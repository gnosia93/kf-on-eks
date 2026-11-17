# 보험 약관 초대형 표 분할 및 하이브리드 검색(Hybrid RAG) 구현 가이드

약관의 **수십 페이지짜리 `[별표]`(장해분류표·질병분류표)의 헤더를 보존하며 분할하는 로직**과, 사용자의 **가입 상품·가입 시기 메타데이터 필터링**을 적용한 하이브리드 검색기(BM25 + Dense Vector + Reranker) 파이썬 구현 코드입니다.

---

## 1. 초대형 별표(Table) 헤더 복제 분할 파이프라인

수백~수천 행에 달하는 `[별표]`를 그대로 두면 LLM 컨텍스트 한도를 초과하고, 단순 분할하면 표의 열(Column) 제목이 사라져 숫자나 코드의 의미를 잃어버립니다. 

아래 코드는 **마크다운 표의 상단 헤더 2줄(`| 열1 | 열2 |` 및 `|---|---|`)을 추출해 두었다가, 지정된 행 수(`rows_per_chunk`)마다 분할할 때 모든 조각의 맨 위에 헤더를 자동으로 복제**합니다.

```python
from typing import List, Dict


def split_large_markdown_table(
    table_lines: List[str],
    metadata: Dict,
    rows_per_chunk: int = 35
) -> List[Dict]:
    """초대형 마크다운 표를 헤더(Header)를 복제한 상태로 안전하게 분할합니다."""
    # 최소 헤더(2줄) + 데이터(1줄) 이상이어야 분할 대상
    if len(table_lines) <= rows_per_chunk + 2:
        return [{
            "text": "\n".join(table_lines),
            "metadata": {**metadata, "is_split_table": False}
        }]

    # 마크다운 표의 헤더(제목 행 + 구분선 행) 추출
    table_header = table_lines[:2]
    data_rows = table_lines[2:]
    
    split_chunks = []
    total_parts = (len(data_rows) + rows_per_chunk - 1) // rows_per_chunk

    for part_idx, start in enumerate(range(0, len(data_rows), rows_per_chunk), start=1):
        batch_rows = data_rows[start : start + rows_per_chunk]
        
        # 상위 경로 + 분할 표 번호 + 원본 표 헤더 + 데이터 행 결합
        prefix = (
            f"[상품: {metadata['product_name']} ({metadata['effective_date']} 개정) | "
            f"{metadata['section']} > {metadata['article']} (표 분할 {part_idx}/{total_parts})]\n"
        )
        reconstructed_table = prefix + "\n".join(table_header + batch_rows)

        split_chunks.append({
            "text": reconstructed_table,
            "metadata": {
                **metadata,
                "is_split_table": True,
                "table_part": f"{part_idx}/{total_parts}"
            }
        })

    return split_chunks
```

---

## 2. 가입 상품·시기 필터링 하이브리드 검색기 (Qdrant + BM25 + Reranker)

보험 약관 RAG에서 가장 치명적인 사고는 **고객이 가입한 2021년 약관 대신 2024년 개정 약관을 근거로 답변하는 것**입니다.
* **1단계 (Pre-Filtering)**: Vector DB(`Qdrant`)의 `Filter` 기능으로 `product_code`와 `effective_date`가 일치하는 문서만 1차 격리합니다.
* **2단계 (Hybrid Retrieval)**: 격리된 문서군 내에서 **의미 벡터 검색(BGE-M3)**과 **키워드 검색(BM25)** 점수를 결합(RRF, Reciprocal Rank Fusion)하여 상위 20개를 추출합니다.
* **3단계 (Reranking)**: Cross-Encoder 리랭커(`bge-reranker-v2-m3`)로 질문과 조항의 논리적 정합성을 재평가해 최상위 `top_k`(3~5개)만 `Qwen-32B`에 전달합니다.

```python
import re
from typing import List, Dict
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class InsuranceHybridRetriever:
    def __init__(self, qdrant_url: str, collection_name: str):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
        # 한국어 약관 및 전문 용어에 강한 임베딩 & 리랭커 모델 로드
        self.embed_model = SentenceTransformer("BAAI/bge-m3")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024)

    @staticmethod
    def _tokenize_ko(text: str) -> List[str]:
        # 한국어 조사/어미 단순 분리 및 조항 번호('제15조'), 질병 코드('I60') 보존 토크나이저
        return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())

    def search(
        self,
        question: str,
        product_code: str,
        effective_date: str,
        top_k: int = 4,
        candidate_k: int = 20
    ) -> List[Dict]:
        # 1. 필수 메타데이터 필터 정의 (다른 상품 및 다른 개정시기 약관 혼입 원천 차단)
        strict_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="product_code",
                    match=models.MatchValue(value=product_code)
                ),
                models.FieldCondition(
                    key="effective_date",
                    match=models.MatchValue(value=effective_date)
                )
            ]
        )

        # 2. Vector 검색 (필터링된 범위 내에서 의미 유사도 상위 candidate_k개 추출)
        query_vector = self.embed_model.encode(question, normalize_embeddings=True).tolist()
        vector_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=strict_filter,
            limit=candidate_k,
            with_payload=True
        )

        if not vector_hits:
            return []

        # 3. 후보군 내 BM25 키워드 재점수화 및 RRF(Reciprocal Rank Fusion) 결합
        candidate_docs = [hit.payload for hit in vector_hits]
        tokenized_corpus = [self._tokenize_ko(doc["text"]) for doc in candidate_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(self._tokenize_ko(question))

        # BM25 순위 계산
        bm25_ranks = {
            idx: rank for rank, idx in enumerate(
                sorted(range(len(candidate_docs)), key=lambda i: bm25_scores[i], reverse=True)
            )
        }

        # RRF 점수 합산 (Vector 순위 + BM25 순위)
        rrf_k = 60
        fused_scores = []
        for idx in range(len(candidate_docs)):
            vec_rank = idx  # vector_hits는 이미 점수순 정렬됨
            bm_rank = bm25_ranks[idx]
            rrf_score = (1.0 / (rrf_k + vec_rank + 1)) + (1.0 / (rrf_k + bm_rank + 1))
            fused_scores.append((idx, rrf_score))

        # RRF 상위 후보 선별
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        rerank_candidates = [candidate_docs[idx] for idx, _ in fused_scores[:12]]

        # 4. Cross-Encoder 리랭킹으로 최종 top_k 선별
        pairs = [[question, doc["text"]] for doc in rerank_candidates]
        rerank_scores = self.reranker.predict(pairs)

        final_ranked = sorted(
            zip(rerank_candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {
                "text": doc["text"],
                "metadata": doc,
                "rerank_score": float(score)
            }
            for doc, score in final_ranked
        ]
```

---

## 3. 사용자 유형별(고객 vs 직원) vLLM 호출 라우팅 예시

검색기에서 뽑아온 최종 약관 조항(`contexts`)을 바탕으로, 요청자의 역할(`user_role`)에 따라 **LoRA 어댑터(`model`)와 시스템 프롬프트를 자동 분기**하여 `Qwen-32B`에 전달합니다.

```python
from openai import OpenAI

vllm_client = OpenAI(base_url="http://cluster.local", api_key="not-used")


def ask_insurance_bot(
    question: str,
    product_code: str,
    effective_date: str,
    user_role: str,  # "customer" 또는 "agent"
    retriever: InsuranceHybridRetriever
) -> str:
    # 1. 가입 상품/시기가 일치하는 약관 조항 검색
    contexts = retriever.search(
        question=question,
        product_code=product_code,
        effective_date=effective_date,
        top_k=4
    )
    if not contexts:
        return "해당 상품 및 가입 시기에 일치하는 약관 정보를 찾을 수 없습니다. 고객센터로 문의해 주세요."

    context_block = "\n\n---\n\n".join([c["text"] for c in contexts])

    # 2. 사용자 권한에 따른 LoRA 어댑터 및 시스템 지침 분기
    if user_role == "agent":
        model_name = "qwen3-32b-agent"
        system_prompt = (
            "당신은 보험사 내부 콜센터 심사 지원 AI입니다. 제공된 약관만을 근거로 "
            "1) 지급/면책 결론, 2) 핵심 근거 조항(관/조/항/호), 3) 예외 및 주의사항을 "
            "개조식(Bullet point)으로 간결하게 요약하세요. 약관에 없는 내용은 절대 추측하지 마세요."
        )
    else:
        model_name = "qwen3-32b-customer"
        system_prompt = (
            "당신은 친절한 보험 약관 안내 챗봇입니다. 제공된 약관만을 근거로 어려운 보험 용어를 "
            "고객이 이해하기 쉽게 풀어서 설명하고, 보장 제외(면책) 조건이 있다면 빠짐없이 안내하세요. "
            "답변 마지막에는 반드시 [근거: 상품명 제N조 제N항]을 표기하세요."
        )

    # 3. 단일 Qwen-32B vLLM 서버의 해당 LoRA 어댑터 호출
    resp = vllm_client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"참고 약관:\n{context_block}\n\n질문: {question}"}
        ]
    )
    return resp.choices[0].message.content.strip()
```
