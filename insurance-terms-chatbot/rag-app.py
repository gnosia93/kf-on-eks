import os
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import mlflow
from mlflow.entities import SpanType

# ---------------------------------------------------------
# 1. 환경 설정 및 클라이언트 초기화
# ---------------------------------------------------------
app = FastAPI(title="Insurance Hybrid RAG API", version="1.0.0")

VLLM_URL = os.environ.get("VLLM_URL", "http://cluster.local")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://cluster.local")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "insurance_terms")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://cluster.local")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "qwen3-insurance-rag"))

vllm_client = OpenAI(base_url=VLLM_URL, api_key="not-used")
qdrant_client = QdrantClient(url=QDRANT_URL)

# 한국어 약관 및 질병코드/조항번호 처리에 강한 임베딩 & 리랭커 모델
embed_model = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024)


# ---------------------------------------------------------
# 2. 요청/응답 데이터 스키마 (가입 상품 및 개정 시기 필수 입력)
# ---------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., description="사용자 또는 상담원 질문")
    product_code: str = Field(..., description="가입 보험 상품 코드 (예: PRD_HEALTH_2024)")
    effective_date: str = Field(..., description="가입 시점 약관 개정일 (YYYY-MM-DD, 예: 2024-01-01)")
    user_role: str = Field("customer", description="사용자 유형: 'customer'(일반고객) 또는 'agent'(콜센터직원)")
    top_k: int = Field(4, ge=1, le=10, description="LLM에 전달할 최종 약관 조항 개수")


class AskResponse(BaseModel):
    answer: str
    used_model: str
    retrieved_articles: List[dict]


# ---------------------------------------------------------
# 3. 메타데이터 필터링 기반 하이브리드 검색기 (Vector + BM25 + Reranker)
# ---------------------------------------------------------
def tokenize_ko(text: str) -> List[str]:
    """한글 단어, 조항 번호('제15조'), 질병 코드('I60')를 보존하는 토크나이저"""
    return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_filtered_clauses(
    question: str,
    product_code: str,
    effective_date: str,
    top_k: int = 4,
    candidate_k: int = 24
) -> List[dict]:
    # [핵심 안전장치] 가입 상품 코드와 약관 개정 시기가 정확히 일치하는 문서만 필터링
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

    # 1단계: 필터링된 범위 내에서 Dense Vector 검색으로 1차 후보군 추출
    query_vec = embed_model.encode(question, normalize_embeddings=True).tolist()
    vector_hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        query_filter=strict_filter,
        limit=candidate_k,
        with_payload=True
    )

    if not vector_hits:
        return []

    # 2단계: 추출된 후보군 내에서 BM25 키워드 점수 계산 및 RRF(Reciprocal Rank Fusion) 결합
    candidates = [hit.payload for hit in vector_hits]
    tokenized_docs = [tokenize_ko(doc["text"]) for doc in candidates]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(tokenize_ko(question))

    # BM25 순위 매핑
    bm25_order = sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {doc_idx: rank for rank, doc_idx in enumerate(bm25_order)}

    # RRF 점수 합산 (의미 검색 순위 + 키워드 검색 순위)
    rrf_k = 60
    fused = []
    for idx in range(len(candidates)):
        vec_rank = idx
        bm_rank = bm25_ranks[idx]
        rrf_score = (1.0 / (rrf_k + vec_rank + 1)) + (1.0 / (rrf_k + bm_rank + 1))
        fused.append((idx, rrf_score))

    fused.sort(key=lambda x: x[1], reverse=True)
    rerank_pool = [candidates[idx] for idx, _ in fused[:12]]

    # 3단계: Cross-Encoder 리랭커로 정밀 재순위화 후 최종 top_k 선별
    pairs = [[question, doc["text"]] for doc in rerank_pool]
    rerank_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(rerank_pool, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # MLflow Retriever 표준 스키마 반환
    return [
        {
            "id": doc.get("doc_id", ""),
            "page_content": doc["text"],
            "metadata": {
                "product_code": doc.get("product_code"),
                "effective_date": doc.get("effective_date"),
                "section": doc.get("section"),
                "article": doc.get("article"),
                "rerank_score": float(score)
            }
        }
        for doc, score in ranked_results
    ]


# ---------------------------------------------------------
# 4. 사용자 유형별(고객 vs 직원) 듀얼 LoRA 생성기
# ---------------------------------------------------------
@mlflow.trace(span_type=SpanType.LLM)
def generate_answer(question: str, contexts: List[dict], user_role: str) -> tuple[str, str]:
    context_text = "\n\n---\n\n".join([c["page_content"] for c in contexts])

    if user_role == "agent":
        model_name = "qwen3-32b-agent"
        system_prompt = (
            "당신은 보험사 내부 콜센터 심사 지원 AI입니다. 제공된 참고 약관만을 근거로 "
            "1) 지급/면책 결론, 2) 핵심 근거 조항(관/조/항/호), 3) 예외 및 심사 유의사항을 "
            "개조식(Bullet point)으로 간결하게 요약하세요. 약관에 없는 내용은 절대 추측하지 마세요."
        )
    else:
        model_name = "qwen3-32b-customer"
        system_prompt = (
            "당신은 친절한 보험 약관 안내 챗봇입니다. 제공된 참고 약관만을 근거로 어려운 보험 용어를 "
            "고객이 이해하기 쉽게 풀어서 설명하고, 보장 제외(면책) 조건이 있다면 빠짐없이 안내하세요. "
            "답변 마지막에는 반드시 [근거: 상품명 제N조 제N항]을 표기하세요."
        )

    resp = vllm_client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"참고 약관:\n{context_text}\n\n질문: {question}"}
        ]
    )

    span = mlflow.get_current_active_span()
    if span and resp.usage:
        span.set_attributes({
            "user_role": user_role,
            "lora_model": model_name,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens
        })

    return resp.choices[0].message.content.strip(), model_name


# ---------------------------------------------------------
# 5. FastAPI 엔드포인트 (루트 Trace 및 헬스체크)
# ---------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@mlflow.trace(span_type=SpanType.CHAIN)
def ask_endpoint(req: AskRequest):
    if req.user_role not in ("customer", "agent"):
        raise HTTPException(status_code=400, detail="user_role은 'customer' 또는 'agent'만 가능합니다.")

    # 1. 메타데이터 필터링 하이브리드 검색
    contexts = retrieve_filtered_clauses(
        question=req.question,
        product_code=req.product_code,
        effective_date=req.effective_date,
        top_k=req.top_k
    )

    # 일치하는 약관이 없으면 LLM 호출 전 즉시 안전 응답 반환 (환각 원천 차단)
    if not contexts:
        return AskResponse(
            answer="요청하신 상품 코드 및 가입 시기에 해당하는 약관 조항을 찾을 수 없습니다. 상품 정보를 다시 확인해 주세요.",
            used_model="none",
            retrieved_articles=[]
        )

    # 2. 역할별 LoRA 어댑터 호출
    answer, used_model = generate_answer(
        question=req.question,
        contexts=contexts,
        user_role=req.user_role
    )

    return AskResponse(
        answer=answer,
        used_model=used_model,
        retrieved_articles=[c["metadata"] for c in contexts]
    )
