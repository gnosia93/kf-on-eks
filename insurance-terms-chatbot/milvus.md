# Milvus 기반 보험 약관 하이브리드 RAG 구축 가이드

Qdrant 대신 **Milvus (`pymilvus` 2.4+)**를 벡터 DB로 사용하여 **가입 상품(`product_code`)·가입 시기(`effective_date`) 메타데이터 필터링**, **Dense Vector + BM25 + Reranker 하이브리드 검색**, 그리고 **고객/직원 듀얼 LoRA 어댑터 서빙**을 구현한 전체 코드입니다.

---

## 1. 약관 코퍼스 Milvus 인덱싱 스크립트 (`index_to_milvus.py`)

파싱된 약관 데이터(`parsed_corpus.jsonl`)를 읽어 `BAAI/bge-m3` 임베딩을 생성한 뒤 Milvus 컬렉션에 적재(`upsert`)하고, **스칼라 필드 인덱스(`product_code`, `effective_date`)**를 생성하여 초고속 메타데이터 필터링을 보장합니다.

```python
import os
import json
from typing import List, Dict
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. 환경 설정
# ---------------------------------------------------------
MILVUS_URI = os.environ.get("MILVUS_URI", "http://cluster.local")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "insurance_terms")
CORPUS_PATH = os.environ.get("CORPUS_PATH", "parsed_corpus.jsonl")
EMBED_MODEL_NAME = "BAAI/bge-m3"
VECTOR_DIM = 1024  # BGE-M3 출력 차원
BATCH_SIZE = 64


def init_milvus_collection(client: MilvusClient, collection_name: str):
    """Milvus 컬렉션 스키마, 벡터 인덱스 및 메타데이터 스칼라 인덱스를 생성합니다."""
    if not client.has_collection(collection_name=collection_name):
        # 1. 스키마 정의 (Primary Key를 문자열로 설정하여 Upsert 멱등성 보장)
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="product_code", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="effective_date", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="article", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

        # 2. 인덱스 설정 (벡터 HNSW 인덱스 + 메타데이터 INVERTED 인덱스)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        # 가입 상품 및 개정 시기 필터링 성능 극대화용 역인덱스
        index_params.add_index(field_name="product_code", index_type="INVERTED")
        index_params.add_index(field_name="effective_date", index_type="INVERTED")

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"[INFO] 신규 Milvus 컬렉션 및 인덱스 생성 완료: {collection_name}")
    else:
        print(f"[INFO] 기존 Milvus 컬렉션 사용: {collection_name}")


def load_corpus(jsonl_path: str) -> List[Dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_documents():
    client = MilvusClient(uri=MILVUS_URI)
    init_milvus_collection(client, COLLECTION_NAME)

    print(f"[INFO] 임베딩 모델 로딩 중: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    records = load_corpus(CORPUS_PATH)
    print(f"[INFO] 총 {len(records)}개 약관 청크 Milvus 적재 시작...")

    for start_idx in range(0, len(records), BATCH_SIZE):
        batch = records[start_idx : start_idx + BATCH_SIZE]
        texts = [item["text"] for item in batch]

        vectors = embed_model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        rows = []
        for item, vec in zip(batch, vectors):
            meta = item["metadata"]
            # 결정론적 PK 생성 -> 재실행 시 중복 없이 덮어쓰기(Upsert)
            pk = f"{meta['product_code']}_{meta['effective_date']}_{item['doc_id']}"

            rows.append({
                "pk": pk,
                "doc_id": item["doc_id"],
                "product_code": meta["product_code"],
                "effective_date": meta["effective_date"],
                "section": meta.get("section", ""),
                "article": meta.get("article", ""),
                "text": item["text"],
                "vector": vec.tolist()
            })

        client.upsert(collection_name=COLLECTION_NAME, data=rows)
        print(f"  -> 진행률: {min(start_idx + BATCH_SIZE, len(records))}/{len(records)} 적재 완료")

    # 검색 가능하도록 컬렉션 메모리 로드 보장
    client.load_collection(collection_name=COLLECTION_NAME)
    print("[SUCCESS] 모든 약관 데이터의 Milvus 인덱싱이 완료되었습니다.")


if __name__ == "__main__":
    index_documents()
```

---

## 2. Milvus 기반 하이브리드 RAG 애플리케이션 (`rag_insurance_milvus.py`)

Milvus의 SQL 스타일 불리언 필터(`filter='product_code == "..." and effective_date == "..."'`)로 다른 시기 약관을 원천 차단한 뒤, **Dense Vector + BM25 + Cross-Encoder Reranker**를 거쳐 `Qwen-32B` 듀얼 LoRA 어댑터를 호출하는 FastAPI 서버 코드입니다.

```python
import os
import re
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import mlflow
from mlflow.entities import SpanType

# ---------------------------------------------------------
# 1. 환경 설정 및 클라이언트 초기화
# ---------------------------------------------------------
app = FastAPI(title="Insurance Milvus Hybrid RAG API", version="1.0.0")

VLLM_URL = os.environ.get("VLLM_URL", "http://cluster.local")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://cluster.local")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "insurance_terms")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://cluster.local")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "qwen3-insurance-rag"))

vllm_client = OpenAI(base_url=VLLM_URL, api_key="not-used")
milvus_client = MilvusClient(uri=MILVUS_URI)

# 한국어 약관 및 조항번호/질병코드 처리용 임베딩 & 리랭커
embed_model = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024)


# ---------------------------------------------------------
# 2. 요청/응답 데이터 스키마
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
# 3. Milvus 메타데이터 필터링 기반 하이브리드 검색기
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
    # [핵심 안전장치] Milvus 스칼라 필터 표현식으로 가입 상품 및 개정 시기 엄격 격리
    # 인젝션 방지를 위해 영숫자/하이픈/언더스코어만 허용
    safe_prod = re.sub(r"[^a-zA-Z0-9_-]", "", product_code)
    safe_date = re.sub(r"[^0-9-]", "", effective_date)
    filter_expr = f'product_code == "{safe_prod}" and effective_date == "{safe_date}"'

    # 1단계: Milvus 사전 필터링(Pre-Filtering) 기반 Dense Vector 검색
    query_vec = embed_model.encode(question, normalize_embeddings=True).tolist()
    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        filter=filter_expr,
        limit=candidate_k,
        output_fields=["doc_id", "product_code", "effective_date", "section", "article", "text"]
    )

    hits = search_res if search_res else []
    if not hits:
        return []

    candidates = [hit["entity"] for hit in hits]

    # 2단계: 후보군 내 BM25 키워드 점수 계산 및 RRF(Reciprocal Rank Fusion) 결합
    tokenized_docs = [tokenize_ko(doc["text"]) for doc in candidates]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(tokenize_ko(question))

    bm25_order = sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {doc_idx: rank for rank, doc_idx in enumerate(bm25_order)}

    rrf_k = 60
    fused = []
    for idx in range(len(candidates)):
        vec_rank = idx
        bm_rank = bm25_ranks[idx]
        rrf_score = (1.0 / (rrf_k + vec_rank + 1)) + (1.0 / (rrf_k + bm_rank + 1))
        fused.append((idx, rrf_score))

    fused.sort(key=lambda x: x, reverse=True)
    rerank_pool = [candidates[idx] for idx, _ in fused[:12]]

    # 3단계: Cross-Encoder 리랭커로 논리적 정합성 재평가 후 최종 top_k 선별
    pairs = [[question, doc["text"]] for doc in rerank_pool]
    rerank_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(rerank_pool, rerank_scores),
        key=lambda x: x,
        reverse=True
    )[:top_k]

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


