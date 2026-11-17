flowchart TB
    %% 사용자 영역
    subgraph Users["1. 사용자 접점 (Client)"]
        U1["일반 고객 앱/웹\n(user_role: customer)"]
        U2["콜센터 상담원 화면\n(user_role: agent)"]
    end

    %% 오프라인 학습 및 데이터 파이프라인
    subgraph Offline["2. 오프라인 데이터 & MLOps 파이프라인 (Kubeflow + S3)"]
        PDF["보험 약관 PDF\n(조항 + 별표)"] --> PARSE["Docling 파서 & 계층 청커\n(표 헤더 복제 + 메타데이터 태깅)"]
        PARSE --> EMB["BGE-M3 임베딩 생성"]
        EMB --> MILVUS[("Milvus Vector DB\n- HNSW 벡터 인덱스\n- INVERTED 인덱스\n(product_code, effective_date)")]

        PARSE --> SYN["합성 Q&A 데이터 생성\n(고객용 / 직원용 페르소나)"]
        SYN --> KFP["Kubeflow Pipelines\n(QLoRA 학습 -> 평가 -> 승격)"]
        KFP --> S3[("AWS S3\nserving/CURRENT\n- adapter-customer\n- adapter-agent")]
    end

    %% 온라인 서빙 파이프라인
    subgraph Online["3. 온라인 하이브리드 RAG & 서빙 (Kubernetes llm 네임스페이스)"]
        RAG["FastAPI RAG 서버\n(rag_insurance_milvus.py)"]
        
        subgraph HybridSearch["3단 하이브리드 검색기"]
            S1["1) Milvus 사전 필터링 + Dense 검색\n(상품코드 & 가입시기 격리)"]
            S2["2) BM25 키워드 검색 + RRF 결합\n(조항번호 · 질병코드 매칭)"]
            S3["3) Cross-Encoder 리랭커\n(bge-reranker-v2-m3 -> Top-K)"]
            S1 --> S2 --> S3
        end

        subgraph VLLM["vLLM 단일 추론 서버 (GPU)"]
            BASE["Base Model: Qwen-32B"]
            LORA1["LoRA 1: qwen3-32b-customer\n(쉬운 용어 + 근거 조항)"]
            LORA2["LoRA 2: qwen3-32b-agent\n(심사 결론 + 개조식 요약)"]
            BASE --- LORA1
            BASE --- LORA2
        end

        MLFLOW[("MLflow Server\nTracing & Evaluation\n(RETRIEVER / LLM Span)")]
    end

    %% 연결 흐름
    U1 & U2 -->|"POST /ask\n(질문, 상품코드, 가입일, 역할)"| RAG
    RAG --> S1
    MILVUS -.->|"필터링된 약관 청크 반환"| S1
    S3 -->|"최종 선별 약관 조항 (Top-K)"| RAG

    S3 -.->|"어댑터 자동 동기화\n(Init Container)"| VLLM
    RAG -->|"customer 요청"| LORA1
    RAG -->|"agent 요청"| LORA2
    LORA1 & LORA2 -->|"생성 답변 반환"| RAG

    RAG -.->|"@mlflow.trace\n(입출력 · 지연시간 · 토큰수)"| MLFLOW
    RAG -->|"최종 답변 + 근거 조항"| U1 & U2



```
[오프라인 파이프라인]
  보험 약관 PDF ──▶ Docling 파싱/청킹 ──▶ Milvus 적재 (product_code, effective_date 인덱싱)
                         │
                         ▼
                  합성 Q&A 생성 ──▶ KFP QLoRA 학습/승격 ──▶ S3 (customer / agent 어댑터)
                                                                │ (파드 시작 시 동기화)
[온라인 추론 파이프라인]                                        ▼
  고객 / 상담원 ──▶ FastAPI RAG 서버 ──▶ [1] Milvus 사전필터링 + Vector 검색
                          │             [2] BM25 키워드 결합 (RRF)
                          │             [3] Cross-Encoder 리랭커 (Top-K)
                          ▼
                    vLLM (Qwen-32B) ──▶ customer 어댑터 (쉬운 설명 + 근거 조항)
                          │         └──▶ agent 어댑터 (심사 결론 + 개조식 요약)
                          ▼
                    MLflow Tracing (검색 문서, 프롬프트, 토큰 수 전 과정 기록)
```



## Qwen-32B 기반 보험 약관 챗봇 구축 아키텍처 (일반 고객용 & 콜센터 직원용) ##

단일 `Qwen-32B` 서빙 서버 위에 **Multi-LoRA(다중 어댑터)**와 **하이브리드 RAG** 구조를 결합하여, 일반 고객용 챗봇과 내부 콜센터 직원용 어시스턴트를 동시에 구축하는 실무 로드맵입니다.


### 1. 핵심 아키텍처 전략: 단일 Qwen-32B + 듀얼 LoRA 어댑터 ###

GPU 메모리를 효율적으로 사용하면서 두 사용자층(고객 vs 직원)의 응답 톤과 정보 깊이를 완벽히 분리하는 방식입니다.

* **Base 모델 (`Qwen-32B`, AWQ/FP8 양자화 권장)**
  * 한국어 이해력과 복잡한 조건부 논리(약관 해석) 능력이 우수합니다.
  * 긴 약관 조항과 표 데이터를 수용하기 위해 컨텍스트 길이를 **16K~32K 이상** 확보합니다. (단일 A100 80GB 또는 L40S 2장 TP=2 권장)
* **LoRA 어댑터 1 (`adapter-customer`: 일반 고객용)**
  * **학습 방향**: 어려운 보험 용어를 쉬운 일상 언어로 풀어서 설명하고, 면책 조항(보장 제외 사항)을 친절하게 안내하며, 약관에 없는 내용은 상담원 연결을 유도합니다.
* **LoRA 어댑터 2 (`adapter-agent`: 내부 콜센터 직원용)**
  * **학습 방향**: 고객 응대용 미사여구를 배제하고, **약관 조항 번호(예: 제15조 제2항), 지급 심사 기준, 예외 특약 코드, 유사 민원 사례**를 핵심 요약(Bullet point) 형태로 즉각 도출합니다.


### 2. 데이터 및 검색기(RAG) 구축 전략 (보험 약관 특화) ###

보험 약관은 계층 구조(장-절-조-항-호)와 지급률 표(Table)가 핵심이므로 정교한 파싱 및 검색 파이프라인이 필요합니다.

#### ① 문서 파싱 및 청킹 (Parsing & Chunking) ####
* PDF 약관을 단순 텍스트로 분할하면 표 구조가 깨지고 조항 간 참조("제1항에 따라...")가 유실됩니다.
* 전용 문서 파서(Docling, Upstage Document Parse, Marker 등)를 사용하여 표를 **Markdown Table** 형태로 보존합니다.
* 모든 청크(Chunk)의 메타데이터에 `[상품명 > 담보명 > 제N조 제목]`을 상위 컨텍스트로 항상 포함하여 저장합니다.

#### ② 하이브리드 검색기 (Hybrid Search + Reranker) ####
* **키워드 검색 (BM25)**: `"제15조"`, `"뇌혈관질환(I60~I69)"`, 특약 코드 등 정확한 용어 매칭에 필수적입니다.
* **벡터 검색 (Dense Embedding)**: *"운전하다가 사고 났는데 동승자도 보상되나요?"* 같은 일반 고객의 구어체 질문을 관련 약관 조항과 연결합니다. (`BGE-M3` 또는 `Qwen3-Embedding` 계열 권장)
* **리랭커 (Reranker)**: 하이브리드 검색으로 상위 20개 조항을 추출한 뒤, Cross-Encoder 리랭커를 통해 핵심 상위 3~5개 조항만 선별하여 LLM에 전달합니다.


### 3. 단계별 실행 로드맵 ###

| 단계 | 수행 작업 | 활용 기술 및 실무 팁 |
| :--- | :--- | :--- |
| **1단계: 지식 베이스 구축** | 약관 PDF 전처리 및 DB 적재 | Markdown 표 변환 + 메타데이터 태깅 후 Vector DB(Qdrant, Milvus, OpenSearch 등) 적재 |
| **2단계: 학습 데이터셋 생성** | 고객용 / 직원용 Q&A 쌍 구축 | 실제 콜센터 상담 이력(비식별화)과 약관 조항 기반 합성 데이터(Synthetic Data) 각 수천 건 생성 |
| **3단계: 파인튜닝 및 승격** | KFP 파이프라인으로 LoRA 2종 학습 | Kubeflow Pipelines에서 고객용/직원용 어댑터 학습·평가 후 S3 포인터(`serving/CURRENT`) 갱신 |
| **4단계: vLLM 멀티 서빙** | 단일 엔드포인트로 동시 배포 | `--lora-modules customer=... agent=...`로 구동하고, API 요청 시 사용자 권한에 따라 `model` 파라미터 분기 |
| **5단계: 환각 방지 및 평가** | Ragas / MLflow Evaluate 검증 | **Faithfulness(문맥 충실성)** 지표를 최우선 배포 게이트(Gate)로 설정하여 환각 답변 차단 |


### 4. 보험 도메인 필수 안전장치 (Guardrails) ###

* **상품 버전 및 가입 시기 필터링**
  * 보험은 **가입 연도 및 개정 시기(예: 2021년 4월 개정판 vs 2024년 1월 개정판)**에 따라 보장 내용이 완전히 다릅니다.
  * 검색 시 반드시 사용자의 **가입 상품 코드와 가입 일자**를 메타데이터 필터(`where` 조건)로 강제 적용하여 타 시기 약관 혼입을 원천 차단해야 합니다.
* **근거 조항 강제 출력**
  * 답변 말미에 항상 `[근거: OO보험 보통약관 제12조 제1항]` 형식을 출력하도록 시스템 프롬프트 및 LoRA 학습을 적용하여, 직원과 고객 모두 원문을 즉시 교차 검증할 수 있게 합니다.


##  약관 PDF의 표와 조항을 깨지지 않게 파싱·청킹하는 파이프라인 ##
## 고객용/직원용 듀얼 LoRA 학습 데이터셋 구축 방법 ##
## 가입 상품/시기별 메타데이터 필터링이 적용된 하이브리드 RAG 코드 ##
