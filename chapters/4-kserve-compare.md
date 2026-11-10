# 실무 LLM 서빙 스택 & 70B 콜드스타트 방지 (2026)

> 핵심: "KServe vs 다른 거"는 비교 대상이 어긋남. 실무는 **엔진 / 플랫폼 / 콜드스타트 기법**을 조합한다.

## 1. 두 층을 구분해야 함

| 층 | 뭐가 있나 | 실무 기본값 |
|---|---|---|
| **추론 엔진** | vLLM, SGLang, TGI, TensorRT-LLM | **vLLM이 사실상 디폴트 (2026)** |
| **오케스트레이션/플랫폼** | KServe, Ray Serve, llm-d, NIM/Dynamo, 관리형 | 상황 따라 갈림 |

- 엔진은 거의 **vLLM**으로 수렴 ([Qovery](https://www.qovery.com/blog/vllm-tgi-nvidia-nim-self-hosted-inference-server-gpu-cost): "2026 기본값은 vLLM").
- KServe는 그 **위에서 돌리는 플랫폼** 중 하나 → 경쟁이 아니라 **겹쳐 쓰는** 관계.

## 2. 플랫폼은 뭘 많이 쓰나

| 상황 | 많이 쓰는 것 |
|---|---|
| **단일 모델 + OpenAI 엔드포인트만** | **vLLM 자체 서버** + plain Deployment/Helm + KEDA (KServe 생략, 제일 단순) |
| **K8s-native / 여러 모델 / 거버넌스** | **KServe** (+ vLLM 런타임), 또는 신생 **llm-d** |
| **NVIDIA 엔터프라이즈 계약** | **NIM / Triton+TensorRT-LLM / Dynamo** |
| **복잡한 멀티모델 파이프라인** (임베딩→리랭크→LLM) | **Ray Serve** |
| **인프라 안 만지고 싶음** | SageMaker / Bedrock / Vertex / Fireworks / Together |

- 2026 K8s 스택 레퍼런스로 **vLLM + KServe + llm-d + Kueue** 조합이 자주 언급됨 ([essamamdani](https://essamamdani.com/blog/scaling-llm-inference-kubernetes-2026-complete-guide)).
- 단일 모델이면 "Ray Serve/KServe 없이 vLLM 서버가 더 간단"이 중론 ([Spheron](https://www.spheron.network/blog/ray-serve-gpu-cloud-llm-deployment/)).

**→ KServe는 "여러 모델 + 플랫폼 기능(라우팅/카나리/오토스케일/캐시/거버넌스)"이 필요할 때 빛나고, 모델 하나만 빠르게면 vLLM 직접도 흔함.**

## 3. 🔑 70B 콜드스타트 방지 — 실무에서 실제로 쓰는 것

프레임워크보다 **"기법"이 핵심**.

> stock K8s에서 70B 신규 replica가 토큰 내기까지 **~10–12분** (GPU 노드 1–1.5분 + 이미지 풀 4–5분 + 140GB 로드 4–5분) — 오토스케일을 무의미하게 만드는 수준 ([Cerebrium](https://cerebrium.ai/resources/autoscaling-llama-70b-inference-replica-startup)).

실전 레시피 (대부분 병행):

1. **Warm 레플리카 유지** (`minReplicas ≥ 1`, scale-to-zero 금지) — 가장 확실. "모델당 최소 1개는 미리 띄워둬라"가 정설 ([markaicode](https://markaicode.com/architecture/kubernetes-inference-architecture/)). 비용 ↔ 지연 트레이드오프.
2. **빠른 모델 로더** — Run:ai Model Streamer (vLLM `--load-format runai_streamer`) / Tensorizer. 스토리지→GPU 직접 스트리밍. AWS·Azure·GKE 공식 문서화 ([Azure AKS](https://blog.aks.azure.com/2026/07/13/runai-streamer-vllm)).
3. **스토리지 아키텍처** — 노드 로컬 NVMe 캐시(= KServe LocalModelCache) 또는 FSx Lustre 고성능 공유FS. NetEase Games는 이 방식으로 **30초 콜드스타트** 달성 ([CNCF](https://www.cncf.io/blog/2026/05/21/how-netease-games-achieved-30-second-llm-cold-starts-on-kubernetes/)).
4. **컨테이너 이미지 최적화** — vLLM+CUDA 이미지가 10–20GB라 이미지 풀 자체도 병목. DaemonSet 프리풀 / 세컨더리 부트디스크 / 노드 이미지 캐시 ([Medium](https://medium.com/@manikandan_t/the-gpu-cold-starts-nobody-warns-you-about-autoscaling-llm-inference-on-kubernetes-4128cb8743f1)).
5. **예측적 오토스케일** — KEDA + vLLM 메트릭, 또는 긴 기동 지연을 흡수하는 lookahead(선제) 스케일링.

## 4. 현실적 권장 (70B 콜드스타트 목적)

> **Warm pool(min replica 유지) + 빠른 로더(run:ai streamer) + 로컬 NVMe 또는 FSx Lustre**

- **K8s-native 팀** → KServe가 이 패턴(특히 LocalModelCache)을 잘 패키징. (앞서 설계한 구성이 정석 중 하나)
- **단순하게 갈 팀** → vLLM 직접 서버 + 위 기법 수동 조합.
- **NVIDIA 스택** → NIM/Dynamo가 로더·캐시·최적 엔진을 상당 부분 내장.

---

**한 줄 요약**: 엔진은 **vLLM이 표준**, 플랫폼은 조직 상황 따라 **KServe / vLLM 직접 / NIM** 로 갈림. 70B 콜드스타트는 어느 플랫폼이든 **"warm pool + 빠른 로더 + 로컬/고속 스토리지"** 조합으로 잡는 게 공통 패턴.
