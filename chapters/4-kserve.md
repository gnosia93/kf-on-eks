# KServe 서빙 — S3의 모델을 InferenceService로 배포 + 추론 테스트

> 앞 단계에서 파이프라인이 `s3://models/iris/v1/model.joblib` 로 모델을 등록했다는 전제.
> KServe는 Standard 모드로 설치돼 있다고 가정. 검증: KServe v0.21.0

## 전체 그림
```
(파이프라인) register_model ─▶ s3://models/iris/v1/model.joblib
                                        │
                                        ▼
            KServe InferenceService(sklearn, storageUri=s3://models/iris/v1)
                                        │
                                        ▼
                         예측 엔드포인트  ──curl──▶  {"predictions":[0]}
```

## 사전 준비
- KServe 설치됨 (`kubectl get pods -n kserve` → controller Running)
- S3(또는 MinIO)에 `s3://models/iris/v1/model.joblib` 존재
- 서빙용 네임스페이스 생성:
```bash
kubectl create namespace kserve-demo
```

---

## 1. 서빙 네임스페이스에 S3 자격증명 (Secret + ServiceAccount)
> ⚠️ 이 Secret은 **서빙 네임스페이스(kserve-demo)** 에 있어야 함.
> (파이프라인용 s3-creds는 kubeflow 네임스페이스에 따로 있던 것 — 서로 다름!)

`s3-secret.yaml`:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: s3-creds
  namespace: kserve-demo
  annotations:
    serving.kserve.io/s3-endpoint: s3.amazonaws.com   # MinIO면 minio-service.kubeflow:9000
    serving.kserve.io/s3-usehttps: "1"                # MinIO(비TLS)면 "0"
    serving.kserve.io/s3-region: us-east-1
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: <YOUR_KEY>
  AWS_SECRET_ACCESS_KEY: <YOUR_SECRET>
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kserve-s3-sa
  namespace: kserve-demo
secrets:
  - name: s3-creds
```
```bash
kubectl apply -f s3-secret.yaml
```

---

## 2. InferenceService 정의
`inferenceservice.yaml`:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: iris
  namespace: kserve-demo
spec:
  predictor:
    serviceAccountName: kserve-s3-sa          # 위 S3 secret 연결
    model:
      modelFormat:
        name: sklearn
      storageUri: s3://models/iris/v1         # '디렉터리' 경로 (그 안의 model.joblib 로드)
      # protocolVersion: v2                   # 필요 시 V2(Open Inference Protocol) 강제
```
```bash
kubectl apply -f inferenceservice.yaml
```

---

## 3. 준비 상태 확인
```bash
kubectl get inferenceservice iris -n kserve-demo -w
# READY=True 될 때까지 (모델 다운로드 + 서버 기동에 1~2분)

# 파드 확인
kubectl get pods -n kserve-demo
# iris-predictor-... 가 Running
```

---

## 4. 추론 테스트 (curl)
Standard 모드는 인그레스 설정에 따라 접근법이 다릅니다. 가장 확실한 **port-forward** 방식:
```bash
# predictor 서비스 이름 확인
kubectl get svc -n kserve-demo | grep iris
# 보통 iris-predictor

# 포트포워드 (다른 터미널 유지)
kubectl port-forward -n kserve-demo svc/iris-predictor 8081:80
```

**(V1 프로토콜) 예측**:
```bash
curl -H "Content-Type: application/json" \
  http://localhost:8081/v1/models/iris:predict \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'
# 기대: {"predictions":[0]}
```

**(V2 / Open Inference Protocol)** — V1이 404면 런타임이 V2입니다:
```bash
# 준비 확인
curl http://localhost:8081/v2/models/iris

# 예측
curl -H "Content-Type: application/json" \
  http://localhost:8081/v2/models/iris/infer \
  -d '{
    "inputs": [
      {"name": "input-0", "shape": [1, 4], "datatype": "FP32",
       "data": [5.1, 3.5, 1.4, 0.2]}
    ]
  }'
```

> 인그레스/외부 URL로 접근하려면 `kubectl get isvc iris -n kserve-demo` 의 **STATUS URL** 을 사용
> (Knative 모드면 그 URL + Host 헤더로 curl).

---

## 5. 꼭 알아둘 포인트 (gotchas)
| 항목 | 내용 |
|---|---|
| secret 네임스페이스 | S3 secret은 **InferenceService와 같은 네임스페이스(kserve-demo)** 에 있어야 함 |
| storageUri | **파일이 아니라 디렉터리**(s3://.../v1) 를 가리킴. 그 안의 model.joblib 로드 |
| 모델 파일명 | sklearn 런타임은 `model.joblib` 을 찾음 (파이프라인 출력과 일치) |
| 프로토콜 | V1(`/v1/models/iris:predict`) vs V2(`/v2/models/iris/infer`) — 런타임 버전에 따라 다름 |
| 접근 방식 | Standard 모드는 port-forward 또는 인그레스. Knative 모드는 외부 URL + Host 헤더 |
| S3 endpoint | AWS면 s3.amazonaws.com, MinIO면 내부 서비스 주소 + usehttps "0" |

---

## 6. 정리 (cleanup)
```bash
kubectl delete inferenceservice iris -n kserve-demo
kubectl delete -f s3-secret.yaml
```

---

## 7. LLM / GPU 로 업그레이드 (다음 목표)
sklearn → vLLM 로 바꾸면 LLM 서빙:
```yaml
spec:
  predictor:
    model:
      modelFormat:
        name: vllm            # 또는 huggingface 런타임
      storageUri: s3://models/my-llm/v1
      resources:
        limits:
          nvidia.com/gpu: "1"
      args:
        - --max-model-len=8192
        - --gpu-memory-utilization=0.9     # 앞서 배운 vLLM 플래그
```
추론 게이트웨이로 **LiteLLM**, 관측/eval로 **W&B Weave / Langfuse** 를 붙이면 LLMOps 루프 완성.

---

✅ 이것으로 **데이터 → 학습 → 평가 → 게이트 → S3 등록 → KServe 서빙 → 추론** end-to-end가 닫힙니다.
