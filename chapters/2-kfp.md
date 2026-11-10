# K8s에서 Kubeflow Pipelines + KServe 설치 가이드

> 전제: **쿠버네티스 클러스터(v1.32+)가 이미 있고**, `kubectl` 접근이 된다.
> 목표: ML 파이프라인(KFP) + 모델 서빙(KServe) 기반을 깐다. (서빙 모드로 Knative 옵션까지 정리)
> 검증 버전: **KFP 2.18.0**, **KServe v0.21.0**

## 0. 사전 요건 — 도구 확인
```bash
kubectl version --client      # Kubernetes CLI
helm version                  # Helm (KServe에 필요)
git --version                 # git
python --version              # 로컬 KFP SDK용 (3.9+)
```
> 클러스터가 없다면 로컬로: `kind create cluster` (Docker 필요)

---

## 1. Kubeflow Pipelines 설치 (standalone · dev 플레이버)
실험/학습용 최소 설치. 아래 3개 명령으로 끝.
```bash
export PIPELINE_VERSION=2.18.0

# (1) 클러스터 스코프 리소스(CRD 등)
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

# (2) CRD 등록 대기
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# (3) KFP 본체 (dev 플레이버 = 비프로덕션/첫 실험용)
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```
> `2.18.0`은 공식 문서상 **"최신 릴리스로 자동 resolve되는 placeholder"** 이므로 그대로 사용 가능.
> 프로덕션은 dev 플레이버가 아니라 **community distribution**을 사용.

**설치 확인** (약 3분 소요):
```bash
kubectl get pods -n kubeflow
# ml-pipeline, ml-pipeline-ui, minio, mysql, metadata-* 등이 전부 Running
```

**UI 접속**:
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# 브라우저: http://localhost:8080
```

---

## 2. KServe 설치 (Standard 모드 · v0.21.0)
서빙용. **Standard 모드** = Knative/Istio 없이 최소 의존성(Deployment/Service/HPA) → 데모에 가장 간단.

> ⚠️ **보안 습관**: `curl | bash`로 바로 실행하지 말고, **내려받아 내용 확인 후 실행**.
```bash
# (권장) 스크립트를 받아 읽어본 뒤 실행
curl -sL "https://github.com/kserve/kserve/releases/download/v0.21.0/kserve-standard-mode-full-install-with-manifests.sh" -o kserve-install.sh
less kserve-install.sh        # 내용 검토
bash kserve-install.sh
```
> `full-install-with-manifests` = **cert-manager 등 의존성까지 포함**해서 설치.
> 서버리스(scale-to-zero)가 필요하면 Standard 대신 `kserve-knative-mode-full-install-with-manifests.sh` 사용 (아래 Knative 섹션 참고).

**설치 확인**:
```bash
kubectl get pods -n kserve
# kserve-controller-manager 가 Running

kubectl get crd | grep kserve
# inferenceservices.serving.kserve.io 등이 보여야 함
```

---

## 3. 서빙 모드: Standard vs Knative (무엇을 고를까)
**Knative** = 쿠버네티스 위에 **서버리스**(요청 기반 오토스케일 + **scale-to-zero** + 버전/트래픽 관리)를 더하는 CNCF 플랫폼. KServe의 Knative 모드가 이를 사용.

| 항목 | Standard 모드 (이 가이드 기본) | Knative 모드 |
|---|---|---|
| 기반 | K8s Deployment + HPA | Knative Serving |
| scale-to-zero | 없음 (항상 떠 있음) | 있음 (유휴 시 파드 0) |
| 오토스케일 | CPU/메모리(HPA) | 요청 기반 |
| canary(버전 트래픽 분할) | 제한적 | revision 기반 지원 |
| 설치 복잡도 | 간단(의존성 적음) | Knative + 네트워킹 레이어(Istio/Kourier) 추가 |

**GPU 추론 관점**:
- `scale-to-zero`는 **비싼 GPU를 유휴 시 0으로** 내려 비용 절감 → 버스티 트래픽에 유리
- 단, **cold start**: 0에서 다시 뜰 때 파드 기동 + **모델 로딩**(LLM은 특히 느림)이 **첫 요청 지연**으로 나타남

**가이드 권장**: 학습/데모는 **Standard**(간단). 유휴 GPU 비용 절감이 중요한 프로덕션에서 **Knative** 검토.

---

## 4. 로컬 KFP SDK 설치 (파이프라인 작성·제출용)
```bash
python -m venv .venv && source .venv/bin/activate
pip install "kfp>=2.7"
kfp --version      # 2.x 확인
```

---

## 5. 최종 Sanity Check
| 확인 | 명령 | 기대 |
|---|---|---|
| KFP 파드 | `kubectl get pods -n kubeflow` | 전부 Running |
| KFP UI | `http://localhost:8080` (port-forward 중) | 대시보드 열림 |
| KServe 파드 | `kubectl get pods -n kserve` | controller Running |
| KServe CRD | `kubectl get crd | grep kserve` | inferenceservices... |
| 로컬 SDK | `kfp --version` | 2.x |

---

## 6. 설치 순서 한눈에
```
① 도구 확인 (kubectl / helm / git / python, K8s 1.32+)
② KFP 설치 (3 commands) → 파드 Running → UI 포트포워드
③ 서빙 모드 선택 (Standard 권장 / Knative는 scale-to-zero 필요 시)
④ KServe 설치 (스크립트 검토 후 실행) → controller Running → CRD 확인
⑤ 로컬 kfp SDK 설치
⑥ sanity check
```

---

## 7. 다음 단계 (이 가이드 이후)
- **파이프라인 배포**: `data → train → eval → (품질 게이트) → 모델 S3 등록` (KFP v2 Python SDK)
- **서빙**: 등록된 모델을 **KServe InferenceService**로 서빙 → `curl`로 추론 테스트
- **업그레이드**: train을 GPU/분산(Trainer·Volcano)으로, 서빙을 vLLM으로, 관측을 W&B/Langfuse로
