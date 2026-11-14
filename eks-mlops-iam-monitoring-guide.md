# AWS EKS MLOps 운영 완성 가이드: IAM 최소 권한 정책 및 vLLM 모니터링 대시보드

본 문서는 앞서 구성한 **EKS + Karpenter + Kubeflow Pipelines + MLflow + vLLM** 아키텍처를 실제 프로덕션 수준으로 안전하게 운영하기 위한 **AWS IAM 최소 권한(Least Privilege) 정책(JSON)**과 **vLLM 및 GPU 실시간 모니터링(Prometheus + Grafana) 구성 가이드**를 담고 있습니다.

---

## 1. AWS IAM 최소 권한(Least Privilege) 정책 정의

운영 환경에서는 보안을 위해 파드(Pod)별로 필요한 최소한의 AWS 권한만 부여해야 합니다.

### 1.1. MLflow / Kubeflow / vLLM 파드용 S3 아티팩트 접근 정책 (`mlops-s3-policy.json`)

학습된 모델 가중치, 평가 데이터셋, 파이프라인 아티팩트를 S3 버킷에서 읽고 쓰기 위해 IRSA(`mlops-sa`)에 연결할 최소 권한 정책입니다. (`<YOUR_BUCKET_NAME>`을 실제 버킷 이름으로 변경하세요.)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListArtifactBucket",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::<YOUR_BUCKET_NAME>"
    },
    {
      "Sid": "ReadWriteArtifacts",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::<YOUR_BUCKET_NAME>/*"
    }
  ]
}
```

### 1.2. GitHub Actions CI/CD 파이프라인용 IAM 정책 (`github-actions-mlops-policy.json`)

GitHub Actions가 Amazon ECR에 도커 이미지를 푸시하고, EKS 클러스터 내부의 Kubeflow Pipelines API를 호출하기 위해 필요한 최소 권한 정책입니다. (실무에서는 장기 Access Key 대신 **GitHub OIDC Provider**와 연동하여 아래 권한을 가진 IAM Role을 Assume하도록 구성하는 것을 권장합니다.)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuthToken",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ECRPushImage",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "arn:aws:ecr:<AWS_REGION>:<AWS_ACCOUNT_ID>:repository/llm-finetune-pipeline"
    },
    {
      "Sid": "EKSDescribeForKubeconfig",
      "Effect": "Allow",
      "Action": [
        "eks:DescribeCluster"
      ],
      "Resource": "arn:aws:eks:<AWS_REGION>:<AWS_ACCOUNT_ID>:cluster/<YOUR_EKS_CLUSTER_NAME>"
    }
  ]
}
```

> **참고:** `eks:DescribeCluster` 권한을 받은 GitHub Actions 실행 주체가 Kubernetes API를 통해 Kubeflow 파드에 접근하려면, EKS의 `aws-auth` ConfigMap 또는 EKS Access Entry에 해당 IAM Role을 매핑하고 `kubeflow` 네임스페이스에 대한 `pods/portforward` 및 `pods/exec` 권한(Role/RoleBinding)을 부여해야 합니다.

### 1.3. Kubernetes RBAC 설정 (`kfp-ci-rbac.yaml`)

GitHub Actions용 IAM Role이 매핑된 Kubernetes 그룹(`ci-pipeline-group`)이 `kubeflow` 네임스페이스의 파이프라인 API에만 접근할 수 있도록 제한하는 RBAC 설정입니다.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: kfp-trigger-role
  namespace: kubeflow
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["pods/portforward", "services/proxy"]
    verbs: ["create", "get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kfp-trigger-rolebinding
  namespace: kubeflow
subjects:
  - kind: Group
    name: ci-pipeline-group
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: kfp-trigger-role
  apiGroup: rbac.authorization.k8s.io
```

---

## 2. vLLM 및 GPU 실시간 모니터링 구성 (Prometheus + Grafana)

LLM 추론 서빙 환경에서는 단순 CPU/메모리 모니터링을 넘어 **GPU 사용률, VRAM 점유율, 첫 토큰 생성 시간(TTFT), 초당 토큰 처리량(TPS), 대기 큐 길이**를 실시간으로 감시해야 장애와 성능 저하(Drift)에 선제적으로 대응할 수 있습니다.

### 2.1. 모니터링 스택 설치 (kube-prometheus-stack & NVIDIA DCGM Exporter)

EKS 클러스터에 Prometheus, Grafana와 함께 GPU 하드웨어 지표를 수집하는 NVIDIA DCGM Exporter를 설치합니다.

```bash
# 1. Helm 저장소 추가
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update

# 2. Prometheus & Grafana 설치 (monitoring 네임스페이스)
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# 3. NVIDIA DCGM Exporter 설치 (GPU 하드웨어 메트릭 수집)
helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace monitoring \
  --set serviceMonitor.enabled=true
```

### 2.2. vLLM 메트릭 수집을 위한 Service 및 ServiceMonitor (`vllm-monitoring.yaml`)

vLLM 서버는 기본적으로 `/metrics` 엔드포인트를 통해 Prometheus 포맷의 LLM 추론 지표를 노출합니다. Prometheus가 이를 자동으로 긁어갈 수 있도록 `Service`와 `ServiceMonitor`를 배포합니다.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-qwen-service
  namespace: mlops
  labels:
    app: vllm-qwen-prod
spec:
  selector:
    app: vllm-qwen-prod
  ports:
    - name: http
      port: 8000
      targetPort: 8000
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-qwen-servicemonitor
  namespace: monitoring
  labels:
    release: kube-prometheus-stack
spec:
  namespaceSelector:
    matchNames:
      - mlops
  selector:
    matchLabels:
      app: vllm-qwen-prod
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

---

## 3. Grafana 대시보드 핵심 모니터링 지표 (PromQL 쿼리)

Grafana에 접속한 뒤 아래 핵심 PromQL 쿼리를 패널로 등록하여 운영 대시보드를 구성합니다.

### 3.1. LLM 추론 성능 및 병목 지표 (vLLM)

| 모니터링 항목 | PromQL 쿼리 예시 | 실무 관제 기준 및 조치 |
| :--- | :--- | :--- |
| **초당 생성 토큰 수 (TPS)** | `rate(vllm:generation_tokens_total[1m])` | 급감 시 GPU 쓰로틀링 또는 배치 크기 병목 점검 |
| **첫 토큰 생성 지연시간 (TTFT P95)** | `histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))` | 2초 초과 시 프롬프트 길이 제한 또는 GPU 노드 증설 |
| **토큰당 생성 시간 (TPOT P95)** | `histogram_quantile(0.95, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le))` | 스트리밍 응답 체감 속도 저하 감지 |
| **KV 캐시 사용률 (%)** | `vllm:gpu_cache_usage_perc * 100` | 90% 이상 지속 시 요청 대기열(Queue) 급증 위험 |
| **대기 중인 요청 수 (Queue)** | `vllm:num_requests_waiting` | 0보다 큰 상태가 지속되면 Karpenter 기반 HPA 스케일아웃 필요 |

### 3.2. GPU 하드웨어 인프라 지표 (NVIDIA DCGM)

| 모니터링 항목 | PromQL 쿼리 예시 | 실무 관제 기준 및 조치 |
| :--- | :--- | :--- |
| **GPU 연산 사용률 (%)** | `avg(DCGM_FI_DEV_GPU_UTIL) by (kubernetes_node)` | 학습 파드가 95% 이상 유지되는지, 서빙 파드가 유휴 상태인지 확인 |
| **GPU VRAM 사용량 (MB)** | `DCGM_FI_DEV_FB_USED` | OOM(Out of Memory) 방지 및 `gpu-memory-utilization` 튜닝 근거 |
| **GPU 온도 (°C)** | `DCGM_FI_DEV_GPU_TEMP` | 85°C 이상 지속 시 하드웨어 노후화 또는 인스턴스 교체 검토 |

---

## 4. 트래픽 급증 시 vLLM 자동 확장(HPA + Karpenter) 연동 팁

Grafana 대시보드에서 `vllm:num_requests_waiting`(대기 요청 수)이 급증하는 패턴이 확인되면, **Prometheus Adapter**와 **Kubernetes HPA(Horizontal Pod Autoscaler)**를 연결해 vLLM 파드 수를 자동으로 늘릴 수 있습니다.

1. **트래픽 급증:** `vllm:num_requests_waiting` 값이 임계치(예: 5개)를 초과합니다.
2. **HPA 동작:** HPA가 `vllm-qwen-prod` Deployment의 `replicas`를 1에서 2로 증가시킵니다.
3. **Karpenter 자동 프로비저닝:** 새로 생성된 vLLM 파드가 `Pending` 상태가 되면, Karpenter가 즉시 새로운 GPU 노드(`g5.xlarge` 등)를 클러스터에 추가해 추론 처리량을 2배로 확장합니다.
4. **트래픽 감소 후 축소:** 트래픽이 줄어들면 HPA가 파드를 1개로 줄이고, Karpenter가 유휴 GPU 노드를 자동으로 반납하여 비용을 최적화합니다.