# EKS + Kubeflow로 Qwen3-8B 파인튜닝부터 RAG까지

> 오픈 모델(Qwen3-8B)을 LoRA로 파인튜닝하고 평가한 뒤 vLLM으로 서빙하고, 그 위에 RAG 앱을 올립니다. 전 과정은 MLflow로 추적합니다.
> 플랫폼: Amazon EKS, Kubeflow Pipelines(KFP), Kubeflow Trainer · 소요 시간: 약 3시간 30분

## 0. 개요

### 0.1 시나리오

사내 문서 QA 봇에서 자주 생기는 두 가지 문제를 파인튜닝으로 줄여 봅니다.

1. 검색된 문서에 답이 없는데도 그럴듯한 답을 지어낸다.
2. 짧게 답하면 될 질문에 장황하게 답한다.

학습 데이터는 SQuAD v2(CC BY-SA 4.0)입니다. 학습셋은 약 1/3, 검증셋은 절반이 문단 안에 답이 없는 질문입니다. 그래서 모르면 모른다고 답하도록 가르칠 수 있습니다.

학습 입력에는 정답 문단 1개와 다른 문서의 문단 2개를 섞고 번호를 붙입니다(RAFT 방식). 실제 RAG처럼 여러 문서 중에서 근거를 골라 짧게 답하는 법을 익히게 하려는 것입니다.

### 0.2 아키텍처

```text
[Kubeflow Pipelines]
  prepare_data ──▶ train_lora ──▶ evaluate ──▶ promote (only if LoRA beats base)
       │               │              │             │
       │        TrainJob (Kubeflow    │             │
       │        Trainer, torchrun     │             │
       │        DDP on N GPU nodes)   │             │
       ▼               ▼              ▼             ▼
  S3: data/ ── adapter/ ── mlflow-artifacts/ ── serving/CURRENT
                                                    │
[Serving]                                           ▼
  user ──▶ RAG app (BM25) ──▶ vLLM (Qwen3-8B + LoRA adapter)
               │
               └── traces ──▶ MLflow ◀── params / loss / eval metrics
```

| 단계 | 실행 위치 | MLflow에 남는 것 |
|---|---|---|
| 데이터 준비 | KFP 컴포넌트 (CPU) | 데이터셋 크기, 답 없는 질문 비율, KFP run ID |
| 학습 | Kubeflow Trainer TrainJob (GPU × N) | 하이퍼파라미터, loss·lr·처리량 곡선, GPU 사용률, LoRA 어댑터 |
| 평가 | KFP 컴포넌트 (GPU, vLLM 오프라인 추론) | base와 LoRA 지표, 예측 샘플 50개 |
| 배포 판단 | KFP 조건 분기 | `promoted` 태그 |
| RAG 요청 | RAG 앱 | 요청별 trace (검색 문서, 생성 결과, 지연 시간) |
| RAG 평가 | 로컬 스크립트 | 모델별 RAG 지표, 검색 recall@k |

KFP는 단계 순서, 조건 분기, 재실행을 맡습니다. Kubeflow Trainer는 멀티노드 torchrun 환경(MASTER_ADDR, RANK 등)을 맡습니다.

### 0.3 버전 (2026년 11월 기준 최신 릴리스로 고정)

| 구성 요소 | 버전 | 비고 |
|---|---|---|
| Amazon EKS | eksctl 기본값 | Kubeflow Trainer는 Kubernetes 1.31 이상 필요 |
| Kubeflow Pipelines | 2.18.0 | standalone 설치 |
| Kubeflow Trainer | 2.4.0 | `TrainJob` API (`trainer.kubeflow.org/v1alpha1`) |
| MLflow | 3.18.0 | 서버 이미지 `ghcr.io/mlflow/mlflow:v3.18.0-full` |
| vLLM | v0.33.0 | CUDA 13.0 이미지. 노드 드라이버가 CUDA 13을 지원해야 함 |
| PEFT | 0.21.3 | |
| 모델 | `Qwen/Qwen3-8B` | Apache 2.0 |

워크샵에서는 컨테이너 이미지를 직접 빌드하지 않습니다. 공개 이미지를 쓰고, 필요한 패키지는 파드가 시작할 때 설치하고, 코드는 ConfigMap으로 넣습니다. 프로덕션에서는 이미지를 빌드해 ECR에 올리세요.

### 0.4 시간표와 비용

| Lab | 내용 | 시간 |
|---|---|---|
| 0 | 사전 준비 | 15분 |
| 1 | EKS 클러스터 | 30분 (생성 대기 약 20분) |
| 2 | KFP, Trainer, MLflow 설치 | 30분 |
| 3 | 학습 코드와 분산 학습 스모크 테스트 | 20분 |
| 4 | 파이프라인 실행 | 50분 (학습 대기 약 30분) |
| 5 | vLLM 서빙 | 20분 |
| 6 | RAG 앱과 MLflow Tracing | 20분 |
| 7 | RAG 평가 | 15분 |
| 8 | 정리 | 10분 |

GPU 노드는 g6e.2xlarge(NVIDIA L40S 48GB) 2대, CPU 노드는 m6i.2xlarge 2대입니다. 쉬는 시간에는 GPU 노드를 0대로 줄이세요. 실습이 끝나면 Lab 8로 모든 리소스를 삭제해야 합니다.

```bash
eksctl scale nodegroup --cluster $CLUSTER_NAME --name gpu --nodes 0   # 일시 중지
eksctl scale nodegroup --cluster $CLUSTER_NAME --name gpu --nodes 2   # 재개
```

---

## Lab 0. 사전 준비 (15분)

### 0-1. 도구

- AWS CLI v2, eksctl(최신), kubectl 1.31 이상, Helm 3.8 이상, jq, envsubst(gettext), Python 3.10 이상
- EKS, IAM, EC2, S3 리소스를 만들 수 있는 AWS 권한
- Docker는 필요 없습니다.

### 0-2. 디렉터리 구조

```text
llm-workshop/
├── env.sh
├── infra/      cluster.yaml, s3-policy.json, storageclass.yaml, gpu-check.yaml
├── platform/   base.yaml, mlflow.yaml
├── code/       common.py, train.py, rag_app.py, rag_eval.py   (ConfigMap으로 배포)
├── training/   ddp-smoke.yaml
├── pipeline/   pipeline.py
└── serving/    vllm.yaml, rag.yaml
```

### 0-3. 환경 변수

`env.sh`

```bash
export AWS_REGION=us-west-2          # 서울(ap-northeast-2)을 쓰려면 0-4에서 g6e 제공 여부부터 확인
export CLUSTER_NAME=llm-workshop
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET=llm-workshop-${ACCOUNT_ID}-${AWS_REGION}
export KFP_VERSION=2.18.0
export TRAINER_VERSION=2.4.0
```

```bash
cd llm-workshop && source env.sh     # 새 터미널을 열 때마다 다시 실행
```

### 0-4. GPU 가용 영역과 쿼터 확인

두 GPU 노드를 같은 AZ에 둡니다. 노드 간 NCCL 통신 지연과 AZ 간 전송 비용을 줄이기 위해서입니다.

```bash
export GPU_AZ=$(aws ec2 describe-instance-type-offerings --region $AWS_REGION \
  --location-type availability-zone --filters Name=instance-type,Values=g6e.2xlarge \
  --query 'InstanceTypeOfferings[0].Location' --output text)
export OTHER_AZ=$(aws ec2 describe-availability-zones --region $AWS_REGION \
  --query "AvailabilityZones[?ZoneName!='${GPU_AZ}' && State=='available'] | [0].ZoneName" --output text)
echo "GPU_AZ=$GPU_AZ OTHER_AZ=$OTHER_AZ"     # GPU_AZ가 None이면 다른 리전을 고르세요
echo "export GPU_AZ=$GPU_AZ OTHER_AZ=$OTHER_AZ" >> env.sh

# "Running On-Demand G and VT instances" 쿼터(vCPU). g6e.2xlarge 2대면 16 이상 필요
aws service-quotas get-service-quota --region $AWS_REGION \
  --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value'
```

### 0-5. S3 버킷과 IAM 정책

데이터, LoRA 어댑터, MLflow 아티팩트를 모두 이 버킷에 둡니다. 파드는 EKS Pod Identity를 통해 이 버킷에만 접근합니다.

`infra/s3-policy.json`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::${BUCKET}" },
    { "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/*" }
  ]
}
```

```bash
aws s3 mb s3://$BUCKET --region $AWS_REGION
envsubst < infra/s3-policy.json > /tmp/s3-policy.json
export S3_POLICY_ARN=$(aws iam create-policy --policy-name ${CLUSTER_NAME}-s3 \
  --policy-document file:///tmp/s3-policy.json --query Policy.Arn --output text)
echo "export S3_POLICY_ARN=$S3_POLICY_ARN" >> env.sh
```

### 0-6. 로컬 Python 환경

파이프라인 제출과 RAG 평가에 씁니다.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "kfp==2.18.0" "kfp-kubernetes==2.18.0" "mlflow-skinny==3.18.0" requests
```

---

## Lab 1. EKS 클러스터 (30분)

### 1-1. 클러스터 정의

`infra/cluster.yaml`

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
availabilityZones: ["${GPU_AZ}", "${OTHER_AZ}"]

iam:
  # 파드별 S3 권한(EKS Pod Identity). 서비스 어카운트는 나중에 만들어져도 된다.
  podIdentityAssociations:
    - namespace: kubeflow          # KFP 파이프라인 단계 파드
      serviceAccountName: pipeline-runner
      permissionPolicyARNs: ["${S3_POLICY_ARN}"]
    - namespace: llm               # TrainJob, vLLM, RAG 앱
      serviceAccountName: llm-workload
      permissionPolicyARNs: ["${S3_POLICY_ARN}"]
    - namespace: mlflow            # MLflow 서버(아티팩트 저장)
      serviceAccountName: mlflow
      permissionPolicyARNs: ["${S3_POLICY_ARN}"]

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: eks-pod-identity-agent
  - name: aws-ebs-csi-driver       # KFP와 MLflow의 PVC용
    useDefaultPodIdentityAssociations: true

managedNodeGroups:
  - name: cpu
    instanceType: m6i.2xlarge
    amiFamily: AmazonLinux2023
    desiredCapacity: 2
    minSize: 2
    maxSize: 3
    volumeSize: 100
    privateNetworking: true
  - name: gpu
    instanceType: g6e.2xlarge        # NVIDIA L40S 48GB x 1
    amiFamily: AmazonLinux2023       # GPU 인스턴스면 eksctl이 가속 AMI와 NVIDIA device plugin을 자동 설치
    availabilityZones: ["${GPU_AZ}"]
    desiredCapacity: 2
    minSize: 0
    maxSize: 2
    volumeSize: 300                  # vLLM 이미지와 모델 가중치(약 16GB) 캐시
    privateNetworking: true
    taints:                          # GPU가 필요 없는 파드는 GPU 노드에 오지 못하게 한다
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
```

```bash
envsubst < infra/cluster.yaml > /tmp/cluster.yaml
eksctl create cluster -f /tmp/cluster.yaml          # 약 20분
kubectl get nodes -L eks.amazonaws.com/nodegroup
```

### 1-2. 기본 StorageClass

KFP(MySQL, 오브젝트 스토어)와 MLflow가 PVC를 씁니다.

`infra/storageclass.yaml`

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer
parameters:
  type: gp3
  encrypted: "true"
```

```bash
kubectl apply -f infra/storageclass.yaml
kubectl annotate storageclass gp2 storageclass.kubernetes.io/is-default-class- 2>/dev/null || true
kubectl get storageclass
```

### 1-3. GPU 확인과 이미지 미리 받기

학습, 평가, 서빙에 모두 vLLM 이미지를 씁니다. 세 단계가 같은 PyTorch·transformers 버전을 쓰게 되고, GPU 노드마다 큰 이미지를 한 번만 받으면 됩니다.

`infra/gpu-check.yaml`

```yaml
# GPU 노드마다 vLLM 이미지를 미리 받아 둔다
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: prepull-vllm
  namespace: default
spec:
  selector:
    matchLabels: {app: prepull-vllm}
  template:
    metadata:
      labels: {app: prepull-vllm}
    spec:
      nodeSelector:
        eks.amazonaws.com/nodegroup: gpu
      tolerations:
        - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
      containers:
        - name: pull
          image: vllm/vllm-openai:v0.33.0
          command: ["sleep", "infinity"]
          resources:
            requests: {cpu: 10m, memory: 32Mi}
---
# 노드 드라이버가 CUDA 13을 지원하는지 확인
apiVersion: v1
kind: Pod
metadata:
  name: gpu-check
  namespace: default
spec:
  restartPolicy: Never
  tolerations:
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  containers:
    - name: smi
      image: vllm/vllm-openai:v0.33.0
      command: ["nvidia-smi"]
      resources:
        limits: {nvidia.com/gpu: 1}
```

```bash
kubectl get nodes -l eks.amazonaws.com/nodegroup=gpu \
  -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'   # 노드마다 1

kubectl apply -f infra/gpu-check.yaml
kubectl rollout status ds/prepull-vllm --timeout=30m
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/gpu-check --timeout=30m
kubectl logs gpu-check | head -15        # "CUDA Version: 13.x" 이상인지 확인
kubectl delete -f infra/gpu-check.yaml   # 이미지는 노드에 캐시된 채로 남는다
```

**확인 포인트**
- GPU allocatable 값이 노드마다 1이다.
- `nvidia-smi` 출력에 CUDA 13 이상이 보인다. 그보다 낮으면 부록 A를 보세요.

---

## Lab 2. 플랫폼 설치 (30분)

### 2-1. Kubeflow Pipelines (standalone)

```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$KFP_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$KFP_VERSION"
kubectl -n kubeflow wait --for=condition=Available deployment --all --timeout=15m
```

### 2-2. Kubeflow Trainer

```bash
helm install kubeflow-trainer oci://ghcr.io/kubeflow/charts/kubeflow-trainer \
  --namespace kubeflow-system --create-namespace \
  --version ${TRAINER_VERSION} \
  --set runtimes.torchDistributed.enabled=true

kubectl get pods -n kubeflow-system        # jobset-controller, kubeflow-trainer-controller
kubectl get clustertrainingruntimes        # torch-distributed
```

### 2-3. 워크로드 네임스페이스와 권한

KFP 단계 파드는 `kubeflow` 네임스페이스의 `pipeline-runner`로 실행됩니다. 이 서비스 어카운트가 `llm` 네임스페이스에서 TrainJob을 만들고 vLLM을 재시작할 수 있게 합니다.

`platform/base.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: llm-workload
  namespace: llm
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: kfp-orchestrator
  namespace: llm
rules:
  - apiGroups: ["trainer.kubeflow.org"]
    resources: ["trainjobs"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kfp-orchestrator
  namespace: llm
subjects:
  - kind: ServiceAccount
    name: pipeline-runner
    namespace: kubeflow
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: kfp-orchestrator
```

```bash
kubectl apply -f platform/base.yaml
```

### 2-4. MLflow 서버

메타데이터는 PVC 위의 SQLite에, 아티팩트는 S3에 저장합니다. 클라이언트는 MLflow 서버를 통해 아티팩트를 올리므로, 학습 파드가 MLflow 때문에 S3 권한을 따로 가질 필요는 없습니다.

MLflow 3.5부터 서버가 Host 헤더를 검사합니다. 클러스터 내부 DNS 이름을 `--allowed-hosts`에 넣지 않으면 파드에서 보낸 요청이 403으로 거절됩니다. 프로브도 같은 이유로 HTTP 대신 TCP를 씁니다.

`platform/mlflow.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlflow
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlflow
  namespace: mlflow
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-db
  namespace: mlflow
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests: {storage: 20Gi}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlflow
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector:
    matchLabels: {app: mlflow}
  template:
    metadata:
      labels: {app: mlflow}
    spec:
      serviceAccountName: mlflow
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:v3.18.0-full
          command: ["mlflow", "server"]
          args:
            - --host=0.0.0.0
            - --port=5000
            - --backend-store-uri=sqlite:////data/mlflow.db
            - --artifacts-destination=s3://${BUCKET}/mlflow-artifacts
            - --allowed-hosts=mlflow.mlflow.svc.cluster.local:5000,mlflow.mlflow:5000,localhost:*,127.0.0.1:*
          env:
            - {name: AWS_DEFAULT_REGION, value: "${AWS_REGION}"}
          ports:
            - {containerPort: 5000}
          readinessProbe:
            tcpSocket: {port: 5000}
            periodSeconds: 10
          resources:
            requests: {cpu: 500m, memory: 1Gi}
          volumeMounts:
            - {name: data, mountPath: /data}
      volumes:
        - name: data
          persistentVolumeClaim: {claimName: mlflow-db}
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: mlflow
spec:
  selector: {app: mlflow}
  ports:
    - {port: 5000, targetPort: 5000}
```

```bash
envsubst '${BUCKET} ${AWS_REGION}' < platform/mlflow.yaml | kubectl apply -f -
kubectl -n mlflow rollout status deploy/mlflow --timeout=10m
kubectl -n mlflow exec deploy/mlflow -- python -c "import boto3; print('boto3 OK')"
```

### 2-5. UI 접속과 권한 확인

```bash
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80 >/dev/null &   # KFP:    http://localhost:8080
kubectl -n mlflow port-forward svc/mlflow 5000:5000 >/dev/null &           # MLflow: http://localhost:5000
eksctl get podidentityassociation --cluster $CLUSTER_NAME                  # 3개 + EBS CSI
```

**확인 포인트**
- 브라우저에서 KFP UI와 MLflow UI가 열린다.
- Pod Identity 연결에 `kubeflow/pipeline-runner`, `llm/llm-workload`, `mlflow/mlflow`가 있다.

---

## Lab 3. 학습 코드와 분산 학습 스모크 테스트 (20분)

### 3-1. 공통 모듈

학습, 평가, RAG 앱이 같은 프롬프트 형식과 지표를 쓰도록 한 파일로 묶습니다. 학습 때와 서빙 때 프롬프트가 조금만 달라도 LoRA 효과가 줄어듭니다.

`code/common.py`

```python
"""Prompt format and metrics shared by training, evaluation and the RAG app."""
import collections
import re
import string

NO_ANSWER = "I don't know."
PUNCT = set(string.punctuation)

SYSTEM_PROMPT = (
    "You answer questions using only the numbered documents. "
    "Reply with the shortest exact phrase from the documents that answers the question. "
    f"If the documents do not contain the answer, reply exactly: {NO_ANSWER}"
)

def build_messages(question, contexts):
    docs = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {question}"},
    ]

def target_text(row):
    return row["answers"][0] if row["answers"] else NO_ANSWER

def is_refusal(text):
    t = text.replace("\u2019", "'").lower()
    return "i don't know" in t or "i do not know" in t

def normalize(text):
    text = "".join(ch for ch in text.lower() if ch not in PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def score_predictions(rows, preds):
    """SQuAD v2 style metrics. A correct refusal on an unanswerable question scores 1."""
    em, f1, false_refusal, no_ans = [], [], [], []
    for row, pred in zip(rows, preds):
        refused = is_refusal(pred)
        if row["answers"]:
            false_refusal.append(float(refused))
            em.append(0.0 if refused else max(float(normalize(pred) == normalize(a)) for a in row["answers"]))
            f1.append(0.0 if refused else max(token_f1(pred, a) for a in row["answers"]))
        else:
            no_ans.append(float(refused))
    return {
        "has_ans_em": _mean(em),
        "has_ans_f1": _mean(f1),
        "false_refusal_rate": _mean(false_refusal),
        "no_ans_acc": _mean(no_ans),
        "overall_f1": (sum(f1) + sum(no_ans)) / max(1, len(f1) + len(no_ans)),
    }
```

| 지표 | 의미 |
|---|---|
| `has_ans_em`, `has_ans_f1` | 답이 있는 질문에서 정답 완전 일치율, 토큰 F1 |
| `no_ans_acc` | 답이 없는 질문에 "I don't know."로 답한 비율 |
| `false_refusal_rate` | 답이 있는데 모른다고 답한 비율 |
| `overall_f1` | SQuAD v2 방식 종합 점수. 배포 판단에 사용 |

### 3-2. 학습 스크립트

HF Trainer 대신 짧은 학습 루프를 직접 씁니다. torchrun이 넘겨주는 분산 환경, DDP의 gradient 동기화 시점, MLflow 기록 위치를 코드에서 바로 확인할 수 있습니다.

- LoRA r=16, attention과 MLP의 모든 선형층에 적용 (학습 파라미터 약 4,400만 개, 전체의 약 0.5%)
- bf16, gradient checkpointing, gradient accumulation 중에는 `no_sync()`로 통신 생략
- loss는 답변 토큰에만 계산 (프롬프트 토큰은 -100)
- MLflow 기록은 rank 0만 담당

`code/train.py`

```python
"""LoRA fine-tuning of Qwen3-8B with DDP. Launched by torchrun inside a Kubeflow TrainJob."""
import contextlib
import json
import math
import os
import sys
import time

import boto3
import mlflow
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_messages, target_text  # noqa: E402

CFG = {
    "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"),
    "epochs": int(os.environ.get("EPOCHS", "1")),
    "lr": float(os.environ.get("LR", "2e-4")),
    "micro_batch": int(os.environ.get("MICRO_BATCH", "4")),
    "grad_accum": int(os.environ.get("GRAD_ACCUM", "4")),
    "max_len": int(os.environ.get("MAX_LEN", "1024")),
    "lora_r": int(os.environ.get("LORA_R", "16")),
    "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
    "warmup_steps": int(os.environ.get("WARMUP_STEPS", "10")),
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def s3_split(uri):
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key

def read_jsonl(uri):
    bucket, key = s3_split(uri)
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]

def upload_dir(local_dir, uri):
    bucket, prefix = s3_split(uri)
    s3 = boto3.client("s3")
    for name in os.listdir(local_dir):
        s3.upload_file(os.path.join(local_dir, name), bucket, f"{prefix.rstrip('/')}/{name}")


"""
row = {
    "question": "입사 1년 미만 직원의 연차는 며칠인가요?",
    "contexts": [
        "입사 1년 미만 근로자는 1개월 개근 시 1일의 유급휴가가 발생합니다.",
        "연차 유급휴가는 발생일로부터 1년간 사용할 수 있습니다."
    ],
    "answers": ["월 1일"]   # 답이 없는 부정 샘플일 경우 빈 리스트 []
}
"""
def encode(tok, row):
    # Same chat template as inference: thinking disabled, answer followed by <|im_end|>.
    prompt = tok.apply_chat_template(build_messages(row["question"], row["contexts"]),
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(target_text(row) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > CFG["max_len"]:
        return None
    return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}


"""
collate 결과

[input_ids]      (빈칸은 pad_id로 채움)
[ [10, 20, 30, 40, 50],
  [11, 22, 33, pad, pad] ]

[attention_mask] (글자 있는 곳은 1, 패딩 빈칸은 0)
[ [ 1,  1,  1,  1,  1],
  [ 1,  1,  1,  0,  0] ]

[labels]         (패딩 빈칸은 -100으로 유지해 오차 계산 제외)
[ [-100, -100, -100, 40, 50],
  [-100, -100,   33, -100, -100] ]


배치 크기가 2인 경우 예시
batch = [
    # 0번째 데이터 (encode 함수의 반환값 1건)
    {
        "input_ids": [151644, 8948, 198, ..., 151645],
        "labels":    [-100,   -100, -100, ..., 151645]
    },
    # 1번째 데이터 (encode 함수의 반환값 1건)
    {
        "input_ids": [151644, 8948, 198, ..., 151645],
        "labels":    [-100,   -100, -100, ..., 151645]
    }
]
"""
def collate(batch, pad_id):
    width = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        labels[i, :n] = torch.tensor(x["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}

def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    prefix = os.environ["S3_PREFIX"]

    tok = AutoTokenizer.from_pretrained(CFG["base_model"])
    rows = read_jsonl(f"{prefix}/data/train.jsonl")
    data = [e for e in (encode(tok, r) for r in rows) if e is not None]

    # device_map은 거대한 모델을 여러 장치(GPU 0, GPU 1, CPU 등)에 쪼개어 배치하기 위해 {"모듈_이름": 장치_번호} 형태로 작성하는 지도(Map).
    model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], dtype=torch.bfloat16,
                                                 device_map={"": local_rank})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                                             lora_dropout=0.05, target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    ddp = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=42)
    loader = DataLoader(data, batch_size=CFG["micro_batch"], sampler=sampler,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    trainable = [p for p in ddp.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=CFG["lr"], weight_decay=0.0)
    total_steps = CFG["epochs"] * (len(loader) // CFG["grad_accum"])

    def lr_lambda(step):  # linear warmup, then cosine decay
        if step < CFG["warmup_steps"]:
            return (step + 1) / CFG["warmup_steps"]
        progress = (step - CFG["warmup_steps"]) / max(1, total_steps - CFG["warmup_steps"])
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if rank == 0:
        mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"], log_system_metrics=True)
        mlflow.log_params({**CFG, "world_size": world, "train_examples": len(data),
                           "skipped_too_long": len(rows) - len(data), "total_steps": total_steps,
                           "trainable_params": sum(p.numel() for p in trainable)})

    step, start = 0, time.time()
    ddp.train()
    for epoch in range(CFG["epochs"]):
        sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        run_loss = torch.zeros(1, device="cuda")
        run_tokens = torch.zeros(1, device="cuda")
        t0 = time.time()
        for i, batch in enumerate(loader):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            sync = (i + 1) % CFG["grad_accum"] == 0
            # Skip the gradient all-reduce on accumulation micro-steps.
            with contextlib.nullcontext() if sync else ddp.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = ddp(**batch).loss / CFG["grad_accum"]
                loss.backward()
            run_loss += loss.detach()
            run_tokens += batch["attention_mask"].sum()
            if not sync:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            stats = torch.cat([run_loss, run_tokens])
            dist.all_reduce(stats)
            if rank == 0:
                metrics = {"train_loss": stats[0].item() / world, "lr": sched.get_last_lr()[0],
                           "tokens_per_sec": stats[1].item() / (time.time() - t0)}
                mlflow.log_metrics(metrics, step=step)
                print(f"step {step}/{total_steps} " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()),
                      flush=True)
            run_loss.zero_()
            run_tokens.zero_()
            t0 = time.time()

    if rank == 0:
        out_dir = "/tmp/adapter"
        ddp.module.save_pretrained(out_dir)               # adapter_config.json + adapter_model.safetensors
        upload_dir(out_dir, f"{prefix}/adapter")          # for evaluation and serving
        mlflow.log_artifacts(out_dir, artifact_path="adapter")
        mlflow.log_metric("train_runtime_sec", time.time() - start)
        mlflow.end_run()
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
```

* TIP 
```
import inspect

# 원본 모델의 forward 함수가 받는 모든 파라미터 이름 출력
print(inspect.signature(model.forward))
```




### 3-3. 코드를 ConfigMap으로 배포

TrainJob과 RAG 앱은 `llm` 네임스페이스에서, KFP 평가 단계는 `kubeflow` 네임스페이스에서 이 코드를 마운트합니다. 코드를 고칠 때마다 다시 실행하세요.

```bash
# rag_app.py와 rag_eval.py는 Lab 6, 7에서 작성합니다. 그때 이 명령을 다시 실행하세요.
for ns in llm kubeflow; do
  kubectl create configmap workshop-code --from-file=code/ -n $ns --dry-run=client -o yaml | kubectl apply -f -
done
```

### 3-4. 멀티노드 DDP 스모크 테스트

본 학습 전에 TrainJob이 GPU 노드 2대에 torchrun 환경을 제대로 만드는지, 노드 간 NCCL 통신이 되는지 1분 안에 확인합니다.

`training/ddp-smoke.yaml`

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: ddp-smoke
  namespace: llm
spec:
  runtimeRef:
    name: torch-distributed
  trainer:
    image: vllm/vllm-openai:v0.33.0
    numNodes: 2
    numProcPerNode: 1
    resourcesPerNode:
      limits:
        nvidia.com/gpu: 1
    command: ["bash", "-c"]
    args:
      - |
        cat > /tmp/smoke.py <<'EOF'
        import os, socket, torch, torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        t = torch.tensor([rank + 1.0], device="cuda")
        dist.all_reduce(t)
        print(f"host={socket.gethostname()} rank={rank}/{dist.get_world_size()} "
              f"gpu={torch.cuda.get_device_name()} all_reduce={t.item()}", flush=True)
        dist.destroy_process_group()
        EOF
        python3 -m torch.distributed.run /tmp/smoke.py
  # 런타임을 바꾸지 않고 파드 설정만 덧붙인다. torch-distributed 런타임의 컨테이너 이름은 node
  runtimePatches:
    - manager: workshop.example.com/manual
      trainingRuntimeSpec:
        template:
          spec:
            replicatedJobs:
              - name: node
                template:
                  spec:
                    template:
                      spec:
                        tolerations:
                          - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
```

```bash
kubectl apply -f training/ddp-smoke.yaml
kubectl get trainjob -n llm ddp-smoke -w            # STATE가 Complete가 되면 Ctrl+C
kubectl logs -n llm -l jobset.sigs.k8s.io/jobset-name=ddp-smoke --prefix --tail=-1 | grep all_reduce
kubectl delete -f training/ddp-smoke.yaml
```

**확인 포인트**
- 두 줄이 나오고 둘 다 `all_reduce=3.0`이다 (rank 0의 1과 rank 1의 2를 더한 값).
- torchrun에 노드 수나 마스터 주소를 넘기지 않았습니다. Trainer가 `PET_NNODES`, `PET_NODE_RANK`, `PET_MASTER_ADDR` 같은 환경 변수를 주입하고 torchrun이 그 값을 읽습니다.

---

## Lab 4. 파이프라인: 데이터 → 학습 → 평가 → 배포 판단 (50분)

### 4-1. 파이프라인 정의

| 단계 | 하는 일 |
|---|---|
| `prepare_data` | SQuAD v2를 RAFT 형식으로 변환해 S3에 저장. MLflow run 생성 후 run ID 반환 |
| `train_lora` | TrainJob을 만들고 끝날 때까지 기다린 뒤 어댑터 경로 반환 |
| `evaluate` | vLLM 오프라인 추론으로 base와 LoRA를 같은 평가셋에서 채점 |
| `promote` | LoRA가 `min_gain` 이상 좋을 때만 실행. `serving/CURRENT`를 갱신하고 vLLM 재시작 |

모든 단계가 MLflow run 하나에 기록되므로 데이터, 학습, 평가 결과가 한 화면에서 이어집니다.

`pipeline/pipeline.py`

```python
"""Qwen3-8B RAFT-style LoRA pipeline: prepare -> train (TrainJob) -> evaluate -> promote."""
import os
from typing import NamedTuple

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Metrics, Output

VLLM_IMAGE = "vllm/vllm-openai:v0.33.0"
PY_IMAGE = "python:3.12-slim"
MLFLOW_PKG = "mlflow-skinny==3.18.0"

@dsl.component(base_image=PY_IMAGE, packages_to_install=["datasets", "boto3", MLFLOW_PKG])
def prepare_data(bucket: str, region: str, mlflow_uri: str, experiment: str, base_model: str,
                 n_train: int, n_eval: int, num_distractors: int, kfp_run_id: str) -> str:
    """Convert SQuAD v2 to RAFT format (gold paragraph + distractors) and open an MLflow run."""
    import json
    import random

    import boto3
    import mlflow
    from datasets import load_dataset

    rnd = random.Random(42)
    ds = load_dataset("rajpurkar/squad_v2")

    def unique_contexts(split):
        seen = {}
        for title, ctx in zip(ds[split]["title"], ds[split]["context"]):
            seen.setdefault(ctx, title)
        return [(title, ctx) for ctx, title in seen.items()]

    def build(split, n, pool):
        rows = []
        for r in ds[split].shuffle(seed=42).select(range(n)):
            distractors = []
            while len(distractors) < num_distractors:   # paragraphs from other articles only
                title, ctx = rnd.choice(pool)
                if title != r["title"] and ctx not in distractors:
                    distractors.append(ctx)
            contexts = [r["context"], *distractors]
            rnd.shuffle(contexts)
            rows.append({"id": r["id"], "question": r["question"], "contexts": contexts,
                         "gold_context": r["context"],
                         "answers": list(dict.fromkeys(r["answers"]["text"]))})  # [] = unanswerable
        return rows

    train_rows = build("train", n_train, unique_contexts("train"))
    val_pool = unique_contexts("validation")
    eval_rows = build("validation", n_eval, val_pool)
    corpus = [{"doc_id": f"val-{i:04d}", "title": t, "text": c} for i, (t, c) in enumerate(val_pool)]
    doc_id_of = {d["text"]: d["doc_id"] for d in corpus}
    for r in eval_rows:
        r["gold_doc_id"] = doc_id_of[r["gold_context"]]

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"kfp-{kfp_run_id[:8]}") as run:
        mlflow.set_tags({"kfp_run_id": kfp_run_id, "base_model": base_model})
        mlflow.log_params({
            "dataset": "rajpurkar/squad_v2", "n_train": len(train_rows), "n_eval": len(eval_rows),
            "num_distractors": num_distractors, "corpus_docs": len(corpus),
            "train_unanswerable_ratio": round(sum(not r["answers"] for r in train_rows) / len(train_rows), 3),
        })
        run_id = run.info.run_id

    s3 = boto3.client("s3", region_name=region)

    def put_jsonl(key, rows):
        s3.put_object(Bucket=bucket, Key=key,
                      Body="\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode())

    put_jsonl(f"runs/{run_id}/data/train.jsonl", train_rows)
    put_jsonl(f"runs/{run_id}/data/eval.jsonl", eval_rows)
    put_jsonl("datasets/squad_v2/corpus.jsonl", corpus)          # RAG corpus (Lab 6)
    print(f"run_id={run_id} train={len(train_rows)} eval={len(eval_rows)} corpus={len(corpus)}")
    return run_id

@dsl.component(base_image=PY_IMAGE, packages_to_install=["kubernetes"])
def train_lora(run_id: str, bucket: str, region: str, mlflow_uri: str, image: str,
               base_model: str, namespace: str, num_nodes: int, epochs: int, lr: float,
               micro_batch: int, grad_accum: int) -> str:
    """Create a Kubeflow Trainer TrainJob and block until it completes."""
    import time

    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    config.load_incluster_config()
    api = client.CustomObjectsApi()
    group, version, plural = "trainer.kubeflow.org", "v1alpha1", "trainjobs"
    name = f"qwen3-lora-{run_id[:12]}"
    prefix = f"s3://{bucket}/runs/{run_id}"

    env = {
        "BASE_MODEL": base_model, "EPOCHS": str(epochs), "LR": str(lr),
        "MICRO_BATCH": str(micro_batch), "GRAD_ACCUM": str(grad_accum),
        "S3_PREFIX": prefix, "AWS_DEFAULT_REGION": region,
        "MLFLOW_TRACKING_URI": mlflow_uri, "MLFLOW_RUN_ID": run_id,
        "PYTHONUNBUFFERED": "1", "PIP_BREAK_SYSTEM_PACKAGES": "1",
    }
    script = "\n".join([
        "set -euo pipefail",
        "python3 -m pip install --no-cache-dir -q 'peft==0.21.3' 'mlflow-skinny==3.18.0' "
        "boto3 psutil nvidia-ml-py",
        "python3 -m torch.distributed.run /workspace/code/train.py",
    ])
    pod_patch = {
        "serviceAccountName": "llm-workload",
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "volumes": [
            {"name": "code", "configMap": {"name": "workshop-code"}},
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
        ],
        "containers": [{"name": "node", "volumeMounts": [
            {"name": "code", "mountPath": "/workspace/code"},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ]}],
    }
    body = {
        "apiVersion": f"{group}/{version}",
        "kind": "TrainJob",
        "metadata": {"name": name, "namespace": namespace, "labels": {"mlflow-run-id": run_id}},
        "spec": {
            "runtimeRef": {"name": "torch-distributed"},
            "trainer": {
                "image": image,
                "command": ["bash", "-c", script],
                "numNodes": num_nodes,
                "numProcPerNode": 1,
                "resourcesPerNode": {
                    "requests": {"cpu": "6", "memory": "48Gi"},
                    "limits": {"nvidia.com/gpu": "1", "memory": "56Gi"},
                },
                "env": [{"name": k, "value": v} for k, v in env.items()],
            },
            "runtimePatches": [{
                "manager": "workshop.example.com/kfp",
                "trainingRuntimeSpec": {"template": {"spec": {"replicatedJobs": [{
                    "name": "node",
                    "template": {"spec": {"template": {"spec": pod_patch}}},
                }]}}},
            }],
        },
    }

    try:
        api.create_namespaced_custom_object(group, version, namespace, plural, body)
        print(f"created TrainJob {namespace}/{name}")
    except ApiException as e:
        if e.status != 409:      # 409: already created by a previous attempt of this step
            raise
        print(f"TrainJob {name} already exists, watching it")

    while True:
        job = api.get_namespaced_custom_object(group, version, namespace, plural, name)
        conds = {c["type"]: c for c in job.get("status", {}).get("conditions", [])}
        if conds.get("Complete", {}).get("status") == "True":
            return f"{prefix}/adapter"
        if conds.get("Failed", {}).get("status") == "True":
            raise RuntimeError(f"TrainJob failed: {conds['Failed'].get('message')}")
        print("waiting:", {k: v.get("status") for k, v in conds.items()}, flush=True)
        time.sleep(60)

@dsl.component(base_image=VLLM_IMAGE, packages_to_install=[MLFLOW_PKG, "boto3"])
def evaluate(run_id: str, bucket: str, region: str, mlflow_uri: str, base_model: str,
             adapter_uri: str, min_gain: float, metrics: Output[Metrics]) -> NamedTuple(
                 "EvalOutputs", [("decision", str), ("base_overall_f1", float), ("ft_overall_f1", float)]):
    """Score base vs LoRA on the held-out set with vLLM offline inference."""
    import json
    import os
    import sys
    from collections import namedtuple

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.path.insert(0, "/workspace/code")
    import boto3
    import mlflow
    from common import build_messages, score_predictions
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    s3 = boto3.client("s3", region_name=region)
    body = s3.get_object(Bucket=bucket, Key=f"runs/{run_id}/data/eval.jsonl")["Body"].read().decode()
    rows = [json.loads(line) for line in body.splitlines() if line.strip()]

    adapter_dir = "/tmp/adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    a_bucket, _, a_prefix = adapter_uri.removeprefix("s3://").partition("/")
    for obj in s3.list_objects_v2(Bucket=a_bucket, Prefix=f"{a_prefix}/")["Contents"]:
        s3.download_file(a_bucket, obj["Key"], os.path.join(adapter_dir, os.path.basename(obj["Key"])))

    tok = AutoTokenizer.from_pretrained(base_model)
    prompts = [tok.apply_chat_template(build_messages(r["question"], r["contexts"]), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in rows]
    llm = LLM(model=base_model, enable_lora=True, max_lora_rank=16, max_model_len=4096,
              gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=64)
    base_preds = [o.outputs[0].text.strip() for o in llm.generate(prompts, params)]
    ft_preds = [o.outputs[0].text.strip() for o in llm.generate(
        prompts, params, lora_request=LoRARequest("ft", 1, adapter_dir))]

    base_m, ft_m = score_predictions(rows, base_preds), score_predictions(rows, ft_preds)
    mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_base_{k}": v for k, v in base_m.items()})
        mlflow.log_metrics({f"eval_ft_{k}": v for k, v in ft_m.items()})
        mlflow.log_dict({"samples": [
            {"question": r["question"], "gold": r["answers"] or ["<no answer>"], "base": b, "ft": f}
            for r, b, f in zip(rows[:50], base_preds, ft_preds)]}, "eval/samples.json")

    for k, v in base_m.items():
        metrics.log_metric(f"base_{k}", round(v, 4))
    for k, v in ft_m.items():
        metrics.log_metric(f"ft_{k}", round(v, 4))

    decision = "promote" if ft_m["overall_f1"] >= base_m["overall_f1"] + min_gain else "skip"
    print(json.dumps({"base": base_m, "ft": ft_m, "decision": decision}, indent=2))
    outputs = namedtuple("EvalOutputs", ["decision", "base_overall_f1", "ft_overall_f1"])
    return outputs(decision, base_m["overall_f1"], ft_m["overall_f1"])

@dsl.component(base_image=PY_IMAGE, packages_to_install=["boto3", "kubernetes", MLFLOW_PKG])
def promote(run_id: str, bucket: str, region: str, adapter_uri: str, mlflow_uri: str, namespace: str):
    """Point serving/CURRENT at the new adapter and roll the vLLM deployment."""
    import boto3
    import mlflow
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    boto3.client("s3", region_name=region).put_object(
        Bucket=bucket, Key="serving/CURRENT", Body=adapter_uri.encode())
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.MlflowClient().set_tag(run_id, "promoted", "true")

    config.load_incluster_config()
    patch = {"spec": {"template": {"metadata": {"annotations": {"workshop/adapter-run": run_id}}}}}
    try:
        client.AppsV1Api().patch_namespaced_deployment("vllm", namespace, patch)
        print("vLLM rollout triggered")
    except ApiException as e:
        if e.status != 404:
            raise
        print("vLLM deployment not found yet; it will read serving/CURRENT on first start")

@dsl.pipeline(name="qwen3-8b-raft-lora")
def qwen3_raft_pipeline(
    bucket: str,
    region: str,
    mlflow_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    experiment: str = "qwen3-8b-squad-raft",
    base_model: str = "Qwen/Qwen3-8B",
    namespace: str = "llm",
    n_train: int = 4000,
    n_eval: int = 1000,
    num_distractors: int = 2,
    num_nodes: int = 1,
    epochs: int = 1,
    lr: float = 2e-4,
    micro_batch: int = 4,
    grad_accum: int = 4,
    min_gain: float = 0.02,
):
    prep = prepare_data(bucket=bucket, region=region, mlflow_uri=mlflow_uri, experiment=experiment,
                        base_model=base_model, n_train=n_train, n_eval=n_eval,
                        num_distractors=num_distractors, kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER)

    train = train_lora(run_id=prep.output, bucket=bucket, region=region, mlflow_uri=mlflow_uri,
                       image=VLLM_IMAGE, base_model=base_model, namespace=namespace,
                       num_nodes=num_nodes, epochs=epochs, lr=lr,
                       micro_batch=micro_batch, grad_accum=grad_accum)

    ev = evaluate(run_id=prep.output, bucket=bucket, region=region, mlflow_uri=mlflow_uri,
                  base_model=base_model, adapter_uri=train.output, min_gain=min_gain)
    ev.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")
    ev.set_cpu_request("4").set_memory_request("32Gi")
    ev.set_env_variable("PIP_BREAK_SYSTEM_PACKAGES", "1")
    kubernetes.add_toleration(ev, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    kubernetes.use_config_map_as_volume(ev, config_map_name="workshop-code", mount_path="/workspace/code")

    with dsl.If(ev.outputs["decision"] == "promote"):
        pr = promote(run_id=prep.output, bucket=bucket, region=region,
                     adapter_uri=train.output, mlflow_uri=mlflow_uri, namespace=namespace)
        pr.set_caching_options(False)

    for task in (prep, train, ev):
        task.set_caching_options(False)

if __name__ == "__main__":
    import kfp

    compiler.Compiler().compile(qwen3_raft_pipeline, "qwen3_raft_pipeline.yaml")
    client = kfp.Client(host=os.environ.get("KFP_HOST", "http://localhost:8080"))
    run = client.create_run_from_pipeline_func(
        qwen3_raft_pipeline,
        arguments={"bucket": os.environ["BUCKET"], "region": os.environ["AWS_REGION"],
                   "num_nodes": int(os.environ.get("NUM_NODES", "1"))},
        experiment_name="qwen3-workshop",
        run_name="qwen3-raft-lora",
    )
    print("KFP run id:", run.run_id)
```

**설계 포인트**
- `train_lora`는 GPU를 쓰지 않습니다. TrainJob을 만들고 상태만 확인합니다. GPU는 TrainJob 파드가 씁니다.
- 평가에는 HF `generate` 대신 vLLM 오프라인 추론을 씁니다. 1,000문항 × 2개 모델 채점이 몇 분 안에 끝납니다.
- 서빙할 어댑터 경로는 `serving/CURRENT` 파일 하나로 관리합니다. 롤백할 때는 이 파일만 이전 경로로 되돌리면 됩니다.

### 4-2. 실행

아직 vLLM을 띄우지 않았으므로 GPU 2장을 모두 학습에 씁니다.

```bash
NUM_NODES=2 python pipeline/pipeline.py
```

### 4-3. 관찰

```bash
kubectl get trainjobs -n llm
TJ=$(kubectl get trainjobs -n llm --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
kubectl get pods -n llm -l jobset.sigs.k8s.io/jobset-name=$TJ -o wide       # 두 GPU 노드에 하나씩
kubectl logs -n llm -l jobset.sigs.k8s.io/jobset-name=$TJ --prefix -f      # step 10/125 train_loss=...
```

- KFP UI: 실행 그래프와 `evaluate` 단계의 Metrics 아티팩트
- MLflow UI → `qwen3-8b-squad-raft` → `kfp-xxxxxxxx` run
  - Model metrics: `train_loss`, `lr`, `tokens_per_sec` 곡선
  - System metrics: GPU 사용률, GPU 메모리
  - 평가 후: `eval_base_*`, `eval_ft_*` 지표, `eval/samples.json`, `adapter/` 아티팩트

예상 소요 시간(2노드 기준): 모델 다운로드와 패키지 설치 5분 내외, 학습 125 step 15~20분, 평가 10분 내외.

### 4-4. 결과 기록

| 지표 | base | LoRA |
|---|---|---|
| has_ans_em | | |
| has_ans_f1 | | |
| no_ans_acc | | |
| false_refusal_rate | | |
| overall_f1 | | |

수치는 환경마다 다르니 직접 채워 보세요. 예상되는 경향은 이렇습니다.
- base는 문장으로 길게 답하는 경우가 많아 EM이 낮습니다. LoRA는 짧은 구절로 답합니다. 개선분 일부는 정확도가 아니라 답변 형식이 맞춰진 효과라는 점을 함께 설명하세요.
- 답 없는 질문에서는 base가 문서 밖 지식으로 답을 지어내는 경우가 많습니다. LoRA는 `no_ans_acc`가 크게 오릅니다.
- `false_refusal_rate`가 같이 올라가면 지나치게 몸을 사리는 모델이 된 것입니다. 두 지표를 함께 보세요.

---

## Lab 5. vLLM 서빙 (20분)

base 모델과 LoRA 어댑터를 vLLM 하나로 함께 서빙합니다. 요청의 `model` 값으로 `qwen3-8b`(base)와 `qwen3-8b-squad`(LoRA) 중 하나를 고릅니다.

`serving/vllm.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
  namespace: llm
spec:
  replicas: 1
  strategy:
    type: Recreate        # GPU 1장짜리 파드라 롤링 업데이트를 하면 GPU가 1장 더 필요하다
  selector:
    matchLabels: {app: vllm}
  template:
    metadata:
      labels: {app: vllm}
    spec:
      serviceAccountName: llm-workload
      tolerations:
        - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
      initContainers:
        - name: fetch-adapter      # serving/CURRENT가 가리키는 어댑터를 받는다
          image: public.ecr.aws/aws-cli/aws-cli:latest
          command: ["sh", "-c"]
          args:
            - |
              set -e
              SRC=$(aws s3 cp s3://${BUCKET}/serving/CURRENT -)
              echo "adapter: $SRC"
              aws s3 sync "$SRC" /adapters/qwen3-squad
          env:
            - {name: AWS_DEFAULT_REGION, value: "${AWS_REGION}"}
          volumeMounts:
            - {name: adapters, mountPath: /adapters}
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.33.0
          command: ["vllm", "serve"]
          args:
            - Qwen/Qwen3-8B
            - --served-model-name
            - qwen3-8b
            - --enable-lora
            - --lora-modules
            - qwen3-8b-squad=/adapters/qwen3-squad
            - --max-lora-rank
            - "16"
            - --max-model-len
            - "8192"
            - --gpu-memory-utilization
            - "0.90"
            - --port
            - "8000"
          ports:
            - {name: http, containerPort: 8000}
          resources:
            requests: {cpu: "4", memory: 32Gi}
            limits: {nvidia.com/gpu: 1, memory: 48Gi}
          volumeMounts:
            - {name: adapters, mountPath: /adapters}
            - {name: hf-cache, mountPath: /root/.cache/huggingface}
            - {name: dshm, mountPath: /dev/shm}
          startupProbe:              # 모델 다운로드와 로딩에 수 분이 걸린다
            httpGet: {path: /health, port: 8000}
            periodSeconds: 10
            failureThreshold: 120
          readinessProbe:
            httpGet: {path: /health, port: 8000}
            periodSeconds: 10
      volumes:
        - {name: adapters, emptyDir: {}}
        - {name: hf-cache, emptyDir: {sizeLimit: 60Gi}}
        - {name: dshm, emptyDir: {medium: Memory, sizeLimit: 8Gi}}
---
apiVersion: v1
kind: Service
metadata:
  name: vllm
  namespace: llm
spec:
  selector: {app: vllm}
  ports:
    - {name: http, port: 8000, targetPort: 8000}
```

```bash
envsubst '${BUCKET} ${AWS_REGION}' < serving/vllm.yaml | kubectl apply -f -
kubectl -n llm logs deploy/vllm -c fetch-adapter
kubectl -n llm rollout status deploy/vllm --timeout=30m
```

### 5-1. base와 LoRA를 같은 질문으로 비교

```bash
kubectl -n llm port-forward svc/vllm 8000:8000 >/dev/null &
curl -s localhost:8000/v1/models | jq -r '.data[].id'        # qwen3-8b, qwen3-8b-squad

SYSTEM="You answer questions using only the numbered documents. Reply with the shortest exact phrase from the documents that answers the question. If the documents do not contain the answer, reply exactly: I don't know."
DOC="[1] The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France."

ask() {   # ask <model> <question>
  jq -n --arg m "$1" --arg s "$SYSTEM" \
        --arg u "$(printf 'Documents:\n%s\n\nQuestion: %s' "$DOC" "$2")" '{
    model: $m, temperature: 0, max_tokens: 32,
    chat_template_kwargs: {enable_thinking: false},
    messages: [{role: "system", content: $s}, {role: "user", content: $u}]}' |
  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @- |
  jq -r '.choices[0].message.content'
}

for m in qwen3-8b qwen3-8b-squad; do
  for q in "In what country is Normandy located?" "Who was the king of Norway in 1066?"; do
    printf '%-15s | %-38s | %s\n' "$m" "$q" "$(ask "$m" "$q")"
  done
done
```

**확인 포인트**
- 두 번째 질문의 답은 문서에 없습니다. base는 사전 지식으로 답을 지어낼 수 있고, LoRA는 `I don't know.`로 답해야 합니다.
- Qwen3는 기본으로 thinking 모드입니다. 학습할 때와 똑같이 `enable_thinking: false`를 넘겨야 합니다.

---

## Lab 6. RAG 앱과 MLflow Tracing (20분)

검색은 BM25를 씁니다. SQuAD 질문은 문단과 단어가 많이 겹쳐서 BM25로도 잘 찾고, 임베딩 모델을 따로 띄울 필요가 없습니다. 코퍼스는 SQuAD v2 검증셋의 문단 약 1,200개이며, Lab 4의 `prepare_data`가 S3에 저장해 두었습니다.

`code/rag_app.py`

```python
"""BM25 retrieval + vLLM generation, traced with MLflow."""
import json
import os
import re
import sys

import boto3
import mlflow
from fastapi import FastAPI
from mlflow.entities import SpanType
from openai import OpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_messages  # noqa: E402

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "qwen3-rag-app"))

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def load_corpus(uri):
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]

DOCS = load_corpus(os.environ["CORPUS_URI"])
BM25 = BM25Okapi([tokenize(d["text"]) for d in DOCS])
client = OpenAI(base_url=os.environ["VLLM_URL"], api_key="not-used")
app = FastAPI()

class AskRequest(BaseModel):
    question: str
    model: str = "qwen3-8b-squad"     # "qwen3-8b" for the base model
    top_k: int = 3

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str, top_k: int):
    scores = BM25.get_scores(tokenize(question))
    best = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"id": DOCS[i]["doc_id"], "page_content": DOCS[i]["text"],
             "metadata": {"title": DOCS[i]["title"], "score": float(scores[i])}} for i in best]

@mlflow.trace(span_type=SpanType.LLM)
def generate(model: str, question: str, contexts: list):
    resp = client.chat.completions.create(
        model=model, messages=build_messages(question, contexts), temperature=0, max_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    span = mlflow.get_current_active_span()
    if span and resp.usage:
        span.set_attributes({"prompt_tokens": resp.usage.prompt_tokens,
                             "completion_tokens": resp.usage.completion_tokens})
    return resp.choices[0].message.content.strip()

@mlflow.trace(span_type=SpanType.CHAIN, name="rag_ask")
def rag(question: str, model: str, top_k: int):
    docs = retrieve(question, top_k)
    answer = generate(model, question, [d["page_content"] for d in docs])
    return {"answer": answer, "model": model, "doc_ids": [d["id"] for d in docs],
            "titles": [d["metadata"]["title"] for d in docs]}

@app.post("/ask")
def ask(req: AskRequest):
    return rag(req.question, req.model, req.top_k)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "docs": len(DOCS)}
```

`serving/rag.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-app
  namespace: llm
spec:
  replicas: 1
  selector:
    matchLabels: {app: rag-app}
  template:
    metadata:
      labels: {app: rag-app}
    spec:
      serviceAccountName: llm-workload      # S3에서 코퍼스를 읽는다
      containers:
        - name: app
          image: python:3.12-slim
          workingDir: /workspace/code
          command: ["sh", "-c"]
          args:
            - >-
              pip install --no-cache-dir -q fastapi uvicorn openai rank-bm25 boto3 mlflow-skinny==3.18.0
              && exec uvicorn rag_app:app --host 0.0.0.0 --port 8080
          env:
            - {name: VLLM_URL, value: "http://vllm.llm.svc.cluster.local:8000/v1"}
            - {name: MLFLOW_TRACKING_URI, value: "http://mlflow.mlflow.svc.cluster.local:5000"}
            - {name: MLFLOW_EXPERIMENT, value: qwen3-rag-app}
            - {name: CORPUS_URI, value: "s3://${BUCKET}/datasets/squad_v2/corpus.jsonl"}
            - {name: AWS_DEFAULT_REGION, value: "${AWS_REGION}"}
          ports:
            - {containerPort: 8080}
          startupProbe:
            httpGet: {path: /healthz, port: 8080}
            periodSeconds: 10
            failureThreshold: 30
          resources:
            requests: {cpu: "1", memory: 2Gi}
          volumeMounts:
            - {name: code, mountPath: /workspace/code}
      volumes:
        - name: code
          configMap: {name: workshop-code}
---
apiVersion: v1
kind: Service
metadata:
  name: rag-app
  namespace: llm
spec:
  selector: {app: rag-app}
  ports:
    - {port: 8080, targetPort: 8080}
```

```bash
# rag_app.py를 추가했으니 ConfigMap 갱신 (3-3과 같은 명령)
for ns in llm kubeflow; do
  kubectl create configmap workshop-code --from-file=code/ -n $ns --dry-run=client -o yaml | kubectl apply -f -
done
envsubst '${BUCKET} ${AWS_REGION}' < serving/rag.yaml | kubectl apply -f -
kubectl -n llm rollout status deploy/rag-app --timeout=10m

kubectl -n llm port-forward svc/rag-app 8081:8080 >/dev/null &
for m in qwen3-8b qwen3-8b-squad; do
  for q in "In what country is Normandy located?" "Who was the king of Norway in 1066?"; do
    jq -n --arg q "$q" --arg m "$m" '{question: $q, model: $m}' |
      curl -s localhost:8081/ask -H 'Content-Type: application/json' -d @- | jq -c '{model, answer, titles}'
  done
done
```

### 6-1. Trace 보기

MLflow UI → `qwen3-rag-app` 실험 → **Traces** 탭

- `rag_ask` 아래에 `retrieve`(RETRIEVER)와 `generate`(LLM) span이 있습니다.
- `retrieve`에서 어떤 문단이 몇 점으로 검색됐는지 봅니다. `generate`에서는 입력 프롬프트, 답변, 토큰 수, 지연 시간을 봅니다.
- 답이 틀렸을 때 검색이 문제인지 생성이 문제인지 trace로 나눠서 봅니다.

---

## Lab 7. RAG 평가 (15분)

Lab 4의 평가에서는 정답 문단을 항상 넣어 줬습니다. 여기서는 BM25가 찾은 문단만으로 답하게 해서 실제 서비스에 가까운 조건으로 비교합니다.

`code/rag_eval.py` (로컬에서 실행)

```python
"""Compare base vs LoRA through the RAG app and log the results to MLflow."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import mlflow
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import score_predictions  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("eval_file")
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--top-k", type=int, default=3)
parser.add_argument("--rag-url", default="http://localhost:8081")
parser.add_argument("--training-run-id", default="")
args = parser.parse_args()

with open(args.eval_file) as f:
    rows = [json.loads(line) for line in f if line.strip()][: args.n]

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("qwen3-rag-eval")

for model in ("qwen3-8b", "qwen3-8b-squad"):
    def ask(row):
        resp = requests.post(f"{args.rag_url}/ask", timeout=120,
                             json={"question": row["question"], "model": model, "top_k": args.top_k})
        resp.raise_for_status()
        return resp.json()

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(ask, rows))

    metrics = score_predictions(rows, [r["answer"] for r in results])
    metrics["retrieval_recall_at_k"] = sum(
        row["gold_doc_id"] in res["doc_ids"] for row, res in zip(rows, results)) / len(rows)
    with mlflow.start_run(run_name=f"rag-{model}"):
        mlflow.log_params({"model": model, "retriever": "bm25", "top_k": args.top_k, "n": len(rows)})
        if args.training_run_id:
            mlflow.set_tag("training_run_id", args.training_run_id)
        mlflow.log_metrics(metrics)
    print(model, {k: round(v, 3) for k, v in metrics.items()})
```

```bash
export RUN_ID=$(aws s3 cp s3://$BUCKET/serving/CURRENT - | cut -d/ -f5)
aws s3 cp s3://$BUCKET/runs/$RUN_ID/data/eval.jsonl /tmp/eval.jsonl
python code/rag_eval.py /tmp/eval.jsonl --n 300 --training-run-id $RUN_ID
```

MLflow UI → `qwen3-rag-eval`에서 두 run을 선택하고 **Compare**를 누릅니다.

**토론 포인트**
- `retrieval_recall_at_k`가 낮으면 모델을 바꿔도 개선에 한계가 있습니다. 검색 품질이 상한선을 정합니다.
- Lab 4 평가(정답 문단 보장)와 RAG 평가의 점수 차이가 곧 검색 단계의 손실입니다.
- 답 없는 질문의 근거 문단이 아닌 다른 문단에 우연히 답이 있을 수 있습니다. 드물지만 RAG 평가의 `no_ans_acc`에는 이런 잡음이 섞입니다.

---

## Lab 8. 정리 (10분)

PVC가 붙은 리소스를 먼저 지워야 EBS 볼륨이 남지 않습니다.

```bash
pkill -f "kubectl.*port-forward" || true
kubectl delete namespace llm mlflow --wait=true
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$KFP_VERSION" --ignore-not-found
helm uninstall kubeflow-trainer -n kubeflow-system
eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION --disable-nodegroup-eviction --wait
aws s3 rm s3://$BUCKET --recursive && aws s3 rb s3://$BUCKET
aws iam delete-policy --policy-arn $S3_POLICY_ARN
```

마지막으로 EC2 콘솔의 볼륨 화면에서 `available` 상태로 남은 EBS 볼륨이 없는지 확인하세요.

---

## 부록 A. 트러블슈팅

| 증상 | 원인과 조치 |
|---|---|
| TrainJob 파드가 `Pending` | GPU가 부족합니다. vLLM이 GPU 1장을 쓰고 있으면 `NUM_NODES=1`로 실행하세요. `kubectl describe pod`의 이벤트도 확인하세요. |
| MLflow 요청이 403 또는 `Invalid Host header` | `--allowed-hosts`에 접속하는 `호스트:포트`를 추가하세요. |
| S3 `AccessDenied` | Pod Identity 연결의 네임스페이스와 SA 이름을 확인하세요(`eksctl get podidentityassociation`). 연결을 만든 뒤에는 파드를 재시작해야 합니다. |
| `CUDA driver version is insufficient` | 노드 드라이버가 CUDA 13을 지원하지 않습니다. 최신 EKS 가속 AMI로 노드그룹을 교체하거나, CUDA 12용 이전 vLLM 이미지 태그를 쓰세요. |
| 학습 중 CUDA OOM | 파이프라인 인자를 `micro_batch=2`, `grad_accum=8`로 바꾸세요. A10G(24GB)라면 `micro_batch=1`, `grad_accum=16`부터 시작하세요. |
| pip `externally-managed-environment` 오류 | `PIP_BREAK_SYSTEM_PACKAGES=1`이 설정됐는지 확인하세요. |
| Hugging Face 다운로드 429 | `HF_TOKEN`을 Secret으로 만들어 env로 넣으세요. |
| vLLM init 컨테이너 실패 (`serving/CURRENT` 없음) | 파이프라인이 `skip`으로 판정한 경우입니다. `min_gain`을 낮춰 다시 실행하거나, `echo -n "s3://$BUCKET/runs/$RUN_ID/adapter" \| aws s3 cp - s3://$BUCKET/serving/CURRENT`로 직접 지정하세요. |
| TrainJob 생성이 webhook에서 거부됨 | `runtimePatches`의 컨테이너 이름을 확인하세요. `torch-distributed` 런타임은 `node`입니다. |

## 부록 B. 확장 과제

1. **재학습 루프**: vLLM을 띄운 상태에서 `lr`이나 `n_train`을 바꿔 `NUM_NODES=1`로 다시 실행합니다. 평가를 통과하면 `promote`가 vLLM을 새 어댑터로 재시작합니다.
2. **한국어로 바꾸기**: 한국어 QA 데이터(라이선스 확인 필요)로 교체하고, 다국어 임베딩(예: `BAAI/bge-m3`) 기반 dense 검색이나 BM25와의 하이브리드 검색으로 바꿉니다.
3. **MLflow GenAI 평가**: `mlflow.genai.evaluate`와 LLM judge로 답변이 근거에 기반했는지 채점합니다.
4. **GPU 큐잉**: Kueue로 TrainJob 대기열과 팀별 GPU 쿼터를 관리합니다.
5. **모델 로딩 시간 단축**: 가중치를 S3에 미리 두고 vLLM의 Run:ai Model Streamer(`--load-format runai_streamer`)로 스트리밍 로드합니다.
6. **멀티 LoRA**: 도메인별 어댑터 여러 개를 base 모델 하나에 올리고 `--max-loras`로 동시에 서빙합니다.
7. **이식성**: EKS 전용 부분(eksctl, Pod Identity, EBS)만 바꾸면 KFP, Trainer, MLflow, vLLM 구성은 다른 Kubernetes(예: CoreWeave CKS)와 S3 호환 스토리지에서도 그대로 쓸 수 있습니다.
