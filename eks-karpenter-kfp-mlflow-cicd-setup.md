# AWS EKS + Karpenter 기반 Kubeflow & MLflow 인프라 설치 및 GitHub Actions 연동 가이드

이 문서는 이전 가이드(`eks-karpenter-kubeflow-llm-mlops-guide.md`)를 실제 AWS EKS 클러스터에서 바로 실행하기 위한 **인프라 설치 스크립트(MLflow, Kubeflow Pipelines Standalone)**와 **GitHub Actions 기반 CI/CD 파이프라인 자동 트리거 설정**을 다룹니다.

---

## 1. 전체 CI/CD 및 인프라 연동 흐름

```mermaid
flowchart LR
    Dev[개발자 Git Push / PR] --> GHA[GitHub Actions Runner]
    GHA -->|1. Docker 빌드 및 ECR 푸시| ECR[(Amazon ECR)]
    GHA -->|2. KFP API 호출| KFP[Kubeflow Pipelines
Standalone on EKS]
    KFP -->|3. GPU Pod 생성 요청| Karpenter[Karpenter Autoscaler]
    Karpenter -->|4. g5 GPU 노드 프로비저닝| Node[EC2 GPU Node]
    Node -->|5. 학습/검증 메트릭 & 모델 저장| MLflow[MLflow Server
RDS + S3]
    MLflow -->|6. @Production 승격| vLLM[vLLM Deployment
모델 자동 롤아웃]
```

---

## 2. 사전 준비: AWS S3, RDS(PostgreSQL) 및 IAM(IRSA) 설정

MLflow와 Kubeflow가 아티팩트를 저장하고 메타데이터를 기록할 백엔드 스토리지를 준비합니다.

```bash
export AWS_REGION="ap-northeast-2"
export CLUSTER_NAME="my-eks-cluster"
export BUCKET_NAME="my-company-mlops-artifacts-$(date +%s)"

# 1. 아티팩트 저장용 S3 버킷 생성
aws s3 mb s3://${BUCKET_NAME} --region ${AWS_REGION}

# 2. IRSA(IAM Roles for Service Accounts) 활성화 (미설정 시 실행)
eksctl utils associate-iam-oidc-provider --cluster ${CLUSTER_NAME} --approve

# 3. MLflow 및 KFP Pod가 S3에 접근할 수 있도록 ServiceAccount 생성
eksctl create iamserviceaccount \
  --name mlops-sa \
  --namespace mlops \
  --cluster ${CLUSTER_NAME} \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
  --approve \
  --override-existing-serviceaccounts
```

*(참고: 프로덕션 환경에서는 `AmazonS3FullAccess` 대신 특정 S3 버킷(`arn:aws:s3:::${BUCKET_NAME}/*`)에만 접근 가능한 최소 권한 IAM 정책을 연결하는 것을 권장합니다.)*

---

## 3. MLflow Tracking Server 배포 (Helm / Kubernetes)

팀원들과 Kubeflow 파이프라인이 공통으로 접속할 중앙 MLflow 서버를 EKS의 `mlops` 네임스페이스에 배포합니다.

### 3.1 `mlflow-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: mlops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      serviceAccountName: mlops-sa
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:v2.17.2
          ports:
            - containerPort: 5000
          env:
            - name: AWS_DEFAULT_REGION
              value: "ap-northeast-2"
            # RDS PostgreSQL 접속 정보 (실제 환경에서는 AWS Secrets Manager 권장)
            - name: BACKEND_STORE_URI
              value: "postgresql://mlflow_user:mlflow_pass@my-rds-endpoint.ap-northeast-2.rds.amazonaws.com:5432/mlflow_db"
            - name: ARTIFACT_ROOT
              value: "s3://my-company-mlops-artifacts/mlflow"
          command: ["/bin/bash", "-c"]
          args:
            - |
              pip install psycopg2-binary boto3 && \
              mlflow server \
                --host 0.0.0.0 \
                --port 5000 \
                --backend-store-uri ${BACKEND_STORE_URI} \
                --default-artifact-root ${ARTIFACT_ROOT}
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "1000m"
              memory: "2Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: mlops
spec:
  selector:
    app: mlflow-server
  ports:
    - protocol: TCP
      port: 5000
      targetPort: 5000
  type: ClusterIP
```

```bash
kubectl apply -f mlflow-deployment.yaml
```

---

## 4. Kubeflow Pipelines (Standalone) 설치

전체 Kubeflow 플랫폼(Istio, Dex 등 포함) 대신, **파이프라인 오케스트레이션 기능만 가볍고 빠르게 EKS에 올릴 수 있는 Standalone 버전**을 설치합니다.

```bash
export PIPELINE_VERSION="2.3.0"

# 1. 클러스터 스코프 리소스 설치
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# 2. Kubeflow Pipelines 네임스페이스 및 컴포넌트 배포 (kubeflow 네임스페이스에 설치됨)
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"

# 3. 파드 정상 기동 확인 (모든 파드가 Running이 될 때까지 약 2~3분 대기)
kubectl get pods -n kubeflow
```

### 로컬에서 Kubeflow Pipelines UI 접속 테스트
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# 브라우저에서 http://localhost:8080 접속
```

---

## 5. GitHub Actions에서 Kubeflow 파이프라인 자동 트리거하기

개발자가 학습 코드(`train.py`)나 데이터셋 설정을 수정하고 `main` 브랜치에 Push하면, GitHub Actions가 **① 새 도커 이미지를 빌드해 Amazon ECR에 올리고 ② EKS 내부의 Kubeflow Pipelines API를 호출해 학습 파이프라인을 자동 실행**합니다.

### 5.1 파이프라인 실행 트리거 스크립트 (`scripts/trigger_kfp.py`)

```python
import os
import kfp
from pipeline import llm_finetuning_pipeline

def main():
    kfp_endpoint = os.environ["KFP_ENDPOINT"]  # 예: http://localhost:8080 또는 내부 ALB 주소
    image_uri = os.environ["IMAGE_URI"]
    s3_data_path = os.environ.get("S3_DATA_PATH", "s3://my-company-mlops-artifacts/data/train.jsonl")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service.mlops.svc.cluster.local:5000")

    # 1. KFP 클라이언트 연결
    client = kfp.Client(host=kfp_endpoint)

    # 2. 실험(Experiment) 생성 또는 조회
    experiment = client.create_experiment(name="llm-full-finetuning-ci")

    # 3. 파이프라인 실행(Run) 트리거
    run = client.create_run_from_pipeline_func(
        llm_finetuning_pipeline,
        experiment_name=experiment.display_name,
        run_name=f"ci-run-{os.environ.get('GITHUB_SHA', 'manual')[:7]}",
        arguments={
            "s3_data_path": s3_data_path,
            "mlflow_uri": mlflow_uri,
            "image_uri": image_uri,
        }
    )
    print(f"Kubeflow Pipeline Run 생성 완료! Run ID: {run.run_id}")

if __name__ == "__main__":
    main()
```

### 5.2 GitHub Actions 워크플로우 파일 (`.github/workflows/mlops-train-pipeline.yml`)

GitHub Actions Runner가 EKS 내부의 Kubeflow Pipelines API(`ml-pipeline.kubeflow.svc.cluster.local:8888`)에 안전하게 접근할 수 있도록 **AWS EKS 토큰 인증 후 `kubectl port-forward`를 통해 트리거**하는 표준 패턴입니다.

```yaml
name: Build Image & Trigger Kubeflow LLM Fine-Tuning

on:
  push:
    branches:
      - main
    paths:
      - 'src/**'
      - 'pipeline.py'
      - 'Dockerfile'

env:
  AWS_REGION: ap-northeast-2
  ECR_REPOSITORY: llm-finetune-worker
  EKS_CLUSTER_NAME: my-eks-cluster

jobs:
  build-and-trigger-kfp:
    runs-on: ubuntu-latest
    permissions:
      id-token: write   # AWS OIDC 인증용
      contents: read

    steps:
      - name: 1. Checkout Repository
        uses: actions/checkout@v4

      - name: 2. Configure AWS Credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-mlops-role
          aws-region: ${{ env.AWS_REGION }}

      - name: 3. Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: 4. Build, Tag, and Push Docker Image to ECR
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image_uri=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

      - name: 5. Update kubeconfig for EKS
        run: |
          aws eks update-kubeconfig --region ${{ env.AWS_REGION }} --name ${{ env.EKS_CLUSTER_NAME }}

      - name: 6. Trigger Kubeflow Pipeline Run
        env:
          IMAGE_URI: ${{ steps.build-image.outputs.image_uri }}
          KFP_ENDPOINT: "http://127.0.0.1:8888"
        run: |
          pip install kfp==2.9.0
          # 백그라운드로 KFP API 서비스 포트 포워딩 실행
          kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888 &
          PF_PID=$!
          sleep 5
          
          # 파이프라인 실행 요청
          python scripts/trigger_kfp.py
          
          # 포트 포워딩 프로세스 종료
          kill $PF_PID
```

---

## 6. 운영 팁: `@Production` 모델 승격 시 vLLM 자동 재시작(Rollout) 연동

평가 단계(`evaluate.py`)에서 성능 기준을 통과해 MLflow 레지스트리의 `@Production` 별칭이 새 모델 버전으로 바뀌었을 때, 배포된 vLLM 서버가 새 가중치를 즉시 불러오도록 하려면 다음 두 가지 방법 중 하나를 선택합니다.

1. **파이프라인 마지막 단계에 K8s Rollout 명령 추가 (가장 직관적).**
   `evaluate.py`가 `@Production` 승격을 완료한 직후, Kubernetes Python 클라이언트(`kubernetes.client`)를 호출해 `kubectl rollout restart deployment/vllm-summarizer -n mlops`와 동일한 효과를 내도록 트리거합니다.
2. **Argo CD / GitOps 연동.**
   파이프라인이 Git 레포지토리의 `values.yaml`(모델 버전 태그)을 업데이트하고 PR을 생성하면, Argo CD가 변경을 감지해 vLLM 파드를 무중단(Rolling Update)으로 교체합니다.
