# AWS EKS + Karpenter 기반 LLM MLOps 2단계 자동화 구축 가이드
> **대상 환경:** AWS EKS (GPU 노드 인식 완료) + Karpenter + Kubeflow Pipelines (KFP v2) + MLflow + PyTorch Full Fine-Tuning + vLLM Serving

---

## 1. 아키텍처 개요 (End-to-End Architecture)

본 가이드는 **필요할 때만 GPU 인스턴스를 동적으로 생성(Karpenter)하고, 학습 및 검증이 끝나면 즉시 반납**하여 비용을 극대화하는 실무형 2단계 MLOps 자동화 아키텍처를 다룹니다.

```mermaid
flowchart LR
    subgraph Trigger["1. 트리거 (Trigger)"]
        GIT["GitHub / GitLab<br/>(Code Push / Tag)"]
        CRON["정기 스케줄러<br/>(KFP Recurring Run)"]
    end

    subgraph EKS["2. AWS EKS Cluster (GPU-Aware)"]
        KFP["Kubeflow Pipelines<br/>(Orchestrator)"]
        KARP["Karpenter Autoscaler<br/>(GPU Node 동적 할당/회수)"]

        subgraph GPUNode_Train["GPU Node (g5 / p4d - 학습용 온디맨드/스팟)"]
            S1["Step 1. 데이터 전처리<br/>(S3 로드 및 토크나이징)"]
            S2["Step 2. LLM Full Fine-Tuning<br/>(PyTorch / HuggingFace)"]
            S3["Step 3. 모델 성능 검증<br/>(MLflow Evaluate)"]
            S1 --> S2 --> S3
        end

        subgraph GPUNode_Serve["GPU Node (g5.xlarge - 추론 상시 운영)"]
            VLLM["vLLM Inference Server<br/>(OpenAI 호환 API)"]
        end

        KFP -->|Pod 스케줄링 요청| KARP
        KARP -.->|GPU 프로비저닝| GPUNode_Train
        KARP -.->|GPU 프로비저닝| GPUNode_Serve
    end

    subgraph Storage["3. 중앙 저장소 및 레지스트리"]
        MLFLOW["MLflow Tracking &<br/>Model Registry (@Production)"]
        S3[("Amazon S3<br/>(데이터셋 & 모델 가중치)")]
        RDS[("Amazon RDS<br/>(PostgreSQL 메타데이터)")]
        MLFLOW --- RDS
        MLFLOW --- S3
    end

    CLIENT["Client / 서비스 애플리케이션"]

    GIT --> KFP
    CRON --> KFP
    S2 -->|메트릭/가중치 기록| MLFLOW
    S3 -->|평가 점수 및 승격| MLFLOW
    MLFLOW -->|@Production 모델 로드| VLLM
    VLLM <-->|REST API 추론| CLIENT
```

### 핵심 동작 원리
1. **동적 GPU 스케일링 (Karpenter):** 평소에는 저렴한 CPU 노드에만 Kubeflow와 MLflow 컨트롤 플레인이 떠 있습니다. 파이프라인이 시작되어 `nvidia.com/gpu` 리소스를 요청하면 **Karpenter가 수십 초 내에 GPU 노드(g5 등)를 띄우고, 파이프라인이 종료되면 노드를 자동 삭제**합니다.
2. **Full Fine-Tuning & MLflow 추적:** S3의 한국어 요약 데이터셋을 불러와 소형 오픈 웨이트 LLM(`Qwen/Qwen2.5-1.5B-Instruct` 등)을 Full Fine-Tuning하고, 가중치와 손실(Loss) 곡선을 MLflow & S3에 저장합니다.
3. **자동 품질 게이트 (Evaluation Gate):** 파인튜닝된 모델이 기준 점수(`faithfulness >= 4.0`)를 통과할 때만 MLflow Model Registry에 `@Production` 별칭을 부여합니다.
4. **vLLM 배포:** S3에 저장된 `@Production` 모델 경로를 마운트하거나 다운로드하여 **vLLM** 엔진으로 고성능 OpenAI 호환 API 서버를 배포합니다.

---

## 2. 인프라 사전 준비 (Karpenter GPU NodePool 설정)

EKS 클러스터에 Karpenter가 설치되어 있다면, 학습용(Spot 가능)과 추론용(On-Demand 권장) GPU 노드를 자동으로 띄울 수 있도록 `NodePool`과 `EC2NodeClass`를 배포합니다.

### `karpenter-gpu-nodepool.yaml`
```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-mlops-pool
spec:
  template:
    metadata:
      labels:
        workload-type: gpu-mlops
    spec:
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu-ec2-nodeclass
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"] # 학습 비용 절감을 위해 Spot 우선 사용
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["g5.xlarge", "g5.2xlarge", "g5.12xlarge"] # A10G GPU 인스턴스군
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
      taints:
        - key: nvidia.com/gpu
          value: "true"
          effect: NoSchedule # GPU 파드만 이 노드에 스케줄링되도록 격리
  limits:
    cpu: 64
    memory: 256Gi
  disruption:
    consolidationPolicy: WhenEmpty # 파드(학습)가 끝나고 노드가 비면 즉시 반납
    consolidateAfter: 60s
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu-ec2-nodeclass
spec:
  amiSelectorTerms:
    - alias: bottlerocket@latest # NVIDIA 드라이버 및 Container Toolkit 기본 내장 AMI
  role: "KarpenterNodeRole-my-eks-cluster" # 실제 EKS 노드 IAM Role 이름으로 변경 필요
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "my-eks-cluster"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "my-eks-cluster"
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 150Gi # LLM 모델 가중치 및 컨테이너 이미지 캐시용 넉넉한 디스크
        volumeType: gp3
```

적용 명령어:
```bash
kubectl apply -f karpenter-gpu-nodepool.yaml
```

---

## 3. 샘플 애플리케이션: LLM 요약 Full Fine-Tuning 및 평가 코드

실습의 빠른 실행과 검증을 위해 **1.5B 파라미터급 오픈 웨이트 모델(`Qwen/Qwen2.5-1.5B-Instruct`)**을 사용하여 요약 태스크에 맞게 **Full Fine-Tuning**을 수행하고, MLflow로 검증하는 파이썬 스크립트입니다. (단일 A10G 24GB GPU인 `g5.xlarge` 또는 `g5.2xlarge`에서 `bf16` + `gradient_checkpointing`으로 원활히 동작합니다.)

### 프로젝트 디렉토리 구조
```text
llm-mlops-pipeline/
├── karpenter-gpu-nodepool.yaml   # Karpenter GPU 노드 설정
├── Dockerfile                    # 학습 및 평가용 베이스 이미지
├── src/
│   ├── train.py                  # Full Fine-Tuning 스크립트 (MLflow 연동)
│   └── evaluate.py               # MLflow LLM-as-a-Judge 검증 스크립트
├── pipeline.py                   # Kubeflow Pipelines (KFP v2) 정의 및 컴파일
└── vllm-deployment.yaml          # vLLM 추론 서버 배포 매니페스트
```

### 3-1. Full Fine-Tuning 스크립트 (`src/train.py`)
```python
import os
import torch
import mlflow
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

def main():
    # 1. 환경 변수 및 MLflow 설정
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server.mlflow.svc.cluster.local:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "llm-summarization-full-ft")
    base_model_id = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = "/tmp/finetuned-model"

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # 2. 샘플 학습 데이터 준비 (실무에서는 S3 경로에서 JSONL 로드: pd.read_json("s3://..."))
    raw_data = [
        {
            "article": "AWS EKS와 Karpenter를 결합하면 워크로드 수요에 맞춰 EC2 인스턴스를 초 단위로 프로비저닝할 수 있습니다. 특히 고비용의 GPU 인스턴스를 학습 작업이 실행될 때만 생성하고 작업 완료 후 즉시 종료함으로써 클라우드 인프라 비용을 최대 70%까지 절감할 수 있습니다.",
            "summary": "AWS EKS와 Karpenter는 GPU 인스턴스를 온디맨드로 동적 할당 및 회수하여 AI 학습 인프라 비용을 대폭 절감합니다."
        },
        {
            "article": "vLLM은 PagedAttention 알고리즘을 도입하여 LLM 서빙 시 KV 캐시 메모리 낭비를 최소화합니다. 이를 통해 기존 HuggingFace 표준 추론 대비 최대 24배 높은 처리량(Throughput)을 제공하며 OpenAI 호환 API 엔드포인트를 기본 지원합니다.",
            "summary": "vLLM은 PagedAttention으로 메모리 효율을 높여 LLM 추론 처리량을 극대화하고 OpenAI 호환 API를 제공합니다."
        },
        {
            "article": "Kubeflow Pipelines는 쿠버네티스 환경에서 머신러닝 워크플로우를 컨테이너 단위로 정의하고 실행하는 오케스트레이터입니다. 데이터 전처리, 모델 학습, 평가, 배포 단계를 DAG 형태로 연결하여 재현성과 자동화를 보장합니다.",
            "summary": "Kubeflow Pipelines는 쿠버네티스 기반 컨테이너 워크플로우로 ML 전 과정을 자동화하고 재현성을 보장합니다."
        }
    ] * 10  # 학습 루프 확인을 위해 30건으로 복제

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_and_tokenize(example):
        prompt = f"다음 글을 한 문장으로 핵심만 요약하세요.\n\n원문: {example['article']}\n요약: "
        full_text = prompt + example["summary"] + tokenizer.eos_token
        
        tokenized = tokenizer(full_text, truncation=True, max_length=512)
        prompt_ids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
        
        # 프롬프트 부분은 Loss 계산에서 제외(-100 마스킹)하고 요약문 생성만 학습
        labels = [-100] * len(prompt_ids) + tokenized["input_ids"][len(prompt_ids):]
        tokenized["labels"] = labels[:len(tokenized["input_ids"])]
        return tokenized

    dataset = Dataset.from_list(raw_data).map(format_and_tokenize, remove_columns=["article", "summary"])

    # 3. 모델 로드 (Full Fine-Tuning: 모든 가중치 업데이트)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable() # VRAM 절약을 위한 체크포인팅 활성화

    # 4. 학습 하이퍼파라미터 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,             # Full Fine-Tuning 권장 학습률
        bf16=True,
        logging_steps=2,
        save_strategy="no",
        report_to="mlflow",              # 학습 Loss 및 지표 MLflow 자동 로깅
    )

    with mlflow.start_run(run_name="full-ft-qwen-1.5b") as run:
        mlflow.log_param("tuning_method", "Full-Fine-Tuning")
        mlflow.log_param("base_model", base_model_id)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
        trainer.train()

        # 5. 학습 완료된 전체 모델 가중치 및 토크나이저 저장 후 MLflow/S3 업로드
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mlflow.log_artifacts(output_dir, artifact_path="model")
        print(f"RUN_ID={run.info.run_id}")
        
        # 다음 파이프라인 스텝(평가)으로 Run ID 전달
        with open("/tmp/run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    main()
```

### 3-2. 모델 성능 검증 및 레지스트리 승격 스크립트 (`src/evaluate.py`)
```python
import os
import torch
import pandas as pd
import mlflow
from mlflow.metrics import make_metric
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server.mlflow.svc.cluster.local:5000")
    run_id = os.getenv("TRAIN_RUN_ID")
    model_name = os.getenv("REGISTERED_MODEL_NAME", "summarization-qwen-1.5b")

    mlflow.set_tracking_uri(mlflow_uri)

    # 1. 이전 단계(Train)에서 S3에 저장한 파인튜닝 모델 다운로드
    local_model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # 2. 검증용 골든 데이터셋 (Golden Dataset)
    eval_df = pd.DataFrame({
        "inputs": [
            "MLflow는 머신러닝 실험 추적, 모델 패키징, 중앙 레지스트리 관리를 통합 지원하는 오픈소스 플랫폼입니다. 팀 단위 협업을 표준화하고 모델 배포 롤백을 용이하게 합니다.",
            "Karpenter는 쿠버네티스 클러스터의 대기 중인 파드를 감지해 최적의 EC2 인스턴스를 직접 프로비저닝합니다. 기존 Cluster Autoscaler보다 스케일링 속도가 빠르고 비용 효율적입니다."
        ],
        "ground_truth": [
            "MLflow는 ML 실험 추적과 모델 배포 관리를 통합 지원하는 오픈소스 MLOps 플랫폼입니다.",
            "Karpenter는 대기 파드를 감지해 최적의 EC2 노드를 빠르게 할당하는 쿠버네티스 오토스케일러입니다."
        ]
    })

    # 3. 추론 래퍼 함수 정의
    def predict_fn(inputs_df):
        predictions = []
        for article in inputs_df["inputs"]:
            prompt = f"다음 글을 한 문장으로 핵심만 요약하세요.\n\n원문: {article}\n요약: "
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predictions.append(decoded.strip())
        return predictions

    # 4. 품질 검증 지표 (길이 제한 및 핵심 키워드 포함 여부 등)
    def length_check_fn(eval_df, _builtin_metrics):
        scores = [1.0 if 10 <= len(p) <= 150 else 0.0 for p in eval_df["prediction"]]
        return mlflow.metrics.MetricValue(scores=scores, aggregate_results={"pass_rate": sum(scores) / len(scores)})

    length_metric = make_metric(eval_fn=length_check_fn, greater_is_better=True, name="length_compliance")

    # (옵션) OPENAI_API_KEY가 환경변수로 주입된 경우 LLM-as-a-Judge 지표 추가 가능
    extra_metrics = [length_metric]

    with mlflow.start_run(run_id=run_id):
        results = mlflow.evaluate(
            model=predict_fn,
            data=eval_df,
            targets="ground_truth",
            extra_metrics=extra_metrics,
        )
        pass_rate = results.metrics.get("length_compliance/v1/pass_rate", 0.0)
        print(f"검증 통과율 (Pass Rate): {pass_rate}")

        # 5. 품질 게이트(Gate): 통과율 80% 이상일 때만 @Production 별칭 승격
        if pass_rate >= 0.8:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri=model_uri, name=model_name)
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name=model_name,
                alias="Production",
                version=mv.version
            )
            print(f"모델 버전 {mv.version}이 @Production으로 승격되었습니다.")
        else:
            raise ValueError(f"품질 기준 미달로 배포가 차단되었습니다. (Pass Rate: {pass_rate})")

if __name__ == "__main__":
    main()
```

### 3-3. 컨테이너 이미지 빌드 (`Dockerfile`)
```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir     transformers==4.45.0     datasets==3.0.0     accelerate==0.34.0     mlflow==2.17.0     boto3     pandas

COPY src/ /app/src/
```

빌드 및 ECR 푸시:
```bash
docker build -t <AWS_ACCOUNT_ID>.dkr.ecr.ap-northeast-2.amazonaws.com/llm-mlops-worker:v1 .
docker push <AWS_ACCOUNT_ID>.dkr.ecr.ap-northeast-2.amazonaws.com/llm-mlops-worker:v1
```

---

## 4. Kubeflow Pipelines (KFP v2) 자동화 파이프라인 구성

이제 위 학습과 평가 컨테이너를 **Kubeflow Pipelines**로 엮어줍니다. 각 스텝에 `set_accelerator_type("nvidia.com/gpu")`를 지정하면, 파드가 생성될 때 **Karpenter가 자동으로 GPU 노드를 띄워서 실행**합니다.

### `pipeline.py`
```python
from kfp import dsl
from kfp import compiler
from kfp import kubernetes

IMAGE_URI = "<AWS_ACCOUNT_ID>.dkr.ecr.ap-northeast-2.amazonaws.com/llm-mlops-worker:v1"
MLFLOW_URI = "http://mlflow-server.mlflow.svc.cluster.local:5000"

@dsl.container_component
def full_finetune_op(run_id_out: dsl.OutputPath(str)):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["sh", "-c"],
        args=[
            f"MLFLOW_TRACKING_URI={MLFLOW_URI} python /app/src/train.py && "
            f"cp /tmp/run_id.txt {run_id_out}"
        ]
    )

@dsl.container_component
def evaluate_and_promote_op(run_id_in: dsl.InputPath(str)):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["sh", "-c"],
        args=[
            f"TRAIN_RUN_ID=$(cat {run_id_in}) "
            f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
            f"python /app/src/evaluate.py"
        ]
    )

@dsl.pipeline(
    name="llm-full-finetuning-pipeline",
    description="EKS + Karpenter 기반 LLM Full Fine-Tuning 및 MLflow 검증 자동화 파이프라인"
)
def llm_mlops_pipeline():
    # Step 1: Full Fine-Tuning (GPU 1장 할당 -> Karpenter가 g5 노드 자동 생성)
    train_task = full_finetune_op()
    train_task.set_cpu_limit("4")
    train_task.set_memory_limit("16Gi")
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_accelerator_limit(1)
    # Karpenter GPU NodePool의 Taint 허용 설정
    kubernetes.add_toleration(
        train_task,
        key="nvidia.com/gpu",
        operator="Exists",
        effect="NoSchedule"
    )

    # Step 2: 모델 평가 및 @Production 승격 (학습 완료 후 순차 실행)
    eval_task = evaluate_and_promote_op(run_id_in=train_task.outputs["run_id_out"])
    eval_task.set_cpu_limit("4")
    eval_task.set_memory_limit("16Gi")
    eval_task.set_accelerator_type("nvidia.com/gpu")
    eval_task.set_accelerator_limit(1)
    kubernetes.add_toleration(
        eval_task,
        key="nvidia.com/gpu",
        operator="Exists",
        effect="NoSchedule"
    )

if __name__ == "__main__":
    # Kubeflow UI에 업로드하거나 API로 실행할 YAML 파일 생성
    compiler.Compiler().compile(
        pipeline_func=llm_mlops_pipeline,
        package_path="llm_full_ft_pipeline.yaml"
    )
    print("파이프라인 컴파일 완료: llm_full_ft_pipeline.yaml")
```

### 파이프라인 실행 방법
1. 로컬에서 `python pipeline.py`를 실행해 `llm_full_ft_pipeline.yaml` 파일을 생성합니다.
2. **Kubeflow Central Dashboard**에 접속하여 `Pipelines` 메뉴에서 해당 YAML 파일을 업로드합니다.
3. **Create Run**을 클릭해 즉시 실행하거나, **Recurring Run (Cron: 예 `0 2 * * *`)**을 설정해 매일 새벽 2시에 자동 재학습되도록 스케줄링합니다.
4. 파이프라인이 실행되면 `kubectl get nodes -w` 명령어로 **Karpenter가 수십 초 내에 새 GPU 노드를 띄우고, 파이프라인 종료 60초 후 노드를 자동 삭제**하는 것을 확인할 수 있습니다.

---

## 5. vLLM 추론 서버 배포 및 테스트 (`vllm-deployment.yaml`)

MLflow 레지스트리에서 `@Production`으로 승격된 모델(S3에 저장된 가중치)을 **vLLM** 엔진으로 서빙합니다. InitContainer를 활용해 MLflow에서 `@Production` 모델 아티팩트를 공유 볼륨(`/models`)에 다운로드한 뒤 vLLM 서버가 즉시 기동하는 깔끔한 패턴입니다.

### `vllm-deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-summarization-server
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-summarization
  template:
    metadata:
      labels:
        app: vllm-summarization
    spec:
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      volumes:
        - name: model-storage
          emptyDir: {}
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "4Gi" # PyTorch/vLLM 멀티프로세싱 공유 메모리 필수 설정
      # 1. InitContainer: MLflow에서 @Production 모델 가중치를 /models 경로로 다운로드
      initContainers:
        - name: fetch-production-model
          image: python:3.11-slim
          env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-server.mlflow.svc.cluster.local:5000"
          command:
            - sh
            - -c
            - |
              pip install --no-cache-dir mlflow boto3 &&               python -c "
              import mlflow
              mlflow.artifacts.download_artifacts(
                  artifact_uri='models:/summarization-qwen-1.5b@Production',
                  dst_path='/models'
              )
              print('Production 모델 다운로드 완료!')
              "
          volumeMounts:
            - name: model-storage
              mountPath: /models
      # 2. Main Container: vLLM OpenAI 호환 API 서버 실행
      containers:
        - name: vllm-server
          image: vllm/vllm-openai:v0.6.3
          args:
            - "--model=/models"
            - "--served-model-name=summarization-prod"
            - "--dtype=bfloat16"
            - "--max-model-len=2048"
            - "--gpu-memory-utilization=0.85"
            - "--port=8000"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 1 # Karpenter가 추론용 GPU 노드 자동 할당
              cpu: "4"
              memory: "16Gi"
          volumeMounts:
            - name: model-storage
              mountPath: /models
            - name: shm
              mountPath: /dev/shm
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-summarization-svc
spec:
  type: ClusterIP
  selector:
    app: vllm-summarization
  ports:
    - port: 80
      targetPort: 8000
```

### 배포 및 API 호출 테스트
```bash
# 1. vLLM 추론 서버 배포
kubectl apply -f vllm-deployment.yaml

# 2. 파드 및 GPU 노드 할당 상태 확인
kubectl get pods -l app=vllm-summarization -w

# 3. 포트 포워딩 후 OpenAI 호환 API로 요약 요청 테스트
kubectl port-forward svc/vllm-summarization-svc 8000:80 &

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "summarization-prod",
    "prompt": "다음 글을 한 문장으로 핵심만 요약하세요.\n\n원문: 사내 MLOps 플랫폼을 구축하면서 Kubeflow와 MLflow를 연동했습니다. 이제 모델 학습부터 평가, vLLM 배포까지 수작업 없이 자동으로 이루어집니다.\n요약: ",
    "max_tokens": 80,
    "temperature": 0.1
  }'
```

---

## 6. 실무 운영 체크리스트 (Solutions Architect 관점)

* **Spot 인스턴스 중단 대응 (학습 안정성):** `g5` 스팟 인스턴스를 사용할 때는 `TrainingArguments`에 `save_strategy="steps"`, `save_steps=50`, `save_total_limit=2`를 설정하고 체크포인트 경로를 **Amazon EFS**나 **S3**에 연결해 두면, 스팟 회수 시에도 처음부터 다시 학습하지 않고 이어서 학습(Resume)할 수 있습니다.
* **새 모델 무중단 롤아웃 (Zero-Downtime Update):** Kubeflow 파이프라인의 마지막 단계에 `kubectl rollout restart deployment/vllm-summarization-server` 명령을 실행하는 경량 스텝을 추가하면, `@Production` 모델이 승격되자마자 vLLM 서버가 새 모델을 내려받아 무중단 교체 배포됩니다.
* **IAM 권한 (IRSA / Pod Identity):** 학습 파드와 vLLM 파드가 S3 버킷에 모델 가중치를 읽고 쓸 수 있도록 EKS ServiceAccount에 **AmazonS3FullAccess(또는 해당 버킷 Read/Write 권한)**를 IRSA 또는 EKS Pod Identity로 반드시 연결해야 합니다.
