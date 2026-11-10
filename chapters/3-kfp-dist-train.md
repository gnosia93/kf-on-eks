# KFP LoRA(QLoRA) 파인튜닝 컴포넌트 + 멀티 GPU

> sklearn RandomForest 컴포넌트와 **같은 KFP v2 골격**, 내용만 LoRA 파인튜닝 + GPU로 교체.
> 파인튜닝한 어댑터는 KServe HF(vLLM) 런타임에 그대로 서빙 가능.

---

## 1. LoRA 파인튜닝 컴포넌트 (QLoRA, 멀티 GPU 자동 분산)

```python
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

# CUDA 포함 베이스 + LLM 학습 스택
COMMON = dict(
    base_image="nvcr.io/nvidia/pytorch:24.07-py3",
    packages_to_install=[
        "transformers>=4.44", "peft>=0.12", "trl>=0.9",
        "datasets>=2.20", "accelerate>=0.33", "bitsandbytes>=0.43",
    ],
)

@dsl.component(**COMMON)
def lora_finetune(
    train_data: Input[Dataset],          # jsonl: {"text": "..."} 형식 가정
    adapter_out: Output[Model],          # LoRA 어댑터만 출력 (수십~수백 MB)
    base_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    epochs: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    hf_token: str = "",
):
    import os, torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # 1) 데이터 로드
    ds = load_dataset("json", data_files=train_data.path, split="train")

    # 2) base 모델을 4-bit로 로드(QLoRA) + 가용 GPU들에 자동 분산
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(base_model_id)
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",              # ← 여러 GPU에 base 모델 자동 샤딩(모델 병렬)
        torch_dtype=torch.bfloat16,
    )

    # 3) LoRA 어댑터 설정 (attention + MLP projection)
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # 4) 학습
    os.makedirs(adapter_out.path, exist_ok=True)
    cfg = SFTConfig(
        output_dir=adapter_out.path,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        max_seq_length=2048,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds, peft_config=lora,
        tokenizer=tok,                  # trl 신버전은 processing_class=tok
    )
    trainer.train()

    # 5) LoRA 어댑터만 저장 (base 가중치 제외)
    trainer.save_model(adapter_out.path)  # adapter_config.json + adapter_model.safetensors
    adapter_out.metadata["base_model"] = base_model_id
    adapter_out.metadata["lora_r"] = lora_r
```

---

## 2. 파이프라인에서 GPU 리소스 지정

```python
@dsl.pipeline(name="lora-finetune")
def pipeline(train_uri: str, hf_token: str):
    imp = dsl.importer(artifact_uri=train_uri, artifact_class=Dataset)

    t = lora_finetune(
        train_data=imp.output,
        base_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        hf_token=hf_token,
    )
    t.set_cpu_limit("8").set_memory_limit("64G")
    t.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(4)  # 노드당 GPU 4장
    # GPU 노드 taint가 있으면 toleration/nodeSelector도 추가 (환경 따라)
```

- `device_map="auto"` + `set_accelerator_limit(4)` → **한 파드 안에서 4 GPU에 모델을 쪼개** QLoRA. 70B급도 fit 가능 (모델 병렬).
- **처리량(데이터 병렬/DDP)** 목적이면 `device_map` 대신 `accelerate launch`/`torchrun` 사용 (§3).

---

## 3. "진짜 분산"이 필요하면 — 멀티노드

⚠️ KFP `@dsl.component`는 **파드 1개**로 실행됨. 그래서:

- **단일 노드 멀티 GPU** → 위처럼 `device_map="auto"`(모델 병렬) 또는 컴포넌트 안 `accelerate launch`(DDP/FSDP). ✅ 지금 예제로 충분.
- **멀티노드 분산(DDP/FSDP across nodes)** → KFP 컴포넌트로는 안 되고 **Kubeflow Training Operator의 `PyTorchJob`**(torchrun 분산)으로 띄움:

```python
@dsl.component(base_image="kubeflow/training-sdk:latest",
               packages_to_install=["kubeflow-training"])
def launch_distributed(train_uri: str):
    from kubeflow.training import TrainingClient
    TrainingClient().create_job(
        name="llama-fsdp",
        train_func=my_train_fn,     # torchrun/FSDP 로직
        num_workers=2,              # 노드 2개
        num_procs_per_worker=8,     # 노드당 GPU 8
        resources_per_worker={"nvidia.com/gpu": "8"},
    )
```

→ **LoRA/단일노드는 KFP 컴포넌트**, **대규모 멀티노드 full/FSDP는 Training Operator**로 나눠 쓰는 게 실무 패턴.

---

## 4. 서빙 연결 (KServe HF + vLLM LoRA)

저장된 LoRA 어댑터는 base와 독립 → 어댑터만 얹어 서빙:

```yaml
spec:
  predictor:
    model:
      modelFormat: { name: huggingface }
      storageUri: s3://<bucket>/base/llama3-8b        # base 모델
      args:
        - --enable-lora
        - --max-lora-rank=16
        - --lora-modules=my-adapter=/mnt/adapters/my-lora   # 파인튜닝 산출물 경로
```

base 하나만 로드하고 어댑터 여러 개를 바꿔 끼울 수 있어 GPU/디스크 효율적.

---

## 5. 버전 주의

`trl` / `peft` / `transformers`가 자주 바뀌어 `SFTConfig` / `SFTTrainer` 인자 위치(`max_seq_length`, `dataset_text_field`, `tokenizer` vs `processing_class`)가 버전마다 다를 수 있음. 위는 **trl 0.9~0.11 기준**. 프로덕션에선 패키지 버전 **핀 고정**.
