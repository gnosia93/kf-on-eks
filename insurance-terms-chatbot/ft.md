# 🚀 보험 약관 RAG 특화 LLM 파인튜닝(Fine-Tuning) 완벽 가이드

## 1. 파인튜닝 핵심 개념 (RAG + SFT + QLoRA)

* **SFT (Supervised Fine-Tuning, 지도 미세 조정):** `[시스템 지시문 + 검색된 약관(Context) + 사용자 질문]`을 입력(Input)으로 주고, `[모범 답변]`을 출력(Output)으로 짝지어 모델의 가중치를 업데이트하는 방식입니다.
* **LoRA / QLoRA (경량화 파인튜닝):** 수십억 개(7B~8B 이상)의 파라미터를 전부 수정하면 엄청난 GPU 메모리가 필요합니다. QLoRA는 원본 모델을 **4비트로 압축(Quantization)**해 메모리에 올린 뒤, 옆에 **작은 어댑터(LoRA 레이어, 전체의 약 1~2%)만 붙여서 학습**시키는 최신 표준 기술입니다. (RTX 3090/4090 24GB VRAM 환경에서도 8B 모델 학습 가능)

---

## 2. 파인튜닝용 샘플 데이터 구조 (`train_data.jsonl`)

실무에서 가장 표준으로 쓰이는 **ChatML(Messages) 포맷**입니다. 1줄에 JSON 객체 1개가 들어가는 `.jsonl` 형식으로 저장합니다.

> **💡 핵심 포인트:** 실전 RAG 환경과 똑같이 만들기 위해 `user` 프롬프트 안에 **정답 약관뿐만 아니라 관련 없는 노이즈 약관(오답)도 함께 섞어** 넣어줍니다.

```json
{"messages": [{"role": "system", "content": "당신은 베테랑 보험 약관 상담 전문가입니다. 반드시 제공된 [참고 약관]의 내용만을 근거로 정확하고 친절하게 답변하세요. 약관에 근거가 없는 내용은 절대 추측하지 말고 '제공된 약관에서는 해당 내용을 확인할 수 없습니다.'라고 답변하세요."}, {"role": "user", "content": "[참고 약관]\n문서1: 제12조(골절진단비) 피보험자가 보험기간 중 상해로 인하여 골절로 진단확정된 경우 최초 1회에 한하여 50만원을 지급합니다.\n문서2: 제15조(뇌출혈의 정의) '뇌출혈'이라 함은 한국표준질병사인분류에 있어서 뇌출혈로 분류되는 질병(질병코드 I60~I62)을 말합니다. 진단확정 시 가입금액의 100%를 지급합니다.\n문서3: 제20조(보험료의 납입연체) 보험료를 납입기일까지 납입하지 아니한 경우 납입최고기간을 둡니다.\n\n[질문]\n뇌출혈(질병코드 I61)로 진단받았는데 가입금액이 3,000만원이면 얼마 받나요?"}, {"role": "assistant", "content": "고객님, 제공된 약관 제15조(뇌출혈의 정의)에 따라 질병코드 I61은 보장 대상인 '뇌출혈(I60~I62)'에 해당합니다.\n\n뇌출혈 진단확정 시 가입금액의 100%가 지급되므로, 가입금액 3,000만 원의 100%인 **3,000만 원**을 지급받으실 수 있습니다."}]}
{"messages": [{"role": "system", "content": "당신은 베테랑 보험 약관 상담 전문가입니다. 반드시 제공된 [참고 약관]의 내용만을 근거로 정확하고 친절하게 답변하세요. 약관에 근거가 없는 내용은 절대 추측하지 말고 '제공된 약관에서는 해당 내용을 확인할 수 없습니다.'라고 답변하세요."}, {"role": "user", "content": "[참고 약관]\n문서1: 제5조(암의 정의) '암'이라 함은 한국표준질병사인분류에 있어서 악성신생물(C00~C97)로 분류되는 질병을 말합니다. 단, 기타피부암(C44) 및 갑상선암(C73)은 제외합니다.\n문서2: 제8조(입원일당) 암의 치료를 직접적인 목적으로 4일 이상 계속 입원한 경우 3일 초과 1일당 10만원을 지급합니다.\n\n[질문]\n치아 임플란트 치료를 받았는데 수술비가 지원되나요?"}, {"role": "assistant", "content": "제공된 약관에서는 치아 임플란트 치료 및 수술비 지원에 관한 내용을 확인할 수 없습니다. 가입하신 상품의 치과 관련 특약 약관을 별도로 확인해 주시기 바랍니다."}]}
```

---

## 3. 파인튜닝 전체 파이썬 소스코드 (`train_qlora.py`)

Hugging Face의 `transformers`, `peft`, `trl`(SFTTrainer) 라이브러리를 사용한 표준 QLoRA 학습 코드입니다.

### ① 필수 라이브러리 설치
```bash
pip install torch transformers datasets accelerate peft bitsandbytes trl
```

### ② 학습 스크립트 (`train_qlora.py`)
```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. 기본 설정 (한국어 성능이 우수한 Qwen2.5-7B-Instruct 예시)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "./train_data.jsonl"
OUTPUT_DIR = "./insurance-rag-qwen-lora"

# 2. 4-bit 양자화 설정 (GPU 메모리 절약 핵심)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # Ampere 이상 GPU(RTX 30/40, A100 등)는 bfloat16 권장
)

# 3. 토크나이저 및 베이스 모델 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = prepare_model_for_kbit_training(model)

# 4. LoRA 어댑터 설정 (어떤 레이어를 학습시킬지 결정)
peft_config = LoraConfig(
    r=16,                        # LoRA 랭크 (보통 8, 16, 32 사용. 높을수록 표현력 증가)
    lora_alpha=32,               # 스케일링 계수 (보통 r의 2배 설정)
    target_modules=[             # 트랜스포머의 어텐션 및 FFN 핵심 모듈 지정
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 전체 파라미터 중 약 1~2%만 학습되는지 확인

# 5. 데이터셋 로드 및 ChatML 템플릿 변환 함수
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_chat_template(row):
    # 모델 고유 채팅 템플릿(<|im_start|>user ... <|im_end|>)으로 자동 변환
    formatted_text = tokenizer.apply_chat_template(
        row["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {"text": formatted_text}

dataset = dataset.map(format_chat_template)

# 6. 학습 하이퍼파라미터 설정
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_seq_length=2048,                 # 약관 3~5개 + 질문 길이를 감안해 2048 이상 권장
    per_device_train_batch_size=2,       # GPU 메모리에 맞게 조절 (1~4)
    gradient_accumulation_steps=8,       # 실질 배치 크기 = 2 * 8 = 16 효과
    learning_rate=2e-4,                  # QLoRA 표준 학습률
    num_train_epochs=3,                  # 데이터 1,300건 기준 3 에포크 권장 (과적합 방지)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                           # bfloat16 연산 활성화
    optim="paged_adamw_8bit"             # 메모리 효율적 옵티마이저
)

# 7. 트레이너 실행 및 어댑터 저장
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=sft_config,
)

trainer.train()

# 학습 완료된 LoRA 어댑터 가중치 저장
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ 파인튜닝 완료 및 어댑터 저장 성공!")
```

---

## 4. 학습된 모델로 실전 RAG 추론하기 (`inference.py`)

학습이 끝나면 `OUTPUT_DIR`에 수십~수백 MB 크기의 **LoRA 어댑터 파일**이 생성됩니다. 추론할 때는 원본 모델에 이 어댑터를 결합(`PeftModel`)해서 사용합니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "./insurance-rag-qwen-lora"

# 1. 원본 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 파인튜닝된 보험 약관 LoRA 어댑터 결합
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# 3. Qdrant 검색기에서 가져온 약관과 사용자 질문 조합
retrieved_context = """
문서1: 제15조(뇌출혈의 정의) '뇌출혈'이라 함은 질병코드 I60~I62에 해당하는 질병을 말하며 진단확정 시 가입금액의 100%를 지급합니다.
문서2: [별표3] 장해분류표 - 3급 장해 시 가입금액의 50%를 후유장해보험금으로 추가 지급합니다.
"""
user_question = "뇌출혈(I60) 진단받고 3급 후유장해가 남았습니다. 가입금액 5,000만원이면 총 얼마 받나요?"

messages = [
    {
        "role": "system",
        "content": "당신은 베테랑 보험 약관 상담 전문가입니다. 반드시 제공된 [참고 약관]의 내용만을 근거로 정확하고 친절하게 답변하세요. 약관에 근거가 없는 내용은 절대 추측하지 말고 '제공된 약관에서는 해당 내용을 확인할 수 없습니다.'라고 답변하세요."
    },
    {
        "role": "user",
        "content": f"[참고 약관]\n{retrieved_context}\n\n[질문]\n{user_question}"
    }
]

# 4. 답변 생성
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,      # 사실 기반 RAG에서는 환각 억제를 위해 0.1 이하 낮은 온도 필수
        do_sample=True,
        repetition_penalty=1.05
    )

# 입력 프롬프트 부분을 제외하고 새로 생성된 답변만 디코딩
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("=== AI 상담원 답변 ===")
print(response)
```

---

## 5. 보험 약관 파인튜닝 실전 체크리스트 (실패 방지 팁)

1. **`max_seq_length` 여유 있게 설정하기**
   * 약관 청크 3~5개를 프롬프트에 넣으면 토큰 길이가 금방 1,500~2,500토큰을 넘깁니다. `max_seq_length`를 `512`나 `1024`로 작게 잡으면 뒷부분의 `[질문]`과 `[답변]`이 잘려나가 학습이 망가집니다. 최소 `2048` 또는 `4096`으로 설정하세요.
2. **에포크(Epoch) 욕심부리지 않기**
   * 데이터 1,300건 내외를 10 Epoch 이상 돌리면 모델이 훈련 데이터의 정답 문장을 통째로 암기(과적합)해버려, 새로운 약관이 들어왔을 때 대처 능력이 떨어집니다. **2~3 Epoch**가 가장 이상적입니다.
3. **추론 시 `temperature`는 낮게 (`0.0 ~ 0.1`)**
   * 창의적인 글쓰기가 아니라 약관 조항과 숫자를 정확히 계산해야 하므로, 추론할 때 `temperature`를 낮춰서 일관되고 보수적인 답변이 나오도록 제어해야 합니다.
4. **어댑터 병합(Merge) 후 vLLM 배포**
   * 실제 서비스에 올릴 때는 `model = model.merge_and_unload()` 명령어로 원본 모델과 LoRA 어댑터를 하나로 합쳐 저장한 뒤, **vLLM** 같은 고속 추론 엔진에 올리면 응답 속도를 수배 이상 높일 수 있습니다.
