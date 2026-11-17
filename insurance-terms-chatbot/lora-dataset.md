# 고객용·직원용 듀얼 LoRA 학습 데이터셋 구축 가이드

동일한 보험 약관 조항(Context)을 바탕으로 **일반 고객용 어댑터(`adapter-customer`)**와 **내부 콜센터 직원용 어댑터(`adapter-agent`)**가 서로 다른 관점·어휘·출력 형식으로 응답하도록 학습시키는 합성 데이터(Synthetic Data) 구축 파이프라인입니다.

---

## 1. 어댑터별 학습 데이터 설계 전략

| 구분 | 고객용 어댑터 (`adapter-customer`) | 직원용 어댑터 (`adapter-agent`) |
| :--- | :--- | :--- |
| **질문 스타일** | 구어체, 상황 중심, 모호한 표현<br>*(예: "스키 타다 다리 부러졌는데 수술비 나오나요?")* | 업무 용어, 조항/특약 중심, 조건 확인<br>*(예: "상해수술비 특약에서 스포츠 활동 면책 여부 및 지급률 확인")* |
| **답변 스타일** | • 쉬운 일상 용어로 풀어서 설명<br>• 면책(보장 안 되는) 조건을 부드럽지만 명확히 고지<br>• 하단에 `[근거: 상품명 제N조]` 표기 | • 인사말/미사여구 완전 배제 (개조식 요약)<br>• **[결론] → [근거 조항] → [지급 기준/표 수치] → [심사 유의사항]** 순서 고정 |
| **Negative(거절) 데이터** | 약관에 없는 질문이나 타 상품 질문에 **"제공된 약관에서는 확인할 수 없습니다"**라고 답하는 데이터 **전체의 15~20% 필수 포함** (환각 방지) | 동일하게 약관 근거가 부족할 경우 **"약관 내 명시적 근거 없음(심사역 확인 필요)"**으로 단호히 끊어주는 데이터 **15~20% 포함** |

---

## 2. 데이터 확보 3단계 파이프라인

1. **사내 콜센터 상담 이력 정제 (Seed 데이터)**
   * 실제 상담 녹취록(STT) 및 QA 게시판에서 개인정보(이름, 주민번호, 증권번호)를 마스킹한 뒤, 질문과 모범 답변 쌍을 200~300건 추출합니다.
2. **약관 청크 기반 합성 데이터 대량 생성 (Data Augmentation)**
   * 파싱된 약관 청크(`parsed_corpus.jsonl`)에서 조항 및 `[별표]`를 꺼내 고성능 Teacher 모델(`Qwen-32B` 또는 `GPT-4o`)에게 고객용/직원용 페르소나별 Q&A를 생성하게 합니다.
   * **비율 권장**: 단순 조항 질의(60%) + 본문과 `[별표]` 지급률 표를 함께 계산하는 복합 추론 질의(25%) + 약관에 답이 없는 거절 유도 질의(15%)
3. **자동 검증 및 필터링 (Quality Filtering)**
   * 생성된 답변이 원본 약관(`context`)에 없는 숫자나 조항 번호를 지어냈는지(환각) LLM-as-a-Judge로 검증하여 통과한 데이터만 최종 JSONL(`train_customer.jsonl`, `train_agent.jsonl`)로 저장합니다.

---

## 3. Qwen 파인튜닝용 JSONL 데이터 포맷 예시

### ① 고객용 학습 데이터 예시 (`train_customer.jsonl`)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "당신은 친절한 보험 약관 안내 챗봇입니다. 제공된 참고 약관만을 근거로 고객이 이해하기 쉽게 설명하고, 답변 마지막에 근거 조항을 표기하세요."
    },
    {
      "role": "user",
      "content": "참고 약관:\n[상품: 무배당 종합건강보험 (2024-01-01 개정) | 제2관 보험금의 지급 > 제15조(보험금의 지급사유)]\n회사는 피보험자가 보험기간 중에 상해의 직접결과로써 수술을 받은 경우 수술 1회당 가입금액을 지급합니다. 단, 미용성형 목적의 수술 및 피보험자의 고의적 자해로 인한 수술은 보상하지 않습니다.\n\n질문: 미용 목적으로 쌍꺼풀 수술을 했는데 상해수술비 받을 수 있나요?"
    },
    {
      "role": "assistant",
      "content": "아니요, 미용 성형 목적의 수술은 보장 대상에서 제외되어 상해수술비를 받으실 수 없습니다.\n\n약관상 상해수술비는 보험기간 중 발생한 상해의 직접적인 치료를 목적으로 수술을 받은 경우에만 지급되며, 미용을 위한 성형수술은 면책 사항(보상하지 않는 손해)에 해당합니다.\n\n[근거: 무배당 종합건강보험 제15조(보험금의 지급사유)]"
    }
  ]
}
```

### ② 직원용 학습 데이터 예시 (`train_agent.jsonl`)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "당신은 보험사 내부 콜센터 심사 지원 AI입니다. 제공된 참고 약관만을 근거로 결론, 근거 조항, 예외 사항을 개조식으로 요약하세요."
    },
    {
      "role": "user",
      "content": "참고 약관:\n[상품: 무배당 종합건강보험 (2024-01-01 개정) | 제2관 보험금의 지급 > 제15조(보험금의 지급사유)]\n회사는 피보험자가 보험기간 중에 상해의 직접결과로써 수술을 받은 경우 수술 1회당 가입금액을 지급합니다. 단, 미용성형 목적의 수술 및 피보험자의 고의적 자해로 인한 수술은 보상하지 않습니다.\n\n질문: 미용 목적 쌍꺼풀 수술 건 상해수술비 지급 가능 여부 및 면책 근거"
    },
    {
      "role": "assistant",
      "content": "- **심사 결론**: 지급 불가 (면책 대상)\n- **근거 조항**: 무배당 종합건강보험 제2관 제15조(보험금의 지급사유) 단서 조항\n- **면책 사유**: 미용성형 목적의 수술은 상해의 직접결과로 인한 치료 목적 수술에 해당하지 않음\n- **상담 유의사항**: 치료 목적(안검하수 등 의학적 소견서 동반) 여부가 확인되지 않는 단순 미용 목적일 경우 면책 안내"
    }
  ]
}
```

---

## 4. 합성 데이터 자동 생성 파이썬 스크립트

파싱된 약관 청크(`parsed_corpus.jsonl`)를 읽어 고객용/직원용 데이터셋을 자동으로 생성하는 코드입니다.

```python
import json
from openai import OpenAI

# 사내 vLLM(Qwen-32B) 또는 OpenAI API 연결
client = OpenAI(base_url="http://cluster.local", api_key="not-used")

GEN_PROMPT = """
다음 보험 약관 조항을 읽고, 학습용 질문-답변 쌍을 JSON 형식으로 2개 생성하세요.
1) customer: 일반 고객이 일상적인 구어체로 묻는 질문과, 쉬운 용어로 풀어서 설명하고 끝에 [근거: 조항]을 붙인 친절한 답변
2) agent: 콜센터 직원이 묻는 실무 질문과, [심사 결론 / 근거 조항 / 핵심 조건]을 개조식으로 요약한 간결한 답변

반드시 아래 JSON 스키마만 출력하세요:
{
  "customer": {"question": "...", "answer": "..."},
  "agent": {"question": "...", "answer": "..."}
}

[약관 조항]
"""


def build_dual_datasets(corpus_path: str, out_customer: str, out_agent: str):
    with open(corpus_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    with open(out_customer, "w", encoding="utf-8") as fc, open(out_agent, "w", encoding="utf-8") as fa:
        for chunk in chunks:
            context = chunk["text"]
            resp = client.chat.completions.create(
                model="qwen3-32b",
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": GEN_PROMPT + context}]
            )
            pair = json.loads(resp.choices.message.content)

            # 고객용 데이터 저장
            fc.write(json.dumps({
                "messages": [
                    {
                        "role": "system",
                        "content": "당신은 친절한 보험 약관 안내 챗봇입니다. 제공된 약관만을 근거로 설명하고 끝에 근거 조항을 표기하세요."
                    },
                    {
                        "role": "user",
                        "content": f"참고 약관:\n{context}\n\n질문: {pair['customer']['question']}"
                    },
                    {
                        "role": "assistant",
                        "content": pair["customer"]["answer"]
                    }
                ]
            }, ensure_ascii=False) + "\n")

            # 직원용 데이터 저장
            fa.write(json.dumps({
                "messages": [
                    {
                        "role": "system",
                        "content": "당신은 보험사 내부 콜센터 심사 지원 AI입니다. 제공된 약관만을 근거로 개조식으로 요약하세요."
                    },
                    {
                        "role": "user",
                        "content": f"참고 약관:\n{context}\n\n질문: {pair['agent']['question']}"
                    },
                    {
                        "role": "assistant",
                        "content": pair["agent"]["answer"]
                    }
                ]
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build_dual_datasets(
        corpus_path="parsed_corpus.jsonl",
        out_customer="train_customer.jsonl",
        out_agent="train_agent.jsonl"
    )
```
