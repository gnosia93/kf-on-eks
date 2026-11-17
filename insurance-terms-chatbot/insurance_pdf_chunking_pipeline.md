

# 보험 약관 PDF 표·조항 무결성 파싱 및 청킹 파이프라인

보험 약관 PDF의 **복잡한 병합 셀 표(Table)**와 **`제1관 > 제1조 > ①항 > 1호` 계층 구조**가 깨지지 않도록 보존하는 실무 파싱·청킹 파이프라인 설계 및 파이썬 구현 코드입니다.

---

## 1. 전체 파싱·청킹 파이프라인 구조

1. **레이아웃 인식 파싱 (Layout-Aware Parsing)**
   * PDF에서 본문, 제목, 표(Table), 머리글/바닥글을 시각적으로 분리하고 **표를 Markdown Table(`| 열1 | 열2 |`)로 정확히 변환**합니다.
2. **노이즈 정제 (Noise Cleaning)**
   * 매 페이지 반복되는 약관 이름 머리글과 페이지 번호(`- 12 -`)를 정규표현식으로 제거하여 문맥 단절을 방지합니다.
3. **계층 구조 기반 청킹 (Hierarchical & Structure-Aware Chunking)**
   * **조(Article) 단위 분할**: 단순 글자 수로 자르지 않고 `제N조(조항제목)` 단위를 1차 청크 경계로 삼습니다.
   * **표(Table) 무결성 보존**: 표 내부는 절대 분할하지 않고, 표가 포함된 조항은 하나의 통청크로 유지하거나 표 직전의 설명 문단과 함께 묶습니다.
   * **상위 경로 주입 (Context Injection)**: 분할된 모든 청크의 최상단에 `[상품명 > 관/절 > 제N조 제목]` 경로를 강제 삽입합니다.

---

## 2. 실무 파이썬 구현 코드 (`Docling` + 조항 정규식 청커)

폐쇄망 및 사내 GPU 환경에서도 구축 가능한 **IBM `docling`**으로 PDF를 Markdown으로 변환한 뒤, 보험 약관 전용 계층 청커로 분할하는 코드입니다.

```python
import re
import json
from typing import List, Dict
from docling.document_converter import DocumentConverter


# 1단계: PDF -> Markdown 변환 (복잡한 병합 표를 깨지지 않게 Markdown Table로 복원)
def parse_insurance_pdf(pdf_path: str) -> str:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    md_text = result.document.export_to_markdown()
    
    # 매 페이지 반복되는 페이지 번호 등 불필요한 노이즈 제거
    md_text = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n", md_text)
    return md_text


# 2단계: 보험 약관 계층 구조(관/조) 및 표 무결성 보존 청킹
def chunk_insurance_markdown(
    md_text: str, 
    product_code: str, 
    product_name: str, 
    effective_date: str,
    max_chunk_chars: int = 2000
) -> List[Dict]:
    chunks = []
    current_section = "일반조항"   # 예: 제1관 목적 및 용어의 정의
    current_article = "서문"       # 예: 제1조(목적)
    buffer_lines = []

    # 정규식 패턴: '제1관 ...' 및 '제15조(보험금의 지급사유)' 탐지
    section_pattern = re.compile(r"^(?:#+\s*)?(제\s*\d+\s*[관절장]\s*.*)\$")
    article_pattern = re.compile(r"^(?:#+\s*)?(제\s*\d+\s*조(?:의\s*\d+)?\s*\([^)]+\))")

    def flush_buffer():
        if not buffer_lines:
            return
        body = "\n".join(buffer_lines).strip()
        if not body:
            return

        # 핵심: 모든 청크 본문 최상단에 상위 계층 경로를 주입하여 검색 및 LLM 이해도 극대화
        enriched_text = (
            f"[상품: {product_name} ({effective_date} 개정) | {current_section} > {current_article}]\n"
            f"{body}"
        )
        chunks.append({
            "text": enriched_text,
            "metadata": {
                "product_code": product_code,
                "product_name": product_name,
                "effective_date": effective_date,  # 가입 시기 필터링용 필수 메타데이터
                "section": current_section,
                "article": current_article,
                "has_table": "|" in body and "---" in body
            }
        })
        buffer_lines.clear()

    lines = md_text.splitlines()
    in_table = False

    for line in lines:
        stripped = line.strip()

        # 마크다운 표(| ... |) 내부인지 추적 -> 표가 끝날 때까지는 절대 청크를 자르지 않음
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
        elif in_table and not stripped.startswith("|"):
            in_table = False

        # 새로운 '관/절'이 시작되면 기존 버퍼 저장 후 섹션 갱신
        sec_match = section_pattern.match(stripped)
        if sec_match and not in_table:
            flush_buffer()
            current_section = sec_match.group(1)
            continue

        # 새로운 '제N조(제목)'가 시작되면 기존 버퍼 저장 후 조항 갱신
        art_match = article_pattern.match(stripped)
        if art_match and not in_table:
            flush_buffer()
            current_article = art_match.group(1)
            buffer_lines.append(stripped)
            continue

        buffer_lines.append(line)

        # 단일 조항이 너무 길 경우(예: 2000자 초과), 표가 아닐 때만 '①, ②' 항 단위로 안전하게 분할
        current_len = sum(len(x) for x in buffer_lines)
        if current_len > max_chunk_chars and not in_table:
            if re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]", stripped):
                # 방금 추가한 새 항(① 등)은 다음 청크로 넘기기 위해 잠시 제외
                last_line = buffer_lines.pop()
                flush_buffer()
                buffer_lines.append(last_line)

    flush_buffer()
    return chunks


# 실행 예시
if __name__ == "__main__":
    md_content = parse_insurance_pdf("무배당_종합건강보험_약관_202401.pdf")
    parsed_chunks = chunk_insurance_markdown(
        md_text=md_content,
        product_code="PRD_HEALTH_2024",
        product_name="무배당 종합건강보험",
        effective_date="2024-01-01"
    )

    with open("parsed_corpus.jsonl", "w", encoding="utf-8") as f:
        for idx, item in enumerate(parsed_chunks):
            item["doc_id"] = f"doc_{idx:05d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"총 {len(parsed_chunks)}개의 조항/표 청크 생성 완료!")
```

---

## 3. 실무에서 품질을 결정짓는 3가지 디테일

1. **초대형 표(예: 10페이지짜리 장해분류표·수술분류표) 처리 전략**
   * 약관 후면의 `[별표] 질병 및 재해 분류표`처럼 표 하나가 수천 줄에 달하는 경우 통째로 넣으면 토큰 한도를 초과합니다.
   * 표의 헤더 행(`| 분류번호 | 질병명 | 지급률 |`)을 저장해 두었다가 30~50행 단위로 분할할 때마다 **매 청크 최상단에 헤더 행을 복제해서 삽입**해야 LLM이 각 열의 의미를 유지합니다.
2. **참조 조항 해결 (Parent-Child 청킹)**
   * 약관에는 *"제15조 제1항 제2호에 해당하는 경우..."*처럼 타 조항을 참조하는 문장이 빈번합니다.
   * 검색은 세밀한 '항/표' 단위로 수행하되, vLLM에 컨텍스트로 전달할 때는 해당 **'제N조' 전체 원문(Parent)**을 넘겨주면 Qwen-32B의 추론 정확도가 크게 향상됩니다.
3. **파서 선택 가이드**
   * **사내 폐쇄망 / GPU 직접 운영**: `Docling` 또는 `Marker` (PyTorch 기반 레이아웃 분석 모델 내장)
   * **클라우드 API 활용 가능 시**: `Upstage Document Parse` 또는 `Azure AI Document Intelligence` (한글 약관의 다중 병합 표 인식률이 매우 우수함)
