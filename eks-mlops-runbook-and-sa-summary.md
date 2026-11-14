# AWS EKS + Karpenter + Kubeflow + MLflow 기반 LLM MLOps
## 실전 장애 대응 런북(Runbook) & SA 아키텍처 제안 요약서

본 문서는 앞서 구축한 **EKS 기반 LLM Full Fine-Tuning 및 vLLM 서빙 MLOps 파이프라인**의 운영 안정성을 위한 **실전 장애 대응 매뉴얼(Part 1)**과, 고객사 및 경영진 보고를 위한 **Solutions Architect(SA) 제안 요약서(Part 2)**를 담고 있습니다.

---

# Part 1. 실전 장애 대응 런북 (Troubleshooting Runbook)

운영 및 학습 과정에서 가장 빈번하게 발생하는 4대 장애 유형별 **증상 확인 → 근본 원인 진단 → 즉시 조치(Mitigation) → 재발 방지책**을 정리합니다.

## 1. 학습 파드 GPU Out-Of-Memory (CUDA OOM) 발생

### 증상 및 탐지
* Kubeflow Pipelines의 `full-finetune-step` 태스크가 실패하며 로그에 `torch.OutOfMemoryError: CUDA out of memory` 출력.
* Grafana GPU 대시보드에서 `DCGM_FI_DEV_FB_USED`가 물리적 VRAM 한계(예: A10G 24GB)에 도달한 직후 파드 재시작(`OOMKilled`).

### 즉시 조치 및 해결 방법
1. **배치 크기 축소 및 그래디언트 누적(Gradient Accumulation) 적용**
   * `train.py`의 `per_device_train_batch_size`를 절반(예: `4` → `2` 또는 `1`)으로 줄이고, `gradient_accumulation_steps`를 2배로 늘려 동일한 유효 배치 크기(Effective Batch Size) 유지.
2. **메모리 절감 기법 활성화**
   * `TrainingArguments`에 `gradient_checkpointing=True`를 활성화하여 활성화 함수(Activation) 메모리 점유율을 최대 60%까지 절감.
   * 옵티마이저를 기본 `adamw_torch` 대신 `paged_adamw_8bit` (bitsandbytes)로 변경.
3. **Karpenter 인스턴스 스펙 상향 (파라미터 규모 증가 시)**
   * 모델 크기가 7B 이상일 경우 단일 GPU(`g5.xlarge`, 24GB) 대신 다중 GPU(`g5.12xlarge`, A10G 4장) 또는 `p4d.24xlarge`(A100 40GB/80GB)로 NodePool 요구사항 변경 후 PyTorch FSDP(Fully Sharded Data Parallel) 적용.

---

## 2. Karpenter GPU 노드 프로비저닝 지연 또는 실패 (파드 `Pending` 지속)

### 증상 및 탐지
* 학습 파드 또는 vLLM 신규 파드가 `Pending` 상태에서 5분 이상 멈춰 있음.
* `kubectl describe pod <pod-name>` 실행 시 `0/N nodes are available: Insufficient nvidia.com/gpu` 이벤트 확인.

### 원인 진단 및 즉시 조치
1. **Spot 인스턴스 재고 부족 (InsufficientInstanceCapacity)**
   * Karpenter 컨트롤러 로그 확인:
     ```bash
     kubectl logs -n karpenter -l app.kubernetes.io/name=karpenter --tail=100 | grep -i "insufficient"
     ```
   * **조치:** `gpu-training-nodepool`의 `capacity-type`에 `"spot"`만 있다면 `"on-demand"`를 함께 추가하여 Spot 재고가 없을 때 자동으로 On-Demand로 폴백(Fallback)되도록 수정.
   * 가용 인스턴스 패밀리를 `g5` 단일 지정에서 `["g5", "g6", "p4d"]`로 확장하여 가용 풀 넓히기.
2. **AWS 계정 GPU vCPU 할당량(Service Quota) 초과**
   * Karpenter 로그에 `VcpuLimitExceeded` 에러가 출력되는 경우.
   * **조치:** AWS Service Quotas 콘솔에서 `Running On-Demand G and VT instances` 또는 `All G and VT Spot Instance Requests` 한도 상향 요청.
3. **Taint / Toleration 불일치**
   * NodePool에 설정된 `taints`(`nvidia.com/gpu: NoSchedule`)와 파드 스펙의 `tolerations`가 정확히 일치하는지 검증.

---

## 3. vLLM 추론 서버 응답 지연 급증 (TTFT P95 > 2초) 및 큐 적체

### 증상 및 탐지
* Grafana vLLM 대시보드에서 `vllm:time_to_first_token_seconds` (P95) 급상승 및 `vllm:num_requests_waiting` 대기 큐 지속 증가.

### 원인 진단 및 즉시 조치
1. **KV 캐시 고갈로 인한 요청 대기 (`vllm:gpu_cache_usage_perc` > 95%)**
   * 긴 프롬프트(Long Context) 요청이 동시에 몰려 GPU KV 캐시 블록이 부족한 상태.
   * **조치:** `vllm-deployment.yaml` 실행 인자에서 `--gpu-memory-utilization`을 `0.90`에서 `0.95`로 상향하거나, 서비스 SLA에 맞춰 `--max-model-len` 값을 적절히 제한(예: `8192` → `4096`).
2. **동시 처리량 한계 도달 (Scale-out 지연)**
   * HPA(Horizontal Pod Autoscaler)가 파드를 늘렸으나 Karpenter가 새 GPU 노드를 띄우고 컨테이너 이미지를 받는 데 시간이 걸리는 경우.
   * **조치:**
     * 추론용 NodePool(`gpu-serving-nodepool`)에는 `ConsolidationPolicy: WhenEmpty` 또는 최소 1대의 Warm 노드 유지.
     * ECR 이미지 크기를 경량화하고, EBS 볼륨 크기 및 IOPS(`gp3`, 3000+ IOPS)를 충분히 확보해 이미지 Pull 속도 단축.

---

## 4. 운영 모델 품질 저하(Drift) 및 환각(Hallucination) 증가

### 증상 및 탐지
* 비동기 샘플링 평가(`mlflow.evaluate()`) 결과 `summary_faithfulness` 평균 점수가 기준치(4.0점) 미만으로 하락하거나, 사용자 '싫어요' 피드백 비율 상승.

### 즉시 조치 (긴급 롤백) 및 재학습
1. **MLflow Model Registry를 통한 즉시 롤백 (무중단 복구)**
   * 현재 `@Production` 별칭을 이전 안정 버전(예: `Version 2`)으로 즉시 되돌림:
     ```python
     from mlflow.tracking import MlflowClient
     client = MlflowClient(tracking_uri="http://mlflow-service.kubeflow.svc:5000")
     client.set_registered_model_alias("summarization-qwen-full", "Production", "2")
     ```
   * vLLM 디플로이먼트를 재시작(`kubectl rollout restart deployment/vllm-summarization-server`)하여 2분 내 이전 안정 가중치로 서비스 복구.
2. **오류 데이터 수집 및 재학습 트리거**
   * 환각이 발생한 운영 로그를 추출해 검수 후 `golden_dataset.jsonl`과 학습 데이터셋에 편입하고 Kubeflow 파이프라인 재실행.

---
---

# Part 2. Solutions Architect (SA) 아키텍처 제안 요약서

고객사 의사결정권자(CTO, VP of Engineering) 및 실무 리드에게 본 아키텍처의 **비즈니스 가치, 비용 최적화 구조, 단계별 도입 로드맵**을 설명하기 위한 핵심 요약 자료입니다.

## 1. 핵심 비즈니스 가치 (Executive Summary)

| 해결 과제 (Pain Point) | 아키텍처 해결 방안 | 기대 효과 (Business Impact) |
| :--- | :--- | :--- |
| **높은 GPU 유휴 비용** | **Karpenter 동적 프로비저닝** (학습 시 Spot 할당, 종료 60초 후 즉시 반납) | 상시 GPU 인스턴스 운영 대비 **인프라 비용 최대 60~75% 절감** |
| **모델 품질 검증 부재** | **MLflow Evaluate 자동화 게이트** (환각/정확도 기준 통과 시에만 배포) | 운영 장애 및 잘못된 요약으로 인한 **브랜드 리스크 사전 차단** |
| **수동 배포 및 협업 병목** | **GitOps + Kubeflow + vLLM 표준화** (PR부터 프로덕션 배포까지 자동화) | 신규 모델 개선 주기(Lead Time) **수 주에서 수 시간으로 단축** |
| **데이터 보안 및 종속성** | **VPC 내부 온프레미스형 오픈 웨이트 모델** (S3/RDS/EKS 완전 격리) | 민감한 사내 문서의 **외부 유출 원천 차단 및 API 종속성 탈피** |

---

## 2. 핵심 아키텍처 의사결정 및 트레이드오프 (Design Rationale)

고객 미팅 및 아키텍처 리뷰 시 자주 나오는 질문에 대한 설계 근거입니다.

* **Q1. 왜 상시 클러스터나 Cluster Autoscaler 대신 Karpenter를 선택했는가?**
  * *설계 근거:* 기존 Cluster Autoscaler는 사전에 정의된 NodeGroup 단위로만 확장되어 다양한 GPU 인스턴스(`g5`, `g6`, `p4d`)와 Spot 재고 변화에 유연하게 대응하지 못합니다. Karpenter는 파드의 GPU 리소스 요구사항을 직접 분석해 **가장 저렴하고 가용한 인스턴스를 수십 초 내에 직접 프로비저닝**하며, 작업 종료 즉시 통합(Consolidation)하여 유휴 비용을 0에 가깝게 만듭니다.
* **Q2. 왜 LoRA가 아닌 Full Fine-Tuning 파이프라인을 표준으로 설계했는가?**
  * *설계 근거:* 1.5B~8B 규모의 경량 오픈 웨이트 모델에서는 도메인 특화 문체, 엄격한 출력 포맷, 고도의 요약 압축률을 달성하기 위해 전체 가중치를 업데이트하는 Full Fine-Tuning이 더 높은 품질 상한선을 제공합니다. 또한 추론 시 어댑터 병합 오버헤드 없이 단일 표준 가중치로 vLLM에 서빙할 수 있어 운영 복잡도가 낮습니다. *(단, 30B 이상 초대형 모델 확장 시에는 동일한 Kubeflow 파이프라인 내에서 학습 스크립트만 QLoRA로 전환하면 아키텍처 변경 없이 수용 가능합니다.)*
* **Q3. 실시간 운영 트래픽의 품질(Drift)을 전수 검사하지 않고 비동기 샘플링하는 이유는?**
  * *설계 근거:* 모든 사용자 요청마다 고성능 Judge LLM을 호출하면 추론 응답 지연(Latency)이 2배 이상 증가하고 평가 비용이 폭증합니다. 따라서 **실시간 경로에서는 vLLM의 지연시간과 토큰 처리량만 경량 메트릭으로 수집**하고, **전체 트래픽의 1~5%를 비동기로 샘플링하여 MLflow로 품질을 채점**하는 구조가 비용과 품질 감시 사이의 최적 균형점입니다.

---

## 3. 단계별 도입 로드맵 (Phased Implementation Roadmap)

고객의 현재 성숙도와 예산에 맞춰 리스크 없이 점진적으로 확장하는 3단계 전략을 제안합니다.

```
[1단계: 기반 구축 (1~2주)]          [2단계: 학습/검증 자동화 (3~4주)]     [3단계: 서빙 고도화 및 관측 (5~6주)]
• EKS + Karpenter GPU 노드 구성   • Kubeflow Pipelines 구축         • vLLM 오토스케일링 (HPA + Karpenter)
• 중앙 MLflow + RDS + S3 배포     • PyTorch 학습 + MLflow 평가 게이트  • Prometheus/Grafana 실시간 모니터링
• 프롬프트/베이스라인 성능 측정    • GitHub Actions CI/CD 트리거 연동  • 비동기 샘플링 기반 Drift 감지 및 롤백
```

* **Phase 1 (Quick Win):** 무거운 학습 전에 원본 모델(Base)과 프롬프트 엔지니어링만으로 MLflow에 베이스라인 점수를 확보하여 투자 대비 효과(ROI) 측정 기준을 마련합니다.
* **Phase 2 (Automation):** 도메인 데이터셋을 투입해 Full Fine-Tuning 파이프라인을 자동화하고, 베이스라인을 이긴 모델만 `@Production`으로 자동 승격시킵니다.
* **Phase 3 (Scale & Operate):** 트래픽 변동에 따른 GPU 오토스케일링과 실시간 메트릭/품질 Drift 알람을 연결하여 무인 운영 체계를 완성합니다.
