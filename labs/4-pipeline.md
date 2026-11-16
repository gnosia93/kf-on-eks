### 파이프라인 정의 ###

* prepare_data - SQuAD v2를 RAFT 형식으로 변환해 S3에 저장. MLflow run 생성 후 run ID 반환
* train_lora - TrainJob을 만들고 끝날 때까지 기다린 뒤 어댑터 경로 반환
* evaluate - vLLM 오프라인 추론으로 base와 LoRA를 같은 평가셋에서 채점
* promote - LoRA가 min_gain 이상 좋을 때만 실행. serving/CURRENT를 갱신하고 vLLM 재시작

모든 단계가 MLflow run 하나에 기록되므로 데이터, 학습, 평가 결과가 한 화면에서 이어진다.


