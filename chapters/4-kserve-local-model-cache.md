# KServe LocalModelCache 운영 구축 가이드

> **시나리오**: 70B LLM · 노드당 H100 80GB × 8 GPU · 최대 3 replica · RawDeployment 모드

---

## 1. 입력값 & 채워야 할 플레이스홀더

| 항목 | 값 |
|---|---|
| 모델 크기 | 70B |
| 노드 GPU | H100 80GB × 8 / node |
| 최대 replica | 3 |
| 배포 모드 | RawDeployment |

구축 전에 아래 값을 먼저 확정하세요.

| 플레이스홀더 | 설명 | 예시 |
|---|---|---|
| `<your-bucket>` | 모델이 든 S3 버킷/경로 | `my-models` |
| `<region>` | S3 리전 | `ap-northeast-2` |
| `<your-ns>` | ISVC를 올릴 네임스페이스 | `llm-serving` |
| `<h100-node-1/2>` | 캐시 대상 노드 이름 | `ip-10-0-1-10...` |
| `<YOUR_KEY>` / `<YOUR_SECRET>` | S3 자격증명 | — |

---

## 2. 용량 산정

| 항목 | 계산 | 결과 |
|---|---|---|
| 70B 가중치 (BF16) | 70B × 2B | **~140GB** → 1 GPU 불가 |
| replica당 GPU (TP) | 140GB ÷ 80GB + KV 여유 | **TP=4** (4×80=320GB, KV ~180GB) |
| 노드당 replica | 8 GPU ÷ 4 | **2 replica / node** |
| 필요 노드 수 | ⌈3 ÷ 2⌉ + HA | **2 노드** (용량 4, 3 사용) |
| 노드당 캐시 디스크 | 140GB + 버퍼 | **≥ 150Gi NVMe** |

→ **노드그룹 = H100 노드 2대 (둘 다 캐시)**. H100이면 FP8(TP=2)로 GPU 절반 가능 — §5 참고.

---

## 3. 구축 순서 (한눈에)

```
0️⃣ 사전 준비        (1회)        노드 라벨 + agent 확인
1️⃣ 기능 켜기        (1회)        ConfigMap localModel.enabled=true
2️⃣ 자격증명         (소스별 1회)  S3 Secret + ClusterStorageContainer
3️⃣ 노드그룹         (노드그룹별)  LocalModelNodeGroup (2노드 규모)
4️⃣ 모델 캐시 + 대기  (모델마다)    LocalModelCache → ⏳ 다운로드 완료 GATE
5️⃣ 서빙             (모델마다)    InferenceService (storageUri 일치 GATE)
```

- 🔵 **1회성**: 0️⃣ 1️⃣ 2️⃣
- 🟠 **노드그룹별**: 3️⃣
- 🟢 **모델마다**: 4️⃣ 5️⃣

---

## 4. 단계별 매니페스트

### 0️⃣ 사전 준비 (1회)

```bash
# 캐시 대상 노드 2대에 라벨
kubectl label node <h100-node-1> kserve/localmodel-node=h100-llm
kubectl label node <h100-node-2> kserve/localmodel-node=h100-llm

# agent DaemonSet이 이 노드에 떠 있는지 + hostPath 확인 (3️⃣ local.path와 일치해야 함)
kubectl get daemonset kserve-localmodelnode-agent -n kserve \
  -o jsonpath='{.spec.template.spec.volumes[?(@.name=="models")].hostPath.path}'
```

### 1️⃣ 기능 켜기 (1회)

```bash
kubectl edit configmap inferenceservice-config -n kserve
```

```yaml
localModel: |-
  {
    "enabled": true,
    "jobNamespace": "kserve-localmodel-jobs",
    "defaultJobImage": "kserve/storage-initializer:v0.15.0",
    "fsGroup": 1000
  }
```

### 2️⃣ 자격증명 (S3 소스별 1회)

```yaml
# (a) S3 자격증명 — 다운로드 job 네임스페이스에
apiVersion: v1
kind: Secret
metadata:
  name: s3-creds
  namespace: kserve-localmodel-jobs
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: "<YOUR_KEY>"
  AWS_SECRET_ACCESS_KEY: "<YOUR_SECRET>"
---
# (b) s3:// 처리용 ClusterStorageContainer (다운로드 job 전용)
apiVersion: serving.kserve.io/v1alpha1
kind: ClusterStorageContainer
metadata:
  name: s3-localmodel
spec:
  container:
    name: storage-initializer
    image: kserve/storage-initializer:v0.15.0
    env:
      - name: AWS_ACCESS_KEY_ID
        valueFrom: { secretKeyRef: { name: s3-creds, key: AWS_ACCESS_KEY_ID } }
      - name: AWS_SECRET_ACCESS_KEY
        valueFrom: { secretKeyRef: { name: s3-creds, key: AWS_SECRET_ACCESS_KEY } }
      - name: AWS_DEFAULT_REGION
        value: "<region>"
      - name: S3_ENDPOINT
        value: "s3.<region>.amazonaws.com"
      - name: S3_USE_HTTPS
        value: "1"
    resources:
      requests: { memory: 4Gi, cpu: "2" }
      limits:   { memory: 8Gi, cpu: "4" }
  supportedUriFormats:
    - prefix: s3://
  workloadType: localModelDownloadJob
```

> ⚠️ 이미 `s3://` + `localModelDownloadJob` 용 CSC가 있으면 새로 만들지 말고 편집하세요.
> EKS면 access key 대신 **IRSA**(서비스어카운트 IAM 역할)가 더 깔끔합니다.

### 3️⃣ 노드그룹 (2노드 규모)

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LocalModelNodeGroup
metadata:
  name: h100-llm
spec:
  storageLimit: 300Gi            # 노드당 NVMe 예산 (모델 150Gi + 버퍼)
  persistentVolumeClaimSpec:
    accessModes: [ReadWriteOnce]
    resources: { requests: { storage: 300Gi } }
    storageClassName: local-storage
    volumeMode: Filesystem
  persistentVolumeSpec:
    accessModes: [ReadWriteOnce]
    volumeMode: Filesystem
    capacity: { storage: 300Gi }
    local:
      path: /mnt/models          # 0️⃣에서 확인한 agent hostPath와 반드시 일치
    storageClassName: local-storage
    nodeAffinity:
      required:
        nodeSelectorTerms:
          - matchExpressions:
              - key: kserve/localmodel-node
                operator: In
                values: [ h100-llm ]
```

### 4️⃣ 모델 캐시 + ⏳ 완료 대기 (GATE 1)

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LocalModelCache
metadata:
  name: llama3-70b
spec:
  sourceModelUri: s3://<your-bucket>/models/llama3-70b   # 5️⃣ storageUri와 "완전 일치"
  modelSize: 150Gi
  nodeGroups: [ h100-llm ]
```

```bash
# 🔴 GATE: 2개 노드 모두 NodeDownloaded + copies.available==total 될 때까지 대기
kubectl get localmodelcache llama3-70b \
  -o jsonpath='{.status.copies}{"\n"}{.status.nodeStatus}{"\n"}' -w
```

### 5️⃣ 서빙 — InferenceService (BF16 TP=4, GATE 2: storageUri 일치)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama3-70b
  namespace: <your-ns>
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 3                 # 산정한 최대치 (고정하려면 min도 3)
    model:
      modelFormat: { name: huggingface }
      storageUri: s3://<your-bucket>/models/llama3-70b   # 🔑 sourceModelUri와 동일
      args:
        - --tensor-parallel-size=4
        - --max-model-len=8192
        - --gpu-memory-utilization=0.9
      resources:
        requests: { cpu: "8", memory: 64Gi, nvidia.com/gpu: "4" }
        limits:   { cpu: "8", memory: 64Gi, nvidia.com/gpu: "4" }
```

```bash
kubectl get isvc llama3-70b -n <your-ns>          # READY=True 확인
kubectl get pods -n <your-ns> \
  -l serving.kserve.io/inferenceservice=llama3-70b -o wide
# → h100-llm 노드에 뜨고 storage-initializer 없이 빠르게 Ready면 캐시 적중 ✅
```

---

## 5. H100 최적화 대안 — FP8 (GPU 절반)

5️⃣의 `args` / `resources`만 교체:

```yaml
      args:
        - --tensor-parallel-size=2      # 4 → 2
        - --quantization=fp8
        - --max-model-len=8192
        - --gpu-memory-utilization=0.9
      resources:
        requests: { cpu: "8", memory: 64Gi, nvidia.com/gpu: "2" }   # 4 → 2
        limits:   { cpu: "8", memory: 64Gi, nvidia.com/gpu: "2" }
```

- replica당 **2 GPU** → 노드당 **4 replica** → 3 replica에 GPU 6장 (BF16은 12장).
- 노드그룹은 그대로 **2노드**(HA)로 충분. FP8 체크포인트면 `modelSize`를 ~80Gi로 낮춰도 됨.

---

## 6. 운영 체크리스트

- [ ] **HA 배치**: 3 replica가 한 노드에 몰리지 않게 predictor에 `topologySpreadConstraints` 또는 `podAntiAffinity`로 노드 분산.
- [ ] **LLM 오토스케일**: CPU% HPA는 부정확 → **KEDA + vLLM 메트릭**(대기 요청 수 / KV 사용률) 권장.
- [ ] **이미지 버전 고정**: `storage-initializer` / 런타임 버전을 프로덕션에서 핀 고정.
- [ ] **권한**: hostPath는 `fsGroup` 무시 → 캐시 경로 소유권 `chown -R 1000:1000 /mnt/models` (권한 에러 시).
- [ ] **Cluster Autoscaler 주의**: 동적으로 새로 붙는 노드는 캐시가 없어 재다운로드 → 캐시 노드 풀을 미리 고정 확보.

---

## 7. 🔴 반드시 지킬 2개의 관문

1. **GATE 1 — 다운로드 완료 대기**: `NodeDownloaded` / `copies.available == total` 확인 **전에** ISVC 올리면 캐시 무시하고 S3에서 그냥 받음.
2. **GATE 2 — storageUri 일치**: ISVC `storageUri` == LocalModelCache `sourceModelUri` (문자 그대로 동일)여야 캐시 적중.

---

## 8. 빠른 명령어 레퍼런스

```bash
# 캐시 상태
kubectl get localmodelcache llama3-70b -o yaml
kubectl get localmodelnode <node> -o yaml

# 다운로드 Job / 로그
kubectl get jobs -n kserve-localmodel-jobs
kubectl logs job/<job-name> -n kserve-localmodel-jobs

# 서빙 상태
kubectl get isvc llama3-70b -n <your-ns>
kubectl get pods -n <your-ns> -l serving.kserve.io/inferenceservice=llama3-70b -o wide

# 적용 순서
# 1) ConfigMap 편집  2) Secret+CSC  3) NodeGroup  4) LocalModelCache(+대기)  5) InferenceService
```

# KServe 70B 서빙 — 복잡도 사다리

> 처음부터 Lv.3를 다 할 필요 없음. PoC는 Lv.1, 운영은 Lv.2~3에서 선택.

| 단계 | 하는 일 | 복잡도 | 속도 |
|---|---|---|---|
| **Lv.1 그냥 서빙** | ISVC 1개 (`storageUri: s3://`) | ⭐ | 느림 (매번 다운로드) |
| **Lv.2 공유 PVC** | 모델 1회 복사 + ISVC | ⭐⭐ | 빠름 (캐시 CRD 없음) |
| **Lv.3 LocalModelCache** | 풀세트 (ConfigMap → Secret/CSC → NodeGroup → Cache → ISVC) | ⭐⭐⭐ | 가장 빠름 + 오토스케일 |

## 고르는 법
- **PoC / 일단 뜨는지 확인** → Lv.1
- **운영 · 단순하게** → Lv.2
- **운영 · 빠른 오토스케일 필수** → Lv.3

## 공통
1회성 세팅을 끝내면, 모델 추가 시 실제로 만지는 건 **`LocalModelCache` + `InferenceService` 2개뿐**.


# LocalModelCache vs 공유 PVC — 속도 & 트레이드오프

## 왜 보통 LocalModelCache가 빠른가
- 모델이 **노드 로컬 NVMe**에 있음 → 디스크→GPU 로딩이 **네트워크 홉 없이** 최고 대역폭.
- scale-up 때 여러 파드가 떠도 **네트워크 스토리지 경합이 없음** (각 노드가 자기 로컬 디스크에서 읽음).

## 단, PVC 속도는 백엔드 나름
공유 PVC는 뭘로 만드냐에 따라 천차만별:

| PVC 백엔드 | 속도 |
|---|---|
| **EFS / NFS** | 느릴 수 있음 (네트워크 FS, 대역폭·지연 불리) |
| **FSx for Lustre** | 로컬 NVMe에 필적 (병렬 FS, 수 GB/s). + GPUDirect Storage면 더 빠름 |

대략적인 로딩 속도 순서:

```
로컬 NVMe(LocalModelCache) ≈ FSx Lustre  >  EFS/NFS  ≫  매번 S3 다운로드(Lv.1)
```

→ 즉 **FSx Lustre로 만든 공유 PVC는 LocalModelCache랑 거의 비슷하게** 빠릅니다. EFS로 만들면 느린 거고요.

## 근데 속도만 보면 안 됨 — 트레이드오프

| 항목 | LocalModelCache | 공유 PVC (RWX) |
|---|---|---|
| 로컬 읽기 속도 | **최고** (NVMe) | 백엔드 나름 (Lustre면 최고급) |
| 디스크 사용 | **노드마다 복사본** (총량↑) | 복사본 1개 공유 (절약) |
| 배치 유연성 | 낮음 (rigid 노드 핀닝) | **높음** (아무 노드서나 마운트) |
| 노드 장애 | 약함 (PV가 노드에 묶임) | **강함** (데이터 노드 독립) |
| 멀티노드 서빙 | ❌ (노드 로컬이라 불가) | ✅ **필수** |
| 동시 콜드스타트 | 경합 없음 | 많은 파드가 같은 볼륨 읽으면 **대역 병목** 가능 |

## 결론
- **단일노드 + 극한 속도/오토스케일** → LocalModelCache (70B · TP=4 케이스에 딱).
- **멀티노드거나, 운영 단순·유연성 중시** → **FSx Lustre PVC**가 실전에서 아주 좋은 균형. 속도도 거의 안 밀림.
- **EFS/NFS PVC** = "S3보다야 낫지만 가장 빠른 건 아니다" 정도.

> 표의 "Lv.3가 가장 빠름"은 **로컬 NVMe 기준**. FSx Lustre로 Lv.2를 하면 **Lv.3에 거의 근접**.
> 속도 차이보다 **"노드 핀닝 vs 유연성", "단일노드 vs 멀티노드"** 가 실제 선택 기준이 되는 경우가 많음.
