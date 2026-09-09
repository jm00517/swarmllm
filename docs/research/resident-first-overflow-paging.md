# Resident-first + overflow paging

상태: 별도 후속 RFC.

## 문제

RAM-backed paging은 WebGPU 노드의 **용량 한계**를 넘길 수 있게 해주지만,
dense 모델에서 모든 weight를 매 token마다 RAM -> GPU로 옮기면 PCIe bandwidth가
병목이 된다.

따라서 paging을 기본 실행 방식으로 쓰기보다,
**GPU에 들어가는 weight는 최대한 계속 resident하게 두고 실제 overflow만 paging**
하는 구조가 더 적합하다.

## 목표

노드의 모델 shard를 두 구역으로 나눈다.

```
local shard
├─ resident region
│  └─ 가능한 만큼 VRAM에 영구 상주
└─ overflow region
   └─ 시스템 RAM backing store
      └─ 필요한 window만 GPU slot으로 page-in
```

Swarm 전체에서는 다음 우선순위를 사용한다.

1. aggregate VRAM에 들어가는 만큼 기기 사이에 resident sharding
2. 각 노드 내부에서 resident weight를 최대화
3. 그래도 넘치는 부분만 RAM paging
4. 미래 MoE에서는 overflow expert를 선택적으로 paging

## 왜 이쪽이 더 빠른가

dense decode에서는 한 token을 만들 때 local shard의 weight를 거의 전부 읽는다.

예를 들어 local shard가 12GB이고 GPU weight budget이 6GB라면:

### 전부 paging

```
token마다 약 12GB RAM -> GPU 전송
```

### resident-first

```
6GB resident
+
6GB overflow만 token마다 전송
```

전송량을 절반 수준으로 줄일 수 있다.

Swarm에 다른 GPU 노드가 있으면 overflow 자체를 더 줄일 수 있다.

## scheduler 개념

각 노드는 최소한 다음 값을 광고한다.

```
gpuResidentBudgetGB
cpuWeightBudgetGB
estimatedUploadGBps
estimatedGpuWeightGBps
networkRTT
networkMbps
```

host planner는 모델 layer를 나눌 때 단순한 `contribGB` 대신
resident capacity와 overflow cost를 함께 고려한다.

목표 cost 예시:

```
cost =
  gpu_compute_time
+ network_hidden_transfer
+ overflow_upload_time
```

overflow_upload_time은 대략:

```
overflow_weight_bytes / measured_host_to_device_bandwidth
```

로 추정할 수 있다.

## 구현 방향

### 1. resident/paged split

현재 RAM paging PR의 `WeightPager`를 그대로 사용하되,
모든 large matrix를 pager로 보내지 않는다.

layer 단위로:

```
resident layer
-> 기존 permanent GPUBuffer 경로

overflow layer
-> CPU packed entry + WeightPager
```

를 선택한다.

### 2. hot-set

초기에는 단순히 앞쪽 또는 뒤쪽 contiguous layer를 resident로 둔다.

이후 실제 profile을 기반으로:
- attention-heavy layer
- 비용이 큰 layer
- 자주 재사용되는 MoE expert

등을 우선 resident하게 둘 수 있다.

### 3. multi-device planner

Swarm 전체 GPU resident capacity가 충분하면 paging을 사용하지 않는다.

```
if aggregateResidentCapacity >= modelWeight:
    fully resident swarm
else:
    resident sharding + minimum overflow paging
```

### 4. MoE expert cache

MoE에서는 expert별 hot/cold cache가 가능하다.

```
hot experts  -> VRAM resident
cold experts -> RAM
router result -> missing expert만 page-in
```

dense layer paging보다 이 구조에서 RAM offload의 가치가 훨씬 크다.

## 현재 RAM paging PR과의 관계

현재 진행 중인 RAM paging PR은 이 RFC의 기반 primitive를 만든다.

먼저 다음을 완성한다.

- CPU packed weight backing store
- WeightPager
- single-token paging correctness
- multi-layer window
- batched prefill / speculative path 복구
- instrumentation

그 다음 이 RFC에서:
- resident/paged 혼합 배치
- planner cost model
- automatic budget selection
- MoE expert cache

순서로 확장한다.

즉 현재 PR은 **paging engine**, 이 PR은 **paging을 언제/얼마나 쓸지 결정하는 scheduler 정책**을 담당한다.
