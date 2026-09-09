# WebGPU RAM weight offload

상태: 구현 진행 중.

## 최종 목표

VRAM이 작은 WebGPU 노드가 시스템 RAM을 weight backing store로 사용해서,
전용 VRAM보다 큰 모델 shard를 맡을 수 있게 한다.

예시 목표:

- RTX 4060 8GB
- 시스템 RAM 수십 GB
- GPU weight budget 5~6GB
- 나머지 Q4/Q8 weight는 RAM에 유지
- 필요한 연속 layer window만 VRAM으로 page-in / page-out

이건 **시스템 RAM을 VRAM처럼 직접 쓰는 기능이 아니다.**
WebGPU에는 portable한 host-visible storage-buffer residency API가 없으므로,
CPU 쪽 packed weight를 유지하고 필요한 시점에 명시적으로 GPUBuffer로 업로드해야 한다.

## 현재 구조에서 막히는 지점

현재 fast path는 weight가 GPU에 영구 상주한다고 가정한다.

1. `streamEntryToGPU()`가 tensor 다운로드 중 최종 GPUBuffer를 바로 만든다.
2. `gpuUploadEntry()`는 GPU 업로드 뒤 `e.qs/e.scales/e.data`를 버린다.
3. `Qwen35Engine._init()`은 각 weight의 permanent GPUBuffer를 캡처한 bind group을 한 번만 만든다.
4. decode/prefill은 local shard의 모든 layer를 하나의 command encoder에 기록한 뒤 한 번 submit한다.

따라서 loader만 수정해서는 RAM offload가 완성되지 않는다.

## 목표 구조

```
GGUF range / Cache API
        |
        v
CPU packed-weight cache
(Q4/Q8 nibbles + f16 scales)
        |
        v
WeightPager
  - VRAM budget
  - residency window
  - upload / evict
  - optional double buffering
        |
        v
Qwen35Engine
  window 0 실행 -> submit
  window 1 page-in
  window 1 실행 -> submit
  ...
```

GPU에 계속 상주시킬 것:

- DeltaNet recurrent state
- convolution state
- attention KV cache
- activation scratch buffer
- uniform / pipeline object
- 작은 norm / bias

pageable 대상으로 둘 것:

- attention projection weight
- FFN gate/up/down
- DeltaNet projection matrix
- 필요하면 LM head

## 구현 단계

### Phase 0 — planner

완료.

- `engine/residency.js`
- 연속 layer window planner
- 전송 시간 하한 계산
- unit test

### Phase 1 — CPU packed weight 유지

완료.

- `ggufEntry(..., { cpuBacked: true })` 경로 추가
- CPU-backed 모드에서는 direct-to-GPU streaming 우회
- Q4/Q8 packed typed array를 RAM에 유지
- `Qwen35Engine`에 `keepCpuWeights` 옵션 추가
- 기존 fully-resident 경로는 그대로 유지
- unit test 추가

현재 테스트 스위치는 URL에 `?ramOffload=1`을 붙여 켠다.

중요: 이 단계에서는 **RAM copy를 유지하면서 GPU에도 전체 shard가 상주한다.**
즉 아직 VRAM 절약은 없다. 다음 Phase 2/3에서 `WeightPager`와 reusable GPU slot을
붙여 실제 page-in/page-out으로 바꾼다.

### Phase 2 — WeightPager

새 파일:

- `engine/weight_pager.js`

역할:

- user-defined VRAM budget
- reusable residency window
- upload / evict
- transfer bytes / timing 수집
- 가능하면 double buffering

예상 변경량: 약 180~300 LOC + tests.

### Phase 3 — Qwen35Engine paged binding

가장 큰 변경.

현재 `buildLayer()`, `mv()`, `mvB()`, `guOp()` 등이
permanent GPUBuffer를 물고 있는 bind group을 생성한다.

paged 모드에서는 두 선택지가 있다.

A. window가 바뀔 때 bind group을 다시 만든다.

B. 고정 크기 GPU slot을 만들고 새 weight를 같은 slot에 복사해서 bind group은 유지한다.

B를 우선한다.
브라우저에서 bind-group churn을 줄일 수 있고 실행 경로도 더 예측 가능하다.

예상 변경량: 약 300~500 LOC.

### Phase 4 — residency window 단위 command submission

현재:

```
모든 local layer encode
-> submit 1회
```

paged 모드:

```js
for (const window of pager.windows) {
  await pager.ensure(window);
  const enc = device.createCommandEncoder();
  for (const layer of window.layers) encodeLayer(enc, layer);
  device.queue.submit([enc.finish()]);
}
```

수정 대상:

- single-token decode
- `_runBatchAndRead()`
- `prefillTokens()`
- speculative verify
- worker hidden-state execution

activation과 recurrent/KV state는 window 사이에도 GPU에 유지한다.

예상 변경량: 약 150~250 LOC.

### Phase 5 — room/UI

현재 `contribGB` 하나로 표현되는 용량을 분리한다.

- GPU resident weight budget
- CPU/offload weight budget

예:

```
GPU weight budget: 5.5 GB
CPU weight budget: 28 GB
```

shard planner는 CPU budget보다 큰 shard를 배정하지 않고,
local pager는 GPU budget만큼만 동시에 resident하게 만든다.

예상 변경량: 약 100~180 LOC.

## 대략적인 전체 규모

Qwen3.5/3.8에서 실제로 쓸 수 있는 prototype은
**800~1,400 LOC** 정도로 예상한다.

핵심 난점은 GGUF loading 자체가 아니라
**모든 weight가 영구 GPU resident라는 엔진 가정을 깨는 것**이다.

## 성능

이 기능은 속도 향상이 아니라 **용량 확장 모드**다.

Dense 모델은 token마다 맡은 weight를 거의 전부 읽는다.

예를 들어 20GB/token을 실효 12GB/s PCIe로 옮기면
전송 시간 하한만 약 1.67초/token이다.

따라서 우선순위는:

1. VRAM을 조금 초과하는 shard
2. 한 번 page-in한 weight를 여러 prompt token에 재사용하는 batched prefill
3. 향후 MoE expert paging
4. upload/compute double buffering

aggregate VRAM이 충분하면 기존 fully-resident sharding을 기본값으로 유지한다.

## 1차 구현에서 하지 않을 것

- OS의 shared GPU memory를 직접 제어한다고 가정
- `adapter.limits.maxBufferSize`를 실제 사용 가능한 VRAM으로 간주
- recurrent/KV state paging
- 모든 model architecture 지원
- dense paging에서도 resident mode와 비슷한 decode 속도를 약속
