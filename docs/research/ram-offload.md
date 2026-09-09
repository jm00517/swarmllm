# RAM-backed weight offload for VRAM-limited WebGPU nodes

Status: RFC / implementation scaffold.

## Goal

Let a node contribute more model weight than fits in dedicated VRAM by keeping
quantized weights in system RAM (or browser storage) and paging contiguous layer
windows into a bounded set of GPU buffers.

Example target: an RTX 4060 8 GB node with tens of GB of system RAM should be
able to participate in a larger swarm without pledging only ~6-7 GB of weights.

This is **not** "make system RAM behave like VRAM". WebGPU does not expose a
portable host-visible storage-buffer residency API. The implementation must keep
CPU copies and explicitly upload pages into GPUBuffer objects.

## Why this is not a small loader tweak

The current fast path assumes permanent GPU residency in three places:

1. `streamEntryToGPU()` creates final GPU buffers while the tensor is arriving
   and returns only the GPU copy.
2. `gpuUploadEntry()` drops `e.qs/e.scales/e.data` after upload.
3. `Qwen35Engine._init()` builds matvec ops and bind groups once, capturing
   permanent GPUBuffer handles for every weight in every layer.

The execution path then records the whole local shard into one command encoder:

- single-token decode: all local layers are encoded before submit;
- batched prefill/verify: all local layers are also encoded before submit.

Because CPU -> GPU uploads are asynchronous queue operations, a pager cannot
replace one layer's buffers with the next layer in the middle of that single
command buffer without restructuring the execution loop.

## Proposed architecture

```
GGUF range / Cache API
        |
        v
CPU packed-weight cache
(q4/q8 nibbles + f16 scales)
        |
        v
WeightPager
  - VRAM budget
  - one or two layer windows
  - upload / evict
  - optional double buffering
        |
        v
Qwen35Engine
  run window 0 -> submit
  upload/prefetch window 1
  run window 1 -> submit
  ...
```

Keep **state**, not weights, resident:

- DeltaNet recurrent state
- convolution state
- attention KV cache
- activation scratch buffers
- uniforms / pipeline objects

Page the large matrices:

- attention projections
- FFN gate/up/down
- DeltaNet projection matrices
- optional LM head as a separate pageable object

Norm vectors, biases and other small f32 tensors can stay resident.

## Implementation phases

### Phase 0 - planner (this PR)

- pure contiguous-window planner in `engine/residency.js`
- transfer-time lower-bound helper
- unit tests
- no runtime behavior change

### Phase 1 - retain CPU weights

Touch `engine/gguf.js` and `room.js`.

Add an alternate loading path that repacks q4/q8 into CPU-owned typed arrays and
does **not** call `streamEntryToGPU()` / `gpuUploadEntry()` for pageable
weights.

Use Cache API for persistent storage as today, but keep only the active shard's
packed CPU entries in memory. An IndexedDB/file-backed page cache can be a later
optimization.

Estimated change: ~100-180 LOC.

### Phase 2 - page object + budget

New `engine/weight_pager.js`.

Responsibilities:

- track a user-specified VRAM budget;
- own one or two reusable residency windows;
- upload packed q4/q8 entries;
- destroy evicted GPUBuffer objects;
- expose `ensureWindow(lo, hi)`;
- collect upload timing and bytes moved.

Estimated change: ~180-300 LOC plus tests.

### Phase 3 - dynamic layer bindings

This is the largest change in `engine/qwen35.js`.

Today `buildLayer()`, `mv()`, `mvB()`, `guOp()`, and the batched layer
tables create bind groups once against permanent buffers.

Paged mode needs either:

A. rebuild the layer's weight-dependent ops/bind groups whenever a window becomes
resident; or

B. keep fixed-size GPU buffers and copy each incoming layer into reusable slots,
so bind groups remain stable.

B is preferable. It avoids repeated bind-group allocation and is friendlier to
browsers, but requires slot sizing by tensor shape/family.

Estimated change: ~300-500 LOC.

### Phase 4 - split command submission by residency window

Refactor:

- `_runBatchAndRead()`
- `prefillTokens()`
- single-token layer execution
- hidden-state worker execution

from "encode all layers then submit once" to roughly:

```js
for (const window of pager.windows) {
  await pager.ensure(window);
  const enc = device.createCommandEncoder();
  for (const layer of window.layers) encodeLayer(enc, layer);
  device.queue.submit([enc.finish()]);
  // prefetch next window where possible
}
```

For correctness, the activation buffer and recurrent/KV state remain on GPU
between submits.

Estimated change: ~150-250 LOC.

### Phase 5 - room/UI integration

Separate two concepts that are currently represented by `contribGB`:

- resident GPU weight budget
- CPU/offload weight budget

The shard planner should avoid assigning a node more than its CPU budget, while
the local pager uses only its GPU budget at a time.

Example:

```
GPU weight budget: 5.5 GB
CPU weight budget: 28 GB
```

Estimated change: ~100-180 LOC.

## Rough scope

A useful Qwen3.5/3.8 paged prototype is likely **800-1,400 LOC** including tests
and instrumentation. DenseEngine support can follow after Qwen35Engine works.

The hard part is not GGUF loading. It is breaking the assumption that every
weight buffer is permanently resident while preserving the optimized batched and
speculative paths.

## Expected performance

This mode trades capacity for speed.

For dense models, every token reads essentially all assigned weight bytes. If a
node must move 20 GB/token over an effective 12 GB/s PCIe path, the transfer
lower bound alone is ~1.67 s/token before compute.

Therefore the first useful targets are:

1. shards that exceed VRAM only modestly;
2. large prefill batches, where one uploaded window can serve many prompt tokens;
3. future MoE models, where only selected experts need to be paged;
4. double-buffered upload/compute overlap.

The room should prefer ordinary fully-resident sharding whenever aggregate VRAM
is sufficient, and use RAM offload only as an explicit capacity mode.

## Non-goals for the first implementation

- transparent OS-managed shared GPU memory;
- treating `adapter.limits.maxBufferSize` as available VRAM;
- paging recurrent/KV state;
- arbitrary model architectures;
- promising native-like decode speed while paging dense weights.
