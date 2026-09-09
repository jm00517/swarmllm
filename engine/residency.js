// Helpers for planning RAM-backed weight residency on VRAM-limited WebGPU nodes.
//
// This module is intentionally pure: the first RAM-offload PR only establishes
// the paging model and tests. Runtime integration comes in a follow-up because
// Qwen35Engine currently bakes permanent GPUBuffer handles into bind groups.

export function planLayerWindows(layerBytes, gpuBudgetBytes, reserveBytes = 0) {
  if (!Array.isArray(layerBytes) || !layerBytes.length) return [];
  const usable = Math.floor(gpuBudgetBytes - reserveBytes);
  if (!Number.isFinite(usable) || usable <= 0) throw new Error("GPU budget must exceed reserve");

  const windows = [];
  let lo = 0, bytes = 0;
  for (let i = 0; i < layerBytes.length; i++) {
    const b = Math.floor(layerBytes[i]);
    if (!Number.isFinite(b) || b < 0) throw new Error("layer byte sizes must be non-negative finite numbers");
    if (b > usable) throw new Error(`layer ${i} needs ${b} bytes but only ${usable} bytes are available for paged weights`);
    if (bytes && bytes + b > usable) {
      windows.push({ lo, hi: i, bytes });
      lo = i;
      bytes = 0;
    }
    bytes += b;
  }
  if (bytes || lo < layerBytes.length) windows.push({ lo, hi: layerBytes.length, bytes });
  return windows;
}

export function residencyMode(shardBytes, gpuBudgetBytes, reserveBytes = 0) {
  const usable = Math.max(0, gpuBudgetBytes - reserveBytes);
  return shardBytes <= usable ? "resident" : "paged";
}

export function pagingLowerBoundMs(bytesMovedPerToken, hostToDeviceGBs) {
  if (!(hostToDeviceGBs > 0)) return Infinity;
  return bytesMovedPerToken / (hostToDeviceGBs * 1e9) * 1000;
}
