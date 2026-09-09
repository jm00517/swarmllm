// VRAM-budgeted uploader for RAM-backed packed weights.
//
// Phase 2 primitive: it owns reusable GPUBuffer slots and copies Q4/Q8 packed
// arrays from CPU RAM into those slots. Qwen35Engine integration happens in
// Phase 3; until then this module is independently unit-tested.

const ALIGN = 256;
const align = (n, a = ALIGN) => Math.ceil(n / a) * a;

export function packedEntryBytes(e) {
  if (!e) return 0;
  if (e.kind === "q4" || e.kind === "q8")
    return (e.qs?.byteLength || 0) + (e.scales?.byteLength || 0);
  if (e.kind === "f32") return e.data?.byteLength || 0;
  throw new Error("unsupported weight kind " + e.kind);
}

export function packedEntrySlotBytes(e) {
  if (!e) return 0;
  if (e.kind === "q4" || e.kind === "q8")
    return align(e.qs?.byteLength || 0) + align(e.scales?.byteLength || 0);
  if (e.kind === "f32") return align(e.data?.byteLength || 0);
  throw new Error("unsupported weight kind " + e.kind);
}

function asBytes(v) {
  if (!v) return new Uint8Array(0);
  return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
}

export class WeightPager {
  constructor(device, {
    budgetBytes,
    reserveBytes = 0,
    chunkBytes = 16 * 2 ** 20,
  } = {}) {
    if (!device) throw new Error("WeightPager requires a GPUDevice");
    if (!(budgetBytes > reserveBytes)) throw new Error("GPU budget must exceed reserve");
    this.device = device;
    this.budgetBytes = Math.floor(budgetBytes);
    this.reserveBytes = Math.floor(reserveBytes);
    this.usableBytes = this.budgetBytes - this.reserveBytes;
    this.chunkBytes = Math.max(4, Math.floor(chunkBytes / 4) * 4);
    this.slots = new Map();
    this.allocatedBytes = 0;
    this.stats = {
      uploads: 0,
      bytesMoved: 0,
      uploadMs: 0,
      reallocations: 0,
      cacheHits: 0,
    };
  }

  _usage() {
    return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  }

  _slotNeed(e) {
    if (e.kind === "q4" || e.kind === "q8") return {
      kind: e.kind,
      qs: align(e.qs.byteLength),
      sc: align(e.scales.byteLength),
      total: align(e.qs.byteLength) + align(e.scales.byteLength),
    };
    if (e.kind === "f32") return {
      kind: e.kind,
      buf: align(e.data.byteLength),
      total: align(e.data.byteLength),
    };
    throw new Error("unsupported weight kind " + e.kind);
  }

  _makeBuffer(size, label) {
    const max = this.device.limits?.maxBufferSize;
    if (max && size > max) throw new Error(`${label} needs ${size} bytes, above maxBufferSize ${max}`);
    return this.device.createBuffer({ size, usage: this._usage(), label });
  }

  _destroySlot(s) {
    try { s.qs?.destroy?.(); } catch {}
    try { s.sc?.destroy?.(); } catch {}
    try { s.buf?.destroy?.(); } catch {}
  }

  _ensureSlot(key, e) {
    const need = this._slotNeed(e);
    let s = this.slots.get(key);
    const current = s?.capacity || 0;

    const fits = s && s.kind === need.kind &&
      (need.kind === "f32" ? s.bufSize >= need.buf : s.qsSize >= need.qs && s.scSize >= need.sc);
    if (fits) return s;

    const nextCapacity = need.total;
    if (this.allocatedBytes - current + nextCapacity > this.usableBytes) {
      throw new Error(
        `WeightPager budget exceeded: slot ${key} needs ${(nextCapacity / 2 ** 20).toFixed(1)} MiB, ` +
        `allocated ${(this.allocatedBytes / 2 ** 20).toFixed(1)} MiB / usable ${(this.usableBytes / 2 ** 20).toFixed(1)} MiB`
      );
    }

    if (s) this._destroySlot(s);
    s = { kind: need.kind, capacity: nextCapacity, entry: null };
    if (need.kind === "f32") {
      s.bufSize = need.buf;
      s.buf = this._makeBuffer(need.buf, `weight-pager:${key}:f32`);
    } else {
      s.qsSize = need.qs; s.scSize = need.sc;
      s.qs = this._makeBuffer(need.qs, `weight-pager:${key}:qs`);
      s.sc = this._makeBuffer(need.sc, `weight-pager:${key}:sc`);
    }
    this.allocatedBytes = this.allocatedBytes - current + nextCapacity;
    this.slots.set(key, s);
    this.stats.reallocations++;
    return s;
  }

  _write(buf, src) {
    const bytes = asBytes(src);
    for (let o = 0; o < bytes.byteLength; o += this.chunkBytes) {
      const n = Math.min(this.chunkBytes, bytes.byteLength - o);
      this.device.queue.writeBuffer(buf, o, bytes, o, n);
    }
  }

  // Copy one CPU-backed packed entry into a reusable named slot.
  // Repeating the same entry in the same slot is a no-op.
  async materialize(key, e) {
    if (!e) throw new Error("cannot materialize empty weight");
    const s = this._ensureSlot(key, e);
    if (s.entry === e) {
      this.stats.cacheHits++;
      return e.kind === "f32"
        ? { kind: "f32", buf: s.buf }
        : { kind: e.kind, qs: s.qs, sc: s.sc };
    }

    const t0 = performance.now();
    if (e.kind === "f32") this._write(s.buf, e.data);
    else {
      this._write(s.qs, e.qs);
      this._write(s.sc, e.scales);
    }
    // queue.writeBuffer only enqueues the copy. Waiting here gives callers a
    // correctness barrier before they reuse the slot for compute.
    await this.device.queue.onSubmittedWorkDone?.();
    const dt = performance.now() - t0;

    s.entry = e;
    this.stats.uploads++;
    this.stats.bytesMoved += packedEntryBytes(e);
    this.stats.uploadMs += dt;
    return e.kind === "f32"
      ? { kind: "f32", buf: s.buf }
      : { kind: e.kind, qs: s.qs, sc: s.sc };
  }

  invalidate(key) {
    const s = this.slots.get(key);
    if (s) s.entry = null;
  }

  snapshot() {
    return {
      ...this.stats,
      allocatedBytes: this.allocatedBytes,
      usableBytes: this.usableBytes,
      slots: this.slots.size,
    };
  }

  destroy() {
    for (const s of this.slots.values()) this._destroySlot(s);
    this.slots.clear();
    this.allocatedBytes = 0;
  }
}
