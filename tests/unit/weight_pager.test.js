import {
  WeightPager,
  packedEntryBytes,
  packedEntrySlotBytes,
} from "../../engine/weight_pager.js";
import {
  assert,
  assertEquals,
  assertRejects,
} from "https://deno.land/std@0.224.0/assert/mod.ts";

globalThis.GPUBufferUsage ??= { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4 };

class FakeBuffer {
  constructor(size) { this.size = size; this.destroyed = false; }
  destroy() { this.destroyed = true; }
}
class FakeDevice {
  constructor(maxBufferSize = 1 << 30) {
    this.limits = { maxBufferSize };
    this.created = [];
    this.writes = [];
    this.queue = {
      writeBuffer: (buf, dst, src, srcOff, size) => this.writes.push({ buf, dst, srcOff, size }),
      onSubmittedWorkDone: async () => {},
    };
  }
  createBuffer({ size, label }) {
    const b = new FakeBuffer(size);
    b.label = label;
    this.created.push(b);
    return b;
  }
}

const q4 = (n = 1024) => ({
  kind: "q4",
  qs: new Uint8Array(n),
  scales: new Uint32Array(Math.ceil(n / 64)),
  shape: [1, n * 2],
});

Deno.test("packed weight byte accounting separates payload and slot alignment", () => {
  const e = { kind: "q4", qs: new Uint8Array(300), scales: new Uint8Array(20) };
  assertEquals(packedEntryBytes(e), 320);
  assertEquals(packedEntrySlotBytes(e), 512 + 256);
});

Deno.test("WeightPager reuses a slot and skips re-upload of the same entry", async () => {
  const d = new FakeDevice();
  const p = new WeightPager(d, { budgetBytes: 4 * 2 ** 20, reserveBytes: 1 * 2 ** 20 });
  const e = q4();
  const a = await p.materialize("ffn-gate", e);
  const writes = d.writes.length;
  const b = await p.materialize("ffn-gate", e);

  assert(a.qs === b.qs);
  assertEquals(d.writes.length, writes);
  assertEquals(p.snapshot().uploads, 1);
  assertEquals(p.snapshot().cacheHits, 1);
});

Deno.test("WeightPager grows a reusable slot but keeps within budget", async () => {
  const d = new FakeDevice();
  const p = new WeightPager(d, { budgetBytes: 2 * 2 ** 20, reserveBytes: 256 * 1024 });
  await p.materialize("w", q4(1024));
  const before = p.snapshot().allocatedBytes;
  await p.materialize("w", q4(4096));
  assert(p.snapshot().allocatedBytes > before);
  assert(p.snapshot().allocatedBytes <= p.snapshot().usableBytes);
  assertEquals(p.snapshot().reallocations, 2);
});

Deno.test("WeightPager rejects a slot that exceeds the VRAM budget", async () => {
  const d = new FakeDevice();
  const p = new WeightPager(d, { budgetBytes: 1024, reserveBytes: 256 });
  await assertRejects(() => p.materialize("huge", q4(4096)), Error, "budget exceeded");
});

Deno.test("destroy releases every allocated buffer", async () => {
  const d = new FakeDevice();
  const p = new WeightPager(d, { budgetBytes: 4 * 2 ** 20 });
  await p.materialize("a", q4());
  p.destroy();
  assert(d.created.every((b) => b.destroyed));
  assertEquals(p.snapshot().allocatedBytes, 0);
});
