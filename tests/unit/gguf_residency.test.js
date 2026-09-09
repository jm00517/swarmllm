import { GGML_Q4_0, ggufEntry } from "../../engine/gguf.js";
import { assert, assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

function fakeQ4Tensor() {
  const bytes = new Uint8Array(18);
  // f16 scale = 1.0, then 16 packed Q4 bytes.
  bytes[0] = 0x00;
  bytes[1] = 0x3c;
  bytes.fill(0x88, 2);
  return {
    info: { name: "blk.0.ffn_gate.weight", shape: [1, 32], ggmlType: GGML_Q4_0, nElems: 32, byteLength: 18 },
    bytes,
  };
}

Deno.test("cpuBacked GGUF entry bypasses direct GPU streaming", async () => {
  const { info, bytes } = fakeQ4Tensor();
  let streamed = 0, fetched = 0;
  const G = {
    tensors: { [info.name]: info },
    streamEntry: async () => { streamed++; throw new Error("should not stream"); },
  };

  const e = await ggufEntry(
    G,
    async () => { fetched++; return bytes; },
    info.name,
    false,
    () => {},
    { cpuBacked: true },
  );

  assertEquals(streamed, 0);
  assertEquals(fetched, 1);
  assertEquals(e.kind, "q4");
  assert(e.qs instanceof Uint8Array);
  assert(e.scales instanceof Uint32Array);
  assertEquals(e.qs.byteLength, 16);
});

Deno.test("resident GGUF entry keeps the existing direct-to-GPU path", async () => {
  const { info } = fakeQ4Tensor();
  let streamed = 0, fetched = 0;
  const sentinel = { kind: "q4", shape: info.shape, gpu: { kind: "q4", qs: {}, sc: {} } };
  const G = {
    tensors: { [info.name]: info },
    streamEntry: async () => { streamed++; return sentinel; },
  };

  const e = await ggufEntry(
    G,
    async () => { fetched++; throw new Error("should not fetch CPU bytes"); },
    info.name,
    false,
  );

  assertEquals(streamed, 1);
  assertEquals(fetched, 0);
  assertEquals(e, sentinel);
});
