import {
  pagingLowerBoundMs,
  planLayerWindows,
  residencyMode,
} from "../../engine/residency.js";
import { assertEquals, assertThrows } from "https://deno.land/std@0.224.0/assert/mod.ts";

Deno.test("resident mode when the whole shard fits", () => {
  assertEquals(residencyMode(6, 8, 1), "resident");
  assertEquals(residencyMode(7, 8, 1), "resident");
  assertEquals(residencyMode(8, 8, 1), "paged");
});

Deno.test("layer windows stay contiguous and inside budget", () => {
  assertEquals(
    planLayerWindows([3, 3, 3, 3, 2], 10, 1),
    [
      { lo: 0, hi: 3, bytes: 9 },
      { lo: 3, hi: 5, bytes: 5 },
    ],
  );
});

Deno.test("one layer must fit in the paged-weight budget", () => {
  assertThrows(() => planLayerWindows([10], 10, 1));
});

Deno.test("paging lower bound is just transfer time", () => {
  assertEquals(Math.round(pagingLowerBoundMs(12e9, 12)), 1000);
});
