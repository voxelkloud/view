// PURE. An index max-heap over parallel typed arrays.
//
// The key IS the stored score. The reference recomputes the closure
// `1 / x.weight` on every comparison, which for its `Number.MAX_VALUE` sentinel
// is 5.56e-309 — a subnormal double, compared millions of times a second.

import type { Containment } from "./frustum.js";

/** The heap's slice of {@link LodScratch}. */
export interface HeapArrays {
  heapNode: Int32Array;
  heapKey: Float64Array;
  heapContainment: Uint8Array;
  /** Pop output, written in place so a pop never allocates. */
  popNode: number;
  popKey: number;
  popContainment: Containment;
}

/**
 * Push one entry. Returns the new heap size.
 *
 * No overflow check is needed: a node is pushed at most once per frame, because
 * it is only ever pushed from its unique admitted parent, and the arrays are
 * sized `nodeCount`.
 */
export function heapPush(
  s: HeapArrays,
  size: number,
  node: number,
  key: number,
  containment: Containment,
): number {
  let i = size;
  const { heapNode, heapKey, heapContainment } = s;
  while (i > 0) {
    const parent = (i - 1) >> 1;
    if (heapKey[parent]! >= key) break;
    heapNode[i] = heapNode[parent]!;
    heapKey[i] = heapKey[parent]!;
    heapContainment[i] = heapContainment[parent]!;
    i = parent;
  }
  heapNode[i] = node;
  heapKey[i] = key;
  heapContainment[i] = containment;
  return size + 1;
}

/**
 * Pop the max entry into `s.popNode` / `s.popKey` / `s.popContainment`. Returns
 * the new heap size. Caller must check `size > 0` first.
 */
export function heapPop(s: HeapArrays, size: number): number {
  const { heapNode, heapKey, heapContainment } = s;
  s.popNode = heapNode[0]!;
  s.popKey = heapKey[0]!;
  s.popContainment = heapContainment[0] as Containment;

  const n = size - 1;
  if (n > 0) {
    const lastNode = heapNode[n]!;
    const lastKey = heapKey[n]!;
    const lastContainment = heapContainment[n]!;

    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      if (l >= n) break;
      const r = l + 1;
      const child = r < n && heapKey[r]! > heapKey[l]! ? r : l;
      if (heapKey[child]! <= lastKey) break;
      heapNode[i] = heapNode[child]!;
      heapKey[i] = heapKey[child]!;
      heapContainment[i] = heapContainment[child]!;
      i = child;
    }
    heapNode[i] = lastNode;
    heapKey[i] = lastKey;
    heapContainment[i] = lastContainment;
  }
  return n;
}
