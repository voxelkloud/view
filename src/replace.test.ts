import { describe, expect, it } from "vitest";
import { createReplaceScratch, filterReplacedParents } from "./replace.js";
import type { ReplaceTreeView } from "./replace.js";

/** A parent with `fanout` children, indices 1..fanout. */
function tree(fanout: number, replaces: number[]): ReplaceTreeView {
  const children = Array.from({ length: fanout }, (_, c) => ({ index: c + 1 }));
  return {
    nodeCount: fanout + 1,
    node: (i) => (i === 0 ? { children } : { children: [] }),
    nodeReplaces: Uint8Array.from(replaces),
  };
}

function run(
  t: ReplaceTreeView,
  selection: number[],
  resident: number[],
): number[] {
  const indices = Int32Array.from(selection);
  const out = new Int32Array(selection.length);
  const residentSet = new Set(resident);
  const n = filterReplacedParents(
    t,
    indices,
    selection.length,
    (i) => residentSet.has(i),
    createReplaceScratch(),
    out,
  );
  return Array.from(out.subarray(0, n));
}

describe("filterReplacedParents", () => {
  it("passes an additive tree through untouched", () => {
    // Every octree format, which is the common case and must cost nothing.
    const additive: ReplaceTreeView = {
      nodeCount: 3,
      node: () => ({ children: [] }),
    };
    expect(run(additive, [0, 1, 2], [0, 1, 2])).toEqual([0, 1, 2]);
  });

  it("hides a parent whose children are all selected AND resident", () => {
    const t = tree(2, [1, 0, 0]);
    expect(run(t, [0, 1, 2], [0, 1, 2])).toEqual([1, 2]);
  });

  it("keeps the parent while a child is still streaming", () => {
    // THE reason residency is in this decision at all: dropping the parent here
    // shows nothing where the picture was about to get better.
    const t = tree(2, [1, 0, 0]);
    expect(run(t, [0, 1, 2], [0, 1])).toEqual([0, 1, 2]);
  });

  it("keeps the parent when the scheduler declined a child", () => {
    // A child the budget or the frustum dropped is a region only the parent
    // covers. Conservative on purpose.
    const t = tree(2, [1, 0, 0]);
    expect(run(t, [0, 1], [0, 1, 2])).toEqual([0, 1]);
  });

  it("keeps a REPLACE leaf, which covers nothing but itself", () => {
    const t = tree(0, [1]);
    expect(run(t, [0], [0])).toEqual([0]);
  });

  it("keeps an ADD parent even when every child is ready", () => {
    // The point of the flag: an additive parent's points are its own, and
    // hiding it deletes them from the picture.
    const t = tree(2, [0, 0, 0]);
    expect(run(t, [0, 1, 2], [0, 1, 2])).toEqual([0, 1, 2]);
  });

  it("mixes modes within one tree, because 3D Tiles allows it", () => {
    // Node 0 REPLACE with children 1,2; node 1 ADD with children 3,4.
    const t: ReplaceTreeView = {
      nodeCount: 5,
      node: (i) =>
        i === 0
          ? { children: [{ index: 1 }, { index: 2 }] }
          : i === 1
            ? { children: [{ index: 3 }, { index: 4 }] }
            : { children: [] },
      nodeReplaces: Uint8Array.from([1, 0, 0, 0, 0]),
    };
    expect(run(t, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])).toEqual([1, 2, 3, 4]);
  });

  it("ignores an undefined child slot", () => {
    // An octree's child array is sparse; a tileset's is not. Both arrive here.
    const t: ReplaceTreeView = {
      nodeCount: 3,
      node: (i) =>
        i === 0 ? { children: [undefined, { index: 1 }, undefined] } : { children: [] },
      nodeReplaces: Uint8Array.from([1, 0, 0]),
    };
    expect(run(t, [0, 1], [0, 1])).toEqual([1]);
  });

  it("preserves the selection's order, which is the streaming priority", () => {
    const t: ReplaceTreeView = {
      nodeCount: 4,
      node: () => ({ children: [] }),
      nodeReplaces: new Uint8Array(4),
    };
    expect(run(t, [3, 1, 2, 0], [0, 1, 2, 3])).toEqual([3, 1, 2, 0]);
  });

  it("reuses its scratch across frames without clearing", () => {
    // Epoch stamping: a node selected in frame 1 must not still read as
    // selected in frame 2. Getting this wrong hides a parent whose children
    // left the selection.
    const t = tree(2, [1, 0, 0]);
    const scratch = createReplaceScratch(8);
    const out = new Int32Array(8);
    const resident = new Set([0, 1, 2]);

    const first = filterReplacedParents(
      t,
      Int32Array.from([0, 1, 2]),
      3,
      (i) => resident.has(i),
      scratch,
      out,
    );
    expect(Array.from(out.subarray(0, first))).toEqual([1, 2]);

    // Frame two: only the parent and one child are selected.
    const second = filterReplacedParents(
      t,
      Int32Array.from([0, 1]),
      2,
      (i) => resident.has(i),
      scratch,
      out,
    );
    expect(Array.from(out.subarray(0, second))).toEqual([0, 1]);
  });

  it("grows its scratch for a tree that got bigger", () => {
    const t = tree(2, [1, 0, 0]);
    const scratch = createReplaceScratch(1);
    const out = new Int32Array(4);
    const n = filterReplacedParents(
      t,
      Int32Array.from([0, 1, 2]),
      3,
      () => true,
      scratch,
      out,
    );
    expect(Array.from(out.subarray(0, n))).toEqual([1, 2]);
    expect(scratch.stamp.length).toBeGreaterThanOrEqual(3);
  });
});

describe("the shape the view calls it with", () => {
  /** What `PointCloudView.drawList` does, reproduced exactly. */
  function drawList(
    tree: ReplaceTreeView,
    selection: { indices: Int32Array; count: number },
    resident: Set<number>,
    handle: { visible: Int32Array; scratch: ReturnType<typeof createReplaceScratch> },
  ): { indices: Int32Array; count: number } {
    if (tree.nodeReplaces === undefined) {
      return { indices: selection.indices, count: selection.count };
    }
    if (handle.visible.length < selection.indices.length) {
      handle.visible = new Int32Array(selection.indices.length);
    }
    const count = filterReplacedParents(
      tree,
      selection.indices,
      selection.count,
      (i) => resident.has(i),
      handle.scratch,
      handle.visible,
    );
    return { indices: handle.visible, count };
  }

  it("hands an additive tree its OWN array back, with no copy", () => {
    // The property the hot path depends on: every octree format pays one
    // undefined check per frame and nothing else. A copy here would be 4 KB of
    // memmove per cloud per frame for a behaviour none of them use.
    const additive: ReplaceTreeView = {
      nodeCount: 3,
      node: () => ({ children: [] }),
    };
    const selection = { indices: Int32Array.from([0, 1, 2]), count: 3 };
    const handle = { visible: new Int32Array(0), scratch: createReplaceScratch(0) };
    const out = drawList(additive, selection, new Set([0, 1, 2]), handle);
    expect(out.indices).toBe(selection.indices);
    expect(out.count).toBe(3);
    expect(handle.visible.length).toBe(0);
  });

  it("grows its buffer once and reuses it across frames", () => {
    const t = tree(2, [1, 0, 0]);
    const selection = { indices: Int32Array.from([0, 1, 2]), count: 3 };
    const handle = { visible: new Int32Array(0), scratch: createReplaceScratch(0) };
    const first = drawList(t, selection, new Set([0, 1, 2]), handle);
    const buffer = handle.visible;
    expect(Array.from(first.indices.subarray(0, first.count))).toEqual([1, 2]);

    const second = drawList(t, selection, new Set([0, 1, 2]), handle);
    expect(handle.visible).toBe(buffer);
    expect(Array.from(second.indices.subarray(0, second.count))).toEqual([1, 2]);
  });

  it("shows the parent again when a child is evicted", () => {
    // Eviction is the other direction of the same fact: the moment a child
    // stops being resident, the parent has to come back or its ground goes
    // blank.
    const t = tree(2, [1, 0, 0]);
    const selection = { indices: Int32Array.from([0, 1, 2]), count: 3 };
    const handle = { visible: new Int32Array(0), scratch: createReplaceScratch(0) };
    const resident = new Set([0, 1, 2]);
    expect(drawList(t, selection, resident, handle).count).toBe(2);
    resident.delete(2);
    const after = drawList(t, selection, resident, handle);
    expect(Array.from(after.indices.subarray(0, after.count))).toEqual([0, 1, 2]);
  });
});
