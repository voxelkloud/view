// Task T1: the scheduler stops counting to eight.
//
// Every format this repo read until now indexes space by halving a cube, so
// eight was not an assumption — it was the shape of the data. A tileset tile has
// any number of children in no particular arrangement: a quadtree has four, an
// explicit tileset has whatever the tiler wrote, and `childMask` stops being
// octant occupancy and becomes a tri-state.
//
// These trees are hand-built and deliberately NOT octrees. An octree cannot
// reproduce the failure this guards: with the old `c < 8` loop a ninth child is
// not culled, not deferred, not reported — it is invisible.

import { describe, expect, it } from "vitest";
import { Containment, extractFrustumPlanes } from "./frustum.js";
import {
  createLodScratch,
  createLodSelection,
  resolveLodOptions,
  selectVisible,
} from "./select.js";
import type {
  LodCameraState,
  LodKernels,
  LodTreeView,
} from "./select.js";
import type { PointCloudNode } from "@voxelkloud/core";

interface Built {
  readonly nodes: PointCloudNode[];
  readonly tree: LodTreeView;
}

/**
 * A two-level tree with `fanout` children under the root, laid out in a row
 * along X so every child has its own box and none is nested in another.
 *
 * `childMask` is set to the CHILD COUNT, not to a bit pattern — the tri-state
 * reading, and a value no octant mask could produce once fanout exceeds 8.
 */
function fanoutTree(fanout: number, ownPoints = 1000): Built {
  const nodes: PointCloudNode[] = [];
  const children: (PointCloudNode | undefined)[] = [];
  const root = {
    index: 0,
    name: "root",
    level: 0,
    minX: 0,
    minY: 0,
    minZ: 0,
    maxX: fanout,
    maxY: 1,
    maxZ: 1,
    numPoints: ownPoints,
    childMask: fanout,
    children,
    parent: undefined,
  } as unknown as PointCloudNode;
  nodes.push(root);
  for (let c = 0; c < fanout; c++) {
    const child = {
      index: c + 1,
      name: `c${c}`,
      level: 1,
      minX: c,
      minY: 0,
      minZ: 0,
      maxX: c + 1,
      maxY: 1,
      maxZ: 1,
      numPoints: ownPoints,
      childMask: 0,
      children: [],
      parent: root,
    } as unknown as PointCloudNode;
    nodes.push(child);
    children.push(child);
  }

  const tree: LodTreeView = {
    nodeCount: nodes.length,
    root,
    node: (i) => nodes[i],
    // Flat and generous: every child clears the target at any sane camera, so
    // what the assertions see is the TRAVERSAL and not the metric.
    geometricErrorAt: (l) => 100 / 2 ** l,
    pointSpacingAt: (l) => 100 / 2 ** l,
    boundingRadiusAt: (l) => 10 / 2 ** l,
    tryExpandSync: () => true,
    requestExpand: () => {},
  };
  return { nodes, tree };
}

/** A camera outside the row, looking down the +X axis at all of it. */
function cameraOver(tree: LodTreeView, scratch = createLodScratch()): {
  cam: LodCameraState;
  scratch: ReturnType<typeof createLodScratch>;
} {
  const root = tree.root;
  const cam: LodCameraState = {
    // An identity clip matrix would cull everything outside [-1,1]; the planes
    // are overwritten below with six that contain the whole row instead.
    clipFromAbs: new Float64Array(16),
    camX: (root.minX + root.maxX) / 2,
    camY: -50,
    camZ: 0.5,
    slope: Math.tan(Math.PI / 6),
    viewportHeightPx: 1080,
    orthographic: false,
    orthoProjFactor: 0,
    nearFloor: 0.1,
    depthRange: "zero-to-one",
    reversedDepth: false,
  };
  // Six planes that contain everything: `classifyAabb` then always answers
  // Inside, which is what isolates the traversal from the frustum maths.
  for (let p = 0; p < 6; p++) {
    scratch.planes[p * 4] = 0;
    scratch.planes[p * 4 + 1] = 0;
    scratch.planes[p * 4 + 2] = 0;
    scratch.planes[p * 4 + 3] = 1e9;
  }
  return { cam, scratch };
}

function select(tree: LodTreeView, opts = {}, kernels?: LodKernels) {
  const { cam, scratch } = cameraOver(tree);
  const out = createLodSelection();
  selectVisible(tree, cam, resolveLodOptions(opts), scratch, out, kernels);
  return out;
}

describe("N-ary traversal", () => {
  it("reaches every child of a 12-wide tile, not the first eight", () => {
    // THE regression this task exists for. The old loop stopped at slot 7, so
    // children 8..11 were never pushed and never reported — the selection was
    // silently short by a third.
    const { tree } = fanoutTree(12);
    const out = select(tree, { pointBudget: 1e9 });
    expect(out.count).toBe(13);
    expect(out.limitedBy).toBe("complete");
    const seen = Array.from(out.indices.subarray(0, out.count)).sort(
      (a, b) => a - b,
    );
    expect(seen).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it("handles a quadtree, which is four and not eight", () => {
    const { tree } = fanoutTree(4);
    const out = select(tree, { pointBudget: 1e9 });
    expect(out.count).toBe(5);
    expect(out.levelCounts[1]).toBe(4);
  });

  it("handles fanouts either side of the kernel block boundary", () => {
    for (const fanout of [1, 3, 7, 8, 9, 16, 17, 33]) {
      const { tree } = fanoutTree(fanout);
      const out = select(tree, { pointBudget: 1e9 });
      expect(out.count, `fanout ${fanout}`).toBe(fanout + 1);
    }
  });

  it("reads childMask as a tri-state, not as octant bits", () => {
    // 12 has bits 2 and 3 set. Under the bit reading that would claim two
    // octants and skip ten real children; under the tri-state it means
    // "has children, walk them".
    const { tree, nodes } = fanoutTree(12);
    expect(nodes[0]!.childMask).toBe(12);
    expect(select(tree, { pointBudget: 1e9 }).count).toBe(13);
  });

  it("still treats 0 as a leaf and undefined as unknown", () => {
    const { tree, nodes } = fanoutTree(4);
    const leafRoot = { ...nodes[0]!, childMask: 0 } as PointCloudNode;
    const leaf: LodTreeView = { ...tree, root: leafRoot, node: (i) => (i === 0 ? leafRoot : nodes[i]) };
    expect(select(leaf, { pointBudget: 1e9 }).count).toBe(1);

    const unknownRoot = { ...nodes[0]!, childMask: undefined } as PointCloudNode;
    const unknown: LodTreeView = {
      ...tree,
      root: unknownRoot,
      node: (i) => (i === 0 ? unknownRoot : nodes[i]),
      tryExpandSync: () => false,
    };
    const out = select(unknown, { pointBudget: 1e9 });
    expect(out.count).toBe(1);
    expect(out.needsExpandCount).toBe(1);
  });
});

describe("nodePointCount (T1)", () => {
  it("charges the budget from the array when the tree has one", () => {
    // A tileset seeds a nominal count and corrects it from the decoded tile.
    // The budget must spend the corrected number, or a cloud whose nominal was
    // 10x too high refuses to refine and one 10x too low blows the budget.
    const { tree, nodes } = fanoutTree(4, 1000);
    const counts = new Float64Array(nodes.length);
    counts.fill(100);
    const corrected: LodTreeView = { ...tree, nodePointCount: counts };

    // 2500 admits the root plus ONE child at 1000 apiece — the second would
    // reach 3000 — and all four under the corrected counts.
    expect(select(tree, { pointBudget: 2500 }).count).toBe(2);
    expect(select(tree, { pointBudget: 2500 }).limitedBy).toBe("budget");
    expect(select(corrected, { pointBudget: 2500 }).count).toBe(5);
    expect(select(corrected, { pointBudget: 2500 }).points).toBe(500);
  });

  it("reproduces the node's own counts EXACTLY when filled with them", () => {
    // The same differential rule A1 set for the other three arrays: filling the
    // override with what it overrides must move nothing.
    const { tree, nodes } = fanoutTree(6, 700);
    const counts = Float64Array.from(nodes.map((n) => n.numPoints));
    const plain = select(tree, { pointBudget: 3000 });
    const dense = select({ ...tree, nodePointCount: counts }, { pointBudget: 3000 });
    expect(dense.count).toBe(plain.count);
    expect(dense.points).toBe(plain.points);
    expect(dense.limitedBy).toBe(plain.limitedBy);
    expect(Array.from(dense.indices.subarray(0, dense.count))).toEqual(
      Array.from(plain.indices.subarray(0, plain.count)),
    );
  });
});

describe("the kernel over an N-ary node", () => {
  /** Records the mask of every block it is handed, and survives nothing. */
  function recordingKernel(): LodKernels & { masks: number[] } {
    return {
      planes: new Float64Array(24),
      boxes: new Float64Array(48),
      results: new Float64Array(16),
      params: new Float64Array(11),
      child: new Float64Array(16),
      masks: [],
      selectChildren(mask: number): number {
        this.masks.push(mask);
        return 0;
      },
    };
  }

  it("is called once per block of eight, with the tail block short", () => {
    // 12 children is two crossings: a full block of eight and a tail of four.
    // The kernel itself never changed — it still reads eight slots — which is
    // the point of blocking rather than widening it.
    const { tree } = fanoutTree(12);
    const k = recordingKernel();
    select(tree, { pointBudget: 1e9 }, k);
    expect(k.masks).toEqual([0xff, 0x0f]);
  });

  it("issues exactly one crossing for an eight-slot node", () => {
    const { tree } = fanoutTree(8);
    const k = recordingKernel();
    select(tree, { pointBudget: 1e9 }, k);
    expect(k.masks).toEqual([0xff]);
  });

  it("does not claim slots the tail block does not have", () => {
    const { tree } = fanoutTree(9);
    const k = recordingKernel();
    select(tree, { pointBudget: 1e9 }, k);
    expect(k.masks).toEqual([0xff, 0x01]);
  });
});

describe("containment still propagates through an N-ary node", () => {
  it("marks children Inside without re-testing planes", () => {
    const { tree } = fanoutTree(12);
    const { cam, scratch } = cameraOver(tree);
    // Planes that contain everything, so the root classifies Inside and every
    // child must inherit it rather than being classified again.
    extractFrustumPlanes(new Float64Array(16), scratch.planes, "zero-to-one", false);
    for (let p = 0; p < 6; p++) {
      scratch.planes[p * 4] = 0;
      scratch.planes[p * 4 + 1] = 0;
      scratch.planes[p * 4 + 2] = 0;
      scratch.planes[p * 4 + 3] = 1e9;
    }
    const out = createLodSelection();
    selectVisible(tree, cam, resolveLodOptions({ pointBudget: 1e9 }), scratch, out);
    expect(out.count).toBe(13);
    expect(Containment.Inside).toBe(2);
  });
});
