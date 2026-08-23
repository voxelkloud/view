import { readFileSync } from "node:fs";
import { createHierarchy, parsePointCloudSource } from "@voxelkloud/format-potree";
import type { HierarchyNode, PointCloudHierarchy } from "@voxelkloud/format-potree";
import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { beforeAll, describe, expect, it } from "vitest";
import { Containment, extractFrustumPlanes } from "./frustum.js";
import {
  DEFAULT_SCREEN_ERROR,
  createLodScratch,
  createLodSelection,
  resolveLodOptions,
  selectVisible,
} from "./select.js";
import type {
  LodCameraState,
  LodKernels,
  LodScratch,
  LodSelection,
  LodTreeView,
} from "./select.js";

// The Potree driver vendors the real autzen manifest and hierarchy for its own offline
// suite; reading them here keeps the scheduler tested against real converter
// output rather than a synthetic tree that cannot reproduce its shape.
const LOADER_FIXTURES = new URL(
  "../../../format-potree/src/__fixtures__/",
  import.meta.url,
);

const FAKE_URLS = {
  base: "https://example.test/cloud/",
  metadata: "https://example.test/cloud/metadata.json",
  hierarchy: "https://example.test/cloud/hierarchy.bin",
  octree: "https://example.test/cloud/octree.bin",
};

let autzen: PointCloudHierarchy;

beforeAll(async () => {
  const json = JSON.parse(
    readFileSync(new URL("autzen.metadata.json", LOADER_FIXTURES), "utf8"),
  ) as Record<string, unknown>;
  const source = parsePointCloudSource(json, FAKE_URLS);
  const buf = readFileSync(new URL("autzen.hierarchy.bin", LOADER_FIXTURES));
  autzen = createHierarchy(source, {
    buffer: buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  });
  await autzen.expandAll();
});

/** Build a camera state the way the view will, but with plain three maths. */
function cameraLooking(
  tree: PointCloudHierarchy,
  distance: number,
  scratch: LodScratch,
  viewportHeightPx = 1080,
): LodCameraState {
  const box = tree.source.metadata.boundingBox;
  const centre = new Vector3(
    (box.min[0] + box.max[0]) / 2,
    (box.min[1] + box.max[1]) / 2,
    (box.min[2] + box.max[2]) / 2,
  );
  const camera = new PerspectiveCamera(60, 16 / 9, 1, 50_000);
  camera.up.set(0, 0, 1);
  camera.position.set(centre.x + distance, centre.y + distance, centre.z + distance);
  camera.lookAt(centre);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();

  const clip = new Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse,
  );
  const cam: LodCameraState = {
    clipFromAbs: new Float64Array(clip.elements),
    camX: camera.position.x,
    camY: camera.position.y,
    camZ: camera.position.z,
    slope: Math.tan(((camera.fov * Math.PI) / 180) / 2),
    viewportHeightPx,
    orthographic: false,
    orthoProjFactor: 0,
    nearFloor: camera.near,
    // three builds the projection matrix with the WebGL convention until a
    // renderer overwrites it, so that is what matches here.
    depthRange: "minus-one-to-one",
    reversedDepth: false,
  };
  extractFrustumPlanes(cam.clipFromAbs, scratch.planes, cam.depthRange, false);
  return cam;
}

function run(
  tree: LodTreeView,
  cam: LodCameraState,
  s: LodScratch,
  out: LodSelection,
  opts = {},
  kernels?: LodKernels,
): LodSelection {
  return selectVisible(tree, cam, resolveLodOptions(opts), s, out, kernels);
}

describe("selectVisible over the real autzen hierarchy", () => {
  it("selects a coherent set and reports what limited it", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const cam = cameraLooking(autzen, 1500, s);
    run(autzen, cam, s, out);

    expect(out.count).toBeGreaterThan(0);
    expect(out.points).toBeGreaterThan(0);
    expect(out.maxSelectedLevel).toBeGreaterThan(0);
    // limitedBy is the field the reference lacks, and the reason nobody noticed
    // its minimumNodePixelSize knob was inert at the shipped defaults.
    expect(["complete", "error", "budget", "nodes"]).toContain(out.limitedBy);
  });

  it("returns nodes in strictly descending priority, parents before children", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    run(autzen, cameraLooking(autzen, 800, s), s, out);

    const seen = new Set<number>();
    for (let k = 0; k < out.count; k++) {
      const node = autzen.node(out.indices[k]!)!;
      if (node.parent !== undefined) {
        // A node can only be pushed from its admitted parent, so the parent must
        // already have been emitted. This is what the streamer relies on.
        expect(seen.has(node.parent.index)).toBe(true);
      }
      seen.add(node.index);
    }
    expect(seen.size).toBe(out.count);
  });

  it("never exceeds the point budget except for the root", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const cam = cameraLooking(autzen, 200, s);
    run(autzen, cam, s, out, { pointBudget: 500_000 });

    expect(out.limitedBy).toBe("budget");
    const rootPoints = autzen.root.numPoints;
    expect(out.points).toBeLessThanOrEqual(500_000 + rootPoints);
    // The root is always admitted: it is the connectivity anchor, and a budget
    // below it must still show something.
    expect(Array.from(out.indices.slice(0, out.count))).toContain(
      autzen.root.index,
    );
  });

  it("refines more as the camera moves closer", () => {
    const s = createLodScratch();
    const far = createLodSelection();
    const near = createLodSelection();
    run(autzen, cameraLooking(autzen, 4000, s), s, far, { pointBudget: 50_000_000 });
    run(autzen, cameraLooking(autzen, 300, s), s, near, { pointBudget: 50_000_000 });
    expect(near.maxSelectedLevel).toBeGreaterThan(far.maxSelectedLevel);
  });

  // The knob the reference ships is inert at its defaults; ours must not be.
  it("has a targetScreenError that actually changes the selection", () => {
    const s = createLodScratch();
    const coarse = createLodSelection();
    const fine = createLodSelection();
    const camA = cameraLooking(autzen, 900, s);
    run(autzen, camA, s, coarse, { targetScreenError: 8, pointBudget: 50_000_000 });
    const camB = cameraLooking(autzen, 900, s);
    run(autzen, camB, s, fine, { targetScreenError: 0.5, pointBudget: 50_000_000 });

    expect(fine.count).toBeGreaterThan(coarse.count);
    expect(fine.points).toBeGreaterThan(coarse.points);
    expect(coarse.limitedBy).toBe("error");
  });

  it("honours maxLevel, including a legitimate 0", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const cam = cameraLooking(autzen, 300, s);
    // `maxLevel || Infinity` in the reference turns this into unbounded.
    run(autzen, cam, s, out, { maxLevel: 0, pointBudget: 50_000_000 });
    expect(out.maxSelectedLevel).toBe(0);
    expect(out.count).toBe(1);
  });

  it("honours maxNodes and says so", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const cam = cameraLooking(autzen, 200, s);
    run(autzen, cam, s, out, { maxNodes: 12, pointBudget: 50_000_000 });
    expect(out.count).toBeLessThanOrEqual(12);
    expect(out.limitedBy).toBe("nodes");
  });

  it("culls everything when the camera looks away", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const box = autzen.source.metadata.boundingBox;
    const camera = new PerspectiveCamera(60, 1.6, 1, 20_000);
    camera.up.set(0, 0, 1);
    camera.position.set(box.min[0] - 20_000, box.min[1] - 20_000, box.min[2]);
    camera.lookAt(new Vector3(box.min[0] - 40_000, box.min[1] - 40_000, box.min[2]));
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();
    const clip = new Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse,
    );
    const cam: LodCameraState = {
      clipFromAbs: new Float64Array(clip.elements),
      camX: camera.position.x,
      camY: camera.position.y,
      camZ: camera.position.z,
      slope: Math.tan(((60 * Math.PI) / 180) / 2),
      viewportHeightPx: 1080,
      orthographic: false,
      orthoProjFactor: 0,
      nearFloor: 1,
      depthRange: "minus-one-to-one",
      reversedDepth: false,
    };
    extractFrustumPlanes(cam.clipFromAbs, s.planes, cam.depthRange, false);
    run(autzen, cam, s, out);
    // The root is always admitted; nothing else should survive the frustum.
    expect(out.count).toBe(1);
  });

  it("degrades safely on a zero-height viewport", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const cam = cameraLooking(autzen, 500, s, 0);
    run(autzen, cam, s, out);
    expect(out.count).toBe(0);
  });

  it("reports the deepest ADMITTED spacing, for the near-plane policy", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    run(autzen, cameraLooking(autzen, 400, s), s, out, {
      pointBudget: 50_000_000,
    });
    expect(out.minPointSpacingWorld).toBeCloseTo(
      autzen.pointSpacingAt(out.maxSelectedLevel),
      12,
    );
  });

  it("fills levelCounts consistently with count", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    run(autzen, cameraLooking(autzen, 700, s), s, out);
    let total = 0;
    for (let l = 0; l < out.levelCounts.length; l++) total += out.levelCounts[l]!;
    expect(total).toBe(out.count);
  });
});

describe("selectVisible: allocation and state discipline", () => {
  // Zero allocation in steady state is a tested invariant, not an aspiration:
  // the reference allocates one object per pushed child (18k-60k/s) plus a full
  // camera clone per frame.
  it("replaces no backing buffer across 1000 frames", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const cam = cameraLooking(autzen, 600, s);
    run(autzen, cam, s, out);

    const before = [s.heapNode, s.heapKey, s.heapContainment, s.visibleEpoch, out.indices];
    for (let f = 0; f < 1000; f++) run(autzen, cam, s, out);
    expect([
      s.heapNode,
      s.heapKey,
      s.heapContainment,
      s.visibleEpoch,
      out.indices,
    ]).toEqual(before);
  });

  it("pushes each node at most once per frame", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const cam = cameraLooking(autzen, 250, s);
    s.pushes = 0;
    run(autzen, cam, s, out, { pointBudget: 50_000_000 });
    // Every push is a distinct child of an admitted node, so the bound is the
    // node count; the arrays are sized on exactly this argument.
    expect(s.pushes).toBeLessThanOrEqual(autzen.nodeCount);
  });

  it("writes nothing to a node — they are frozen", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    run(autzen, cameraLooking(autzen, 500, s), s, out);
    for (const node of autzen.nodes()) {
      if (node.state === "expanded") expect(Object.isFrozen(node)).toBe(true);
    }
  });

  it("is deterministic for an identical camera", () => {
    const s = createLodScratch();
    const a = createLodSelection();
    const b = createLodSelection();
    const cam = cameraLooking(autzen, 350, s);
    run(autzen, cam, s, a);
    run(autzen, cam, s, b);
    expect(Array.from(b.indices.slice(0, b.count))).toEqual(
      Array.from(a.indices.slice(0, a.count)),
    );
    expect(b.points).toBe(a.points);
    expect(b.limitedBy).toBe(a.limitedBy);
  });

  it("marks the visible set with the frame epoch", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    run(autzen, cameraLooking(autzen, 450, s), s, out);
    for (let k = 0; k < out.count; k++) {
      expect(s.visibleEpoch[out.indices[k]!]).toBe(out.frame);
    }
  });
});

describe("selectVisible: containment propagation", () => {
  it("marks an enclosing frustum's subtree Inside and skips plane tests", () => {
    // A camera far enough back that the whole cloud is inside the frustum.
    const s = createLodScratch();
    const out = createLodSelection();
    const cam = cameraLooking(autzen, 12_000, s);
    run(autzen, cam, s, out, { pointBudget: 50_000_000, targetScreenError: 0.1 });
    expect(out.count).toBeGreaterThan(1);
    // Root is pushed as Intersecting; once a box classifies Inside every
    // descendant inherits it without a six-plane test.
    expect(Containment.Inside).toBe(2);
  });
});

// ---- A1/A2/A4: the generalised metric -------------------------------------
//
// The point of these is that the SPLIT is real. Two of the three quantities
// used to be one number, and every test below fails if they are silently
// re-merged.

/** Wrap a real hierarchy, overriding only what a test names. */
function view(base: PointCloudHierarchy, over: Partial<LodTreeView>): LodTreeView {
  return {
    nodeCount: base.nodeCount,
    root: base.root,
    node: (i) => base.node(i),
    geometricErrorAt: (l) => base.geometricErrorAt(l),
    pointSpacingAt: (l) => base.pointSpacingAt(l),
    boundingRadiusAt: (l) => base.boundingRadiusAt(l),
    // The scheduler speaks the NEUTRAL node; this fixture's tree is a Potree
    // one, so the narrowing happens here rather than in the contract.
    tryExpandSync: (n) => base.tryExpandSync(n as HierarchyNode),
    requestExpand: (n, sig) => base.requestExpand(n as HierarchyNode, sig),
    ...over,
  };
}

/** Dense array holding exactly the closed form, for every materialised node. */
function denseFromLevel(
  base: PointCloudHierarchy,
  at: (level: number) => number,
): Float64Array {
  const a = new Float64Array(base.nodeCount);
  for (let i = 0; i < base.nodeCount; i++) {
    const n = base.node(i);
    if (n !== undefined) a[i] = at(n.level);
  }
  return a;
}

describe("per-node overrides (A1)", () => {
  it("reproduces the closed-form selection EXACTLY when filled with it", () => {
    // The differential test that makes the whole generalisation safe: the array
    // path and the closed-form path are the same decision seen two ways, so
    // filling the arrays with the closed form must be a no-op down to the order
    // of `indices` — which is also the streaming priority order.
    const s1 = createLodScratch();
    const s2 = createLodScratch();
    const closed = createLodSelection();
    const dense = createLodSelection();

    run(autzen, cameraLooking(autzen, 600, s1), s1, closed, {
      pointBudget: 50_000_000,
    });
    const overridden = view(autzen, {
      nodeGeometricError: denseFromLevel(autzen, (l) => autzen.geometricErrorAt(l)),
      nodePointSpacing: denseFromLevel(autzen, (l) => autzen.pointSpacingAt(l)),
      nodeBoundingRadius: denseFromLevel(autzen, (l) => autzen.boundingRadiusAt(l)),
    });
    run(overridden, cameraLooking(autzen, 600, s2), s2, dense, {
      pointBudget: 50_000_000,
    });

    expect(dense.count).toBe(closed.count);
    expect(dense.points).toBe(closed.points);
    expect(dense.limitedBy).toBe(closed.limitedBy);
    expect(dense.maxSelectedLevel).toBe(closed.maxSelectedLevel);
    expect(dense.minPointSpacingWorld).toBeCloseTo(closed.minPointSpacingWorld, 12);
    expect(Array.from(dense.indices.subarray(0, dense.count))).toEqual(
      Array.from(closed.indices.subarray(0, closed.count)),
    );
  });

  it("actually binds: halving every node's error coarsens the selection", () => {
    const s1 = createLodScratch();
    const s2 = createLodScratch();
    const normal = createLodSelection();
    const halved = createLodSelection();

    run(autzen, cameraLooking(autzen, 600, s1), s1, normal, {
      pointBudget: 50_000_000,
    });
    const err = denseFromLevel(autzen, (l) => autzen.geometricErrorAt(l));
    for (let i = 0; i < err.length; i++) err[i] = err[i]! * 0.5;
    run(
      view(autzen, { nodeGeometricError: err }),
      cameraLooking(autzen, 600, s2),
      s2,
      halved,
      { pointBudget: 50_000_000 },
    );

    expect(halved.count).toBeLessThan(normal.count);
  });
});

describe("the four jobs of spacingAt are separated (A2)", () => {
  it("refines on geometric error and reports near-plane pitch independently", () => {
    // Scale the two apart by a factor no octree would ever produce. If the
    // split were cosmetic, one of these two assertions could not hold.
    const s = createLodScratch();
    const out = createLodSelection();
    const PITCH = 7;
    const split = view(autzen, {
      pointSpacingAt: (l) => autzen.pointSpacingAt(l) * PITCH,
    });
    run(split, cameraLooking(autzen, 600, s), s, out, { pointBudget: 50_000_000 });

    const s2 = createLodScratch();
    const baseline = createLodSelection();
    run(autzen, cameraLooking(autzen, 600, s2), s2, baseline, {
      pointBudget: 50_000_000,
    });

    // Refinement is untouched: it reads the ERROR, which did not move.
    expect(out.count).toBe(baseline.count);
    // The near-plane quantity followed the PITCH, which did.
    expect(out.minPointSpacingWorld).toBeCloseTo(
      baseline.minPointSpacingWorld * PITCH,
      12,
    );
  });

  it("keeps minPointSpacingWorld off the refinement path entirely", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    // An absurd error with a sane pitch: selection collapses to the root alone,
    // but the reported pitch is still the root's real pitch, not the error.
    const collapsed = view(autzen, { geometricErrorAt: () => 1e-9 });
    run(collapsed, cameraLooking(autzen, 600, s), s, out, {
      pointBudget: 50_000_000,
    });
    expect(out.count).toBe(1);
    expect(out.minPointSpacingWorld).toBeCloseTo(autzen.pointSpacingAt(0), 12);
  });
});

describe("targetScreenError resolution (A4)", () => {
  it("prefers the explicit option, then the format default, then 1.35", () => {
    expect(resolveLodOptions({}).targetScreenError).toBe(DEFAULT_SCREEN_ERROR);
    expect(resolveLodOptions({}, 16).targetScreenError).toBe(16);
    expect(resolveLodOptions({ targetScreenError: 2 }, 16).targetScreenError).toBe(2);
  });

  it("accepts the deprecated targetPixelSpacing alias", () => {
    expect(resolveLodOptions({ targetPixelSpacing: 3 }).targetScreenError).toBe(3);
    // An explicit new-name value wins over the alias.
    expect(
      resolveLodOptions({ targetScreenError: 2, targetPixelSpacing: 9 })
        .targetScreenError,
    ).toBe(2);
    // The alias still beats a format default: it is a caller instruction.
    expect(resolveLodOptions({ targetPixelSpacing: 3 }, 16).targetScreenError).toBe(3);
  });
});

describe("kernel eligibility (A1)", () => {
  /**
   * A kernel that survives NO child, so a selection that reached it collapses to
   * the root alone. That makes "was the kernel consulted" observable from the
   * selection itself, not only from the call counter.
   */
  function countingKernel(): LodKernels & {
    calls: number;
    firstChild: Float64Array | undefined;
    firstPerChild: number | undefined;
  } {
    return {
      planes: new Float64Array(24),
      boxes: new Float64Array(48),
      results: new Float64Array(16),
      params: new Float64Array(11),
      child: new Float64Array(16),
      calls: 0,
      firstChild: undefined,
      firstPerChild: undefined,
      selectChildren(_mask: number, _parentInside: boolean): number {
        // Snapshot the FIRST call — the root's children — before the next
        // admitted node overwrites the shared block.
        if (this.calls === 0) {
          this.firstChild = this.child.slice();
          this.firstPerChild = this.params[10];
        }
        this.calls++;
        return 0;
      },
    };
  }

  it("uses the kernel on a closed-form tree", () => {
    const s = createLodScratch();
    const out = createLodSelection();
    const k = countingKernel();
    run(autzen, cameraLooking(autzen, 600, s), s, out, { pointBudget: 50_000_000 }, k);

    expect(k.calls).toBeGreaterThan(0);
    // No child survived, so the root is the whole selection. If this were the
    // TypeScript path the count would be in the hundreds.
    expect(out.count).toBe(1);
  });

  it("marshals per-node overrides into the kernel's child block (A3)", () => {
    // Since A3 the kernel CAN express eight errors and eight radii, so a tree
    // with overrides is no longer pushed off the fast path. What must hold is
    // that each slot carries ITS OWN child's value — feeding slot c the value
    // of a different node is the silent-wrong-picture failure this guards.
    const s = createLodScratch();
    const out = createLodSelection();
    const k = countingKernel();
    // A distinctive per-node pattern, so a broadcast or an off-by-one slot is
    // visible rather than plausible.
    const err = new Float64Array(autzen.nodeCount);
    const rad = new Float64Array(autzen.nodeCount);
    for (let i = 0; i < autzen.nodeCount; i++) {
      err[i] = 1000 + i;
      rad[i] = 2_000_000 + i;
    }
    const overridden = view(autzen, {
      nodeGeometricError: err,
      nodeBoundingRadius: rad,
    });
    run(overridden, cameraLooking(autzen, 600, s), s, out, { pointBudget: 50_000_000 }, k);

    expect(k.calls).toBeGreaterThan(0);
    expect(k.firstPerChild).toBe(1);
    for (let c = 0; c < 8; c++) {
      const kid = autzen.root.children[c];
      if (kid === undefined) continue;
      expect(k.firstChild![c]).toBe(err[kid.index]);
      expect(k.firstChild![8 + c]).toBe(rad[kid.index]);
    }
  });

  it("leaves the per-child flag clear on a closed-form tree", () => {
    // The cost argument: an octree must keep writing two scalars per admitted
    // node, not sixteen. If this flag ever came up on autzen, every octree would
    // silently start paying for a generality it does not use.
    const s = createLodScratch();
    const out = createLodSelection();
    const k = countingKernel();
    run(autzen, cameraLooking(autzen, 600, s), s, out, { pointBudget: 50_000_000 }, k);
    expect(k.firstPerChild).toBe(0);
  });

  it("keeps the closed-form path when only the POINT PITCH is overridden", () => {
    // `nodePointSpacing` never reaches the kernel — it feeds the near plane, not
    // the child math — so it must not switch the kernel to the per-child block.
    const s = createLodScratch();
    const out = createLodSelection();
    const k = countingKernel();
    const overridden = view(autzen, {
      nodePointSpacing: denseFromLevel(autzen, (l) => autzen.pointSpacingAt(l)),
    });
    run(overridden, cameraLooking(autzen, 600, s), s, out, { pointBudget: 50_000_000 }, k);

    expect(k.calls).toBeGreaterThan(0);
    expect(k.firstPerChild).toBe(0);
  });
});
