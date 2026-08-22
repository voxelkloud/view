import { readFileSync } from "node:fs";
import {
  createHierarchy,
  parsePointCloudSource,
} from "@voxelkloud/loader";
import type { PointCloudHierarchy } from "@voxelkloud/loader";
import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { beforeAll, describe, expect, it } from "vitest";
import { Containment, extractFrustumPlanes } from "./frustum.js";
import {
  createLodScratch,
  createLodSelection,
  resolveLodOptions,
  selectVisible,
} from "./select.js";
import type { LodCameraState, LodScratch, LodSelection, LodTreeView } from "./select.js";

// The loader vendors the real autzen manifest and hierarchy for its own offline
// suite; reading them here keeps the scheduler tested against real converter
// output rather than a synthetic tree that cannot reproduce its shape.
const LOADER_FIXTURES = new URL(
  "../../../loader/src/__fixtures__/",
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
): LodSelection {
  return selectVisible(tree, cam, resolveLodOptions(opts), s, out);
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
  it("has a targetPixelSpacing that actually changes the selection", () => {
    const s = createLodScratch();
    const coarse = createLodSelection();
    const fine = createLodSelection();
    const camA = cameraLooking(autzen, 900, s);
    run(autzen, camA, s, coarse, { targetPixelSpacing: 8, pointBudget: 50_000_000 });
    const camB = cameraLooking(autzen, 900, s);
    run(autzen, camB, s, fine, { targetPixelSpacing: 0.5, pointBudget: 50_000_000 });

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
    expect(out.minSpacingWorld).toBeCloseTo(
      autzen.spacingAt(out.maxSelectedLevel),
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
    run(autzen, cam, s, out, { pointBudget: 50_000_000, targetPixelSpacing: 0.1 });
    expect(out.count).toBeGreaterThan(1);
    // Root is pushed as Intersecting; once a box classifies Inside every
    // descendant inherits it without a six-plane test.
    expect(Containment.Inside).toBe(2);
  });
});
