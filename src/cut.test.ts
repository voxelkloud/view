import { readFileSync } from "node:fs";
import { createHierarchy, parsePointCloudSource } from "@voxelkloud/format-potree";
import type { PointCloudHierarchy } from "@voxelkloud/format-potree";
import type { PointCloudNode } from "@voxelkloud/core";
import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { beforeAll, describe, expect, it } from "vitest";
import { CUT_WIDTH, MAX_CUT_DEPTH, OctreeCut, localDepth } from "./cut.js";
import { extractFrustumPlanes } from "./lod/frustum.js";
import {
  createLodScratch,
  createLodSelection,
  resolveLodOptions,
  selectVisible,
} from "./lod/select.js";
import type { LodCameraState, LodScratch } from "./lod/select.js";

const LOADER_FIXTURES = new URL(
  "./__fixtures__/",
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

function cameraLooking(
  tree: PointCloudHierarchy,
  distance: number,
  scratch: LodScratch,
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
    viewportHeightPx: 1080,
    orthographic: false,
    orthoProjFactor: 0,
    nearFloor: camera.near,
    depthRange: "minus-one-to-one",
    reversedDepth: false,
  };
  extractFrustumPlanes(cam.clipFromAbs, scratch.planes, cam.depthRange, false);
  return cam;
}

/**
 * GROUND TRUTH. Descends the real hierarchy in ABSOLUTE float64 using each
 * node's own stored box, with no halving arithmetic and no packed offsets.
 *
 * Deliberately a different program from the one under test: the encoding could
 * be wrong in exactly the way the walk is wrong and a self-consistent pair
 * would still agree.
 */
function deepestSelected(
  root: PointCloudNode,
  epoch: Int32Array,
  frame: number,
  p: readonly [number, number, number],
): number {
  let node = root;
  let depth = 0;
  for (;;) {
    let next: PointCloudNode | undefined;
    for (let c = 0; c < 8; c++) {
      const child = node.children[c];
      if (child === undefined) continue;
      if (epoch[child.index] !== frame) continue;
      if (
        p[0] >= child.minX && p[0] < child.maxX &&
        p[1] >= child.minY && p[1] < child.maxY &&
        p[2] >= child.minZ && p[2] < child.maxZ
      ) {
        next = child;
        break;
      }
    }
    if (next === undefined) return depth;
    node = next;
    depth++;
  }
}

/**
 * A point INSIDE a node and off every split plane, at any depth.
 *
 * Never the box centre. A node's centre is the exact corner where all eight of
 * its children meet, and the two descents reach it by different arithmetic —
 * this one halves a box `depth` times, the tree's own boxes come from
 * `makeChildNode` recomputing `max - min` at each level — so they land an ULP
 * apart and pick different octants. The fractions below are non-dyadic, so no
 * amount of halving ever puts them on a boundary.
 */
function probe(n: PointCloudNode): [number, number, number] {
  return [
    n.minX + 0.3717 * (n.maxX - n.minX),
    n.minY + 0.6291 * (n.maxY - n.minY),
    n.minZ + 0.2143 * (n.maxZ - n.minZ),
  ];
}

/** The steady state the walk is designed around: everything selected has landed. */
function allResident(sel: { indices: Int32Array; count: number }): Set<number> {
  const r = new Set<number>();
  for (let k = 0; k < sel.count; k++) r.add(sel.indices[k]!);
  return r;
}

/** Deterministic, so a failure is reproducible. */
function lcg(seed: number): () => number {
  let x = seed >>> 0;
  return () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    return x / 4294967296;
  };
}

describe("OctreeCut over the real autzen selection", () => {
  it("agrees with an independent descent at every selected node's centre", () => {
    const s = createLodScratch();
    const sel = createLodSelection();
    const cam = cameraLooking(autzen, 300, s);
    selectVisible(autzen, cam, resolveLodOptions({ targetScreenError: 0.4 }), s, sel);
    expect(sel.count).toBeGreaterThan(100);

    const cut = new OctreeCut(4096);
    cut.build(autzen.root, (i) => autzen.node(i), s.visibleEpoch, sel.frame, 4096, allResident(sel));
    expect(cut.entryCount).toBe(sel.count);

    const root = autzen.root;
    const rootMin = [root.minX, root.minY, root.minZ] as const;
    const rootSize = [
      root.maxX - root.minX,
      root.maxY - root.minY,
      root.maxZ - root.minZ,
    ] as const;

    for (let k = 0; k < sel.count; k++) {
      const n = autzen.node(sel.indices[k]!)!;
      const c = probe(n);
      const got = localDepth(cut.bytes, c, rootMin, rootSize);
      expect(got).toBe(deepestSelected(root, s.visibleEpoch, sel.frame, c));
      // The invariant the whole feature rests on: a point can never be told to
      // draw itself COARSER than the node it came from.
      expect(got).toBeGreaterThanOrEqual(n.level);
    }
  });

  // The one case the walk is allowed to get wrong, stated so a future reader
  // does not take it for a bug. A point exactly on a split plane is ambiguous
  // in the tree too — `makeChildNode` builds the low sibling's max by
  // subtraction and the high sibling's min by addition, and in float64 those
  // are not always the same number — so the two descents may pick different
  // octants there, in EITHER direction. The shader runs the same walk in
  // float32, which widens the ambiguous band to about half a millimetre on
  // autzen's 4655 m root.
  //
  // Both directions are bounded and neither is visible. One level too shallow
  // is a splat twice too wide, and the shader clamps that away by never sizing
  // below the drawn slab's own level — the worst case there is the size drawn
  // TODAY. One level too deep is a splat half as wide, i.e. a sub-pixel
  // pinhole, on the ~2.5e-4 of points that land within a millimetre of a plane.
  it("may disagree by one level, and only on a split plane", () => {
    const s = createLodScratch();
    const sel = createLodSelection();
    const cam = cameraLooking(autzen, 300, s);
    selectVisible(autzen, cam, resolveLodOptions({ targetScreenError: 0.4 }), s, sel);

    const cut = new OctreeCut(4096);
    cut.build(autzen.root, (i) => autzen.node(i), s.visibleEpoch, sel.frame, 4096, allResident(sel));
    const root = autzen.root;
    const rootMin = [root.minX, root.minY, root.minZ] as const;
    const rootSize = [
      root.maxX - root.minX,
      root.maxY - root.minY,
      root.maxZ - root.minZ,
    ] as const;

    let disagreed = 0;
    for (let k = 0; k < sel.count; k++) {
      const n = autzen.node(sel.indices[k]!)!;
      const centre: [number, number, number] = [
        (n.minX + n.maxX) / 2,
        (n.minY + n.maxY) / 2,
        (n.minZ + n.maxZ) / 2,
      ];
      const got = localDepth(cut.bytes, centre, rootMin, rootSize);
      const want = deepestSelected(root, s.visibleEpoch, sel.frame, centre);
      if (got !== want) disagreed++;
      expect(Math.abs(want - got)).toBeLessThanOrEqual(1);
    }
    // Rare, and confined to the degenerate probe. If this ever reaches a large
    // fraction of the selection, the arithmetic has drifted and the clamp is
    // carrying the feature rather than backstopping it.
    expect(disagreed / sel.count).toBeLessThan(0.05);
  });

  it("agrees at random points across the root box", () => {
    const s = createLodScratch();
    const sel = createLodSelection();
    const cam = cameraLooking(autzen, 300, s);
    selectVisible(autzen, cam, resolveLodOptions({ targetScreenError: 0.4 }), s, sel);

    const cut = new OctreeCut(4096);
    cut.build(autzen.root, (i) => autzen.node(i), s.visibleEpoch, sel.frame, 4096, allResident(sel));

    const root = autzen.root;
    const rootMin = [root.minX, root.minY, root.minZ] as const;
    const rootSize = [
      root.maxX - root.minX,
      root.maxY - root.minY,
      root.maxZ - root.minZ,
    ] as const;
    const rnd = lcg(0xc0ffee);

    let deeperThanRoot = 0;
    for (let k = 0; k < 20_000; k++) {
      const p: [number, number, number] = [
        rootMin[0] + rnd() * rootSize[0],
        rootMin[1] + rnd() * rootSize[1],
        rootMin[2] + rnd() * rootSize[2],
      ];
      const got = localDepth(cut.bytes, p, rootMin, rootSize);
      expect(got).toBe(deepestSelected(root, s.visibleEpoch, sel.frame, p));
      if (got > 0) deeperThanRoot++;
    }
    // Guards the guard: a cut that terminated immediately everywhere would
    // agree with a broken oracle on all 20k samples and prove nothing.
    expect(deeperThanRoot).toBeGreaterThan(1000);
  });

  it("addresses past the first texture row, where the offset needs 2 bytes", () => {
    const s = createLodScratch();
    const sel = createLodSelection();
    const cam = cameraLooking(autzen, 120, s);
    selectVisible(
      autzen,
      cam,
      resolveLodOptions({ targetScreenError: 0.1, pointBudget: 50_000_000 }),
      s,
      sel,
    );
    // Only meaningful if the cut is big enough that a child offset overflows a
    // single byte and spills into `b`.
    expect(sel.count).toBeGreaterThan(256);

    const cut = new OctreeCut(4096);
    cut.build(autzen.root, (i) => autzen.node(i), s.visibleEpoch, sel.frame, 4096, allResident(sel));

    const root = autzen.root;
    const rootMin = [root.minX, root.minY, root.minZ] as const;
    const rootSize = [
      root.maxX - root.minX,
      root.maxY - root.minY,
      root.maxZ - root.minZ,
    ] as const;
    for (let k = 0; k < sel.count; k++) {
      const n = autzen.node(sel.indices[k]!)!;
      const c = probe(n);
      expect(localDepth(cut.bytes, c, rootMin, rootSize)).toBe(
        deepestSelected(root, s.visibleEpoch, sel.frame, c),
      );
    }
  });

  it("stops at what has landed, not at what was selected", () => {
    const s = createLodScratch();
    const sel = createLodSelection();
    const cam = cameraLooking(autzen, 300, s);
    selectVisible(autzen, cam, resolveLodOptions({ targetScreenError: 0.4 }), s, sel);

    const root = autzen.root;
    const rootMin = [root.minX, root.minY, root.minZ] as const;
    const rootSize = [
      root.maxX - root.minX,
      root.maxY - root.minY,
      root.maxZ - root.minZ,
    ] as const;

    // Mid-load: everything down to level 3 has arrived, the rest is in flight.
    const landed = new Set<number>();
    for (let k = 0; k < sel.count; k++) {
      const n = autzen.node(sel.indices[k]!)!;
      if (n.level <= 3) landed.add(n.index);
    }

    const full = new OctreeCut(4096);
    full.build(autzen.root, (i) => autzen.node(i), s.visibleEpoch, sel.frame, 4096, allResident(sel));
    const partial = new OctreeCut(4096);
    partial.build(autzen.root, (i) => autzen.node(i), s.visibleEpoch, sel.frame, 4096, landed);

    expect(partial.entryCount).toBe(landed.size);
    expect(partial.entryCount).toBeLessThan(full.entryCount);

    let deeper = 0;
    for (let k = 0; k < sel.count; k++) {
      const n = autzen.node(sel.indices[k]!)!;
      const c = probe(n);
      const d = localDepth(partial.bytes, c, rootMin, rootSize);
      // The whole point: a splat is never sized for data that has not landed.
      expect(d).toBeLessThanOrEqual(3);
      if (localDepth(full.bytes, c, rootMin, rootSize) > d) deeper++;
    }
    // The two cuts must genuinely differ, or the assertion above is vacuous.
    expect(deeper).toBeGreaterThan(0);
  });

  it("emits one terminating entry when nothing was selected", () => {
    const s = createLodScratch();
    const cut = new OctreeCut(64);
    // Frame 999 was never stamped, so no node is a member.
    cut.build(autzen.root, (i) => autzen.node(i), s.visibleEpoch, 999, 64, new Set());
    expect(cut.entryCount).toBe(1);
    expect(cut.bytes[0]).toBe(0);
    expect(localDepth(cut.bytes, [0, 0, 0], [0, 0, 0], [1, 1, 1])).toBe(0);
  });

  it("keeps the texture a whole number of CUT_WIDTH rows", () => {
    const cut = new OctreeCut(4096);
    expect(cut.map.image.width).toBe(CUT_WIDTH);
    expect(cut.map.image.height).toBe(4096 / CUT_WIDTH);
    expect(MAX_CUT_DEPTH).toBeGreaterThan(6);
  });
});
