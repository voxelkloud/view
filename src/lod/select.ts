import type { HierarchyNode } from "@voxelkloud/loader";
import { Containment, classifyAabb } from "./frustum.js";
import type { DepthRange } from "./frustum.js";
import { heapPop, heapPush } from "./heap.js";

const MAX_LEVELS = 33;

/**
 * The slice of `PointCloudHierarchy` the selector needs.
 *
 * `PointCloudHierarchy` satisfies this STRUCTURALLY — no adapter, no extraction
 * step. Declaring against the narrow interface is what lets the same tests run
 * over hand-built synthetic trees AND the real vendored autzen hierarchy.
 */
export interface LodTreeView {
  readonly nodeCount: number;
  readonly root: HierarchyNode;
  node(index: number): HierarchyNode | undefined;
  /** Per-level constant: `radiusAt(L) === radiusAt(0) / 2 ** L`, exactly. */
  radiusAt(level: number): number;
  /** `metadata.spacing / 2 ** level`. */
  spacingAt(level: number): number;
  /** SYNCHRONOUS, never throws, safe from a render loop. */
  tryExpandSync(node: HierarchyNode): boolean;
  /** Fire-and-forget. A documented no-op inside a backoff window. */
  requestExpand(node: HierarchyNode, signal?: AbortSignal): void;
}

/**
 * The camera reduced to what the scheduler reads.
 *
 * EVERY spatial quantity is in ABSOLUTE CRS and float64, matching
 * `HierarchyNode.minX..maxZ`. Plain numbers plus one `Float64Array` — no three
 * types anywhere, which is what keeps this module three-free and therefore
 * importable under SSR, in a worker, and in vitest.
 */
export interface LodCameraState {
  /**
   * `proj * viewInverse * root.matrixWorld * translate(-cloudOrigin)`,
   * column-major, 16 elements, computed in float64 on the CPU.
   */
  clipFromAbs: Float64Array;
  /** Camera eye in ABSOLUTE CRS. The float64 ULP at 635,577 m is 1.4e-10 m. */
  camX: number;
  camY: number;
  camZ: number;
  /** `tan(fov / 2)`, the VERTICAL half-angle. Must be > 0. Perspective only. */
  slope: number;
  /** Framebuffer height in DEVICE pixels (`cssHeight * dpr`). */
  viewportHeightPx: number;
  orthographic: boolean;
  /** `viewportHeightPx / orthoHeightWorld`. Constant in distance. */
  orthoProjFactor: number;
  /**
   * Lower clamp on `distanceToCentre - radius`. Replaces the reference's
   * `Number.MAX_VALUE` sentinel, which makes every node containing the camera
   * tie at exactly MAX_VALUE and destroys coarse-first ordering along the very
   * descent the camera is inside. Set it to `camera.near`.
   */
  nearFloor: number;
  depthRange: DepthRange;
  reversedDepth: boolean;
}

export interface LodOptions {
  /**
   * THE PRIMARY CONTROL. Target inter-point spacing on screen, in DEVICE pixels.
   * Refinement into a child stops when its projected point spacing would fall
   * below this. Default 1.35.
   */
  readonly targetPixelSpacing?: number;
  /**
   * Points DRAWN per frame. Default 3,000,000 (48 MB of GPU vertex data at
   * 16 B/pt). Charged from `node.numPoints`, which is THIS NODE'S OWN LAYER and
   * never a subtree total — treating it as cumulative under-refines by the
   * measured own-vs-subtree ratio at the frontier, 1.96 at level 5.
   */
  readonly pointBudget?: number;
  /** Hard cap on selected nodes. Default 4096. Sizes `selection.indices`. */
  readonly maxNodes?: number;
  /**
   * Default `Infinity`. Resolved with `??`, NEVER `||` — the reference's
   * `maxLevel || Infinity` turns a legitimate 0 into unbounded.
   */
  readonly maxLevel?: number;
  /**
   * Budget-rejected nodes to walk past before abandoning the heap. Default 32.
   * The reference `break`s on the first node that does not fit, abandoning a
   * measured 191 already-allocated heap entries on autzen at 80 m and
   * under-spending the budget exactly at the refinement frontier.
   */
  readonly maxBudgetSkips?: number;
}

export type ResolvedLodOptions = Required<LodOptions>;

/**
 * The optional wasm kernel surface.
 *
 * Structural, and declared HERE rather than imported, so `lod/` keeps its
 * zero-dependency import graph: `@voxelkloud/wasm-core` satisfies this shape
 * without this module ever naming the package. A caller that has no kernels
 * passes nothing and gets the TypeScript path, which stays the oracle the
 * kernels are differential-tested against.
 */
export interface LodKernels {
  readonly planes: Float64Array;
  readonly boxes: Float64Array;
  readonly results: Float64Array;
  readonly params: Float64Array;
  selectChildren(mask: number, parentInside: boolean): number;
}

export function resolveLodOptions(o: LodOptions = {}): ResolvedLodOptions {
  return {
    targetPixelSpacing: o.targetPixelSpacing ?? 1.35,
    pointBudget: o.pointBudget ?? 3_000_000,
    maxNodes: o.maxNodes ?? 4096,
    maxLevel: o.maxLevel ?? Number.POSITIVE_INFINITY,
    maxBudgetSkips: o.maxBudgetSkips ?? 32,
  };
}

/**
 * Reused across frames, grown by DOUBLING only when `tree.nodeCount` grows.
 * `selectVisible` performs ZERO allocation in steady state.
 */
export interface LodScratch {
  readonly planes: Float64Array;
  heapNode: Int32Array;
  heapKey: Float64Array;
  heapContainment: Uint8Array;
  /** `visibleEpoch[i] === frame` IS the membership test. */
  visibleEpoch: Int32Array;
  popNode: number;
  popKey: number;
  popContainment: Containment;
  frame: number;
  capacity: number;
  /** Test hook, asserted against an analytic bound. */
  pushes: number;
}

export function createLodScratch(capacity = 0): LodScratch {
  return {
    planes: new Float64Array(24),
    heapNode: new Int32Array(capacity),
    heapKey: new Float64Array(capacity),
    heapContainment: new Uint8Array(capacity),
    visibleEpoch: new Int32Array(capacity),
    popNode: -1,
    popKey: 0,
    popContainment: Containment.Outside,
    frame: 0,
    capacity,
    pushes: 0,
  };
}

/** Overwritten in place every frame; the arrays are never reallocated. */
export interface LodSelection {
  /**
   * `count` valid entries, in POP ORDER — strictly descending screen spacing.
   * Load-bearing twice: it is the streaming priority order, and it guarantees a
   * node's parent appears before it.
   */
  indices: Int32Array;
  count: number;
  /** Sum of own-layer `numPoints`. Within budget except that the ROOT is
   *  always admitted. */
  points: number;
  frame: number;
  /**
   * Which constraint stopped refinement. `"error"` is the good case — the target
   * spacing was met. This field exists because the reference has none, which is
   * why nobody noticed `minimumNodePixelSize` is INERT at its shipped defaults:
   * 150 px and 50 px give byte-identical autzen selections at 1500/300/80 m.
   */
  limitedBy: "complete" | "error" | "budget" | "nodes";
  /** Min `spacingAt(level)` over ADMITTED nodes only. Drives near/far. */
  minSpacingWorld: number;
  maxSelectedLevel: number;
  /** Admitted count per level. Free, and what makes an LOD bug legible. */
  readonly levelCounts: Int32Array;
  /** Nodes whose children are unknown and could not be expanded synchronously. */
  needsExpand: Int32Array;
  needsExpandCount: number;
}

export function createLodSelection(maxNodes = 4096): LodSelection {
  return {
    indices: new Int32Array(maxNodes),
    count: 0,
    points: 0,
    frame: 0,
    limitedBy: "complete",
    minSpacingWorld: 0,
    maxSelectedLevel: 0,
    levelCounts: new Int32Array(MAX_LEVELS),
    needsExpand: new Int32Array(maxNodes),
    needsExpandCount: 0,
  };
}

export function ensureLodCapacity(
  s: LodScratch,
  out: LodSelection,
  nodeCount: number,
  maxNodes: number,
): void {
  if (nodeCount > s.capacity) {
    let cap = Math.max(s.capacity, 16);
    while (cap < nodeCount) cap *= 2;
    s.heapNode = new Int32Array(cap);
    s.heapKey = new Float64Array(cap);
    s.heapContainment = new Uint8Array(cap);
    const epoch = new Int32Array(cap);
    epoch.set(s.visibleEpoch);
    s.visibleEpoch = epoch;
    s.capacity = cap;
  }
  if (out.indices.length < maxNodes) {
    out.indices = new Int32Array(maxNodes);
    out.needsExpand = new Int32Array(maxNodes);
  }
}

/**
 * Best-first descent over the octree, writing the visible set into `out`.
 *
 * Pure apart from `tree.tryExpandSync`, which Task 3 guarantees is synchronous
 * and never throws. NOTHING is written to a node — every per-frame bit lives in
 * an index-keyed side array, which is a requirement rather than an optimisation,
 * because Task 3 freezes expanded nodes.
 */
export function selectVisible(
  tree: LodTreeView,
  cam: LodCameraState,
  opts: ResolvedLodOptions,
  s: LodScratch,
  out: LodSelection,
  kernels?: LodKernels,
): LodSelection {
  ensureLodCapacity(s, out, tree.nodeCount, opts.maxNodes);
  const frame = ++s.frame;
  let heap = 0;
  let n = 0;
  let pts = 0;
  let skips = 0;
  let ne = 0;
  let limitedBy: LodSelection["limitedBy"] = "complete";
  let minSpacing = Infinity;
  let deepest = 0;
  out.levelCounts.fill(0);

  if (!(cam.slope > 0) || !(cam.viewportHeightPx > 0)) {
    out.count = 0;
    out.points = 0;
    out.frame = frame;
    out.limitedBy = "complete";
    out.needsExpandCount = 0;
    out.minSpacingWorld = tree.spacingAt(0);
    out.maxSelectedLevel = 0;
    return out;
  }

  // Everything except the per-level radius and spacing is constant for the
  // whole frame, so it is written into the kernel block ONCE rather than on
  // every admitted node.
  if (kernels !== undefined) {
    const p = kernels.params;
    p[0] = cam.camX;
    p[1] = cam.camY;
    p[2] = cam.camZ;
    p[5] = cam.nearFloor;
    p[6] = cam.slope;
    p[7] = cam.viewportHeightPx;
    p[8] = cam.orthoProjFactor;
    p[9] = cam.orthographic ? 1 : 0;
  }

  const rootIndex = tree.root.index;
  heap = heapPush(s, heap, rootIndex, Infinity, Containment.Intersecting);

  while (heap > 0) {
    if (n >= opts.maxNodes) {
      limitedBy = "nodes";
      break;
    }
    heap = heapPop(s, heap);
    const i = s.popNode;
    const containment = s.popContainment;
    let node = tree.node(i);
    if (node === undefined) continue;

    // BUDGET: skip and continue, never break. The ROOT is exempt — it is the
    // connectivity anchor and 0.1% of the cloud, and a budget below it must
    // still show something.
    if (i !== rootIndex && pts + node.numPoints > opts.pointBudget) {
      limitedBy = "budget";
      if (++skips > opts.maxBudgetSkips) break;
      continue;
    }

    pts += node.numPoints;
    s.visibleEpoch[i] = frame;
    out.indices[n++] = i;
    const lvl = node.level;
    out.levelCounts[lvl] = (out.levelCounts[lvl] ?? 0) + 1;
    if (lvl > deepest) deepest = lvl;
    const sp = tree.spacingAt(lvl);
    if (sp < minSpacing) minSpacing = sp;

    if (node.childMask === 0) continue; // 80.5% of autzen nodes
    if (node.childMask === undefined) {
      if (!tree.tryExpandSync(node)) {
        out.needsExpand[ne++] = i;
        continue;
      }
      // RE-FETCH: expansion freezes the node and the index is the stable key,
      // not the object identity.
      const refreshed = tree.node(i);
      if (refreshed === undefined) continue;
      node = refreshed;
      if (node.childMask === undefined || node.childMask === 0) continue;
    }
    if (lvl + 1 > opts.maxLevel) continue;

    const rChild = tree.radiusAt(lvl + 1);
    const spChild = tree.spacingAt(lvl + 1);

    if (kernels !== undefined) {
      // ONE boundary crossing per admitted node, not per child: the arithmetic
      // per child is comparable to the cost of a crossing, so per-child calls
      // would be a wash.
      const { boxes, results, params } = kernels;
      let mask = 0;
      for (let c = 0; c < 8; c++) {
        const child = node.children[c];
        if (child === undefined) continue;
        const o = c * 6;
        boxes[o] = child.minX;
        boxes[o + 1] = child.minY;
        boxes[o + 2] = child.minZ;
        boxes[o + 3] = child.maxX;
        boxes[o + 4] = child.maxY;
        boxes[o + 5] = child.maxZ;
        mask |= 1 << c;
      }
      params[3] = rChild;
      params[4] = spChild;

      const survived = kernels.selectChildren(
        mask,
        containment === Containment.Inside,
      );
      for (let c = 0; c < 8; c++) {
        if (((survived >> c) & 1) === 0) continue;
        const child = node.children[c]!;
        const key = results[c * 2 + 1]!;
        if (key < opts.targetPixelSpacing) {
          if (limitedBy === "complete") limitedBy = "error";
          continue;
        }
        heap = heapPush(s, heap, child.index, key, results[c * 2] as Containment);
        s.pushes++;
      }
      continue;
    }

    for (let c = 0; c < 8; c++) {
      const child = node.children[c];
      if (child === undefined) continue;

      // CULL AT PUSH — the reference culls at pop, so every off-screen child is
      // allocated, heaped, popped and only then discarded. Containment
      // propagates: an Inside parent means every descendant is Inside, so the
      // six-plane test is skipped for the whole subtree.
      const cc =
        containment === Containment.Inside
          ? Containment.Inside
          : classifyAabb(
              s.planes,
              child.minX,
              child.minY,
              child.minZ,
              child.maxX,
              child.maxY,
              child.maxZ,
            );
      if (cc === Containment.Outside) continue;

      const dx = cam.camX - (child.minX + child.maxX) * 0.5;
      const dy = cam.camY - (child.minY + child.maxY) * 0.5;
      const dz = cam.camZ - (child.minZ + child.maxZ) * 0.5;
      // A FINITE clamp, not a MAX_VALUE sentinel: it keeps the key monotone in
      // level (deeper = smaller radius = smaller key = parent first), which the
      // sentinel destroys precisely when the camera is close and ordering
      // matters most.
      const d = Math.max(Math.hypot(dx, dy, dz) - rChild, cam.nearFloor);
      const pf = cam.orthographic
        ? cam.orthoProjFactor
        : (0.5 * cam.viewportHeightPx) / (cam.slope * d);

      const key = spChild * pf; // projected point spacing, device px
      if (key < opts.targetPixelSpacing) {
        if (limitedBy === "complete") limitedBy = "error";
        continue;
      }
      heap = heapPush(s, heap, child.index, key, cc);
      s.pushes++;
    }
  }

  out.count = n;
  out.points = pts;
  out.frame = frame;
  out.limitedBy = limitedBy;
  out.needsExpandCount = ne;
  out.maxSelectedLevel = deepest;
  out.minSpacingWorld = minSpacing === Infinity ? tree.spacingAt(0) : minSpacing;
  return out;
}

/**
 * Ask the hierarchy to fetch the chunks the last selection could not expand
 * synchronously. Fire-and-forget; a no-op inside Task 3's backoff window.
 *
 * Under Task 3's default prefetch policy the whole hierarchy.bin arrives in ONE
 * request, so `tryExpandSync` succeeds for every node and this list is empty in
 * practice — descent costs zero frames of latency.
 */
export function resolveExpansions(
  tree: LodTreeView,
  sel: LodSelection,
  signal?: AbortSignal,
): void {
  for (let k = 0; k < sel.needsExpandCount; k++) {
    const node = tree.node(sel.needsExpand[k]!);
    if (node !== undefined) tree.requestExpand(node, signal);
  }
}
