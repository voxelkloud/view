// The NEUTRAL node, not a driver's. `lod/` still declares its TREE surface
// structurally — that is what keeps this module graph free of runtime imports —
// but the node type is shared vocabulary and worth naming once.
import type { PointCloudNode as HierarchyNode } from "@voxelkloud/core";
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
  /**
   * THE REFINEMENT QUANTITY, world units. Multiplied by the projection factor
   * and compared against `targetScreenError`; the product is also the heap key.
   * For an octree this equals `pointSpacingAt`; for a tile format it is the
   * tile's geometric error.
   */
  geometricErrorAt(level: number): number;
  /**
   * THE POINT PITCH, world units. Drives `minPointSpacingWorld`, and through it
   * the near plane. NEVER the refinement key — see `nodeGeometricError`.
   */
  pointSpacingAt(level: number): number;
  /** Per-level constant: `boundingRadiusAt(L) === boundingRadiusAt(0) / 2 ** L`. */
  boundingRadiusAt(level: number): number;
  /**
   * Dense per-node overrides indexed by `node.index`, for formats whose LOD
   * quantities are not a closed form of the level. `undefined` on every octree
   * format, which is the common case and costs nothing: they are hoisted once
   * per frame, so the branch is constant for the whole traversal.
   *
   * A driver that fills `nodeGeometricError` owes MONOTONICITY
   * (`error[child] <= error[parent]`), clamped at expansion time. The
   * scheduler's best-first ordering is only correct because the key cannot rise
   * as it descends, and it does not re-check.
   */
  readonly nodeGeometricError?: Float64Array | undefined;
  readonly nodePointSpacing?: Float64Array | undefined;
  readonly nodeBoundingRadius?: Float64Array | undefined;
  /**
   * Own-layer point count, overriding `node.numPoints`, for a format that only
   * learns the count when the payload arrives — a `tileset.json` declares none.
   * The budget is charged from this when it is present.
   */
  readonly nodePointCount?: Float64Array | undefined;
  /** Format-native default for `targetScreenError`, device px. */
  readonly defaultScreenError?: number | undefined;
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
   * THE PRIMARY CONTROL. Target screen-space error, in DEVICE pixels.
   * Refinement into a child stops when its projected geometric error would fall
   * below this.
   *
   * Named "error" rather than "spacing" because the quantity is only a point
   * spacing on point octrees — where it is exactly that, and 1.35 px is the
   * calibrated default. On a tile format the same product is a projected tile
   * error and wants a different number, which is what `defaultScreenError` on
   * the tree carries. Resolution order: this option, then the tree's default,
   * then {@link DEFAULT_SCREEN_ERROR}.
   */
  readonly targetScreenError?: number;
  /** @deprecated Renamed to {@link targetScreenError}. */
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
  /**
   * Fraction of `pointBudget` withheld from refinement PAST
   * {@link targetScreenError}. Default 0.15.
   *
   * The budget is spent in two tiers, and this splits them. A node whose
   * projected error still exceeds the target is quality the caller ASKED for,
   * and it may spend the whole budget. A node below the target is a bonus, and
   * bonus nodes may only spend down to `pointBudget * (1 - budgetHeadroom)`.
   * Because the heap is best-first the two tiers never interleave: every
   * required node is popped before every bonus one.
   *
   * Withheld rather than spent because the leftover is what absorbs a camera
   * move — the next frame's selection can grow into it without evicting, and
   * the frame that grows is the frame that was already paying to stream.
   *
   * Setting this to 1 restores the pre-headroom behaviour: refinement stops
   * dead at the target and the budget ceiling is never reached from below.
   */
  readonly budgetHeadroom?: number;
  /**
   * Hard floor on projected geometric error, DEVICE pixels. Refinement never
   * descends past it however much budget is free. Clamped to at most
   * `targetScreenError`.
   *
   * Default `targetScreenError / 4` — {@link BONUS_LEVELS} octree levels past
   * the target, since a level halves the spacing. RELATIVE by default and not a
   * constant, because a constant would make `targetScreenError` inert: with a
   * fat budget every target above the floor would refine to the same floor, and
   * the one knob that is supposed to trade quality for cost would stop doing
   * anything. Asking for 8 px has to stay cheaper than asking for 0.5 px.
   *
   * Two levels is a 4x oversample cap. Past roughly 0.5 px absolute there is
   * nothing left to win in any case: the material draws `2 * sizeMultiplier`
   * spacings per splat, so below `minPixelSize / 2` every splat is pinned at
   * `minPixelSize` and the extra points land in pixels that are already
   * painted. Name this explicitly to cap there instead.
   */
  readonly minScreenError?: number;
}

/** Resolved options. The deprecated alias is collapsed away, never carried. */
export interface ResolvedLodOptions {
  readonly targetScreenError: number;
  readonly pointBudget: number;
  readonly maxNodes: number;
  readonly maxLevel: number;
  readonly maxBudgetSkips: number;
  readonly budgetHeadroom: number;
  /** Already clamped to `<= targetScreenError`. */
  readonly minScreenError: number;
}

/** The point-octree calibration. See `metric.ts` for where 1.35 px comes from. */
export const DEFAULT_SCREEN_ERROR = 1.35;

/**
 * How many octree levels past `targetScreenError` the bonus tier may refine
 * when the budget allows, and so the default `minScreenError` divisor.
 *
 * Two, and it is the knee rather than a round number. Swept on autzen at the
 * 1.35 default against a 3M budget, points selected:
 *
 * | levels | close (0.25) | wide (0.9) |
 * | --- | --- | --- |
 * | 0 | 852,023 (L4) | 98,301 (L2) |
 * | 1 | 1,759,577 (L5) | 308,828 (L3) |
 * | 2 | **2,549,260 (L6)** | **1,002,851 (L4)** |
 * | 3 | 2,549,260 (L6) | 2,549,741 (L5) |
 *
 * At two the close camera has already saturated the headroom, so a third level
 * buys it nothing — while for the wide camera the third level is the one that
 * runs it all the way to the ceiling for a picture 6 km away that cannot show
 * the difference. One level leaves the close camera at 1.76M and a level short
 * of the data that resolves stadium seating and parking bays.
 */
export const BONUS_LEVELS = 2;

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
  /** 11 f64. Index 10 is the per-child flag, written once per frame. */
  readonly params: Float64Array;
  /** 16 f64: `[geometricError; 8]` then `[boundingRadius; 8]`. */
  readonly child: Float64Array;
  selectChildren(mask: number, parentInside: boolean): number;
}

export function resolveLodOptions(
  o: LodOptions = {},
  /** The tree's `defaultScreenError`, when the caller named no target. */
  formatDefault: number = DEFAULT_SCREEN_ERROR,
): ResolvedLodOptions {
  const target = o.targetScreenError ?? o.targetPixelSpacing ?? formatDefault;
  return {
    targetScreenError: target,
    pointBudget: o.pointBudget ?? 3_000_000,
    maxNodes: o.maxNodes ?? 4096,
    maxLevel: o.maxLevel ?? Number.POSITIVE_INFINITY,
    maxBudgetSkips: o.maxBudgetSkips ?? 32,
    // Clamped rather than validated: 1 is "no bonus tier", and a negative or
    // >1 headroom is a caller slip that should not be able to invert the cap.
    budgetHeadroom: Math.min(Math.max(o.budgetHeadroom ?? 0.15, 0), 1),
    // NEVER above the target: the floor bounds how far PAST the ask the bonus
    // tier may go, so a floor above the ask is meaningless.
    minScreenError: Math.min(
      o.minScreenError ?? target * 2 ** -BONUS_LEVELS,
      target,
    ),
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
   * Which constraint stopped refinement, worst first. This field exists because
   * the reference has none, which is why nobody noticed `minimumNodePixelSize`
   * is INERT at its shipped defaults: 150 px and 50 px give byte-identical
   * autzen selections at 1500/300/80 m.
   *
   * - `"budget"` — the target was NOT met: the full budget bound first.
   * - `"nodes"` — `maxNodes` bound first. Same meaning, different ceiling.
   * - `"headroom"` — target met, and the bonus tier then hit
   *   `pointBudget * (1 - budgetHeadroom)`.
   * - `"error"` — target met, and refinement stopped at `minScreenError`. The
   *   best case that still left something on the table.
   * - `"complete"` — the tree ran out. Every point in the cloud is selected.
   *
   * Read it with {@link achievedScreenError}: this says WHAT bound, that says
   * how much it cost.
   */
  limitedBy: "complete" | "error" | "headroom" | "budget" | "nodes";
  /**
   * The WORST projected geometric error left un-refined, device px — the error
   * in the sloppiest region of the picture. 0 when nothing was declined.
   *
   * Compare it against `targetScreenError` to know whether the ask was met:
   * above means some region is coarser than requested, at or below means every
   * region met it and the number is the quality actually delivered.
   */
  achievedScreenError: number;
  /**
   * Min POINT PITCH over ADMITTED nodes only — never the geometric error.
   * Drives near/far, whose rule is calibrated against the 24-bit depth quantum
   * in units of point pitch.
   */
  minPointSpacingWorld: number;
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
    achievedScreenError: 0,
    minPointSpacingWorld: 0,
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
  // HOISTED ONCE PER FRAME. These are constant for the whole traversal, so the
  // branch on each is perfectly predicted and the closed-form formats — every
  // octree, which is the common case — pay nothing for the generality.
  const errArr = tree.nodeGeometricError;
  const spcArr = tree.nodePointSpacing;
  const radArr = tree.nodeBoundingRadius;
  const cntArr = tree.nodePointCount;
  // Whether the child loop must feed the kernel eight errors and eight radii
  // rather than one of each. Constant for the frame, so the kernel is told once
  // and the closed-form formats — every octree — keep writing exactly the two
  // scalars they wrote before this generality existed.
  const perChild = errArr !== undefined || radArr !== undefined;
  const useKernel = kernels !== undefined;
  const frame = ++s.frame;
  let heap = 0;
  let n = 0;
  let pts = 0;
  let skips = 0;
  let ne = 0;
  // Which ceilings were actually touched, resolved into `limitedBy` once at the
  // end. Flags rather than an assign-as-you-go string because the precedence is
  // worst-first and the traversal meets them in no particular order.
  let hitNodes = false;
  let hitBudget = false;
  let hitHeadroom = false;
  let hitFloor = false;
  /** Worst projected error left un-refined. See `achievedScreenError`. */
  let worst = 0;
  let minSpacing = Infinity;
  const target = opts.targetScreenError;
  const floorPx = opts.minScreenError;
  /**
   * The ceiling for refinement PAST the target. Floored, so an integer point
   * count is never compared against a fraction.
   */
  const bonusBudget = Math.floor(opts.pointBudget * (1 - opts.budgetHeadroom));
  let deepest = 0;
  out.levelCounts.fill(0);

  if (!(cam.slope > 0) || !(cam.viewportHeightPx > 0)) {
    out.count = 0;
    out.points = 0;
    out.frame = frame;
    out.limitedBy = "complete";
    out.achievedScreenError = 0;
    out.needsExpandCount = 0;
    out.minPointSpacingWorld = spcArr?.[tree.root.index] ?? tree.pointSpacingAt(0);
    out.maxSelectedLevel = 0;
    return out;
  }

  // Everything except the per-level radius and spacing is constant for the
  // whole frame, so it is written into the kernel block ONCE rather than on
  // every admitted node.
  if (useKernel) {
    const p = kernels!.params;
    p[0] = cam.camX;
    p[1] = cam.camY;
    p[2] = cam.camZ;
    p[5] = cam.nearFloor;
    p[6] = cam.slope;
    p[7] = cam.viewportHeightPx;
    p[8] = cam.orthoProjFactor;
    p[9] = cam.orthographic ? 1 : 0;
    p[10] = perChild ? 1 : 0;
  }

  const rootIndex = tree.root.index;
  heap = heapPush(s, heap, rootIndex, Infinity, Containment.Intersecting);

  while (heap > 0) {
    if (n >= opts.maxNodes) {
      hitNodes = true;
      break;
    }
    heap = heapPop(s, heap);
    const i = s.popNode;
    // CAPTURED, not read later: the next `heapPop` overwrites `s.popKey`, and
    // this node's key is what decides which of the two budget tiers it is in.
    const key = s.popKey;
    const containment = s.popContainment;
    let node = tree.node(i);
    if (node === undefined) continue;

    // BUDGET: skip and continue, never break. The ROOT is exempt — it is the
    // connectivity anchor and 0.1% of the cloud, and a budget below it must
    // still show something.
    //
    // TWO CEILINGS. A node still above the target is quality that was asked
    // for and may spend the whole budget; one below it is a bonus and may only
    // spend down to the headroom. The heap is best-first and the key never
    // rises as it descends, so the tiers cannot interleave — every required
    // node is popped before every bonus one, and this branch flips exactly once
    // per traversal.
    const required = key >= target;
    const cap = required ? opts.pointBudget : bonusBudget;
    const own = cntArr !== undefined ? cntArr[i]! : node.numPoints;
    if (i !== rootIndex && pts + own > cap) {
      if (required) hitBudget = true;
      else hitHeadroom = true;
      if (key > worst) worst = key;
      if (++skips > opts.maxBudgetSkips) break;
      continue;
    }

    pts += own;
    s.visibleEpoch[i] = frame;
    out.indices[n++] = i;
    const lvl = node.level;
    out.levelCounts[lvl] = (out.levelCounts[lvl] ?? 0) + 1;
    if (lvl > deepest) deepest = lvl;
    // POINT PITCH, not the refinement key: this feeds the near plane.
    const sp = spcArr !== undefined ? spcArr[i]! : tree.pointSpacingAt(lvl);
    if (sp < minSpacing) minSpacing = sp;

    // TRI-STATE, not a bit test: `0` is a leaf whatever the format, and an
    // octree's octant bits are a meaning this loop deliberately does not read.
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

    // The closed-form values for the child level. When a per-node array is
    // present these are the fallback; when it is not, they ARE the value.
    const rChildLevel = tree.boundingRadiusAt(lvl + 1);
    const eChildLevel = tree.geometricErrorAt(lvl + 1);

    const kids = node.children;
    const numKids = kids.length;

    if (useKernel) {
      // ONE boundary crossing per EIGHT children, not per child: the arithmetic
      // per child is comparable to the cost of a crossing, so per-child calls
      // would be a wash. Eight is the kernel's fixed block — every octree has
      // exactly that many slots and pays one crossing per admitted node, and an
      // N-ary tile tree pays ceil(N/8) without the kernel changing a byte.
      const { boxes, results, params, child: childBlock } = kernels!;
      params[3] = rChildLevel;
      params[4] = eChildLevel;

      for (let base = 0; base < numKids; base += 8) {
        const span = numKids - base < 8 ? numKids - base : 8;
        let mask = 0;
        for (let c = 0; c < span; c++) {
          const child = kids[base + c];
          if (child === undefined) continue;
          const o = c * 6;
          boxes[o] = child.minX;
          boxes[o + 1] = child.minY;
          boxes[o + 2] = child.minZ;
          boxes[o + 3] = child.maxX;
          boxes[o + 4] = child.maxY;
          boxes[o + 5] = child.maxZ;
          // Filled in the SAME pass that fills the boxes, so the per-child path
          // costs writes and not a second walk of the child slots.
          if (perChild) {
            childBlock[c] =
              errArr !== undefined ? errArr[child.index]! : eChildLevel;
            childBlock[8 + c] =
              radArr !== undefined ? radArr[child.index]! : rChildLevel;
          }
          mask |= 1 << c;
        }

        const survived = kernels!.selectChildren(
          mask,
          containment === Containment.Inside,
        );
        for (let c = 0; c < span; c++) {
          if (((survived >> c) & 1) === 0) continue;
          const child = kids[base + c]!;
          const ckey = results[c * 2 + 1]!;
          // FLOOR: below this no amount of budget can change a pixel.
          if (ckey < floorPx) {
            hitFloor = true;
            if (ckey > worst) worst = ckey;
            continue;
          }
          // A bonus child that cannot possibly fit. `pts` only grows, so
          // failing the headroom cap here means failing it at every later pop
          // too — pruning at push is what makes `budgetHeadroom: 1` cost
          // nothing instead of burning `maxBudgetSkips` on doomed pops.
          if (ckey < target && pts >= bonusBudget) {
            hitHeadroom = true;
            if (ckey > worst) worst = ckey;
            continue;
          }
          heap = heapPush(
            s,
            heap,
            child.index,
            ckey,
            results[c * 2] as Containment,
          );
          s.pushes++;
        }
      }
      continue;
    }

    for (let c = 0; c < numKids; c++) {
      const child = kids[c];
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

      const rChild = radArr !== undefined ? radArr[child.index]! : rChildLevel;
      const eChild = errArr !== undefined ? errArr[child.index]! : eChildLevel;

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

      const ckey = eChild * pf; // projected geometric error, device px
      // FLOOR: below this no amount of budget can change a pixel.
      if (ckey < floorPx) {
        hitFloor = true;
        if (ckey > worst) worst = ckey;
        continue;
      }
      // A bonus child that cannot possibly fit. `pts` only grows, so failing
      // the headroom cap here means failing it at every later pop too —
      // pruning at push is what makes `budgetHeadroom: 1` cost nothing
      // instead of burning `maxBudgetSkips` on doomed pops.
      if (ckey < target && pts >= bonusBudget) {
        hitHeadroom = true;
        if (ckey > worst) worst = ckey;
        continue;
      }
      heap = heapPush(s, heap, child.index, ckey, cc);
      s.pushes++;
    }
  }

  // Whatever is still on the heap was never even looked at, and the max-heap
  // root is the worst of it. Without this a traversal abandoned by
  // `maxBudgetSkips` or `maxNodes` would report the error of the last node it
  // happened to reject rather than of the region it gave up on.
  if (heap > 0 && s.heapKey[0]! > worst) worst = s.heapKey[0]!;

  out.count = n;
  out.points = pts;
  out.frame = frame;
  // WORST FIRST. `budget` and `nodes` mean the target was missed; `headroom`
  // and `error` mean it was met and only the bonus tier was cut short.
  out.limitedBy = hitNodes
    ? "nodes"
    : hitBudget
      ? "budget"
      : hitHeadroom
        ? "headroom"
        : hitFloor
          ? "error"
          : "complete";
  out.achievedScreenError = worst;
  out.needsExpandCount = ne;
  out.maxSelectedLevel = deepest;
  out.minPointSpacingWorld =
    minSpacing === Infinity ? tree.pointSpacingAt(0) : minSpacing;
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
