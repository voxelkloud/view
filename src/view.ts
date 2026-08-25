// THE SEAM IS CLOSED. This file used to import the Potree driver outright,
// because the node payload was the one step no neutral contract covered.
// `PointReader` is that contract: a renderer walks a neutral tree and asks a
// reader for a node's vertices, and Potree, COPC and EPT answer the same way.
// Nothing here names a format any more.
import { VoxelkloudError, isVoxelkloudError } from "@voxelkloud/core";
import type {
  DecodedPointData,
  NodeDecompress,
  PointCloudNode,
  PointCloudSourceBase,
  PointCloudTreeBase,
  PointReader,
  PointReaderFactory,
} from "@voxelkloud/core";
import {
  Matrix4,
  PerspectiveCamera,
  Scene,
  Vector2,
  Vector3,
  WebGPURenderer,
} from "three/webgpu";
import { OctreeCut } from "./cut.js";
import {
  createPointMaterial,
  resolvePointMaterialOptions,
  scalarAttributeFor,
} from "./material.js";
import type { PointCloudMaterial, PointMaterialOptions } from "./material.js";
import { extractFrustumPlanes } from "./lod/frustum.js";
import type { DepthRange } from "./lod/frustum.js";
import { suggestNearFar } from "./lod/metric.js";
import {
  createLodScratch,
  createLodSelection,
  resolveExpansions,
  resolveLodOptions,
  selectVisible,
} from "./lod/select.js";
import type {
  LodCameraState,
  LodOptions,
  LodScratch,
  LodSelection,
  ResolvedLodOptions,
} from "./lod/select.js";
import { createEdlPipeline } from "./edl.js";
import type { EdlOptions, EdlPipeline } from "./edl.js";
import { PointCloudObject3D } from "./object.js";
import { pickPoint as pickPointOnClouds } from "./pick.js";
import type { PickPointOptions, PickResult } from "./pick.js";
import { GroundIndex, type GroundIndexOptions } from "./ground.js";
import { extractProfile as extractProfileFromCloud } from "./profile/index.js";
import type {
  ProfileBatch,
  ProfileExtractionOptions,
  ProfileQuery,
} from "./profile/index.js";
import { createReplaceScratch, filterReplacedParents } from "./replace.js";
import type { ReplaceScratch, ReplaceTreeView } from "./replace.js";
import { ArenaSink } from "./sink-arena.js";
import { buildTriangleBvh, trianglesFromObject } from "./deviation.js";
import { OverlayRenderer } from "./overlay.js";
import { ComputeRasterizer, ComputeSink, OVERLAY_DEPTH_FORMAT } from "./sink-compute.js";
import { PointsRasterizer, PointsSink } from "./sink-points.js";
import { PerNodeSink } from "./sink.js";
import type { PointSink } from "./sink.js";

/**
 * The elevation ramp's domain, in the frame `pointOffset.z` actually lives in.
 *
 * Task 4 emits positions RELATIVE to `metadata.boundingBox.min`, but
 * `tightBoundingBox` is absolute CRS. Feeding the absolute range to the shader
 * subtracts an absolute elevation from a relative one, so `t` clamps to 0 for
 * every point and the whole cloud renders as the ramp's first stop. On the
 * az-usfs survey that is (0 - 1972.67) / 621.94; on autzen (0 - 406) / 209.
 *
 * It only shows on a cloud with NO RGB, since every other cloud uses vertex
 * colour — which is why it survived until a 3DEP survey turned up.
 */
export function cloudRelativeElevationRange(
  source: PointCloudSourceBase,
): [number, number] {
  const originZ = source.bounds.min[2];
  const tight = source.tightBoundingBox;
  return [tight.min[2] - originZ, tight.max[2] - originZ];
}

/** The slice of a tree the pitch helpers read. */
export interface PitchSource {
  pointSpacingAt(level: number): number;
  readonly nodePointSpacing?: Float64Array | undefined;
}

/**
 * ONE node's point pitch: the dense per-node array when the tree has one, the
 * level's closed form otherwise.
 *
 * The array wins because it is the exception a driver goes to the trouble of
 * filling. Task A2 split the refinement error from the point pitch and wired
 * this into the scheduler; every consumer on the RENDER side — the pitch a slab
 * is stamped with, the near-plane floor — kept reading the closed form, which
 * is correct only while the two agree. They agree on every octree and on
 * nothing else.
 *
 * Declared against a STRUCTURAL slice rather than `PointCloudTreeBase`, the
 * same move `lod/select.ts` makes with `LodTreeView`: it is what lets the test
 * hand this a hand-built tree with a pitch array that is deliberately not
 * `s0 / 2 ** level`.
 */
export function pitchOf(
  tree: PitchSource,
  index: number,
  level: number,
): number {
  const arr = tree.nodePointSpacing;
  if (arr === undefined) return tree.pointSpacingAt(level);
  const v = arr[index];
  return v !== undefined && v > 0 ? v : tree.pointSpacingAt(level);
}

/**
 * The scalar ramp's domain, in the units the DECODED lane actually carries.
 *
 * The manifest's `min`/`max` are in source units, but Task 4's `f32` lane
 * applies the attribute's own `normalization` constants when it has any:
 * `f32 = (v - offset) * scale`. Handing the shader the raw range would then
 * compare a normalised value against an un-normalised domain — on a 16-bit
 * intensity that is `t = 0.7 / 65535`, i.e. the whole cloud at the ramp's first
 * stop, which is the exact shape of the elevation bug above.
 *
 * The transform is read off the READER rather than recomputed, so this cannot
 * drift from what the decoder did.
 */
export function scalarRangeFor(
  source: PointCloudSourceBase,
  reader: PointReader,
  name: string,
): [number, number] {
  const attribute = source.attributesByName.get(name);
  const lo = attribute?.min[0] ?? 0;
  const hi = attribute?.max[0] ?? 1;
  const pack = reader.packingFor(name);
  const apply = (v: number) =>
    pack === undefined ? v : (v - pack.offset) * pack.scale;
  const a = apply(lo);
  const b = apply(hi);
  // A degenerate or absent declared range (min === max, or a manifest that
  // simply did not fill them in) would make every point land at ramp stop 0.
  // Classification ignores the range entirely; intensity gets the 16-bit LAS
  // default, which is right far more often than a collapsed one.
  if (!(b > a)) return [0, 65535];
  return [a, b];
}

export interface ViewStats {
  frameMs: number;
  selectMs: number;
  visibleNodes: number;
  visiblePoints: number;
  residentNodes: number;
  residentMB: number;
  drawCalls: number;
  /** Slabs in the arena, or 0 under `sinkMode: "per-node"`. */
  slabs: number;
  /** Decoded nodes waiting for their turn at the per-frame upload budget. */
  /**
   * Instance slots dispatched per frame, live or not.
   *
   * Against `visiblePoints` this is the compaction headroom: the arena masks
   * dead points instead of compacting them out, so the gap between the two is
   * vertex work spent on points nobody is looking at.
   */
  residentPoints: number;
  pendingAttach: number;
  /** True on frames where the reduced moving-camera budget was in force. */
  movingBudget: boolean;
  loading: number;
  maxLevel: number;
  /**
   * The first node-load failure, if any. Surfaced rather than swallowed: a
   * BROTLI cloud with no decompressor otherwise renders as an empty scene with
   * no explanation, which is exactly the failure the loader's actionable error
   * message exists to prevent.
   */
  lastError: string | undefined;
  /**
   * Which constraint stopped refinement. `"error"` means the target spacing was
   * met — the good case. The reference has no such field, which is why nobody
   * noticed its quality knob was inert at the shipped defaults.
   */
  limitedBy: LodSelection["limitedBy"];
  /**
   * Worst projected geometric error left un-refined this frame, device px —
   * across every cloud, so it describes the sloppiest region on screen.
   *
   * The number that makes {@link limitedBy} actionable: `"budget"` at 3.0 px is
   * a picture worth spending on, `"budget"` at 1.4 px against a 1.35 target is
   * a ceiling that is doing its job.
   */
  achievedScreenError: number;
}

export interface PointCloudViewOptions {
  readonly canvas: HTMLCanvasElement;
  readonly lod?: LodOptions;
  readonly material?: PointMaterialOptions;
  /** Concurrent node fetches. Default 12. */
  readonly maxConcurrentLoads?: number;
  /** Evict least-recently-selected nodes past this. Default 512 MiB. */
  readonly maxResidentBytes?: number;
  /**
   * Decoded bytes staged into the sink per frame. Default 8 MiB.
   *
   * Uncapped, a burst of completions all upload in one frame: measured on
   * autzen at target 0.5, p50 sits at 19.8 ms while p95 reaches 72 ms, and the
   * spike follows resident node count rather than anything being drawn.
   */
  readonly maxAttachBytesPerFrame?: number;
  /**
   * A whole-payload decompressor, for a driver that needs one supplied from
   * outside: a BROTLI Potree cloud or a zstandard EPT one, neither of which any
   * current browser can decode on its own.
   *
   * ```ts
   * const { brotliDecompress } = await import("@voxelkloud/loader/brotli");
   * ```
   *
   * Passed straight through to the reader factory `addCloud` was given.
   */
  readonly decompress?: NodeDecompress;
  /**
   * How point data reaches the GPU.
   *
   * `"arena"` (default) packs nodes into level-partitioned slabs: one draw call
   * per slab instead of one per node, which on autzen is ~20-40 rather than the
   * measured 338-1184. It also bounds a leak — three's `Bindings.delete` never
   * destroys the per-object uniform buffer, so per-node meshes leak one per node
   * ever attached.
   *
   * `"per-node"` is the simpler fallback, kept because it is the reference the
   * arena is checked against.
   */
  /**
   * Which rasteriser draws the points. Default `"auto"`, which is the COMPUTE
   * one wherever WebGPU gives us a device.
   *
   * That default is a measurement, not a preference: at a 3M budget INP is
   * ~320 ms through the instanced path and 56-72 ms through compute, against
   * Potree's 136 ms, at the same idle fps and the same point count. The cost
   * the instanced path cannot shed is per INSTANCE, established by elimination
   * rather than guessed — see {@link ComputeSink}.
   *
   * `"auto"` resolves in order: COMPUTE on WebGPU, POINTS on WebGL 2, and the
   * instanced arena only if neither is reachable. All three were measured on one
   * host in one window, same camera and same selected points:
   *
   * | | INP | idle fps |
   * | --- | --- | --- |
   * | compute (WebGPU) | 72 ms | 59.9 |
   * | points (WebGL 2) | 72 ms | 59.9 |
   * | Potree 1.8, for scale | 88 ms | 59.9 |
   * | instanced (WebGL 2) | 656 ms | 7.5 |
   *
   * `"arena"` and `"per-node"` are the instanced paths, and they are NOT
   * obsolete — they are the only ones that draw through three's scene graph, so
   * they are the only ones that compose with other three content: gizmos,
   * meshes, overlays that need to occlude and be occluded. They are the
   * COMPOSITION path now, not the performance one. Name one explicitly to pin
   * it.
   */
  readonly sinkMode?: "auto" | "arena" | "per-node" | "compute" | "points";
  /**
   * Clear colour for the compute rasteriser, linear 0..1. Ignored by the
   * instanced paths, which clear through three.
   */
  readonly background?: readonly [number, number, number];
  /**
   * Use the WebGL 2 backend even where WebGPU is available.
   *
   * The fallback path, made reachable on purpose. Compute shaders are WebGPU
   * only, so this also forces the instanced rasteriser regardless of
   * {@link sinkMode} — `view.rasterizer` will say `"instanced"`.
   */
  readonly forceWebGL?: boolean;
  /**
   * Cancel an in-flight node fetch once the camera has moved past it and the
   * fetch queue is saturated. Default `true`.
   *
   * Default `false`, and the default is measured rather than chosen. Under a
   * panning camera it saves 0.58 MB of 105.79 — 0.5% — and in an A/B on a quiet
   * host it cost ~64 ms of INP (320/344 on, 280/256 off). Trading the one Core
   * Web Vital already behind for half a percent of bytes is a bad deal, so it
   * ships off and stays available for a workload where the bytes matter more.
   */
  readonly abortSuperseded?: boolean;
  /**
   * Node liveness flips the arena may perform per frame. Default 128.
   *
   * Each flip rewrites one node's alpha bytes AND marks that node's whole
   * colour range for re-upload — `4 * count` bytes for a flag change. Under a
   * damped camera nodes cross the selection threshold continuously, so this
   * ceiling is also the per-frame upload ceiling.
   */
  readonly maxLivenessOpsPerFrame?: number;
  /**
   * Fraction of `pointBudget` to spend while the camera is moving. Default 0.35;
   * 1 disables the behaviour.
   *
   * Interaction to Next Paint scales with the number of points drawn, and the
   * cost is fill rate rather than anything on the CPU. Measured on autzen at a
   * 0.25 framing over 20 Mbit, a scripted orbit drag: 3.0M points gives an INP
   * of 280-344 ms, 1.0M gives 112-128 — the difference between failing the
   * 200 ms threshold and matching the fastest arm in the benchmark. Seven other
   * mechanisms were eliminated by measurement before this one was found: input
   * delay, handler cost, long tasks, per-frame CPU, the attach budget, the
   * arena's liveness re-uploads, and the decode backlog.
   *
   * DEFAULT 1 — OFF — because it was implemented, measured, and did not work.
   * Shrinking the SELECTION during motion changed INP not at all (328-368 ms
   * against 328-344 before), while shrinking the whole budget to 1M moved it to
   * 112-128. The difference between those two experiments is resident versus
   * selected, and it is the answer: the arena MASKS rather than compacts, so
   * `setVisible` zeroes a point's alpha but `geometry.instanceCount` still
   * covers every resident point. Hiding removes fragment work and leaves vertex
   * work untouched, and INP here is bound by the latter.
   *
   * The knob stays because a fill-rate-bound GPU would benefit where this one
   * did not. The real lever is compacting the draw range — see the note on
   * `setVisible` in sink-arena.ts.
   */
  readonly movingBudgetScale?: number;
  /**
   * Frames the camera must hold still before the full budget returns. Default 6.
   */
  readonly movingSettleFrames?: number;
  /**
   * Eye-dome lighting. Omitted means off.
   *
   * Off by default because it costs a full-screen pass and a depth-texture
   * round trip that a cloud with good RGB does not need. It earns its keep on
   * the colourless ones — an intensity or single-hue cloud has no other cue for
   * shape.
   */
  readonly edl?: EdlOptions;
}

export interface ViewProfileOptions extends ProfileExtractionOptions {
  readonly cloudIndex?: number;
}

interface CloudHandle {
  readonly source: PointCloudSourceBase;
  readonly hierarchy: PointCloudTreeBase;
  readonly object: PointCloudObject3D;
  readonly material: PointCloudMaterial;
  /**
   * The selected cut, rebuilt every frame and read by the vertex stage to size
   * each splat by the pitch of the finest data at its own position rather than
   * by the level of the node it came from.
   */
  readonly cut: OctreeCut;
  readonly sink: PointSink & { residentBytes: number; nodeCount: number };
  /** Set when this cloud draws through the compute rasteriser. */
  readonly computeSink: ComputeSink | undefined;
  /** Set when this cloud draws through the WebGL 2 points rasteriser. */
  readonly pointsSink: PointsSink | undefined;
  readonly reader: PointReader;
  readonly scratch: LodScratch;
  readonly selection: LodSelection;
  readonly cam: LodCameraState;
  /** index -> last frame it was selected. Drives eviction. */
  readonly lastSeen: Map<number, number>;
  readonly resident: Set<number>;
  /**
   * The draw list, when it is not the selection itself.
   *
   * Only allocated for a tree that marks REPLACE refinement; every octree
   * format hands `selection.indices` straight through.
   */
  visible: Int32Array;
  replaceScratch: ReplaceScratch;
  readonly inFlight: Map<number, AbortController>;
  readonly failed: Set<number>;
  /**
   * Decoded nodes waiting to be staged into the sink.
   *
   * Attaching is a memcpy into a slab plus an update range three uploads on the
   * next render. Doing every completion in the frame it lands makes the upload
   * cost a burst: measured on autzen at target 0.5, frame p50 stays near vsync
   * at 19.8 ms while p95 hits 72 ms, and the spike tracks RESIDENT node count
   * (109 -> 926), not visible points or draw calls. Spreading the staging over
   * frames is what turns that burst into a slope.
   */
  readonly pending: Array<{ index: number; data: DecodedPointData; level: number }>;
  /**
   * Indices sitting in `pending`.
   *
   * Residency is only marked at DRAIN time, so without this a node that has
   * finished loading but not yet been staged is neither in flight nor resident
   * — and `stream` dispatches it a second time. The duplicate then finds its
   * block already allocated, the attach returns 0, and it is reported as a sink
   * refusal.
   */
  readonly queued: Set<number>;
  prevMinSpacing: number;
  /**
   * The finest point pitch seen anywhere in this cloud so far, world units.
   *
   * A PROPERTY OF THE CLOUD, never of a selection — feeding a per-frame
   * quantity back into `cam.nearFloor` closes the loop documented at
   * `updateCamera`. It only ever DECREASES, as lazy expansion discovers deeper
   * nodes and as attached tiles report their measured pitch, so it converges
   * instead of oscillating.
   *
   * Replaces `pointSpacingAt(maxLevel)`, which is that same number only while
   * the pitch is a closed form of the level. A tileset carries a per-node pitch
   * and the deepest level is not the finest one.
   */
  minPitchWorld: number;
  /** How far `minPitchWorld` has scanned `nodePointSpacing`. */
  scannedNodes: number;
}

/**
 * The cloud-wide point-pitch floor, refreshed incrementally.
 *
 * On a closed-form tree this is `pointSpacingAt(maxLevel)` and nothing changed:
 * that number only ever falls as expansion discovers deeper chunks, so folding
 * it into a running minimum returns exactly the same value.
 *
 * On a per-node tree the deepest level is NOT the finest, so the floor is the
 * minimum over the array. Each node is scanned once ever — O(nodes added) per
 * frame, not O(nodeCount) — and entries a driver fills LATER, after a tile is
 * decoded, are folded in at attach time instead. Both paths only ever lower it,
 * which is the invariant `updateCamera` depends on.
 */
export function cloudPitchFloor(h: {
  readonly hierarchy: PitchSource & {
    readonly nodeCount: number;
    readonly maxLevel: number;
  };
  minPitchWorld: number;
  scannedNodes: number;
}): number {
  const arr = h.hierarchy.nodePointSpacing;
  if (arr === undefined) {
    const closed = h.hierarchy.pointSpacingAt(h.hierarchy.maxLevel);
    if (closed < h.minPitchWorld) h.minPitchWorld = closed;
    return h.minPitchWorld;
  }
  const n = Math.min(h.hierarchy.nodeCount, arr.length);
  for (let i = h.scannedNodes; i < n; i++) {
    const v = arr[i]!;
    if (v > 0 && v < h.minPitchWorld) h.minPitchWorld = v;
  }
  h.scannedNodes = n;
  return h.minPitchWorld;
}

/**
 * A point cloud viewer over three's WebGPU renderer.
 *
 * Construction is synchronous and touches no GPU; `init()` is the async,
 * cancellable step that creates the device. Nothing happens at module scope, so
 * importing this package is safe under SSR and in Node.
 */
export class PointCloudView {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly renderer: WebGPURenderer;
  readonly stats: ViewStats = {
    frameMs: 0,
    selectMs: 0,
    visibleNodes: 0,
    visiblePoints: 0,
    residentNodes: 0,
    residentMB: 0,
    drawCalls: 0,
    slabs: 0,
    residentPoints: 0,
    pendingAttach: 0,
    movingBudget: false,
    loading: 0,
    maxLevel: 0,
    lastError: undefined,
    limitedBy: "complete",
    achievedScreenError: 0,
  };

  private readonly clouds: CloudHandle[] = [];
  private readonly groundIndices = new Map<number, GroundIndex>();
  /**
   * Whether the CALLER named a target. Only when they did not may a cloud's
   * format-native `defaultScreenError` take over — an explicit 1.35 must not be
   * silently replaced by a driver's 16.
   */
  private readonly targetNamed: boolean;
  /** Set once, by the first cloud that carries a format default. */
  private targetAdopted = false;
  private readonly lodOptions: { -readonly [K in keyof ResolvedLodOptions]: ResolvedLodOptions[K] };
  private readonly materialOptions;
  private readonly maxConcurrent: number;
  private readonly maxResidentBytes: number;
  private readonly maxAttachBytes: number;
  private readonly abortSuperseded: boolean;
  private readonly movingBudgetScale: number;
  private readonly movingSettleFrames: number;
  /** Camera world matrix as of the last frame, to detect motion without an API. */
  private readonly lastCam = new Float64Array(16);
  /** Frames since the camera last changed. */
  private stillFrames = 0;
  /** `lodOptions`, or a budget-reduced copy while the camera is moving. */
  private frameLodOptions!: ResolvedLodOptions;
  private readonly scratchMatrix = new Matrix4();
  private readonly scratchInverse = new Matrix4();
  private readonly scratchVec = new Vector3();
  private readonly drawSize = new Vector2();
  private readonly clipFromCloudM = new Matrix4();
  private readonly viewFromCloudM = new Matrix4();
  /**
   * The screen-sized half of the compute path, shared by every cloud so they
   * accumulate into ONE depth buffer and therefore occlude each other. Created
   * after `init`, because it needs the device three opened.
   */
  private raster: ComputeRasterizer | undefined;
  /** The WebGL 2 points path's frame-level half, and the context it draws with. */
  private pointsRaster: PointsRasterizer | undefined;
  private gl: WebGL2RenderingContext | undefined;
  private overlay: OverlayRenderer | undefined;
  /** What actually drew, after capability resolution. Public so a caller can
   *  see that a compute request fell back rather than guess from a frame time. */
  rasterizer: "compute" | "points" | "instanced" = "instanced";
  private depthRange: DepthRange = "minus-one-to-one";
  private frame = 0;
  private initialized = false;
  private disposed = false;
  private dirty = true;
  private edl: EdlPipeline | undefined;
  /** Held so a cloud added after the cut still gets it. */
  private clipPlanes: Float32Array | undefined;

  constructor(private readonly options: PointCloudViewOptions) {
    this.camera = new PerspectiveCamera(60, 1, 1, 20_000);
    // Potree v2 data is Z-up CRS; three defaults to Y-up, which puts every
    // terrain cloud on its side.
    this.camera.up.set(0, 0, 1);
    this.renderer = new WebGPURenderer({
      canvas: options.canvas,
      antialias: false,
      // Forcing the fallback is how it gets TESTED. A path that only runs on
      // hardware nobody on the team has is a path nobody finds out is broken.
      ...(options.forceWebGL === true ? { forceWebGL: true } : {}),
    });
    // NOT optional. Without it the model-view product is computed in float32 IN
    // THE SHADER, and cloud-relative positions come out WORSE than absolute
    // ones — a measured 34.1 px of jitter against 29.2 px. With it, 0.46 px.
    this.renderer.highPrecision = true;
    this.targetNamed =
      options.lod?.targetScreenError !== undefined ||
      options.lod?.targetPixelSpacing !== undefined;
    this.lodOptions = resolveLodOptions(options.lod);
    this.materialOptions = options.material ?? {};
    this.maxConcurrent = options.maxConcurrentLoads ?? 12;
    this.maxResidentBytes = options.maxResidentBytes ?? 512 * 1024 * 1024;
    this.maxAttachBytes = options.maxAttachBytesPerFrame ?? 8 * 1024 * 1024;
    this.abortSuperseded = options.abortSuperseded ?? false;
    this.movingBudgetScale = Math.min(Math.max(options.movingBudgetScale ?? 1, 0.05), 1);
    this.movingSettleFrames = Math.max(options.movingSettleFrames ?? 6, 1);
  }

  /** Whether the caller asked for compute, explicitly or by taking the default. */
  private wantsCompute(): boolean {
    const mode = this.options.sinkMode ?? "auto";
    return (mode === "auto" || mode === "compute") && this.options.forceWebGL !== true;
  }

  /** Points is the WebGL 2 answer, so `"auto"` reaches it only where compute
   *  did not — which `forceWebGL` also makes reachable on purpose. */
  private wantsPoints(): boolean {
    const mode = this.options.sinkMode ?? "auto";
    return mode === "points" || (mode === "auto" && this.raster === undefined);
  }

  async init(): Promise<void> {
    if (this.initialized) return;
    await this.renderer.init();
    // Read AFTER init: three constructs a Camera with the WebGL convention and
    // Renderer.render overwrites it at first render, so the camera is not a
    // trustworthy source for this.
    this.depthRange =
      this.renderer.coordinateSystem === 2000 ? "zero-to-one" : "minus-one-to-one";
    // AFTER `renderer.init`: PostProcessing builds a quad and a render target
    // against the live backend, so it needs a device.
    if (this.options.edl !== undefined && !this.wantsCompute()) {
      this.edl = createEdlPipeline(
        this.renderer,
        this.scene,
        this.camera,
        this.options.edl,
      );
    }
    // The compute path needs the raw device and swapchain three just opened.
    // Reached through the backend rather than re-created, because a second
    // device would mean a second copy of every buffer.
    if (this.wantsCompute()) {
      const backend = (this.renderer as unknown as {
        backend?: { device?: GPUDevice; context?: GPUCanvasContext };
      }).backend;
      const device = backend?.device;
      const context = backend?.context;
      if (device !== undefined && context !== undefined && typeof navigator !== "undefined") {
        this.raster = new ComputeRasterizer(
          device,
          context,
          navigator.gpu.getPreferredCanvasFormat(),
          {
            background: this.options.background ?? [0, 0, 0],
            ...(this.options.edl !== undefined
              ? {
                  edl: {
                    strength: this.options.edl.strength ?? 1,
                    radius: this.options.edl.radius ?? 1.4,
                  },
                }
              : {}),
          },
        );
        this.rasterizer = "compute";
      }
    }
    // POINTS, where compute did not reach. `backend.gl` is the WebGL 2 context
    // three opened — the same seam as `backend.device` on the other side.
    if (this.raster === undefined && this.wantsPoints()) {
      const backend = (this.renderer as unknown as { backend?: { gl?: WebGL2RenderingContext } }).backend;
      const gl = backend?.gl;
      if (gl !== undefined && typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext) {
        this.pointsRaster = new PointsRasterizer(gl, {
          background: this.options.background ?? [0, 0, 0],
          ...(this.options.edl !== undefined
            ? {
                edl: {
                  strength: this.options.edl.strength ?? 1,
                  radius: this.options.edl.radius ?? 1.4,
                },
              }
            : {}),
        });
        this.gl = gl;
        this.rasterizer = "points";
      }
    }
    this.initialized = true;
    this.dirty = true;
  }

  /**
   * Add a cloud. `hierarchy` must already have its root expanded.
   *
   * `openPoints` is the driver's reader factory — `loadPointCloud` returns one
   * bound to the source, or take it from `format.openPoints`. It is a FACTORY
   * rather than a ready reader because the attribute selection depends on this
   * view's colour mode, which the caller does not know.
   */
  addCloud(
    source: PointCloudSourceBase,
    hierarchy: PointCloudTreeBase,
    openPoints: PointReaderFactory,
  ): PointCloudObject3D {
    const origin = source.bounds.min;
    const sceneOrigin =
      this.clouds.length === 0 ? origin : this.clouds[0]!.object.getSceneOrigin();
    const object = new PointCloudObject3D(origin, sceneOrigin);

    // A colour mode that reads a per-point scalar needs that attribute both
    // DECODED and in a lane three can bind. `_getVertexFormat` has no entry for
    // a 1-component Uint8Array, so a raw uint8 classification resolves to
    // `undefined` in the pipeline descriptor and the device rejects the draw —
    // Task 4's `scalarFormat: "gpu"` with an `f32` lane is what avoids that.
    const colorMode = this.materialOptions.colorMode;
    const scalarAttribute =
      colorMode === undefined ? undefined : scalarAttributeFor(colorMode);
    if (
      scalarAttribute !== undefined &&
      !source.attributesByName.has(scalarAttribute)
    ) {
      throw new VoxelkloudError(
        "unsupported-point-data",
        `Colour mode ${JSON.stringify(colorMode?.kind)} reads a per-point ` +
          `attribute named ${JSON.stringify(scalarAttribute)}, which this ` +
          `cloud does not have. It has: ` +
          `${source.attributes.map((a) => JSON.stringify(a.name)).join(", ")}.`,
      );
    }

    // Naming the scalar deselects colour, which is right: none of the modes
    // that read a scalar also read RGB, and not fetching it halves the bytes
    // per point on a cloud that has both.
    const reader = openPoints({
      computeBounds: true,
      ...(this.options.decompress !== undefined
        ? { decompress: this.options.decompress }
        : {}),
      ...(scalarAttribute !== undefined
        ? {
            attributes: [scalarAttribute],
            scalarFormat: "gpu" as const,
            lanes: { [scalarAttribute]: "f32" as const },
          }
        : {}),
    });

    const materialOptions = {
      // CLOUD-RELATIVE, because that is the frame `pointOffset` is in. The tight
      // bounds are absolute CRS, and feeding those straight in subtracts an
      // absolute elevation from a relative one: on autzen that is (0 - 406) /
      // 209, clamped to 0 for every point, so the whole cloud renders as the
      // ramp's first stop. It only shows on a cloud with no RGB, which is why it
      // survived until a 3DEP survey turned up.
      elevationRange: cloudRelativeElevationRange(source),
      ...(scalarAttribute !== undefined
        ? { scalarRange: scalarRangeFor(source, reader, scalarAttribute) }
        : {}),
      ...this.materialOptions,
    };
    const material = createPointMaterial(materialOptions);
    // The compute path needs the SAME resolved numbers the material derived —
    // splat multiplier, pixel clamps, ramp ranges — so both rasterisers size a
    // splat identically and a comparison between them is a comparison.
    const resolved = resolvePointMaterialOptions(materialOptions);
    // Task 4 writes the SOURCE alpha when the colour attribute has four
    // elements, so whether the arena must stamp it is a per-cloud fact, decided
    // here rather than sniffed per node.
    const color = source.attributes.find((a) => a.role === "color");
    const needsAlphaStamp = (color?.numElements ?? 3) >= 4;
    const cut = new OctreeCut(this.lodOptions.maxNodes);
    material.uCutMap.value = cut.map;

    // Resolved here, not in `init`: a caller who never awaited `init` has no
    // device yet, and silently drawing through a different rasteriser than the
    // one they asked for is worse than saying so. `rasterizer` reports what won.
    const useCompute = this.raster !== undefined;
    if (this.wantsCompute() && !useCompute) {
      this.rasterizer = "instanced";
      console.warn(
        "voxelkloud: compute rasteriser unavailable, falling back to instanced. " +
          "`await view.init()` before `addCloud` if the device was simply not ready yet.",
      );
    }
    const root0 = hierarchy.root;
    const computeSink = useCompute
      ? new ComputeSink(
          (this.renderer as unknown as { backend: { device: GPUDevice } }).backend.device,
          this.raster!,
          cut,
          {
            // CLOUD-LOCAL, the same frame `pointOffset` and the arena use.
            min: [root0.minX - origin[0], root0.minY - origin[1], root0.minZ - origin[2]],
            size: [root0.maxX - root0.minX, root0.maxY - root0.minY, root0.maxZ - root0.minZ],
          },
          {
            pointBudget: this.lodOptions.pointBudget,
            colorMode: resolved.colorMode,
            sizeMultiplier: resolved.sizeMultiplier,
            minPixelSize: resolved.minPixelSize,
            maxPixelSize: resolved.maxPixelSize,
            elevationRange: resolved.elevationRange,
            scalarRange: resolved.scalarRange,
            background: this.options.background ?? [0, 0, 0],
          },
          scalarAttribute,
        )
      : undefined;
    computeSink?.setClipPlanes(this.clipPlanes);
    const pointsSink =
      this.pointsRaster !== undefined && this.gl !== undefined
        ? new PointsSink(
            this.gl,
            cut,
            {
              // CLOUD-LOCAL, the same frame `pointOffset` and the arena use.
              min: [root0.minX - origin[0], root0.minY - origin[1], root0.minZ - origin[2]],
              size: [root0.maxX - root0.minX, root0.maxY - root0.minY, root0.maxZ - root0.minZ],
            },
            {
              pointBudget: this.lodOptions.pointBudget,
              colorMode: resolved.colorMode,
              sizeMultiplier: resolved.sizeMultiplier,
              minPixelSize: resolved.minPixelSize,
              maxPixelSize: resolved.maxPixelSize,
              elevationRange: resolved.elevationRange,
              scalarRange: resolved.scalarRange,
              background: this.options.background ?? [0, 0, 0],
            },
            scalarAttribute,
          )
        : undefined;
    const sink =
      computeSink ??
      pointsSink ??
      ((this.options.sinkMode ?? "auto") === "per-node"
        ? new PerNodeSink(object, material, scalarAttribute)
        : new ArenaSink(
            object,
            material,
            (l) => hierarchy.pointSpacingAt(l),
            (l) => hierarchy.boundingRadiusAt(l),
            needsAlphaStamp,
            this.options.maxLivenessOpsPerFrame !== undefined
              ? { maxLivenessOpsPerFrame: this.options.maxLivenessOpsPerFrame }
              : {},
            scalarAttribute,
          ));

    // A tile format's error is not a point pitch, so its calibrated threshold is
    // not 1.35 px. Adopt it only when the caller named none, and only once: two
    // clouds with different defaults would otherwise fight over a knob that is
    // global to the view.
    if (
      !this.targetNamed &&
      !this.targetAdopted &&
      hierarchy.defaultScreenError !== undefined
    ) {
      this.lodOptions.targetScreenError = hierarchy.defaultScreenError;
      this.targetAdopted = true;
    }

    // CLOUD-LOCAL, the frame `pointOffset` is in. `origin` is `source.bounds.min`
    // and the root node's box IS `source.bounds`, so this reduces to a zero
    // corner and the cube extent — written out rather than assumed, so a format
    // whose root box is not the indexing volume still walks the right box.
    const root = hierarchy.root;
    material.uRootMin.value = {
      x: root.minX - origin[0],
      y: root.minY - origin[1],
      z: root.minZ - origin[2],
    };
    material.uRootSize.value = {
      x: root.maxX - root.minX,
      y: root.maxY - root.minY,
      z: root.maxZ - root.minZ,
    };

    this.scene.add(object);
    this.clouds.push({
      source,
      hierarchy,
      object,
      material,
      cut,
      sink,
      computeSink,
      pointsSink,
      reader,
      scratch: createLodScratch(),
      selection: createLodSelection(this.lodOptions.maxNodes),
      cam: {
        clipFromAbs: new Float64Array(16),
        camX: 0,
        camY: 0,
        camZ: 0,
        slope: 1,
        viewportHeightPx: 1,
        orthographic: false,
        orthoProjFactor: 0,
        nearFloor: 1,
        depthRange: this.depthRange,
        reversedDepth: false,
      },
      lastSeen: new Map(),
      resident: new Set(),
      visible: new Int32Array(0),
      replaceScratch: createReplaceScratch(0),
      inFlight: new Map(),
      failed: new Set(),
      pending: [],
      queued: new Set(),
      prevMinSpacing: pitchOf(hierarchy, hierarchy.root.index, 0),
      minPitchWorld: pitchOf(hierarchy, hierarchy.root.index, 0),
      scannedNodes: 0,
    });
    this.dirty = true;
    return object;
  }

  /** Frame the camera on a cloud's TIGHT bounds — never the cubic octree box,
   *  which on autzen is 22x taller than the data and would aim at empty sky. */
  frameCloud(index = 0): void {
    const h = this.clouds[index];
    if (h === undefined) return;
    const b = h.source.tightBoundingBox;
    const origin = h.object.getSceneOrigin();
    const cx = (b.min[0] + b.max[0]) / 2 - origin[0];
    const cy = (b.min[1] + b.max[1]) / 2 - origin[1];
    const cz = (b.min[2] + b.max[2]) / 2 - origin[2];
    const span = Math.max(
      b.max[0] - b.min[0],
      b.max[1] - b.min[1],
      b.max[2] - b.min[2],
    );
    const d = span * 0.9;
    this.camera.position.set(cx + d, cy - d, cz + d * 0.6);
    this.camera.lookAt(cx, cy, cz);
    this.camera.near = Math.max(span / 5000, 0.1);
    this.camera.far = span * 20;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  /** Where `frameCloud` aims — the orbit target for controls. */
  targetFor(index = 0): Vector3 {
    const h = this.clouds[index];
    if (h === undefined) return new Vector3();
    const b = h.source.tightBoundingBox;
    const o = h.object.getSceneOrigin();
    return new Vector3(
      (b.min[0] + b.max[0]) / 2 - o[0],
      (b.min[1] + b.max[1]) / 2 - o[1],
      (b.min[2] + b.max[2]) / 2 - o[2],
    );
  }

  /**
   * The overlay renderer, or undefined when the scene holds nothing to draw.
   *
   * Checked per frame so an app that adds nothing pays nothing beyond a walk
   * of the scene's direct children.
   */
  private overlayFor(): OverlayRenderer | undefined {
    if (this.scene.children.length === 0) return undefined;
    const device = (this.renderer as unknown as { backend?: { device?: GPUDevice } }).backend
      ?.device;
    if (device === undefined) return undefined;
    this.overlay ??= new OverlayRenderer(
      device,
      navigator.gpu.getPreferredCanvasFormat(),
      OVERLAY_DEPTH_FORMAT,
    );
    // Nobody else updates these. On the compute path `renderer.render(scene)`
    // never runs, so three never walks the graph — a mesh whose local
    // transform was set still has an IDENTITY world matrix, and draws at the
    // scene origin instead of where the app put it. That cost an afternoon.
    this.scene.updateMatrixWorld(true);
    const clouds = new Set(this.clouds.map((h) => h.object as unknown as object));
    return this.overlay.collect(this.scene, clouds) ? this.overlay : undefined;
  }

  /**
   * A synchronous height/obstacle index over the points currently resident.
   *
   * Exists for callers that run inside a frame — a vehicle, a first-person
   * walker, a collision probe — and cannot await a traversal. It answers from
   * the cut the scheduler has already chosen, so the answer is as detailed as
   * what is on screen and no more; `support` on each sample says how much data
   * stood behind it.
   *
   * The index rebuilds itself only when the resident set changes, so calling
   * this every frame is free after the first.
   */
  groundIndex(cloudIndex = 0, options?: GroundIndexOptions): GroundIndex | undefined {
    const h = this.clouds[cloudIndex];
    if (h === undefined) return undefined;
    let index = this.groundIndices.get(cloudIndex);
    if (index === undefined) {
      index = new GroundIndex();
      this.groundIndices.set(cloudIndex, index);
    }
    // The selected-node count and the newest node index together change
    // whenever the cut does, and both are already maintained per frame.
    const token =
      h.selection.count * 1_000_003 +
      (h.selection.count > 0 ? h.selection.indices[h.selection.count - 1]! : 0);
    index.rebuild(
      {
        selection: h.selection.indices,
        selectionCount: h.selection.count,
        sceneOrigin: h.object.getSceneOrigin() as [number, number, number],
        cloudOrigin: h.object.cloudOrigin as [number, number, number],
        node: (i: number) => h.hierarchy.node(i),
        readPoints: (i: number) => h.sink.readPoints(i),
      },
      token,
      options ?? {},
    );
    return index;
  }

  /**
   * B5 — mede a nuvem contra esta malha.
   *
   * Constrói a BVH uma vez, entrega-a a cada nuvem e corre o kernel. O
   * resultado entra no lugar do escalar, então desenhar o desvio é
   * `setColorMode("scalar")` com a faixa que a tolerância pedir — a rampa, a
   * legenda e o relatório já existem (DEC-B8).
   *
   * `maxDistance` corta a travessia cedo E limita a rampa: um telhado a 40 m do
   * modelo não é um desvio, é outro prédio.
   *
   * Devolve quantos pontos foram medidos, ou 0 se não havia malha nem pontos.
   */
  setDeviationMesh(model: { updateMatrixWorld(force?: boolean): void; traverse(cb: (o: unknown) => void): void }, maxDistance = 5): number {
    if (this.disposed) return 0;
    const tris = trianglesFromObject(model as unknown as Parameters<typeof trianglesFromObject>[0]);
    const bvh = buildTriangleBvh(tris);
    let measured = 0;
    for (const h of this.clouds) {
      const e = h.object.matrixWorld.elements;
      h.computeSink?.setDeviationMesh(bvh.nodes, bvh.tris, [e[12]!, e[13]!, e[14]!], maxDistance);
      measured += h.computeSink?.runDeviation() ?? 0;
    }
    this.dirty = true;
    return measured;
  }

  /** Remede depois de a nuvem ter carregado mais pontos. */
  refreshDeviation(): number {
    let n = 0;
    for (const h of this.clouds) n += h.computeSink?.runDeviation() ?? 0;
    this.dirty = true;
    return n;
  }

  /**
   * Cross-section planes, in SCENE coordinates: `[nx, ny, nz, d]` each, four at
   * most. The positive half-space survives, which is three's convention.
   *
   * They cut the POINTS and the MODEL with one call, and that is the whole
   * decision (DEC-B6): a section that slices the scan and leaves the wall
   * standing is worse than no section, because it looks like an answer.
   */
  setClipPlanes(planes: Float32Array | undefined): void {
    this.clipPlanes = planes;
    this.overlay?.setClipPlanes(planes);
    for (const h of this.clouds) h.computeSink?.setClipPlanes(planes);
    this.dirty = true;
  }

  /**
   * Which BIM elements are drawn: one entry per dense feature index, zero to
   * hide. `undefined` shows everything. This is how isolating a storey or a
   * class of element works — the geometry never moves.
   */
  setElementVisibility(mask: Uint32Array | undefined): void {
    this.overlay?.setVisibility(mask);
    this.dirty = true;
  }

  /**
   * Highlight one BIM element, by the same dense index `pickElement` returns.
   * `undefined` clears it. Takes effect on the next frame.
   */
  setSelectedElement(feature: number | undefined): void {
    this.overlay?.setSelected(feature);
    this.dirty = true;
  }

  /**
   * Which BIM element is under the cursor, as the dense feature index the
   * converter wrote. `undefined` when there is no model there — including when
   * the point cloud is in front of it, which is deliberate: an element hidden
   * behind the scan should not be selectable.
   *
   * Only answers on the compute path, because that is where the mesh is drawn
   * and where the points' depth buffer exists to test against.
   *
   * Coordinates are CSS pixels, as an event gives them; the pixel ratio is
   * applied here so a caller never has to remember it.
   */
  async pickElement(screenX: number, screenY: number): Promise<number | undefined> {
    if (this.disposed) return undefined;
    const depth = this.raster?.depth;
    if (depth === undefined) return undefined;
    const overlay = this.overlayFor();
    if (overlay === undefined) return undefined;
    const ratio = this.renderer.getPixelRatio();
    return overlay.pickFeature(
      this.camera,
      depth,
      this.drawSize.x,
      this.drawSize.y,
      screenX * ratio,
      screenY * ratio,
    );
  }

  /**
   * Pick the closest point currently selected by the scheduler.
   */
  pickPoint(
    screenX: number,
    screenY: number,
    options: PickPointOptions = {},
  ): PickResult | undefined {
    if (this.disposed) return undefined;
    if (!(this.drawSize.x > 0) || !(this.drawSize.y > 0)) return undefined;
    return pickPointOnClouds(
      this.camera,
      this.drawSize.x,
      this.drawSize.y,
      screenX,
      screenY,
      this.clouds.map((h, cloudIndex) => ({
        cloudIndex,
        cloudOrigin: h.object.cloudOrigin,
        sceneOrigin: h.object.getSceneOrigin(),
        selection: h.selection.indices,
        selectionCount: h.selection.count,
        node: (index: number) => h.hierarchy.node(index),
        readPoints: (index: number) => h.sink.readPoints(index),
      })),
      options,
    );
  }

  /**
   * Extract a profile independently from the current LOD selection.
   *
   * The returned async iterable yields point batches as nodes are read, so a
   * caller can progressively fill a 2D profile view instead of waiting for the
   * whole query to finish.
   */
  extractProfile(
    query: ProfileQuery,
    options: ViewProfileOptions = {},
  ): AsyncGenerator<ProfileBatch> {
    const { cloudIndex = 0, ...profileOptions } = options;
    const h = this.clouds[cloudIndex];
    if (h === undefined) {
      throw new RangeError(`No point cloud exists at index ${cloudIndex}.`);
    }
    return extractProfileFromCloud(
      {
        source: h.source,
        hierarchy: h.hierarchy,
        // The view's own reader, which already has this cloud's decompressor
        // and attribute selection. A profile that wants a different selection
        // passes `pointData` and gets its own.
        openPoints: () => h.reader,
        readPoints: (index: number) => h.sink.readPoints(index),
      },
      query,
      profileOptions,
    );
  }

  /**
   * Whether the EDL pass exists on this view.
   *
   * Surfaced because {@link setEdl} is a no-op without it, and a measurement
   * run that swept EDL strength against a view built without the pass produced
   * three identical stops that read as "EDL is free". A caller measuring
   * anything about EDL must be able to assert this first.
   */
  get hasEdl(): boolean {
    return this.edl !== undefined;
  }

  /**
   * Eye-dome lighting knobs, live.
   *
   * Uniform writes, so none of these recompile a shader or rebuild the pass.
   * Whether EDL exists AT ALL is fixed at construction: turning it on later
   * would mean building a render target mid-frame.
   */
  setEdl(o: EdlOptions): void {
    if (this.edl === undefined) return;
    if (o.strength !== undefined) this.edl.setStrength(o.strength);
    if (o.radius !== undefined) this.edl.setRadius(o.radius);
    if (o.opacity !== undefined) this.edl.setOpacity(o.opacity);
    this.dirty = true;
  }

  setSize(width: number, height: number, pixelRatio = 1): void {
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(width, height, false);
    this.drawSize.set(width * pixelRatio, height * pixelRatio);
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  invalidate(): void {
    this.dirty = true;
  }

  /**
   * Change the LOD quality target at runtime — the primary control, in device
   * pixels of projected geometric error. On a point octree that error IS the
   * inter-point spacing; on a tile format it is not, which is why the parameter
   * is no longer named for a spacing.
   *
   * Lower refines further. Worth turning down when `stats.limitedBy` reports
   * `"error"` while frame time is flat: that combination means the target was
   * met, the bonus tier ran down to `minScreenError`, and there is still budget
   * and frame to spare.
   */
  setTargetScreenError(px: number): void {
    this.lodOptions.targetScreenError = px;
    this.dirty = true;
  }

  /** @deprecated Renamed to {@link setTargetScreenError}. */
  setTargetPixelSpacing(px: number): void {
    this.setTargetScreenError(px);
  }

  /**
   * Points drawn per frame.
   *
   * Raise this BEFORE lowering `targetScreenError`: measured on autzen at a
   * 0.25 framing, the 3M default binds at a target of 1.0 and below, so the
   * quality knob does nothing until the budget has room. `stats.limitedBy`
   * reports which of the two is actually holding — `"headroom"` means this
   * ceiling, `"budget"` means it bound before the target was even met.
   */
  setPointBudget(points: number): void {
    this.lodOptions.pointBudget = points;
    this.dirty = true;
  }

  renderFrame(): boolean {
    if (!this.initialized || this.disposed) return false;
    const t0 = performance.now();
    this.frame++;
    // Camera motion, detected from the world matrix rather than announced by
    // the app: a viewer embedded behind React or Vue has no reliable place to
    // call setInteracting, and the matrix is the ground truth either way.
    {
      // position + quaternion, NOT matrixWorld: three only refreshes
      // matrixWorld inside render(), so at the top of renderFrame it still
      // describes the PREVIOUS frame. Controls write position and quaternion
      // synchronously, so they are the honest "has the camera moved" test.
      const cp = this.camera.position;
      const cq = this.camera.quaternion;
      const c = this.lastCam;
      const moved =
        c[0] !== cp.x || c[1] !== cp.y || c[2] !== cp.z ||
        c[3] !== cq.x || c[4] !== cq.y || c[5] !== cq.z || c[6] !== cq.w;
      if (moved) {
        this.stillFrames = 0;
        c[0] = cp.x; c[1] = cp.y; c[2] = cp.z;
        c[3] = cq.x; c[4] = cq.y; c[5] = cq.z; c[6] = cq.w;
      } else if (this.stillFrames < this.movingSettleFrames) {
        this.stillFrames++;
      }
      const moving = this.stillFrames < this.movingSettleFrames;
      this.stats.movingBudget = moving;
      this.frameLodOptions =
        moving && this.movingBudgetScale < 1
          ? {
              ...this.lodOptions,
              pointBudget: Math.max(
                1,
                Math.round(this.lodOptions.pointBudget * this.movingBudgetScale),
              ),
            }
          : this.lodOptions;
    }

    this.camera.updateMatrixWorld(true);
    this.renderer.getDrawingBufferSize(this.drawSize);

    let selectMs = 0;
    let visibleNodes = 0;
    let visiblePoints = 0;
    let residentNodes = 0;
    let residentBytes = 0;
    let loading = 0;
    let maxLevel = 0;
    let slabs = 0;
    let limitedBy: LodSelection["limitedBy"] = "complete";
    let achievedScreenError = 0;

    for (const h of this.clouds) {
      // Near/far FIRST, from the previous frame's selection, so the culling
      // frustum and the render frustum are the same matrix within a frame. The
      // reference writes camera.near at the END of its traversal, closing an
      // oscillating feedback loop on a stationary camera.
      const viewDepth = this.camera.position.distanceTo(this.targetFor(0)) * 2;
      const { near, far } = suggestNearFar(h.prevMinSpacing, viewDepth);
      if (near / this.camera.near > 2 || this.camera.near / near > 2) {
        this.camera.near = near;
        this.camera.far = Math.max(far, near + 10);
        this.camera.updateProjectionMatrix();
      }

      this.buildCameraState(h);

      const ts = performance.now();
      selectVisible(h.hierarchy, h.cam, this.frameLodOptions, h.scratch, h.selection);
      selectMs += performance.now() - ts;
      resolveExpansions(h.hierarchy, h.selection);

      // AFTER the selection and BEFORE the draw, from the same frame's epoch
      // stamps. A cut built from last frame's selection would size splats
      // against nodes the sink is about to hide.
      h.cut.build(
        h.hierarchy.root,
        (i) => h.hierarchy.node(i),
        h.scratch.visibleEpoch,
        h.selection.frame,
        this.lodOptions.maxNodes,
        h.resident,
      );
      // Cheap and unconditional: `OctreeCut` replaces its texture only when
      // `maxNodes` grows, and re-assigning the same object is a no-op.
      h.material.uCutMap.value = h.cut.map;

      this.stream(h);
      this.drainPending(h);
      // THE REPLACE FILTER, and it sits here rather than in the scheduler
      // because the fact it turns on — whether a node's children are RESIDENT —
      // is only known at this point in the frame. See `replace.ts`.
      const drawn = this.drawList(h);
      h.sink.setVisible(drawn.indices, drawn.count);
      h.sink.commit();

      h.prevMinSpacing = h.selection.minPointSpacingWorld;
      visibleNodes += h.selection.count;
      visiblePoints += h.selection.points;
      residentNodes += h.sink.nodeCount;
      residentBytes += h.sink.residentBytes;
      loading += h.inFlight.size;
      if (h.sink instanceof ArenaSink) slabs += h.sink.slabCount;
      maxLevel = Math.max(maxLevel, h.selection.maxSelectedLevel);
      if (h.selection.limitedBy !== "complete") limitedBy = h.selection.limitedBy;
      // The WORST region across every cloud, which is what the number means.
      if (h.selection.achievedScreenError > achievedScreenError) {
        achievedScreenError = h.selection.achievedScreenError;
      }
    }

    // THREE RASTERISERS, one draw. The compute path replaces `renderer.render`
    // rather than running beside it: with a compute sink the scene holds an
    // empty group per cloud, so three would only clear what compute just drew.
    //
    // Clear once, let every cloud accumulate into the shared depth and colour
    // buffers — which is what makes clouds occlude each other — then resolve
    // once. EDL lives inside that resolve, so there is no post-process here.
    if (this.raster !== undefined) {
      const enc = this.raster.begin(this.drawSize.x, this.drawSize.y);
      if (enc !== undefined) {
        for (const h of this.clouds) {
          h.object.updateMatrixWorld();
          h.computeSink?.dispatch(
            enc,
            this.camera,
            h.object.matrixWorld,
            this.drawSize.x,
            this.drawSize.y,
          );
        }
        // Anything the app put in the scene, composited INSIDE the resolve
        // pass: mutual occlusion with the points, because the fragment shader
        // reads their depth buffer directly.
        const overlay = this.overlayFor();
        this.raster.end(
          enc,
          overlay === undefined
            ? undefined
            : (pass) =>
                overlay.record(
                  pass,
                  this.camera,
                  this.raster!.depth!,
                  this.drawSize.x,
                  this.drawSize.y,
                ),
        );
      }
    }
    // THE POINTS PATH, same shape as compute: clear once, every cloud draws into
    // the shared depth buffer — which is what makes clouds occlude each other —
    // then one resolve. EDL lives in that resolve.
    else if (this.pointsRaster !== undefined) {
      this.pointsRaster.begin(this.drawSize.x, this.drawSize.y);
      for (const h of this.clouds) {
        if (h.pointsSink === undefined) continue;
        h.object.updateMatrixWorld();
        this.clipFromCloudM
          .multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse)
          .multiply(h.object.matrixWorld);
        this.viewFromCloudM.multiplyMatrices(this.camera.matrixWorldInverse, h.object.matrixWorld);
        h.pointsSink.draw(this.camera, this.clipFromCloudM, this.viewFromCloudM, this.drawSize.y);
      }
      this.pointsRaster.end(this.camera, this.drawSize.x, this.drawSize.y);
    }
    // EDL owns the draw when it is on: it renders the scene into its own target
    // and composites, so calling `renderer.render` as well would draw the frame
    // twice and throw the first one away.
    else if (this.edl !== undefined) this.edl.render();
    else this.renderer.render(this.scene, this.camera);

    this.stats.frameMs = performance.now() - t0;
    this.stats.selectMs = selectMs;
    this.stats.visibleNodes = visibleNodes;
    this.stats.visiblePoints = visiblePoints;
    this.stats.residentNodes = residentNodes;
    this.stats.residentMB = residentBytes / (1024 * 1024);
    this.stats.drawCalls = this.renderer.info.render.drawCalls;
    this.stats.slabs = slabs;
    this.stats.loading = loading;
    this.stats.maxLevel = maxLevel;
    this.stats.limitedBy = limitedBy;
    this.stats.achievedScreenError = achievedScreenError;
    this.dirty = false;
    return true;
  }

  /**
   * The ONE float64 fold: one 4x4 chain and one inverse per cloud per frame.
   *
   * Node boxes are then read verbatim with zero per-node arithmetic. Subtracting
   * the cloud origin from every node box inside the selection loop would be six
   * subtractions across up to 4377 nodes AND would put a rounding decision in
   * the hot path.
   */
  private buildCameraState(h: CloudHandle): void {
    const origin = h.object.cloudOrigin;
    // The cloud's local space is ABS minus its own origin — the frame Task 4
    // emits float32 offsets in.
    const localFromAbs = this.scratchInverse.makeTranslation(
      -origin[0],
      -origin[1],
      -origin[2],
    );

    const m = this.scratchMatrix
      .multiplyMatrices(
        this.camera.projectionMatrix,
        this.camera.matrixWorldInverse,
      )
      .multiply(h.object.matrixWorld)
      .multiply(localFromAbs);
    // three's Matrix4.elements are ordinary JS doubles, so this whole chain is
    // exact even at 635 km.
    h.cam.clipFromAbs.set(m.elements);

    // Camera eye, expressed in ABSOLUTE CRS so it can be differenced against
    // node boxes directly.
    const absFromWorld = this.scratchInverse
      .copy(h.object.matrixWorld)
      .multiply(
        new Matrix4().makeTranslation(-origin[0], -origin[1], -origin[2]),
      )
      .invert();
    this.scratchVec
      .setFromMatrixPosition(this.camera.matrixWorld)
      .applyMatrix4(absFromWorld);
    h.cam.camX = this.scratchVec.x;
    h.cam.camY = this.scratchVec.y;
    h.cam.camZ = this.scratchVec.z;

    h.cam.slope = Math.tan(((this.camera.fov * Math.PI) / 180) / 2);
    // DEVICE pixels, matching what the material reads from viewportSize.y. The
    // reference uses CSS pixels and silently halves every threshold on a 2x
    // display.
    h.cam.viewportHeightPx = this.drawSize.y;
    // NOT `this.camera.near`, which is where this used to come from and which
    // closed a feedback loop through the scheduler:
    //
    //   selection -> minPointSpacingWorld (view.ts, end of renderFrame)
    //             -> suggestNearFar -> camera.near
    //             -> cam.nearFloor
    //             -> the `d` clamp in select.ts, which sets the prune threshold
    //             -> selection
    //
    // `d` is the denominator of the projected-spacing key, so the previous
    // frame's selection decided this frame's prune threshold, with positive
    // gain in both directions: deeper selection lowered the floor and went
    // deeper still, shallower raised it to the 100 ceiling and collapsed. The
    // 2x gate on writing camera.near did not open the loop, it just turned
    // oscillation into a latch. MEASURED on autzen at a fixed camera, sweeping
    // only targetScreenError: 1.35 -> 1.10M points at level 5, 1.0 -> 4.17M at
    // level 6, 0.7 -> 1.25M at level FOUR. A stricter target selecting less
    // detail is not a tuning problem, it is the loop latching.
    //
    // This floor is a property of the CLOUD, so nothing the scheduler decides
    // can feed back into it. It still moves as lazy expansion discovers deeper
    // chunks, but only ever downward, so it converges instead of oscillating —
    // and a caller that ran `expandAll()` has it constant from the first frame.
    h.cam.nearFloor = cloudPitchFloor(h);
    h.cam.depthRange = this.depthRange;

    extractFrustumPlanes(
      h.cam.clipFromAbs,
      h.scratch.planes,
      h.cam.depthRange,
      h.cam.reversedDepth,
    );
  }

  /**
   * The selection, minus every node its children have taken over.
   *
   * Returns the selection's own array untouched for an additive tree, which is
   * every octree format — so the common case costs one undefined check and no
   * copy.
   *
   * NOTE: `h.cut` is built from the per-frame epoch stamps the SELECTION wrote,
   * not from this list, so a node hidden here still contributes to the cut.
   * That is the existing behaviour and left alone deliberately; on a tileset it
   * is worth a look, and on an octree it cannot arise.
   */
  private drawList(h: CloudHandle): { indices: Int32Array; count: number } {
    const tree = h.hierarchy as unknown as ReplaceTreeView;
    if (tree.nodeReplaces === undefined) {
      return { indices: h.selection.indices, count: h.selection.count };
    }
    if (h.visible.length < h.selection.indices.length) {
      h.visible = new Int32Array(h.selection.indices.length);
    }
    const count = filterReplacedParents(
      tree,
      h.selection.indices,
      h.selection.count,
      (i) => h.resident.has(i),
      h.replaceScratch,
      h.visible,
    );
    return { indices: h.visible, count };
  }

  private stream(h: CloudHandle): void {
    const now = this.frame;
    for (let k = 0; k < h.selection.count; k++) {
      h.lastSeen.set(h.selection.indices[k]!, now);
    }

    // Cancel work the camera has moved past, but only when the queue is full.
    //
    // A fetch nothing is competing with is cheaper to finish than to redo: its
    // bytes are already partly paid for, and the node may well be selected
    // again. Under saturation the trade flips — a slot held by a node nobody is
    // looking at is a slot the node on screen is waiting for.
    if (this.abortSuperseded && h.inFlight.size >= this.maxConcurrent) {
      for (const [i, c] of h.inFlight) {
        // STALE, not merely absent this frame.
        //
        // A node that left the selection one frame ago has usually not left at
        // all: with damped controls the camera creeps for a dozen frames after
        // a drag and nodes oscillate across the error threshold, so cancelling
        // on first absence throws away work that is about to be wanted again.
        // Measured, that cost 64 ms of INP — on the one Core Web Vital already
        // behind — to save 0.5% of bytes. Requiring the node to stay gone tells
        // a camera that has genuinely moved on from threshold jitter.
        if (now - (h.lastSeen.get(i) ?? 0) < ABORT_STALE_FRAMES) continue;
        c.abort();
        h.inFlight.delete(i);
      }
    }

    // ONE fetch until the first node lands, then the full width.
    //
    // Dispatch is already in strict priority order, so the node that turns the
    // canvas from black to drawn is always first in this loop — but with twelve
    // fetches in flight it gets a twelfth of the pipe. Measured on autzen over
    // 20 Mbit: the root's own request takes 583 ms at width 12 and 203 ms at
    // width 1, and time-to-first-pixel moves 1224 -> 866 ms.
    //
    // The narrowing is scoped to `resident.size === 0` because holding it for
    // the whole load is a bad trade in the other direction: a permanent width
    // of 1 costs 15.7 s of visual completeness (33.9 -> 49.6 s) and 24 s of
    // stream time. One node serialised, everything after it parallel.
    const width = h.resident.size === 0 ? 1 : this.maxConcurrent;
    for (let k = 0; k < h.selection.count; k++) {
      if (h.inFlight.size >= width) break;
      const i = h.selection.indices[k]!;
      if (
        h.resident.has(i) ||
        h.inFlight.has(i) ||
        h.queued.has(i) ||
        h.failed.has(i)
      ) {
        continue;
      }
      // Where the seam used to be. `hasPayload` is the driver's answer to
      // "does this node have bytes at all", which is a real question with three
      // different answers: 47 of autzen's nodes carry no payload of their own,
      // a COPC placeholder has none until its hierarchy page lands, and an EPT
      // node always does. Asking beats discovering it in a catch.
      const node: PointCloudNode | undefined = h.hierarchy.node(i);
      if (node === undefined || node.numPoints === 0) continue;
      if (!h.reader.hasPayload(node)) continue;

      const controller = new AbortController();
      h.inFlight.set(i, controller);
      void h.reader
        .read(node, { signal: controller.signal, computeBounds: true })
        .then((data) => {
          h.inFlight.delete(i);
          if (this.disposed) return;
          // Queued, not staged: see CloudHandle.pending.
          h.pending.push({ index: i, data, level: node.level });
          h.queued.add(i);
          this.dirty = true;
        })
        .catch((err: unknown) => {
          h.inFlight.delete(i);
          // An abort is US cancelling, not the node being unreadable, and it
          // must never reach `failed` — that set is terminal by design, so
          // blacklisting a node the camera merely panned away from would leave
          // a permanent hole the moment it panned back.
          if (err instanceof Error && err.name === "AbortError") return;
          // A real failure is terminal until something clears it — never a
          // per-frame retry, which is exactly the reference's storm.
          h.failed.add(i);
          if (this.stats.lastError === undefined) {
            this.stats.lastError = isVoxelkloudError(err)
              ? `${err.code}: ${err.message}`
              : String(err);
          }
        });
    }

    if (h.sink.residentBytes > this.maxResidentBytes) this.evict(h);
  }

  /**
   * Stage queued nodes into the sink under a per-frame byte budget.
   *
   * Ordered by the CURRENT selection: a node the camera still wants goes in
   * before one it has already moved past, so a pan that outruns streaming
   * spends its upload budget on what is actually on screen rather than on a
   * backlog nobody is looking at.
   */
  private drainPending(h: CloudHandle): void {
    if (h.pending.length === 0) return;
    const frame = this.frame;
    for (let k = 0; k < h.selection.count; k++) {
      h.lastSeen.set(h.selection.indices[k]!, frame);
    }
    // Most recently selected first; anything not in this frame's selection
    // sorts last.
    h.pending.sort(
      (a, b) => (h.lastSeen.get(b.index) ?? 0) - (h.lastSeen.get(a.index) ?? 0),
    );

    let staged = 0;
    let n = 0;
    while (n < h.pending.length && staged < this.maxAttachBytes) {
      const p = h.pending[n]!;
      n++;
      h.queued.delete(p.index);
      const pitch = pitchOf(h.hierarchy, p.index, p.level);
      // The floor moves DOWNWARD here too, not only as the hierarchy grows: a
      // driver that measures a tile's pitch from its decoded points can only
      // learn it at this moment.
      if (pitch < h.minPitchWorld) h.minPitchWorld = pitch;
      const bytes = h.sink.attach(p.index, p.data, pitch, p.level);
      if (bytes > 0) {
        h.resident.add(p.index);
        staged += bytes;
      } else if (p.data.numPoints > 0) {
        // A refusal used to be doubly silent: nothing drew, and the node was
        // still counted resident, so the HUD agreed everything was fine.
        h.failed.add(p.index);
        this.stats.lastError ??=
          `sink refused ${p.data.nodeName} (${p.data.numPoints} points)`;
      }
    }
    h.pending.splice(0, n);
    this.stats.residentPoints = h.sink.residentPoints;
    this.stats.pendingAttach = h.pending.length;
    if (h.pending.length > 0) this.dirty = true;
  }

  /** Drop the least-recently-selected resident nodes, never the root. */
  private evict(h: CloudHandle): void {
    const candidates: Array<[number, number]> = [];
    for (const i of h.resident) {
      if (i === h.hierarchy.root.index) continue;
      candidates.push([i, h.lastSeen.get(i) ?? 0]);
    }
    candidates.sort((a, b) => a[1] - b[1]);
    const target = this.maxResidentBytes * 0.85;
    for (const [i] of candidates) {
      if (h.sink.residentBytes <= target) break;
      h.sink.detach(i);
      h.resident.delete(i);
    }
  }

  dispose(): void {
    this.raster?.dispose();
    this.raster = undefined;
    this.pointsRaster?.dispose();
    this.pointsRaster = undefined;
    this.overlay?.dispose();
    this.overlay = undefined;
    if (this.disposed) return;
    this.disposed = true;
    for (const h of this.clouds) {
      for (const [, c] of h.inFlight) c.abort();
      h.inFlight.clear();
      h.pending.length = 0;
      h.queued.clear();
      h.sink.dispose();
      h.cut.dispose();
      h.computeSink?.dispose();
      h.material.dispose();
      this.scene.remove(h.object);
    }
    this.clouds.length = 0;
    this.edl?.dispose();
    this.edl = undefined;
    this.renderer.dispose();
  }
}

/**
 * Frames a node must stay out of the selection before its in-flight fetch is
 * cancelled. Eight is ~130 ms at 60 Hz — longer than damping's settle, shorter
 * than any deliberate camera move.
 */
const ABORT_STALE_FRAMES = 8;

export function createPointCloudView(
  options: PointCloudViewOptions,
): PointCloudView {
  return new PointCloudView(options);
}
