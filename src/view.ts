import type {
  DecodedPointData,
  PointCloudHierarchy,
  PointCloudSource,
  PointRecordLayout,
} from "@voxelkloud/loader";
import {
  VoxelkloudError,
  createPointLayout,
  isVoxelkloudError,
  loadPointData,
} from "@voxelkloud/loader";
import type { BrotliDecompress } from "@voxelkloud/loader";
import {
  Matrix4,
  PerspectiveCamera,
  Scene,
  Vector2,
  Vector3,
  WebGPURenderer,
} from "three/webgpu";
import { createPointMaterial, scalarAttributeFor } from "./material.js";
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
import { extractProfile as extractProfileFromCloud } from "./profile/index.js";
import type {
  ProfileBatch,
  ProfileExtractionOptions,
  ProfileQuery,
} from "./profile/index.js";
import { ArenaSink } from "./sink-arena.js";
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
  source: PointCloudSource,
): [number, number] {
  const originZ = source.metadata.boundingBox.min[2];
  const tight = source.tightBoundingBox;
  return [tight.min[2] - originZ, tight.max[2] - originZ];
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
 * The transform is read off the LAYOUT rather than recomputed, so this cannot
 * drift from what the decoder did.
 */
export function scalarRangeFor(
  source: PointCloudSource,
  layout: PointRecordLayout,
  name: string,
): [number, number] {
  const attribute = source.attributesByName.get(name);
  const lo = attribute?.min[0] ?? 0;
  const hi = attribute?.max[0] ?? 1;
  const pack = layout.fields.find((f) => f.name === name)?.pack;
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
  pendingAttach: number;
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
   * A brotli decompressor, for BROTLI-encoded clouds in an environment without
   * a native one — which is every current browser.
   *
   * ```ts
   * const { brotliDecompress } = await import("@voxelkloud/loader/brotli");
   * ```
   */
  readonly decompress?: BrotliDecompress;
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
  readonly sinkMode?: "arena" | "per-node";
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
  readonly source: PointCloudSource;
  readonly hierarchy: PointCloudHierarchy;
  readonly object: PointCloudObject3D;
  readonly material: PointCloudMaterial;
  readonly sink: PointSink & { residentBytes: number; nodeCount: number };
  readonly layout: PointRecordLayout;
  readonly scratch: LodScratch;
  readonly selection: LodSelection;
  readonly cam: LodCameraState;
  /** index -> last frame it was selected. Drives eviction. */
  readonly lastSeen: Map<number, number>;
  readonly resident: Set<number>;
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
    pendingAttach: 0,
    loading: 0,
    maxLevel: 0,
    lastError: undefined,
    limitedBy: "complete",
  };

  private readonly clouds: CloudHandle[] = [];
  private readonly lodOptions: { -readonly [K in keyof ResolvedLodOptions]: ResolvedLodOptions[K] };
  private readonly materialOptions;
  private readonly maxConcurrent: number;
  private readonly maxResidentBytes: number;
  private readonly maxAttachBytes: number;
  private readonly scratchMatrix = new Matrix4();
  private readonly scratchInverse = new Matrix4();
  private readonly scratchVec = new Vector3();
  private readonly drawSize = new Vector2();
  private depthRange: DepthRange = "minus-one-to-one";
  private frame = 0;
  private initialized = false;
  private disposed = false;
  private dirty = true;
  private edl: EdlPipeline | undefined;

  constructor(private readonly options: PointCloudViewOptions) {
    this.camera = new PerspectiveCamera(60, 1, 1, 20_000);
    // Potree v2 data is Z-up CRS; three defaults to Y-up, which puts every
    // terrain cloud on its side.
    this.camera.up.set(0, 0, 1);
    this.renderer = new WebGPURenderer({
      canvas: options.canvas,
      antialias: false,
    });
    // NOT optional. Without it the model-view product is computed in float32 IN
    // THE SHADER, and cloud-relative positions come out WORSE than absolute
    // ones — a measured 34.1 px of jitter against 29.2 px. With it, 0.46 px.
    this.renderer.highPrecision = true;
    this.lodOptions = resolveLodOptions(options.lod);
    this.materialOptions = options.material ?? {};
    this.maxConcurrent = options.maxConcurrentLoads ?? 12;
    this.maxResidentBytes = options.maxResidentBytes ?? 512 * 1024 * 1024;
    this.maxAttachBytes = options.maxAttachBytesPerFrame ?? 8 * 1024 * 1024;
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
    if (this.options.edl !== undefined) {
      this.edl = createEdlPipeline(
        this.renderer,
        this.scene,
        this.camera,
        this.options.edl,
      );
    }
    this.initialized = true;
    this.dirty = true;
  }

  /** Add a cloud. `hierarchy` must already have its root expanded. */
  addCloud(
    source: PointCloudSource,
    hierarchy: PointCloudHierarchy,
  ): PointCloudObject3D {
    const origin = source.metadata.boundingBox.min;
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
    const layout = createPointLayout(source, {
      computeBounds: true,
      ...(scalarAttribute !== undefined
        ? {
            attributes: [scalarAttribute],
            scalarFormat: "gpu" as const,
            lanes: { [scalarAttribute]: "f32" as const },
          }
        : {}),
    });

    const material = createPointMaterial({
      // CLOUD-RELATIVE, because that is the frame `pointOffset` is in. The tight
      // bounds are absolute CRS, and feeding those straight in subtracts an
      // absolute elevation from a relative one: on autzen that is (0 - 406) /
      // 209, clamped to 0 for every point, so the whole cloud renders as the
      // ramp's first stop. It only shows on a cloud with no RGB, which is why it
      // survived until a 3DEP survey turned up.
      elevationRange: cloudRelativeElevationRange(source),
      ...(scalarAttribute !== undefined
        ? { scalarRange: scalarRangeFor(source, layout, scalarAttribute) }
        : {}),
      ...this.materialOptions,
    });
    // Task 4 writes the SOURCE alpha when the colour attribute has four
    // elements, so whether the arena must stamp it is a per-cloud fact, decided
    // here rather than sniffed per node.
    const color = source.attributes.find((a) => a.role === "color");
    const needsAlphaStamp = (color?.numElements ?? 3) >= 4;
    const sink =
      (this.options.sinkMode ?? "arena") === "per-node"
        ? new PerNodeSink(object, material, scalarAttribute)
        : new ArenaSink(
            object,
            material,
            (l) => hierarchy.spacingAt(l),
            (l) => hierarchy.radiusAt(l),
            needsAlphaStamp,
            {},
            scalarAttribute,
          );

    this.scene.add(object);
    this.clouds.push({
      source,
      hierarchy,
      object,
      material,
      sink,
      layout,
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
      inFlight: new Map(),
      failed: new Set(),
      pending: [],
      queued: new Set(),
      prevMinSpacing: source.metadata.spacing,
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
    const { cloudIndex = 0, decompress, ...profileOptions } = options;
    const h = this.clouds[cloudIndex];
    if (h === undefined) {
      throw new RangeError(`No point cloud exists at index ${cloudIndex}.`);
    }
    return extractProfileFromCloud(
      {
        source: h.source,
        hierarchy: h.hierarchy,
        readPoints: (index: number) => h.sink.readPoints(index),
      },
      query,
      {
        ...profileOptions,
        ...(decompress !== undefined
          ? { decompress }
          : this.options.decompress !== undefined
            ? { decompress: this.options.decompress }
            : {}),
      },
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
   * pixels of inter-point spacing.
   *
   * Lower refines further. Worth turning down when `stats.limitedBy` reports
   * `"error"` while frame time is flat: that combination means the target was
   * met with budget and frame both to spare.
   */
  setTargetPixelSpacing(px: number): void {
    this.lodOptions.targetPixelSpacing = px;
    this.dirty = true;
  }

  /**
   * Points drawn per frame.
   *
   * Raise this BEFORE lowering `targetPixelSpacing`: measured on autzen at a
   * 0.25 framing, the 3M default binds at a target of 1.0 and below, so the
   * quality knob does nothing until the budget has room. `stats.limitedBy`
   * reports which of the two is actually holding.
   */
  setPointBudget(points: number): void {
    this.lodOptions.pointBudget = points;
    this.dirty = true;
  }

  renderFrame(): boolean {
    if (!this.initialized || this.disposed) return false;
    const t0 = performance.now();
    this.frame++;

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
      selectVisible(h.hierarchy, h.cam, this.lodOptions, h.scratch, h.selection);
      selectMs += performance.now() - ts;
      resolveExpansions(h.hierarchy, h.selection);

      this.stream(h);
      this.drainPending(h);
      h.sink.setVisible(h.selection.indices, h.selection.count);
      h.sink.commit();

      h.prevMinSpacing = h.selection.minSpacingWorld;
      visibleNodes += h.selection.count;
      visiblePoints += h.selection.points;
      residentNodes += h.sink.nodeCount;
      residentBytes += h.sink.residentBytes;
      loading += h.inFlight.size;
      if (h.sink instanceof ArenaSink) slabs += h.sink.slabCount;
      maxLevel = Math.max(maxLevel, h.selection.maxSelectedLevel);
      if (h.selection.limitedBy !== "complete") limitedBy = h.selection.limitedBy;
    }

    // EDL owns the draw when it is on: it renders the scene into its own target
    // and composites, so calling `renderer.render` as well would draw the frame
    // twice and throw the first one away.
    if (this.edl !== undefined) this.edl.render();
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
    //   selection -> minSpacingWorld (view.ts, end of renderFrame)
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
    // only targetPixelSpacing: 1.35 -> 1.10M points at level 5, 1.0 -> 4.17M at
    // level 6, 0.7 -> 1.25M at level FOUR. A stricter target selecting less
    // detail is not a tuning problem, it is the loop latching.
    //
    // This floor is a property of the CLOUD, so nothing the scheduler decides
    // can feed back into it. It still moves as lazy expansion discovers deeper
    // chunks, but only ever downward, so it converges instead of oscillating —
    // and a caller that ran `expandAll()` has it constant from the first frame.
    h.cam.nearFloor = h.hierarchy.spacingAt(h.hierarchy.maxLevel);
    h.cam.depthRange = this.depthRange;

    extractFrustumPlanes(
      h.cam.clipFromAbs,
      h.scratch.planes,
      h.cam.depthRange,
      h.cam.reversedDepth,
    );
  }

  private stream(h: CloudHandle): void {
    const now = this.frame;
    for (let k = 0; k < h.selection.count; k++) {
      h.lastSeen.set(h.selection.indices[k]!, now);
    }

    // Dispatch in selection order, which is already strictly descending
    // priority, so a camera pan re-prioritises pending work for free.
    for (let k = 0; k < h.selection.count; k++) {
      if (h.inFlight.size >= this.maxConcurrent) break;
      const i = h.selection.indices[k]!;
      if (
        h.resident.has(i) ||
        h.inFlight.has(i) ||
        h.queued.has(i) ||
        h.failed.has(i)
      ) {
        continue;
      }
      const node = h.hierarchy.node(i);
      if (node === undefined || node.numPoints === 0) continue;
      if (node.byteSize === undefined) continue;

      const controller = new AbortController();
      h.inFlight.set(i, controller);
      void loadPointData(
        h.source,
        node,
        {
          signal: controller.signal,
          computeBounds: true,
          decompress: this.options.decompress,
        },
        h.layout,
      )
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
          // A failure is terminal for this node until something clears it —
          // never a per-frame retry, which is exactly the reference's storm.
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
      const bytes = h.sink.attach(
        p.index,
        p.data,
        h.hierarchy.spacingAt(p.level),
        p.level,
      );
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
    if (this.disposed) return;
    this.disposed = true;
    for (const h of this.clouds) {
      for (const [, c] of h.inFlight) c.abort();
      h.inFlight.clear();
      h.pending.length = 0;
      h.queued.clear();
      h.sink.dispose();
      h.material.dispose();
      this.scene.remove(h.object);
    }
    this.clouds.length = 0;
    this.edl?.dispose();
    this.edl = undefined;
    this.renderer.dispose();
  }
}

export function createPointCloudView(
  options: PointCloudViewOptions,
): PointCloudView {
  return new PointCloudView(options);
}
