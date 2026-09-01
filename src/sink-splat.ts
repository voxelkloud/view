import type { DecodedPointData } from "@voxelkloud/format-potree";
import type { Matrix4, PerspectiveCamera } from "three";
import { toZeroToOneDepth } from "./clip.js";
import {
  applySortOrder,
  buildSplatNodeGeometry,
  cameraPositionSignature,
  gatherVisibleGeometry,
  hasGaussianAttributes,
  sortBackToFront,
  type SplatNodeGeometry,
} from "./sink-splat-geometry.js";
import { SPLAT_WGSL } from "./splat-wgsl.js";
import type { PointReadback, PointSink } from "./sink.js";

export interface SplatSinkOptions {
  /**
   * Multiplies each Gaussian's world-space scale before it becomes a
   * billboard half-extent — the same knob `GaussianPocViewer.tsx` exposes as
   * `?gauss=`/`?kernel=`. 2 is that prototype's default.
   */
  readonly scaleFactor?: number;
  /** Screen-space alpha floor below which a resort is not worth its cost.
   * Rounds the camera eye to this many units per meter before comparing —
   * see `cameraPositionSignature`. */
  readonly resortGridPerMeter?: number;
}

const CORNER_VERTICES = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
const UNIFORM_BYTES = 80; // mat4x4<f32> (64) + screenW/screenH/ignoreDepth/_pad (16)

/**
 * A {@link PointSink} that draws real alpha-blended Gaussian splats — the
 * production home for the rasterizer prototyped in
 * `demo/app/src/GaussianSplatRasterizer.tsx`. See §Chunk 1 Step 4 of
 * `docs/superpowers/plans/2026-08-31-live-gaussian-splat-capture.md`.
 *
 * DELIBERATELY NOT a `BlockAllocator`-backed GPU arena like `ComputeSink`.
 * That trade only pays for itself at LiDAR-survey scale (tens of millions of
 * points, streamed and evicted continuously); a live splat capture is one
 * flat set in the hundreds of thousands. This keeps one small typed-array
 * bundle per attached node (mirroring the already-proven
 * `GaussianPocViewer.tsx` prototype's `Map<nodeIndex, geometry>`) and
 * re-gathers + depth-sorts the VISIBLE subset into upload buffers only when
 * the camera has moved enough to change the draw order — see
 * `cameraPositionSignature`. Revisit if a capture ever approaches arena
 * scale.
 *
 * Composition with the point-cloud renderer: `draw()` is meant to be called
 * from inside `ComputeRasterizer.end()`'s second render pass, the same slot
 * `OverlayRenderer.record()` uses and for the same reason — the resolve pass
 * binds the point depth as read-write storage (atomics), and a render pass
 * cannot also read it, so anything that needs to READ that depth (to occlude
 * against opaque points) needs its own pass. `draw()`'s signature mirrors
 * `OverlayRenderer.record()`'s for exactly that reason.
 *
 * NOT wired into `PointCloudView`'s cloud-loading pipeline yet — no format
 * or dataset-detection code picks this sink automatically for a Gaussian
 * COPC. That selection is real remaining work, left for whoever wires a
 * trained `.ply` through to the live viewer (Chunk 3).
 */
export class SplatSink implements PointSink {
  private readonly blocks = new Map<number, SplatNodeGeometry>();
  private readonly scaleFactor: number;
  private readonly resortGridPerMeter: number;

  private visibleIndices: readonly number[] = [];
  private lastSortSignature = "";

  private readonly pipeline: GPURenderPipeline;
  private readonly uniformBuffer: GPUBuffer;
  private readonly cornerBuffer: GPUBuffer;
  private centerBuffer: GPUBuffer;
  private axisUBuffer: GPUBuffer;
  private axisVBuffer: GPUBuffer;
  private colorBuffer: GPUBuffer;
  private opacityBuffer: GPUBuffer;
  private drawCapacity = 0;
  private drawCount = 0;

  private bindGroup: GPUBindGroup | undefined;
  private boundDepth: GPUBuffer | undefined;
  private bytesResident = 0;
  private disposed = false;

  constructor(
    private readonly device: GPUDevice,
    format: GPUTextureFormat,
    options: SplatSinkOptions = {},
  ) {
    this.scaleFactor = options.scaleFactor ?? 2;
    this.resortGridPerMeter = options.resortGridPerMeter ?? 40;

    const module = device.createShaderModule({ code: SPLAT_WGSL, label: "voxelkloud-splat" });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: 8,
            stepMode: "vertex",
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
          },
          {
            arrayStride: 12,
            stepMode: "instance",
            attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }],
          },
          {
            arrayStride: 12,
            stepMode: "instance",
            attributes: [{ shaderLocation: 2, offset: 0, format: "float32x3" }],
          },
          {
            arrayStride: 12,
            stepMode: "instance",
            attributes: [{ shaderLocation: 3, offset: 0, format: "float32x3" }],
          },
          {
            arrayStride: 4,
            stepMode: "instance",
            attributes: [{ shaderLocation: 4, offset: 0, format: "unorm8x4" }],
          },
          {
            arrayStride: 4,
            stepMode: "instance",
            attributes: [{ shaderLocation: 5, offset: 0, format: "float32" }],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
      // No depthStencil here: this pass tests against the POINT CLOUD's depth
      // via the `pointDepth` storage binding read in the fragment shader
      // (same technique `overlay.ts`'s `WGSL` const uses), not a bound depth
      // attachment. It never writes depth either way — splats are drawn
      // back-to-front and blended, so nothing downstream should occlude
      // against a single splat's depth.
    });

    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.cornerBuffer = device.createBuffer({
      size: CORNER_VERTICES.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.cornerBuffer, 0, CORNER_VERTICES);

    this.centerBuffer = this.instanceBuffer(12);
    this.axisUBuffer = this.instanceBuffer(12);
    this.axisVBuffer = this.instanceBuffer(12);
    this.colorBuffer = this.instanceBuffer(4);
    this.opacityBuffer = this.instanceBuffer(4);
  }

  private instanceBuffer(strideBytes: number): GPUBuffer {
    return this.device.createBuffer({
      size: Math.max(4, strideBytes),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }

  // ---- PointSink ------------------------------------------------------

  attach(index: number, data: DecodedPointData, _spacingWorld: number, _level: number): number {
    if (this.disposed) return 0;
    if (this.blocks.has(index)) return 0;
    if (data.numPoints === 0) return 0;
    if (!hasGaussianAttributes(data)) return 0;

    const geometry = buildSplatNodeGeometry(data, this.scaleFactor);
    this.blocks.set(index, geometry);
    this.bytesResident += geometry.bytes;
    return geometry.bytes;
  }

  detach(index: number): void {
    const geometry = this.blocks.get(index);
    if (geometry === undefined) return;
    this.bytesResident -= geometry.bytes;
    this.blocks.delete(index);
  }

  get residentPoints(): number {
    let total = 0;
    for (const g of this.blocks.values()) total += g.count;
    return total;
  }

  get residentBytes(): number {
    return this.bytesResident;
  }

  get nodeCount(): number {
    return this.blocks.size;
  }

  setVisible(indices: Int32Array, count: number): void {
    this.visibleIndices = Array.from(indices.subarray(0, count));
  }

  /** Nothing to flush: geometry is built once at `attach` time (CPU-only),
   * and the GPU upload happens in `draw`, gated on the camera actually
   * having moved — same reasoning as `PerNodeSink.commit`. */
  commit(): void {}

  readPoints(index: number): PointReadback | undefined {
    const geometry = this.blocks.get(index);
    if (geometry === undefined) return undefined;
    return {
      positions: geometry.centers,
      start: 0,
      count: geometry.count,
      colors: geometry.colors,
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.blocks.clear();
    this.uniformBuffer.destroy();
    this.cornerBuffer.destroy();
    this.centerBuffer.destroy();
    this.axisUBuffer.destroy();
    this.axisVBuffer.destroy();
    this.colorBuffer.destroy();
    this.opacityBuffer.destroy();
  }

  // ---- draw -------------------------------------------------------------

  /**
   * Draw the currently-visible splats into `pass`, occluding against the
   * point cloud's own depth (`pointDepth`, from `ComputeRasterizer.depth`).
   * Called from inside `ComputeRasterizer.end()`'s second pass — the same
   * seam `OverlayRenderer.record()` uses, and this method's signature
   * mirrors it deliberately.
   */
  draw(
    pass: GPURenderPassEncoder,
    camera: PerspectiveCamera,
    modelMatrix: Matrix4,
    pointDepth: GPUBuffer,
    width: number,
    height: number,
  ): void {
    if (this.disposed) return;
    this.refreshSortedUpload(camera, modelMatrix);
    if (this.drawCount === 0) return;

    if (this.bindGroup === undefined || this.boundDepth !== pointDepth) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: pointDepth } },
        ],
      });
      this.boundDepth = pointDepth;
    }

    const viewProj = toZeroToOneDepth(
      camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse).multiply(modelMatrix),
    );
    const uniform = new ArrayBuffer(UNIFORM_BYTES);
    const f = new Float32Array(uniform);
    f.set(viewProj.elements as unknown as ArrayLike<number>, 0);
    f[16] = width;
    f[17] = height;
    f[18] = 0; // ignoreDepth
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniform);

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.cornerBuffer);
    pass.setVertexBuffer(1, this.centerBuffer);
    pass.setVertexBuffer(2, this.axisUBuffer);
    pass.setVertexBuffer(3, this.axisVBuffer);
    pass.setVertexBuffer(4, this.colorBuffer);
    pass.setVertexBuffer(5, this.opacityBuffer);
    pass.draw(6, this.drawCount);
  }

  private refreshSortedUpload(camera: PerspectiveCamera, modelMatrix: Matrix4): void {
    const eye: [number, number, number] = [
      camera.matrixWorld.elements[12]!,
      camera.matrixWorld.elements[13]!,
      camera.matrixWorld.elements[14]!,
    ];
    const signature = `${cameraPositionSignature(eye, this.resortGridPerMeter)}|${this.visibleIndices.join(".")}`;
    if (signature === this.lastSortSignature) return;
    this.lastSortSignature = signature;

    const visibleBlocks = this.visibleIndices
      .map((index) => this.blocks.get(index))
      .filter((g): g is SplatNodeGeometry => g !== undefined)
      .map((geometry) => ({ geometry }));
    const gathered = gatherVisibleGeometry(visibleBlocks);
    if (gathered.count === 0) {
      this.drawCount = 0;
      return;
    }

    // Sort in MODEL space (cloud-local, what `centers` already holds) using
    // the eye transformed into that same space — cheaper than transforming
    // every centre into world space just to sort them.
    const inverseModel = modelMatrix.clone().invert();
    const eyeLocal = eye.slice() as [number, number, number];
    applyMatrix4InPlace(inverseModel.elements as unknown as ArrayLike<number>, eyeLocal);
    const forward = normalize([-eyeLocal[0], -eyeLocal[1], -eyeLocal[2]]);
    const order = sortBackToFront(gathered.centers, gathered.count, eyeLocal, forward);
    const sorted = applySortOrder(gathered, order);

    if (sorted.count > this.drawCapacity) {
      this.centerBuffer.destroy();
      this.axisUBuffer.destroy();
      this.axisVBuffer.destroy();
      this.colorBuffer.destroy();
      this.opacityBuffer.destroy();
      this.centerBuffer = this.instanceBuffer(sorted.count * 12);
      this.axisUBuffer = this.instanceBuffer(sorted.count * 12);
      this.axisVBuffer = this.instanceBuffer(sorted.count * 12);
      this.colorBuffer = this.instanceBuffer(sorted.count * 4);
      this.opacityBuffer = this.instanceBuffer(sorted.count * 4);
      this.drawCapacity = sorted.count;
    }

    const q = this.device.queue;
    q.writeBuffer(this.centerBuffer, 0, sorted.centers as unknown as BufferSource);
    q.writeBuffer(this.axisUBuffer, 0, sorted.axisU as unknown as BufferSource);
    q.writeBuffer(this.axisVBuffer, 0, sorted.axisV as unknown as BufferSource);
    q.writeBuffer(this.colorBuffer, 0, sorted.colors as unknown as BufferSource);
    q.writeBuffer(this.opacityBuffer, 0, sorted.opacities as unknown as BufferSource);
    this.drawCount = sorted.count;
  }
}

function applyMatrix4InPlace(e: ArrayLike<number>, v: [number, number, number]): void {
  const [x, y, z] = v;
  v[0] = e[0]! * x + e[4]! * y + e[8]! * z + e[12]!;
  v[1] = e[1]! * x + e[5]! * y + e[9]! * z + e[13]!;
  v[2] = e[2]! * x + e[6]! * y + e[10]! * z + e[14]!;
}

function normalize(v: readonly [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}
