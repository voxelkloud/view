import type { Camera, Object3D } from "three/webgpu";

/**
 * Meshes drawn alongside the points, inside the compute path's own resolve
 * pass — a vehicle, a measurement gizmo, a model to compare against.
 *
 * WHY THIS EXISTS AT ALL. On the compute path the view resolves points
 * straight to the swapchain and never calls `renderer.render(scene)`, so
 * anything added to `view.scene` was invisible: an API you could add to and
 * never see. Rendering three's scene as a second pass afterwards does not
 * work either — `WebGPURenderer` owns its targets and overwrites the
 * swapchain rather than compositing with a raw-device write in the same
 * frame. That was tried and reverted; this is the version that composes.
 *
 * HOW OCCLUSION WORKS. The points' depth lives in a storage buffer as the
 * IEEE bit pattern of the EYE depth (see compute-wgsl), not in a depth
 * attachment. So each fragment here recomputes its own eye depth, reads the
 * point depth at its pixel, and discards when a point is nearer. Meshes
 * occlude each other through a real depth attachment. The result is mutual
 * occlusion between geometry and points with no depth copy anywhere.
 */

const WGSL = /* wgsl */ `
struct U {
  mvp        : mat4x4<f32>,
  color      : vec4<f32>,
  screenW    : f32,
  screenH    : f32,
  hasTexture : f32,
  _pad       : f32,
};

@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> pointDepth : array<u32>;
@group(1) @binding(0) var samp : sampler;
@group(1) @binding(1) var tex  : texture_2d<f32>;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  // The clip w IS the eye depth, which is what the point pass stored.
  @location(0) eyeDepth : f32,
  @location(1) uv : vec2<f32>,
};

@vertex
fn vs(@location(0) position : vec3<f32>, @location(1) uv : vec2<f32>) -> VSOut {
  var out : VSOut;
  let clip = u.mvp * vec4<f32>(position, 1.0);
  out.pos = clip;
  out.eyeDepth = clip.w;
  out.uv = uv;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let x = u32(clamp(in.pos.x, 0.0, u.screenW - 1.0));
  let y = u32(clamp(in.pos.y, 0.0, u.screenH - 1.0));
  let stored = pointDepth[y * u32(u.screenW) + x];
  // Both are bit patterns of non-negative floats, which are monotonic in the
  // value — so this compares depths without decoding either.
  if (bitcast<u32>(in.eyeDepth) > stored) {
    discard;
  }
  if (u.hasTexture > 0.5) {
    let t = textureSample(tex, samp, in.uv);
    return vec4<f32>(t.rgb * u.color.rgb, t.a * u.color.a);
  }
  return u.color;
}
`;

/** 256 is the minimum dynamic-offset alignment WebGPU guarantees. */
const UNIFORM_STRIDE = 256;

interface CachedGeometry {
  vertex: GPUBuffer;
  /** UVs, zero-filled when the geometry has none, so one pipeline serves all. */
  uv: GPUBuffer;
  index: GPUBuffer | undefined;
  indexFormat: GPUIndexFormat;
  version: number;
}

interface DrawItem {
  geometryId: number;
  matrix: Float32Array;
  color: [number, number, number, number];
  start: number;
  count: number;
  indexed: boolean;
  transparent: boolean;
  /** A three texture's image, or undefined for a flat-coloured draw. */
  image: TexSource | undefined;
  imageId: number;
}

type TexSource = ImageBitmap | HTMLImageElement | HTMLCanvasElement;

type ThreeMesh = Object3D & {
  isMesh?: boolean;
  visible: boolean;
  matrixWorld: { elements: ArrayLike<number> };
  geometry?: {
    id: number;
    attributes: {
      position?: { array: ArrayLike<number>; itemSize: number; count: number; version: number };
      uv?: { array: ArrayLike<number>; itemSize: number; count: number };
    };
    index?: { array: ArrayLike<number>; count: number } | null;
    groups: { start: number; count: number; materialIndex?: number }[];
    drawRange: { start: number; count: number };
  };
  material?: unknown;
};

const colorOf = (material: unknown): [number, number, number, number] => {
  const m = material as
    | { color?: { r: number; g: number; b: number }; opacity?: number; transparent?: boolean }
    | undefined;
  const c = m?.color;
  const a = m?.transparent === true ? (m.opacity ?? 1) : 1;
  return c === undefined ? [1, 1, 1, a] : [c.r, c.g, c.b, a];
};

const isTransparent = (material: unknown): boolean =>
  (material as { transparent?: boolean } | undefined)?.transparent === true;

/** three's `material.map`, if it carries a decoded image we can upload. */
function textureOf(material: unknown): { image: TexSource; id: number } | undefined {
  const map = (material as { map?: { image?: unknown; id?: number } } | undefined)?.map;
  const img = map?.image;
  if (img === undefined || img === null) return undefined;
  const ok =
    (typeof ImageBitmap !== "undefined" && img instanceof ImageBitmap) ||
    (typeof HTMLImageElement !== "undefined" && img instanceof HTMLImageElement) ||
    (typeof HTMLCanvasElement !== "undefined" && img instanceof HTMLCanvasElement);
  return ok ? { image: img as TexSource, id: map!.id ?? 0 } : undefined;
}

export class OverlayRenderer {
  private readonly device: GPUDevice;
  private pipeline: GPURenderPipeline | undefined;
  private layout: GPUBindGroupLayout | undefined;
  private uniform: GPUBuffer | undefined;
  private uniformCapacity = 0;
  private bindGroup: GPUBindGroup | undefined;
  private boundDepth: GPUBuffer | undefined;
  private readonly geometries = new Map<number, CachedGeometry>();
  private readonly textures = new Map<number, { texture: GPUTexture; bind: GPUBindGroup }>();
  private texLayout: GPUBindGroupLayout | undefined;
  private sampler: GPUSampler | undefined;
  /** Bound when a draw has no texture, so one pipeline serves both cases. */
  private blankBind: GPUBindGroup | undefined;
  private readonly scratch = new Float32Array(UNIFORM_STRIDE / 4);
  private draws: DrawItem[] = [];
  private disposed = false;

  constructor(
    device: GPUDevice,
    private readonly format: GPUTextureFormat,
    private readonly depthFormat: GPUTextureFormat,
  ) {
    this.device = device;
  }

  /**
   * Collect drawable meshes from a scene graph, skipping `exclude` (the point
   * clouds' own groups, which hold no drawable geometry on this path).
   *
   * Returns false when there is nothing to draw, so the caller can skip
   * attaching a depth buffer and binding a pipeline at all.
   */
  collect(root: Object3D, exclude: ReadonlySet<object>): boolean {
    this.draws.length = 0;
    const walk = (node: Object3D): void => {
      if (exclude.has(node as unknown as object)) return;
      const mesh = node as ThreeMesh;
      if (mesh.visible === false) return;
      if (mesh.isMesh === true && mesh.geometry !== undefined) this.addMesh(mesh);
      for (const child of node.children) walk(child);
    };
    for (const child of root.children) walk(child);
    // Transparent last: no sorting beyond that, which is enough for gizmos and
    // wrong only for transparent surfaces that overlap each other.
    this.draws.sort((a, b) => Number(a.transparent) - Number(b.transparent));
    return this.draws.length > 0;
  }

  private addMesh(mesh: ThreeMesh): void {
    const geo = mesh.geometry!;
    const position = geo.attributes.position;
    if (position === undefined || position.itemSize !== 3) return;
    this.upload(geo);

    const matrix = new Float32Array(mesh.matrixWorld.elements as ArrayLike<number>);
    const indexed = geo.index != null;
    const total = indexed ? geo.index!.count : position.count;
    const materials = Array.isArray(mesh.material) ? mesh.material : null;

    // A geometry with groups and a material array is one draw per group —
    // which is how a six-faced box gets six colours.
    if (materials !== null && geo.groups.length > 0) {
      for (const g of geo.groups) {
        const material = materials[g.materialIndex ?? 0];
        const tex = textureOf(material);
        this.draws.push({
          geometryId: geo.id,
          matrix,
          color: colorOf(material),
          start: g.start,
          count: g.count,
          indexed,
          transparent: isTransparent(material),
          image: tex?.image,
          imageId: tex?.id ?? 0,
        });
      }
      return;
    }
    const material = materials !== null ? materials[0] : mesh.material;
    const tex = textureOf(material);
    this.draws.push({
      geometryId: geo.id,
      matrix,
      color: colorOf(material),
      start: geo.drawRange.start,
      count: Math.min(geo.drawRange.count, total),
      indexed,
      transparent: isTransparent(material),
      image: tex?.image,
      imageId: tex?.id ?? 0,
    });
  }

  /** Upload (or refresh) one geometry's buffers, keyed by three's own id. */
  private upload(geo: NonNullable<ThreeMesh["geometry"]>): void {
    const position = geo.attributes.position!;
    const cached = this.geometries.get(geo.id);
    if (cached !== undefined && cached.version === position.version) return;
    cached?.vertex.destroy();
    cached?.uv.destroy();
    cached?.index?.destroy();

    const verts = new Float32Array(position.array as ArrayLike<number>);
    const vertex = this.device.createBuffer({
      size: Math.max(4, verts.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertex, 0, verts);

    // A geometry with no UVs still needs the buffer bound, so it gets zeros
    // rather than a second pipeline.
    const uvSrc = geo.attributes.uv;
    const uvs =
      uvSrc !== undefined && uvSrc.itemSize === 2
        ? new Float32Array(uvSrc.array as ArrayLike<number>)
        : new Float32Array(position.count * 2);
    const uv = this.device.createBuffer({
      size: Math.max(4, uvs.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uv, 0, uvs);

    let index: GPUBuffer | undefined;
    let indexFormat: GPUIndexFormat = "uint32";
    if (geo.index != null) {
      const src = geo.index.array;
      const data =
        src instanceof Uint16Array ? new Uint32Array(src) : new Uint32Array(src as ArrayLike<number>);
      index = this.device.createBuffer({
        size: Math.max(4, data.byteLength),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(index, 0, data);
      indexFormat = "uint32";
    }
    this.geometries.set(geo.id, { vertex, uv, index, indexFormat, version: position.version });
  }

  /** Upload one image once and keep its bind group. */
  private textureBind(image: TexSource, id: number): GPUBindGroup {
    const hit = this.textures.get(id);
    if (hit !== undefined) return hit.bind;
    const width = "width" in image ? image.width : 1;
    const height = "height" in image ? image.height : 1;
    const texture = this.device.createTexture({
      size: { width: Math.max(1, width), height: Math.max(1, height) },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: image as GPUCopyExternalImageSource },
      { texture },
      { width: Math.max(1, width), height: Math.max(1, height) },
    );
    const bind = this.device.createBindGroup({
      layout: this.texLayout!,
      entries: [
        { binding: 0, resource: this.sampler! },
        { binding: 1, resource: texture.createView() },
      ],
    });
    this.textures.set(id, { texture, bind });
    return bind;
  }

  /** A 1x1 white texture, bound when a draw carries none. */
  private blank(): GPUBindGroup {
    if (this.blankBind !== undefined) return this.blankBind;
    const texture = this.device.createTexture({
      size: { width: 1, height: 1 },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      { width: 1, height: 1 },
    );
    this.blankBind = this.device.createBindGroup({
      layout: this.texLayout!,
      entries: [
        { binding: 0, resource: this.sampler! },
        { binding: 1, resource: texture.createView() },
      ],
    });
    return this.blankBind;
  }

  /**
   * Record the draws. `pointDepth` is the compute path's depth buffer, read
   * per fragment so points can hide geometry.
   */
  record(
    pass: GPURenderPassEncoder,
    camera: Camera,
    pointDepth: GPUBuffer,
    width: number,
    height: number,
  ): void {
    if (this.disposed || this.draws.length === 0) return;
    this.ensurePipeline();
    this.ensureUniform(this.draws.length);
    if (this.pipeline === undefined || this.uniform === undefined) return;
    if (this.bindGroup === undefined || this.boundDepth !== pointDepth) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.layout!,
        entries: [
          { binding: 0, resource: { buffer: this.uniform, size: UNIFORM_STRIDE } },
          { binding: 1, resource: { buffer: pointDepth } },
        ],
      });
      this.boundDepth = pointDepth;
    }

    const cam = camera as unknown as {
      projectionMatrix: { elements: ArrayLike<number> };
      matrixWorldInverse: { elements: ArrayLike<number> };
    };
    // Same remap the point path applies: three hands back a WebGL-range
    // projection here regardless of what the renderer reports, and WebGPU
    // clips anything with ndc.z < 0.
    const viewProj = zeroToOne(
      multiply(cam.projectionMatrix.elements, cam.matrixWorldInverse.elements),
    );

    pass.setPipeline(this.pipeline);
    for (const [i, draw] of this.draws.entries()) {
      const geo = this.geometries.get(draw.geometryId);
      if (geo === undefined || draw.count === 0) continue;

      const mvp = multiply(viewProj, draw.matrix);
      this.scratch.set(mvp, 0);
      this.scratch.set(draw.color, 16);
      this.scratch[20] = width;
      this.scratch[21] = height;
      this.scratch[22] = draw.image !== undefined ? 1 : 0;
      this.device.queue.writeBuffer(
        this.uniform,
        i * UNIFORM_STRIDE,
        this.scratch.buffer,
        0,
        UNIFORM_STRIDE,
      );

      pass.setBindGroup(0, this.bindGroup, [i * UNIFORM_STRIDE]);
      pass.setBindGroup(
        1,
        draw.image !== undefined ? this.textureBind(draw.image, draw.imageId) : this.blank(),
      );
      pass.setVertexBuffer(0, geo.vertex);
      pass.setVertexBuffer(1, geo.uv);
      if (draw.indexed && geo.index !== undefined) {
        pass.setIndexBuffer(geo.index, geo.indexFormat);
        pass.drawIndexed(draw.count, 1, draw.start);
      } else {
        pass.draw(draw.count, 1, draw.start);
      }
    }
  }

  private ensureUniform(count: number): void {
    if (this.uniform !== undefined && this.uniformCapacity >= count) return;
    this.uniform?.destroy();
    this.uniformCapacity = Math.max(16, count * 2);
    this.uniform = this.device.createBuffer({
      size: this.uniformCapacity * UNIFORM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bindGroup = undefined;
  }

  private ensurePipeline(): void {
    if (this.pipeline !== undefined) return;
    const module = this.device.createShaderModule({ code: WGSL });
    this.layout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: UNIFORM_STRIDE },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });
    this.texLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.layout, this.texLayout],
      }),
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
          },
          {
            arrayStride: 8,
            attributes: [{ shaderLocation: 1, offset: 0, format: "float32x2" }],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [
          {
            format: this.format,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: this.depthFormat,
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const g of this.geometries.values()) {
      g.vertex.destroy();
      g.uv.destroy();
      g.index?.destroy();
    }
    for (const t of this.textures.values()) t.texture.destroy();
    this.textures.clear();
    this.geometries.clear();
    this.uniform?.destroy();
    this.draws.length = 0;
  }
}

/** z' = (z + w) / 2, in place — see clip.ts for why this is needed. */
function zeroToOne(m: Float32Array): Float32Array {
  for (let c = 0; c < 4; c++) {
    const z = c * 4 + 2;
    m[z] = 0.5 * (m[z]! + m[c * 4 + 3]!);
  }
  return m;
}

/** Column-major 4x4 multiply, matching three's element order. */
function multiply(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r]! * b[c * 4 + k]!;
      out[c * 4 + r] = sum;
    }
  return out;
}
