import { Matrix3 } from "three/webgpu";
import type { Camera, Object3D } from "three/webgpu";
import type { MeshVertexLayout } from "./mesh-layout.js";
import { quantisedMeshLayout } from "./mesh-layout.js";

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

/**
 * The BIM model's own pipeline: the same occlusion test as the flat path, plus
 * the two things a building needs and a gizmo does not — a normal to shade by
 * and the element id each vertex belongs to.
 *
 * The position arrives as three uint16 over the model's bounding box and is put
 * back in metres by the SAME model matrix that already places the mesh: three's
 * GLTFLoader folds the dequantisation scale into the node transform, so the
 * shader multiplies and asks no questions.
 *
 * The normal is transformed by its own matrix and not by the model matrix,
 * because the dequantisation scale is per-axis and therefore NON-UNIFORM. Using
 * the upper 3x3 of the model matrix would tilt every normal by however much the
 * building's box is longer than it is tall, and the result reads as bad
 * lighting rather than as a bug.
 */
/**
 * The same geometry, drawn once into an id target so a click can be resolved.
 *
 * A separate PASS and not a second attachment on the resolve pass, because
 * picking is a click and not a frame: paying 4 bytes per pixel of bandwidth on
 * every frame to answer a question nobody asked is the wrong trade. The scissor
 * narrows it to the one pixel under the cursor, so the cost is a draw call per
 * material and nothing else.
 *
 * It keeps the point-depth discard, which is a product decision as much as a
 * technical one: a wall hidden behind the scan should not be selectable, or the
 * user picks things they cannot see.
 */
const PICK_WGSL = /* wgsl */ `
struct U {
  mvp        : mat4x4<f32>,
  color      : vec4<f32>,
  screenW    : f32,
  screenH    : f32,
  hasTexture : f32,
  _pad       : f32,
  normalMat  : mat4x4<f32>,
  selected   : f32,
  clipCount  : f32,
  _pad2      : vec2<f32>,
  clip       : array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> pointDepth : array<u32>;
// Um por elemento: 0 esconde. Storage e não uniform porque a seleção cabe num
// float e a visibilidade não — são dezenas de milhares de elementos, e o que
// muda quando alguém desliga um pavimento é qual, não quanta, geometria existe.
@group(0) @binding(2) var<storage, read> visible : array<u32>;


// Mantém o semi-espaço POSITIVO: descarta quando dot(n, p) + d < 0. Mesma
// convenção do three, para quem já tem planos não ter de os inverter.
fn clipped(p : vec3<f32>) -> bool {
  let n = u32(u.clipCount);
  for (var i : u32 = 0u; i < n; i = i + 1u) {
    let pl = u.clip[i];
    if (dot(pl.xyz, p) + pl.w < 0.0) { return true; }
  }
  return false;
}

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) eyeDepth : f32,
  @location(1) @interpolate(flat) featureId : u32,
  @location(2) local : vec3<f32>,
};

@vertex
fn vs(
  @location(0) qpos : vec4<u32>,
  @location(1) nrm  : vec4<f32>,
  @location(2) fid  : vec2<u32>,
) -> VSOut {
  var out : VSOut;
  let clip = u.mvp * vec4<f32>(f32(qpos.x), f32(qpos.y), f32(qpos.z), 1.0);
  out.pos = clip;
  out.eyeDepth = clip.w;
  out.featureId = fid.x;
  out.local = vec3<f32>(f32(qpos.x), f32(qpos.y), f32(qpos.z));
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) u32 {
  // Cortado fora é não-clicável, pelo mesmo motivo que escondido é.
  if (clipped(in.local)) { discard; }
  // Escondido é não-clicável: selecionar o que não se vê seria pior que não
  // poder selecionar.
  if (in.featureId < arrayLength(&visible) && visible[in.featureId] == 0u) {
    discard;
  }
  let x = u32(clamp(in.pos.x, 0.0, u.screenW - 1.0));
  let y = u32(clamp(in.pos.y, 0.0, u.screenH - 1.0));
  if (bitcast<u32>(in.eyeDepth) > pointDepth[y * u32(u.screenW) + x]) {
    discard;
  }
  return in.featureId;
}
`;

const MESH_WGSL = /* wgsl */ `
struct U {
  mvp        : mat4x4<f32>,
  color      : vec4<f32>,
  screenW    : f32,
  screenH    : f32,
  hasTexture : f32,
  _pad       : f32,
  normalMat  : mat4x4<f32>,
  // Qual elemento está selecionado, ou -1. Um uniform e não um atributo: a
  // seleção muda a cada clique e a geometria não muda nunca, e reenviar um
  // prédio para pintar uma parede seria pagar megabytes por um booleano.
  selected   : f32,
  clipCount  : f32,
  _pad2      : vec2<f32>,
  // DEC-B6. Já no espaço LOCAL deste draw, transformados na CPU: assim o
  // shader não precisa da matriz de modelo e o uniform não estoura os 256 B.
  // Um plano é covariante, então a transformação é a TRANSPOSTA da matriz —
  // usar a matriz direta inclina o corte junto com a escala de desquantização,
  // que é por eixo.
  clip       : array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> pointDepth : array<u32>;
// Um por elemento: 0 esconde. Storage e não uniform porque a seleção cabe num
// float e a visibilidade não — são dezenas de milhares de elementos, e o que
// muda quando alguém desliga um pavimento é qual, não quanta, geometria existe.
@group(0) @binding(2) var<storage, read> visible : array<u32>;


// Mantém o semi-espaço POSITIVO: descarta quando dot(n, p) + d < 0. Mesma
// convenção do three, para quem já tem planos não ter de os inverter.
fn clipped(p : vec3<f32>) -> bool {
  let n = u32(u.clipCount);
  for (var i : u32 = 0u; i < n; i = i + 1u) {
    let pl = u.clip[i];
    if (dot(pl.xyz, p) + pl.w < 0.0) { return true; }
  }
  return false;
}

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) eyeDepth : f32,
  @location(1) normal : vec3<f32>,
  @location(2) @interpolate(flat) featureId : u32,
  // Interpolada, para o corte ser por FRAGMENTO: um triângulo que atravessa o
  // plano tem de ser cortado, não aceite ou rejeitado inteiro.
  @location(3) local : vec3<f32>,
};

@vertex
fn vs(
  @location(0) qpos : vec4<u32>,
  @location(1) nrm  : vec4<f32>,
  @location(2) fid  : vec2<u32>,
) -> VSOut {
  var out : VSOut;
  let clip = u.mvp * vec4<f32>(f32(qpos.x), f32(qpos.y), f32(qpos.z), 1.0);
  out.pos = clip;
  out.eyeDepth = clip.w;
  out.normal = normalize((u.normalMat * vec4<f32>(nrm.xyz, 0.0)).xyz);
  out.featureId = fid.x;
  out.local = vec3<f32>(f32(qpos.x), f32(qpos.y), f32(qpos.z));
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  if (clipped(in.local)) { discard; }
  if (in.featureId < arrayLength(&visible) && visible[in.featureId] == 0u) {
    discard;
  }
  let x = u32(clamp(in.pos.x, 0.0, u.screenW - 1.0));
  let y = u32(clamp(in.pos.y, 0.0, u.screenH - 1.0));
  let stored = pointDepth[y * u32(u.screenW) + x];
  if (bitcast<u32>(in.eyeDepth) > stored) {
    discard;
  }
  // Two lights and an ambient floor, no shadows: enough for a wall to read as a
  // wall from any angle, and cheap enough not to matter next to the point pass.
  let n = normalize(in.normal);
  let key = max(dot(n, normalize(vec3<f32>(0.35, 0.75, 0.55))), 0.0);
  let fill = max(dot(n, normalize(vec3<f32>(-0.5, 0.2, -0.6))), 0.0);
  let lit = 0.34 + 0.52 * key + 0.14 * fill;
  var rgb = u.color.rgb * lit;
  // O elemento selecionado: puxado para o azul da marca e clareado, mas ainda
  // sombreado — pintar de chapado esconderia a forma dele, que é justamente o
  // que a pessoa clicou para ver.
  if (u.selected >= 0.0 && f32(in.featureId) == u.selected) {
    rgb = mix(rgb, vec3<f32>(0.30, 0.76, 1.0), 0.65) + vec3<f32>(0.10 * lit);
  }
  return vec4<f32>(rgb, u.color.a);
}
`;

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

/** Cleared into the id target to mean "no element here"; 0 is a real index. */
const NO_FEATURE = 0xffffffff;

/** 256 is the minimum dynamic-offset alignment WebGPU guarantees. */
const UNIFORM_STRIDE = 256;

interface CachedGeometry {
  vertex: GPUBuffer;
  /** UVs, zero-filled when the geometry has none, so one pipeline serves all. */
  uv: GPUBuffer;
  index: GPUBuffer | undefined;
  indexFormat: GPUIndexFormat;
  version: number;
  /** Set when the geometry is a quantised BIM model rather than a plain mesh. */
  mesh: MeshVertexLayout | undefined;
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
  /** The 3x3 that transforms this draw's normals, padded to a mat4. */
  normalMatrix: Float32Array | undefined;
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
      /** Present only on a BIM model; what marks the geometry as one. */
      _feature_id_0?: unknown;
    };
    // `array` is typed loosely because the two paths want different things
    // from it: the float path reads it as numbers, the quantised one binds its
    // bytes. Narrowing happens where each path uses it.
    index?: { array: ArrayLike<number> & Partial<ArrayBufferView>; count: number } | null;
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
  private meshPipeline: GPURenderPipeline | undefined;
  private pickPipeline: GPURenderPipeline | undefined;
  private pickTarget: { id: GPUTexture; depth: GPUTexture; w: number; h: number } | undefined;
  private pickRead: GPUBuffer | undefined;
  private selected = -1;
  /** Planos em coordenadas de CENA, 4 no máximo. `undefined` = sem corte. */
  private clip: Float32Array | undefined;
  private visible: GPUBuffer | undefined;
  private boundVisibility: GPUBuffer | undefined;
  private emptyVisible: GPUBuffer | undefined;
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
    // Only a BIM geometry pays for this: the others shade flat and never read
    // it. Padded into a mat4 because WGSL's mat3x3 has a 16-byte column stride
    // anyway, and a mat4 is one fewer thing to get wrong.
    let normalMatrix: Float32Array | undefined;
    if (geo.attributes._feature_id_0 !== undefined) {
      const n = new Matrix3().getNormalMatrix(mesh.matrixWorld).elements;
      normalMatrix = new Float32Array([
        n[0]!, n[1]!, n[2]!, 0,
        n[3]!, n[4]!, n[5]!, 0,
        n[6]!, n[7]!, n[8]!, 0,
        0, 0, 0, 1,
      ]);
    }
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
          normalMatrix,
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
      normalMatrix,
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

    // A BIM model arrives quantised and interleaved and is bound AS IT LIES —
    // unpacking it to float32 here would undo the 2x the converter bought and
    // cost a copy of the building on every upload.
    const mesh = quantisedMeshLayout(geo as unknown as Parameters<typeof quantisedMeshLayout>[0]);
    if (mesh !== undefined) {
      const src = new Uint8Array(mesh.source, mesh.byteOffset, mesh.byteLength);
      const vertex = this.device.createBuffer({
        size: Math.max(4, src.byteLength),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(vertex, 0, src);
      // The mesh pipeline binds no second vertex buffer; this exists so the
      // cache entry has one shape.
      const uv = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX });
      let index: GPUBuffer | undefined;
      let indexFormat: GPUIndexFormat = "uint32";
      if (geo.index != null) {
        // The source width is KEPT here, unlike the float path: a model whose
        // batch fits in uint16 indices should not have its index buffer
        // doubled on the way to the GPU.
        const arr = geo.index.array as unknown as ArrayBufferView & { BYTES_PER_ELEMENT: number };
        indexFormat = arr.BYTES_PER_ELEMENT === 2 ? "uint16" : "uint32";
        // `writeBuffer` takes a multiple of FOUR bytes, and a uint16 index
        // buffer is only a multiple of two: indices come in threes, so `3 * T *
        // 2` is 4-aligned exactly when the triangle count is even. Half of all
        // batches are therefore rejected outright. Pad to the next word — the
        // extra index is never drawn, because `drawIndexed` is told the real
        // count.
        const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
        const padded =
          bytes.byteLength % 4 === 0
            ? bytes
            : (() => {
                const out = new Uint8Array(bytes.byteLength + (4 - (bytes.byteLength % 4)));
                out.set(bytes);
                return out;
              })();
        index = this.device.createBuffer({
          size: Math.max(4, padded.byteLength),
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(index, 0, padded as unknown as BufferSource);
      }
      this.geometries.set(geo.id, {
        vertex, uv, index, indexFormat, version: position.version, mesh,
      });
      return;
    }

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
    this.geometries.set(geo.id, { vertex, uv, index, indexFormat, version: position.version, mesh: undefined });
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
    this.recordInto(pass, camera, pointDepth, width, height, "draw");
  }

  /**
   * The draw loop, shared by the frame and by picking. The two differ in three
   * places and nowhere else — which pipeline, whether the flat path runs at
   * all, and whether the texture group is bound — so they are one function
   * with a mode rather than two that drift apart.
   */
  private recordInto(
    pass: GPURenderPassEncoder,
    camera: Camera,
    pointDepth: GPUBuffer,
    width: number,
    height: number,
    mode: "draw" | "pick",
  ): void {
    if (this.disposed || this.draws.length === 0) return;
    this.ensurePipeline();
    this.ensureUniform(this.draws.length);
    if (this.pipeline === undefined || this.uniform === undefined) return;
    if (
      this.bindGroup === undefined ||
      this.boundDepth !== pointDepth ||
      this.boundVisibility !== this.visible
    ) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.layout!,
        entries: [
          { binding: 0, resource: { buffer: this.uniform, size: UNIFORM_STRIDE } },
          { binding: 1, resource: { buffer: pointDepth } },
          { binding: 2, resource: { buffer: this.visibilityBuffer() } },
        ],
      });
      this.boundDepth = pointDepth;
      this.boundVisibility = this.visible;
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

    // One `setPipeline` per RUN of draws, not per draw: the list is already
    // sorted, and a building is a run of meshes followed by whatever gizmos the
    // tools put up.
    let bound: GPURenderPipeline | undefined;
    for (const [i, draw] of this.draws.entries()) {
      const geo = this.geometries.get(draw.geometryId);
      if (geo === undefined || draw.count === 0) continue;

      // Picking answers "which ELEMENT", so a gizmo or the vehicle has no id to
      // give and is skipped rather than drawn into the id target as noise.
      if (mode === "pick" && geo.mesh === undefined) continue;

      const want =
        geo.mesh === undefined
          ? this.pipeline!
          : mode === "pick"
            ? this.ensurePickPipeline(geo.mesh)
            : this.ensureMeshPipeline(geo.mesh);
      if (want !== bound) {
        pass.setPipeline(want);
        bound = want;
      }

      const mvp = multiply(viewProj, draw.matrix);
      this.scratch.set(mvp, 0);
      this.scratch.set(draw.color, 16);
      this.scratch[20] = width;
      this.scratch[21] = height;
      this.scratch[22] = draw.image !== undefined ? 1 : 0;
      if (draw.normalMatrix !== undefined) this.scratch.set(draw.normalMatrix, 24);
      this.scratch[40] = this.selected;
      const planes = this.clip;
      const count = planes === undefined ? 0 : Math.min(4, planes.length >> 2);
      this.scratch[41] = count;
      for (let p = 0; p < count; p++) {
        // P_local = transpose(M) * P_scene, com M em ordem de coluna.
        const m = draw.matrix;
        const a = planes![p * 4]!;
        const b = planes![p * 4 + 1]!;
        const c = planes![p * 4 + 2]!;
        const d = planes![p * 4 + 3]!;
        const o = 44 + p * 4;
        this.scratch[o] = m[0]! * a + m[1]! * b + m[2]! * c + m[3]! * d;
        this.scratch[o + 1] = m[4]! * a + m[5]! * b + m[6]! * c + m[7]! * d;
        this.scratch[o + 2] = m[8]! * a + m[9]! * b + m[10]! * c + m[11]! * d;
        this.scratch[o + 3] = m[12]! * a + m[13]! * b + m[14]! * c + m[15]! * d;
      }
      this.device.queue.writeBuffer(
        this.uniform,
        i * UNIFORM_STRIDE,
        this.scratch.buffer,
        0,
        UNIFORM_STRIDE,
      );

      pass.setBindGroup(0, this.bindGroup, [i * UNIFORM_STRIDE]);
      pass.setVertexBuffer(0, geo.vertex);
      if (geo.mesh === undefined && mode === "draw") {
        // The texture group and the UV buffer belong to the flat path only; the
        // mesh pipeline's layout does not declare them.
        pass.setBindGroup(
          1,
          draw.image !== undefined ? this.textureBind(draw.image, draw.imageId) : this.blank(),
        );
        pass.setVertexBuffer(1, geo.uv);
      }
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

  /**
   * The mesh pipeline, built lazily and only when a BIM layer actually turns
   * up. It shares group 0 — the uniform and the points' depth buffer — with the
   * flat path, and declares NO texture group: a model is shaded, not textured.
   */
  private ensureMeshPipeline(layout: MeshVertexLayout): GPURenderPipeline {
    if (this.meshPipeline !== undefined) return this.meshPipeline;
    this.ensurePipeline(); // for `this.layout`, the shared group 0
    const module = this.device.createShaderModule({ code: MESH_WGSL });
    this.meshPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout!] }),
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [{ arrayStride: layout.arrayStride, attributes: [...layout.attributes] }],
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
      // `back` and not `none`: a building is closed geometry and half its
      // triangles face away, so culling them is a free halving of fragment work
      // that a gizmo — which may be a single open quad — cannot afford.
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: this.depthFormat, depthWriteEnabled: true, depthCompare: "less" },
    });
    return this.meshPipeline;
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
        {
          binding: 2,
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

  /**
   * Which elements are drawn: one `u32` per dense feature index, zero to hide.
   * `undefined` shows everything.
   *
   * The array is uploaded, not diffed: a mask for the largest open IFC there is
   * (sixty5, 16,401 elements) is 64 kB, and a toggle that costs 64 kB of
   * PCIe is not a toggle worth optimising.
   */
  setVisibility(mask: Uint32Array | undefined): void {
    if (mask === undefined || mask.length === 0) {
      this.visible?.destroy();
      this.visible = undefined;
      return;
    }
    const bytes = mask.byteLength;
    if (this.visible === undefined || this.visible.size < bytes) {
      this.visible?.destroy();
      this.visible = this.device.createBuffer({
        size: Math.max(4, bytes),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.visible, 0, mask as unknown as BufferSource);
  }

  /** A one-element "everything visible" buffer, so the binding always exists. */
  private visibilityBuffer(): GPUBuffer {
    if (this.visible !== undefined) return this.visible;
    this.emptyVisible ??= this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Length 1: `arrayLength` then makes every real featureId fall past the end
    // and the shader's bounds check lets it through.
    this.device.queue.writeBuffer(this.emptyVisible, 0, new Uint32Array([1]));
    return this.emptyVisible;
  }

  /**
   * Cross-section planes, in SCENE coordinates: `[nx, ny, nz, d]` each, at most
   * four. The positive half-space survives — `dot(n, p) + d >= 0` — which is
   * three's convention, so a caller that already has planes does not invert
   * them.
   *
   * They are transformed into each draw's own space at record time, on the CPU.
   * A plane is covariant, so that transform is the TRANSPOSE of the model
   * matrix: using the matrix itself tilts the cut along with the per-axis
   * dequantisation scale, and the section comes out skewed by however much the
   * building's box is longer than it is tall.
   */
  setClipPlanes(planes: Float32Array | undefined): void {
    this.clip = planes === undefined || planes.length === 0 ? undefined : planes;
  }

  /** Highlight one element, or `undefined` to clear. Costs one float per draw. */
  setSelected(feature: number | undefined): void {
    this.selected = feature ?? -1;
  }

  /**
   * Which element sits under the cursor, as the DENSE FEATURE INDEX the
   * converter wrote — not the IFC expressID. The index -> expressID table
   * lives in `props.json` beside the GLB (DEC-B4), because 4.4M-wide ids do not
   * fit a vertex attribute and a model has at most tens of thousands of parts.
   *
   * `undefined` means no model geometry under that pixel: either the point
   * cloud is in front of it, or there is nothing there.
   *
   * Coordinates are DEVICE pixels, top-left origin — the same frame `record`
   * draws in, so a caller multiplies CSS pixels by the pixel ratio and does not
   * flip anything.
   */
  async pickFeature(
    camera: Camera,
    pointDepth: GPUBuffer,
    width: number,
    height: number,
    x: number,
    y: number,
  ): Promise<number | undefined> {
    if (this.disposed || this.draws.length === 0) return undefined;
    if (!this.draws.some((d) => this.geometries.get(d.geometryId)?.mesh !== undefined))
      return undefined;
    const px = Math.floor(x);
    const py = Math.floor(y);
    if (px < 0 || py < 0 || px >= width || py >= height) return undefined;

    const target = this.ensurePickTarget(width, height);
    this.pickRead ??= this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: target.id.createView(),
          // Every valid index is >= 0, so "nothing" needs a value outside the
          // range rather than zero, which is the first element.
          clearValue: { r: NO_FEATURE, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: target.depth.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
      },
    });
    // One pixel of work: the scissor is what keeps this cheap enough to run on
    // a click without thinking about it.
    pass.setScissorRect(px, py, 1, 1);
    this.recordInto(pass, camera, pointDepth, width, height, "pick");
    pass.end();

    enc.copyTextureToBuffer(
      { texture: target.id, origin: { x: px, y: py } },
      // 256 is the row alignment `copyTextureToBuffer` requires, even for the
      // four bytes actually wanted.
      { buffer: this.pickRead, bytesPerRow: 256 },
      { width: 1, height: 1 },
    );
    this.device.queue.submit([enc.finish()]);

    await this.pickRead.mapAsync(GPUMapMode.READ);
    const value = new Uint32Array(this.pickRead.getMappedRange(0, 4))[0];
    this.pickRead.unmap();
    return value === NO_FEATURE ? undefined : value;
  }

  private ensurePickTarget(w: number, h: number): {
    id: GPUTexture;
    depth: GPUTexture;
    w: number;
    h: number;
  } {
    const hit = this.pickTarget;
    if (hit !== undefined && hit.w === w && hit.h === h) return hit;
    hit?.id.destroy();
    hit?.depth.destroy();
    const id = this.device.createTexture({
      size: { width: w, height: h },
      format: "r32uint",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depth = this.device.createTexture({
      size: { width: w, height: h },
      format: this.depthFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.pickTarget = { id, depth, w, h };
    return this.pickTarget;
  }

  private ensurePickPipeline(layout: MeshVertexLayout): GPURenderPipeline {
    if (this.pickPipeline !== undefined) return this.pickPipeline;
    this.ensurePipeline();
    const module = this.device.createShaderModule({ code: PICK_WGSL });
    this.pickPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout!] }),
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [{ arrayStride: layout.arrayStride, attributes: [...layout.attributes] }],
      },
      // No blending: an id is a value, not a colour, and averaging two of them
      // names a third element that is not there.
      fragment: { module, entryPoint: "fs", targets: [{ format: "r32uint" }] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: this.depthFormat, depthWriteEnabled: true, depthCompare: "less" },
    });
    return this.pickPipeline;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const g of this.geometries.values()) {
      g.vertex.destroy();
      g.uv.destroy();
      g.index?.destroy();
    }
    this.visible?.destroy();
    this.emptyVisible?.destroy();
    this.pickTarget?.id.destroy();
    this.pickTarget?.depth.destroy();
    this.pickRead?.destroy();
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
