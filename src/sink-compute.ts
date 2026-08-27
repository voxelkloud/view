import type { DecodedPointData } from "@voxelkloud/format-potree";
import type { Matrix4, PerspectiveCamera } from "three";
import { COMPUTE_WGSL } from "./compute-wgsl.js";
import { DEVIATION_WGSL } from "./deviation-wgsl.js";
import type { OctreeCut } from "./cut.js";
import type { ColorMode } from "./material.js";
import { CLASS_ATTRIBUTE } from "./material-options.js";
import type { PointReadback, PointSink } from "./sink.js";
import { toZeroToOneDepth } from "./clip.js";

/** The index the shader switches on, in the order {@link ColorMode} declares. */
const MODE_INDEX: Record<ColorMode["kind"], number> = {
  rgb: 0,
  flat: 1,
  elevation: 2,
  level: 3,
  intensity: 4,
  classification: 5,
  // A mesma via escalar que `intensity`: o kernel do desvio escreve no `col`
  // que o modo 4 lê. Um modo próprio, e não `intensity` reaproveitado, para
  // que a UI e o relatório saibam que a unidade é metros de desvio.
  deviation: 4,
};

export interface ComputeSinkOptions {
  readonly pointBudget: number;
  readonly colorMode: ColorMode;
  readonly sizeMultiplier: number;
  readonly minPixelSize: number;
  readonly maxPixelSize: number;
  readonly elevationRange: readonly [number, number];
  readonly scalarRange: readonly [number, number];
  readonly background: readonly [number, number, number];
  readonly edl?: { readonly strength: number; readonly radius: number } | undefined;
}

interface Block {
  readonly start: number;
  readonly count: number;
  readonly slot: number;
  readonly level: number;
}

/**
 * The decoder's arrays are typed over `ArrayBufferLike`, which TypeScript will
 * not narrow to the `ArrayBuffer` WebGPU's signatures ask for. The runtime
 * value is always a plain `ArrayBuffer` here — nothing in this package produces
 * a `SharedArrayBuffer` — so this is the cast, in one place and named, rather
 * than sprinkled at every call.
 */
const gpuData = (v: ArrayBufferView): GPUAllowSharedBufferSource =>
  v as unknown as GPUAllowSharedBufferSource;

/**
 * The screen-sized half of the compute rasteriser, owned by the VIEW rather
 * than by a cloud.
 *
 * The split is not tidiness: depth, accumulation and the resolve are per FRAME,
 * not per cloud. A sink that cleared and resolved for itself would erase the
 * cloud drawn before it, so two clouds would render as one. Here the frame is
 * cleared once, every cloud's points accumulate into the SAME depth and colour
 * buffers — which is also what makes them occlude each other correctly — and
 * one resolve reads the result out.
 */
/**
 * RGBA bytes for the GPU, whatever width the driver decoded.
 *
 * Only `Uint8Array` used to pass here and everything else was dropped in
 * silence — which turned any LAS carrying 16-bit RGB into a pure black cloud,
 * because the colour buffer stayed zero. That is most aerial survey data.
 *
 * The narrowing is by the WIDEST channel seen, not a fixed `>> 8`: plenty of
 * files store 0..255 inside 16-bit fields, and shifting those leaves every
 * channel at zero — the same trap the CLI's thumbnail renderer fell into on
 * this exact dataset.
 */
function narrowColors(src: ArrayLike<number> | undefined): Uint8Array | undefined {
  if (src === undefined) return undefined;
  if (src instanceof Uint8Array) return src;
  if (!(src instanceof Uint16Array)) return undefined;
  let max = 0;
  for (let i = 0; i < src.length; i++) if (src[i]! > max) max = src[i]!;
  const out = new Uint8Array(src.length);
  if (max <= 255) {
    for (let i = 0; i < src.length; i++) out[i] = src[i]!;
    return out;
  }
  const scale = 255 / max;
  for (let i = 0; i < src.length; i++) out[i] = Math.min(255, Math.round(src[i]! * scale));
  return out;
}

/** Depth format for the overlay attachment. 24-bit plus stencil is universal. */
export const OVERLAY_DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

/**
 * Shader messages that are NOT errors, drained into a shared array.
 *
 * `createShaderModule` never rejects and `getCompilationInfo` is async, so a
 * module that compiled with warnings looks exactly like one that compiled
 * clean — right up until a driver that treats the warning as fatal draws
 * nothing. Fire-and-forget: nothing here should delay a frame.
 */
function collectShaderMessages(
  module: GPUShaderModule,
  label: string,
  into: string[] | undefined,
): void {
  if (into === undefined) return;
  void module
    .getCompilationInfo()
    .then((info) => {
      for (const m of info.messages) {
        // Errors already travel through `uncapturederror`; duplicating them
        // here would spend an event budget twice on one fault.
        if (m.type === "error" || into.length >= 24) continue;
        into.push(`${label}:${m.lineNum}:${m.linePos} ${m.type}: ${m.message}`);
      }
    })
    .catch(() => {
      /* A browser without compilation info is not a failure worth reporting. */
    });
}

export class ComputeRasterizer {
  private depthBuf: GPUBuffer | undefined;
  /**
   * A real depth attachment, used ONLY so overlay meshes occlude each other.
   * The points' own depth stays in `depthBuf`; nothing is copied between them.
   */
  private overlayDepth: GPUTexture | undefined;
  private accumBuf: GPUBuffer | undefined;
  private bind: GPUBindGroup | undefined;
  private pixels = 0;
  private width = 0;
  private height = 0;
  private generation = 0;
  private disposed = false;

  readonly module: GPUShaderModule;
  /** Bumped whenever depth/accum are recreated, so sinks know to rebind. */
  get bufferGeneration(): number {
    return this.generation;
  }

  private readonly layout: GPUBindGroupLayout;
  private readonly clearPipe: GPUComputePipeline;
  private readonly drawPipe: GPURenderPipeline;
  private readonly uniBuf: GPUBuffer;
  private readonly uniform = new ArrayBuffer(UNIFORM_BYTES);
  private readonly uf = new Float32Array(this.uniform);

  constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    format: GPUTextureFormat,
    private readonly options: {
      readonly background: readonly [number, number, number];
      readonly edl?: { readonly strength: number; readonly radius: number } | undefined;
      /** Where non-error shader messages go. Shared with the view. */
      readonly warnings?: string[] | undefined;
    },
  ) {
    this.module = device.createShaderModule({ code: COMPUTE_WGSL, label: "voxelkloud-compute" });
    collectShaderMessages(this.module, "compute", options.warnings);
    this.uniBuf = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const rw = { type: "storage" } as const;
    const CF = GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT;
    // Only the three bindings `clearPass` and `fsResolve` statically use. A
    // layout must cover what an ENTRY POINT touches, not what the module
    // declares, so the resolve does not carry the point buffers.
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 2, visibility: CF, buffer: rw },
        { binding: 3, visibility: CF, buffer: rw },
        { binding: 4, visibility: CF, buffer: { type: "uniform" } },
      ],
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.clearPipe = device.createComputePipeline({
      layout: pl,
      compute: { module: this.module, entryPoint: "clearPass" },
    });
    this.drawPipe = device.createRenderPipeline({
      layout: pl,
      vertex: { module: this.module, entryPoint: "vsResolve" },
      fragment: { module: this.module, entryPoint: "fsResolve", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
  }

  get depth(): GPUBuffer | undefined {
    return this.depthBuf;
  }
  get accum(): GPUBuffer | undefined {
    return this.accumBuf;
  }

  private resize(width: number, height: number): void {
    if (width === this.width && height === this.height && this.depthBuf !== undefined) return;
    this.width = width;
    this.height = height;
    this.pixels = width * height;
    this.depthBuf?.destroy();
    this.accumBuf?.destroy();
    const mk = (bytes: number): GPUBuffer =>
      this.device.createBuffer({
        size: Math.max(4, bytes),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    this.depthBuf = mk(this.pixels * 4);
    this.accumBuf = mk(this.pixels * 16);
    this.overlayDepth?.destroy();
    this.overlayDepth = this.device.createTexture({
      size: { width, height },
      format: OVERLAY_DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.bind = this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 2, resource: { buffer: this.depthBuf } },
        { binding: 3, resource: { buffer: this.accumBuf } },
        { binding: 4, resource: { buffer: this.uniBuf } },
      ],
    });
    this.generation++;
  }

  /** Clear the frame. Returns the encoder every sink then dispatches into. */
  begin(width: number, height: number): GPUCommandEncoder | undefined {
    if (this.disposed || width <= 0 || height <= 0) return undefined;
    this.resize(width, height);
    const o = this.options;
    this.uf[16] = width;
    this.uf[17] = height;
    this.uf[41] = o.edl?.strength ?? 0;
    this.uf[42] = o.edl?.radius ?? 1.4;
    this.uf[43] = o.background[0];
    this.uf[44] = o.background[1];
    this.uf[45] = o.background[2];
    this.device.queue.writeBuffer(this.uniBuf, 0, this.uniform);
    const enc = this.device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setBindGroup(0, this.bind!);
    cp.setPipeline(this.clearPipe);
    cp.dispatchWorkgroups(Math.ceil(this.pixels / WORKGROUP));
    cp.end();
    return enc;
  }

  /**
   * Resolve to the swapchain and submit everything the frame encoded.
   *
   * `overlay` runs inside the SAME pass, right after the resolve triangle, so
   * app geometry composites with the points instead of overwriting them.
   */
  end(enc: GPUCommandEncoder, overlay?: (pass: GPURenderPassEncoder) => void): void {
    const view = this.context.getCurrentTexture().createView();
    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    });
    rp.setPipeline(this.drawPipe);
    rp.setBindGroup(0, this.bind!);
    rp.draw(3);
    rp.end();

    // A SECOND pass, not a continuation of the first.
    //
    // The resolve binds the point depth as read-write storage (its shader uses
    // atomics), and the overlay reads the same buffer. WebGPU forbids a
    // buffer being writable and readable in one synchronisation scope, and a
    // render pass is one scope — so the overlay gets its own, loading the
    // colour the resolve just wrote instead of clearing it.
    if (overlay !== undefined) {
      const op = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: "load", storeOp: "store" }],
        depthStencilAttachment: {
          view: this.overlayDepth!.createView(),
          depthClearValue: 1,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      overlay(op);
      op.end();
    }
    this.device.queue.submit([enc.finish()]);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.depthBuf?.destroy();
    this.accumBuf?.destroy();
    this.overlayDepth?.destroy();
    this.uniBuf.destroy();
  }
}

/**
 * The point-range allocator, with no GPU in it.
 *
 * Extracted so it can be TESTED. It is the part of this sink that corrupts
 * silently: a free run that fails to coalesce fragments the arena until an
 * allocation that should fit does not, and the only symptom is a frame quietly
 * drawing fewer points than the budget while every other counter looks right.
 * A wrong `start` is worse — it hands out a range that overlaps a live node.
 */
export class BlockAllocator {
  /** Address-ordered and coalesced, the same shape the slab arena keeps. */
  private readonly free: { start: number; count: number }[] = [];
  private high = 0;

  constructor(private cap: number) {}

  /** The high-water mark, which is what a frame must dispatch over. */
  get used(): number {
    return this.high;
  }
  get capacity(): number {
    return this.cap;
  }
  get freeRunCount(): number {
    return this.free.length;
  }
  /** Points inside free runs — the fragmentation, in points. */
  get freePoints(): number {
    let n = 0;
    for (const r of this.free) n += r.count;
    return n;
  }

  release(start: number, count: number): void {
    if (count <= 0) return;
    const free = this.free;
    let i = 0;
    while (i < free.length && free[i]!.start < start) i++;
    free.splice(i, 0, { start, count });
    const prev = free[i - 1];
    if (prev !== undefined && prev.start + prev.count === start) {
      prev.count += count;
      free.splice(i, 1);
      i--;
    }
    const run = free[i]!;
    const next = free[i + 1];
    if (next !== undefined && run.start + run.count === next.start) {
      run.count += next.count;
      free.splice(i + 1, 1);
    }
  }

  /**
   * First fit, then the bump pointer. Returns -1 only when the caller declined
   * to grow; `grow` is the caller's because it has GPU buffers to copy.
   */
  allocate(count: number): number {
    for (let i = 0; i < this.free.length; i++) {
      const r = this.free[i]!;
      if (r.count < count) continue;
      const start = r.start;
      if (r.count === count) this.free.splice(i, 1);
      else {
        r.start += count;
        r.count -= count;
      }
      return start;
    }
    if (this.high + count > this.cap) return -1;
    const start = this.high;
    this.high += count;
    return start;
  }

  /** Raise the ceiling. Never lowers it, and never moves a live range. */
  setCapacity(cap: number): void {
    if (cap > this.cap) this.cap = cap;
  }

  reset(): void {
    this.free.length = 0;
    this.high = 0;
  }
}

/**
 * Point slots per u32 of `live`; the cap on distinct resident nodes.
 *
 * Exportado porque {@link packNodeMeta} lhe reserva EXACTAMENTE 16 bits: subir
 * isto acima de 65536 faz dois nós partilharem slot e consultarem a liveness um
 * do outro, sem erro nenhum. Um teste tranca a invariante.
 */
export const MAX_SLOTS = 65_536;
// 256 e não 192 desde a DEC-B6: os quatro planos de corte precisam de
// alinhamento de 16 bytes, logo entram em 192 e o struct fecha em 256. Depois
// veio a faixa de altura, um vec2 que só cabia DEPOIS do array de planos, e o
// struct fecha em 272. Manter este número em sincronia com o `struct U` do
// WGSL é obrigatório — um uniform mais curto que o struct dá layout inválido, e
// o cabeçalho de `compute-wgsl` avisa que isso NÃO lança: os passes silenciam e
// a tela fica preta com todos os contadores da CPU corretos.
const UNIFORM_BYTES = 272;

/**
 * Faixa de altura DESLIGADA.
 *
 * Um sentinela em vez de um booleano no uniform: nenhum ponto real passa de
 * 3.4e38, então o teste no shader é o mesmo com ou sem filtro e o caso comum
 * não paga nem um branch nem quatro bytes de flag.
 */
export const Z_RANGE_OFF: readonly [number, number] = [-3.4e38, 3.4e38];
const WORKGROUP = 256;

/**
 * `nmeta`, um u32 por ponto: nível nos 8 bits baixos, slot do nó dono nos 16
 * seguintes, classe ASPRS no byte de topo.
 *
 * O byte de topo estava livre porque {@link MAX_SLOTS} é 65536: o slot nunca
 * passa de 16 bits ainda que o layout lhe reservasse 24. É o que torna a classe
 * gratuita — sem buffer novo, num layout de bind que já está no limite de oito
 * storage buffers que o WebGPU garante por estágio, e onde um nono não lançaria
 * erro, faria os passes silenciarem.
 *
 * O shader tem de ler o slot com máscara (`slotOf`), nunca com um `>> 8` cru:
 * sem ela a classe entra no índice e o ponto consulta a liveness de outro nó.
 *
 * @param present marcado por código visto, para a UI listar só o que existe.
 * Escrito aqui porque este laço já toca todo ponto, e uma segunda passada por
 * 20 milhões de pontos para contar o mesmo seria pura repetição.
 */
export function packNodeMeta(
  level: number,
  slot: number,
  count: number,
  classes: ArrayLike<number> | undefined,
  present?: Uint8Array,
): Uint32Array {
  const base = (level & 0xff) | ((slot & 0xffff) << 8);
  const meta = new Uint32Array(count);
  if (classes === undefined) {
    meta.fill(base);
    return meta;
  }
  for (let i = 0; i < count; i++) {
    // A lane f32 carrega o código cru. Um valor fora de 0..255 não é uma classe
    // ASPRS; cai em 255, que o shader já trata como "fora do padrão" junto com
    // tudo acima de 18 — e não em 0, que é uma classe real ("created").
    const raw = classes[i] ?? 0;
    const code = raw >= 0 && raw <= 255 ? raw | 0 : 255;
    if (present !== undefined) present[code] = 1;
    meta[i] = base | (code << 24);
  }
  return meta;
}

/**
 * A {@link PointSink} that also DRAWS, by software-rasterising its points in
 * compute shaders instead of handing them to three as instanced quads.
 *
 * Why it exists, in one measurement: INP at 3M points is ~320 ms through the
 * instanced path and 56–72 ms through this one, against Potree's 136 ms. The
 * cost the instanced path cannot shed is PER INSTANCE — the attribute step for
 * `pointOffset`, `color` and `scalarValue`, once per splat — and that was
 * established by elimination, not guessed: pinning the splat to 1 px left INP
 * unchanged (so not fill rate), a 3-vertex envelope left it unchanged (so not
 * per-vertex), and a sweep showed INP linear in instance count. `gl.POINTS`
 * would avoid it, but WGSL has no point-size builtin and three's WebGPU backend
 * maps `isPoints` to `point-list` topology, silently ignoring `sizeNode`, so
 * every point would rasterise as one unattenuated pixel.
 *
 * Three passes, all with 32-bit atomics: `atomicMin` the depth, `atomicAdd` the
 * colour and a count, then a fullscreen resolve that averages. One invocation
 * per point, no instancing, no envelope, no attribute step. The averaging is a
 * bonus the instanced path does not get — it is antialiasing.
 *
 * It keeps a CPU mirror of positions and colour even though the render never
 * reads it, because {@link PointSink.readPoints} is what makes `pickPoint`
 * synchronous, and this pipeline could otherwise only answer a pick after a
 * GPU readback. That is the same memory the slab arena already spends, so it is
 * not a new cost — but it IS the thing to remove if `pickPoint` ever grows an
 * async form, since the depth buffer here can pick better than the CPU can.
 */
export class ComputeSink implements PointSink {
  private readonly blocks = new Map<number, Block>();
  private readonly alloc: BlockAllocator;
  private devNodes: GPUBuffer | undefined;
  private devTris: GPUBuffer | undefined;
  private devUniform: GPUBuffer | undefined;
  private devPipeline: GPUComputePipeline | undefined;
  private devLayout: GPUBindGroupLayout | undefined;
  private devBind: GPUBindGroup | undefined;
  private devBoundPos: GPUBuffer | undefined;
  private devToScene: [number, number, number] = [0, 0, 0];
  private devMaxDistance = 5;
  private capacity: number;
  private slotCount = 0;

  private posBuf: GPUBuffer;
  private colBuf: GPUBuffer;
  private pitchBuf: GPUBuffer;
  private metaBuf: GPUBuffer;
  private readonly liveBuf: GPUBuffer;
  private readonly cutBuf: GPUBuffer;
  private readonly uniBuf: GPUBuffer;

  /** The CPU mirror `readPoints` answers from. */
  private posCpu: Float32Array;
  private colCpu: Uint8Array;
  private scalarCpu: Float32Array | undefined;

  private readonly live = new Uint32Array(MAX_SLOTS);
  private readonly uniform = new ArrayBuffer(UNIFORM_BYTES);
  private readonly uf = new Float32Array(this.uniform);
  private readonly uu = new Uint32Array(this.uniform);
  /**
   * Cross-section planes in SCENE coordinates, `[nx, ny, nz, d]` each, four at
   * most; the positive half-space survives. The SAME array the overlay gets, so
   * a cut can never disagree between the points and the model (DEC-B6).
   */
  private clipPlanes: Float32Array | undefined;

  private bind: GPUBindGroup | undefined;
  private bindStale = true;

  private readonly layout: GPUBindGroupLayout;
  private readonly depthPipe: GPUComputePipeline;
  private readonly colorPipe: GPUComputePipeline;
  private boundGeneration = -1;

  private readonly mode: number;
  private cutDepth = 1;
  private maxLevel = 1;
  /** Bitmask das classes escondidas, lido pelo shader em qualquer modo. */
  private classHidden = 0;
  private zLo = Z_RANGE_OFF[0];
  private zHi = Z_RANGE_OFF[1];
  /**
   * Que códigos ASPRS esta nuvem realmente contém, um flag por código.
   *
   * Acumulado no `attach`, que já percorre todo ponto para empacotar a classe,
   * porque a alternativa é a UI listar as 20 classes do padrão numa nuvem que
   * tem quatro. Só cresce: um nó descarregado não retira o que já se viu, e
   * isso é o que se quer — a lista da UI não deve piscar com o streaming.
   */
  private readonly classPresent = new Uint8Array(256);
  private disposed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly raster: ComputeRasterizer,
    private readonly cut: OctreeCut,
    private readonly rootBox: {
      readonly min: readonly [number, number, number];
      readonly size: readonly [number, number, number];
    },
    private readonly options: ComputeSinkOptions,
    private readonly scalarAttribute: string | undefined = undefined,
  ) {
    this.mode = MODE_INDEX[options.colorMode.kind];
    // Sized to the budget with slack, then DOUBLED on demand rather than
    // refused: the arena grows a new slab when one fills, and a sink that
    // refused instead would draw less than the budget while every counter said
    // otherwise. That failure mode is not hypothetical — it is how the first
    // streaming build of this measured itself void.
    //
    // Capped at what the DEVICE can bind: posBuf is the widest lane at 12
    // bytes/point, and a bind group over the limit fails validation and every
    // pass after it goes invalid — the sink would render nothing at all, which
    // is strictly worse than rendering the capped budget.
    this.capacity = Math.min(
      this.maxBindablePoints(),
      Math.max(1 << 16, Math.ceil(options.pointBudget * 1.25)),
    );
    this.alloc = new BlockAllocator(this.capacity);
    this.posBuf = this.storage(this.capacity * 12);
    this.colBuf = this.storage(this.capacity * 4);
    this.pitchBuf = this.storage(this.capacity * 4);
    this.metaBuf = this.storage(this.capacity * 4);
    this.posCpu = new Float32Array(this.capacity * 3);
    this.colCpu = new Uint8Array(this.capacity * 4);
    if (scalarAttribute !== undefined) this.scalarCpu = new Float32Array(this.capacity);
    this.liveBuf = this.storage(MAX_SLOTS * 4);
    this.cutBuf = this.storage(Math.max(4, cut.bytes.byteLength));
    this.uniBuf = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const ro = { type: "read-only-storage" } as const;
    const rw = { type: "storage" } as const;
    const C = GPUShaderStage.COMPUTE;
    // All nine, because `depthPass` and `colorPass` touch all nine — which is
    // also exactly the WebGPU guarantee for storage buffers per stage, and why
    // level and node slot share one u32 instead of taking a binding each.
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: ro },
        { binding: 1, visibility: C, buffer: ro },
        { binding: 2, visibility: C, buffer: rw },
        { binding: 3, visibility: C, buffer: rw },
        { binding: 4, visibility: C, buffer: { type: "uniform" } },
        { binding: 5, visibility: C, buffer: ro },
        { binding: 6, visibility: C, buffer: ro },
        { binding: 7, visibility: C, buffer: ro },
        { binding: 8, visibility: C, buffer: ro },
      ],
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const compute = (entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({ layout: pl, compute: { module: raster.module, entryPoint } });
    this.depthPipe = compute("depthPass");
    this.colorPipe = compute("colorPass");
  }

  /** The largest point count whose position buffer still binds as storage. */
  private maxBindablePoints(): number {
    return Math.floor(this.device.limits.maxStorageBufferBindingSize / 12);
  }

  private storage(bytes: number): GPUBuffer {
    return this.device.createBuffer({
      size: Math.max(4, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  /**
   * B5 — carrega a malha contra a qual medir, e mede.
   *
   * Um pipeline SÓ SEU. O de pontos já usa os oito storage buffers que o
   * WebGPU garante por estágio; um nono não lança, faz o layout voltar
   * inválido e todos os passes silenciarem. Este usa quatro.
   *
   * `toScene` é o deslocamento nuvem->cena: os pontos vivem em coordenadas
   * locais da nuvem e a BVH em coordenadas de cena. Somar por ponto é mais
   * barato que reconstruir a árvore.
   *
   * O RESULTADO SUBSTITUI A COR. O escalar mora no `col` (o shader de pontos lê
   * `bitcast<f32>(col[i])` no modo 4), e é isso que faz o desvio herdar rampa,
   * faixa e legenda sem UI nova — ao preço de a cor RGB daquele ponto deixar de
   * existir até o nó ser reanexado.
   */
  setDeviationMesh(
    nodes: Float32Array,
    tris: Float32Array,
    toScene: readonly [number, number, number],
    maxDistance: number,
  ): void {
    if (this.disposed) return;
    if (nodes.length === 0 || tris.length === 0) {
      this.devNodes = undefined;
      return;
    }
    this.devNodes?.destroy();
    this.devTris?.destroy();
    this.devNodes = this.storage(nodes.byteLength);
    this.devTris = this.storage(tris.byteLength);
    this.device.queue.writeBuffer(this.devNodes, 0, nodes as unknown as BufferSource);
    this.device.queue.writeBuffer(this.devTris, 0, tris as unknown as BufferSource);

    this.devUniform ??= this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.devToScene = [toScene[0], toScene[1], toScene[2]];
    this.devMaxDistance = maxDistance;
    this.devBind = undefined;
  }

  /**
   * Corre o kernel sobre TODOS os pontos residentes.
   *
   * Sobre todos e não sobre os novos: acompanhar quais chegaram é contabilidade
   * de alocador para poupar um passo que já é barato perto do frame, e um
   * ponto que escapasse ficaria com a cor antiga no meio do mapa de desvio —
   * um erro que se lê como dado.
   */
  runDeviation(): number {
    if (this.disposed || this.devNodes === undefined || this.devTris === undefined) return 0;
    const count = this.alloc.used;
    if (count === 0) return 0;

    if (this.devPipeline === undefined) {
      const module = this.device.createShaderModule({ code: DEVIATION_WGSL });
      this.devLayout = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      this.devPipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.devLayout] }),
        compute: { module, entryPoint: "main" },
      });
    }

    // Rebind sempre que a arena cresce: `grow` TROCA os buffers, e um bind
    // group guardado apontaria para os antigos — que ainda existem, ainda leem,
    // e devolvem o desvio de outra cena.
    if (this.devBind === undefined || this.devBoundPos !== this.posBuf) {
      this.devBind = this.device.createBindGroup({
        layout: this.devLayout!,
        entries: [
          { binding: 0, resource: { buffer: this.posBuf } },
          { binding: 1, resource: { buffer: this.colBuf } },
          { binding: 2, resource: { buffer: this.devNodes } },
          { binding: 3, resource: { buffer: this.devTris } },
          { binding: 4, resource: { buffer: this.devUniform! } },
        ],
      });
      this.devBoundPos = this.posBuf;
    }

    // Espelha `struct U` de deviation-wgsl BYTE A BYTE. Ver o comentário lá
    // sobre por que o vec3 vem primeiro.
    const u = new ArrayBuffer(32);
    const f = new Float32Array(u);
    f[0] = this.devToScene[0];
    f[1] = this.devToScene[1];
    f[2] = this.devToScene[2];
    f[3] = this.devMaxDistance;
    new Uint32Array(u, 16, 1)[0] = count;
    this.device.queue.writeBuffer(this.devUniform!, 0, u);

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.devPipeline);
    pass.setBindGroup(0, this.devBind);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    this.device.queue.submit([enc.finish()]);
    return count;
  }

  /**
   * Lê de volta da GPU uma amostra de posições e desvios, para conferir o
   * kernel contra a referência em CPU (`distanceToBvh`).
   *
   * Existe porque `readPoints` devolve a cópia que a CPU subiu, e o kernel do
   * desvio só escreve na GPU: comparar contra ela compararia o dado com ele
   * mesmo. As posições saem em coordenadas LOCAIS DA NUVEM, que é como estão
   * no buffer — quem compara soma o mesmo `toScene` que o kernel somou.
   */
  async readDeviationSample(
    n: number,
  ): Promise<{ positions: Float32Array; deviations: Float32Array } | undefined> {
    if (this.disposed) return undefined;
    const count = Math.min(n, this.alloc.used);
    if (count === 0) return undefined;

    const posBytes = count * 12;
    const colBytes = count * 4;
    const rb = this.device.createBuffer({
      size: posBytes + colBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.posBuf, 0, rb, 0, posBytes);
    enc.copyBufferToBuffer(this.colBuf, 0, rb, posBytes, colBytes);
    this.device.queue.submit([enc.finish()]);

    await rb.mapAsync(GPUMapMode.READ);
    const positions = new Float32Array(rb.getMappedRange(0, posBytes).slice(0));
    const deviations = new Float32Array(rb.getMappedRange(posBytes, colBytes).slice(0));
    rb.unmap();
    rb.destroy();
    return { positions, deviations };
  }

  /** O offset que o kernel usou, para a verificação comparar com o mesmo. */
  get deviationToScene(): readonly [number, number, number] {
    return this.devToScene;
  }

  get residentPoints(): number {
    return this.alloc.used;
  }

  /** Capacity, not live points: this is what the GPU is actually holding. */
  get residentBytes(): number {
    return this.capacity * (this.scalarCpu === undefined ? 20 : 24);
  }

  get nodeCount(): number {
    return this.blocks.size;
  }

  /** Free runs outstanding. A rising count with flat `residentPoints` is
   *  fragmentation, which is the thing that would make growth spurious. */
  get freeRunCount(): number {
    return this.alloc.freeRunCount;
  }

  private grow(need: number): void {
    // Same ceiling as the constructor: past it the bind group fails validation,
    // so growth stops there and `attach` refuses the node instead.
    const max = this.maxBindablePoints();
    if (this.capacity >= max) return;
    let cap = this.capacity;
    while (cap < need && cap < max) cap *= 2;
    cap = Math.min(cap, max);
    const copy = (old: GPUBuffer, stride: number): GPUBuffer => {
      const next = this.storage(cap * stride);
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(old, 0, next, 0, this.capacity * stride);
      this.device.queue.submit([enc.finish()]);
      old.destroy();
      return next;
    };
    this.posBuf = copy(this.posBuf, 12);
    this.colBuf = copy(this.colBuf, 4);
    this.pitchBuf = copy(this.pitchBuf, 4);
    this.metaBuf = copy(this.metaBuf, 4);
    const pos = new Float32Array(cap * 3);
    pos.set(this.posCpu);
    this.posCpu = pos;
    const col = new Uint8Array(cap * 4);
    col.set(this.colCpu);
    this.colCpu = col;
    if (this.scalarCpu !== undefined) {
      const sc = new Float32Array(cap);
      sc.set(this.scalarCpu);
      this.scalarCpu = sc;
    }
    this.capacity = cap;
    this.alloc.setCapacity(cap);
    this.bindStale = true;
  }

  attach(
    index: number,
    data: DecodedPointData,
    spacingWorld: number,
    level: number,
  ): number {
    if (this.disposed) return 0;
    if (this.blocks.has(index)) return 0;
    if (data.numPoints === 0) return 0;
    // Cloud-local float32 is the frame the whole pipeline agrees on; anything
    // else would need a rebase this sink deliberately does not do, because the
    // reader already produced the right frame for the arena.
    if (!(data.positions instanceof Float32Array)) return 0;
    if (this.slotCount >= MAX_SLOTS) return 0;

    const n = data.numPoints;
    let start = this.alloc.allocate(n);
    if (start < 0) {
      this.grow(this.alloc.used + n);
      start = this.alloc.allocate(n);
      if (start < 0) return 0;
    }
    const slot = this.slotCount++;
    this.blocks.set(index, { start, count: n, slot, level });

    const q = this.device.queue;
    q.writeBuffer(this.posBuf, start * 12, gpuData(data.positions), 0, n * 3);
    this.posCpu.set(data.positions.subarray(0, n * 3), start * 3);

    let staged = data.positions.byteLength;
    const scalar =
      this.scalarAttribute === undefined
        ? undefined
        : (data.attributesByName.get(this.scalarAttribute)?.array as Float32Array | undefined);
    if (scalar !== undefined) {
      // The RAW value: a classification code stays an integer the palette can
      // index, and intensity is normalised in the shader from the declared
      // range — normalising per node would make one intensity two colours and
      // read as banding along node boundaries.
      q.writeBuffer(this.colBuf, start * 4, gpuData(scalar), 0, n);
      this.scalarCpu?.set(scalar.subarray(0, n), start);
      staged += 4 * n;
    } else {
      const colors = narrowColors(data.colors?.array);
      if (colors !== undefined) {
        // RGBA bytes ARE the u32 the shader reads, on a little-endian host:
        // byte 0 lands in the low 8 bits, which is where `shade` looks for red.
        //
        // O byte de topo é forçado a 255 porque ele deixou de ser ignorado: o
        // kernel do desvio carimba ali a distância à malha, e o corte esconde
        // o que estiver ABAIXO do limiar. Um ponto acabado de chegar ainda não
        // foi carimbado, e tem de nascer "longe" — nascer com o alfa da origem
        // faria a nuvem piscar buracos onde o streaming ainda não passou, com
        // o alfa 128 do cinza de fallback a esconder metade do levantamento.
        q.writeBuffer(this.colBuf, start * 4, gpuData(colors), 0, n * 4);
        this.colCpu.set(colors.subarray(0, n * 4), start * 4);
        staged += colors.byteLength;
      } else {
        // A cloud with no colour is normal, not an error: LAS point format 1
        // carries intensity and classification and no RGB.
        const grey = new Uint8Array(n * 4).fill(0x80);
        q.writeBuffer(this.colBuf, start * 4, gpuData(grey));
        this.colCpu.set(grey, start * 4);
      }
    }

    const pitch = new Float32Array(n).fill(spacingWorld);
    q.writeBuffer(this.pitchBuf, start * 4, gpuData(pitch));

    // A classe é lida à parte do escalar de propósito: no modo classificação os
    // dois são o mesmo atributo, mas em RGB ou intensidade o escalar é outro —
    // ou nenhum — e esconder uma classe tem de valer em todos.
    const classes = data.attributesByName.get(CLASS_ATTRIBUTE)?.array;
    const meta = packNodeMeta(level, slot, n, classes, this.classPresent);
    q.writeBuffer(this.metaBuf, start * 4, gpuData(meta));
    if (level > this.maxLevel) this.maxLevel = level;
    return staged;
  }

  detach(index: number): void {
    const block = this.blocks.get(index);
    if (block === undefined) return;
    this.alloc.release(block.start, block.count);
    this.blocks.delete(index);
    // The SLOT is retired, never reused, and that is what makes detaching free:
    // the freed points keep pointing at a slot whose `live` entry stays 0
    // forever, so a frame rejects them without rewriting a byte of point data.
    // Reusing slot ids would mean scrubbing the freed range on every detach.
    this.live[block.slot] = 0;
  }

  setVisible(indices: Int32Array, count: number): void {
    this.live.fill(0, 0, this.slotCount);
    // The cut is only as deep as the DRAWN set, so the shader's descent stops
    // where the data does instead of walking levels no node reached.
    let deepest = 0;
    for (let i = 0; i < count; i++) {
      const block = this.blocks.get(indices[i]!);
      if (block === undefined) continue;
      this.live[block.slot] = 1;
      if (block.level > deepest) deepest = block.level;
    }
    this.cutDepth = deepest + 1;
  }

  commit(): void {
    if (this.disposed) return;
    const q = this.device.queue;
    q.writeBuffer(this.liveBuf, 0, gpuData(this.live), 0, Math.max(1, this.slotCount));
    const bytes = this.cut.entryCount * 4;
    if (bytes > 0 && bytes <= this.cutBuf.size) {
      q.writeBuffer(this.cutBuf, 0, gpuData(this.cut.bytes), 0, bytes);
    }
  }

  readPoints(index: number): PointReadback | undefined {
    const block = this.blocks.get(index);
    if (block === undefined) return undefined;
    return {
      positions: this.posCpu,
      start: block.start,
      count: block.count,
      ...(this.scalarCpu === undefined
        ? { colors: this.colCpu }
        : { scalars: this.scalarCpu }),
    };
  }

  private rebind(): void {
    const depth = this.raster.depth;
    const accum = this.raster.accum;
    if (depth === undefined || accum === undefined) return;
    if (!this.bindStale && this.boundGeneration === this.raster.bufferGeneration) return;
    this.bind = this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.posBuf } },
        { binding: 1, resource: { buffer: this.colBuf } },
        { binding: 2, resource: { buffer: depth } },
        { binding: 3, resource: { buffer: accum } },
        { binding: 4, resource: { buffer: this.uniBuf } },
        { binding: 5, resource: { buffer: this.pitchBuf } },
        { binding: 6, resource: { buffer: this.metaBuf } },
        { binding: 7, resource: { buffer: this.cutBuf } },
        { binding: 8, resource: { buffer: this.liveBuf } },
      ],
    });
    this.bindStale = false;
    this.boundGeneration = this.raster.bufferGeneration;
  }

  /**
   * Accumulate this cloud's points into the frame the rasteriser cleared.
   *
   * `clipFromCloud` composes the camera with the cloud's own model matrix,
   * because the buffers hold CLOUD-LOCAL positions — the same frame the slab
   * arena stages, so both paths read one convention and neither rebases.
   */
  setClipPlanes(planes: Float32Array | undefined): void {
    this.clipPlanes = planes === undefined || planes.length === 0 ? undefined : planes;
  }

  /** Ver {@link PointCloudView.setHiddenClasses} — aqui é só o uniform. */
  setClassHidden(mask: number): void {
    this.classHidden = mask >>> 0;
  }

  /** Ver {@link PointCloudView.setZRange} — Z LOCAL DA NUVEM, já convertido. */
  setZRange(lo: number, hi: number): void {
    this.zLo = lo;
    this.zHi = hi;
  }

  /**
   * Os códigos ASPRS vistos até agora, crescente e sem repetição.
   *
   * Vazio numa nuvem sem classificação — que é diferente de "ainda não chegou
   * nó nenhum", e a UI tem de tratar os dois casos igual: não listar nada.
   */
  get presentClasses(): readonly number[] {
    const out: number[] = [];
    for (let code = 0; code < 256; code++) {
      if (this.classPresent[code] === 1) out.push(code);
    }
    return out;
  }

  dispatch(
    enc: GPUCommandEncoder,
    camera: PerspectiveCamera,
    modelMatrix: Matrix4,
    width: number,
    height: number,
  ): void {
    if (this.disposed || this.alloc.used === 0) return;
    this.rebind();
    if (this.bind === undefined) return;
    const o = this.options;

    const m = toZeroToOneDepth(
      camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse).multiply(modelMatrix),
    );
    this.uf.set(m.elements, 0);
    this.uf[16] = width;
    this.uf[17] = height;
    // The high-water mark, not the live count: a free list scatters live nodes,
    // so the frame covers everything resident and rejects per point. That is
    // the instanced arm's draw-to-high-water model exactly, one dispatch
    // instead of one draw.
    this.uu[18] = this.alloc.used;
    this.uu[19] = 1;
    this.uf[20] = camera.projectionMatrix.elements[5]!;
    this.uf[21] = o.sizeMultiplier;
    this.uf[22] = o.minPixelSize;
    this.uf[23] = o.maxPixelSize;
    this.uf[24] = this.rootBox.min[0];
    this.uf[25] = this.rootBox.min[1];
    this.uf[26] = this.rootBox.min[2];
    this.uu[27] = this.cutDepth + 1;
    this.uf[28] = this.rootBox.size[0];
    this.uf[29] = this.rootBox.size[1];
    this.uf[30] = this.rootBox.size[2];
    this.uu[31] = this.cut.entryCount > 0 ? 1 : 0;
    const flat = o.colorMode.kind === "flat" ? o.colorMode.color : [0, 0, 0];
    this.uf[32] = flat[0]!;
    this.uf[33] = flat[1]!;
    this.uf[34] = flat[2]!;
    this.uu[35] = this.mode;
    this.uf[36] = o.elevationRange[0];
    this.uf[37] = o.elevationRange[1];
    this.uf[38] = o.scalarRange[0];
    this.uf[39] = o.scalarRange[1];
    this.uf[40] = this.maxLevel;

    // DEC-B6. De cena para local-da-nuvem é a translação do modelo, e para um
    // plano covariante isso é só ajustar a distância: dot(n, p+t)+d >= 0 vira
    // dot(n, p) + (d + dot(n, t)) >= 0. Sem rotação envolvida, sem transposta.
    const planes = this.clipPlanes;
    const nPlanes = planes === undefined ? 0 : Math.min(4, planes.length >> 2);
    this.uf[46] = nPlanes;
    // O slot 47 era o padding antes de 'clip'; agora carrega a máscara de
    // classes escondidas (u32), e o array continua alinhado a 16 bytes.
    this.uu[47] = this.classHidden;
    // Slots 64/65: a faixa de altura, depois do array de planos. Ver
    // {@link UNIFORM_BYTES} — foi por causa destes dois que ele cresceu.
    this.uf[64] = this.zLo;
    this.uf[65] = this.zHi;
    const e = modelMatrix.elements;
    for (let i = 0; i < nPlanes; i++) {
      const nx = planes![i * 4]!;
      const ny = planes![i * 4 + 1]!;
      const nz = planes![i * 4 + 2]!;
      const d = planes![i * 4 + 3]!;
      const k = 48 + i * 4;
      this.uf[k] = nx;
      this.uf[k + 1] = ny;
      this.uf[k + 2] = nz;
      this.uf[k + 3] = d + nx * e[12]! + ny * e[13]! + nz * e[14]!;
    }
    this.device.queue.writeBuffer(this.uniBuf, 0, this.uniform);

    const groups = Math.ceil(this.alloc.used / WORKGROUP);
    const cp = enc.beginComputePass();
    cp.setBindGroup(0, this.bind);
    cp.setPipeline(this.depthPipe);
    cp.dispatchWorkgroups(groups);
    cp.setPipeline(this.colorPipe);
    cp.dispatchWorkgroups(groups);
    cp.end();
  }


  dispose(): void {
    this.devNodes?.destroy();
    this.devTris?.destroy();
    this.devUniform?.destroy();
    if (this.disposed) return;
    this.disposed = true;
    for (const b of [
      this.posBuf,
      this.colBuf,
      this.pitchBuf,
      this.metaBuf,
      this.liveBuf,
      this.cutBuf,
      this.uniBuf,
    ]) {
      b?.destroy();
    }
    this.blocks.clear();
    this.alloc.reset();
  }
}
