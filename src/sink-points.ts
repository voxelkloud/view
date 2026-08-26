import type { DecodedPointData } from "@voxelkloud/format-potree";
import type { Matrix4, PerspectiveCamera } from "three";
import type { OctreeCut } from "./cut.js";
import type { ColorMode } from "./material.js";
import { FS, FS_EDL, VS, VS_POST } from "./points-glsl.js";
import { BlockAllocator, MAX_SLOTS, packNodeMeta } from "./sink-compute.js";
import { CLASS_ATTRIBUTE } from "./material-options.js";
import type { PointReadback, PointSink } from "./sink.js";

// NO GPU PICK HERE, deliberately, and the id-writing shader in points-glsl.ts
// is what a future one would use.
//
// It would be a better pick than the CPU one: WebGL 2's `readPixels` is
// synchronous, so unlike the compute path it needs no async API, and the depth
// test filters visibility for free — a point that lost it never wrote its id, so
// what comes back is only what the user can see. `pick.ts` ranks by screen
// distance with depth as a tiebreak and can return a point hidden behind the
// splat that was actually clicked.
//
// It is not here because it would save nothing yet: `readPoints` also feeds
// `ground.ts`, so the CPU mirror stays either way, and a linked program with no
// caller is dead weight.

/** The index the shader switches on, in the order {@link ColorMode} declares. */
const MODE_INDEX: Record<ColorMode["kind"], number> = {
  rgb: 0,
  flat: 1,
  elevation: 2,
  level: 3,
  intensity: 4,
  classification: 5,
  // O caminho WebGL 2 desenha a mesma via escalar, mas NÃO tem o kernel que a
  // preenche: o desvio é um passo de compute, e este arm é o fallback para
  // quem não tem WebGPU. Aqui o modo existe para o tipo fechar e desenha o
  // escalar que houver — que sem o kernel é o atributo original da nuvem.
  deviation: 4,
};

export interface PointsSinkOptions {
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

const CUT_W = 1024;
const LIVE_W = 1024;

/**
 * A {@link PointSink} that draws its own points with `gl.POINTS` on WebGL 2.
 *
 * Why it exists, in one measurement: the instanced path through three's WebGL
 * fallback costs ~40 ns per point — 8.5 fps at a 3M budget, against Potree's
 * 59.9 and 88 ms of INP. Neither our shader nor fill rate explains it. Pinning
 * the splat to 1 px changed nothing, and disabling the octree cut walk entirely
 * changed nothing. It is the backend's instanced draw, and no amount of shader
 * work reaches it.
 *
 * `gl_PointSize` is the way out and is exactly what Potree uses. It cannot be
 * reached through three's node system — `GLSLNodeBuilder` writes
 * `gl_PointSize = 1.0` after our code, the GLSL mirror of WGSL having no
 * point-size builtin at all — so this owns the draw, as {@link ComputeSink}
 * does on WebGPU.
 *
 * THE LEVER is draw calls, not per-point cost. Potree creates a `THREE.Points`
 * per octree NODE: about two thousand draws for a 3M frame on autzen, at a
 * ~1446-point median node. This keeps every point in one buffer set and issues
 * ONE, dispatching over the high-water mark and rejecting per point — the same
 * model the compute rasteriser uses, and the reason both land at 72 ms where
 * the path they replace sits at 656.
 */
export class PointsSink implements PointSink {
  private readonly blocks = new Map<number, { start: number; count: number; slot: number; level: number }>();
  private readonly alloc: BlockAllocator;
  private capacity: number;
  private slotCount = 0;
  private refusedNodes = 0;
  private evictedNodes = 0;
  private grewTimes = 0;
  private growMs = 0;
  /** Only delete the program if this sink built it; a shared one outlives it. */
  private readonly ownsProg: boolean = true;
  private maxAttachMs = 0;
  private maxCommitMs = 0;
  private attachCalls = 0;
  private frameNo = 0;

  private posBuf: WebGLBuffer;
  private colBuf: WebGLBuffer;
  private pitchBuf: WebGLBuffer;
  private metaBuf: WebGLBuffer;
  private vao: WebGLVertexArrayObject;

  /** The CPU mirror `readPoints` answers from, which is what keeps `pickPoint`
   *  synchronous. The same memory the slab arena already spends. */
  private posCpu: Float32Array;
  private colCpu: Uint8Array;
  private scalarCpu: Float32Array | undefined;
  /** Ver {@link ComputeSink.setClassHidden} — aqui é o mesmo uniform, em GL. */
  private classHidden = 0;
  /** Ver {@link ComputeSink.presentClasses}. Preenchido por `packNodeMeta`. */
  private readonly classPresent = new Uint8Array(256);

  private readonly live = new Uint8Array(MAX_SLOTS);
  private readonly lastLive: number[] = [];
  private readonly nodeOfSlot: (number | undefined)[] = [];

  private readonly prog: WebGLProgram;
  private readonly cutTex: WebGLTexture;
  private readonly liveTex: WebGLTexture;
  private readonly emptyVao: WebGLVertexArrayObject;
  private readonly locCache = new Map<WebGLProgram, Record<string, WebGLUniformLocation | null>>();
  private readonly mode: number;
  private cutDepth = 1;
  private maxLevel = 1;
  private disposed = false;

  /** `ALIASED_POINT_SIZE_RANGE`. A hardware ceiling with no guaranteed value,
   *  and the one limit `gl.POINTS` has that instanced quads do not. */
  readonly maxPointSizePx: number;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    private readonly cut: OctreeCut,
    private readonly rootBox: {
      readonly min: readonly [number, number, number];
      readonly size: readonly [number, number, number];
    },
    private readonly options: PointsSinkOptions,
    private readonly scalarAttribute: string | undefined = undefined,
    /** Shared, and already warmed, from `PointsRasterizer`. */
    sharedProgram?: WebGLProgram,
  ) {
    this.mode = MODE_INDEX[options.colorMode.kind];
    // 1.6x, not 1.25. Under a moving camera the frontier pulls in nodes before
    // the ones it left go cold, and the compute spike measured that settling
    // near 1.6x the budget. Too tight a capacity does not fail — it evicts and
    // reloads in a loop, or grows.
    this.capacity = Math.max(1 << 16, Math.ceil(options.pointBudget * 1.6));
    this.alloc = new BlockAllocator(this.capacity);
    this.maxPointSizePx =
      (gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array)[1] ?? 0;

    this.prog = sharedProgram ?? link(gl, VS, FS, "points");
    this.ownsProg = sharedProgram === undefined;

    this.posBuf = buffer(gl, this.capacity * 12);
    this.colBuf = buffer(gl, this.capacity * 4);
    this.pitchBuf = buffer(gl, this.capacity * 4);
    this.metaBuf = buffer(gl, this.capacity * 4);
    this.posCpu = new Float32Array(this.capacity * 3);
    this.colCpu = new Uint8Array(this.capacity * 4);
    if (scalarAttribute !== undefined) this.scalarCpu = new Float32Array(this.capacity);
    this.vao = this.buildVao();
    this.emptyVao = gl.createVertexArray()!;

    this.cutTex = intTexture(gl, gl.RGBA8UI, CUT_W, Math.max(1, Math.ceil(cut.bytes.byteLength / 4 / CUT_W)));
    this.liveTex = intTexture(gl, gl.R8UI, LIVE_W, MAX_SLOTS / LIVE_W);
  }

  private buildVao(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const bind = (buf: WebGLBuffer, loc: number, size: number, type: number, norm: boolean): void => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, type, norm, 0, 0);
    };
    bind(this.posBuf, gl.getAttribLocation(this.prog, "aPos"), 3, gl.FLOAT, false);
    bind(this.colBuf, gl.getAttribLocation(this.prog, "aColor"), 4, gl.UNSIGNED_BYTE, true);
    bind(this.pitchBuf, gl.getAttribLocation(this.prog, "aPitch"), 1, gl.FLOAT, false);
    // INTEIRO, não float: `vertexAttribIPointer` entrega os 32 bits sem passar
    // por uma conversão para float, que é o que faz o byte da classe caber.
    gl.bindBuffer(gl.ARRAY_BUFFER, this.metaBuf);
    const metaLoc = gl.getAttribLocation(this.prog, "aMeta");
    gl.enableVertexAttribArray(metaLoc);
    gl.vertexAttribIPointer(metaLoc, 1, gl.UNSIGNED_INT, 0, 0);
    gl.bindVertexArray(null);
    return vao;
  }

  get residentPoints(): number {
    return this.alloc.used;
  }
  get residentBytes(): number {
    return this.capacity * (this.scalarCpu === undefined ? 20 : 24);
  }
  get nodeCount(): number {
    return this.blocks.size;
  }
  /** Non-zero VOIDS a measurement: the allocator refused nodes the cut asked
   *  for, so the frame drew less than the budget. */
  get refused(): number {
    return this.refusedNodes;
  }
  get evicted(): number {
    return this.evictedNodes;
  }
  /** Buffer doublings and what they cost. A reallocation copies every resident
   *  byte, so one landing mid-drag is a hitch the INP measurement will find. */
  get grew(): { times: number; ms: number } {
    return { times: this.grewTimes, ms: Math.round(this.growMs) };
  }
  /** Worst single call of each, which is what an INP measurement finds. */
  get worst(): { attachMs: number; commitMs: number; attaches: number } {
    return {
      attachMs: +this.maxAttachMs.toFixed(1),
      commitMs: +this.maxCommitMs.toFixed(1),
      attaches: this.attachCalls,
    };
  }
  resetWorst(): void {
    this.maxAttachMs = 0;
    this.maxCommitMs = 0;
  }

  private grow(need: number): void {
    const t0 = performance.now();
    this.grewTimes++;
    const gl = this.gl;
    let cap = this.capacity;
    while (cap < need) cap *= 2;
    const copy = (old: WebGLBuffer, stride: number): WebGLBuffer => {
      const next = buffer(gl, cap * stride);
      gl.bindBuffer(gl.COPY_READ_BUFFER, old);
      gl.bindBuffer(gl.COPY_WRITE_BUFFER, next);
      gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, 0, 0, this.capacity * stride);
      gl.deleteBuffer(old);
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
    gl.deleteVertexArray(this.vao);
    this.vao = this.buildVao();
    this.growMs += performance.now() - t0;
  }

  private evictOne(): boolean {
    let victim = -1;
    let oldest = this.frameNo;
    for (let slot = 0; slot < this.slotCount; slot++) {
      if (this.nodeOfSlot[slot] === undefined) continue;
      if (this.lastLive[slot]! >= this.frameNo) continue;
      if (this.lastLive[slot]! < oldest) {
        oldest = this.lastLive[slot]!;
        victim = slot;
      }
    }
    if (victim < 0) return false;
    const nodeIndex = this.nodeOfSlot[victim]!;
    const b = this.blocks.get(nodeIndex)!;
    this.alloc.release(b.start, b.count);
    this.blocks.delete(nodeIndex);
    // The SLOT is retired, never reused: the freed points keep pointing at a
    // slot whose `live` byte stays 0 forever, so a frame rejects them without
    // rewriting a byte of point data.
    this.nodeOfSlot[victim] = undefined;
    this.live[victim] = 0;
    this.evictedNodes++;
    return true;
  }

  attach(index: number, data: DecodedPointData, spacingWorld: number, level: number): number {
    const tA = performance.now();
    const staged = this.attachInner(index, data, spacingWorld, level);
    const dt = performance.now() - tA;
    if (dt > this.maxAttachMs) this.maxAttachMs = dt;
    this.attachCalls++;
    return staged;
  }

  private attachInner(index: number, data: DecodedPointData, spacingWorld: number, level: number): number {
    if (this.disposed || this.blocks.has(index) || data.numPoints === 0) return 0;
    if (!(data.positions instanceof Float32Array)) return 0;
    if (this.slotCount >= MAX_SLOTS) {
      this.refusedNodes++;
      return 0;
    }
    const n = data.numPoints;
    // EVICT FIRST, grow only when eviction has nothing left to give.
    //
    // The order matters more than it looks. A doubling copies every resident
    // byte — measured at 142 ms for one call mid-drag, which is the whole INP
    // number by itself, because INP reports the WORST frame of an interaction.
    // Releasing a run costs microseconds. Growing first meant paying the
    // expensive fix for a problem the cheap one solves: under a moving camera
    // the LOD frontier advances and residency climbed from 3.0M to 4.8M, so the
    // allocator hit its ceiling and doubled instead of dropping what had gone
    // cold. Growth is now what happens when everything resident is genuinely
    // still being drawn, which is a budget that does not fit rather than a
    // frontier that moved.
    let start = this.alloc.allocate(n);
    while (start < 0) {
      if (this.evictOne()) {
        start = this.alloc.allocate(n);
        continue;
      }
      if (this.capacity * 2 > (1 << 26)) {
        this.refusedNodes++;
        return 0;
      }
      this.grow(this.alloc.used + n);
      start = this.alloc.allocate(n);
    }

    const gl = this.gl;
    const slot = this.slotCount++;
    this.nodeOfSlot.push(index);
    this.lastLive.push(this.frameNo);
    this.blocks.set(index, { start, count: n, slot, level });
    if (level > this.maxLevel) this.maxLevel = level;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, start * 12, data.positions, 0, n * 3);
    this.posCpu.set(data.positions.subarray(0, n * 3), start * 3);
    let staged = data.positions.byteLength;

    const scalar =
      this.scalarAttribute === undefined
        ? undefined
        : (data.attributesByName.get(this.scalarAttribute)?.array as Float32Array | undefined);
    const colors = data.colors?.array instanceof Uint8Array ? data.colors.array : undefined;
    const rgba = new Uint8Array(n * 4);
    if (scalar !== undefined) {
      // Little-endian into the first two bytes: a 16-bit intensity does not fit
      // one normalised byte, and the shader reassembles it. The RAW value —
      // a classification code stays an integer the palette can index, and
      // intensity is normalised in the shader from the declared range.
      for (let k = 0; k < n; k++) {
        const v = Math.max(0, Math.round(scalar[k] ?? 0));
        rgba[k * 4] = v & 0xff;
        rgba[k * 4 + 1] = (v >> 8) & 0xff;
        rgba[k * 4 + 3] = 255;
      }
      this.scalarCpu?.set(scalar.subarray(0, n), start);
      staged += 4 * n;
    } else if (colors !== undefined) {
      rgba.set(colors.subarray(0, n * 4));
      for (let k = 0; k < n; k++) rgba[k * 4 + 3] = 255;
      staged += colors.byteLength;
    } else {
      // A cloud with no colour is normal, not an error: LAS point format 1
      // carries intensity and classification and no RGB.
      rgba.fill(0x80);
      for (let k = 0; k < n; k++) rgba[k * 4 + 3] = 255;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, start * 4, rgba);
    this.colCpu.set(rgba, start * 4);

    const pitch = new Float32Array(n).fill(spacingWorld);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pitchBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, start * 4, pitch);
    // Nível, slot e CLASSE num atributo só, no mesmo layout do braço compute —
    // ver {@link packNodeMeta}. Era `slot * 32 + level` num float, e a classe
    // não cabia ali: level, slot e classe somam 29 bits e float32 só é exato
    // até 2^24. Ligado como INTEIRO o mesmo lane de 4 bytes dá os 32, então a
    // classe entra sem custo nenhum — e os dois rasterizadores passam a
    // empacotar pelo mesmo código, que é o que impede os dois layouts de
    // divergirem em silêncio.
    const classes = data.attributesByName.get(CLASS_ATTRIBUTE)?.array;
    const meta = packNodeMeta(level, slot, n, classes, this.classPresent);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.metaBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, start * 4, meta);
    return staged;
  }

  detach(index: number): void {
    const b = this.blocks.get(index);
    if (b === undefined) return;
    this.alloc.release(b.start, b.count);
    this.blocks.delete(index);
    this.nodeOfSlot[b.slot] = undefined;
    this.live[b.slot] = 0;
  }

  setVisible(indices: Int32Array, count: number): void {
    this.frameNo++;
    this.live.fill(0, 0, this.slotCount);
    let deepest = 0;
    for (let i = 0; i < count; i++) {
      const b = this.blocks.get(indices[i]!);
      if (b === undefined) continue;
      this.live[b.slot] = 1;
      this.lastLive[b.slot] = this.frameNo;
      if (b.level > deepest) deepest = b.level;
    }
    this.cutDepth = deepest + 1;
  }

  commit(): void {
    if (this.disposed) return;
    const tC = performance.now();
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.liveTex);
    gl.texSubImage2D(
      gl.TEXTURE_2D, 0, 0, 0, LIVE_W, Math.max(1, Math.ceil(this.slotCount / LIVE_W)),
      gl.RED_INTEGER, gl.UNSIGNED_BYTE, this.live, 0,
    );
    if (this.cut.entryCount > 0) {
      gl.bindTexture(gl.TEXTURE_2D, this.cutTex);
      gl.texSubImage2D(
        gl.TEXTURE_2D, 0, 0, 0, CUT_W, Math.max(1, Math.ceil(this.cut.entryCount / CUT_W)),
        gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, this.cut.bytes, 0,
      );
    }
    const dt = performance.now() - tC;
    if (dt > this.maxCommitMs) this.maxCommitMs = dt;
  }

  readPoints(index: number): PointReadback | undefined {
    const b = this.blocks.get(index);
    if (b === undefined) return undefined;
    return {
      positions: this.posCpu,
      start: b.start,
      count: b.count,
      ...(this.scalarCpu === undefined ? { colors: this.colCpu } : { scalars: this.scalarCpu }),
    };
  }

  private uniforms(program: WebGLProgram): Record<string, WebGLUniformLocation | null> {
    let u = this.locCache.get(program);
    if (u === undefined) {
      // Looked up PER PROGRAM and cached: a location belongs to one program
      // object, and reusing one across programs writes to the wrong slot in
      // silence.
      const names = [
        "uClipFromCloud", "uViewFromCloud", "uProjScale", "uSizeMul", "uMinPx",
        "uMaxPx", "uCut", "uLive", "uUseMask", "uRootMin", "uRootSize",
        "uCutDepth", "uUseCut", "uMode", "uFlatColor", "uElevRange",
        "uScalarRange", "uMaxLevel", "uRound", "uClassHidden",
      ];
      u = Object.fromEntries(names.map((n) => [n, this.gl.getUniformLocation(program, n)]));
      this.locCache.set(program, u);
    }
    return u;
  }

  private bindUniforms(
    program: WebGLProgram,
    camera: PerspectiveCamera,
    clipFromCloud: Matrix4,
    viewFromCloud: Matrix4,
    height: number,
  ): void {
    const gl = this.gl;
    const u = this.uniforms(program);
    const o = this.options;
    gl.uniformMatrix4fv(u["uClipFromCloud"]!, false, clipFromCloud.elements);
    gl.uniformMatrix4fv(u["uViewFromCloud"]!, false, viewFromCloud.elements);
    gl.uniform1f(u["uProjScale"]!, 0.5 * height * camera.projectionMatrix.elements[5]!);
    gl.uniform1f(u["uSizeMul"]!, o.sizeMultiplier);
    gl.uniform1f(u["uMinPx"]!, o.minPixelSize);
    gl.uniform1f(u["uMaxPx"]!, o.maxPixelSize);
    gl.uniform1ui(u["uClassHidden"]!, this.classHidden);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cutTex);
    gl.uniform1i(u["uCut"]!, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.liveTex);
    gl.uniform1i(u["uLive"]!, 1);
    gl.uniform1i(u["uUseMask"]!, 1);
    gl.uniform3f(u["uRootMin"]!, this.rootBox.min[0], this.rootBox.min[1], this.rootBox.min[2]);
    gl.uniform3f(u["uRootSize"]!, this.rootBox.size[0], this.rootBox.size[1], this.rootBox.size[2]);
    gl.uniform1i(u["uCutDepth"]!, this.cutDepth + 1);
    gl.uniform1i(u["uUseCut"]!, this.cut.entryCount > 0 ? 1 : 0);
    gl.uniform1i(u["uRound"]!, 1);
    gl.uniform1i(u["uMode"]!, this.mode);
    const flat = o.colorMode.kind === "flat" ? o.colorMode.color : [0, 0, 0];
    gl.uniform3f(u["uFlatColor"]!, flat[0]!, flat[1]!, flat[2]!);
    gl.uniform2f(u["uElevRange"]!, o.elevationRange[0], o.elevationRange[1]);
    gl.uniform2f(u["uScalarRange"]!, o.scalarRange[0], o.scalarRange[1]);
    gl.uniform1f(u["uMaxLevel"]!, Math.max(1, this.maxLevel));
  }

  /**
   * Draw this cloud's points. ONE `drawArrays` over the high-water mark, with
   * the per-point mask rejecting what this frame did not select — the same
   * draw-to-high-water model the slab arena uses, and the reason a 3M frame is
   * one draw call where Potree issues about two thousand.
   *
   * It does NOT clear and it does NOT bind a framebuffer. Both are per FRAME,
   * not per cloud: a sink that cleared for itself would erase the cloud drawn
   * before it, which is exactly how the compute path had to be split into a
   * rasteriser and a sink after the fact.
   */
  draw(
    camera: PerspectiveCamera,
    clipFromCloud: Matrix4,
    viewFromCloud: Matrix4,
    height: number,
  ): void {
    if (this.disposed || this.alloc.used === 0) return;
    const gl = this.gl;
    gl.useProgram(this.prog);
    this.bindUniforms(this.prog, camera, clipFromCloud, viewFromCloud, height);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.alloc.used);
    gl.bindVertexArray(null);
  }

  /** Ver {@link PointCloudView.setHiddenClasses} — aqui é só o uniform. */
  setClassHidden(mask: number): void {
    this.classHidden = mask >>> 0;
  }

  /** Ver {@link PointCloudView.presentClasses}. */
  get presentClasses(): readonly number[] {
    const out: number[] = [];
    for (let code = 0; code < 256; code++) {
      if (this.classPresent[code] === 1) out.push(code);
    }
    return out;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    for (const b of [this.posBuf, this.colBuf, this.pitchBuf, this.metaBuf]) gl.deleteBuffer(b);
    gl.deleteVertexArray(this.vao);
    gl.deleteVertexArray(this.emptyVao);
    gl.deleteTexture(this.cutTex);
    gl.deleteTexture(this.liveTex);
    if (this.ownsProg) gl.deleteProgram(this.prog);

    this.blocks.clear();
    this.alloc.reset();
  }
}

/**
 * Non-fatal GL shader messages, drained into a shared array.
 *
 * `getShaderInfoLog` is not only for failures: a shader that compiles and
 * links can still carry a driver note, and on WebGL that note is the only
 * warning channel there is. It is also where "compiled fine here, black there"
 * usually leaves its fingerprint.
 */
function collectGlMessages(
  gl: WebGL2RenderingContext,
  prog: WebGLProgram,
  shaders: readonly WebGLShader[],
  label: string,
  into: string[] | undefined,
): void {
  if (into === undefined) return;
  const push = (what: string, log: string | null): void => {
    const text = (log ?? "").trim();
    if (text === "" || into.length >= 24) return;
    into.push(`${label}/${what}: ${text.slice(0, 300)}`);
  };
  for (const sh of shaders) push("shader", gl.getShaderInfoLog(sh));
  push("link", gl.getProgramInfoLog(prog));
}

function link(
  gl: WebGL2RenderingContext,
  vs: string,
  fs: string,
  label: string,
  warnings?: string[],
): WebGLProgram {
  const compile = (type: number, src: string): WebGLShader => {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(`${label} shader: ${gl.getShaderInfoLog(sh) ?? "?"}`);
    }
    return sh;
  };
  const prog = gl.createProgram()!;
  const shv = compile(gl.VERTEX_SHADER, vs);
  const shf = compile(gl.FRAGMENT_SHADER, fs);
  gl.attachShader(prog, shv);
  gl.attachShader(prog, shf);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`${label} link: ${gl.getProgramInfoLog(prog) ?? "?"}`);
  }
  collectGlMessages(gl, prog, [shv, shf], label, warnings);
  return prog;
}

function buffer(gl: WebGL2RenderingContext, bytes: number): WebGLBuffer {
  const b = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, Math.max(4, bytes), gl.DYNAMIC_DRAW);
  return b;
}

function intTexture(gl: WebGL2RenderingContext, fmt: number, w: number, h: number): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, Math.max(1, h));
  return t;
}

/**
 * The screen-sized half of the WebGL 2 points path, owned by the VIEW.
 *
 * The split is not tidiness: clearing and the EDL post-process are per FRAME,
 * not per cloud. A sink that cleared for itself would erase the cloud drawn
 * before it, and EDL reading a target one sink owned would shade only that
 * sink's points. The compute rasteriser learned this the same way — after the
 * fact — and this one is built with the seam already in place.
 */
export class PointsRasterizer {
  /**
   * The point program, linked HERE rather than in the sink.
   *
   * `VS`/`FS` are constants, so the program never depended on the cloud — but
   * it was built in `PointsSink`, which is constructed in `addCloud`, which
   * runs only after metadata AND hierarchy have landed. Linking it here puts
   * the compile alongside the network instead of after it, and every cloud
   * shares the one program.
   */
  readonly pointProg: WebGLProgram;
  private readonly edlProg: WebGLProgram;
  private readonly emptyVao: WebGLVertexArrayObject;
  private target: WebGLFramebuffer | null = null;
  private targetColor: WebGLTexture | null = null;
  private targetDepth: WebGLTexture | null = null;
  private w = 0;
  private h = 0;
  private disposed = false;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    private readonly options: {
      readonly background: readonly [number, number, number];
      readonly edl?: { readonly strength: number; readonly radius: number } | undefined;
      /** Where non-fatal shader messages go. Shared with the view. */
      readonly warnings?: string[] | undefined;
    },
  ) {
    this.edlProg = link(gl, VS_POST, FS_EDL, "points-edl", options.warnings);
    this.emptyVao = gl.createVertexArray()!;
    this.pointProg = link(gl, VS, FS, "points", options.warnings);
    // WARM IT. `linkProgram` succeeding does not mean the driver has generated
    // machine code: most defer that to the first draw that uses the program,
    // and the first draw here is the first frame the user waits for. One
    // vertex, with colour writes masked off and no attributes bound, forces the
    // work now — while the hierarchy is still on the wire.
    gl.colorMask(false, false, false, false);
    gl.bindVertexArray(this.emptyVao);
    gl.useProgram(this.pointProg);
    gl.drawArrays(gl.POINTS, 0, 1);
    gl.bindVertexArray(null);
    gl.useProgram(null);
    gl.colorMask(true, true, true, true);
  }

  private get edlOn(): boolean {
    const e = this.options.edl;
    return e !== undefined && e.strength > 0;
  }

  // EDL needs the depth SAMPLEABLE, which the default framebuffer's is not — so
  // the scene goes to a target with a depth TEXTURE attached and a fullscreen
  // pass reads it. The compute rasteriser avoids this entirely: its depth is
  // already a storage buffer its resolve can read.
  private ensureTarget(width: number, height: number): void {
    if (this.w === width && this.h === height && this.target !== null) return;
    const gl = this.gl;
    this.w = width;
    this.h = height;
    if (this.target !== null) gl.deleteFramebuffer(this.target);
    if (this.targetColor !== null) gl.deleteTexture(this.targetColor);
    if (this.targetDepth !== null) gl.deleteTexture(this.targetDepth);
    const mk = (fmt: number): WebGLTexture => {
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, width, height);
      return t;
    };
    this.targetColor = mk(gl.RGBA8);
    this.targetDepth = mk(gl.DEPTH_COMPONENT24);
    this.target = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.target);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.targetColor, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.targetDepth, 0);
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (st !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`points EDL target incomplete: 0x${st.toString(16)}`);
    }
  }

  /** Clear the frame. Every cloud then draws into it, sharing one depth buffer,
   *  which is what makes clouds occlude each other. */
  begin(width: number, height: number): void {
    if (this.disposed) return;
    const gl = this.gl;
    if (this.edlOn) {
      this.ensureTarget(width, height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.target);
    }
    gl.viewport(0, 0, width, height);
    // STATE THIS PATH OWNS, set explicitly rather than inherited. three's
    // renderer is alive alongside this one and leaves its own state behind —
    // blending in particular, which turns every splat into a read-modify-write
    // and at this overdraw is the difference between 64 ms and 128 ms of INP for
    // the identical draw. A renderer that borrows state measures whatever the
    // last one left.
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.SCISSOR_TEST);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.depthFunc(gl.LESS);
    gl.colorMask(true, true, true, true);
    const bg = this.options.background;
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  /** Resolve to the screen. A no-op without EDL, which draws straight through. */
  end(camera: PerspectiveCamera, width: number, height: number): void {
    if (this.disposed || !this.edlOn || this.target === null) return;
    const gl = this.gl;
    const e = this.options.edl!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.edlProg);
    const g = (n: string): WebGLUniformLocation | null => gl.getUniformLocation(this.edlProg, n);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.targetColor);
    gl.uniform1i(g("uColor"), 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.targetDepth);
    gl.uniform1i(g("uDepth"), 3);
    gl.uniform2f(g("uTexel"), 1 / width, 1 / height);
    gl.uniform1f(g("uNear"), camera.near);
    gl.uniform1f(g("uFar"), camera.far);
    gl.uniform1f(g("uStrength"), e.strength);
    gl.uniform1f(g("uRadius"), e.radius);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    gl.enable(gl.DEPTH_TEST);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    gl.deleteProgram(this.edlProg);
    gl.deleteVertexArray(this.emptyVao);
    if (this.target !== null) gl.deleteFramebuffer(this.target);
    if (this.targetColor !== null) gl.deleteTexture(this.targetColor);
    if (this.targetDepth !== null) gl.deleteTexture(this.targetDepth);
  }
}
