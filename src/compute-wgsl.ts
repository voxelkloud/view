/**
 * The compute rasteriser's shader, as one module.
 *
 * Kept apart from {@link ComputeSink} so the WGSL reads as WGSL: it is the part
 * a graphics reviewer checks, and burying it inside allocator bookkeeping makes
 * that harder than it needs to be.
 *
 * Eight storage buffers is not a style choice, it is the WebGPU guarantee for
 * `maxStorageBuffersPerShaderStage`. Level and node slot therefore share one
 * u32, and the colour ramp and ASPRS palette are compile-time constants rather
 * than buffers. Exceeding the limit does NOT throw — the layout comes back
 * invalid and every pass silently no-ops, which renders a black frame while
 * every CPU-side counter stays correct.
 */
export const COMPUTE_WGSL = `
struct U {
  clipFromCloud : mat4x4<f32>,
  screen    : vec2<f32>,
  count     : u32,
  useMask   : u32,
  p11       : f32,
  sizeMul   : f32,
  minPx     : f32,
  maxPx     : f32,
  rootMin   : vec3<f32>,
  cutDepth  : u32,
  rootSize  : vec3<f32>,
  useCut    : u32,
  flatColor : vec3<f32>,
  mode      : u32,
  elevMin   : f32,
  elevMax   : f32,
  scalarMin : f32,
  scalarMax : f32,
  maxLevel  : f32,
  edlStr    : f32,
  edlRadius : f32,
  bgR       : f32,
  bgG       : f32,
  bgB       : f32,
  // DEC-B6: os mesmos planos que cortam a malha, aqui já em coordenadas
  // LOCAIS DA NUVEM. A conversão é uma soma — de cena para nuvem é uma
  // translação, e para um plano isso é d += dot(n, t).
  clipCount : f32,
  // Bitmask das classes ESCONDIDAS, em qualquer modo de cor: bit n esconde o
  // código n para 0..18, bit 31 esconde o balde "fora do padrão". Vive no
  // que era o padding antes de 'clip', então o layout não mexe.
  classHidden : u32,
  clip      : array<vec4<f32>, 4>,
  // A faixa de ALTURA em que um ponto é desenhado, em Z local-da-nuvem — o
  // mesmo frame de 'pos' e de 'elevMin/elevMax'. Acrescentada DEPOIS do array
  // de planos, e não num padding no meio: os offsets de tudo o que já existia
  // ficam onde estavam, e o custo é um vec4 a mais no buffer.
  zRange    : vec2<f32>,
  // A camera da FOTO, em coordenadas LOCAIS DA NUVEM: leva um ponto ao espaco
  // de recorte dela, e e o que substitui a homografia em CSS.
  //
  // A diferenca e o pressuposto que cai. A homografia rebatia a imagem sobre UM
  // plano, entao todo o relevo virava desalinhamento e a foto parecia flutuar;
  // aqui cada ponto pergunta que pixel o fotografou, na geometria que ele
  // realmente tem. Entra DEPOIS do array de planos e da faixa de altura, pela
  // mesma razao que elas: os offsets de tudo o que ja existia ficam onde estao.
  photoClipFromCloud : mat4x4<f32>,
  // Quanto da foto entra na cor, 0..1. ZERO DESLIGA -- e o teste que o caso
  // comum paga, em vez de um bind group diferente por causa de uma feature que
  // a maioria das cenas nao usa.
  photoMix  : f32,
};

@group(0) @binding(0) var<storage, read>       pos   : array<f32>;
@group(0) @binding(1) var<storage, read>       col   : array<u32>;
@group(0) @binding(2) var<storage, read_write> depth : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> accum : array<atomic<u32>>;
@group(0) @binding(4) var<uniform>             u     : U;
@group(0) @binding(5) var<storage, read>       pitch : array<f32>;
// Level in the low 8 bits, owning node slot in the next 16, ASPRS class in the
// top 8. The slot only ever needs 16 because MAX_SLOTS is 65536, so the class
// rides in bits that were already allocated and cost nothing — which is the
// only reason it can be here at all, with the eight-buffer limit above.
@group(0) @binding(6) var<storage, read>       nmeta : array<u32>;
// Per cut slot: child mask in the low 8 bits, first-child slot in the high 24,
// the same bytes 'OctreeCut' feeds the instanced material's DataTexture.
@group(0) @binding(7) var<storage, read>       cut   : array<u32>;
// Whether each node is in THIS frame's draw list, indexed by the slot in nmeta.
@group(0) @binding(8) var<storage, read>       live  : array<u32>;

// A foto e o seu sampler. NAO sao storage buffers: o limite que este shader ja
// toca -- 'maxStorageBuffersPerShaderStage', garantido em 8 e onde ele esta
// exactamente -- conta so buffers. Texturas amostradas tem limite proprio, de
// 16, entao isto cabe sem tirar nada de la.
//
// Existem SEMPRE, mesmo sem foto: o WebGPU exige uma entrada por binding do
// layout, e sem foto o que esta ligado e uma textura 1x1 que 'photoMix' a zero
// nunca chega a ler.
@group(0) @binding(9)  var photoTex  : texture_2d<f32>;
@group(0) @binding(10) var photoSamp : sampler;

const RAMP = array<vec3<f32>, 5>(
  vec3<f32>(0.19, 0.07, 0.23), vec3<f32>(0.21, 0.36, 0.55), vec3<f32>(0.13, 0.57, 0.55),
  vec3<f32>(0.48, 0.74, 0.32), vec3<f32>(0.99, 0.91, 0.15));

fn rampAt(t : f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0) * 4.0;
  let i = min(u32(x), 3u);
  return mix(RAMP[i], RAMP[i + 1u], x - f32(i));
}

const CLASSES = array<vec3<f32>, 19>(
  vec3<f32>(0.42, 0.42, 0.46), vec3<f32>(0.62, 0.62, 0.66), vec3<f32>(0.55, 0.42, 0.28),
  vec3<f32>(0.35, 0.60, 0.30), vec3<f32>(0.28, 0.68, 0.32), vec3<f32>(0.18, 0.50, 0.24),
  vec3<f32>(0.85, 0.45, 0.35), vec3<f32>(0.90, 0.20, 0.35), vec3<f32>(0.75, 0.70, 0.30),
  vec3<f32>(0.24, 0.50, 0.85), vec3<f32>(0.50, 0.35, 0.55), vec3<f32>(0.34, 0.34, 0.40),
  vec3<f32>(0.70, 0.70, 0.55), vec3<f32>(0.95, 0.75, 0.25), vec3<f32>(0.95, 0.60, 0.15),
  vec3<f32>(0.80, 0.50, 0.60), vec3<f32>(0.95, 0.85, 0.40), vec3<f32>(0.60, 0.55, 0.70),
  vec3<f32>(1.00, 0.10, 0.55));
const UNKNOWN_CLASS = vec3<f32>(0.0, 0.85, 0.8);

// O SLOT do nó dono, em 16 bits. O byte de topo do 'nmeta' carrega a classe, e
// um '>> 8u' cru traria a classe junto e indexaria 'live' fora do nó certo.
fn slotOf(i : u32) -> u32 {
  return (nmeta[i] >> 8u) & 0xffffu;
}

// A classe deste ponto está desligada? Vale em TODOS os modos de cor: o código
// viaja no byte de topo do 'nmeta', não no 'col' que muda de significado com o
// modo, então esconder uma classe e depois trocar para RGB mantém-na escondida.
// Ambos os passes chamam isto: um ponto escondido que ainda escrevesse
// profundidade seguiria ocluindo o que está atrás dele — invisível, mas
// presente, que é pior que um bug visível.
fn classOff(i : u32) -> bool {
  if (u.classHidden == 0u) { return false; }
  let code = nmeta[i] >> 24u;
  let bit = select(31u, code, code <= 18u);
  return ((u.classHidden >> bit) & 1u) == 1u;
}

// FORA DA FAIXA DE ALTURA. O irmão geométrico de 'classOff', e pela mesma razão
// chamado dos dois passes: um ponto que não se vê mas ainda escreve
// profundidade continua a ocluir o que está atrás dele.
//
// Existe porque nem toda nuvem traz o ruído classificado. Num levantamento
// aéreo da NOAA os retornos que voam já vêm em classe 18 e o filtro de classes
// resolve; num scan terrestre cru, num E57 convertido ou numa nuvem de
// fotogrametria não vem classificação nenhuma, e a única coisa que separa o
// que voa do levantamento é a altura.
//
// Desligada é [-3.4e38, 3.4e38] — nenhum ponto real ultrapassa isso, então o
// caso comum não paga um uniform de flag nem um branch a mais.
fn zOff(i : u32) -> bool {
  let z = pos[i * 3u + 2u];
  return z < u.zRange.x || z > u.zRange.y;
}

// Screen pixel for point i, or ok=false when it is off screen or behind the eye.
//
// Depth is the IEEE bit pattern of the EYE depth. For non-negative floats that
// pattern is monotonic in the value, so atomicMin over the bits is atomicMin
// over the depth — and it buys two things a quantised NDC integer cannot. The
// colour pass gets a RELATIVE tolerance, meaning the same thing near and far.
// And EDL gets a real linear depth to take log2 of, which is what makes its
// response Potree's.
struct Splat { ok : bool, x : i32, y : i32, d : u32, r : f32, p : f32 };

// THE RECONSTRUCTION FILTER, and its width is in PIXELS — not in splat radii.
//
// That distinction is the whole fix. A splat's radius says how much it COVERS;
// the filter says how much a point at some pixel distance should say about that
// pixel. Weighting by a falloff over the disc conflates the two and does almost
// nothing: at an 8 px splat a point 2 px away weighs 0.94 against 0.98 for one
// at 1 px, so they still average to a blur — and normalising by the weight sum
// cancels the falloff entirely for an isolated splat.
//
// A fixed 0.7 px sigma instead makes a point centred on the pixel outweigh one
// 2 px away by ~50x, so overlapping coarse splats stop blending and the edge
// comes back hard. FLOOR is what keeps coverage: an isolated splat still paints
// its whole disc, because normalising a constant weight returns the colour
// unchanged. Shrinking the splat would have sharpened too — and opened holes.
//
// It also self-adjusts. At fine LOD every splat is 1-2 px, so the weights are
// close and the average survives: the antialiasing stays where it helps and
// disappears where it smeared.
const FILTER_INV = 1.0204;   // 1 / (2 * 0.7^2)
const FILTER_FLOOR = 0.004;

fn project(i : u32) -> Splat {
  var s : Splat;
  s.ok = false;
  let p = vec4<f32>(pos[i * 3u], pos[i * 3u + 1u], pos[i * 3u + 2u], 1.0);
  // Antes da projeção: um ponto cortado fora não custa nem a multiplicação.
  let nClip = u32(u.clipCount);
  for (var ci : u32 = 0u; ci < nClip; ci = ci + 1u) {
    let pl = u.clip[ci];
    if (dot(pl.xyz, p.xyz) + pl.w < 0.0) { return s; }
  }
  let c = u.clipFromCloud * p;
  if (c.w <= 0.0) { return s; }
  let ndc = c.xyz / c.w;
  if (ndc.x < -1.5 || ndc.x > 1.5 || ndc.y < -1.5 || ndc.y > 1.5) { return s; }
  if (ndc.z < 0.0 || ndc.z > 1.0) { return s; }
  s.x = i32((ndc.x * 0.5 + 0.5) * u.screen.x);
  s.y = i32((1.0 - (ndc.y * 0.5 + 0.5)) * u.screen.y);
  s.d = bitcast<u32>(c.w);

  // LOCAL DEPTH: descend the selected cut to the deepest node containing this
  // point and size by THAT pitch. A point sized by its own node instead draws
  // 2**(D-L) too wide everywhere behind the frontier, painting over finer data
  // already on the GPU.
  var shrink : u32 = 0u;
  if (u.useCut == 1u) {
    var slot : u32 = 0u;
    var bMin = u.rootMin;
    var bSize = u.rootSize;
    var d2 : u32 = 0u;
    for (var step_i : u32 = 0u; step_i < u.cutDepth; step_i = step_i + 1u) {
      let word = cut[slot];
      let mask = word & 0xffu;
      // Three BIG-endian bytes after the mask, which little-endian u32 loads
      // deliver reversed. Decoded here rather than re-encoded on the CPU, so
      // both consumers of 'OctreeCut' read one layout.
      let first = ((word >> 8u) & 0xffu) * 65536u
                + ((word >> 16u) & 0xffu) * 256u
                + ((word >> 24u) & 0xffu);
      let half = bSize * 0.5;
      let mid = bMin + half;
      let cx = select(0u, 1u, p.x >= mid.x);
      let cy = select(0u, 1u, p.y >= mid.y);
      let cz = select(0u, 1u, p.z >= mid.z);
      let idx = cx * 4u + cy * 2u + cz;
      if ((mask & (1u << idx)) == 0u) { break; }
      var below : u32 = 0u;
      for (var b : u32 = 0u; b < idx; b = b + 1u) {
        below = below + ((mask >> b) & 1u);
      }
      slot = first + below;
      bMin = bMin + vec3<f32>(f32(cx), f32(cy), f32(cz)) * half;
      bSize = half;
      d2 = d2 + 1u;
    }
    // Never shallower than the point's own node: a drawn point's ancestors are
    // all selected, so a shallower answer is float32 ambiguity at a split
    // plane, and clamping degrades it to per-node sizing rather than a blob.
    let myLevel = nmeta[i] & 0xffu;
    // MEASURED CAP OF ONE LEVEL, and the number is not taste.
    //
    // The cut says "the deepest resident node here is at depth D", and shrinking
    // a coarse point by the full 2**(D-L) assumes the levels in between deliver
    // enough points to refill the area it gave up. They do not always, and the
    // gap is visible: with the full shrink, 5.4% of the screen went from painted
    // at 13 s to background at 25 s — COVERAGE FALLING as more data arrived,
    // which is the opposite of what streaming should do. Potree loses 1.2% over
    // the same window, because it never shrinks: it sizes each point by its own
    // node's spacing.
    //
    // Capping the shrink at one level takes that to 0.7%, better than Potree,
    // and it is what took Speed Index from 5083 to 1918 (Potree: 1594) with
    // visual completeness arriving at 21 s instead of 25 s.
    //
    // Two alternatives were measured and are worse. A bigger splat with the full
    // shrink closes the holes only at 1.6x, and blurs: 13.1 of surface detail
    // against 14.1 here. A cap of two levels keeps more detail (15.9) but leaves
    // 3.6% falling and only 49% of the frame hole-free, against 59% here.
    if (d2 > myLevel) { shrink = min(d2 - myLevel, 1u); }
  }
  let projFactor = 0.5 * u.screen.y * u.p11 / c.w;
  let localPitch = pitch[i] / exp2(f32(shrink));
  s.r = clamp(localPitch * 2.0 * u.sizeMul * projFactor, u.minPx, u.maxPx) * 0.5;
  // The pitch the cut actually resolved, in WORLD units, which is what the
  // colour pass needs to decide whether two points are one surface.
  s.p = localPitch;
  s.ok = true;
  return s;
}

// 'col[i]' carries whatever THIS mode needs — RGBA bytes, a raw intensity or a
// class code — because only one mode is live per cloud and a second per-point
// buffer would be the ninth binding. The scalar lanes arrive as f32 bits, which
// is the same 'scalarFormat: "gpu"' lane the instanced material reads.
//
// No sRGB->linear conversion, unlike that material: it renders through three
// into a linear working space and back out, while this writes straight to a
// non-sRGB swapchain, so converting here would decode twice.
fn baseShade(i : u32, z : f32) -> vec3<f32> {
  switch u.mode {
    case 1u: { return u.flatColor; }
    case 2u: { return rampAt((z - u.elevMin) / max(u.elevMax - u.elevMin, 1e-9)); }
    case 3u: { return rampAt(f32(nmeta[i] & 0xffu) / max(u.maxLevel, 1.0)); }
    case 4u: {
      return rampAt((bitcast<f32>(col[i]) - u.scalarMin) / max(u.scalarMax - u.scalarMin, 1e-9));
    }
    case 5u: {
      let code = u32(max(bitcast<f32>(col[i]), 0.0));
      if (code > 18u) { return UNKNOWN_CLASS; }
      return CLASSES[code];
    }
    default: {
      let rgba = col[i];
      return vec3<f32>(f32(rgba & 0xffu), f32((rgba >> 8u) & 0xffu), f32((rgba >> 16u) & 0xffu)) / 255.0;
    }
  }
}

/**
 * A cor com a foto projetada por cima, quando ha uma.
 *
 * Isto e textura projetiva: o ponto vai ao espaco de recorte da camera da foto
 * e le o pixel que o fotografou. Nao ha plano nenhum no meio, entao o relevo
 * deixa de ser erro -- e como a cor vive no ponto, a foto passa a ser oclusa
 * pela nuvem e a orbitar como geometria, que era o que a imagem em DOM por
 * cima da tela nunca podia fazer.
 *
 * O QUE ISTO AINDA NAO FAZ e o teste de profundidade a partir da foto. Um
 * projetor ilumina tudo o que esta na sua frente, inclusive o que estava
 * ESCONDIDO atras de um telhado: essa parede recebe textura que a lente nunca
 * viu. Num voo a nadir sobre terreno o efeito e pequeno e aparece nas faces
 * verticais; a correccao e um mapa de profundidade da foto, um passe a mais.
 */
fn photoOver(i : u32, base : vec3<f32>) -> vec3<f32> {
  if (u.photoMix <= 0.0) { return base; }
  let world = vec4<f32>(pos[i * 3u], pos[i * 3u + 1u], pos[i * 3u + 2u], 1.0);
  let c = u.photoClipFromCloud * world;
  // Atras da lente. Sem este corte a divisao espelha a cena e a foto cola-se
  // tambem ao que esta por tras da camera -- o artefacto classico do projetor.
  if (c.w <= 1e-9) { return base; }
  let ndc = c.xy / c.w;
  if (any(abs(ndc) > vec2<f32>(1.0, 1.0))) { return base; }
  // NDC cresce para cima, a textura cresce para baixo.
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  // 'textureSampleLevel' e nao 'textureSample': um shader de compute nao tem
  // derivadas, entao o nivel de mipmap tem de ser explicito. O implicito nem
  // compila aqui.
  let s = textureSampleLevel(photoTex, photoSamp, uv, 0.0);
  return mix(base, s.rgb, u.photoMix * s.a);
}

fn shade(i : u32, z : f32) -> vec3<f32> {
  return photoOver(i, baseShade(i, z));
}

@compute @workgroup_size(256)
fn clearPass(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(u.screen.x) * u32(u.screen.y)) { return; }
  atomicStore(&depth[i], 0x7f7fffffu);  // FLT_MAX
  let base = i * 4u;
  atomicStore(&accum[base + 0u], 0u);
  atomicStore(&accum[base + 1u], 0u);
  atomicStore(&accum[base + 2u], 0u);
  atomicStore(&accum[base + 3u], 0u);
}

@compute @workgroup_size(256)
fn depthPass(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  if (live[slotOf(i)] == 0u) { return; }
  if (classOff(i) || zOff(i)) { return; }
  let s = project(i);
  if (!s.ok) { return; }
  let ri = i32(ceil(s.r - 0.5));
  let r2 = s.r * s.r;
  let W = i32(u.screen.x);
  let H = i32(u.screen.y);
  for (var dy = -ri; dy <= ri; dy = dy + 1) {
    let y = s.y + dy;
    if (y < 0 || y >= H) { continue; }
    for (var dx = -ri; dx <= ri; dx = dx + 1) {
      let x = s.x + dx;
      if (x < 0 || x >= W) { continue; }
      if (f32(dx * dx + dy * dy) > r2) { continue; }
      atomicMin(&depth[u32(y * W + x)], s.d);
    }
  }
}

@compute @workgroup_size(256)
fn colorPass(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  if (live[slotOf(i)] == 0u) { return; }
  if (classOff(i) || zOff(i)) { return; }
  let s = project(i);
  if (!s.ok) { return; }
  let rgb = shade(i, pos[i * 3u + 2u]) * 255.0;
  let cr = u32(clamp(rgb.x, 0.0, 255.0));
  let cg = u32(clamp(rgb.y, 0.0, 255.0));
  let cb = u32(clamp(rgb.z, 0.0, 255.0));
  let ri = i32(ceil(s.r - 0.5));
  let r2 = s.r * s.r;
  let W = i32(u.screen.x);
  let H = i32(u.screen.y);
  for (var dy = -ri; dy <= ri; dy = dy + 1) {
    let y = s.y + dy;
    if (y < 0 || y >= H) { continue; }
    for (var dx = -ri; dx <= ri; dx = dx + 1) {
      let x = s.x + dx;
      if (x < 0 || x >= W) { continue; }
      let d2 = f32(dx * dx + dy * dy);
      if (d2 > r2) { continue; }
      let idx = u32(y * W + x);
      // WORLD units from the resolved pitch, not a percentage of the depth.
      // Two points within a sampling step are one surface; past that they are
      // different surfaces and must not blend. The old '+1%' meant 10 m at a
      // kilometre, which is how a tree came to average with the ground behind
      // it — and a silhouette that averages is a silhouette that is not an
      // edge. Loose at coarse levels and tight at fine ones, which is right:
      // coarse data resolves depth coarsely. The relative term is a floor for
      // numerical safety, not the criterion.
      let near = bitcast<f32>(atomicLoad(&depth[idx]));
      if (bitcast<f32>(s.d) > near + max(s.p * 2.0, near * 0.005)) { continue; }
      // Weighted, so accum[3] is now a sum of WEIGHTS rather than a count. The
      // resolve divides by it either way, so its arithmetic is unchanged.
      let wi = max(1u, u32((FILTER_FLOOR + exp(-d2 * FILTER_INV)) * 255.0));
      let base = idx * 4u;
      atomicAdd(&accum[base + 0u], cr * wi);
      atomicAdd(&accum[base + 1u], cg * wi);
      atomicAdd(&accum[base + 2u], cb * wi);
      atomicAdd(&accum[base + 3u], wi);
    }
  }
}

struct VOut { @builtin(position) pos : vec4<f32> };
@vertex
fn vsResolve(@builtin(vertex_index) vi : u32) -> VOut {
  // One oversized triangle, not a quad: fewer vertices and no seam.
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  var o : VOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  return o;
}

@fragment
fn fsResolve(@builtin(position) fc : vec4<f32>) -> @location(0) vec4<f32> {
  let W = u32(u.screen.x);
  let H = u32(u.screen.y);
  let idx = u32(fc.y) * W + u32(fc.x);
  let base = idx * 4u;
  // The sum of filter WEIGHTS, not a count of points. Dividing by it is the
  // normalisation that keeps an isolated splat at full colour, which is what
  // lets the filter sharpen without opening the holes a smaller splat would.
  // At most 255 per contribution times 255 of colour is 65k, so a u32 takes
  // 66,000 points on one pixel before it could wrap.
  let n = atomicLoad(&accum[base + 3u]);
  if (n == 0u) { return vec4<f32>(u.bgR, u.bgG, u.bgB, 1.0); }
  var c = vec3<f32>(
    f32(atomicLoad(&accum[base + 0u])),
    f32(atomicLoad(&accum[base + 1u])),
    f32(atomicLoad(&accum[base + 2u]))) / f32(n) / 255.0;

  // Eye-dome lighting, folded into the resolve.
  //
  // In the instanced pipeline EDL is a separate two-pass post-process, because
  // the depth it needs is a texture it has to be handed. Here the depth buffer
  // is already bound to this shader, so the whole effect is this block: no
  // extra pass, no extra target, no extra bandwidth.
  //
  // Potree's formulation on purpose, so an 'edlStrength' transfers without
  // retuning: response is the mean POSITIVE log2-depth difference over an
  // 8-neighbour ring, shade = exp(-response * 300 * strength). LOG depth so the
  // metric is a RATIO and means the same thing 10 m and 10 km from the camera.
  if (u.edlStr > 0.0) {
    let RING = array<vec2<f32>, 8>(
      vec2<f32>(1.0, 0.0), vec2<f32>(0.7071, 0.7071), vec2<f32>(0.0, 1.0), vec2<f32>(-0.7071, 0.7071),
      vec2<f32>(-1.0, 0.0), vec2<f32>(-0.7071, -0.7071), vec2<f32>(0.0, -1.0), vec2<f32>(0.7071, -0.7071));
    let zc = log2(bitcast<f32>(atomicLoad(&depth[idx])));
    var response = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
      let o = RING[k] * u.edlRadius;
      let sx = i32(fc.x) + i32(round(o.x));
      let sy = i32(fc.y) + i32(round(o.y));
      if (sx < 0 || sy < 0 || sx >= i32(W) || sy >= i32(H)) { continue; }
      let d = bitcast<f32>(atomicLoad(&depth[u32(sy) * W + u32(sx)]));
      // Centre MINUS neighbour, darkening where the centre is farther. A
      // neighbour with no point reads FLT_MAX, so the term goes negative and
      // contributes nothing — which is why no silhouette against the background
      // grows a dark halo. The post-process needs an explicit depth threshold
      // for that case; the 'n == 0' early-out above is exact instead, because
      // this pipeline knows whether a pixel has geometry.
      response = response + max(0.0, zc - log2(d));
    }
    c = c * exp(-response / 8.0 * 300.0 * u.edlStr);
  }
  return vec4<f32>(c, 1.0);
}
`;
