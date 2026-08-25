/**
 * O kernel do scan-vs-BIM: a distância de cada ponto residente à malha.
 *
 * PASSO PRÓPRIO, e não mais um no shader de pontos. Aquele já usa os oito
 * storage buffers que o WebGPU garante por estágio, e o nono não lança nada —
 * o layout volta inválido e todos os passes silenciam com a tela preta e os
 * contadores da CPU corretos. Um pipeline separado tem os seus próprios oito.
 *
 * Escreve no `col`, que é onde o escalar já vive (o shader de pontos lê
 * `bitcast<f32>(col[i])` no modo 4). Portanto o desvio herda rampa, faixa,
 * legenda e relatório sem uma linha de UI nova — que é a DEC-B8 inteira. O
 * preço, dito sem rodeios: a cor RGB daquele ponto é destruída até o nó ser
 * reanexado. É um MODO, e trocar de modo já reanexa neste produto.
 *
 * A travessia é a mesma de `deviation.ts`, que é o oráculo dela: as duas têm de
 * concordar, e o teste em TypeScript é quem diz.
 */
export const DEVIATION_WGSL = `
struct U {
  count     : u32,
  // Nuvem -> cena. Os pontos vivem em coordenadas locais da nuvem e a BVH em
  // coordenadas de cena; somar aqui é mais barato que reconstruir a árvore.
  toScene   : vec3<f32>,
  // Acima disto o ponto não interessa: um telhado a 40 m do modelo não é um
  // desvio, é outro prédio. Corta a travessia cedo e limita a rampa.
  maxDist   : f32,
  _pad      : vec3<f32>,
};

@group(0) @binding(0) var<storage, read>       pos   : array<f32>;
@group(0) @binding(1) var<storage, read_write> col   : array<u32>;
@group(0) @binding(2) var<storage, read>       nodes : array<f32>;
@group(0) @binding(3) var<storage, read>       tris  : array<f32>;
@group(0) @binding(4) var<uniform>             u     : U;

const NODE_STRIDE : u32 = 8u;
const TRI_STRIDE  : u32 = 9u;
// Uma BVH por mediana sobre 250k triângulos tem ~16 níveis; 32 é o dobro e a
// pilha é registrador, não memória. Estourar seria silencioso, então a
// travessia também para quando enche.
const STACK_MAX   : i32 = 32;

fn boxDistSq(p : vec3<f32>, lo : vec3<f32>, hi : vec3<f32>) -> f32 {
  let d = max(max(lo - p, p - hi), vec3<f32>(0.0));
  return dot(d, d);
}

/** As sete regiões de Voronoi do triângulo. Espelha \`pointTriangleDistanceSq\`. */
fn triDistSq(p : vec3<f32>, a : vec3<f32>, b : vec3<f32>, c : vec3<f32>) -> f32 {
  let ab = b - a;
  let ac = c - a;
  let ap = p - a;
  let d1 = dot(ab, ap);
  let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return dot(ap, ap); }

  let bp = p - b;
  let d3 = dot(ab, bp);
  let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return dot(bp, bp); }

  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    let v = d1 / (d1 - d3);
    let q = ap - v * ab;
    return dot(q, q);
  }

  let cp = p - c;
  let d5 = dot(ab, cp);
  let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return dot(cp, cp); }

  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    let w = d2 / (d2 - d6);
    let q = ap - w * ac;
    return dot(q, q);
  }

  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    let q = bp + w * (cp - bp);
    return dot(q, q);
  }

  let denom = 1.0 / (va + vb + vc);
  let v = vb * denom;
  let w = vc * denom;
  let q = ap - (v * ab + w * ac);
  return dot(q, q);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }

  let p = vec3<f32>(pos[i * 3u], pos[i * 3u + 1u], pos[i * 3u + 2u]) + u.toScene;

  var best = u.maxDist * u.maxDist;
  var stack : array<u32, 32>;
  var sp : i32 = 0;
  stack[0] = 0u;
  sp = 1;

  while (sp > 0) {
    sp = sp - 1;
    let n = stack[sp];
    let b = n * NODE_STRIDE;
    let lo = vec3<f32>(nodes[b], nodes[b + 1u], nodes[b + 2u]);
    let hi = vec3<f32>(nodes[b + 4u], nodes[b + 5u], nodes[b + 6u]);
    // Podar contra a caixa antes de descer É a árvore. Sem isto a travessia
    // visita tudo e a BVH não serve para nada.
    if (boxDistSq(p, lo, hi) >= best) { continue; }

    let marker = nodes[b + 7u];
    if (marker >= 0.0) {
      let first = u32(nodes[b + 3u]);
      let n_tri = u32(marker);
      for (var t : u32 = 0u; t < n_tri; t = t + 1u) {
        let o = (first + t) * TRI_STRIDE;
        let d = triDistSq(
          p,
          vec3<f32>(tris[o], tris[o + 1u], tris[o + 2u]),
          vec3<f32>(tris[o + 3u], tris[o + 4u], tris[o + 5u]),
          vec3<f32>(tris[o + 6u], tris[o + 7u], tris[o + 8u]),
        );
        best = min(best, d);
      }
    } else if (sp + 2 <= STACK_MAX) {
      stack[sp] = n + 1u;
      stack[sp + 1] = u32(nodes[b + 3u]);
      sp = sp + 2;
    }
  }

  col[i] = bitcast<u32>(sqrt(best));
}
`;
