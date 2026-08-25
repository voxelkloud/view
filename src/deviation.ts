// B5 — a distância de cada ponto à superfície de projeto mais próxima.
//
// A pergunta que a Cintoo vende como produto inteiro. Aqui é um campo escalar:
// o desvio entra na mesma via que intensidade, elevação e classificação já
// usam, e portanto herda a rampa, a faixa, a legenda e o relatório sem escrever
// UI nenhuma (DEC-B8).
//
// ESTE ARQUIVO É A METADE TESTÁVEL. A travessia acontece na GPU, num passo de
// compute próprio — o shader de pontos já usa os oito storage buffers que o
// WebGPU garante, e um nono faz o layout voltar inválido em silêncio. Mas a
// árvore é construída aqui, na CPU, e a mesma distância ponto-triângulo que o
// WGSL calcula está escrita abaixo em TypeScript. Uma é o oráculo da outra: se
// divergirem, o teste diz, em vez de o mapa de cores mentir por dez
// centímetros.

/**
 * Uma BVH achatada em dois arrays, que é a forma que a GPU consegue ler.
 *
 * Cada nó ocupa 8 floats: `minX minY minZ, A, maxX maxY maxZ, B`. Num nó
 * INTERNO, `A` é o índice do filho direito (o esquerdo é sempre o próximo nó,
 * o que poupa um campo) e `B` é -1. Numa FOLHA, `A` é o primeiro triângulo e
 * `B` é quantos. O sinal de `B` é portanto o discriminante, e cabe no mesmo
 * float — WGSL não tem união e um campo de tipo custaria 16 bytes de padding.
 */
export interface TriangleBvh {
  /** 8 floats por nó. */
  readonly nodes: Float32Array;
  /** 9 floats por triângulo: três vértices em coordenadas de CENA. */
  readonly tris: Float32Array;
  readonly nodeCount: number;
  readonly triCount: number;
}

const NODE_STRIDE = 8;
const TRI_STRIDE = 9;
/** Abaixo disto um nó vira folha: mais fundo custa mais travessia do que poupa. */
const LEAF_SIZE = 4;

/** Distância ao quadrado de `p` ao triângulo `abc`, sem raiz — a raiz sai no fim. */
export function pointTriangleDistanceSq(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  // Ericson, "Real-Time Collision Detection", 5.1.5. As sete regiões de Voronoi
  // do triângulo, na ordem em que a maioria dos pontos cai: três vértices, três
  // arestas, o interior.
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return apx * apx + apy * apy + apz * apz;

  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return bpx * bpx + bpy * bpy + bpz * bpz;

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const qx = apx - v * abx, qy = apy - v * aby, qz = apz - v * abz;
    return qx * qx + qy * qy + qz * qz;
  }

  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return cpx * cpx + cpy * cpy + cpz * cpz;

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const qx = apx - w * acx, qy = apy - w * acy, qz = apz - w * acz;
    return qx * qx + qy * qy + qz * qz;
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    const qx = bpx + w * (cpx - bpx), qy = bpy + w * (cpy - bpy), qz = bpz + w * (cpz - bpz);
    return qx * qx + qy * qy + qz * qz;
  }

  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  const qx = apx - (v * abx + w * acx);
  const qy = apy - (v * aby + w * acy);
  const qz = apz - (v * abz + w * acz);
  return qx * qx + qy * qy + qz * qz;
}

/** Distância ao quadrado de `p` à caixa, zero quando dentro. */
export function pointBoxDistanceSq(
  px: number, py: number, pz: number,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): number {
  const dx = px < minX ? minX - px : px > maxX ? px - maxX : 0;
  const dy = py < minY ? minY - py : py > maxY ? py - maxY : 0;
  const dz = pz < minZ ? minZ - pz : pz > maxZ ? pz - maxZ : 0;
  return dx * dx + dy * dy + dz * dz;
}

/**
 * Constrói a BVH a partir de triângulos já em coordenadas de cena.
 *
 * Divisão pela MEDIANA do eixo mais longo, e não SAH: uma malha de BIM é
 * paredes e lajes alinhadas aos eixos, onde a mediana já separa bem, e a
 * árvore é construída uma vez por modelo em vez de por frame.
 */
export function buildTriangleBvh(tris: Float32Array): TriangleBvh {
  const triCount = Math.floor(tris.length / TRI_STRIDE);
  if (triCount === 0) {
    return { nodes: new Float32Array(0), tris, nodeCount: 0, triCount: 0 };
  }

  const order = new Uint32Array(triCount);
  const cx = new Float32Array(triCount);
  const cy = new Float32Array(triCount);
  const cz = new Float32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    order[t] = t;
    const o = t * TRI_STRIDE;
    cx[t] = (tris[o]! + tris[o + 3]! + tris[o + 6]!) / 3;
    cy[t] = (tris[o + 1]! + tris[o + 4]! + tris[o + 7]!) / 3;
    cz[t] = (tris[o + 2]! + tris[o + 5]! + tris[o + 8]!) / 3;
  }

  // Um nó por folha mais os internos: 2n-1 no pior caso, e a árvore nunca
  // passa disso porque cada divisão consome pelo menos um triângulo.
  const nodes = new Float32Array(Math.max(1, 2 * triCount) * NODE_STRIDE);
  let nodeCount = 0;

  const build = (lo: number, hi: number): number => {
    const self = nodeCount++;
    const b = self * NODE_STRIDE;
    let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
    let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
    for (let i = lo; i < hi; i++) {
      const o = order[i]! * TRI_STRIDE;
      for (let v = 0; v < 3; v++) {
        const x = tris[o + v * 3]!, y = tris[o + v * 3 + 1]!, z = tris[o + v * 3 + 2]!;
        if (x < mnX) mnX = x; if (x > mxX) mxX = x;
        if (y < mnY) mnY = y; if (y > mxY) mxY = y;
        if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
      }
    }
    nodes[b] = mnX; nodes[b + 1] = mnY; nodes[b + 2] = mnZ;
    nodes[b + 4] = mxX; nodes[b + 5] = mxY; nodes[b + 6] = mxZ;

    const count = hi - lo;
    if (count <= LEAF_SIZE) {
      nodes[b + 3] = lo;
      nodes[b + 7] = count; // >= 0 marca folha
      return self;
    }

    const ex = mxX - mnX, ey = mxY - mnY, ez = mxZ - mnZ;
    const axis = ex >= ey && ex >= ez ? cx : ey >= ez ? cy : cz;
    const mid = (lo + hi) >> 1;
    // `sort` sobre uma fatia é O(n log n) por nível; suficiente para dezenas de
    // milhares de triângulos e muito mais simples de ler que um nth_element.
    const slice = Array.from(order.subarray(lo, hi)).sort((p, q) => axis[p]! - axis[q]!);
    order.set(slice, lo);

    build(lo, mid);
    const right = build(mid, hi);
    nodes[b + 3] = right;
    nodes[b + 7] = -1; // interno
    return self;
  };
  build(0, triCount);

  // Os triângulos são reordenados para que uma folha seja um intervalo
  // contíguo: a GPU lê `[first, first+count)` sem uma tabela de indireção.
  const sorted = new Float32Array(triCount * TRI_STRIDE);
  for (let i = 0; i < triCount; i++) {
    sorted.set(tris.subarray(order[i]! * TRI_STRIDE, order[i]! * TRI_STRIDE + TRI_STRIDE), i * TRI_STRIDE);
  }

  return { nodes: nodes.subarray(0, nodeCount * NODE_STRIDE), tris: sorted, nodeCount, triCount };
}

/**
 * A referência em CPU da travessia que o WGSL faz. Existe para o teste — e para
 * quem duvidar do mapa de cores poder conferir um ponto à mão.
 */
export function distanceToBvh(bvh: TriangleBvh, px: number, py: number, pz: number): number {
  if (bvh.nodeCount === 0) return Infinity;
  const n8 = bvh.nodes;
  let best = Infinity;
  const stack: number[] = [0];
  while (stack.length > 0) {
    const n = stack.pop()!;
    const b = n * NODE_STRIDE;
    // Podar contra a caixa ANTES de descer é a árvore inteira: sem isto a
    // travessia visita tudo e a BVH não serve para nada. A comparação é com
    // `best` ao QUADRADO — tirar a raiz aqui seria uma por nó visitado.
    if (
      pointBoxDistanceSq(px, py, pz, n8[b]!, n8[b + 1]!, n8[b + 2]!, n8[b + 4]!, n8[b + 5]!, n8[b + 6]!) >=
      best
    )
      continue;
    const marker = n8[b + 7]!;
    if (marker >= 0) {
      const first = n8[b + 3]!;
      for (let t = first; t < first + marker; t++) {
        const o = t * TRI_STRIDE;
        const d = pointTriangleDistanceSq(px, py, pz,
          bvh.tris[o]!, bvh.tris[o + 1]!, bvh.tris[o + 2]!,
          bvh.tris[o + 3]!, bvh.tris[o + 4]!, bvh.tris[o + 5]!,
          bvh.tris[o + 6]!, bvh.tris[o + 7]!, bvh.tris[o + 8]!);
        if (d < best) best = d;
      }
    } else {
      // Filho esquerdo é sempre o nó seguinte; o direito está no campo. Empurrar
      // os dois sem ordenar é mais lento que descer primeiro pelo mais próximo,
      // e a versão do WGSL faz o mesmo — as duas têm de concordar.
      stack.push(n + 1, n8[b + 3]!);
    }
  }
  return Math.sqrt(best);
}

export const BVH_NODE_STRIDE = NODE_STRIDE;
export const BVH_TRI_STRIDE = TRI_STRIDE;

/**
 * Os triângulos de um objeto three, já em coordenadas de CENA.
 *
 * Lê o atributo de posição CRU — que num modelo BIM é uint16 quantizado — e
 * aplica a `matrixWorld`, que é onde a desquantização vive. Ler valores
 * "desquantizados" do atributo daria as unidades de quantum, e a BVH mediria
 * desvios em quanta em vez de metros.
 */
export function trianglesFromObject(root: {
  updateMatrixWorld(force?: boolean): void;
  traverse(cb: (o: unknown) => void): void;
}): Float32Array {
  root.updateMatrixWorld(true);
  const out: number[] = [];

  root.traverse((node) => {
    const mesh = node as {
      isMesh?: boolean;
      visible?: boolean;
      geometry?: {
        attributes: { position?: { count: number; getX(i: number): number; getY(i: number): number; getZ(i: number): number } };
        index?: { count: number; getX(i: number): number } | null;
      };
      matrixWorld?: { elements: ArrayLike<number> };
    };
    if (mesh.isMesh !== true || mesh.visible === false) return;
    const pos = mesh.geometry?.attributes.position;
    const m = mesh.matrixWorld?.elements;
    if (pos === undefined || m === undefined) return;

    const at = (i: number): [number, number, number] => {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      return [
        m[0]! * x + m[4]! * y + m[8]! * z + m[12]!,
        m[1]! * x + m[5]! * y + m[9]! * z + m[13]!,
        m[2]! * x + m[6]! * y + m[10]! * z + m[14]!,
      ];
    };

    const idx = mesh.geometry?.index;
    const n = idx != null ? idx.count : pos.count;
    for (let t = 0; t + 2 < n; t += 3) {
      const a = at(idx != null ? idx.getX(t) : t);
      const b = at(idx != null ? idx.getX(t + 1) : t + 1);
      const c = at(idx != null ? idx.getX(t + 2) : t + 2);
      out.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    }
  });

  return new Float32Array(out);
}
