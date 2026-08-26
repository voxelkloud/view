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
  /**
   * O índice denso do elemento a que cada triângulo pertence, ou vazio quando a
   * malha não os traz. É isto que transforma "0,21 m de desvio" em "0,21 m da
   * parede #1506" — e sem isso um tópico BCF não seleciona nada no Revit.
   */
  readonly features: Uint32Array;
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
export function buildTriangleBvh(tris: Float32Array, features?: Uint32Array): TriangleBvh {
  const triCount = Math.floor(tris.length / TRI_STRIDE);
  if (triCount === 0) {
    return { nodes: new Float32Array(0), tris, features: new Uint32Array(0), nodeCount: 0, triCount: 0 };
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
  const sortedFeat = new Uint32Array(features === undefined ? 0 : triCount);
  for (let i = 0; i < triCount; i++) {
    sorted.set(tris.subarray(order[i]! * TRI_STRIDE, order[i]! * TRI_STRIDE + TRI_STRIDE), i * TRI_STRIDE);
    if (features !== undefined) sortedFeat[i] = features[order[i]!] ?? 0;
  }

  return {
    nodes: nodes.subarray(0, nodeCount * NODE_STRIDE),
    tris: sorted,
    features: sortedFeat,
    nodeCount,
    triCount,
  };
}

/**
 * A referência em CPU da travessia que o WGSL faz. Existe para o teste — e para
 * quem duvidar do mapa de cores poder conferir um ponto à mão.
 */
export function distanceToBvh(bvh: TriangleBvh, px: number, py: number, pz: number): number {
  return nearestOnBvh(bvh, px, py, pz).distance;
}

/**
 * O mesmo que {@link distanceToBvh}, mas devolve também de QUE elemento é a
 * superfície mais próxima. `feature` é -1 quando a malha não trouxe ids ou
 * quando nada foi encontrado.
 */
export function nearestOnBvh(
  bvh: TriangleBvh,
  px: number,
  py: number,
  pz: number,
): { distance: number; feature: number } {
  if (bvh.nodeCount === 0) return { distance: Infinity, feature: -1 };
  const n8 = bvh.nodes;
  let best = Infinity;
  let bestTri = -1;
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
        if (d < best) {
          best = d;
          bestTri = t;
        }
      }
    } else {
      // Filho esquerdo é sempre o nó seguinte; o direito está no campo. Empurrar
      // os dois sem ordenar é mais lento que descer primeiro pelo mais próximo,
      // e a versão do WGSL faz o mesmo — as duas têm de concordar.
      stack.push(n + 1, n8[b + 3]!);
    }
  }
  return {
    distance: Math.sqrt(best),
    feature: bestTri >= 0 && bvh.features.length > 0 ? (bvh.features[bestTri] ?? -1) : -1,
  };
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
}): { tris: Float32Array; features: Uint32Array } {
  root.updateMatrixWorld(true);
  const out: number[] = [];
  const feats: number[] = [];

  root.traverse((node) => {
    const mesh = node as {
      isMesh?: boolean;
      visible?: boolean;
      geometry?: {
        attributes: {
          position?: { count: number; getX(i: number): number; getY(i: number): number; getZ(i: number): number };
          _feature_id_0?: { getX(i: number): number };
        };
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

    const fid = mesh.geometry?.attributes._feature_id_0;
    const idx = mesh.geometry?.index;
    const n = idx != null ? idx.count : pos.count;
    for (let t = 0; t + 2 < n; t += 3) {
      const i0 = idx != null ? idx.getX(t) : t;
      const a = at(i0);
      const b = at(idx != null ? idx.getX(t + 1) : t + 1);
      const c = at(idx != null ? idx.getX(t + 2) : t + 2);
      out.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
      // O id do PRIMEIRO vértice. Um triângulo nunca cruza elementos — o
      // conversor agrupa por material, mas cada triângulo veio de uma malha só.
      feats.push(fid === undefined ? 0 : fid.getX(i0));
    }
  });

  return { tris: new Float32Array(out), features: new Uint32Array(feats) };
}

/** Uma região contígua de pontos fora da tolerância. Vira um tópico BCF. */
export interface DeviationCluster {
  readonly count: number;
  readonly centre: readonly [number, number, number];
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly meanDeviation: number;
  readonly maxDeviation: number;
  /** Índices densos dos elementos mais atingidos, o mais atingido primeiro. */
  readonly features: readonly number[];
}

/**
 * Agrupa os pontos fora da tolerância em regiões.
 *
 * Um mapa de cores mostra que há divergência; um tópico BCF precisa de dizer
 * ONDE, e "onde" tem de ser uma região e não um ponto — 4.812 tópicos, um por
 * ponto, é a mesma coisa que nenhum.
 *
 * Voxel + componentes conexas em 26-vizinhança. Não é k-means nem DBSCAN de
 * propósito: os dois precisam de parâmetros que ninguém sabe escolher e de uma
 * semente que torna o resultado instável entre execuções. Uma grade fixa dá
 * sempre o mesmo agrupamento para o mesmo dado, que é o que um relatório
 * precisa.
 *
 * Pontos SATURADOS no teto são descartados: eles não são um desvio, são "não
 * há modelo aqui" — o chão em volta do prédio, as árvores. Incluí-los faria o
 * maior tópico ser sempre o terreno.
 */
export function clusterDeviation(
  positions: Float32Array,
  deviations: Float32Array,
  options: {
    readonly tolerance: number;
    readonly ceiling: number;
    /** Aresta do voxel, em metros. Dois pontos a mais que isto não se juntam. */
    readonly cell?: number;
    /** Regiões menores que isto são ruído do scan, não divergência. */
    readonly minPoints?: number;
    readonly bvh?: TriangleBvh;
  },
): DeviationCluster[] {
  const cell = options.cell ?? 2;
  const minPoints = options.minPoints ?? 40;
  const ceil = options.ceiling * 0.999;

  // chave do voxel -> índices dos pontos nele
  const cells = new Map<string, number[]>();
  const n = deviations.length;
  for (let i = 0; i < n; i++) {
    const d = deviations[i]!;
    if (!(d > options.tolerance) || d >= ceil || !Number.isFinite(d)) continue;
    const kx = Math.floor(positions[i * 3]! / cell);
    const ky = Math.floor(positions[i * 3 + 1]! / cell);
    const kz = Math.floor(positions[i * 3 + 2]! / cell);
    const key = `${kx},${ky},${kz}`;
    let list = cells.get(key);
    if (list === undefined) cells.set(key, (list = []));
    list.push(i);
  }

  const seen = new Set<string>();
  const out: DeviationCluster[] = [];
  for (const start of cells.keys()) {
    if (seen.has(start)) continue;
    // Largura primeiro sobre os 26 vizinhos: uma parede inteira fora do prumo é
    // UM problema, não um por voxel.
    const stack = [start];
    seen.add(start);
    const members: number[] = [];
    while (stack.length > 0) {
      const key = stack.pop()!;
      const list = cells.get(key);
      if (list === undefined) continue;
      members.push(...list);
      const [cx, cy, cz] = key.split(",").map(Number) as [number, number, number];
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            if (dx === 0 && dy === 0 && dz === 0) continue;
            const k = `${cx + dx},${cy + dy},${cz + dz}`;
            if (cells.has(k) && !seen.has(k)) {
              seen.add(k);
              stack.push(k);
            }
          }
        }
      }
    }
    if (members.length < minPoints) continue;

    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    let sum = 0;
    let peak = 0;
    const hits = new Map<number, number>();
    // Amostra para a atribuição de elemento: uma consulta por ponto num
    // agrupamento de dezenas de milhares seria minutos de CPU, e o que se quer
    // é o nome do elemento, não um censo.
    const step = Math.max(1, Math.floor(members.length / 200));
    for (let m = 0; m < members.length; m++) {
      const i = members[m]!;
      const d = deviations[i]!;
      sum += d;
      if (d > peak) peak = d;
      for (let k = 0; k < 3; k++) {
        const v = positions[i * 3 + k]!;
        if (v < min[k]!) min[k] = v;
        if (v > max[k]!) max[k] = v;
      }
      if (options.bvh !== undefined && m % step === 0) {
        const f = nearestOnBvh(options.bvh, positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!).feature;
        if (f >= 0) hits.set(f, (hits.get(f) ?? 0) + 1);
      }
    }

    out.push({
      count: members.length,
      centre: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
      min,
      max,
      meanDeviation: sum / members.length,
      maxDeviation: peak,
      features: [...hits.entries()].sort((a, b) => b[1] - a[1]).map(([f]) => f),
    });
  }

  // O pior primeiro: quem lê um relatório lê as primeiras linhas.
  return out.sort((a, b) => b.maxDeviation - a.maxDeviation || b.count - a.count);
}

/**
 * Onde um raio encontra a malha, e em que elemento.
 *
 * É a metade do alinhamento manual que fala com o MODELO: para casar um canto
 * do telhado no scan com o mesmo canto no projeto, é preciso saber onde no
 * projeto o cursor caiu — e isso é um raio contra a mesma árvore que o desvio
 * já usa.
 */
export function raycastBvh(
  bvh: TriangleBvh,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
): { point: [number, number, number]; distance: number; feature: number } | undefined {
  if (bvh.nodeCount === 0) return undefined;
  const n8 = bvh.nodes;
  const inv = [1 / dx, 1 / dy, 1 / dz];
  let best = Infinity;
  let bestTri = -1;
  const stack: number[] = [0];

  while (stack.length > 0) {
    const n = stack.pop()!;
    const b = n * NODE_STRIDE;
    // Slab test. Escrito com min/max em vez de ramos por eixo porque um raio
    // paralelo a um eixo dá inv = +-Infinity, e min/max com Infinity resolve-se
    // sozinho enquanto uma comparação encadeada devolve NaN.
    let t0 = -Infinity;
    let t1 = best;
    for (let k = 0; k < 3; k++) {
      const o = k === 0 ? ox : k === 1 ? oy : oz;
      const a = (n8[b + k]! - o) * inv[k]!;
      const c = (n8[b + 4 + k]! - o) * inv[k]!;
      t0 = Math.max(t0, Math.min(a, c));
      t1 = Math.min(t1, Math.max(a, c));
    }
    if (t0 > t1 || t1 < 0) continue;

    const marker = n8[b + 7]!;
    if (marker >= 0) {
      const first = n8[b + 3]!;
      for (let t = first; t < first + marker; t++) {
        const o = t * TRI_STRIDE;
        const hit = rayTriangle(
          ox, oy, oz, dx, dy, dz,
          bvh.tris[o]!, bvh.tris[o + 1]!, bvh.tris[o + 2]!,
          bvh.tris[o + 3]!, bvh.tris[o + 4]!, bvh.tris[o + 5]!,
          bvh.tris[o + 6]!, bvh.tris[o + 7]!, bvh.tris[o + 8]!,
        );
        if (hit >= 0 && hit < best) {
          best = hit;
          bestTri = t;
        }
      }
    } else {
      stack.push(n + 1, n8[b + 3]!);
    }
  }

  if (bestTri < 0) return undefined;
  return {
    point: [ox + dx * best, oy + dy * best, oz + dz * best],
    distance: best,
    feature: bvh.features.length > 0 ? (bvh.features[bestTri] ?? -1) : -1,
  };
}

/** Möller–Trumbore. Devolve `t` ao longo do raio, ou -1. Dois lados. */
function rayTriangle(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  // Dois lados de propósito: uma parede vista por dentro tem de poder ser
  // clicada, e o alinhamento acontece com a câmera em qualquer sítio.
  if (Math.abs(det) < 1e-12) return -1;
  const invDet = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * invDet;
  if (u < 0 || u > 1) return -1;
  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * invDet;
  if (v < 0 || u + v > 1) return -1;
  const t = (e2x * qx + e2y * qy + e2z * qz) * invDet;
  return t > 1e-9 ? t : -1;
}

/** O que a B4 resolve: quanto rodar em Z e para onde mover. */
export interface Alignment {
  /** Radianos, em torno de Z. */
  readonly yaw: number;
  readonly translation: readonly [number, number, number];
  /** Distância média entre os pares depois de aplicar, em metros. */
  readonly residual: number;
  readonly pairs: number;
}

/**
 * Resolve o alinhamento a partir de pares (ponto no modelo, ponto na nuvem).
 *
 * SÓ GUINADA E TRANSLAÇÃO, não rotação livre. Um prédio está a prumo: os graus
 * de liberdade reais são para onde ele aponta e onde está. Deixar o solver
 * inclinar o modelo faz com que três cliques imprecisos o deitem de lado — e o
 * resultado parece melhor no resíduo e é pior na tela.
 *
 * A guinada sai em forma fechada: com os dois conjuntos centrados, o ângulo que
 * minimiza o erro quadrático é `atan2(sum(x_m*y_c - y_m*x_c), sum(x_m*x_c +
 * y_m*y_c))`. Sem iteração, sem semente, sem mínimo local.
 */
export function solveAlignment(
  pairs: readonly {
    readonly model: readonly [number, number, number];
    readonly cloud: readonly [number, number, number];
  }[],
): Alignment | undefined {
  const n = pairs.length;
  if (n === 0) return undefined;

  const cm = [0, 0, 0];
  const cc = [0, 0, 0];
  for (const p of pairs) {
    for (let k = 0; k < 3; k++) {
      cm[k]! += p.model[k]!;
      cc[k]! += p.cloud[k]!;
    }
  }
  for (let k = 0; k < 3; k++) {
    cm[k]! /= n;
    cc[k]! /= n;
  }

  // Um par só não define orientação: fica translação pura, que é exatamente o
  // que a pessoa espera do primeiro clique.
  let yaw = 0;
  if (n >= 2) {
    let num = 0;
    let den = 0;
    for (const p of pairs) {
      const mx = p.model[0]! - cm[0]!;
      const my = p.model[1]! - cm[1]!;
      const qx = p.cloud[0]! - cc[0]!;
      const qy = p.cloud[1]! - cc[1]!;
      num += mx * qy - my * qx;
      den += mx * qx + my * qy;
    }
    if (Math.abs(num) > 1e-12 || Math.abs(den) > 1e-12) yaw = Math.atan2(num, den);
  }

  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const translation: [number, number, number] = [
    cc[0]! - (cos * cm[0]! - sin * cm[1]!),
    cc[1]! - (sin * cm[0]! + cos * cm[1]!),
    cc[2]! - cm[2]!,
  ];

  let sum = 0;
  for (const p of pairs) {
    const x = cos * p.model[0]! - sin * p.model[1]! + translation[0];
    const y = sin * p.model[0]! + cos * p.model[1]! + translation[1];
    const z = p.model[2]! + translation[2];
    sum += Math.hypot(x - p.cloud[0]!, y - p.cloud[1]!, z - p.cloud[2]!);
  }

  return { yaw, translation, residual: sum / n, pairs: n };
}
