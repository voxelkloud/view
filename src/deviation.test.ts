import { describe, expect, it } from "vitest";
import { Group, Vector3 } from "three";
import {
  buildTriangleBvh,
  clusterDeviation,
  raycastBvh,
  solveAlignment,
  distanceToBvh,
  pointBoxDistanceSq,
  pointTriangleDistanceSq,
} from "./deviation.js";
import type { Alignment } from "./deviation.js";

/** Um triângulo no plano z = 0, com catetos de comprimento 1 na origem. */
const UNIT = [0, 0, 0, 1, 0, 0, 0, 1, 0] as const;
const distTo = (p: readonly [number, number, number], t: readonly number[] = UNIT): number =>
  Math.sqrt(pointTriangleDistanceSq(p[0], p[1], p[2], ...(t as [number, number, number, number, number, number, number, number, number])));

describe("pointTriangleDistanceSq", () => {
  it("mede a perpendicular quando o ponto cai DENTRO do triângulo", () => {
    expect(distTo([0.25, 0.25, 3])).toBeCloseTo(3, 10);
    expect(distTo([0.25, 0.25, -0.5])).toBeCloseTo(0.5, 10);
  });

  it("é zero sobre a superfície", () => {
    expect(distTo([0.3, 0.3, 0])).toBeCloseTo(0, 10);
  });

  it("cai no VÉRTICE quando o ponto está além do canto", () => {
    // Fora dos três semi-planos: a resposta é a distância ao vértice, não a
    // perpendicular ao plano — que é o erro clássico e daria 0 aqui.
    expect(distTo([-1, -1, 0])).toBeCloseTo(Math.SQRT2, 10);
    expect(distTo([3, 0, 0])).toBeCloseTo(2, 10);
  });

  it("cai na ARESTA quando o ponto está ao lado dela", () => {
    // Frente à hipotenusa, que vai de (1,0,0) a (0,1,0).
    expect(distTo([1, 1, 0])).toBeCloseTo(Math.SQRT1_2, 10);
    // Ao lado do cateto em y = 0.
    expect(distTo([0.5, -2, 0])).toBeCloseTo(2, 10);
  });

  it("compõe a perpendicular com o afastamento lateral", () => {
    // Fora pelo vértice (0,0,0) em xy, e 4 acima: sqrt(2) no plano, 4 fora.
    expect(distTo([-1, -1, 4])).toBeCloseTo(Math.hypot(Math.SQRT2, 4), 10);
  });
});

describe("pointBoxDistanceSq", () => {
  it("é zero dentro da caixa", () => {
    expect(pointBoxDistanceSq(0.5, 0.5, 0.5, 0, 0, 0, 1, 1, 1)).toBe(0);
  });
  it("mede por face, aresta e canto", () => {
    expect(pointBoxDistanceSq(2, 0.5, 0.5, 0, 0, 0, 1, 1, 1)).toBeCloseTo(1, 10);
    expect(pointBoxDistanceSq(2, 2, 0.5, 0, 0, 0, 1, 1, 1)).toBeCloseTo(2, 10);
    expect(pointBoxDistanceSq(2, 2, 2, 0, 0, 0, 1, 1, 1)).toBeCloseTo(3, 10);
  });
});

/** Uma grelha de triângulos no plano z = 0, `n` por `n` células. */
function grid(n: number): Float32Array {
  const out = new Float32Array(n * n * 2 * 9);
  let k = 0;
  const put = (...v: number[]): void => {
    out.set(v, k);
    k += 9;
  };
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      put(i, j, 0, i + 1, j, 0, i, j + 1, 0);
      put(i + 1, j, 0, i + 1, j + 1, 0, i, j + 1, 0);
    }
  }
  return out;
}

describe("buildTriangleBvh", () => {
  it("mantém todos os triângulos, apenas reordenados", () => {
    const tris = grid(4);
    const bvh = buildTriangleBvh(tris);
    expect(bvh.triCount).toBe(32);
    expect(bvh.tris.length).toBe(tris.length);
    // A soma das coordenadas é invariante a permutação; se um triângulo se
    // perdesse ou duplicasse na reordenação, isto muda.
    const sum = (a: Float32Array): number => a.reduce((s, v) => s + v, 0);
    expect(sum(bvh.tris)).toBeCloseTo(sum(tris), 4);
  });

  it("dá a MESMA distância que a força bruta, em 200 pontos aleatórios", () => {
    // O teste que importa: a árvore é uma otimização, e uma otimização que
    // muda a resposta é um bug. Poda errada num nó devolve uma distância maior
    // e ninguém repara — o mapa de cores só fica um pouco diferente.
    const tris = grid(6);
    const bvh = buildTriangleBvh(tris);
    const brute = (x: number, y: number, z: number): number => {
      let best = Infinity;
      for (let t = 0; t < tris.length / 9; t++) {
        const o = t * 9;
        const d = pointTriangleDistanceSq(x, y, z,
          tris[o]!, tris[o + 1]!, tris[o + 2]!,
          tris[o + 3]!, tris[o + 4]!, tris[o + 5]!,
          tris[o + 6]!, tris[o + 7]!, tris[o + 8]!);
        if (d < best) best = d;
      }
      return Math.sqrt(best);
    };
    // Sequência determinística: um teste que falha uma vez em vinte é pior que
    // não ter teste.
    let seed = 12345;
    const rnd = (): number => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let i = 0; i < 200; i++) {
      const x = rnd() * 10 - 2;
      const y = rnd() * 10 - 2;
      const z = rnd() * 6 - 3;
      expect(distanceToBvh(bvh, x, y, z)).toBeCloseTo(brute(x, y, z), 5);
    }
  });

  it("aceita uma malha vazia sem explodir", () => {
    const bvh = buildTriangleBvh(new Float32Array(0));
    expect(bvh.triCount).toBe(0);
    expect(distanceToBvh(bvh, 0, 0, 0)).toBe(Infinity);
  });

  it("resolve um único triângulo, que é o caso degenerado da árvore", () => {
    const bvh = buildTriangleBvh(new Float32Array(UNIT));
    expect(bvh.nodeCount).toBe(1);
    expect(distanceToBvh(bvh, 0.25, 0.25, 2)).toBeCloseTo(2, 10);
  });
});

describe("clusterDeviation", () => {
  /** `n` pontos numa esfera de raio `r` em torno de `c`, com desvio `d`. */
  function blob(
    c: readonly [number, number, number],
    r: number,
    n: number,
    d: number,
    seed: number,
  ): { pos: number[]; dev: number[] } {
    const pos: number[] = [];
    const dev: number[] = [];
    let s = seed;
    const rnd = (): number => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let i = 0; i < n; i++) {
      pos.push(c[0] + (rnd() - 0.5) * r, c[1] + (rnd() - 0.5) * r, c[2] + (rnd() - 0.5) * r);
      dev.push(d);
    }
    return { pos, dev };
  }

  const join = (...parts: { pos: number[]; dev: number[] }[]) => ({
    positions: new Float32Array(parts.flatMap((p) => p.pos)),
    deviations: new Float32Array(parts.flatMap((p) => p.dev)),
  });

  it("separa duas regiões distantes e junta o que é contíguo", () => {
    const { positions, deviations } = join(
      blob([0, 0, 0], 3, 200, 0.3, 1),
      blob([50, 0, 0], 3, 200, 0.5, 2),
    );
    const cs = clusterDeviation(positions, deviations, { tolerance: 0.05, ceiling: 2 });
    expect(cs).toHaveLength(2);
    // O pior primeiro: quem lê um relatório lê as primeiras linhas.
    expect(cs[0]!.maxDeviation).toBeCloseTo(0.5, 6);
    expect(cs[1]!.maxDeviation).toBeCloseTo(0.3, 6);
    expect(cs[0]!.centre[0]).toBeCloseTo(50, 0);
  });

  it("ignora o que está DENTRO da tolerância", () => {
    const { positions, deviations } = join(blob([0, 0, 0], 3, 300, 0.02, 3));
    expect(clusterDeviation(positions, deviations, { tolerance: 0.05, ceiling: 2 })).toHaveLength(0);
  });

  it("descarta os saturados, que são ausência de modelo e não desvio", () => {
    // O terreno em volta do prédio satura no teto. Se entrasse, o maior tópico
    // seria sempre o chão.
    const { positions, deviations } = join(blob([0, 0, 0], 5, 500, 2, 4));
    expect(clusterDeviation(positions, deviations, { tolerance: 0.05, ceiling: 2 })).toHaveLength(0);
  });

  it("descarta regiões pequenas demais, que são ruído do scan", () => {
    const { positions, deviations } = join(blob([0, 0, 0], 1, 10, 0.4, 5));
    expect(clusterDeviation(positions, deviations, { tolerance: 0.05, ceiling: 2 })).toHaveLength(0);
    expect(
      clusterDeviation(positions, deviations, { tolerance: 0.05, ceiling: 2, minPoints: 5 }),
    ).toHaveLength(1);
  });

  it("dá o MESMO agrupamento duas vezes", () => {
    // Um relatório que muda de tópicos entre duas exportações do mesmo dado é
    // pior que não ter relatório. É por isto que não há k-means aqui.
    const { positions, deviations } = join(
      blob([0, 0, 0], 4, 300, 0.3, 6),
      blob([30, 10, 0], 4, 300, 0.6, 7),
    );
    const a = clusterDeviation(positions, deviations, { tolerance: 0.05, ceiling: 2 });
    const b = clusterDeviation(positions, deviations, { tolerance: 0.05, ceiling: 2 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it("nomeia os elementos atingidos quando a malha traz ids", () => {
    // Dois triângulos, elementos 7 e 9, e uma nuvem de pontos sobre o 9.
    const tris = new Float32Array([
      0, 0, 0, 10, 0, 0, 0, 10, 0,
      100, 0, 0, 110, 0, 0, 100, 10, 0,
    ]);
    const bvh = buildTriangleBvh(tris, new Uint32Array([7, 9]));
    const { positions, deviations } = join(blob([104, 4, 0.3], 1, 100, 0.3, 8));
    const cs = clusterDeviation(positions, deviations, {
      tolerance: 0.05,
      ceiling: 2,
      bvh,
      minPoints: 10,
    });
    expect(cs).toHaveLength(1);
    expect(cs[0]!.features[0]).toBe(9);
  });
});

describe("raycastBvh", () => {
  const quad = new Float32Array([
    // dois triângulos formando um quadrado 10x10 no plano z = 5
    0, 0, 5, 10, 0, 5, 0, 10, 5,
    10, 0, 5, 10, 10, 5, 0, 10, 5,
  ]);

  it("acerta o plano e devolve o ponto, não só a distância", () => {
    const bvh = buildTriangleBvh(quad, new Uint32Array([3, 3]));
    const hit = raycastBvh(bvh, 4, 4, 20, 0, 0, -1)!;
    expect(hit).toBeDefined();
    expect(hit.point[2]).toBeCloseTo(5, 6);
    expect(hit.point[0]).toBeCloseTo(4, 6);
    expect(hit.distance).toBeCloseTo(15, 6);
    expect(hit.feature).toBe(3);
  });

  it("erra quando o raio passa ao lado", () => {
    const bvh = buildTriangleBvh(quad);
    expect(raycastBvh(bvh, 40, 40, 20, 0, 0, -1)).toBeUndefined();
  });

  it("ignora o que está ATRÁS da origem do raio", () => {
    // Câmera abaixo do plano olhando para baixo: o quadrado está atrás.
    const bvh = buildTriangleBvh(quad);
    expect(raycastBvh(bvh, 4, 4, 1, 0, 0, -1)).toBeUndefined();
  });

  it("aceita raio paralelo a um eixo sem devolver NaN", () => {
    // inv = ±Infinity no slab test. Uma comparação encadeada daria NaN aqui e a
    // caixa seria descartada em silêncio.
    const bvh = buildTriangleBvh(quad);
    const hit = raycastBvh(bvh, 5, 5, 100, 0, 0, -1);
    expect(hit?.point[2]).toBeCloseTo(5, 6);
  });

  it("atravessa de baixo para cima — uma parede vista por dentro", () => {
    const bvh = buildTriangleBvh(quad);
    expect(raycastBvh(bvh, 5, 5, 0, 0, 0, 1)?.point[2]).toBeCloseTo(5, 6);
  });
});

describe("solveAlignment", () => {
  const apply = (a: Alignment, p: readonly [number, number, number]) => [
    Math.cos(a.yaw) * p[0] - Math.sin(a.yaw) * p[1] + a.translation[0],
    Math.sin(a.yaw) * p[0] + Math.cos(a.yaw) * p[1] + a.translation[1],
    p[2] + a.translation[2],
  ];

  it("um par só dá translação pura, que é o que a pessoa espera do 1º clique", () => {
    const a = solveAlignment([{ model: [0, 0, 0], cloud: [10, 20, 30] }])!;
    expect(a.yaw).toBe(0);
    expect(a.translation).toEqual([10, 20, 30]);
    expect(a.residual).toBeCloseTo(0, 9);
  });

  it("recupera uma guinada e uma translação conhecidas, exatamente", () => {
    const yaw = 0.35;
    const t: [number, number, number] = [185940, 428238, 11.1];
    const model: [number, number, number][] = [
      [0, 0, 0], [24, 0, 0], [24, 25, 0], [0, 25, 15],
    ];
    const pairs = model.map((m) => ({
      model: m,
      cloud: [
        Math.cos(yaw) * m[0] - Math.sin(yaw) * m[1] + t[0],
        Math.sin(yaw) * m[0] + Math.cos(yaw) * m[1] + t[1],
        m[2] + t[2],
      ] as [number, number, number],
    }));
    const a = solveAlignment(pairs)!;
    expect(a.yaw).toBeCloseTo(yaw, 9);
    expect(a.translation[0]).toBeCloseTo(t[0], 4);
    expect(a.translation[1]).toBeCloseTo(t[1], 4);
    expect(a.translation[2]).toBeCloseTo(t[2], 6);
    expect(a.residual).toBeLessThan(1e-3);
  });

  it("aguenta cliques imprecisos e diz quanto errou", () => {
    // O caso real: ninguém acerta o mesmo canto ao centímetro em duas janelas.
    // O que importa é que a solução continue perto E que o RESÍDUO conte a
    // verdade — é ele que a UI mostra para a pessoa saber se pode confiar.
    const yaw = 0.35;
    const t: [number, number, number] = [185940, 428238, 11.1];
    const model: [number, number, number][] = [
      [0, 0, 0], [24, 0, 0], [24, 25, 0], [0, 25, 15],
    ];
    const noise = [0.12, -0.09, 0.15, -0.11, 0.08, -0.14, 0.1, -0.07, 0.13, 0.05, -0.06, 0.09];
    const pairs = model.map((m, i) => ({
      model: m,
      cloud: [
        Math.cos(yaw) * m[0] - Math.sin(yaw) * m[1] + t[0] + noise[i * 3]!,
        Math.sin(yaw) * m[0] + Math.cos(yaw) * m[1] + t[1] + noise[i * 3 + 1]!,
        m[2] + t[2] + noise[i * 3 + 2]!,
      ] as [number, number, number],
    }));
    const a = solveAlignment(pairs)!;
    expect(Math.abs(a.yaw - yaw)).toBeLessThan(0.02);
    expect(Math.hypot(a.translation[0] - t[0], a.translation[1] - t[1])).toBeLessThan(0.2);
    // O resíduo tem de ser da ordem do ruído, nem zero nem enorme: zero seria
    // um solver que sobreajusta e mente sobre a qualidade.
    expect(a.residual).toBeGreaterThan(0.05);
    expect(a.residual).toBeLessThan(0.4);
  });

  it("NÃO inclina o modelo, mesmo quando os pares pedem", () => {
    // Um prédio está a prumo. Deixar o solver inclinar faz três cliques
    // imprecisos deitarem-no de lado, com resíduo menor e resultado pior.
    const pairs = [
      { model: [0, 0, 0] as const, cloud: [0, 0, 0] as const },
      { model: [10, 0, 0] as const, cloud: [10, 0, 10] as const },
      { model: [0, 10, 0] as const, cloud: [0, 10, 0] as const },
    ];
    const a = solveAlignment(pairs)!;
    // A única saída é guinada + translação; o desnível vira resíduo, e é assim
    // que a pessoa descobre que clicou errado.
    expect(a.residual).toBeGreaterThan(1);
    const flat = apply(a, [10, 0, 0]);
    expect(flat[2]).toBeCloseTo(a.translation[2], 9);
  });
});

// A UI da B4 não aplica a solução ao modelo: aplica-a ao PIVÔ, por cima do que
// já lá está. Essa composição é o passo que erra em silêncio — um prédio rodado
// em torno da origem da cena, a centenas de metros dali, ainda tem resíduo
// baixo no solver e está no sítio errado no ecrã. Aqui está a mesma aritmética
// que `bim.ts` escreve, contra o Object3D do three a dizer onde o modelo acaba.
describe("compor o alinhamento sobre o pivô", () => {
  const compose = (pivot: Group, a: Alignment): void => {
    const c = Math.cos(a.yaw);
    const s = Math.sin(a.yaw);
    const px = pivot.position.x;
    const py = pivot.position.y;
    pivot.position.set(
      c * px - s * py + a.translation[0],
      s * px + c * py + a.translation[1],
      pivot.position.z + a.translation[2],
    );
    pivot.rotation.z += a.yaw;
    pivot.updateMatrixWorld(true);
  };

  /** Onde um ponto local do modelo cai na cena, com o pivô como está. */
  const world = (pivot: Group, p: [number, number, number]): [number, number, number] => {
    const v = new Vector3(p[0], p[1], p[2]).applyMatrix4(pivot.matrixWorld);
    return [v.x, v.y, v.z];
  };

  const LOCAL: [number, number, number][] = [
    [0, 0, 0],
    [12, 0, 0],
    [12, 8, 3],
    [0, 8, 0],
  ];

  it("leva o modelo exatamente para onde os pares dizem", () => {
    // O pivô já está deslocado e rodado, como está sempre depois da B4 grosseira.
    const pivot = new Group();
    pivot.position.set(-137.5, 402.25, 11.3);
    pivot.rotation.z = 0.42;
    pivot.updateMatrixWorld(true);

    // A verdade: o modelo está 0.3 rad e (4, -7, 0.6) fora do sítio.
    const yaw = 0.3;
    const t: [number, number, number] = [4, -7, 0.6];
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const truth = (p: [number, number, number]): [number, number, number] => [
      c * p[0] - s * p[1] + t[0],
      s * p[0] + c * p[1] + t[1],
      p[2] + t[2],
    ];

    const pairs = LOCAL.map((l) => {
      const m = world(pivot, l);
      return { model: m, cloud: truth(m) };
    });

    const a = solveAlignment(pairs);
    expect(a).toBeDefined();
    expect(a!.residual).toBeLessThan(1e-6);

    compose(pivot, a!);

    // O que importa não é o pivô, é onde o modelo aterra.
    for (const l of LOCAL) {
      const got = world(pivot, l);
      const want = truth(
        // ...comparado com onde estava ANTES de compor.
        (() => {
          const p = new Vector3(l[0], l[1], l[2]);
          const before = new Group();
          before.position.set(-137.5, 402.25, 11.3);
          before.rotation.z = 0.42;
          before.updateMatrixWorld(true);
          p.applyMatrix4(before.matrixWorld);
          return [p.x, p.y, p.z] as [number, number, number];
        })(),
      );
      expect(got[0]).toBeCloseTo(want[0], 6);
      expect(got[1]).toBeCloseTo(want[1], 6);
      expect(got[2]).toBeCloseTo(want[2], 6);
    }
  });

  it("não move nada quando os pares já coincidem", () => {
    const pivot = new Group();
    pivot.position.set(80, -20, 5);
    pivot.rotation.z = 1.1;
    pivot.updateMatrixWorld(true);
    const pairs = LOCAL.map((l) => {
      const m = world(pivot, l);
      return { model: m, cloud: m };
    });
    const a = solveAlignment(pairs)!;
    compose(pivot, a);
    expect(pivot.position.x).toBeCloseTo(80, 6);
    expect(pivot.position.y).toBeCloseTo(-20, 6);
    expect(pivot.position.z).toBeCloseTo(5, 6);
    expect(pivot.rotation.z).toBeCloseTo(1.1, 6);
  });

  it("compor duas vezes converge em vez de divergir", () => {
    // Ninguém acerta à primeira: clica-se, aplica-se, olha-se, clica-se outra
    // vez. A segunda passagem tem de aproximar. Se a composição estivesse
    // errada, a primeira aplicação parecia certa e a segunda mandava o prédio
    // para longe — que é como este bug se manifesta na prática.
    const pivot = new Group();
    pivot.position.set(10, 10, 0);
    pivot.updateMatrixWorld(true);

    const target = new Group();
    target.position.set(-55, 130, 2.5);
    target.rotation.z = 0.9;
    target.updateMatrixWorld(true);
    const cloud = LOCAL.map((l) => world(target, l));

    let err = Infinity;
    for (let pass = 0; pass < 2; pass++) {
      const pairs = LOCAL.map((l, i) => ({ model: world(pivot, l), cloud: cloud[i]! }));
      compose(pivot, solveAlignment(pairs)!);
      err = Math.max(
        ...LOCAL.map((l, i) => {
          const g = world(pivot, l);
          return Math.hypot(g[0] - cloud[i]![0], g[1] - cloud[i]![1], g[2] - cloud[i]![2]);
        }),
      );
    }
    expect(err).toBeLessThan(1e-6);
  });
});
