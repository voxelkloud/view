import { describe, expect, it } from "vitest";
import {
  buildTriangleBvh,
  distanceToBvh,
  pointBoxDistanceSq,
  pointTriangleDistanceSq,
} from "./deviation.js";

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
