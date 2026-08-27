import { describe, expect, it } from "vitest";
import { GROUND_TAIL, GroundLevel } from "./ground-level.js";

/** `n` points at height `z`, in the `3 * n` float32 layout a node carries. */
function slab(n: number, z: number): Float32Array {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) out[i * 3 + 2] = z;
  return out;
}

/** Stream `n` points at `z` in, a node at a time, so `settled` moves as it would. */
function stream(g: GroundLevel, n: number, z: number, cls?: number, chunk = 50_000): void {
  for (let left = n; left > 0; left -= chunk) {
    const c = Math.min(chunk, left);
    g.add(slab(c, z), c, cls === undefined ? undefined : new Uint8Array(c).fill(cls));
  }
}

describe("GroundLevel", () => {
  it("says nothing until the cloud has streamed in far enough", () => {
    // The failure this guards: a COPC root carries the noise at the cloud's own
    // rate, so an answer given from it is not rough — it is in the tail.
    const g = new GroundLevel(-333, 337, 5_000_000);
    stream(g, 199_999, -1);
    expect(g.settled).toBe(false);
    expect(g.get()).toBeUndefined();
    g.add(slab(1, -1), 1);
    expect(g.settled).toBe(true);
    expect(g.get()).toBeDefined();
  });

  it("does not make a small cloud wait for points it does not have", () => {
    const g = new GroundLevel(0, 100, 30_000);
    stream(g, 30_000, 40);
    expect(g.settled).toBe(true);
    expect(g.get()).toBeDefined();
  });

  it("ignores a tail the minimum would be defined by", () => {
    // The shape of the Morro Bay tile, scaled down: a survey sitting in a
    // narrow band with a thin smear of noise hundreds of metres below it.
    const g = new GroundLevel(-333, 337, 500_000);
    stream(g, 400_000, -1);
    stream(g, 400, -333);

    // The declared floor is -333. The ground is not.
    const z = g.get()!;
    expect(z).toBeGreaterThan(-10);
    expect(z).toBeLessThanOrEqual(-1);
  });

  it("follows the ground when the noise is thicker than the cut", () => {
    // 1% below, against a 0.2% budget: this noise is NOT dismissable, and the
    // estimate has to stay in it rather than pretend the survey starts higher.
    const g = new GroundLevel(-100, 100, 500_000);
    stream(g, 396_000, 0);
    stream(g, 4_000, -100);
    expect(g.get()!).toBeLessThan(-50);
  });

  it("lands under the data, never through it", () => {
    // A clean cloud: every point at 40, nothing below. The estimate must not
    // come out above the survey, or imagery placed on it cuts the cloud.
    const g = new GroundLevel(0, 100, 200_000);
    stream(g, 200_000, 40);
    const z = g.get()!;
    expect(z).toBeLessThanOrEqual(40);
    // …and not so far under that it is useless: one bucket of 100/1024.
    expect(z).toBeGreaterThan(40 - 100 / 1024 - 1e-6);
  });

  it("answers a flat cloud with its one height", () => {
    const g = new GroundLevel(7, 7, 20_000);
    stream(g, 20_000, 7);
    expect(g.get()).toBe(7);
  });

  it("clamps points that fall outside the declared range", () => {
    // A tight box is a claim, and a decoded point may sit a quantum outside it.
    // Bucketing must not write out of bounds or lose the point.
    const g = new GroundLevel(0, 100, 200_000);
    stream(g, 100_000, 500);
    stream(g, 100_000, -500);
    expect(g.get()).toBe(0);
  });

  it("reads the other tail from the same histogram", () => {
    // O que `robustZRange` faz: a mesma fração, lida da ponta de cima. Uma
    // nuvem com ruído nas DUAS pontas tem de perder as duas.
    const g = new GroundLevel(-100, 100, 500_000);
    stream(g, 500, -100); // 0,125% — dentro do orçamento, logo descartável
    stream(g, 200_000, -10);
    stream(g, 199_000, 10);
    stream(g, 500, 100);
    const lo = g.get(GROUND_TAIL)!;
    const hi = g.get(1 - GROUND_TAIL)!;
    // A faixa fecha sobre os dados e não sobre as duas pontas.
    expect(lo).toBeGreaterThan(-11);
    expect(lo).toBeLessThan(0);
    expect(hi).toBeGreaterThan(0);
    expect(hi).toBeLessThan(11);
  });

  it("counts the survey and not the file's own noise", () => {
    // A forma do ladrilho de Morro Bay: uma banda de ruído alto que é 1,45%
    // dos pontos — sete vezes a cauda descartada, e CONTÍNUA com o terreno,
    // logo nenhum percentil a alcança. O que a alcança é o rótulo.
    const g = new GroundLevel(0, 700, 500_000);
    stream(g, 400_000, 340); // terreno
    stream(g, 6_000, 640, 18); // ruído alto, muito acima da cauda de 0,2%
    stream(g, 200, 10, 7); // ruído baixo
    const hi = g.get(1 - GROUND_TAIL)!;
    expect(hi).toBeLessThan(400);
    expect(g.get()!).toBeGreaterThan(300);
  });

  it("does not let dropped noise dilute the tail", () => {
    // Duas nuvens com o MESMO levantamento e quantidades diferentes de ruído
    // rotulado têm de dar a mesma resposta: o ruído sai do numerador e do
    // denominador, senão a fração passa a depender de quanto lixo veio junto.
    const survey = (g: GroundLevel) => {
      stream(g, 100_000, 10);
      stream(g, 100_000, 20);
      stream(g, 100_000, 30);
    };
    const a = new GroundLevel(0, 100, 400_000);
    survey(a);
    const b = new GroundLevel(0, 100, 400_000);
    survey(b);
    stream(b, 300_000, 99, 18);
    expect(b.get()).toBe(a.get());
    expect(b.get(1 - GROUND_TAIL)).toBe(a.get(1 - GROUND_TAIL));
  });

  it("takes the fraction as a knob", () => {
    const g = new GroundLevel(0, 100, 200_000);
    stream(g, 180_000, 90);
    stream(g, 20_000, 10);
    // Below the 10% that sits low, and above it.
    expect(g.get(0.05)!).toBeLessThan(20);
    expect(g.get(0.5)!).toBeGreaterThan(80);
  });
});
