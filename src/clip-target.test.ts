import { describe, expect, it } from "vitest";
import { PointCloudView } from "./index.js";

/**
 * Quem o corte atinge.
 *
 * O DEC-B6 diz que uma SEÇÃO corta os dois — fatiar o scan e deixar a parede em
 * pé parece uma resposta e não é. O alvo `"model"` é a excepção deliberada, o
 * "tirar o telhado para ver dentro", e a falha que este teste tranca é a que
 * não lança nada: um corte de apresentação que também comesse a nuvem, ou um
 * `"all"` que deixasse de comer.
 */
function viewWithSpies() {
  const canvas = {
    getContext: () => null,
    addEventListener() {},
    removeEventListener() {},
    width: 100,
    height: 100,
    style: {},
    clientWidth: 100,
    clientHeight: 100,
  } as unknown as HTMLCanvasElement;
  const view = new PointCloudView({ canvas });

  const seen: { points: (Float32Array | undefined)[]; model: (Float32Array | undefined)[] } = {
    points: [],
    model: [],
  };
  const internals = view as unknown as {
    clouds: Array<Record<string, unknown>>;
    overlay: unknown;
  };
  internals.clouds.push({
    computeSink: {
      setClipPlanes: (p: Float32Array | undefined) => seen.points.push(p),
    },
  });
  internals.overlay = {
    setClipPlanes: (p: Float32Array | undefined) => seen.model.push(p),
  };
  return { view, seen };
}

const PLANE = new Float32Array([0, 0, -1, 12]);

describe("setClipPlanes targeting", () => {
  it("corta os dois por omissão, que é o DEC-B6", () => {
    const { view, seen } = viewWithSpies();
    view.setClipPlanes(PLANE);
    expect(seen.model).toEqual([PLANE]);
    expect(seen.points).toEqual([PLANE]);
  });

  it('com "model" corta o BIM e deixa a nuvem intacta', () => {
    const { view, seen } = viewWithSpies();
    view.setClipPlanes(PLANE, "model");
    expect(seen.model).toEqual([PLANE]);
    // O ponto todo da feature: a nuvem tem de receber `undefined`, e não ficar
    // sem chamada — senão um corte "all" anterior continuaria a valer nela.
    expect(seen.points).toEqual([undefined]);
  });

  it('volta a cortar a nuvem ao trocar de "model" para "all"', () => {
    const { view, seen } = viewWithSpies();
    view.setClipPlanes(PLANE, "model");
    view.setClipPlanes(PLANE, "all");
    expect(seen.points).toEqual([undefined, PLANE]);
  });

  it("limpar o corte limpa os dois em qualquer alvo", () => {
    const { view, seen } = viewWithSpies();
    view.setClipPlanes(PLANE, "model");
    view.setClipPlanes(undefined, "model");
    expect(seen.model).toEqual([PLANE, undefined]);
    expect(seen.points).toEqual([undefined, undefined]);
  });
});
