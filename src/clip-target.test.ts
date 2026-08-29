import { PointCloudObject3D } from "./object.js";
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

/**
 * A união que {@link PointCloudView.frameClouds} enquadra.
 *
 * A falha que isto tranca não lança nada: enquadrar só a primeira nuvem deixa
 * os ladrilhos vizinhos fora do ecrã, e a pessoa arrasta à procura de dados que
 * pensa que não carregaram. Todas as nuvens partilham a origem de cena da
 * primeira, e é essa subtração que se verifica aqui.
 */
describe("targetForClouds", () => {
  function viewWithClouds(
    boxes: { min: [number, number, number]; max: [number, number, number] }[],
    origin: [number, number, number],
  ) {
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
    (view as unknown as { clouds: unknown[] }).clouds.push(
      ...boxes.map((b) => ({
        source: { tightBoundingBox: b },
        // O objeto DE VERDADE, e não um duplo com `getSceneOrigin`: é ele que
        // sabe converter para a cena, e um duplo aqui deixaria de cobrir a
        // colocação — que é precisamente o que pode partir esta conta.
        object: new PointCloudObject3D(origin, origin),
      })),
    );
    return view;
  }

  it("centra no conjunto, não na primeira", () => {
    const view = viewWithClouds(
      [
        { min: [0, 0, 0], max: [10, 10, 10] },
        { min: [90, 0, 0], max: [100, 10, 10] },
      ],
      [0, 0, 0],
    );
    const t = view.targetForClouds();
    // 50 e não 5: a primeira nuvem sozinha daria 5, e é esse o erro.
    expect(t.x).toBe(50);
    expect(t.y).toBe(5);
  });

  it("subtrai a origem de cena partilhada", () => {
    const view = viewWithClouds([{ min: [1000, 2000, 0], max: [1010, 2010, 4] }], [1000, 2000, 0]);
    const t = view.targetForClouds();
    expect([t.x, t.y, t.z]).toEqual([5, 5, 2]);
  });

  it("segue uma nuvem COLOCADA para onde ela foi posta", () => {
    // Sem isto, enquadrar um projeto multi-CRS apontaria a câmera para onde a
    // nuvem estaria se nunca tivesse sido reprojetada — ou seja, para o vazio.
    const view = viewWithClouds([{ min: [0, 0, 0], max: [10, 10, 0] }], [0, 0, 0]);
    const clouds = (view as unknown as { clouds: { object: PointCloudObject3D }[] }).clouds;
    clouds[0]!.object.setPlacement({
      yaw: 0,
      scale: 1,
      pivot: [5, 5, 0],
      at: [1005, 2005, 0],
    });
    const t = view.targetForClouds();
    expect([t.x, t.y]).toEqual([1005, 2005]);
  });

  it("devolve a origem quando não há nuvem nenhuma", () => {
    const view = viewWithClouds([], [0, 0, 0]);
    expect(view.targetForClouds().lengthSq()).toBe(0);
  });
});
