import { CloudFrame } from "./object.js";
import { PerspectiveCamera } from "three/webgpu";
import { describe, expect, it } from "vitest";
import { PointCloudObject3D, PointCloudView, pickPoint } from "./index.js";

function makeCamera() {
  const camera = new PerspectiveCamera(60, 1, 1, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(10, 0, 10);
  camera.lookAt(10, 0, 0);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();
  return camera;
}

describe("pickPoint", () => {
  it("returns the closest selected point with absolute coordinates and color", () => {
    const camera = makeCamera();

    const result = pickPoint(
      camera,
      1000,
      1000,
      500,
      500,
      [
        {
          cloudIndex: 0,
          frame: new CloudFrame([10, 0, 0], [0, 0, 0]),
          selection: new Int32Array([7]),
          selectionCount: 1,
          node: (index: number) =>
            index === 7
              ? {
                  index: 7,
                  level: 3,
                  minX: 9,
                  minY: -1,
                  minZ: -1,
                  maxX: 13,
                  maxY: 1,
                  maxZ: 1,
                }
              : undefined,
          readPoints: (index: number) =>
            index === 7
              ? {
                  positions: new Float32Array([0, 0, 0, 2, 0, 0]),
                  start: 0,
                  count: 2,
                  colors: new Uint8Array([1, 2, 3, 255, 9, 8, 7, 255]),
                  scalars: new Float32Array([1.5, 2.5]),
                }
              : undefined,
        },
      ] as const,
      { maxDistancePx: 24 },
    );

    expect(result).toEqual({
      position: [10, 0, 0],
      cloudIndex: 0,
      nodeIndex: 7,
      pointIndex: 0,
      screenDistancePx: expect.any(Number),
      color: [1, 2, 3],
      scalarValue: 1.5,
    });
  });

  it("returns scene coordinates for overlays when the cloud is rebased", () => {
    const camera = makeCamera();

    const result = pickPoint(
      camera,
      1000,
      1000,
      500,
      500,
      [
        {
          cloudIndex: 0,
          frame: new CloudFrame([1010, 2000, 3000], [1000, 2000, 3000]),
          selection: new Int32Array([7]),
          selectionCount: 1,
          node: (index: number) =>
            index === 7
              ? {
                  index: 7,
                  level: 3,
                  minX: 1009,
                  minY: 1999,
                  minZ: 2999,
                  maxX: 1013,
                  maxY: 2001,
                  maxZ: 3001,
                }
              : undefined,
          readPoints: () => ({
            positions: new Float32Array([0, 0, 0]),
            start: 0,
            count: 1,
          }),
        },
      ] as const,
      { maxDistancePx: 24 },
    );

    expect(result).toMatchObject({
      position: [1010, 2000, 3000],
      scenePosition: [10, 0, 0],
    });
  });

  it("returns undefined when nothing falls within the picking cone", () => {
    const camera = makeCamera();

    expect(
      pickPoint(
        camera,
        1000,
        1000,
        500,
        500,
        [
          {
            cloudIndex: 0,
            frame: new CloudFrame([10, 0, 0], [0, 0, 0]),
            selection: new Int32Array([7]),
            selectionCount: 1,
            node: () => ({
              index: 7,
              level: 3,
              minX: 20,
              minY: 20,
              minZ: 20,
              maxX: 21,
              maxY: 21,
              maxZ: 21,
            }),
            readPoints: () => ({
              positions: new Float32Array([20, 20, 20]),
              start: 0,
              count: 1,
            }),
          },
        ] as const,
        { maxDistancePx: 3 },
      ),
    ).toBeUndefined();
  });
});

describe("PointCloudView pickPoint", () => {
  it("delegates to the current cloud selection", () => {
    const canvas = {
      getContext: () => null,
      addEventListener() {},
      removeEventListener() {},
      width: 1000,
      height: 1000,
      style: {},
      clientWidth: 1000,
      clientHeight: 1000,
    } as unknown as HTMLCanvasElement;
    const view = new PointCloudView({ canvas });
    view.setSize(1000, 1000, 1);
    view.camera.up.set(0, 0, 1);
    view.camera.position.set(10, 0, 10);
    view.camera.lookAt(10, 0, 0);
    view.camera.updateMatrixWorld(true);
    view.camera.updateProjectionMatrix();

    const cloud = new PointCloudObject3D([10, 0, 0], [0, 0, 0]);
    (view as unknown as { clouds: Array<Record<string, unknown>> }).clouds.push({
      cloudOrigin: [10, 0, 0],
      object: cloud,
      selection: {
        indices: new Int32Array([7]),
        count: 1,
      },
      hierarchy: {
        node: (index: number) =>
          index === 7
            ? {
                index: 7,
                level: 3,
                minX: 9,
                minY: -1,
                minZ: -1,
                maxX: 13,
                maxY: 1,
                maxZ: 1,
              }
            : undefined,
      },
      sink: {
        readPoints: (index: number) =>
          index === 7
            ? {
                positions: new Float32Array([0, 0, 0]),
                start: 0,
                count: 1,
                colors: new Uint8Array([1, 2, 3, 255]),
                scalars: new Float32Array([1.5]),
              }
            : undefined,
      },
    });

    expect(view.pickPoint(500, 500)).toEqual({
      position: [10, 0, 0],
      cloudIndex: 0,
      nodeIndex: 7,
      pointIndex: 0,
      screenDistancePx: expect.any(Number),
      color: [1, 2, 3],
      scalarValue: 1.5,
    });
  });
});

describe("pickPoint numa nuvem COLOCADA", () => {
  /**
   * A propriedade que faz a medição valer alguma coisa num projeto multi-CRS:
   * o ponto é encontrado ONDE ESTÁ DESENHADO — na cena, depois da colocação —
   * mas a coordenada devolvida é a do sistema em que a camada foi entregue.
   *
   * Se `position` viesse no sistema do projeto, a medição de quem carregou um
   * scan em RD New sairia em UTM sem ninguém lhe ter pedido, e não bateria com
   * o levantamento dele. Se `scenePosition` viesse antes da colocação, o pino
   * da anotação apareceria a quilómetros do ponto clicado.
   */
  it("acha o ponto no sítio desenhado e devolve a coordenada NATIVA", () => {
    // A nuvem vive em (1000, 2000) e é levada para junto da câmera por uma
    // colocação que roda 90° e não escala.
    const nativa: [number, number, number] = [1000, 2000, 0];
    const frame = new CloudFrame(nativa, [0, 0, 0], {
      yaw: Math.PI / 2,
      scale: 1,
      pivot: [1000, 2000, 0],
      at: [10, 0, 0],
    });

    const camera = makeCamera();
    const result = pickPoint(
      camera,
      1000,
      1000,
      500,
      500,
      [
        {
          cloudIndex: 0,
          frame,
          selection: new Int32Array([7]),
          selectionCount: 1,
          node: (index: number) =>
            index === 7
              ? {
                  index: 7,
                  level: 3,
                  minX: 999,
                  minY: 1999,
                  minZ: -1,
                  maxX: 1001,
                  maxY: 2001,
                  maxZ: 1,
                }
              : undefined,
          readPoints: (index: number) =>
            index === 7
              ? { positions: new Float32Array([0, 0, 0]), start: 0, count: 1 }
              : undefined,
        },
      ] as const,
      { maxDistancePx: 24 },
    );

    expect(result).toBeDefined();
    // A câmera olha para (10, 0, 0), que é para onde o pivô foi mandado.
    expect(result!.scenePosition![0]).toBeCloseTo(10, 6);
    expect(result!.scenePosition![1]).toBeCloseTo(0, 6);
    // E a coordenada devolvida é a da NUVEM, não a da cena.
    expect(result!.position[0]).toBeCloseTo(nativa[0], 6);
    expect(result!.position[1]).toBeCloseTo(nativa[1], 6);
  });
});
