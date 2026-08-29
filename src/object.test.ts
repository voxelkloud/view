/**
 * A colocação de uma nuvem cujas coordenadas não são as do projeto.
 *
 * Dois grupos de teste, e o primeiro é o que importa mais: SEM colocação, a
 * `CloudFrame` tem de dar exatamente os mesmos números que a subtração de
 * origens que ela substituiu em quatro arquivos. Essa é a parte que já
 * funcionava, é a que toda a gente usa, e é a que uma generalização silenciosa
 * partiria.
 *
 * O segundo grupo é a álgebra nova: ida e volta, o pivô que aterra onde foi
 * mandado, e a caixa que cresce ao rodar em vez de encolher.
 */
import { describe, expect, it } from "vitest";
import { Vector3 } from "three";
import { CloudFrame, PointCloudObject3D, type CloudPlacement } from "./object.js";

const out = () => ({ x: 0, y: 0, z: 0 });

const CLOUD: [number, number, number] = [636001.76, 848935.2, 406.14];
const SCENE: [number, number, number] = [637000, 849000, 400];

describe("CloudFrame sem colocação — o comportamento antigo, intacto", () => {
  const f = new CloudFrame(CLOUD, SCENE, undefined);

  it("diz que é só translação", () => {
    expect(f.isTranslationOnly).toBe(true);
  });

  it("local para cena é a soma de `cloudOrigin - sceneOrigin`", () => {
    const o = out();
    f.localToScene(10, 20, 30, o);
    expect(o.x).toBeCloseTo(10 + CLOUD[0] - SCENE[0], 9);
    expect(o.y).toBeCloseTo(20 + CLOUD[1] - SCENE[1], 9);
    expect(o.z).toBeCloseTo(30 + CLOUD[2] - SCENE[2], 9);
  });

  it("cena para absoluto é a soma de `sceneOrigin`", () => {
    const o = out();
    f.sceneToAbs(-500, 60, 12, o);
    expect(o.x).toBeCloseTo(-500 + SCENE[0], 9);
    expect(o.y).toBeCloseTo(60 + SCENE[1], 9);
    expect(o.z).toBeCloseTo(12 + SCENE[2], 9);
  });

  it("absoluto para cena é a subtração de `sceneOrigin`", () => {
    const o = out();
    f.absToScene(CLOUD[0], CLOUD[1], CLOUD[2], o);
    expect(o.x).toBeCloseTo(CLOUD[0] - SCENE[0], 9);
    expect(o.y).toBeCloseTo(CLOUD[1] - SCENE[1], 9);
    expect(o.z).toBeCloseTo(CLOUD[2] - SCENE[2], 9);
  });

  it("a caixa da cena é a caixa absoluta deslocada, sem crescer", () => {
    const b = new Float64Array(6);
    f.sceneBox(100, 200, 0, 140, 260, 10, b);
    expect([...b]).toEqual([
      100 - SCENE[0],
      200 - SCENE[1],
      0 - SCENE[2],
      140 - SCENE[0],
      260 - SCENE[1],
      10 - SCENE[2],
    ]);
  });

  it("escala de comprimento 1", () => {
    expect(f.lengthScale).toBe(1);
  });
});

describe("CloudFrame com colocação", () => {
  // 30°, e uma escala bem longe de 1 para que um fator esquecido não passe.
  const p: CloudPlacement = {
    yaw: Math.PI / 6,
    scale: 0.3048006096012192,
    pivot: [1000, 2000, 50],
    at: [500_000, 7_000_000, 120],
  };
  const f = new CloudFrame(CLOUD, SCENE, p);

  it("põe o pivô exatamente onde foi mandado", () => {
    // O pivô é o ponto onde o ajuste foi ancorado: é onde a transformação tem
    // de ser exata, não aproximada.
    const o = out();
    f.absToScene(p.pivot[0], p.pivot[1], p.pivot[2], o);
    expect(o.x).toBeCloseTo(p.at[0] - SCENE[0], 6);
    expect(o.y).toBeCloseTo(p.at[1] - SCENE[1], 6);
    expect(o.z).toBeCloseTo(p.at[2] - SCENE[2], 6);
  });

  it("volta ao ponto de partida — ida e volta", () => {
    const o = out();
    const back = out();
    for (const q of [
      [0, 0, 0],
      [1234.5, -678.9, 42],
      [p.pivot[0], p.pivot[1], p.pivot[2]],
    ] as const) {
      f.absToScene(q[0], q[1], q[2], o);
      f.sceneToAbs(o.x, o.y, o.z, back);
      expect(back.x).toBeCloseTo(q[0], 6);
      expect(back.y).toBeCloseTo(q[1], 6);
      expect(back.z).toBeCloseTo(q[2], 6);
    }
  });

  it("local e absoluto contam a mesma história", () => {
    // `local` é `abs - cloudOrigin`: as duas rotas para a cena têm de coincidir,
    // ou o laço de picking e o laço de perfil discordariam sobre o mesmo ponto.
    const viaLocal = out();
    const viaAbs = out();
    f.localToScene(10, 20, 30, viaLocal);
    f.absToScene(10 + CLOUD[0], 20 + CLOUD[1], 30 + CLOUD[2], viaAbs);
    expect(viaLocal.x).toBeCloseTo(viaAbs.x, 6);
    expect(viaLocal.y).toBeCloseTo(viaAbs.y, 6);
    expect(viaLocal.z).toBeCloseTo(viaAbs.z, 6);
  });

  it("preserva distâncias a menos da escala — é uma semelhança", () => {
    const a = out();
    const b = out();
    f.absToScene(0, 0, 0, a);
    f.absToScene(300, 400, 0, b);
    // 500 na nuvem, 500 * escala na cena. Se houvesse cisalhamento ou escala
    // não uniforme, este número mudaria com a direção.
    expect(Math.hypot(b.x - a.x, b.y - a.y)).toBeCloseTo(500 * p.scale, 6);
  });

  it("não é só translação, e diz que não é", () => {
    expect(f.isTranslationOnly).toBe(false);
    expect(f.lengthScale).toBe(p.scale);
  });

  it("um yaw de 90° troca os eixos", () => {
    const g = new CloudFrame([0, 0, 0], [0, 0, 0], {
      yaw: Math.PI / 2,
      scale: 1,
      pivot: [0, 0, 0],
      at: [0, 0, 0],
    });
    const o = out();
    g.absToScene(10, 0, 0, o);
    expect(o.x).toBeCloseTo(0, 9);
    expect(o.y).toBeCloseTo(10, 9);
  });
});

describe("CloudFrame.sceneBox sob rotação", () => {
  const g = new CloudFrame([0, 0, 0], [0, 0, 0], {
    yaw: Math.PI / 4,
    scale: 1,
    pivot: [0, 0, 0],
    at: [0, 0, 0],
  });

  it("cresce, nunca encolhe — a caixa rodada já não é alinhada aos eixos", () => {
    const b = new Float64Array(6);
    // Um quadrado de 10 m a 45° tem uma diagonal de 14.14 m em cada eixo.
    g.sceneBox(-5, -5, 0, 5, 5, 2, b);
    expect(b[3]! - b[0]!).toBeCloseTo(10 * Math.SQRT2, 6);
    expect(b[4]! - b[1]!).toBeCloseTo(10 * Math.SQRT2, 6);
    // Z não roda.
    expect(b[5]! - b[2]!).toBeCloseTo(2, 9);
  });

  it("contém todos os oito cantos rodados", () => {
    // A propriedade que importa: uma caixa que não contivesse um canto faria a
    // rejeição raio-caixa do picking descartar geometria em silêncio.
    const b = new Float64Array(6);
    g.sceneBox(100, 200, 0, 140, 260, 10, b);
    const o = out();
    for (const x of [100, 140])
      for (const y of [200, 260])
        for (const z of [0, 10]) {
          g.absToScene(x, y, z, o);
          expect(o.x).toBeGreaterThanOrEqual(b[0]! - 1e-9);
          expect(o.y).toBeGreaterThanOrEqual(b[1]! - 1e-9);
          expect(o.z).toBeGreaterThanOrEqual(b[2]! - 1e-9);
          expect(o.x).toBeLessThanOrEqual(b[3]! + 1e-9);
          expect(o.y).toBeLessThanOrEqual(b[4]! + 1e-9);
          expect(o.z).toBeLessThanOrEqual(b[5]! + 1e-9);
        }
  });
});

describe("PointCloudObject3D", () => {
  it("a matriz do three concorda com a frame", () => {
    // É a garantia que faz o desenho e a medição olharem para o mesmo sítio: o
    // caminho de compute projeta por `matrixWorld` e o picking converte pela
    // `CloudFrame`. Se divergissem, o ponto medido não seria o ponto visto.
    const o3d = new PointCloudObject3D(CLOUD, SCENE);
    o3d.setPlacement({
      yaw: 0.7,
      scale: 1.0001,
      pivot: [636500, 849200, 400],
      at: [500_000, 7_000_000, 100],
    });
    const f = o3d.frame();
    const esperado = out();
    for (const q of [
      [0, 0, 0],
      [123, -456, 78],
    ] as const) {
      f.localToScene(q[0], q[1], q[2], esperado);
      const v = new Vector3(q[0], q[1], q[2]).applyMatrix4(o3d.matrixWorld);
      expect(v.x).toBeCloseTo(esperado.x, 6);
      expect(v.y).toBeCloseTo(esperado.y, 6);
      expect(v.z).toBeCloseTo(esperado.z, 6);
    }
  });

  it("sem colocação a matriz continua a ser a translação de sempre", () => {
    const o3d = new PointCloudObject3D(CLOUD, SCENE);
    expect(o3d.position.x).toBeCloseTo(CLOUD[0] - SCENE[0], 9);
    expect(o3d.position.y).toBeCloseTo(CLOUD[1] - SCENE[1], 9);
    expect(o3d.position.z).toBeCloseTo(CLOUD[2] - SCENE[2], 9);
    expect(o3d.scale.x).toBe(1);
  });

  it("tirar a colocação devolve a nuvem ao sítio de origem", () => {
    const o3d = new PointCloudObject3D(CLOUD, SCENE);
    o3d.setPlacement({ yaw: 1, scale: 2, pivot: [0, 0, 0], at: [9, 9, 9] });
    o3d.setPlacement(undefined);
    expect(o3d.position.x).toBeCloseTo(CLOUD[0] - SCENE[0], 9);
    expect(o3d.scale.x).toBe(1);
    expect(o3d.frame().isTranslationOnly).toBe(true);
  });
});
