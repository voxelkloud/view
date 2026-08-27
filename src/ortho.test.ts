import { describe, expect, it } from "vitest";
import { orthoSampleClouds, type OrthoCloudContext } from "./ortho.js";

/**
 * A vista de cima que alimenta o alinhamento automático.
 *
 * Nada aqui lança quando erra. Um eixo trocado alinha a foto noventa graus
 * fora e o resultado parece um erro de pose; uma média em vez do ponto mais
 * alto produz uma cor que não existe no terreno, e a correlação encontra o
 * mínimo no sítio errado com toda a confiança.
 */

/** Uma nuvem de teste com pontos em local-da-nuvem e cor RGB por ponto. */
const cloudOf = (
  points: readonly (readonly [number, number, number])[],
  colors: readonly (readonly [number, number, number])[],
  origin: readonly [number, number, number] = [0, 0, 0],
): OrthoCloudContext => {
  const pos = new Float32Array(points.flatMap((p) => [p[0], p[1], p[2]]));
  const col = new Uint8Array(colors.flatMap((c) => [c[0], c[1], c[2]]));
  return {
    cloudIndex: 0,
    cloudOrigin: origin,
    sceneOrigin: [0, 0, 0],
    selection: Int32Array.from([0]),
    selectionCount: 1,
    readPoints: () => ({ positions: pos, start: 0, count: points.length, colors: col }),
  };
};

const BRANCO = [255, 255, 255] as const;
const PRETO = [0, 0, 0] as const;

describe("orthoSampleClouds", () => {
  it("põe cada ponto na célula da sua posição", () => {
    // Grelha 2x2 sobre [0,0]..[2,2]: um ponto branco no canto inferior-esquerdo.
    const s = orthoSampleClouds([cloudOf([[0.5, 0.5, 0]], [BRANCO])], [0, 0], [2, 2], 2, 2);
    expect(s.hits[0]).toBe(1);
    expect(s.luma[0]).toBe(255);
    // As outras três ficam vazias, e vazio NÃO é preto.
    expect([...s.hits].filter((h) => h === 1)).toHaveLength(1);
  });

  it("o eixo Y cresce para cima na grelha, como no mundo", () => {
    const s = orthoSampleClouds(
      [cloudOf([[0.5, 1.5, 0]], [BRANCO])],
      [0, 0],
      [2, 2],
      2,
      2,
    );
    // Célula (x=0, y=1) = índice 2 numa grelha de 2 de largura.
    expect(s.hits[2]).toBe(1);
    expect(s.hits[0]).toBe(0);
  });

  it("o ponto MAIS ALTO ganha a célula, e não a média", () => {
    // Chão preto com uma copa branca por cima, na mesma célula. Uma média daria
    // cinzento — uma cor que não existe em sítio nenhum do terreno, e que a
    // lente nunca viu.
    const s = orthoSampleClouds(
      [
        cloudOf(
          [
            [0.5, 0.5, 0],
            [0.5, 0.5, 10],
          ],
          [PRETO, BRANCO],
        ),
      ],
      [0, 0],
      [1, 1],
      1,
      1,
    );
    expect(s.luma[0]).toBe(255);
  });

  it("a ordem de chegada não muda o resultado", () => {
    // O mesmo par, invertido: quem chega primeiro não pode ganhar por isso.
    const s = orthoSampleClouds(
      [
        cloudOf(
          [
            [0.5, 0.5, 10],
            [0.5, 0.5, 0],
          ],
          [BRANCO, PRETO],
        ),
      ],
      [0, 0],
      [1, 1],
      1,
      1,
    );
    expect(s.luma[0]).toBe(255);
  });

  it("o que cai fora do retângulo é descartado, não dobrado para dentro", () => {
    const s = orthoSampleClouds(
      [
        cloudOf(
          [
            [-5, 0.5, 0],
            [50, 0.5, 0],
            [0.5, -5, 0],
          ],
          [BRANCO, BRANCO, BRANCO],
        ),
      ],
      [0, 0],
      [2, 2],
      2,
      2,
    );
    expect([...s.hits].filter((h) => h === 1)).toHaveLength(0);
  });

  it("a nuvem é lida no frame da CENA, não no dela", () => {
    // Mesmos números no buffer, origem 100 m a leste: cai 100 m a leste.
    const s = orthoSampleClouds(
      [cloudOf([[0.5, 0.5, 0]], [BRANCO], [100, 0, 0])],
      [100, 0],
      [2, 2],
      2,
      2,
    );
    expect(s.hits[0]).toBe(1);
  });

  it("uma nuvem sem cor não inventa cinzento", () => {
    const pos = new Float32Array([0.5, 0.5, 0]);
    const semCor: OrthoCloudContext = {
      cloudIndex: 0,
      cloudOrigin: [0, 0, 0],
      sceneOrigin: [0, 0, 0],
      selection: Int32Array.from([0]),
      selectionCount: 1,
      readPoints: () => ({ positions: pos, start: 0, count: 1 }),
    };
    const s = orthoSampleClouds([semCor], [0, 0], [1, 1], 1, 1);
    expect(s.hits[0]).toBe(0);
  });

  it("`cloudIndex` escolhe uma nuvem e ignora as outras", () => {
    const a = { ...cloudOf([[0.5, 0.5, 0]], [BRANCO]), cloudIndex: 0 };
    const b = { ...cloudOf([[1.5, 0.5, 0]], [BRANCO]), cloudIndex: 1 };
    const so = orthoSampleClouds([a, b], [0, 0], [2, 2], 2, 2, 1);
    expect(so.hits[0]).toBe(0);
    expect(so.hits[1]).toBe(1);
  });

  it("a escala diz quantos metros vale uma célula", () => {
    const s = orthoSampleClouds([], [10, 20], [50, 100], 25, 25);
    expect(s.scale).toEqual([2, 4]);
    expect(s.min).toEqual([10, 20]);
  });
});
