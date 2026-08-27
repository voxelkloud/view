/**
 * Uma vista de cima da COR da nuvem, sobre um retângulo do chão, na CPU.
 *
 * Existe para alinhar uma foto aérea com o levantamento sem GPU nenhuma: a
 * correlação precisa das duas imagens na mesma grelha, e a nuvem só existe em
 * pontos. Isto rasteriza-os uma vez, e o resultado é pequeno o suficiente para
 * atravessar a fronteira de um worker sem custo.
 *
 * O PONTO MAIS ALTO ganha a célula, e não a média. Uma média mistura o chão
 * com a copa que está por cima dele e produz uma cor que não existe em sítio
 * nenhum do terreno — e a foto, tirada de cima, vê exatamente o que está por
 * cima. Guardar o mais alto é a mesma pergunta que a lente fez.
 */

export interface OrthoReadback {
  readonly positions: Float32Array | Int32Array;
  readonly start: number;
  readonly count: number;
  readonly colors?: Uint8Array | Uint16Array;
}

export interface OrthoCloudContext {
  readonly cloudIndex: number;
  readonly cloudOrigin: readonly [number, number, number];
  readonly sceneOrigin: readonly [number, number, number];
  readonly selection: Int32Array;
  readonly selectionCount: number;
  readPoints(index: number): OrthoReadback | undefined;
}

export interface OrthoSample {
  readonly width: number;
  readonly height: number;
  /** Luminância 0..255 por célula; 0 onde nada caiu. */
  readonly luma: Uint8Array;
  /** 1 onde caiu pelo menos um ponto. Distingue "preto" de "vazio". */
  readonly hits: Uint8Array;
  /** O canto inferior-esquerdo, em coordenadas de CENA. */
  readonly min: readonly [number, number];
  /** Metros por célula, nos dois eixos. */
  readonly scale: readonly [number, number];
}

/** Rec. 601, a mesma que a foto vai usar — as duas têm de concordar. */
const lumaOf = (r: number, g: number, b: number): number =>
  (r * 299 + g * 587 + b * 114) / 1000;

/**
 * Rasteriza a cor das nuvens sobre `[min, min + size]`, numa grelha de
 * `width x height`.
 *
 * `cloudIndex` a `undefined` usa todas: um projeto pode ter o levantamento
 * repartido em várias, e alinhar contra metade dele seria alinhar contra uma
 * borda que não é borda nenhuma.
 */
export function orthoSampleClouds(
  clouds: readonly OrthoCloudContext[],
  min: readonly [number, number],
  size: readonly [number, number],
  width: number,
  height: number,
  cloudIndex?: number,
): OrthoSample {
  const luma = new Uint8Array(width * height);
  const hits = new Uint8Array(width * height);
  // A altura de quem já ganhou cada célula. -Infinity é "ainda ninguém".
  const top = new Float32Array(width * height).fill(-Infinity);

  const sx = width / Math.max(size[0], 1e-9);
  const sy = height / Math.max(size[1], 1e-9);

  for (const cloud of clouds) {
    if (cloudIndex !== undefined && cloud.cloudIndex !== cloudIndex) continue;
    // Os buffers estão em local-da-nuvem; a cena é isso mais esta translação.
    const ox = cloud.cloudOrigin[0] - cloud.sceneOrigin[0];
    const oy = cloud.cloudOrigin[1] - cloud.sceneOrigin[1];
    const oz = cloud.cloudOrigin[2] - cloud.sceneOrigin[2];

    for (let k = 0; k < cloud.selectionCount; k++) {
      const read = cloud.readPoints(cloud.selection[k]!);
      if (read === undefined || read.count === 0) continue;
      const pos = read.positions;
      const col = read.colors;
      // Sem cor não há o que correlacionar, e uma nuvem sem RGB não serve
      // para isto — dizê-lo com células vazias é melhor que inventar cinzento.
      if (col === undefined) continue;
      const per = col.length / Math.max(read.count, 1) >= 4 ? 4 : 3;
      const norm = col instanceof Uint16Array ? 1 / 257 : 1;

      for (let i = 0; i < read.count; i++) {
        const p = (read.start + i) * 3;
        const gx = ((pos[p]! + ox - min[0]) * sx) | 0;
        if (gx < 0 || gx >= width) continue;
        const gy = ((pos[p + 1]! + oy - min[1]) * sy) | 0;
        if (gy < 0 || gy >= height) continue;
        const z = pos[p + 2]! + oz;
        const cell = gy * width + gx;
        if (z <= top[cell]!) continue;
        const c = (read.start + i) * per;
        top[cell] = z;
        luma[cell] = lumaOf(col[c]! * norm, col[c + 1]! * norm, col[c + 2]! * norm) | 0;
        hits[cell] = 1;
      }
    }
  }

  return {
    width,
    height,
    luma,
    hits,
    min: [min[0], min[1]],
    scale: [size[0] / width, size[1] / height],
  };
}
