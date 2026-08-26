// The half of the point material that is JUST NUMBERS — colour modes, option
// defaults, the scalar-attribute question.
//
// Split out for ONE reason, and it is measurable: the other half builds a
// `NodeMaterial` out of TSL, both of which live in `three/webgpu` — three's
// WebGPU build, 357 kB gzipped against 115 kB for the core. `view.ts` needs
// these numbers on EVERY path (the compute rasteriser sizes its splats from
// exactly the same resolved options, so the two agree by construction), but it
// needs the material itself only on the instanced path. Kept in one file, the
// numbers dragged the WebGPU build into every bundle that ever asked what a
// colour mode was.

/**
 * How points are coloured.
 *
 * `intensity` and `classification` read a SCALAR instance attribute rather than
 * vertex colour, which is why they need Task 4's `scalarFormat: "gpu"` lane:
 * three's `_getVertexFormat` has no `Uint8Array` entry at itemSize 1, so a raw
 * uint8 attribute resolves to `undefined` straight into the pipeline descriptor
 * and the device rejects it. The `f32` lane sidesteps that and carries the RAW
 * value — normalisation happens in the shader from the attribute's declared
 * range, so a classification code stays an integer the palette can index.
 */
export type ColorMode =
  | { readonly kind: "rgb" }
  | { readonly kind: "flat"; readonly color: readonly [number, number, number] }
  | { readonly kind: "elevation" }
  | { readonly kind: "level" }
  /** Continuous ramp over the attribute's declared min/max. */
  | { readonly kind: "intensity" }
  /** Discrete palette, indexed by the ASPRS class code. */
  | { readonly kind: "classification" }
  /**
   * B5 — distância à malha de projeto, em metros.
   *
   * Percorre a MESMA via escalar que `intensity`, e de propósito: a rampa, a
   * faixa, a legenda e o relatório já a conhecem. A diferença é de onde vem o
   * número — nenhum arquivo o traz, o kernel de `deviation-wgsl` escreve-o —
   * e por isso `scalarAttributeFor` não pede atributo nenhum. Pedir um faria a
   * nuvem ser recusada por não ter um campo que ela não devia ter.
   */
  | { readonly kind: "deviation" };

/**
 * The per-point class, by verbatim source name.
 *
 * Named once because two places need the same string for different reasons:
 * the colour mode that paints by it, and the decode that fetches it whether or
 * not any mode paints by it — hiding a class has to work in every mode.
 */
export const CLASS_ATTRIBUTE = "classification";

/** The attribute a colour mode needs, or `undefined` for the built-in ones. */
export function scalarAttributeFor(mode: ColorMode): string | undefined {
  if (mode.kind === "intensity") return "intensity";
  if (mode.kind === "classification") return CLASS_ATTRIBUTE;
  return undefined;
}

export interface PointMaterialOptions {
  readonly colorMode?: ColorMode;
  /**
   * Scales the derived world diameter. 1.0 covers with no holes; Potree ships
   * the equivalent of 0.85, a deliberate 15% under-cover tuned by eye.
   */
  readonly sizeMultiplier?: number;
  /** Device pixels. Below ~1 a splat starts dropping out of rasterisation. */
  readonly minPixelSize?: number;
  /**
   * Device pixels. This is a FILL-RATE control disguised as a taste control: at
   * 3M points a 2 px mean splat is ~7.6x overdraw at 1080p, 3 px is 17x and
   * 5 px is 47x.
   */
  readonly maxPixelSize?: number;
  /** CRS units, for the elevation ramp. */
  readonly elevationRange?: readonly [number, number];
  /** Declared min/max of the scalar attribute, for the intensity ramp. */
  readonly scalarRange?: readonly [number, number];
}

export type ResolvedPointMaterialOptions = Required<PointMaterialOptions>;

export function resolvePointMaterialOptions(
  o: PointMaterialOptions = {},
): ResolvedPointMaterialOptions {
  return {
    colorMode: o.colorMode ?? { kind: "rgb" },
    sizeMultiplier: o.sizeMultiplier ?? 1,
    minPixelSize: o.minPixelSize ?? 1,
    maxPixelSize: o.maxPixelSize ?? 8,
    elevationRange: o.elevationRange ?? [0, 1],
    scalarRange: o.scalarRange ?? [0, 1],
  };
}
