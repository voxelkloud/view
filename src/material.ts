import { DoubleSide, NodeMaterial, SRGBColorSpace } from "three/webgpu";
import {
  Discard,
  Fn,
  If,
  attribute,
  cameraProjectionMatrix,
  colorSpaceToWorking,
  float,
  int,
  mix,
  modelViewMatrix,
  positionGeometry,
  uniform,
  varyingProperty,
  vec3,
  vec4,
  viewportSize,
} from "three/tsl";

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
  | { readonly kind: "classification" };

/** The attribute a colour mode needs, or `undefined` for the built-in ones. */
export function scalarAttributeFor(mode: ColorMode): string | undefined {
  if (mode.kind === "intensity") return "intensity";
  if (mode.kind === "classification") return "classification";
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

/** A `NodeMaterial` with the uniform handles kept reachable for live updates. */
export interface PointCloudMaterial extends NodeMaterial {
  uSizeMultiplier: { value: number };
  uMinPixelSize: { value: number };
  uMaxPixelSize: { value: number };
  uElevMin: { value: number };
  uElevMax: { value: number };
  uMaxLevel: { value: number };
  uScalarMin: { value: number };
  uScalarMax: { value: number };
  colorMode: ColorMode;
}

/**
 * A five-stop analytic ramp, authored in sRGB like every other colour path so
 * the single `colorSpaceToWorking` in the fragment assembler covers it.
 *
 * Analytic rather than a `DataTexture`: fewer moving parts, and nothing to
 * sample from a vertex stage.
 */
const RAMP: ReadonlyArray<readonly [number, number, number]> = [
  [0.19, 0.07, 0.23],
  [0.21, 0.36, 0.55],
  [0.13, 0.57, 0.55],
  [0.48, 0.74, 0.32],
  [0.99, 0.91, 0.15],
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ramp(t: any): any {
  const u = t.clamp(0, 1).mul(RAMP.length - 1);
  // `any` because the accumulator changes node type on the first mix(): it
  // starts as a ConstNode and becomes a MathNode. TSL's types describe each
  // node precisely, which is exactly what makes an accumulator awkward.
  let c: any = vec3(...RAMP[0]!);
  for (let i = 1; i < RAMP.length; i++) {
    c = mix(c, vec3(...RAMP[i]!), u.sub(i - 1).clamp(0, 1));
  }
  return c;
}

/**
 * The ASPRS LAS 1.4 standard classes, 0 to 18.
 *
 * A palette rather than a ramp because the codes are NOMINAL, not ordinal:
 * ground (2) is not "less" than building (6), so interpolating between them
 * would produce a colour that means nothing.
 *
 * All nineteen are listed, not just the common ones, so {@link UNKNOWN_CLASS}
 * genuinely means "a code outside the standard" — a user-defined class, or a
 * misread byte.
 */
const CLASSES: ReadonlyArray<readonly [number, readonly [number, number, number]]> = [
  [0, [0.42, 0.42, 0.46]], // never classified
  [1, [0.62, 0.62, 0.66]], // unclassified
  [2, [0.55, 0.42, 0.28]], // ground
  [3, [0.35, 0.6, 0.3]], // low vegetation
  [4, [0.28, 0.68, 0.32]], // medium vegetation
  [5, [0.18, 0.5, 0.24]], // high vegetation
  [6, [0.85, 0.45, 0.35]], // building
  [7, [0.9, 0.2, 0.35]], // low point (noise)
  [8, [0.75, 0.7, 0.3]], // reserved (was model key-point)
  [9, [0.24, 0.5, 0.85]], // water
  [10, [0.5, 0.35, 0.55]], // rail
  [11, [0.34, 0.34, 0.4]], // road surface
  [12, [0.7, 0.7, 0.55]], // overlap
  [13, [0.95, 0.75, 0.25]], // wire, guard (shield)
  [14, [0.95, 0.6, 0.15]], // wire, conductor (phase)
  [15, [0.8, 0.5, 0.6]], // transmission tower
  [16, [0.95, 0.85, 0.4]], // wire-structure connector
  [17, [0.6, 0.55, 0.7]], // bridge deck
  [18, [1.0, 0.1, 0.55]], // high noise
];

/**
 * Anything outside 0..18.
 *
 * Deliberately a colour no standard class uses. An earlier version fell back to
 * a grey one shade off "unclassified", which on a survey that is 80% class 1 —
 * measured, on the az-usfs tiles — makes "I have no colour for this" and
 * "unclassified" the same picture.
 */
const UNKNOWN_CLASS: readonly [number, number, number] = [0.0, 0.85, 0.8];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function classificationColor(code: any): any {
  let c: any = vec3(...UNKNOWN_CLASS);
  for (const [value, rgb] of CLASSES) {
    // `mix(a, b, step)` with a 0/1 selector: TSL has no switch, and a chain of
    // If() blocks in the fragment stage would each need their own varying.
    c = mix(c, vec3(...rgb), code.sub(value).abs().lessThan(0.5).select(1, 0));
  }
  return c;
}

/**
 * The point material: a view-aligned quad per point, expanded in the vertex
 * stage.
 *
 * NOT `Points` + `PointsNodeMaterial`. three's WebGPU backend maps any
 * `object.isPoints` to `point-list` topology and WGSL has no point-size builtin,
 * so `sizeNode` is silently ignored there and every point rasterises as one
 * pixel with no attenuation. Instanced quads are the only path to a sized splat.
 */
export function createPointMaterial(
  options: PointMaterialOptions = {},
): PointCloudMaterial {
  const o = resolvePointMaterialOptions(options);
  const m = new NodeMaterial() as PointCloudMaterial;

  // Shared uniforms — writing `.value` is free and never triggers a recompile.
  const uSizeMultiplier = uniform(o.sizeMultiplier);
  const uMinPixelSize = uniform(o.minPixelSize);
  const uMaxPixelSize = uniform(o.maxPixelSize);
  const uElevMin = uniform(o.elevationRange[0]);
  const uElevMax = uniform(o.elevationRange[1]);
  const uMaxLevel = uniform(1);
  const uScalarMin = uniform(o.scalarRange[0]);
  const uScalarMax = uniform(o.scalarRange[1]);

  // Per-object uniforms. `onObjectUpdate` is the same mechanism three uses for
  // `highpModelViewMatrix`; these land in the OBJECT bind group, so they do not
  // touch the pipeline cache key and every slab still shares one pipeline. That
  // is what makes adaptive point size cost zero bytes per point.
  const uNodeSpacing = uniform(1).onObjectUpdate(
    ({ object }) =>
      (object?.userData['spacingWorld'] as number | undefined) ?? 1,
  );
  const uNodeLevel = uniform(0).onObjectUpdate(
    ({ object }) => (object?.userData['level'] as number | undefined) ?? 0,
  );

  m.vertexNode = Fn(() => {
    // ARENA LIVENESS. `color.w` is 255 for a live, currently-selected slot and 0
    // otherwise; multiplying the diameter by it collapses a dead instance to a
    // zero-area quad that rasterises nothing. voxelkloud OWNS this byte — v1
    // stamps it and warns when the source attribute really had four elements.
    // Never read it as opacity.
    const alive = attribute('color', 'vec4').w;

    // Task 4's float32 cloud-relative offsets, through the float64-computed
    // model-view. With `renderer.highPrecision = true` this resolves to
    // `highpModelViewMatrix`, a single uniform read with no shader-side multiply
    // of two large translations.
    const viewPos = modelViewMatrix
      .mul(vec4(attribute('pointOffset', 'vec3'), 1))
      .toVar('vkViewPos');

    const z = viewPos.z.negate().max(1e-6).toVar('vkZ');

    // Pixels per world unit — exactly the reference's projFactor.
    //   Potree: (0.5 * domHeight) / (tan(fov/2) * distance)
    //   here:   0.5 * H * P11 / z, where P11 === 1 / tan(fov/2)
    // NOT three's own attenuation, which uses `scale / -positionView.z` with
    // `scale = 0.5 * height` — the same expression WITHOUT the 1/slope term, so
    // at a 60-degree fov its points come out 1.73x smaller and the splat size
    // decalibrates against the LOD metric. `viewportSize.y` is the drawing
    // buffer height in physical pixels, matching the scheduler's device-pixel
    // viewport exactly.
    const projFactor = float(0.5)
      .mul(viewportSize.y)
      .mul(cameraProjectionMatrix.element(int(1)).y)
      .div(z)
      .toVar('vkPF');

    // A WORLD diameter, never a pixel size. A Potree level is a maximal
    // Poisson-disc sample with minimum distance s_L = spacing / 2**L;
    // maximality means every surface location is within s_L of a sample, so
    // discs of radius s_L — diameter 2*s_L — cover with no holes.
    const dWorld = uNodeSpacing.mul(2).mul(uSizeMultiplier).toVar('vkD0');

    // Clamp in PIXELS, then derive the world diameter back from the clamped
    // pixel size, so the quad and any radius varying always agree.
    const px = dWorld.mul(projFactor).toVar('vkPx');
    const pxC = px.clamp(uMinPixelSize, uMaxPixelSize).toVar('vkPxC');
    const d = dWorld.mul(pxC.div(px.max(1e-6))).mul(alive).toVar('vkD');

    varyingProperty('vec2', 'vkCorner').assign(positionGeometry.xy);

    // Expand in VIEW space: all four corners share a view z, so the splat is a
    // view-plane-parallel disc and the perspective foreshortening is exact.
    //
    // Built from explicit components rather than `vec4(vec2, float, float)`:
    // TSL counts a swizzled vec2 against vec4's arity and warns at graph-build
    // time, which would otherwise be a permanent line of console noise.
    const half = d.mul(0.5);
    return cameraProjectionMatrix.mul(
      vec4(
        viewPos.x.add(positionGeometry.x.mul(half)),
        viewPos.y.add(positionGeometry.y.mul(half)),
        viewPos.z,
        viewPos.w,
      ),
    );
  })();

  m.colorNode = Fn(() => {
    const corner = varyingProperty('vec2', 'vkCorner');
    // Round splats. NOT `pointUV`: `PointUVNode.generate` returns a literal GLSL
    // string referencing `gl_PointCoord`, which does not exist in WGSL.
    If(corner.dot(corner).greaterThan(1), () => {
      Discard();
    });

    // Every mode returns an sRGB-ENCODED vec3, converted exactly once below.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let srgb: any;
    switch (o.colorMode.kind) {
      case 'flat':
        srgb = vec3(...o.colorMode.color);
        break;
      case 'elevation':
        srgb = ramp(
          attribute('pointOffset', 'vec3')
            .z.sub(uElevMin)
            .div(uElevMax.sub(uElevMin).max(1e-9)),
        );
        break;
      case 'level':
        srgb = ramp(uNodeLevel.div(uMaxLevel.max(1)));
        break;
      case 'intensity':
        srgb = ramp(
          attribute('scalarValue', 'float')
            .sub(uScalarMin)
            .div(uScalarMax.sub(uScalarMin).max(1e-9)),
        );
        break;
      case 'classification':
        srgb = classificationColor(attribute('scalarValue', 'float'));
        break;
      default:
        srgb = attribute('color', 'vec4').xyz;
        break;
    }

    // ONE conversion, in one place. Potree colour bytes are display-referred
    // sRGB and three's working space is Linear-sRGB; without this the cloud
    // renders visibly washed out and every measurement screenshot misleads.
    const lin = colorSpaceToWorking(srgb, SRGBColorSpace);
    return vec4(lin.x, lin.y, lin.z, 1);
  })();

  m.transparent = false;
  m.depthWrite = true;
  // A point cloud is data, not a lit render.
  m.toneMapped = false;
  // A view-aligned quad is never backfacing, so this costs nothing and removes a
  // whole class of "why is my cloud invisible" winding bug.
  m.side = DoubleSide;

  Object.assign(m, {
    uScalarMin,
    uScalarMax,
    uSizeMultiplier,
    uMinPixelSize,
    uMaxPixelSize,
    uElevMin,
    uElevMax,
    uMaxLevel,
    colorMode: o.colorMode,
  });
  return m;
}
