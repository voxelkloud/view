import {
  DataTexture,
  DoubleSide,
  NearestFilter,
  NodeMaterial,
  RGBAFormat,
  SRGBColorSpace,
  UnsignedByteType,
} from "three/webgpu";
import {
  Break,
  Discard,
  Fn,
  If,
  Loop,
  attribute,
  bitAnd,
  cameraProjectionMatrix,
  colorSpaceToWorking,
  exp2,
  float,
  int,
  ivec2,
  mix,
  modelViewMatrix,
  positionGeometry,
  shiftLeft,
  shiftRight,
  step,
  texture,
  uint,
  uniform,
  varyingProperty,
  vec3,
  vec4,
  viewportSize,
} from "three/tsl";
import { CUT_WIDTH, CUT_WIDTH_SHIFT, MAX_CUT_DEPTH } from "./cut.js";

// Re-exported, so `./material.js` stays the one import path it always was.
export type {
  ColorMode,
  PointMaterialOptions,
  ResolvedPointMaterialOptions,
} from "./material-options.js";
export { scalarAttributeFor, resolvePointMaterialOptions } from "./material-options.js";
import type {
  ColorMode,
  PointMaterialOptions,
  ResolvedPointMaterialOptions,
} from "./material-options.js";
import { resolvePointMaterialOptions } from "./material-options.js";

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
  /**
   * The octree-cut texture the vertex walk reads. Assign `.value` per frame;
   * see {@link OctreeCut}. Until something does, it holds a 1x1 all-zero texel
   * whose empty child mask stops the walk at depth 0 — which, through the
   * `max(depth, level)` clamp, is exactly per-node sizing.
   */
  uCutMap: { value: DataTexture };
  /**
   * Bitmask of HIDDEN classes for the classification mode: bit `n` set hides
   * class `n` for 0..18, and bit 31 hides everything outside the standard
   * range. Zero — the default — hides nothing. Meaningless in other modes,
   * where no class code streams at all.
   */
  uClassHidden: { value: number };
  /** CLOUD-LOCAL min corner of the root box — the frame `pointOffset` is in. */
  uRootMin: { value: { x: number; y: number; z: number } };
  /** CLOUD-LOCAL extent of the root box, per axis. */
  uRootSize: { value: { x: number; y: number; z: number } };
  colorMode: ColorMode;
}

/**
 * A cut that terminates immediately.
 *
 * Not an error path: a cloud drawn before its first selection, a
 * `PerNodeSink`, and any driver that never builds a cut all read this, get
 * depth 0, and fall through the clamp to the node's own level.
 */
function emptyCut(): DataTexture {
  const t = new DataTexture(new Uint8Array(4), 1, 1, RGBAFormat, UnsignedByteType);
  t.minFilter = NearestFilter;
  t.magFilter = NearestFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
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
  const uClassHidden = uniform(0, "uint");

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

  // The octree cut, and the box the walk descends. Cloud-local, because
  // `pointOffset` is.
  const uCutMap = texture(emptyCut());
  const uRootMin = uniform(vec3(0, 0, 0));
  const uRootSize = uniform(vec3(1, 1, 1));

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

    // LOCAL DEPTH. Descend the selected cut to the deepest node that contains
    // this point, which is what decides how wide its splat may be. Sizing by
    // the point's OWN node instead is the bug this exists to fix: every level
    // above the frontier then draws 2**(D-L) too wide and paints over finer
    // data that is already resident and already correct.
    //
    // From the ROOT rather than from the drawn node, which is the one place
    // this diverges from the reference. The arena batches slabs by LEVEL, so
    // there is no per-node uniform to carry a start offset and no per-point
    // lane to spend four bytes on; the price is that the walk runs D steps
    // instead of D - L.
    const p = attribute('pointOffset', 'vec3');
    const bMin = uRootMin.toVar('vkBMin');
    const bSize = uRootSize.toVar('vkBSize');
    const slot = int(0).toVar('vkSlot');
    const depth = int(0).toVar('vkDepth');

    Loop(MAX_CUT_DEPTH, () => {
      // Linear slot to texel. CUT_WIDTH is a power of two precisely so this is
      // a shift and a mask rather than an integer divide per step.
      const texel = uCutMap
        .load(
          ivec2(
            bitAnd(slot, int(CUT_WIDTH - 1)),
            shiftRight(slot, int(CUT_WIDTH_SHIFT)),
          ),
        )
        .toVar('vkTexel');
      const mask = int(texel.r.mul(255).round()).toVar('vkMask');

      const half = bSize.mul(0.5).toVar('vkHalf');
      // `step(edge, x)` is `x >= edge`, matching the half-open split
      // `makeChildNode` builds with `lo += size/2`.
      const c = step(bMin.add(half), p).toVar('vkC');
      // (x << 2) | (y << 1) | z — the format's own octant numbering.
      const idx = int(c.x.mul(4).add(c.y.mul(2)).add(c.z)).toVar('vkIdx');

      // Not selected, so its data is not on the GPU and this is as deep as the
      // picture goes here.
      If(bitAnd(mask, shiftLeft(int(1), idx)).equal(int(0)), () => {
        Break();
      });

      const first = int(texel.g.mul(255).round())
        .mul(65536)
        .add(int(texel.b.mul(255).round()).mul(256))
        .add(int(texel.a.mul(255).round()));

      // Siblings are contiguous in breadth-first order, so the child sits at
      // the run start plus the number of selected octants sorting before it.
      // Unrolled at graph-build time: eight bits, no loop, no divergence.
      const below = int(0).toVar('vkBelow');
      for (let b = 0; b < 7; b++) {
        below.addAssign(
          bitAnd(shiftRight(mask, int(b)), int(1)).mul(
            idx.greaterThan(int(b)).select(int(1), int(0)),
          ),
        );
      }

      slot.assign(first.add(below));
      bMin.addAssign(c.mul(half));
      bSize.assign(half);
      depth.addAssign(int(1));
    });

    // CLAMPED AT THE NODE'S OWN LEVEL, never below it. The walk cannot
    // legitimately land shallower — a drawn point's own node is selected and so
    // is every ancestor — so a shallower answer only ever comes from float32
    // ambiguity within about half a millimetre of a split plane. Clamping makes
    // that case degrade to the size drawn before this feature existed rather
    // than to a blob, and it is also what makes the empty-cut placeholder
    // behave exactly like per-node sizing.
    // MEASURED CAP OF ONE LEVEL, and the number is not taste.
    //
    // The cut says "the deepest resident node here is at depth D", and shrinking
    // a coarse point by the full 2**(D-L) assumes the levels in between deliver
    // enough points to refill the area it gave up. They do not always, and the
    // gap is visible: with the full shrink, 5.4% of the screen went from painted
    // at 13 s to background at 25 s — COVERAGE FALLING as more data arrived,
    // which is the opposite of what streaming should do. Potree loses 1.2% over
    // the same window, because it never shrinks: it sizes each point by its own
    // node's spacing.
    //
    // Capping the shrink at one level takes that to 0.7%, better than Potree,
    // and it is what took Speed Index from 5083 to 1918 (Potree: 1594) with
    // visual completeness arriving at 21 s instead of 25 s.
    //
    // Two alternatives were measured and are worse. A bigger splat with the full
    // shrink closes the holes only at 1.6x, and blurs: 13.1 of surface detail
    // against 14.1 here. A cap of two levels keeps more detail (15.9) but leaves
    // 3.6% falling and only 49% of the frame hole-free, against 59% here.
    //
    // Applied in BOTH rasterisers, and it has to be: they share this cut, so a
    // cap in one and not the other would make them size splats differently and
    // any A/B between them would be comparing two pictures, not two pipelines.
    //
    // Clamped as a FLOAT, and that is not cosmetic. TSL emits the literals of
    // `.max()`/`.min()` as floats whatever the node type, so an integer
    // `vkShrink` produced `max(int, 0.0)` — which WGSL accepts and GLSL ES does
    // not, since it has no such overload. The WebGL 2 fallback failed to
    // compile its vertex shader and rendered nothing at all, silently, because
    // nobody runs it: WebGPU is available on every machine here. `exp2` below
    // wants a float anyway, so this also drops a cast.
    const shrink = float(depth.sub(int(uNodeLevel)))
      .max(0)
      .min(1)
      .toVar('vkShrink');

    // A WORLD diameter, never a pixel size. A Potree level is a maximal
    // Poisson-disc sample with minimum distance s_L = spacing / 2**L;
    // maximality means every surface location is within s_L of a sample, so
    // discs of radius s_L — diameter 2*s_L — cover with no holes. Here s is the
    // LOCAL pitch: the node's own, halved once per level the cut goes deeper.
    //
    // Scaling `uNodeSpacing` rather than recomputing from a root spacing keeps
    // any per-node or per-format pitch override in play — only the local shrink
    // is applied here.
    const dWorld = uNodeSpacing
      .div(exp2(shrink))
      .mul(2)
      .mul(uSizeMultiplier)
      .toVar('vkD0');

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
      case 'classification': {
        const code = attribute('scalarValue', 'float');
        // Class toggles: one bit per standard code, bit 31 for anything
        // outside 0..18 — same bucket the UNKNOWN_CLASS colour paints. A
        // discard, not a zero-size quad: the class only exists in the
        // fragment stage, where the scalar attribute lands.
        const bit = code.add(0.5).toInt().clamp(0, 31);
        const bitU = bit.lessThanEqual(int(18)).select(bit, int(31)).toUint();
        If(
          bitAnd(shiftRight(uClassHidden, bitU), uint(1)).equal(uint(1)),
          () => {
            Discard();
          },
        );
        srgb = classificationColor(code);
        break;
      }
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
    uCutMap,
    uRootMin,
    uRootSize,
    uScalarMin,
    uScalarMax,
    uClassHidden,
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
