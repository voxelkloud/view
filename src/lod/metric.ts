// PURE. The screen-space error metric and the depth-range policy.

/**
 * World-to-pixel scale at a given view distance, for a perspective camera.
 *
 * `(0.5 * viewportHeightPx) / (tan(fov/2) * distance)` — a world span of `s` at
 * distance `d` covers `s * projFactor` device pixels vertically.
 *
 * `viewportHeightPx` must be DEVICE pixels (`cssHeight * dpr`). The reference
 * uses CSS pixels, which silently halves every threshold on a 2x display, and
 * the material reads three's `viewportSize.y`, which is physical — so scheduler
 * and shader must agree.
 */
export function projectionFactorPerspective(
  slope: number,
  viewportHeightPx: number,
  distance: number,
): number {
  return (0.5 * viewportHeightPx) / (slope * distance);
}

/** For an orthographic camera the factor is constant in distance. */
export function projectionFactorOrthographic(
  viewportHeightPx: number,
  orthoHeightWorld: number,
): number {
  return viewportHeightPx / orthoHeightWorld;
}

/**
 * Projected geometric error, in device pixels — THE control quantity.
 *
 * On a point octree the geometric error IS the inter-point spacing, which is
 * what the calibration below is about. On a tile format the same product is a
 * projected tile error; same units, different meaning, different threshold.
 *
 * Proportional to the reference's `screenPixelRadius` by the per-cloud constant
 * `k = metadata.spacing / rootHalfDiagonal`, so it induces an identical heap
 * ordering while being scale-free. Measured: autzen k = 9.0211e-3 (and the same
 * for any stock converter output, since the converter sets
 * `spacing = boxSize / 128`, giving `k = 2 / (128 * sqrt(3))`), synthetic
 * k = 1.4434e-1. So the reference's `minimumNodePixelSize: 150` means 1.35 px on
 * real data and 21.65 px on the synthetic fixture — a 16x swing in effective LOD
 * from an unchanged knob.
 */
export function screenErrorPx(
  geometricErrorAtLevel: number,
  projFactor: number,
): number {
  return geometricErrorAtLevel * projFactor;
}

/** @deprecated Renamed to {@link screenErrorPx}. */
export const screenSpacingPx = screenErrorPx;

/** Projected bounding-sphere radius in device pixels. Diagnostics and HUD. */
export function screenPixelRadius(
  boundingRadiusAtLevel: number,
  projFactor: number,
): number {
  return boundingRadiusAtLevel * projFactor;
}

export interface NearFarOptions {
  /** Multiple of the deepest admitted spacing used as the near plane. */
  readonly nearSpacingMultiple?: number;
  readonly minNear?: number;
  readonly maxNear?: number;
  readonly farHeadroom?: number;
  readonly minFarSpan?: number;
}

export interface NearFar {
  readonly near: number;
  readonly far: number;
}

/**
 * Derive near/far from the deepest ADMITTED POINT PITCH and the view depth.
 *
 * The argument is a point pitch and never a geometric error. The whole rule is
 * a comparison against the 24-bit depth quantum in units of point pitch, so a
 * tile error handed to it would silently misplace the near plane.
 *
 * With a fixed near of 0.1 the 24-bit depth quantum exceeds the LOD-implied
 * point pitch beyond roughly 1 km, so it fails across the far half of autzen
 * (headroom 1.0x at 1 km, 0.2x at 4.6 km). With this rule at level 6
 * (spacing 0.568 m, near 5.68 m) headroom is 54x at 1 km and 10x at 4.6 km, so a
 * plain depth24plus buffer is sufficient and neither reversed-Z nor logarithmic
 * depth is needed.
 *
 * Computed from ADMITTED nodes only. The reference minimises over popped-but-
 * CULLED nodes and writes the result into `camera.near` at the END of the frame,
 * so this frame's traversal sets next frame's frustum — a genuine oscillating
 * feedback loop on a stationary camera. The caller applies this at the START of a
 * frame instead, so the culling frustum and the render frustum always match.
 */
export function suggestNearFar(
  minPointSpacingWorld: number,
  viewDepth: number,
  options: NearFarOptions = {},
): NearFar {
  const multiple = options.nearSpacingMultiple ?? 10;
  const minNear = options.minNear ?? 0.01;
  const maxNear = options.maxNear ?? 100;
  const headroom = options.farHeadroom ?? 1.5;
  const minSpan = options.minFarSpan ?? 10_000;

  const near = Math.min(
    Math.max(minPointSpacingWorld * multiple, minNear),
    maxNear,
  );
  const far = Math.max(headroom * viewDepth, near + minSpan);
  return { near, far };
}

/**
 * Rough fragment count for the HUD, so fill-rate cost is visible rather than
 * discovered.
 *
 * A 2 px mean splat over 3M points is ~15.7M fragments — about 7.6x overdraw at
 * 1080p; 3 px is 17x and 5 px is 47x. The reference's `minSize` default of 2.0
 * already puts a viewer at ~8x before anyone touches a setting.
 */
export function estimateFragments(
  pointCount: number,
  meanPixelDiameter: number,
): number {
  const r = meanPixelDiameter * 0.5;
  return Math.round(pointCount * Math.PI * r * r);
}
