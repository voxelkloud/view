// PURE. Zero imports: no three, no DOM, no GPU.
//
// This file and its siblings under lod/ are the seam a wasm kernel cuts along,
// so everything here is a plain function over numbers and typed arrays.

/**
 * Which clip-space depth convention `clipFromWorld` was built for.
 *
 * There is NO DEFAULT and there never will be. three constructs a `Camera` with
 * `WebGLCoordinateSystem` and then `Renderer.render` OVERWRITES it to the
 * backend's value at first render — so read it from `renderer.coordinateSystem`
 * AFTER `init()`, never from the camera. Using the WebGL near plane against a
 * WebGPU matrix does not fail loudly: it puts one plane in the wrong place, and
 * the bug presents as flicker near the camera.
 */
export type DepthRange = "zero-to-one" | "minus-one-to-one";

/**
 * - `Outside` — behind at least one plane. Cull, and prune the whole subtree.
 * - `Intersecting` — straddles. Children must be re-tested. Conservative: a box
 *   outside a frustum CORNER reports this. Correct and cheap.
 * - `Inside` — fully contained, so EVERY descendant is too. Propagates, which
 *   skips the six-plane test for an entire subtree.
 */
export const Containment = { Outside: 0, Intersecting: 1, Inside: 2 } as const;
export type Containment = (typeof Containment)[keyof typeof Containment];

/** 6 planes x [nx, ny, nz, d]. */
export const FRUSTUM_PLANE_FLOATS = 24;

/**
 * Extract six normalised world-space planes from a clip-from-world matrix.
 *
 * Ordered right, left, bottom, top, far, near — three's own `Frustum` order, so
 * the suite can diff coefficient-for-coefficient against
 * `Frustum.setFromProjectionMatrix`.
 *
 * Normalised so `n·p + d` is a SIGNED DISTANCE IN METRES. float64 throughout: on
 * autzen `d` is of order 6.4e5 and `n·p + d` is a difference of two such values,
 * which is ~1e-10 in float64 but ~0.05 m in float32. A wasm port must use f64
 * here, not f32.
 *
 * @param clipFromWorld column-major, 16 elements.
 * @param out 24 doubles, overwritten in place.
 */
export function extractFrustumPlanes(
  clipFromWorld: ArrayLike<number>,
  out: Float64Array,
  depthRange: DepthRange,
  reversedDepth = false,
): Float64Array {
  const m = clipFromWorld;
  const me0 = m[0]!;
  const me1 = m[1]!;
  const me2 = m[2]!;
  const me3 = m[3]!;
  const me4 = m[4]!;
  const me5 = m[5]!;
  const me6 = m[6]!;
  const me7 = m[7]!;
  const me8 = m[8]!;
  const me9 = m[9]!;
  const me10 = m[10]!;
  const me11 = m[11]!;
  const me12 = m[12]!;
  const me13 = m[13]!;
  const me14 = m[14]!;
  const me15 = m[15]!;

  set(out, 0, me3 - me0, me7 - me4, me11 - me8, me15 - me12); // right
  set(out, 1, me3 + me0, me7 + me4, me11 + me8, me15 + me12); // left
  set(out, 2, me3 + me1, me7 + me5, me11 + me9, me15 + me13); // bottom
  set(out, 3, me3 - me1, me7 - me5, me11 - me9, me15 - me13); // top

  // The only place the two conventions differ. Under WebGPU's 0..1 depth the
  // near plane is the raw third row; under WebGL's -1..1 it is w + z.
  if (reversedDepth) {
    set(out, 4, me2, me6, me10, me14); // far
    set(out, 5, me3 - me2, me7 - me6, me11 - me10, me15 - me14); // near
  } else if (depthRange === "zero-to-one") {
    set(out, 4, me3 - me2, me7 - me6, me11 - me10, me15 - me14); // far
    set(out, 5, me2, me6, me10, me14); // near
  } else {
    set(out, 4, me3 - me2, me7 - me6, me11 - me10, me15 - me14); // far
    set(out, 5, me3 + me2, me7 + me6, me11 + me10, me15 + me14); // near
  }

  return out;
}

function set(
  out: Float64Array,
  plane: number,
  nx: number,
  ny: number,
  nz: number,
  d: number,
): void {
  const len = Math.hypot(nx, ny, nz);
  const k = len === 0 ? 0 : 1 / len;
  const o = plane * 4;
  out[o] = nx * k;
  out[o + 1] = ny * k;
  out[o + 2] = nz * k;
  out[o + 3] = d * k;
}

/**
 * Classify an axis-aligned box against the frustum.
 *
 * Uses the p-vertex / n-vertex pair: the p-vertex is the box corner furthest
 * along the plane normal, the n-vertex the one furthest against it. If the
 * p-vertex is behind a plane the box is entirely outside; if any n-vertex is
 * behind, the box straddles.
 */
export function classifyAabb(
  planes: Float64Array,
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
): Containment {
  let intersecting = false;
  for (let p = 0; p < 6; p++) {
    const o = p * 4;
    const nx = planes[o]!;
    const ny = planes[o + 1]!;
    const nz = planes[o + 2]!;
    const d = planes[o + 3]!;

    const pX = nx > 0 ? maxX : minX;
    const pY = ny > 0 ? maxY : minY;
    const pZ = nz > 0 ? maxZ : minZ;
    if (nx * pX + ny * pY + nz * pZ + d < 0) return Containment.Outside;

    const nX = nx > 0 ? minX : maxX;
    const nY = ny > 0 ? minY : maxY;
    const nZ = nz > 0 ? minZ : maxZ;
    if (nx * nX + ny * nY + nz * nZ + d < 0) intersecting = true;
  }
  return intersecting ? Containment.Intersecting : Containment.Inside;
}

/**
 * Boolean intersection test.
 *
 * Deliberately the same algorithm as three's `Frustum.intersectsBox`, so the two
 * agree exactly and the suite can use three as an oracle.
 */
export function intersectsAabb(
  planes: Float64Array,
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
): boolean {
  for (let p = 0; p < 6; p++) {
    const o = p * 4;
    const nx = planes[o]!;
    const ny = planes[o + 1]!;
    const nz = planes[o + 2]!;
    const pX = nx > 0 ? maxX : minX;
    const pY = ny > 0 ? maxY : minY;
    const pZ = nz > 0 ? maxZ : minZ;
    if (nx * pX + ny * pY + nz * pZ + planes[o + 3]! < 0) return false;
  }
  return true;
}
