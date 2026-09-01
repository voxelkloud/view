import type { DecodedPointData } from "@voxelkloud/format-potree";

/**
 * The CPU-only half of {@link SplatSink} (see `sink-splat.ts`) — no `GPUDevice`,
 * no `GPUBuffer`, so it can be unit tested the way `BlockAllocator` and
 * `packNodeMeta` are in `sink-compute.ts`. Everything here is either building a
 * node's billboard geometry once at attach time, or resorting/regathering the
 * currently visible set once per significant camera move.
 */

export interface SplatNodeGeometry {
  readonly count: number;
  /** xyz per point, cloud-local — the SAME frame `ComputeSink.attach` writes
   * `data.positions` into unchanged, no re-centring or rescaling here. */
  readonly centers: Float32Array;
  /** The billboard's two widest axes, world-scaled, already converted from
   * the source splat's Y-up-visual convention to this pipeline's Z-up. */
  readonly axisU: Float32Array;
  readonly axisV: Float32Array;
  /** rgba8, alpha always 255 — opacity travels in `opacities`, not here,
   * matching `GaussianPocViewer.tsx`'s split (the material multiplies them). */
  readonly colors: Uint8Array;
  readonly opacities: Float32Array;
  readonly bytes: number;
}

const REQUIRED_ATTRIBUTES = [
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
  "opacity",
] as const;

/**
 * `data.attributesByName` must carry the Gaussian extras `packages/wasm-3dgs`
 * writes into COPC (see its `EXTRA_DIMENSION_NAMES`) — already activated:
 * scale linear (not log), opacity 0-1 (not pre-sigmoid), rotation normalized
 * w,x,y,z (not reordered). A cloud missing any of them is not a Gaussian
 * splat cloud, and attaching it would silently draw garbage axes.
 */
export function hasGaussianAttributes(data: DecodedPointData): boolean {
  return REQUIRED_ATTRIBUTES.every((name) => {
    const attr = data.attributesByName.get(name);
    return attr?.array instanceof Float32Array;
  });
}

/**
 * Rotation matrix, as its three column vectors — quaternion order is w,x,y,z,
 * matching `rot_0`=w, the same convention `crates/voxelkloud-3dgs/src/ply.rs`
 * and `splat.rs` both decode into. Un-normalized input is fine; normalized
 * here.
 */
export function rotationColumns(
  wRaw: number,
  xRaw: number,
  yRaw: number,
  zRaw: number,
): [[number, number, number], [number, number, number], [number, number, number]] {
  const len = Math.hypot(wRaw, xRaw, yRaw, zRaw) || 1;
  const w = wRaw / len;
  const x = xRaw / len;
  const y = yRaw / len;
  const z = zRaw / len;
  const m00 = 1 - 2 * (y * y + z * z);
  const m01 = 2 * (x * y - w * z);
  const m02 = 2 * (x * z + w * y);
  const m10 = 2 * (x * y + w * z);
  const m11 = 1 - 2 * (x * x + z * z);
  const m12 = 2 * (y * z - w * x);
  const m20 = 2 * (x * z - w * y);
  const m21 = 2 * (y * z + w * x);
  const m22 = 1 - 2 * (x * x + y * y);
  return [
    [m00, m10, m20],
    [m01, m11, m21],
    [m02, m12, m22],
  ];
}

/**
 * Rotation-derived vectors are still in the SOURCE splat's Y-up-visual
 * convention — only `position` was converted to Z-up upstream, by
 * `GaussianPoint::splat_visual_up_to_z_up` in the Rust crate, and only for
 * position. Every axis this file builds needs the same rotation applied by
 * hand, or the billboards would tilt against the points they sit on.
 */
export function sourceSplatVectorToZUp(
  v: readonly [number, number, number],
): [number, number, number] {
  return [v[0], v[2], -v[1]];
}

/**
 * One node's worth of billboard geometry, built once at `attach` time — the
 * expensive part (quaternion → matrix, axis picking, Z-up conversion) never
 * repeats per frame, only the sort does.
 */
export function buildSplatNodeGeometry(
  data: DecodedPointData,
  scaleFactor: number,
): SplatNodeGeometry {
  // Cloud-local float32 is the frame the whole pipeline agrees on — same
  // guard `ComputeSink.attach` makes before writing `data.positions`
  // straight to the GPU with no frame decode of its own.
  if (!(data.positions instanceof Float32Array)) {
    throw new Error("SplatSink expects float32 point positions.");
  }
  const n = data.numPoints;
  const centers = new Float32Array(n * 3);
  const axisU = new Float32Array(n * 3);
  const axisV = new Float32Array(n * 3);
  const colors = new Uint8Array(n * 4);
  const opacities = new Float32Array(n);

  const scale0 = requireFloatAttr(data, "scale_0");
  const scale1 = requireFloatAttr(data, "scale_1");
  const scale2 = requireFloatAttr(data, "scale_2");
  const rot0 = requireFloatAttr(data, "rot_0");
  const rot1 = requireFloatAttr(data, "rot_1");
  const rot2 = requireFloatAttr(data, "rot_2");
  const rot3 = requireFloatAttr(data, "rot_3");
  const opacity = requireFloatAttr(data, "opacity");
  const rgba = data.colors?.array instanceof Uint8Array ? data.colors.array : undefined;

  for (let i = 0; i < n; i++) {
    centers[i * 3] = data.positions[i * 3]!;
    centers[i * 3 + 1] = data.positions[i * 3 + 1]!;
    centers[i * 3 + 2] = data.positions[i * 3 + 2]!;

    const scales = [
      Math.max(0.0001, scale0[i]!),
      Math.max(0.0001, scale1[i]!),
      Math.max(0.0001, scale2[i]!),
    ] as const;
    const order = [0, 1, 2].sort((a, b) => scales[b]! - scales[a]!);
    const columns = rotationColumns(rot0[i]!, rot1[i]!, rot2[i]!, rot3[i]!);
    const u = sourceSplatVectorToZUp(columns[order[0]!]!);
    const v = sourceSplatVectorToZUp(columns[order[1]!]!);
    const su = scales[order[0]!]! * scaleFactor;
    const sv = scales[order[1]!]! * scaleFactor;
    axisU[i * 3] = u[0] * su;
    axisU[i * 3 + 1] = u[1] * su;
    axisU[i * 3 + 2] = u[2] * su;
    axisV[i * 3] = v[0] * sv;
    axisV[i * 3 + 1] = v[1] * sv;
    axisV[i * 3 + 2] = v[2] * sv;

    if (rgba !== undefined) {
      colors[i * 4] = rgba[i * 4]!;
      colors[i * 4 + 1] = rgba[i * 4 + 1]!;
      colors[i * 4 + 2] = rgba[i * 4 + 2]!;
    } else {
      colors[i * 4] = 200;
      colors[i * 4 + 1] = 200;
      colors[i * 4 + 2] = 200;
    }
    colors[i * 4 + 3] = 255;
    opacities[i] = clamp01(opacity[i]!);
  }

  const bytes =
    centers.byteLength + axisU.byteLength + axisV.byteLength + colors.byteLength + opacities.byteLength;
  return { count: n, centers, axisU, axisV, colors, opacities, bytes };
}

function requireFloatAttr(data: DecodedPointData, name: string): Float32Array {
  const attr = data.attributesByName.get(name);
  if (!(attr?.array instanceof Float32Array)) {
    throw new Error(`SplatSink expected a float32 "${name}" attribute.`);
  }
  return attr.array;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/** One sample point per visible block, for the depth-sort throttle below. */
export interface VisibleBlock {
  readonly geometry: SplatNodeGeometry;
}

export interface GatheredGeometry {
  readonly count: number;
  readonly centers: Float32Array;
  readonly axisU: Float32Array;
  readonly axisV: Float32Array;
  readonly colors: Uint8Array;
  readonly opacities: Float32Array;
}

/** Concatenates every visible block's arrays into one flat set, in whatever
 * order `blocks` arrives — the caller sorts afterward. */
export function gatherVisibleGeometry(blocks: readonly VisibleBlock[]): GatheredGeometry {
  let total = 0;
  for (const b of blocks) total += b.geometry.count;
  const centers = new Float32Array(total * 3);
  const axisU = new Float32Array(total * 3);
  const axisV = new Float32Array(total * 3);
  const colors = new Uint8Array(total * 4);
  const opacities = new Float32Array(total);
  let offset = 0;
  for (const b of blocks) {
    const g = b.geometry;
    centers.set(g.centers, offset * 3);
    axisU.set(g.axisU, offset * 3);
    axisV.set(g.axisV, offset * 3);
    colors.set(g.colors, offset * 4);
    opacities.set(g.opacities, offset);
    offset += g.count;
  }
  return { count: total, centers, axisU, axisV, colors, opacities };
}

/**
 * Back-to-front draw order — farthest first — the correct order for the
 * premultiplied "over" blend `sink-splat.ts`'s pipeline uses. Returns an
 * index permutation rather than reordering in place, so the caller can reuse
 * one scratch set of destination arrays across frames.
 */
export function sortBackToFront(
  centers: Float32Array,
  count: number,
  eye: readonly [number, number, number],
  forward: readonly [number, number, number],
): Uint32Array {
  const depths = new Float32Array(count);
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    depths[i] =
      (centers[i * 3]! - eye[0]) * forward[0] +
      (centers[i * 3 + 1]! - eye[1]) * forward[1] +
      (centers[i * 3 + 2]! - eye[2]) * forward[2];
    order[i] = i;
  }
  // `Uint32Array` has no `.sort(comparator)` with index semantics preserved
  // across engines the same way a plain array does here, so sort a plain
  // array view and copy back — count is per-capture (hundreds of thousands),
  // not LiDAR-survey scale, so this is not the hot path `BlockAllocator` is.
  const plain = Array.from(order);
  plain.sort((a, b) => depths[b]! - depths[a]!);
  return Uint32Array.from(plain);
}

/** Writes `src[order[i]]` into `dst[i]` for every typed-array field at once. */
export function applySortOrder(geometry: GatheredGeometry, order: Uint32Array): GatheredGeometry {
  const { count } = geometry;
  const centers = new Float32Array(count * 3);
  const axisU = new Float32Array(count * 3);
  const axisV = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 4);
  const opacities = new Float32Array(count);
  for (let dst = 0; dst < count; dst++) {
    const src = order[dst]!;
    centers[dst * 3] = geometry.centers[src * 3]!;
    centers[dst * 3 + 1] = geometry.centers[src * 3 + 1]!;
    centers[dst * 3 + 2] = geometry.centers[src * 3 + 2]!;
    axisU[dst * 3] = geometry.axisU[src * 3]!;
    axisU[dst * 3 + 1] = geometry.axisU[src * 3 + 1]!;
    axisU[dst * 3 + 2] = geometry.axisU[src * 3 + 2]!;
    axisV[dst * 3] = geometry.axisV[src * 3]!;
    axisV[dst * 3 + 1] = geometry.axisV[src * 3 + 1]!;
    axisV[dst * 3 + 2] = geometry.axisV[src * 3 + 2]!;
    colors[dst * 4] = geometry.colors[src * 4]!;
    colors[dst * 4 + 1] = geometry.colors[src * 4 + 1]!;
    colors[dst * 4 + 2] = geometry.colors[src * 4 + 2]!;
    colors[dst * 4 + 3] = geometry.colors[src * 4 + 3]!;
    opacities[dst] = geometry.opacities[src]!;
  }
  return { count, centers, axisU, axisV, colors, opacities };
}

/**
 * A resort/reupload is only worth its cost when the camera moved enough to
 * change the back-to-front order in a way anyone would see — same throttle
 * idea as `GaussianPocViewer.tsx`'s `cameraDepthSortKey`, but keyed off the
 * eye POSITION directly (this sink has no yaw/pitch/distance state of its
 * own; the view hands it a three.js camera).
 */
export function cameraPositionSignature(
  eye: readonly [number, number, number],
  step: number,
): string {
  return [
    Math.round(eye[0] * step),
    Math.round(eye[1] * step),
    Math.round(eye[2] * step),
  ].join(":");
}
