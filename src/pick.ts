import type { PerspectiveCamera } from "three/webgpu";
import { Ray, Vector3 } from "three/webgpu";
import { projectionFactorPerspective } from "./lod/metric.js";

export interface PickResult {
  /** CRS absoluto, float64. */
  readonly position: readonly [number, number, number];
  /** Scene-relative position, for overlays rendered with this view's camera. */
  readonly scenePosition?: readonly [number, number, number];
  readonly cloudIndex: number;
  readonly nodeIndex: number;
  readonly pointIndex: number;
  readonly screenDistancePx: number;
  readonly color?: readonly [number, number, number];
  readonly scalarValue?: number;
}

export interface PickPointOptions {
  readonly maxDistancePx?: number;
  readonly cloudIndex?: number;
}

interface PickNode {
  readonly index: number;
  readonly level: number;
  readonly minX: number;
  readonly minY: number;
  readonly minZ: number;
  readonly maxX: number;
  readonly maxY: number;
  readonly maxZ: number;
}

export interface PickReadback {
  readonly positions: Float32Array | Int32Array;
  readonly start: number;
  readonly count: number;
  readonly colors?: Uint8Array | Uint16Array;
  readonly scalars?: ArrayLike<number>;
}

interface PickCloudContext {
  readonly cloudIndex: number;
  readonly cloudOrigin: readonly [number, number, number];
  readonly sceneOrigin: readonly [number, number, number];
  readonly selection: Int32Array;
  readonly selectionCount: number;
  node(index: number): PickNode | undefined;
  readPoints(index: number): PickReadback | undefined;
}

const DEFAULT_MAX_DISTANCE_PX = 8;
const point = new Vector3();
const projected = new Vector3();

function sceneOffset(
  cloudOrigin: readonly [number, number, number],
  sceneOrigin: readonly [number, number, number],
): [number, number, number] {
  return [
    cloudOrigin[0] - sceneOrigin[0],
    cloudOrigin[1] - sceneOrigin[1],
    cloudOrigin[2] - sceneOrigin[2],
  ];
}

function intersectsExpandedBox(
  ray: Ray,
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
  margin: number,
): boolean {
  let tMin = 0;
  let tMax = Number.POSITIVE_INFINITY;

  const test = (origin: number, dir: number, lo: number, hi: number): boolean => {
    if (Math.abs(dir) < 1e-12) return origin >= lo && origin <= hi;
    const inv = 1 / dir;
    let t1 = (lo - origin) * inv;
    let t2 = (hi - origin) * inv;
    if (t1 > t2) [t1, t2] = [t2, t1];
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    return tMin <= tMax;
  };

  return (
    test(ray.origin.x, ray.direction.x, minX - margin, maxX + margin) &&
    test(ray.origin.y, ray.direction.y, minY - margin, maxY + margin) &&
    test(ray.origin.z, ray.direction.z, minZ - margin, maxZ + margin) &&
    tMax >= 0
  );
}

function screenDistancePx(
  camera: PerspectiveCamera,
  viewportWidthPx: number,
  viewportHeightPx: number,
  x: number,
  y: number,
  z: number,
  screenX: number,
  screenY: number,
): number {
  projected.set(x, y, z).project(camera);
  const px = (projected.x * 0.5 + 0.5) * viewportWidthPx;
  const py = (1 - (projected.y * 0.5 + 0.5)) * viewportHeightPx;
  return Math.hypot(px - screenX, py - screenY);
}

/**
 * Pick the closest point to a screen coordinate from the currently selected
 * nodes of one or more clouds.
 *
 * The helper is pure: it only reads camera matrices and the CPU mirror owned by
 * the sinks, so tests can exercise it without a GPU device or DOM.
 */
export function pickPoint(
  camera: PerspectiveCamera,
  viewportWidthPx: number,
  viewportHeightPx: number,
  screenX: number,
  screenY: number,
  clouds: readonly PickCloudContext[],
  options: PickPointOptions = {},
): PickResult | undefined {
  if (!(viewportWidthPx > 0) || !(viewportHeightPx > 0)) return undefined;

  const maxDistancePx = options.maxDistancePx ?? DEFAULT_MAX_DISTANCE_PX;
  const ndcX = (screenX / viewportWidthPx) * 2 - 1;
  const ndcY = -(screenY / viewportHeightPx) * 2 + 1;
  const origin = camera.position.clone();
  const target = new Vector3(ndcX, ndcY, 0.5).unproject(camera);
  const direction = target.sub(origin).normalize();
  const ray = new Ray(origin, direction);
  const slope = Math.tan(((camera.fov ?? 60) * Math.PI) / 360);

  let best: PickResult | undefined;
  let bestDepth = Number.POSITIVE_INFINITY;

  for (const cloud of clouds) {
    if (options.cloudIndex !== undefined && cloud.cloudIndex !== options.cloudIndex) {
      continue;
    }

    const offset = sceneOffset(cloud.cloudOrigin, cloud.sceneOrigin);

    for (let k = 0; k < cloud.selectionCount; k++) {
      const nodeIndex = cloud.selection[k]!;
      const node = cloud.node(nodeIndex);
      const read = cloud.readPoints(nodeIndex);
      if (node === undefined || read === undefined || read.count === 0) continue;

      const nodeCentreX = (node.minX + node.maxX) * 0.5 - cloud.sceneOrigin[0];
      const nodeCentreY = (node.minY + node.maxY) * 0.5 - cloud.sceneOrigin[1];
      const nodeCentreZ = (node.minZ + node.maxZ) * 0.5 - cloud.sceneOrigin[2];
      const depthToCentre = Math.max(
        Math.hypot(
          nodeCentreX - ray.origin.x,
          nodeCentreY - ray.origin.y,
          nodeCentreZ - ray.origin.z,
        ),
        1e-6,
      );
      const proj = projectionFactorPerspective(
        slope,
        viewportHeightPx,
        depthToCentre,
      );
      if (!(proj > 0)) continue;
      const worldRadius = maxDistancePx / proj;
      if (
        !intersectsExpandedBox(
          ray,
          node.minX - cloud.sceneOrigin[0],
          node.minY - cloud.sceneOrigin[1],
          node.minZ - cloud.sceneOrigin[2],
          node.maxX - cloud.sceneOrigin[0],
          node.maxY - cloud.sceneOrigin[1],
          node.maxZ - cloud.sceneOrigin[2],
          worldRadius,
        )
      ) {
        continue;
      }

      for (let i = 0; i < read.count; i++) {
        const j = read.start + i;
        const sx = read.positions[3 * j]! + offset[0];
        const sy = read.positions[3 * j + 1]! + offset[1];
        const sz = read.positions[3 * j + 2]! + offset[2];
        point.set(sx, sy, sz);
        const depth = (point.x - ray.origin.x) * ray.direction.x +
          (point.y - ray.origin.y) * ray.direction.y +
          (point.z - ray.origin.z) * ray.direction.z;
        if (depth < 0) continue;
        const dist = screenDistancePx(
          camera,
          viewportWidthPx,
          viewportHeightPx,
          sx,
          sy,
          sz,
          screenX,
          screenY,
        );
        if (dist > maxDistancePx) continue;

        if (best === undefined || dist < best.screenDistancePx - 1e-6) {
          const scene: [number, number, number] = [sx, sy, sz];
          const abs: [number, number, number] = [
            sx + cloud.sceneOrigin[0],
            sy + cloud.sceneOrigin[1],
            sz + cloud.sceneOrigin[2],
          ];
          const colors = read.colors;
          const color =
            colors === undefined
              ? undefined
              : ([colors[4 * j]!, colors[4 * j + 1]!, colors[4 * j + 2]!] as const);
          const scalar =
            read.scalars === undefined ? undefined : Number(read.scalars[j]!);
          best = {
            position: abs,
            ...(cloud.sceneOrigin[0] !== 0 ||
            cloud.sceneOrigin[1] !== 0 ||
            cloud.sceneOrigin[2] !== 0
              ? { scenePosition: scene }
              : {}),
            cloudIndex: cloud.cloudIndex,
            nodeIndex,
            pointIndex: i,
            screenDistancePx: dist,
            ...(color !== undefined ? { color } : {}),
            ...(scalar !== undefined ? { scalarValue: scalar } : {}),
          };
          bestDepth = depth;
          continue;
        }

        if (
          Math.abs(dist - best.screenDistancePx) <= 1e-6 &&
          depth < bestDepth
        ) {
          const scene: [number, number, number] = [sx, sy, sz];
          const abs: [number, number, number] = [
            sx + cloud.sceneOrigin[0],
            sy + cloud.sceneOrigin[1],
            sz + cloud.sceneOrigin[2],
          ];
          const colors = read.colors;
          const color =
            colors === undefined
              ? undefined
              : ([colors[4 * j]!, colors[4 * j + 1]!, colors[4 * j + 2]!] as const);
          const scalar =
            read.scalars === undefined ? undefined : Number(read.scalars[j]!);
          best = {
            position: abs,
            ...(cloud.sceneOrigin[0] !== 0 ||
            cloud.sceneOrigin[1] !== 0 ||
            cloud.sceneOrigin[2] !== 0
              ? { scenePosition: scene }
              : {}),
            cloudIndex: cloud.cloudIndex,
            nodeIndex,
            pointIndex: i,
            screenDistancePx: dist,
            ...(color !== undefined ? { color } : {}),
            ...(scalar !== undefined ? { scalarValue: scalar } : {}),
          };
          bestDepth = depth;
        }
      }
    }
  }

  return best;
}
