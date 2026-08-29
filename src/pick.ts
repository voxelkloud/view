import type { PerspectiveCamera } from "three";
import { Ray, Vector3 } from "three";
import { projectionFactorPerspective } from "./lod/metric.js";
import type { CloudFrame } from "./object.js";

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
  /**
   * Onde esta nuvem está na cena — a translação de sempre, ou a semelhança que
   * a traz de outro sistema de coordenadas. Substituiu o par
   * `cloudOrigin`/`sceneOrigin` que estava aqui: as três conversões que este
   * arquivo fazia à mão só valiam enquanto a relação fosse uma translação, e
   * nada no código dizia isso.
   */
  readonly frame: CloudFrame;
  readonly selection: Int32Array;
  readonly selectionCount: number;
  node(index: number): PickNode | undefined;
  readPoints(index: number): PickReadback | undefined;
}

const DEFAULT_MAX_DISTANCE_PX = 8;
const point = new Vector3();
const projected = new Vector3();
/** Destinos reutilizados: este laço corre por milhões de pontos. */
const scenePt = { x: 0, y: 0, z: 0 };
const absPt = { x: 0, y: 0, z: 0 };
const centrePt = { x: 0, y: 0, z: 0 };
const nodeBox = new Float64Array(6);

/**
 * O resultado de um acerto, montado num sítio só.
 *
 * Vivia duplicado — uma cópia no ramo "é o melhor até agora" e outra no
 * desempate por profundidade — e as duas construíam as mesmas seis coisas a
 * partir dos mesmos índices. Duas cópias de uma conversão de coordenadas são
 * dois sítios para a corrigir e um para esquecer.
 */
function hit(
  cloud: PickCloudContext,
  read: PickReadback,
  j: number,
  i: number,
  nodeIndex: number,
  sx: number,
  sy: number,
  sz: number,
  dist: number,
): PickResult {
  cloud.frame.sceneToAbs(sx, sy, sz, absPt);
  const colors = read.colors;
  const color =
    colors === undefined
      ? undefined
      : ([colors[4 * j]!, colors[4 * j + 1]!, colors[4 * j + 2]!] as const);
  const scalar = read.scalars === undefined ? undefined : Number(read.scalars[j]!);
  return {
    position: [absPt.x, absPt.y, absPt.z],
    // Só quando as duas diferem. Numa nuvem que define a origem da cena e não
    // tem colocação elas são o MESMO número, e devolver as duas faria quem lê
    // supor que há ali uma distinção a fazer.
    ...(absPt.x !== sx || absPt.y !== sy || absPt.z !== sz
      ? { scenePosition: [sx, sy, sz] as [number, number, number] }
      : {}),
    cloudIndex: cloud.cloudIndex,
    nodeIndex,
    pointIndex: i,
    screenDistancePx: dist,
    ...(color !== undefined ? { color } : {}),
    ...(scalar !== undefined ? { scalarValue: scalar } : {}),
  };
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

    const frame = cloud.frame;

    for (let k = 0; k < cloud.selectionCount; k++) {
      const nodeIndex = cloud.selection[k]!;
      const node = cloud.node(nodeIndex);
      const read = cloud.readPoints(nodeIndex);
      if (node === undefined || read === undefined || read.count === 0) continue;

      frame.absToScene(
        (node.minX + node.maxX) * 0.5,
        (node.minY + node.maxY) * 0.5,
        (node.minZ + node.maxZ) * 0.5,
        centrePt,
      );
      const depthToCentre = Math.max(
        Math.hypot(
          centrePt.x - ray.origin.x,
          centrePt.y - ray.origin.y,
          centrePt.z - ray.origin.z,
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
      // A caixa do nó JÁ NA CENA. Sob rotação ela deixa de estar alinhada aos
      // eixos, e `sceneBox` devolve o limite alinhado que a contém — maior,
      // nunca menor, porque encolher aqui descartaria geometria em silêncio.
      frame.sceneBox(node.minX, node.minY, node.minZ, node.maxX, node.maxY, node.maxZ, nodeBox);
      if (
        !intersectsExpandedBox(
          ray,
          nodeBox[0]!,
          nodeBox[1]!,
          nodeBox[2]!,
          nodeBox[3]!,
          nodeBox[4]!,
          nodeBox[5]!,
          worldRadius,
        )
      ) {
        continue;
      }

      for (let i = 0; i < read.count; i++) {
        const j = read.start + i;
        frame.localToScene(
          read.positions[3 * j]!,
          read.positions[3 * j + 1]!,
          read.positions[3 * j + 2]!,
          scenePt,
        );
        const sx = scenePt.x;
        const sy = scenePt.y;
        const sz = scenePt.z;
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
          best = hit(cloud, read, j, i, nodeIndex, sx, sy, sz, dist);
          bestDepth = depth;
          continue;
        }

        if (
          Math.abs(dist - best.screenDistancePx) <= 1e-6 &&
          depth < bestDepth
        ) {
          best = hit(cloud, read, j, i, nodeIndex, sx, sy, sz, dist);
          bestDepth = depth;
        }
      }
    }
  }

  return best;
}
