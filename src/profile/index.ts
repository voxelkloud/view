// FORMAT-SPECIFIC BY CONSTRUCTION, and left that way deliberately.
//
// A profile reads node PAYLOADS and reports node NAMES, and both are the
// driver's vocabulary — `byteSize === 0` is how Potree says a node carries no
// points of its own, and `r047` is how it names one. Widening the traversal to
// the neutral tree while the reporting stays Potree would buy nothing and cost
// a cast at every step. It generalises when the node-payload contract exists,
// which is Task B2.
import type { DecodedAttribute, DecodedPointData, NodeDecompress, OpenPointsOptions, PointCloudNode, PointCloudSourceBase, PointCloudTreeBase, PointDataOptions, PointReader, PointReaderFactory, ReadPointsOptions } from "@voxelkloud/core";

import type { PointReadback } from "../sink.js";

export type ProfileVec2 = readonly [number, number];
export type ProfileVec3 = readonly [number, number, number];

export interface VerticalProfileQuery {
  readonly kind: "vertical";
  /** Absolute CRS XY polyline. `width` is the full corridor width. */
  readonly points: readonly ProfileVec2[];
  readonly width: number;
  readonly zRange?: readonly [number, number];
}

export interface HorizontalProfileQuery {
  readonly kind: "horizontal";
  readonly z: number;
  /** Full accepted Z thickness around `z`. */
  readonly thickness: number;
  /** Absolute CRS XY polygon. Omitted means the horizontal slice has no XY cut. */
  readonly footprint?: readonly ProfileVec2[];
}

export type ProfileQuery = VerticalProfileQuery | HorizontalProfileQuery;

export type ProfileLimitedBy =
  | "complete"
  | "maxDepth"
  | "maxNodes"
  | "maxPoints"
  | "error";

export type ProfileBatchSource = "resident" | "loaded";

export type ProfileAttributeValue = number | readonly number[];

export interface ProfilePoint {
  /** Absolute CRS coordinates. */
  readonly position: ProfileVec3;
  /** Generic 2D profile coordinate: mileage/Z for vertical, X/Y for horizontal. */
  readonly u: number;
  readonly v: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly mileage?: number;
  readonly elevation?: number;
  readonly lateralOffset?: number;
  readonly color?: readonly [number, number, number];
  readonly scalarValue?: number;
  readonly attributes?: Readonly<Record<string, ProfileAttributeValue>>;
}

export interface ProfileNodeError {
  readonly nodeIndex: number;
  readonly nodeName: string;
  readonly message: string;
}

export interface ProfileStats {
  readonly visitedNodes: number;
  readonly intersectingNodes: number;
  readonly scannedPoints: number;
  readonly acceptedPoints: number;
  readonly residentNodes: number;
  readonly loadedNodes: number;
  readonly failedNodes: number;
  readonly elapsedMs: number;
  readonly limitedBy: ProfileLimitedBy;
}

export interface ProfilePointBatch {
  readonly kind: "points";
  readonly done: false;
  readonly source: ProfileBatchSource;
  readonly nodeIndex: number;
  readonly nodeName: string;
  readonly nodeLevel: number;
  readonly points: readonly ProfilePoint[];
  readonly stats: ProfileStats;
  readonly errors: readonly ProfileNodeError[];
}

export interface ProfileSummaryBatch {
  readonly kind: "summary";
  readonly done: true;
  readonly points: readonly [];
  readonly stats: ProfileStats;
  readonly errors: readonly ProfileNodeError[];
}

export type ProfileBatch = ProfilePointBatch | ProfileSummaryBatch;

export interface ProfileCloudContext {
  readonly source: PointCloudSourceBase;
  readonly hierarchy: PointCloudTreeBase;
  /**
   * The driver's reader factory. Called once per extraction with the attribute
   * selection the query needs, so a profile that wants only position does not
   * pay for colour on every node it touches.
   */
  readonly openPoints: PointReaderFactory;
  readonly readPoints?: (nodeIndex: number) => PointReadback | undefined;
}

export interface ProfileExtractionOptions {
  /** Accepted profile points before traversal stops. Default 1,000,000. */
  readonly maxPoints?: number;
  /**
   * Inclusive octree depth limit. Unlimited by default.
   *
   * Was the Potree manifest's declared depth, which no neutral source carries —
   * and the traversal already stops at a childless node, so the limit is a
   * caller's budget rather than a safety net.
   */
  readonly maxDepth?: number;
  /** Materialised/intersecting nodes visited before traversal stops. */
  readonly maxNodes?: number;
  readonly signal?: AbortSignal;
  readonly decompress?: NodeDecompress;
  /**
   * Attribute selection for nodes that are not already resident. Defaults to
   * position plus colour, the same default a reader takes on its own.
   */
  readonly pointData?: PointDataOptions;
  /** A reader to use instead of opening one. Takes precedence over `pointData`. */
  readonly reader?: PointReader;
  /**
   * Prefer `PointSink.readPoints()` when it has enough data for the request.
   * Enabled by default.
   */
  readonly preferResident?: boolean;
}

interface MutableProfileStats {
  visitedNodes: number;
  intersectingNodes: number;
  scannedPoints: number;
  acceptedPoints: number;
  residentNodes: number;
  loadedNodes: number;
  failedNodes: number;
  elapsedMs: number;
  limitedBy: ProfileLimitedBy;
}

interface QueueEntry {
  readonly node: PointCloudNode;
  readonly distance2: number;
  readonly sequence: number;
}

interface VerticalGeometry {
  readonly kind: "vertical";
  readonly points: readonly ProfileVec2[];
  readonly halfWidth: number;
  readonly zMin: number;
  readonly zMax: number;
  readonly segmentLengths: readonly number[];
  readonly cumulative: readonly number[];
}

interface HorizontalGeometry {
  readonly kind: "horizontal";
  readonly zMin: number;
  readonly zMax: number;
  readonly footprint?: readonly ProfileVec2[];
  readonly footprintBounds?: Rect;
}

type ProfileGeometry = VerticalGeometry | HorizontalGeometry;

interface Rect {
  readonly minX: number;
  readonly minY: number;
  readonly maxX: number;
  readonly maxY: number;
}

interface AcceptedPointGeometry {
  readonly u: number;
  readonly v: number;
  readonly mileage?: number;
  readonly elevation?: number;
  readonly lateralOffset?: number;
}

const DEFAULT_MAX_POINTS = 1_000_000;
const LIMIT_PRIORITY: Record<ProfileLimitedBy, number> = {
  complete: 0,
  maxDepth: 1,
  maxNodes: 2,
  maxPoints: 3,
  error: 4,
};

export async function* extractProfile(
  context: ProfileCloudContext,
  query: ProfileQuery,
  options: ProfileExtractionOptions = {},
): AsyncGenerator<ProfileBatch> {
  const geometry = createProfileGeometry(query);
  const maxPoints = positiveLimit(
    options.maxPoints ?? DEFAULT_MAX_POINTS,
    "maxPoints",
  );
  const maxNodes = positiveLimit(
    options.maxNodes ?? Number.POSITIVE_INFINITY,
    "maxNodes",
  );
  const maxDepth = depthLimit(options.maxDepth ?? Number.POSITIVE_INFINITY);
  const pointData = options.pointData ?? {};
  const ownsReader = options.reader === undefined;
  const reader =
    options.reader ??
    context.openPoints({
      ...(pointData as OpenPointsOptions),
      computeBounds: false,
      ...(options.decompress !== undefined
        ? { decompress: options.decompress }
        : {}),
    });
  const preferResident =
    (options.preferResident ?? true) &&
    canUseResidentReadback(context.source, pointData);
  const includeResidentColor = shouldEmitResidentColor(context.source, pointData);
  // The decompressor is a reader-wide concern, not a per-node one, so it was
  // handed to `openPoints` above rather than repeated on every read.
  const loadOptions: ReadPointsOptions = {
    signal: options.signal,
    computeBounds: false,
  };

  const started = nowMs();
  const stats: MutableProfileStats = {
    visitedNodes: 0,
    intersectingNodes: 0,
    scannedPoints: 0,
    acceptedPoints: 0,
    residentNodes: 0,
    loadedNodes: 0,
    failedNodes: 0,
    elapsedMs: 0,
    limitedBy: "complete",
  };
  const errors: ProfileNodeError[] = [];
  const queue = new ProfileNodeQueue();
  let sequence = 0;

  if (nodeIntersectsProfile(context.hierarchy.root, geometry)) {
    queue.push({
      node: context.hierarchy.root,
      distance2: nodeProfileDistance2(context.hierarchy.root, geometry),
      sequence: sequence++,
    });
  }

  while (queue.size > 0) {
    throwIfAborted(options.signal);
    if (stats.visitedNodes >= maxNodes) {
      setLimitedBy(stats, "maxNodes");
      break;
    }
    if (stats.acceptedPoints >= maxPoints) {
      setLimitedBy(stats, "maxPoints");
      break;
    }

    const entry = queue.pop()!;
    const node = entry.node;
    stats.visitedNodes++;
    if (!nodeIntersectsProfile(node, geometry)) continue;
    stats.intersectingNodes++;

    const readback =
      preferResident ? context.readPoints?.(node.index) : undefined;
    if (readback !== undefined) {
      const points = collectReadbackPoints(
        context.source,
        node,
        readback,
        geometry,
        maxPoints - stats.acceptedPoints,
        includeResidentColor,
        stats,
      );
      stats.residentNodes++;
      if (points.length > 0) {
        stats.elapsedMs = nowMs() - started;
        yield {
          kind: "points",
          done: false,
          source: "resident",
          nodeIndex: node.index,
          nodeName: node.name,
          nodeLevel: node.level,
          points,
          stats: snapshotStats(stats),
          errors: errors.slice(),
        };
      }
      if (stats.acceptedPoints >= maxPoints) {
        setLimitedBy(stats, "maxPoints");
        break;
      }
    } else if (node.numPoints > 0) {
      const data = await loadProfileNode(
        context,
        reader,
        node,
        loadOptions,
        stats,
        errors,
      );
      if (data !== undefined) {
        const points = collectDecodedPoints(
          data,
          geometry,
          maxPoints - stats.acceptedPoints,
          stats,
        );
        stats.loadedNodes++;
        if (points.length > 0) {
          stats.elapsedMs = nowMs() - started;
          yield {
            kind: "points",
            done: false,
            source: "loaded",
            nodeIndex: node.index,
            nodeName: node.name,
            nodeLevel: node.level,
            points,
            stats: snapshotStats(stats),
            errors: errors.slice(),
          };
        }
        if (stats.acceptedPoints >= maxPoints) {
          setLimitedBy(stats, "maxPoints");
          break;
        }
      }
    }

    if (node.level >= maxDepth) {
      if (node.childMask !== 0) setLimitedBy(stats, "maxDepth");
      continue;
    }

    const expanded = await expandProfileNode(context, node, options.signal, errors);
    if (!expanded) {
      stats.failedNodes++;
      if (node.childMask === undefined) setLimitedBy(stats, "error");
      continue;
    }

    for (const child of node.children) {
      if (child === undefined) continue;
      if (!nodeIntersectsProfile(child, geometry)) continue;
      queue.push({
        node: child,
        distance2: nodeProfileDistance2(child, geometry),
        sequence: sequence++,
      });
    }
  }

  stats.elapsedMs = nowMs() - started;
  yield {
    kind: "summary",
    done: true,
    points: [],
    stats: snapshotStats(stats),
    errors,
  };
}

function createProfileGeometry(query: ProfileQuery): ProfileGeometry {
  if (query.kind === "vertical") return createVerticalGeometry(query);
  return createHorizontalGeometry(query);
}

function createVerticalGeometry(query: VerticalProfileQuery): VerticalGeometry {
  if (query.points.length < 2) {
    throw new RangeError("A vertical profile needs at least two XY points.");
  }
  const width = finitePositive(query.width, "width");
  const zRange = query.zRange ?? [
    Number.NEGATIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  ];
  const zMin = Math.min(zRange[0], zRange[1]);
  const zMax = Math.max(zRange[0], zRange[1]);
  const segmentLengths: number[] = [];
  const cumulative: number[] = [0];
  let total = 0;
  for (let i = 0; i < query.points.length - 1; i++) {
    const a = query.points[i]!;
    const b = query.points[i + 1]!;
    assertFinitePoint2(a, `points[${i}]`);
    if (i === query.points.length - 2) assertFinitePoint2(b, `points[${i + 1}]`);
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    segmentLengths.push(length);
    total += length;
    cumulative.push(total);
  }
  if (!(total > 0)) {
    throw new RangeError("A vertical profile polyline must have non-zero length.");
  }
  return {
    kind: "vertical",
    points: query.points,
    halfWidth: width / 2,
    zMin,
    zMax,
    segmentLengths,
    cumulative,
  };
}

function createHorizontalGeometry(
  query: HorizontalProfileQuery,
): HorizontalGeometry {
  finite(query.z, "z");
  const thickness = finitePositive(query.thickness, "thickness");
  const half = thickness / 2;
  if (query.footprint !== undefined && query.footprint.length < 3) {
    throw new RangeError("A horizontal profile footprint needs at least 3 points.");
  }
  query.footprint?.forEach((p, i) => assertFinitePoint2(p, `footprint[${i}]`));
  return {
    kind: "horizontal",
    zMin: query.z - half,
    zMax: query.z + half,
    ...(query.footprint !== undefined
      ? {
          footprint: query.footprint,
          footprintBounds: polygonBounds(query.footprint),
        }
      : {}),
  };
}

async function loadProfileNode(
  context: ProfileCloudContext,
  reader: PointReader,
  node: PointCloudNode,
  loadOptions: ReadPointsOptions,
  stats: MutableProfileStats,
  errors: ProfileNodeError[],
): Promise<DecodedPointData | undefined> {
  const expanded = await expandProfileNode(
    context,
    node,
    loadOptions.signal,
    errors,
  );
  const hasPayload = reader.hasPayload(node);
  // A node that neither expanded nor carries points is a dead end the caller
  // should know about; one that simply has no payload of its own is normal.
  if (!expanded && !hasPayload) {
    setLimitedBy(stats, "error");
    return undefined;
  }
  if (!hasPayload) return undefined;
  try {
    return await reader.read(node, loadOptions);
  } catch (error) {
    stats.failedNodes++;
    setLimitedBy(stats, "error");
    errors.push({
      nodeIndex: node.index,
      nodeName: node.name,
      message: error instanceof Error ? error.message : String(error),
    });
    return undefined;
  }
}

/**
 * Make a node's children known, or report why not.
 *
 * `childMask !== undefined` is the neutral spelling of "expanded": a driver's
 * own `state` field is not on the tree contract, and the mask is what the
 * scheduler keys off too.
 */
async function expandProfileNode(
  context: ProfileCloudContext,
  node: PointCloudNode,
  signal: AbortSignal | undefined,
  errors: ProfileNodeError[],
): Promise<boolean> {
  if (node.childMask !== undefined) return true;
  if (context.hierarchy.tryExpandSync(node)) return true;
  try {
    await context.hierarchy.expand(node, signal === undefined ? {} : { signal });
    return true;
  } catch (error) {
    errors.push({
      nodeIndex: node.index,
      nodeName: node.name,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

function collectReadbackPoints(
  source: PointCloudSourceBase,
  node: PointCloudNode,
  readback: PointReadback,
  geometry: ProfileGeometry,
  remaining: number,
  includeColor: boolean,
  stats: MutableProfileStats,
): ProfilePoint[] {
  const points: ProfilePoint[] = [];
  const origin = source.bounds.min;
  const limit = Math.min(readback.count, remaining);

  for (let i = 0; i < readback.count && points.length < limit; i++) {
    stats.scannedPoints++;
    const j = readback.start + i;
    const x = origin[0] + Number(readback.positions[3 * j]!);
    const y = origin[1] + Number(readback.positions[3 * j + 1]!);
    const z = origin[2] + Number(readback.positions[3 * j + 2]!);
    const accepted = acceptPoint(geometry, x, y, z);
    if (accepted === undefined) continue;
    const color = includeColor ? colorAt(readback.colors, j) : undefined;
    const scalar =
      readback.scalars === undefined ? undefined : Number(readback.scalars[j]!);
    points.push({
      position: [x, y, z],
      x,
      y,
      z,
      ...accepted,
      ...(color !== undefined ? { color } : {}),
      ...(scalar !== undefined ? { scalarValue: scalar } : {}),
    });
    stats.acceptedPoints++;
  }

  if (points.length >= remaining) setLimitedBy(stats, "maxPoints");
  return points;
}

function collectDecodedPoints(
  data: DecodedPointData,
  geometry: ProfileGeometry,
  remaining: number,
  stats: MutableProfileStats,
): ProfilePoint[] {
  const points: ProfilePoint[] = [];
  const frame = data.frame;
  const limit = Math.min(data.numPoints, remaining);

  for (let i = 0; i < data.numPoints && points.length < limit; i++) {
    stats.scannedPoints++;
    const x = frame.origin[0] + Number(data.positions[3 * i]!) * frame.scale[0];
    const y = frame.origin[1] + Number(data.positions[3 * i + 1]!) * frame.scale[1];
    const z = frame.origin[2] + Number(data.positions[3 * i + 2]!) * frame.scale[2];
    const accepted = acceptPoint(geometry, x, y, z);
    if (accepted === undefined) continue;
    const color = colorAt(data.colors?.array, i, data.colors?.maxValue);
    const attributes = attributesAt(data.attributes, i);
    points.push({
      position: [x, y, z],
      x,
      y,
      z,
      ...accepted,
      ...(color !== undefined ? { color } : {}),
      ...(attributes !== undefined ? { attributes } : {}),
    });
    stats.acceptedPoints++;
  }

  if (points.length >= remaining) setLimitedBy(stats, "maxPoints");
  return points;
}

function acceptPoint(
  geometry: ProfileGeometry,
  x: number,
  y: number,
  z: number,
): AcceptedPointGeometry | undefined {
  if (geometry.kind === "vertical") {
    if (z < geometry.zMin || z > geometry.zMax) return undefined;
    const closest = closestPolylinePoint(geometry, x, y);
    if (closest.distance2 > geometry.halfWidth * geometry.halfWidth) {
      return undefined;
    }
    return {
      u: closest.mileage,
      v: z,
      mileage: closest.mileage,
      elevation: z,
      lateralOffset: closest.lateralOffset,
    };
  }

  if (z < geometry.zMin || z > geometry.zMax) return undefined;
  if (geometry.footprint !== undefined && !pointInPolygon([x, y], geometry.footprint)) {
    return undefined;
  }
  return { u: x, v: y };
}

function attributesAt(
  attributes: readonly DecodedAttribute[],
  pointIndex: number,
): Readonly<Record<string, ProfileAttributeValue>> | undefined {
  if (attributes.length === 0) return undefined;
  const out: Record<string, ProfileAttributeValue> = {};
  for (const attribute of attributes) {
    const base = pointIndex * attribute.itemSize;
    if (attribute.itemSize === 1) {
      let value = Number(attribute.array[base]!);
      if (attribute.inverse !== undefined) {
        value = value * attribute.inverse.scale + attribute.inverse.offset;
      }
      out[attribute.name] = value;
      continue;
    }
    const values: number[] = [];
    for (let k = 0; k < attribute.itemSize; k++) {
      values.push(Number(attribute.array[base + k]!));
    }
    out[attribute.name] = values;
  }
  return out;
}

function colorAt(
  colors: Uint8Array | Uint16Array | undefined,
  pointIndex: number,
  maxValue?: number,
): readonly [number, number, number] | undefined {
  if (colors === undefined) return undefined;
  const scale = (maxValue ?? (colors instanceof Uint16Array ? 65535 : 255)) / 255;
  return [
    Math.max(0, Math.min(255, Math.round(Number(colors[4 * pointIndex]!) / scale))),
    Math.max(
      0,
      Math.min(255, Math.round(Number(colors[4 * pointIndex + 1]!) / scale)),
    ),
    Math.max(
      0,
      Math.min(255, Math.round(Number(colors[4 * pointIndex + 2]!) / scale)),
    ),
  ];
}

function nodeIntersectsProfile(
  node: PointCloudNode,
  geometry: ProfileGeometry,
): boolean {
  if (node.maxZ < geometry.zMin || node.minZ > geometry.zMax) return false;
  if (geometry.kind === "vertical") {
    const rect = {
      minX: node.minX - geometry.halfWidth,
      minY: node.minY - geometry.halfWidth,
      maxX: node.maxX + geometry.halfWidth,
      maxY: node.maxY + geometry.halfWidth,
    };
    for (let i = 0; i < geometry.points.length - 1; i++) {
      const a = geometry.points[i]!;
      const b = geometry.points[i + 1]!;
      if (segmentIntersectsRect(a, b, rect)) return true;
    }
    return false;
  }

  if (geometry.footprint === undefined) return true;
  const rect = {
    minX: node.minX,
    minY: node.minY,
    maxX: node.maxX,
    maxY: node.maxY,
  };
  if (!rectsOverlap(rect, geometry.footprintBounds!)) return false;
  return polygonIntersectsRect(geometry.footprint, rect);
}

function nodeProfileDistance2(
  node: PointCloudNode,
  geometry: ProfileGeometry,
): number {
  if (geometry.kind === "horizontal") {
    const dz =
      node.maxZ < geometry.zMin
        ? geometry.zMin - node.maxZ
        : node.minZ > geometry.zMax
          ? node.minZ - geometry.zMax
          : 0;
    return dz * dz;
  }

  const cx = (node.minX + node.maxX) * 0.5;
  const cy = (node.minY + node.maxY) * 0.5;
  return closestPolylinePoint(geometry, cx, cy).distance2;
}

function closestPolylinePoint(
  geometry: VerticalGeometry,
  x: number,
  y: number,
): { distance2: number; mileage: number; lateralOffset: number } {
  let bestDistance2 = Number.POSITIVE_INFINITY;
  let bestMileage = 0;
  let bestOffset = 0;
  for (let i = 0; i < geometry.points.length - 1; i++) {
    const a = geometry.points[i]!;
    const b = geometry.points[i + 1]!;
    const length = geometry.segmentLengths[i]!;
    if (!(length > 0)) continue;
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const t = Math.max(
      0,
      Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / (length * length)),
    );
    const px = a[0] + dx * t;
    const py = a[1] + dy * t;
    const ox = x - px;
    const oy = y - py;
    const distance2 = ox * ox + oy * oy;
    if (distance2 >= bestDistance2) continue;
    bestDistance2 = distance2;
    bestMileage = geometry.cumulative[i]! + length * t;
    bestOffset = (dx * (y - a[1]) - dy * (x - a[0])) / length;
  }
  return { distance2: bestDistance2, mileage: bestMileage, lateralOffset: bestOffset };
}

function polygonIntersectsRect(points: readonly ProfileVec2[], rect: Rect): boolean {
  for (const p of points) {
    if (pointInRect(p, rect)) return true;
  }
  const corners: ProfileVec2[] = [
    [rect.minX, rect.minY],
    [rect.maxX, rect.minY],
    [rect.maxX, rect.maxY],
    [rect.minX, rect.maxY],
  ];
  for (const corner of corners) {
    if (pointInPolygon(corner, points)) return true;
  }
  for (let i = 0; i < points.length; i++) {
    const a = points[i]!;
    const b = points[(i + 1) % points.length]!;
    if (segmentIntersectsRect(a, b, rect)) return true;
  }
  return false;
}

function polygonBounds(points: readonly ProfileVec2[]): Rect {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  return { minX, minY, maxX, maxY };
}

function pointInPolygon(
  point: ProfileVec2,
  polygon: readonly ProfileVec2[],
): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i]!;
    const b = polygon[j]!;
    if (pointOnSegment(point, a, b)) return true;
    const crosses =
      a[1] > point[1] !== b[1] > point[1] &&
      point[0] <
        ((b[0] - a[0]) * (point[1] - a[1])) / (b[1] - a[1]) + a[0];
    if (crosses) inside = !inside;
  }
  return inside;
}

function segmentIntersectsRect(a: ProfileVec2, b: ProfileVec2, rect: Rect): boolean {
  if (pointInRect(a, rect) || pointInRect(b, rect)) return true;
  const p0: ProfileVec2 = [rect.minX, rect.minY];
  const p1: ProfileVec2 = [rect.maxX, rect.minY];
  const p2: ProfileVec2 = [rect.maxX, rect.maxY];
  const p3: ProfileVec2 = [rect.minX, rect.maxY];
  return (
    segmentsIntersect(a, b, p0, p1) ||
    segmentsIntersect(a, b, p1, p2) ||
    segmentsIntersect(a, b, p2, p3) ||
    segmentsIntersect(a, b, p3, p0)
  );
}

function segmentsIntersect(
  a: ProfileVec2,
  b: ProfileVec2,
  c: ProfileVec2,
  d: ProfileVec2,
): boolean {
  const o1 = orient(a, b, c);
  const o2 = orient(a, b, d);
  const o3 = orient(c, d, a);
  const o4 = orient(c, d, b);
  if (o1 === 0 && pointOnSegment(c, a, b)) return true;
  if (o2 === 0 && pointOnSegment(d, a, b)) return true;
  if (o3 === 0 && pointOnSegment(a, c, d)) return true;
  if (o4 === 0 && pointOnSegment(b, c, d)) return true;
  return (o1 > 0) !== (o2 > 0) && (o3 > 0) !== (o4 > 0);
}

function orient(a: ProfileVec2, b: ProfileVec2, c: ProfileVec2): number {
  const v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  return Math.abs(v) < 1e-12 ? 0 : v;
}

function pointOnSegment(
  point: ProfileVec2,
  a: ProfileVec2,
  b: ProfileVec2,
): boolean {
  const cross =
    (point[0] - a[0]) * (b[1] - a[1]) -
    (point[1] - a[1]) * (b[0] - a[0]);
  if (Math.abs(cross) > 1e-9) return false;
  return (
    point[0] >= Math.min(a[0], b[0]) - 1e-9 &&
    point[0] <= Math.max(a[0], b[0]) + 1e-9 &&
    point[1] >= Math.min(a[1], b[1]) - 1e-9 &&
    point[1] <= Math.max(a[1], b[1]) + 1e-9
  );
}

function pointInRect(point: ProfileVec2, rect: Rect): boolean {
  return (
    point[0] >= rect.minX &&
    point[0] <= rect.maxX &&
    point[1] >= rect.minY &&
    point[1] <= rect.maxY
  );
}

function rectsOverlap(a: Rect, b: Rect): boolean {
  return a.maxX >= b.minX && a.minX <= b.maxX && a.maxY >= b.minY && a.minY <= b.maxY;
}

function canUseResidentReadback(
  source: PointCloudSourceBase,
  pointData: PointDataOptions,
): boolean {
  if (pointData.positionFormat !== undefined && pointData.positionFormat !== "float32") {
    return false;
  }
  if (pointData.origin !== undefined && pointData.origin !== "cloud") return false;
  if (pointData.colorFormat === "native") return false;

  const attributes = pointData.attributes;
  if (attributes === undefined || Array.isArray(attributes) && attributes.length === 0) {
    return true;
  }
  if (attributes === "all") return false;
  const color = source.attributes.find((a) => a.role === "color")?.name;
  return attributes.every((name) => name === color);
}

function shouldEmitResidentColor(
  source: PointCloudSourceBase,
  pointData: PointDataOptions,
): boolean {
  const color = source.attributes.find((a) => a.role === "color")?.name;
  if (color === undefined) return false;
  const attributes = pointData.attributes;
  if (attributes === undefined) return true;
  if (attributes === "all") return true;
  return attributes.includes(color);
}

function positiveLimit(value: number, name: string): number {
  if (value === Number.POSITIVE_INFINITY) return value;
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive number.`);
  }
  return Math.floor(value);
}

function depthLimit(value: number | undefined): number {
  if (value === undefined) return Number.POSITIVE_INFINITY;
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError("maxDepth must be a non-negative number.");
  }
  return Math.floor(value);
}

function finitePositive(value: number, name: string): number {
  finite(value, name);
  if (!(value > 0)) throw new RangeError(`${name} must be greater than zero.`);
  return value;
}

function finite(value: number, name: string): void {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite.`);
}

function assertFinitePoint2(point: ProfileVec2, name: string): void {
  finite(point[0], `${name}[0]`);
  finite(point[1], `${name}[1]`);
}

function setLimitedBy(
  stats: MutableProfileStats,
  limitedBy: ProfileLimitedBy,
): void {
  if (LIMIT_PRIORITY[limitedBy] > LIMIT_PRIORITY[stats.limitedBy]) {
    stats.limitedBy = limitedBy;
  }
}

function snapshotStats(stats: MutableProfileStats): ProfileStats {
  return { ...stats };
}

function nowMs(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  throw signal.reason instanceof Error
    ? signal.reason
    : new DOMException("The operation was aborted.", "AbortError");
}

class ProfileNodeQueue {
  private readonly heap: QueueEntry[] = [];

  get size(): number {
    return this.heap.length;
  }

  push(entry: QueueEntry): void {
    const heap = this.heap;
    let i = heap.length;
    heap.push(entry);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (compareQueueEntry(heap[parent]!, entry) <= 0) break;
      heap[i] = heap[parent]!;
      i = parent;
    }
    heap[i] = entry;
  }

  pop(): QueueEntry | undefined {
    const heap = this.heap;
    const first = heap[0];
    const last = heap.pop();
    if (first === undefined || last === undefined || heap.length === 0) {
      return first;
    }

    let i = 0;
    for (;;) {
      const left = 2 * i + 1;
      if (left >= heap.length) break;
      const right = left + 1;
      const child =
        right < heap.length &&
        compareQueueEntry(heap[right]!, heap[left]!) < 0
          ? right
          : left;
      if (compareQueueEntry(last, heap[child]!) <= 0) break;
      heap[i] = heap[child]!;
      i = child;
    }
    heap[i] = last;
    return first;
  }
}

function compareQueueEntry(a: QueueEntry, b: QueueEntry): number {
  if (a.node.level !== b.node.level) return a.node.level - b.node.level;
  if (a.distance2 !== b.distance2) return a.distance2 - b.distance2;
  if (a.node.index !== b.node.index) return a.node.index - b.node.index;
  return a.sequence - b.sequence;
}
