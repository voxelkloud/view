import type { CloudFrame } from "./object.js";
/**
 * Synchronous spatial queries against the points currently resident on the
 * GPU-bound cut — the S1 surface.
 *
 * The contract that makes this usable in a game loop:
 *
 *  - It answers from what is LOADED, at whatever detail the scheduler has
 *    chosen, and says so. A caller that wants certainty must wait for a finer
 *    cut; a caller driving a car at 60 fps wants an answer this frame.
 *  - It never allocates per query once warm. A `heightAt` that produced
 *    garbage would stutter the very loop it exists to serve.
 *
 * The index is a flat XY hash over the resident points, rebuilt only when it
 * has to be. Two things keep that cheap, and both were learned the hard way by
 * driving a car through a city scan at 15 fps:
 *
 *  - A FOCUS WINDOW. Indexing every resident point means scanning millions per
 *    rebuild; a vehicle only ever asks about the ground within a few dozen
 *    metres of itself. Nodes outside the window are rejected by their bounds
 *    before a single point is read.
 *  - A REBUILD FLOOR. In a dense scan the cut changes almost every frame, so
 *    "rebuild when the cut changes" degenerates into "rebuild every frame".
 *    A stale index is harmless — the ground does not move — so rebuilds are
 *    spaced by time and by how far the focus has travelled.
 */

export interface GroundSample {
  /** Scene-space height at the query point. */
  readonly z: number;
  /** Points that contributed. Zero means no data — `z` is then a fallback. */
  readonly support: number;
  /** Horizontal distance to the nearest contributing point, in scene units. */
  readonly spread: number;
}

export interface GroundNormal {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** Destinos reutilizados: os dois laços correm por milhões de pontos. */
const scenePt = { x: 0, y: 0, z: 0 };
const nodeBox = new Float64Array(6);

interface ResidentSource {
  readonly selection: Int32Array | Uint32Array | readonly number[];
  readonly selectionCount: number;
  /** Onde esta nuvem está na cena — ver {@link CloudFrame}. */
  readonly frame: CloudFrame;
  node(index: number): { minX: number; minY: number; maxX: number; maxY: number } | undefined;
  readPoints(index: number):
    | {
        // Widened to match the sink's own type. Both sink paths hand back
        // scene-relative floats — the same assumption `pickPoint` makes.
        readonly positions: Float32Array | Int32Array;
        readonly start: number;
        readonly count: number;
      }
    | undefined;
}

export interface GroundIndexOptions {
  /** Cell edge in scene units. */
  readonly cellSize?: number;
  /** Centre of the window to index, in scene XY. Omitted means everything. */
  readonly focus?: readonly [number, number];
  /** Half-width of that window. Omitted means unbounded. */
  readonly radius?: number;
  /** Never rebuild more often than this, unless the focus left the window. */
  readonly minRebuildMs?: number;
  /** Injectable clock, for tests. */
  readonly now?: number;
}

/** Cell edge in scene units. Tuned so a car-sized query touches 1-4 cells. */
const DEFAULT_CELL = 1.0;

export class GroundIndex {
  private cell = DEFAULT_CELL;
  private minX = 0;
  private minY = 0;
  private nx = 0;
  private ny = 0;
  /** Per cell: the highest Z seen, and how many points landed in it. */
  private top = new Float32Array(0);
  private count = new Int32Array(0);
  /** Lowest Z, so a caller can tell a roof from the floor under it. */
  private low = new Float32Array(0);
  private builtFrom = -1;
  private builtAt = -Infinity;
  private focusX = 0;
  private focusY = 0;
  private radius = Infinity;
  private empty = true;

  get isEmpty(): boolean {
    return this.empty;
  }

  /** Cell size actually in use, in scene units. */
  get resolution(): number {
    return this.cell;
  }

  /**
   * Rebuild from the resident cut, if it is worth rebuilding.
   *
   * `token` changes when the resident set changes. Calling this every frame is
   * the intended use: the guards below decide whether anything actually runs.
   */
  rebuild(source: ResidentSource, token: number, options: GroundIndexOptions = {}): void {
    const cellSize = options.cellSize ?? DEFAULT_CELL;
    const radius = options.radius ?? Infinity;
    const focusX = options.focus?.[0] ?? 0;
    const focusY = options.focus?.[1] ?? 0;
    const minMs = options.minRebuildMs ?? 0;
    const now = options.now ?? (typeof performance !== "undefined" ? performance.now() : 0);

    if (!this.empty) {
      const movedFar =
        Number.isFinite(radius) &&
        Math.hypot(focusX - this.focusX, focusY - this.focusY) > radius / 3;
      const cutChanged = token !== this.builtFrom;
      // Nothing to do, or too soon to bother. A stale ground is still ground.
      if (!movedFar && !cutChanged) return;
      if (!movedFar && now - this.builtAt < minMs) return;
    }

    this.builtFrom = token;
    this.builtAt = now;
    this.focusX = focusX;
    this.focusY = focusY;
    this.radius = radius;
    this.cell = cellSize;

    const frame = source.frame;

    // Pass 1: the XY extent of the resident nodes, so the grid covers exactly
    // what is loaded rather than the whole cloud.
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const winMinX = focusX - radius;
    const winMaxX = focusX + radius;
    const winMinY = focusY - radius;
    const winMaxY = focusY + radius;
    for (let k = 0; k < source.selectionCount; k++) {
      const node = source.node(source.selection[k]!);
      if (node === undefined) continue;
      // A caixa do nó na CENA. Sob rotação já não está alinhada aos eixos, e
      // `sceneBox` devolve o limite que a contém — maior, nunca menor, para
      // que a rejeição por bounds não descarte um nó que toca a janela.
      frame.sceneBox(node.minX, node.minY, 0, node.maxX, node.maxY, 0, nodeBox);
      const nx0 = nodeBox[0]!;
      const ny0 = nodeBox[1]!;
      const nx1 = nodeBox[3]!;
      const ny1 = nodeBox[4]!;
      // Reject by bounds before any point of this node is read.
      if (nx1 < winMinX || nx0 > winMaxX || ny1 < winMinY || ny0 > winMaxY) continue;
      if (nx0 < minX) minX = nx0;
      if (ny0 < minY) minY = ny0;
      if (nx1 > maxX) maxX = nx1;
      if (ny1 > maxY) maxY = ny1;
    }
    // Clamp the grid to the window: a node may straddle its edge.
    minX = Math.max(minX, winMinX);
    minY = Math.max(minY, winMinY);
    maxX = Math.min(maxX, winMaxX);
    maxY = Math.min(maxY, winMaxY);
    if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
      this.empty = true;
      return;
    }

    const nx = Math.max(1, Math.min(4096, Math.ceil((maxX - minX) / this.cell) + 1));
    const ny = Math.max(1, Math.min(4096, Math.ceil((maxY - minY) / this.cell) + 1));
    const cells = nx * ny;
    if (this.top.length < cells) {
      this.top = new Float32Array(cells);
      this.low = new Float32Array(cells);
      this.count = new Int32Array(cells);
    }
    this.top.fill(-Infinity, 0, cells);
    this.low.fill(Infinity, 0, cells);
    this.count.fill(0, 0, cells);
    this.minX = minX;
    this.minY = minY;
    this.nx = nx;
    this.ny = ny;

    // Pass 2: bin every resident point.
    let seen = 0;
    for (let k = 0; k < source.selectionCount; k++) {
      const index = source.selection[k]!;
      const node = source.node(index);
      if (node !== undefined) {
        frame.sceneBox(node.minX, node.minY, 0, node.maxX, node.maxY, 0, nodeBox);
        const nx0 = nodeBox[0]!;
        const ny0 = nodeBox[1]!;
        const nx1 = nodeBox[3]!;
        const ny1 = nodeBox[4]!;
        if (nx1 < winMinX || nx0 > winMaxX || ny1 < winMinY || ny0 > winMaxY) continue;
      }
      const read = source.readPoints(index);
      if (read === undefined || read.count === 0) continue;
      const positions = read.positions;
      for (let i = 0; i < read.count; i++) {
        const j = 3 * (read.start + i);
        frame.localToScene(positions[j]!, positions[j + 1]!, positions[j + 2]!, scenePt);
        const sx = scenePt.x;
        const sy = scenePt.y;
        const sz = scenePt.z;
        const cx = ((sx - minX) / this.cell) | 0;
        const cy = ((sy - minY) / this.cell) | 0;
        if (cx < 0 || cy < 0 || cx >= nx || cy >= ny) continue;
        const at = cy * nx + cx;
        if (sz > this.top[at]!) this.top[at] = sz;
        if (sz < this.low[at]!) this.low[at] = sz;
        this.count[at]!++;
        seen++;
      }
    }
    this.empty = seen === 0;
  }

  /**
   * Surface height at an XY, from the highest returns nearby.
   *
   * `radius` widens the search until something is found, which is what keeps a
   * vehicle from falling through the gap between two laser sweeps. `support`
   * tells the caller how much data stood behind the answer.
   */
  heightAt(x: number, y: number, radius = 1.5): GroundSample {
    if (this.empty) return { z: 0, support: 0, spread: Infinity };
    const span = Math.max(1, Math.ceil(radius / this.cell));
    const cx = ((x - this.minX) / this.cell) | 0;
    const cy = ((y - this.minY) / this.cell) | 0;

    let best = -Infinity;
    let support = 0;
    let spread = Infinity;
    for (let j = cy - span; j <= cy + span; j++) {
      if (j < 0 || j >= this.ny) continue;
      for (let i = cx - span; i <= cx + span; i++) {
        if (i < 0 || i >= this.nx) continue;
        const at = j * this.nx + i;
        const n = this.count[at]!;
        if (n === 0) continue;
        support += n;
        const dx = (i + 0.5) * this.cell + this.minX - x;
        const dy = (j + 0.5) * this.cell + this.minY - y;
        const d = Math.hypot(dx, dy);
        if (d < spread) spread = d;
        if (this.top[at]! > best) best = this.top[at]!;
      }
    }
    return support === 0
      ? { z: 0, support: 0, spread: Infinity }
      : { z: best, support, spread };
  }

  /** Lowest return nearby — the floor beneath a canopy or a roof. */
  floorAt(x: number, y: number, radius = 1.5): GroundSample {
    if (this.empty) return { z: 0, support: 0, spread: Infinity };
    const span = Math.max(1, Math.ceil(radius / this.cell));
    const cx = ((x - this.minX) / this.cell) | 0;
    const cy = ((y - this.minY) / this.cell) | 0;
    let best = Infinity;
    let support = 0;
    for (let j = cy - span; j <= cy + span; j++) {
      if (j < 0 || j >= this.ny) continue;
      for (let i = cx - span; i <= cx + span; i++) {
        if (i < 0 || i >= this.nx) continue;
        const at = j * this.nx + i;
        if (this.count[at] === 0) continue;
        support += this.count[at]!;
        if (this.low[at]! < best) best = this.low[at]!;
      }
    }
    return support === 0 ? { z: 0, support: 0, spread: Infinity } : { z: best, support, spread: 0 };
  }

  /**
   * Surface normal from four height samples around the point.
   *
   * Central differences rather than a plane fit: a vehicle needs the tilt to
   * settle, not the statistically best plane, and four lookups is a tenth of
   * the cost.
   */
  normalAt(
    x: number,
    y: number,
    step = 1.5,
    surface: "top" | "floor" = "top",
  ): GroundNormal {
    const at = (px: number, py: number): number =>
      surface === "floor" ? this.floorAt(px, py).z : this.heightAt(px, py).z;
    const e = at(x + step, y);
    const w = at(x - step, y);
    const n = at(x, y + step);
    const s = at(x, y - step);
    const dzdx = (e - w) / (2 * step);
    const dzdy = (n - s) / (2 * step);
    const len = Math.hypot(dzdx, dzdy, 1);
    return { x: -dzdx / len, y: -dzdy / len, z: 1 / len };
  }

  /**
   * How much the surface rises within `radius` — a cheap obstacle test.
   *
   * Returns the tallest point above `fromZ`, which is what a car needs to know
   * to refuse to drive into a wall without any collision mesh existing.
   */
  riseAhead(x: number, y: number, fromZ: number, radius = 2): number {
    if (this.empty) return 0;
    const span = Math.max(1, Math.ceil(radius / this.cell));
    const cx = ((x - this.minX) / this.cell) | 0;
    const cy = ((y - this.minY) / this.cell) | 0;
    let rise = 0;
    for (let j = cy - span; j <= cy + span; j++) {
      if (j < 0 || j >= this.ny) continue;
      for (let i = cx - span; i <= cx + span; i++) {
        if (i < 0 || i >= this.nx) continue;
        const at = j * this.nx + i;
        if (this.count[at] === 0) continue;
        const d = this.top[at]! - fromZ;
        if (d > rise) rise = d;
      }
    }
    return rise;
  }
}
