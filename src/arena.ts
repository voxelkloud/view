import {
  BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  Sphere,
  Vector3,
} from "three";
import type { Group, Material } from "three";
import type { PointReadback } from "./sink.js";

/**
 * Where one node's points live inside a slab.
 *
 * `pointOffset` occupies elements `[3*start, 3*(start+count))` and `color`
 * `[4*start, 4*(start+count))`. Page alignment keeps both byte offsets and both
 * sizes multiples of 4, which `device.queue.writeBuffer` requires.
 */
export interface ArenaBlock {
  readonly level: number;
  /**
   * Which slab list this block lives in. EQUAL TO `level` for every tree whose
   * point pitch is a closed form of the level — which is every octree format —
   * and a spacing bucket otherwise. See {@link PointArena.allocate}.
   */
  readonly key: number;
  readonly slab: number;
  /** First point slot. */
  readonly start: number;
  /** Slots used; whole pages beyond this are padding. */
  readonly count: number;
  readonly pages: number;
}

export interface ArenaOptions {
  /**
   * Point slots per slab. Default 524,288 — 6.3 MiB of positions plus 2.1 MiB
   * of colour. The FIRST slab of a level is sized to the level's own demand so
   * levels 0-2 do not each waste a full slab: autzen's level 0 is 10,833 points.
   */
  readonly slabCapacity?: number;
  /** Allocation granularity, in points. Default 1024. */
  readonly pageSize?: number;
  /**
   * Show/hide alpha ops per frame. Default 128. Both directions are safe to
   * defer — a deferred hide over-draws for one frame, a deferred show leaves a
   * hole — so shows are prioritised.
   */
  readonly maxLivenessOpsPerFrame?: number;
}

interface FreeRun {
  page: number;
  pages: number;
}

interface Slab {
  readonly level: number;
  readonly key: number;
  /** What the mesh's `spacingWorld` uniform was stamped with. */
  readonly spacingWorld: number;
  readonly capacity: number;
  readonly pages: number;
  readonly positions: Float32Array;
  readonly colors: Uint8Array;
  /** Present only when the material's colour mode reads a scalar attribute. */
  readonly scalars: Float32Array | undefined;
  readonly geometry: InstancedBufferGeometry;
  readonly mesh: Mesh;
  readonly posAttr: InstancedBufferAttribute;
  readonly colorAttr: InstancedBufferAttribute;
  readonly scalarAttr: InstancedBufferAttribute | undefined;
  /** Address-ordered, coalesced on free. */
  free: FreeRun[];
  /** The high-water mark IS `instanceCount`: everything below is drawn and
   *  masked by alpha, everything above has never been written. */
  highWater: number;
  dirty: boolean;
}

/**
 * The unit quad, four corners and six indices.
 *
 * A three-vertex triangle circumscribing the same disc was tried and reverted.
 * It cut vertices per splat by 25% and INP did not move (296-352 ms against
 * 320-324), while the triangle's 30% larger envelope cost idle frame rate
 * (58.1-59.2 fps against 59.9).
 *
 * That null result sharpened the model. INP scales with INSTANCES — 3.0M gives
 * ~320 ms, 2.0M gives 180, 1.0M gives 120, 750k gives 76 — and not with
 * vertices per instance. So the cost is per-instance work: stepping the
 * instanced attributes and fetching `pointOffset`, `color` and `scalarValue`
 * once per splat, which happens whatever the envelope is. Potree pays none of
 * it because `gl.POINTS` has no instancing at all.
 *
 * The consequence is that no envelope change can close this gap; only drawing
 * fewer instances, or leaving the instanced-draw model altogether.
 */
const CORNERS = new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
const CORNER_INDEX = [0, 1, 2, 0, 2, 3];

/**
 * A slab key for a pitch that is NOT the level's closed form.
 *
 * QUARTER-OCTAVE buckets: two pitches share a slab only within a factor of
 * 2^(1/4), so a point is drawn within 9% of its own pitch — well under the
 * error the splat's own quad rounding already carries, and enough to keep a
 * tileset from opening one slab per distinct tile error.
 *
 * NEGATIVE, so a bucket can never collide with a plain level key: levels are
 * small non-negative integers and these are not.
 */
function bucketKey(level: number, spacingWorld: number): number {
  const q = Math.round(Math.log2(spacingWorld) * 4);
  const clamped = q < -2048 ? -2048 : q > 2047 ? 2047 : q;
  return -(1 + level * 4096 + (clamped + 2048));
}

/**
 * A pitch-partitioned slab allocator for point data.
 *
 * ONE `Mesh` per slab instead of one per node, which is the whole point: autzen
 * at a 3M budget selects 338-1184 nodes, so per-node meshes mean that many draw
 * calls and that many bind-group updates every frame. It also bounds a leak —
 * three's `Bindings.delete` is `DataMap.delete` and never destroys the
 * per-object uniform buffer, so per-node objects leak one buffer per node ever
 * attached, where an arena leaks at most one per slab.
 *
 * Slabs are partitioned BY POINT PITCH because the material reads
 * `spacingWorld` and `level` as per-object uniforms. Those land in the object
 * bind group and so never touch the pipeline cache key — which is what makes
 * adaptive point size cost zero bytes per point. Mixing pitches in one slab
 * would force size to become per-point data.
 *
 * For every format whose pitch is a closed form of the level — Potree v2, COPC,
 * EPT, every octree — the partition IS the level, exactly and by construction:
 * `allocate` compares the pitch it is given against `pointSpacingAt(level)` and
 * keys on the level when they are the same number. A format that carries a
 * per-node pitch (a tileset, whose tiles at one depth are not one size) keys on
 * a QUARTER-OCTAVE bucket instead, so points inside a slab are drawn within 9%
 * of their own pitch and the slab count stays bounded. That the two cases are
 * decided by one equality, in one place, is what keeps the common one free.
 *
 * Sub-range draws are NOT available: `RenderObject.getDrawParameters` never
 * offsets `firstInstance` off the BatchedMesh path, and `geometry.groups` bound
 * firstVertex/vertexCount, which a 6-index quad does not vary. So the whole slab
 * is drawn every frame and per-point liveness is carried in the colour ALPHA
 * byte — the material multiplies the splat diameter by it, collapsing a dead or
 * unselected slot to a zero-area quad. Masking is required rather than an
 * optimisation: a resident set several times the point budget would otherwise
 * be pure overdraw.
 */
export class PointArena {
  private readonly slabsByKey = new Map<number, Slab[]>();
  private readonly allSlabs: Slab[] = [];
  private readonly slabCapacity: number;
  private readonly pageSize: number;
  readonly maxLivenessOpsPerFrame: number;
  private bytes = 0;

  constructor(
    private readonly parent: Group,
    private readonly material: Material,
    /** POINT PITCH at a level — the world-space size a point is drawn at.
     *  Never a geometric error: those diverge on non-octree formats. */
    private readonly pointSpacingAt: (level: number) => number,
    private readonly boundingRadiusAt: (level: number) => number,
    options: ArenaOptions = {},
    /**
     * Allocate a per-point scalar lane, for the intensity and classification
     * colour modes. Decided once per cloud, because it changes the slab layout.
     */
    private readonly withScalar = false,
  ) {
    this.slabCapacity = options.slabCapacity ?? 524_288;
    this.pageSize = options.pageSize ?? 1024;
    this.maxLivenessOpsPerFrame = options.maxLivenessOpsPerFrame ?? 128;
  }

  /**
   * Instance slots the vertex stage actually processes, summed over slabs.
   *
   * `slab.highWater`, not attached points and not `residentBytes` — the first
   * undercounts because a freed block below the high-water mark still costs a
   * vertex invocation, and the second is slab CAPACITY, which includes page
   * slack. This is the number the draw dispatches, so it is the one the
   * mask-versus-compact question turns on.
   */
  get residentPoints(): number {
    let n = 0;
    for (const slab of this.allSlabs) n += slab.highWater;
    return n;
  }

  get residentBytes(): number {
    return this.bytes;
  }
  get slabCount(): number {
    return this.allSlabs.length;
  }
  /** Points actually drawn, i.e. the sum of every slab's high-water mark. */
  get drawnSlots(): number {
    let n = 0;
    for (const s of this.allSlabs) n += s.highWater;
    return n;
  }

  /**
   * Allocate `numPoints` slots for a node at `level` whose points are drawn at
   * `spacingWorld`.
   *
   * `spacingWorld` is optional and defaults to the level's closed form, which
   * is what every octree format wants and what this did before per-node pitches
   * existed. Passing the SAME number the closed form would produce is also
   * free: the key is then the level and not a bucket, so nothing about the
   * partition moves.
   */
  allocate(
    level: number,
    numPoints: number,
    spacingWorld?: number,
  ): ArenaBlock | undefined {
    if (numPoints <= 0) return undefined;
    const closed = this.pointSpacingAt(level);
    const pitch =
      spacingWorld === undefined ||
      !Number.isFinite(spacingWorld) ||
      spacingWorld <= 0
        ? closed
        : spacingWorld;
    const key = pitch === closed ? level : bucketKey(level, pitch);
    const pages = Math.ceil(numPoints / this.pageSize);
    let list = this.slabsByKey.get(key);
    if (list === undefined) {
      list = [];
      this.slabsByKey.set(key, list);
    }

    for (let s = 0; s < list.length; s++) {
      const block = this.allocateIn(list[s]!, s, pages, numPoints);
      if (block !== undefined) return block;
    }

    // Only the FIRST slab of a level is sized down to demand — autzen's level 0
    // is 10,833 points and a full slab per level would waste most of it. Every
    // subsequent slab is full capacity, because a level that has already
    // overflowed once will keep growing, and sizing those to demand too would
    // produce dozens of small slabs and defeat the whole point of the arena.
    // A slab must also never be smaller than the request it is being created
    // for, or a single large node could never be placed.
    const wanted =
      list.length === 0
        ? Math.min(this.slabCapacity, Math.max(16_384, pages * this.pageSize))
        : this.slabCapacity;
    const capacity = Math.max(wanted, pages * this.pageSize);
    const slab = this.createSlab(level, key, pitch, capacity);
    list.push(slab);
    return this.allocateIn(slab, list.length - 1, pages, numPoints);
  }

  private allocateIn(
    slab: Slab,
    slabIndex: number,
    pages: number,
    numPoints: number,
  ): ArenaBlock | undefined {
    for (let i = 0; i < slab.free.length; i++) {
      const run = slab.free[i]!;
      if (run.pages < pages) continue;
      const page = run.page;
      run.page += pages;
      run.pages -= pages;
      if (run.pages === 0) slab.free.splice(i, 1);
      const start = page * this.pageSize;
      const end = start + numPoints;
      if (end > slab.highWater) slab.highWater = end;
      slab.dirty = true;
      return {
        level: slab.level,
        key: slab.key,
        slab: slabIndex,
        start,
        count: numPoints,
        pages,
      };
    }
    return undefined;
  }

  /** Return a block's pages, coalescing with neighbours. */
  free(block: ArenaBlock): void {
    const slab = this.slabsByKey.get(block.key)?.[block.slab];
    if (slab === undefined) return;
    const page = block.start / this.pageSize;

    let i = 0;
    while (i < slab.free.length && slab.free[i]!.page < page) i++;
    slab.free.splice(i, 0, { page, pages: block.pages });

    // Coalesce with the following run, then the preceding one.
    const cur = slab.free[i]!;
    const next = slab.free[i + 1];
    if (next !== undefined && cur.page + cur.pages === next.page) {
      cur.pages += next.pages;
      slab.free.splice(i + 1, 1);
    }
    const prev = i > 0 ? slab.free[i - 1] : undefined;
    if (prev !== undefined && prev.page + prev.pages === cur.page) {
      prev.pages += cur.pages;
      slab.free.splice(i, 1);
    }
    slab.dirty = true;
  }

  /**
   * Copy one node's decoded arrays into its block.
   *
   * Two `TypedArray.set` calls and, when the source carried its own alpha, one
   * strided stamp. There is deliberately NO per-point arithmetic: Task 4 already
   * emits `Float32Array(3n)` relative to the shared cloud origin, and point size
   * comes from a per-object uniform rather than per-point data.
   *
   * `colors` is optional because a cloud can legitimately have no colour: LAS
   * point format 1 carries intensity and classification and no RGB, which is
   * most 3DEP lidar. The colour buffer is still allocated and still written,
   * because its ALPHA byte is the liveness mask the material multiplies the
   * splat diameter by — a colourless cloud with an unwritten colour block would
   * have alpha 0 everywhere and draw absolutely nothing.
   */
  stage(
    block: ArenaBlock,
    positions: Float32Array,
    colors: Uint8Array | undefined,
    needsAlphaStamp: boolean,
    scalars?: Float32Array | undefined,
  ): void {
    const slab = this.slabsByKey.get(block.key)?.[block.slab];
    if (slab === undefined) return;
    slab.positions.set(positions.subarray(0, 3 * block.count), 3 * block.start);
    if (colors === undefined) {
      // White, fully live. RGB is unused by the elevation/flat/level colour
      // modes a colourless cloud must use anyway.
      slab.colors.fill(255, 4 * block.start, 4 * (block.start + block.count));
    } else {
      slab.colors.set(colors.subarray(0, 4 * block.count), 4 * block.start);
      if (needsAlphaStamp) {
        const end = 4 * (block.start + block.count);
        for (let o = 4 * block.start + 3; o < end; o += 4) slab.colors[o] = 255;
      }
    }
    if (slab.scalars !== undefined && slab.scalarAttr !== undefined) {
      if (scalars !== undefined) {
        slab.scalars.set(scalars.subarray(0, block.count), block.start);
      } else {
        slab.scalars.fill(0, block.start, block.start + block.count);
      }
      slab.scalarAttr.addUpdateRange(block.start, block.count);
    }
    slab.posAttr.addUpdateRange(3 * block.start, 3 * block.count);
    slab.colorAttr.addUpdateRange(4 * block.start, 4 * block.count);
    slab.dirty = true;
  }

  readPoints(block: ArenaBlock): PointReadback | undefined {
    const slab = this.slabsByKey.get(block.key)?.[block.slab];
    if (slab === undefined) return undefined;
    return {
      positions: slab.positions,
      start: block.start,
      count: block.count,
      colors: slab.colors,
      ...(slab.scalars !== undefined ? { scalars: slab.scalars } : {}),
    };
  }

  /** Write `v` into every alpha byte of a block — the liveness mask. */
  setAlpha(block: ArenaBlock, v: number): void {
    const slab = this.slabsByKey.get(block.key)?.[block.slab];
    if (slab === undefined) return;
    const end = 4 * (block.start + block.count);
    for (let o = 4 * block.start + 3; o < end; o += 4) slab.colors[o] = v;
    slab.colorAttr.addUpdateRange(4 * block.start, 4 * block.count);
    slab.dirty = true;
  }

  /**
   * Flush pending writes.
   *
   * `needsUpdate` is a version bump; three re-reads only on a version change and
   * issues ONE `writeBuffer` per accumulated update range, then clears them.
   * Usage stays `StaticDrawUsage` — `DynamicDrawUsage` would re-upload the whole
   * slab every frame.
   */
  commit(): void {
    for (const slab of this.allSlabs) {
      if (!slab.dirty) continue;
      slab.posAttr.needsUpdate = true;
      slab.colorAttr.needsUpdate = true;
      if (slab.scalarAttr !== undefined) slab.scalarAttr.needsUpdate = true;
      slab.geometry.instanceCount = slab.highWater;
      slab.mesh.visible = slab.highWater > 0;
      slab.dirty = false;
    }
  }

  private createSlab(
    level: number,
    key: number,
    spacingWorld: number,
    capacity: number,
  ): Slab {
    const pages = Math.ceil(capacity / this.pageSize);
    const slots = pages * this.pageSize;
    const positions = new Float32Array(3 * slots);
    const colors = new Uint8Array(4 * slots);
    const scalars = this.withScalar ? new Float32Array(slots) : undefined;

    const geometry = new InstancedBufferGeometry();
    // A fresh corner attribute and index per geometry: three binds a dispose
    // listener to the first render object that used a geometry, so sharing one
    // instance would make disposal order load-bearing.
    geometry.setAttribute("position", new BufferAttribute(CORNERS.slice(), 3));
    geometry.setIndex(CORNER_INDEX.slice());

    const posAttr = new InstancedBufferAttribute(positions, 3, false);
    // `normalized: true` is load-bearing: it yields unorm8x4 at 4 B/pt and a
    // float vec4 in the shader. False makes three widen the array to
    // Uint32Array IN PLACE — 4x the GPU memory and an integer-typed value — and
    // it still renders, just wrong.
    const colorAttr = new InstancedBufferAttribute(colors, 4, true);
    geometry.setAttribute("pointOffset", posAttr);
    geometry.setAttribute("color", colorAttr);
    const scalarAttr =
      scalars === undefined
        ? undefined
        : new InstancedBufferAttribute(scalars, 1, false);
    if (scalarAttr !== undefined) {
      geometry.setAttribute("scalarValue", scalarAttr);
    }
    geometry.instanceCount = 0;

    // Explicit, because the only geometry three could infer from is the unit
    // quad. Never culled: visibility is the scheduler's decision, made against
    // the same frustum.
    geometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

    const mesh = new Mesh(geometry, this.material);
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.visible = false;
    mesh.updateMatrix();
    mesh.userData['spacingWorld'] = spacingWorld;
    mesh.userData['level'] = level;
    this.parent.add(mesh);

    const slab: Slab = {
      level,
      key,
      spacingWorld,
      capacity: slots,
      pages,
      positions,
      colors,
      scalars,
      geometry,
      mesh,
      posAttr,
      colorAttr,
      scalarAttr,
      free: [{ page: 0, pages }],
      highWater: 0,
      dirty: false,
    };
    this.allSlabs.push(slab);
    this.bytes +=
      positions.byteLength + colors.byteLength + (scalars?.byteLength ?? 0);
    void this.boundingRadiusAt;
    return slab;
  }

  dispose(): void {
    for (const slab of this.allSlabs) {
      this.parent.remove(slab.mesh);
      slab.geometry.dispose();
    }
    this.allSlabs.length = 0;
    this.slabsByKey.clear();
    this.bytes = 0;
  }
}
