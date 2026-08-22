import {
  BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  Sphere,
  Vector3,
} from "three/webgpu";
import type { Group, Material } from "three/webgpu";
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

const CORNERS = new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
const CORNER_INDEX = [0, 1, 2, 0, 2, 3];

/**
 * A level-partitioned slab allocator for point data.
 *
 * ONE `Mesh` per slab instead of one per node, which is the whole point: autzen
 * at a 3M budget selects 338-1184 nodes, so per-node meshes mean that many draw
 * calls and that many bind-group updates every frame. It also bounds a leak —
 * three's `Bindings.delete` is `DataMap.delete` and never destroys the
 * per-object uniform buffer, so per-node objects leak one buffer per node ever
 * attached, where an arena leaks at most one per slab.
 *
 * Slabs are partitioned BY LEVEL because the material reads `spacingWorld` and
 * `level` as per-object uniforms. Those land in the object bind group and so
 * never touch the pipeline cache key — which is what makes adaptive point size
 * cost zero bytes per point. Mixing levels in one slab would force size to
 * become per-point data.
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
  private readonly slabsByLevel = new Map<number, Slab[]>();
  private readonly allSlabs: Slab[] = [];
  private readonly slabCapacity: number;
  private readonly pageSize: number;
  readonly maxLivenessOpsPerFrame: number;
  private bytes = 0;

  constructor(
    private readonly parent: Group,
    private readonly material: Material,
    private readonly spacingAt: (level: number) => number,
    private readonly radiusAt: (level: number) => number,
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

  /** Allocate `numPoints` slots at `level`, growing the level if needed. */
  allocate(level: number, numPoints: number): ArenaBlock | undefined {
    if (numPoints <= 0) return undefined;
    const pages = Math.ceil(numPoints / this.pageSize);
    let list = this.slabsByLevel.get(level);
    if (list === undefined) {
      list = [];
      this.slabsByLevel.set(level, list);
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
    const slab = this.createSlab(level, capacity);
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
      return { level: slab.level, slab: slabIndex, start, count: numPoints, pages };
    }
    return undefined;
  }

  /** Return a block's pages, coalescing with neighbours. */
  free(block: ArenaBlock): void {
    const slab = this.slabsByLevel.get(block.level)?.[block.slab];
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
    const slab = this.slabsByLevel.get(block.level)?.[block.slab];
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
    const slab = this.slabsByLevel.get(block.level)?.[block.slab];
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
    const slab = this.slabsByLevel.get(block.level)?.[block.slab];
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

  private createSlab(level: number, capacity: number): Slab {
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
    mesh.userData['spacingWorld'] = this.spacingAt(level);
    mesh.userData['level'] = level;
    this.parent.add(mesh);

    const slab: Slab = {
      level,
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
    void this.radiusAt;
    return slab;
  }

  dispose(): void {
    for (const slab of this.allSlabs) {
      this.parent.remove(slab.mesh);
      slab.geometry.dispose();
    }
    this.allSlabs.length = 0;
    this.slabsByLevel.clear();
    this.bytes = 0;
  }
}
