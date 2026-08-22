import type { DecodedPointData } from "@voxelkloud/loader";
import type { Group, Material } from "three/webgpu";
import { PointArena } from "./arena.js";
import type { ArenaBlock, ArenaOptions } from "./arena.js";
import type { PointReadback, PointSink } from "./sink.js";

/**
 * A {@link PointSink} backed by a level-partitioned slab arena.
 *
 * Collapses one draw call per node into one per slab. Per-point liveness rides
 * in the colour ALPHA byte, which the material multiplies the splat diameter by
 * — a dead or unselected slot becomes a zero-area quad. voxelkloud OWNS that
 * byte: Task 4 writes the SOURCE alpha when the attribute really has four
 * elements, so it is stamped to 255 at staging rather than assumed.
 *
 * Liveness ops are bounded per frame. Shows are issued before hides because a
 * deferred show leaves a visible hole while a deferred hide only over-draws for
 * one frame.
 */
export class ArenaSink implements PointSink {
  private readonly arena: PointArena;
  private readonly blocks = new Map<number, ArenaBlock>();
  /** 1 when a node's alpha is currently 255. */
  private readonly shown = new Map<number, number>();
  private readonly shownEpoch = new Map<number, number>();
  private shownList: number[] = [];
  private epoch = 0;

  /**
   * @param needsAlphaStamp whether the SOURCE colour attribute carries its own
   * alpha, decided once per cloud from `numElements >= 4`. Task 4 writes the
   * source value in that case and the format maximum otherwise, so an RGB
   * source already has alpha 255 and the strided stamp is pure waste.
   */
  constructor(
    parent: Group,
    material: Material,
    spacingAt: (level: number) => number,
    radiusAt: (level: number) => number,
    private readonly needsAlphaStamp: boolean,
    options: ArenaOptions = {},
    /**
     * The attribute the material's colour mode reads per point, if any —
     * `"intensity"` or `"classification"`. Decided once per cloud, because it
     * changes the slab layout.
     */
    private readonly scalarAttribute: string | undefined = undefined,
  ) {
    this.arena = new PointArena(
      parent,
      material,
      spacingAt,
      radiusAt,
      options,
      scalarAttribute !== undefined,
    );
  }

  get residentBytes(): number {
    return this.arena.residentBytes;
  }
  get nodeCount(): number {
    return this.blocks.size;
  }
  get slabCount(): number {
    return this.arena.slabCount;
  }
  /**
   * `true` when the source colour carried its own alpha, which this sink
   * overwrites to carry liveness. Surfaced so the decision is visible rather
   * than silent.
   */
  get discardedColorAlpha(): boolean {
    return this.needsAlphaStamp;
  }

  attach(
    index: number,
    data: DecodedPointData,
    _spacingWorld: number,
    level: number,
  ): number {
    if (this.blocks.has(index)) return 0;
    if (data.numPoints === 0) return 0;
    if (!(data.positions instanceof Float32Array)) return 0;
    // A cloud with no colour attribute is normal, not an error: LAS point
    // format 1 carries intensity and classification and no RGB. Refusing it
    // here is what made a 50.7M-point 3DEP survey load its whole hierarchy,
    // decode every node, and draw nothing at all.
    const colors =
      data.colors?.array instanceof Uint8Array ? data.colors.array : undefined;

    const block = this.arena.allocate(level, data.numPoints);
    if (block === undefined) return 0;

    // Task 4's `f32` gpu lane, so the raw value arrives as a Float32Array and
    // the shader normalises it from the declared range.
    const scalar =
      this.scalarAttribute === undefined
        ? undefined
        : (data.attributesByName.get(this.scalarAttribute)?.array as
            | Float32Array
            | undefined);
    this.arena.stage(
      block,
      data.positions,
      colors,
      this.needsAlphaStamp,
      scalar,
    );
    this.blocks.set(index, block);
    // Newly attached nodes start visible; setVisible will hide them if the
    // scheduler did not ask for them.
    this.shown.set(index, 1);
    this.shownList.push(index);
    // What was STAGED this call, which is what the view's per-frame upload
    // budget is spending. The scalar lane is real staged bytes, so leaving it
    // out would let a scalar colour mode overshoot that budget by a third.
    // Slab residency is a different number and comes from `residentBytes`.
    return (
      data.positions.byteLength +
      (colors?.byteLength ?? 0) +
      (scalar !== undefined ? 4 * data.numPoints : 0)
    );
  }

  detach(index: number): void {
    const block = this.blocks.get(index);
    if (block === undefined) return;
    // Zero the alpha before freeing: the pages may be reallocated before the
    // next commit, but until they are written the slots must draw nothing.
    this.arena.setAlpha(block, 0);
    this.arena.free(block);
    this.blocks.delete(index);
    this.shown.delete(index);
    this.shownEpoch.delete(index);
  }

  setVisible(indices: Int32Array, count: number): void {
    const epoch = ++this.epoch;
    const max = this.arena.maxLivenessOpsPerFrame;
    let ops = 0;

    // Shows first: a deferred show is a hole, a deferred hide is one frame of
    // overdraw.
    for (let k = 0; k < count; k++) {
      const i = indices[k]!;
      this.shownEpoch.set(i, epoch);
      if (this.shown.get(i) === 1) continue;
      const block = this.blocks.get(i);
      if (block === undefined || ops >= max) continue;
      this.arena.setAlpha(block, 255);
      this.shown.set(i, 1);
      this.shownList.push(i);
      ops++;
    }

    let w = 0;
    for (let k = 0; k < this.shownList.length; k++) {
      const i = this.shownList[k]!;
      if (this.shownEpoch.get(i) !== epoch && ops < max) {
        const block = this.blocks.get(i);
        if (block !== undefined) this.arena.setAlpha(block, 0);
        this.shown.set(i, 0);
        ops++;
        continue;
      }
      this.shownList[w++] = i;
    }
    this.shownList.length = w;
  }

  commit(): void {
    this.arena.commit();
  }

  readPoints(index: number): PointReadback | undefined {
    const block = this.blocks.get(index);
    if (block === undefined) return undefined;
    return this.arena.readPoints(block);
  }

  dispose(): void {
    this.arena.dispose();
    this.blocks.clear();
    this.shown.clear();
    this.shownEpoch.clear();
    this.shownList.length = 0;
  }
}
