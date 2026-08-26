// The OCTREE CUT: which nodes the LOD pass selected, in a form the vertex stage
// can walk.
//
// Exists to answer one question per point: how deep does the SELECTION go at
// this position? A splat has to be as wide as the point pitch of the finest
// data present where it lands, not the pitch of the node it happens to have
// come from. Those are the same number only on the frontier. Everywhere behind
// it — every ancestor of a refined region, and on an octree that is every level
// above the deepest — sizing by the node's own level draws a splat 2**(D-L)
// times too wide, and the coarse layers paint over the fine ones that are
// already resident and already correct. Measured on autzen at a 0.25 framing
// with a 3M budget: a level-0 point lands 28 px wide (clipped to `maxPixelSize`
// at 8) in a region whose level-6 data wants 0.44 px.
//
// The reference solves the same problem with a `visibleNodes` RGBA8 texture and
// a per-node `uVNStart`, walked in the vertex shader. This is that structure,
// with two differences. It is built BREADTH-FIRST so a node's selected children
// are contiguous and one 24-bit offset addresses them all, and the walk starts
// at the ROOT rather than at the drawn node — because the arena batches slabs
// by LEVEL, not by node, so there is no per-node uniform to hang a start offset
// on and no per-point lane to spend 4 bytes on. The cost of that choice is the
// walk length: D steps rather than D - L.
import { DataTexture, NearestFilter, RGBAFormat, UnsignedByteType } from "three";
import type { PointCloudNode } from "@voxelkloud/core";

/**
 * Texels per row. A power of two so the shader's divide and modulo are a shift
 * and a mask, which is also why it is a constant rather than an option: it is
 * baked into the addressing on both sides.
 */
export const CUT_WIDTH = 1024;
export const CUT_WIDTH_SHIFT = 10;

/**
 * How deep the shader walk may go, and so the cap on reported depth.
 *
 * Bounded by float32, not by taste. The walk halves a cloud-local box `depth`
 * times and compares a `pointOffset` against its centre; at autzen's 4655 m
 * root extent the f32 ULP near the far corner is 4.9e-4 m, and level 20 cells
 * are 4.4e-3 m — 9x margin. Level 23 is 5.5e-4 m and the comparison stops
 * meaning anything. No stock converter emits 20 levels: autzen is 6.
 */
export const MAX_CUT_DEPTH = 20;

/** Bytes per entry: `[childMask, first >> 16, first >> 8, first]`. */
const STRIDE = 4;

/**
 * One entry per selected node, breadth-first from the root.
 *
 * - `r` — occupancy of the SELECTED children, bit `c` for octant `c`. Never the
 *   node's own `childMask`: a child that exists but was not selected must
 *   terminate the walk, because its data is not on the GPU.
 * - `g,b,a` — index of this node's first selected child, 24-bit big-endian.
 *   Breadth-first order is what makes one offset enough; the siblings that
 *   follow are addressed by counting set mask bits below the octant.
 */
export class OctreeCut {
  private data: Uint8Array;
  private queue: Int32Array;
  private texture: DataTexture;
  private capacity: number;
  /** Entries written by the last {@link build}. */
  entryCount = 0;

  constructor(maxNodes = 4096) {
    this.capacity = Math.max(1, maxNodes);
    this.data = new Uint8Array(this.capacity * STRIDE);
    this.queue = new Int32Array(this.capacity);
    this.texture = this.makeTexture();
  }

  get map(): DataTexture {
    return this.texture;
  }

  private makeTexture(): DataTexture {
    const rows = Math.max(1, Math.ceil(this.capacity / CUT_WIDTH));
    // The backing store is `capacity * STRIDE` and the texture is
    // `CUT_WIDTH * rows * STRIDE`, which is only the same when capacity is a
    // multiple of the row width. Pad, and let the tail stay zero — a zero mask
    // terminates the walk, so unreachable slots are safe by construction.
    const bytes = CUT_WIDTH * rows * STRIDE;
    if (this.data.length < bytes) {
      const grown = new Uint8Array(bytes);
      grown.set(this.data);
      this.data = grown;
    }
    const t = new DataTexture(this.data, CUT_WIDTH, rows, RGBAFormat, UnsignedByteType);
    // NEAREST and no mips: every read is an exact texel fetch of a bitfield.
    // A filtered read would average two nodes' masks into a number that means
    // nothing.
    t.minFilter = NearestFilter;
    t.magFilter = NearestFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  }

  /** Rows in the current texture. The shader needs no equivalent: it addresses
   *  by shift and mask off a linear slot. */
  get rows(): number {
    return this.texture.image.height;
  }

  /**
   * Rebuild from a finished selection. Allocation-free once `maxNodes` settles.
   *
   * `epoch[i] === frame` IS the selection membership test — the same side array
   * `selectVisible` stamps, read rather than copied.
   *
   * `resident` narrows that to what is actually ON THE GPU, and it is not
   * optional bookkeeping. The cut says how fine the picture is at a position,
   * and a selected-but-still-loading node is a promise, not a picture: honour
   * it and every coarse splat above it shrinks for the data's arrival, opening
   * holes that stay open for as long as the fetch takes. Built over what has
   * landed, a load coarsens and then sharpens, which is the behaviour a
   * streaming viewer is supposed to have.
   *
   * A node whose parent has not landed is simply unreachable from the root and
   * shrinks nothing — conservative in the safe direction. Streaming runs in the
   * selection's own best-first order, so parents almost always arrive first
   * anyway.
   */
  build(
    root: PointCloudNode,
    node: (index: number) => PointCloudNode | undefined,
    epoch: Int32Array,
    frame: number,
    maxNodes: number,
    resident: ReadonlySet<number>,
  ): void {
    if (maxNodes > this.capacity) {
      this.capacity = maxNodes;
      this.data = new Uint8Array(this.capacity * STRIDE);
      this.queue = new Int32Array(this.capacity);
      this.texture.dispose();
      this.texture = this.makeTexture();
    }
    const data = this.data;
    const q = this.queue;

    // A cut with no root is a cut with no nodes. One empty entry rather than
    // zero, so the walk always has a slot 0 to read and terminates at depth 0.
    if (epoch[root.index] !== frame || !resident.has(root.index)) {
      data[0] = 0;
      data[1] = 0;
      data[2] = 0;
      data[3] = 0;
      this.entryCount = 1;
      this.texture.needsUpdate = true;
      return;
    }

    let qn = 0;
    q[qn++] = root.index;
    // Breadth-first order and allocation order are THE SAME SEQUENCE, so a
    // node's queue position is its entry slot and `write` is always the slot
    // its first child will land in. That identity is the whole reason the
    // encoding needs one offset instead of eight.
    let write = 1;
    for (let qi = 0; qi < qn; qi++) {
      const n = node(q[qi]!);
      let mask = 0;
      const first = write;
      if (n !== undefined) {
        for (let c = 0; c < 8; c++) {
          const child = n.children[c];
          if (child === undefined) continue;
          if (epoch[child.index] !== frame) continue;
          if (!resident.has(child.index)) continue;
          mask |= 1 << c;
          // Cannot overflow: `qn` only ever reaches the number of selected
          // nodes, which `selectVisible` capped at `maxNodes`.
          q[qn++] = child.index;
          write++;
        }
      }
      const o = qi * STRIDE;
      data[o] = mask;
      data[o + 1] = (first >>> 16) & 0xff;
      data[o + 2] = (first >>> 8) & 0xff;
      data[o + 3] = first & 0xff;
    }
    this.entryCount = qn;
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.texture.dispose();
  }

  /**
   * The encoded bytes: per slot, the child mask then the first child's slot as
   * three big-endian bytes, `STRIDE` apart. Read `entryCount` for how many
   * slots are live.
   *
   * Also what the compute rasterizer uploads as a storage buffer, so that path
   * reads the SAME bytes the vertex walk reads instead of re-encoding the
   * layout. Two encoders of one format is how the two drift.
   */
  get bytes(): Uint8Array {
    return this.data;
  }
}

/**
 * THE ORACLE. The same walk the vertex shader runs, in TypeScript.
 *
 * The shader is the fast path and this is the definition; a differential test
 * over a synthetic tree pins them together, which is the only way a 24-bit
 * offset packed into three colour channels stays honest.
 *
 * @param p CLOUD-LOCAL position, the frame `pointOffset` is in.
 * @param rootMin  cloud-local min corner of the root box
 * @param rootSize cloud-local extent of the root box, per axis
 * @returns depth of the deepest SELECTED node containing `p`
 */
export function localDepth(
  cut: Uint8Array,
  p: readonly [number, number, number],
  rootMin: readonly [number, number, number],
  rootSize: readonly [number, number, number],
  maxDepth = MAX_CUT_DEPTH,
): number {
  let bx = rootMin[0];
  let by = rootMin[1];
  let bz = rootMin[2];
  let sx = rootSize[0];
  let sy = rootSize[1];
  let sz = rootSize[2];
  let slot = 0;
  let depth = 0;

  for (let i = 0; i < maxDepth; i++) {
    const o = slot * STRIDE;
    const mask = cut[o]!;
    if (mask === 0) return depth;
    const hx = sx * 0.5;
    const hy = sy * 0.5;
    const hz = sz * 0.5;
    // `>=` on the centre, matching `step()` in the shader, matching
    // `makeChildNode`'s `lo += size/2` half-open split.
    const cx = p[0] >= bx + hx ? 1 : 0;
    const cy = p[1] >= by + hy ? 1 : 0;
    const cz = p[2] >= bz + hz ? 1 : 0;
    // (x << 2) | (y << 1) | z — the format's own octant numbering, the one
    // `makeChildNode` decodes with 0b100 / 0b010 / 0b001.
    const idx = cx * 4 + cy * 2 + cz;
    if ((mask & (1 << idx)) === 0) return depth;

    const first = (cut[o + 1]! << 16) | (cut[o + 2]! << 8) | cut[o + 3]!;
    // Siblings are contiguous, so the child's slot is the run start plus the
    // number of selected octants that sort before this one.
    let below = 0;
    for (let b = 0; b < idx; b++) below += (mask >> b) & 1;
    slot = first + below;

    bx += cx * hx;
    by += cy * hy;
    bz += cz * hz;
    sx = hx;
    sy = hy;
    sz = hz;
    depth++;
  }
  return depth;
}
