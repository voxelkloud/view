import { Group, MeshBasicMaterial } from "three/webgpu";
import { beforeEach, describe, expect, it } from "vitest";
import { PointArena } from "./arena.js";
import type { ArenaBlock } from "./arena.js";

const PAGE = 1024;

function makeArena(options = {}) {
  const parent = new Group();
  const material = new MeshBasicMaterial();
  const arena = new PointArena(
    parent,
    material,
    (l) => 36.37 / 2 ** l,
    (l) => 4031.7 / 2 ** l,
    { pageSize: PAGE, slabCapacity: 64 * PAGE, ...options },
  );
  return { arena, parent };
}

/** Every allocation must be page-aligned, or writeBuffer's 4-byte alignment
 *  requirement is only satisfied by luck. */
function expectAligned(block: ArenaBlock) {
  expect(block.start % PAGE).toBe(0);
  expect((3 * block.start * 4) % 4).toBe(0);
  expect((4 * block.start) % 4).toBe(0);
}

describe("PointArena: allocation", () => {
  let arena: PointArena;
  let parent: Group;
  beforeEach(() => {
    ({ arena, parent } = makeArena());
  });

  it("allocates page-aligned, non-overlapping blocks within a level", () => {
    const blocks: ArenaBlock[] = [];
    for (let i = 0; i < 20; i++) {
      const b = arena.allocate(3, 500 + i * 137)!;
      expect(b).toBeDefined();
      expectAligned(b);
      blocks.push(b);
    }
    // No two blocks in the same slab may overlap.
    const bySlab = new Map<number, ArenaBlock[]>();
    for (const b of blocks) {
      const list = bySlab.get(b.slab) ?? [];
      list.push(b);
      bySlab.set(b.slab, list);
    }
    for (const [, list] of bySlab) {
      list.sort((a, b) => a.start - b.start);
      for (let i = 1; i < list.length; i++) {
        const prev = list[i - 1]!;
        expect(list[i]!.start).toBeGreaterThanOrEqual(
          prev.start + prev.pages * PAGE,
        );
      }
    }
  });

  it("puts different levels in different slabs", () => {
    // Slabs are level-partitioned because the material reads spacing and level
    // as per-object uniforms; mixing levels would force size to become
    // per-point data.
    const a = arena.allocate(2, 1000)!;
    const b = arena.allocate(5, 1000)!;
    expect(a.level).toBe(2);
    expect(b.level).toBe(5);
    expect(arena.slabCount).toBe(2);
    expect(parent.children).toHaveLength(2);
  });

  it("reuses freed pages", () => {
    const a = arena.allocate(1, 2000)!;
    const before = arena.slabCount;
    arena.free(a);
    const b = arena.allocate(1, 2000)!;
    expect(b.start).toBe(a.start);
    expect(arena.slabCount).toBe(before);
  });

  it("coalesces adjacent free runs", () => {
    const a = arena.allocate(1, PAGE)!;
    const b = arena.allocate(1, PAGE)!;
    const c = arena.allocate(1, PAGE)!;
    expect(b.start).toBe(a.start + PAGE);
    expect(c.start).toBe(b.start + PAGE);

    arena.free(a);
    arena.free(c);
    arena.free(b); // bridges the two runs
    // A single 3-page request must now fit where the three 1-page blocks were.
    const big = arena.allocate(1, 3 * PAGE)!;
    expect(big.start).toBe(a.start);
    expect(big.pages).toBe(3);
  });

  it("grows a new slab when the current one is full", () => {
    const cap = 64 * PAGE;
    const first = arena.allocate(1, cap)!;
    expect(first.slab).toBe(0);
    const second = arena.allocate(1, PAGE)!;
    expect(second.slab).toBe(1);
    expect(arena.slabCount).toBe(2);
  });

  it("sizes the first slab to demand rather than always full capacity", () => {
    // autzen's level 0 is 10,833 points; a full 524k slab per level would waste
    // most of it.
    const { arena: small } = makeArena({ slabCapacity: 524_288 });
    small.allocate(0, 10_833);
    // 6.3 MiB positions + 2.1 MiB colour for a full slab would be ~8.4 MB.
    expect(small.residentBytes).toBeLessThan(1_500_000);
  });

  it("honours a node larger than the nominal slab capacity", () => {
    const { arena: tight } = makeArena({ slabCapacity: 2 * PAGE });
    const big = tight.allocate(1, 10 * PAGE)!;
    expect(big).toBeDefined();
    expect(big.count).toBe(10 * PAGE);
  });

  it("refuses a zero-point allocation", () => {
    expect(arena.allocate(1, 0)).toBeUndefined();
  });
});

describe("PointArena: staging and liveness", () => {
  it("copies positions and colours into the block's slots", () => {
    const { arena, parent } = makeArena();
    const block = arena.allocate(2, 3)!;
    const pos = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const col = new Uint8Array([10, 11, 12, 200, 13, 14, 15, 201, 16, 17, 18, 202]);
    arena.stage(block, pos, col, false);
    arena.commit();

    const mesh = parent.children[0] as unknown as { geometry: { attributes: Record<string, { array: ArrayLike<number> }> } };
    const p = mesh.geometry.attributes['pointOffset']!.array;
    const c = mesh.geometry.attributes['color']!.array;
    for (let i = 0; i < 9; i++) expect(p[3 * block.start + i]).toBe(pos[i]);
    for (let i = 0; i < 12; i++) expect(c[4 * block.start + i]).toBe(col[i]);
  });

  it("stamps alpha to 255 when the source carried its own", () => {
    const { arena, parent } = makeArena();
    const block = arena.allocate(2, 2)!;
    // A source alpha of 7 would otherwise shrink the splat to 7/255 of its size,
    // because the material multiplies the diameter by colour alpha.
    arena.stage(
      block,
      new Float32Array(6),
      new Uint8Array([1, 2, 3, 7, 4, 5, 6, 9]),
      true,
    );
    const mesh = parent.children[0] as unknown as { geometry: { attributes: Record<string, { array: ArrayLike<number> }> } };
    const c = mesh.geometry.attributes['color']!.array;
    expect(c[4 * block.start + 3]).toBe(255);
    expect(c[4 * block.start + 7]).toBe(255);
    // The colour channels are untouched.
    expect(c[4 * block.start]).toBe(1);
  });

  it("writes liveness into exactly the block's alpha bytes", () => {
    const { arena, parent } = makeArena();
    const a = arena.allocate(1, PAGE)!;
    const b = arena.allocate(1, PAGE)!;
    arena.stage(a, new Float32Array(3 * PAGE), new Uint8Array(4 * PAGE).fill(255), false);
    arena.stage(b, new Float32Array(3 * PAGE), new Uint8Array(4 * PAGE).fill(255), false);

    arena.setAlpha(a, 0);
    const mesh = parent.children[0] as unknown as { geometry: { attributes: Record<string, { array: ArrayLike<number> }> } };
    const c = mesh.geometry.attributes['color']!.array;
    // a is masked off...
    expect(c[4 * a.start + 3]).toBe(0);
    expect(c[4 * (a.start + PAGE - 1) + 3]).toBe(0);
    // ...and b is untouched.
    expect(c[4 * b.start + 3]).toBe(255);
  });

  it("tracks a high-water mark and draws exactly that many instances", () => {
    const { arena, parent } = makeArena();
    arena.allocate(1, 100);
    const second = arena.allocate(1, 50)!;
    arena.commit();
    const mesh = parent.children[0] as unknown as { geometry: { instanceCount: number } };
    // Everything below the high-water mark is drawn and masked by alpha; there
    // is no sub-range draw available on the instanced path.
    expect(mesh.geometry.instanceCount).toBe(second.start + 50);
    expect(arena.drawnSlots).toBe(second.start + 50);
  });

  it("hides a slab that holds nothing", () => {
    const { arena, parent } = makeArena();
    arena.commit();
    const a = arena.allocate(1, 10)!;
    void a;
    arena.commit();
    const mesh = parent.children[0] as unknown as { visible: boolean };
    expect(mesh.visible).toBe(true);
  });
});

describe("PointArena: draw-call reduction", () => {
  // The whole reason the arena exists.
  it("collapses ~1000 nodes into a handful of draws", () => {
    const { arena, parent } = makeArena({ slabCapacity: 524_288 });
    // A realistic autzen frontier: ~1000 nodes spread over levels 3-7 at a
    // median of ~1450 points each.
    let nodes = 0;
    for (let level = 3; level <= 7; level++) {
      for (let n = 0; n < 200; n++) {
        arena.allocate(level, 1450);
        nodes++;
      }
    }
    expect(nodes).toBe(1000);
    // One draw per slab. Per-node meshes would be 1000.
    expect(arena.slabCount).toBeLessThanOrEqual(12);
    expect(parent.children.length).toBe(arena.slabCount);
  });
});

describe("PointArena: disposal", () => {
  it("removes every mesh and releases its accounting", () => {
    const { arena, parent } = makeArena();
    arena.allocate(1, 5000);
    arena.allocate(4, 5000);
    expect(parent.children.length).toBeGreaterThan(0);
    expect(arena.residentBytes).toBeGreaterThan(0);
    arena.dispose();
    expect(parent.children).toHaveLength(0);
    expect(arena.residentBytes).toBe(0);
    expect(arena.slabCount).toBe(0);
  });
});

describe("PointArena: colourless clouds", () => {
  // LAS point format 1 carries intensity and classification and NO RGB, which
  // is most 3DEP lidar. Refusing those made a 50.7M-point survey load its whole
  // hierarchy, decode every node, and draw nothing.
  it("writes a live white block when the source has no colour", () => {
    const { arena, parent } = makeArena();
    const block = arena.allocate(2, 3)!;
    arena.stage(block, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]), undefined, false);
    arena.commit();

    const mesh = parent.children[0] as unknown as {
      geometry: { attributes: Record<string, { array: ArrayLike<number> }> };
    };
    const c = mesh.geometry.attributes['color']!.array;
    const p = mesh.geometry.attributes['pointOffset']!.array;
    // Positions still land.
    expect(p[3 * block.start]).toBe(1);
    // Alpha is the liveness mask the material multiplies the diameter by, so an
    // unwritten colour block would mean alpha 0 and a cloud that draws nothing.
    for (let i = 0; i < 3; i++) {
      expect(c[4 * (block.start + i) + 3]).toBe(255);
    }
  });

  it("leaves neighbouring blocks untouched when filling", () => {
    const { arena, parent } = makeArena();
    const a = arena.allocate(1, PAGE)!;
    const b = arena.allocate(1, PAGE)!;
    arena.stage(a, new Float32Array(3 * PAGE), new Uint8Array(4 * PAGE).fill(7), false);
    arena.stage(b, new Float32Array(3 * PAGE), undefined, false);
    const mesh = parent.children[0] as unknown as {
      geometry: { attributes: Record<string, { array: ArrayLike<number> }> };
    };
    const c = mesh.geometry.attributes['color']!.array;
    expect(c[4 * a.start]).toBe(7);
    expect(c[4 * b.start]).toBe(255);
  });
});
