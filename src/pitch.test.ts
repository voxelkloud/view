// Task T0: the per-node point pitch, from the tree's dense array all the way to
// the `spacingWorld` uniform the material reads.
//
// Task A2 split the refinement error from the point pitch and wired the split
// into the scheduler. The RENDER side kept reading the level's closed form —
// correct while the two agree, which they do on every octree and on nothing
// else. These tests are written against a tree whose pitch is deliberately NOT
// `s0 / 2 ** level`, because that is the only shape that can tell the two apart.

import { Group, MeshBasicMaterial } from "three/webgpu";
import type { Mesh } from "three/webgpu";
import { describe, expect, it } from "vitest";
import { PointArena } from "./arena.js";
import { ArenaSink } from "./sink-arena.js";
import { PerNodeSink } from "./sink.js";
import { cloudPitchFloor, pitchOf } from "./view.js";
import type { PitchSource } from "./view.js";

const S0 = 36.37;
const closedForm: PitchSource = {
  pointSpacingAt: (l) => S0 / 2 ** l,
};

/** A tileset-shaped tree: one depth, three very different tile errors. */
function perNode(values: readonly number[]): PitchSource & {
  nodeCount: number;
  maxLevel: number;
} {
  return {
    pointSpacingAt: (l) => S0 / 2 ** l,
    nodePointSpacing: Float64Array.from(values),
    nodeCount: values.length,
    maxLevel: 1,
  };
}

describe("pitchOf", () => {
  it("falls back to the closed form when the tree has no array", () => {
    expect(pitchOf(closedForm, 7, 3)).toBeCloseTo(S0 / 8, 12);
  });

  it("prefers the dense array over the level", () => {
    const tree = perNode([12.5, 0.4, 3.25]);
    expect(pitchOf(tree, 0, 1)).toBe(12.5);
    expect(pitchOf(tree, 1, 1)).toBe(0.4);
    expect(pitchOf(tree, 2, 1)).toBe(3.25);
  });

  it("falls back for an entry the driver has not filled yet", () => {
    // A tileset learns a tile's pitch from its decoded points, so the slot is 0
    // until the content lands. 0 would draw invisible points.
    const tree = perNode([12.5, 0, 3.25]);
    expect(pitchOf(tree, 1, 1)).toBeCloseTo(S0 / 2, 12);
    expect(pitchOf(tree, 99, 2)).toBeCloseTo(S0 / 4, 12);
  });
});

describe("cloudPitchFloor", () => {
  it("is pointSpacingAt(maxLevel) on a closed-form tree", () => {
    const h = {
      hierarchy: { ...closedForm, nodeCount: 500, maxLevel: 6 },
      minPitchWorld: S0,
      scannedNodes: 0,
    };
    expect(cloudPitchFloor(h)).toBeCloseTo(S0 / 64, 12);
  });

  it("only ever moves downward as the tree deepens", () => {
    // The invariant `updateCamera` depends on: a floor that can RISE closes a
    // feedback loop through suggestNearFar and latches.
    const hierarchy = { ...closedForm, nodeCount: 10, maxLevel: 4 };
    const h = { hierarchy, minPitchWorld: S0, scannedNodes: 0 };
    const deep = cloudPitchFloor(h);
    // maxLevel going BACKWARDS is not something a tree does, but the floor must
    // not follow it if it did.
    h.hierarchy = { ...closedForm, nodeCount: 10, maxLevel: 2 };
    expect(cloudPitchFloor(h)).toBe(deep);
  });

  it("takes the minimum over the array, not the deepest level", () => {
    // THE POINT: on a tileset the finest tile is not the deepest one.
    const h = {
      hierarchy: perNode([12.5, 0.4, 3.25]),
      minPitchWorld: Number.POSITIVE_INFINITY,
      scannedNodes: 0,
    };
    expect(cloudPitchFloor(h)).toBe(0.4);
  });

  it("scans each node exactly once and folds in later growth", () => {
    const values = [12.5, 3.25];
    const hierarchy = perNode(values);
    const h = {
      hierarchy,
      minPitchWorld: Number.POSITIVE_INFINITY,
      scannedNodes: 0,
    };
    expect(cloudPitchFloor(h)).toBe(3.25);
    expect(h.scannedNodes).toBe(2);

    const grown = perNode([...values, 0.8]);
    const h2 = { ...h, hierarchy: grown };
    expect(cloudPitchFloor(h2)).toBe(0.8);
    expect(h2.scannedNodes).toBe(3);
  });
});

function meshesOf(parent: Group): Mesh[] {
  return parent.children as Mesh[];
}

function decoded(numPoints: number, nodeIndex = 0) {
  return {
    nodeIndex,
    nodeName: `t${nodeIndex}`,
    numPoints,
    positions: new Float32Array(3 * numPoints),
    frame: {
      format: "float32" as const,
      origin: [0, 0, 0] as [number, number, number],
      scale: [1, 1, 1] as [number, number, number],
      originPolicy: "cloud" as const,
      maxPositionError: 0,
    },
    colors: undefined,
    attributes: [],
    attributesByName: new Map(),
    bounds: undefined,
    transferList: [],
    byteLength: 0,
  };
}

describe("PointArena: the pitch a slab is stamped with", () => {
  function makeArena() {
    const parent = new Group();
    const arena = new PointArena(
      parent,
      new MeshBasicMaterial(),
      (l) => S0 / 2 ** l,
      (l) => 4031.7 / 2 ** l,
      { pageSize: 1024, slabCapacity: 64 * 1024 },
    );
    return { arena, parent };
  }

  it("keys on the level when the pitch IS the closed form", () => {
    // The property that keeps every octree format free: same partition, same
    // slab count, same keys as before per-node pitches existed.
    const { arena, parent } = makeArena();
    const a = arena.allocate(3, 1000, S0 / 8)!;
    const b = arena.allocate(3, 1000)!;
    expect(a.key).toBe(3);
    expect(b.key).toBe(3);
    expect(a.slab).toBe(b.slab);
    expect(parent.children).toHaveLength(1);
    expect(meshesOf(parent)[0]!.userData['spacingWorld']).toBeCloseTo(S0 / 8, 12);
  });

  it("splits one level into separate slabs for far-apart pitches", () => {
    const { arena, parent } = makeArena();
    const coarse = arena.allocate(1, 1000, 12.5)!;
    const fine = arena.allocate(1, 1000, 0.4)!;
    expect(coarse.key).not.toBe(fine.key);
    expect(parent.children).toHaveLength(2);
    const stamped = meshesOf(parent)
      .map((m) => m.userData['spacingWorld'] as number)
      .sort((x, y) => x - y);
    expect(stamped[0]).toBeCloseTo(0.4, 12);
    expect(stamped[1]).toBeCloseTo(12.5, 12);
  });

  it("shares one slab within a quarter octave", () => {
    // The bound that keeps a tileset from opening one slab per distinct tile
    // error: 2^(1/4) apart at most, so a point draws within 9% of its pitch.
    const { arena, parent } = makeArena();
    const a = arena.allocate(1, 1000, 4.0)!;
    const b = arena.allocate(1, 1000, 4.1)!;
    expect(a.key).toBe(b.key);
    expect(parent.children).toHaveLength(1);
  });

  it("ignores a non-finite or non-positive pitch", () => {
    const { arena, parent } = makeArena();
    expect(arena.allocate(2, 100, 0)!.key).toBe(2);
    expect(arena.allocate(2, 100, Number.NaN)!.key).toBe(2);
    expect(arena.allocate(2, 100, -1)!.key).toBe(2);
    expect(parent.children).toHaveLength(1);
    expect(meshesOf(parent)[0]!.userData['spacingWorld']).toBeCloseTo(S0 / 4, 12);
  });

  it("frees a bucketed block back into its own slab", () => {
    // `free` looks the slab up by key, not by level: getting that wrong loses
    // the pages silently.
    const { arena } = makeArena();
    const block = arena.allocate(1, 2048, 12.5)!;
    const before = arena.residentBytes;
    arena.free(block);
    const again = arena.allocate(1, 2048, 12.5)!;
    expect(again.start).toBe(block.start);
    expect(arena.residentBytes).toBe(before);
  });
});

describe("the pitch reaches the sink", () => {
  it("ArenaSink stamps the pitch it was handed", () => {
    const parent = new Group();
    const sink = new ArenaSink(
      parent,
      new MeshBasicMaterial(),
      (l) => S0 / 2 ** l,
      (l) => 4031.7 / 2 ** l,
      false,
      { pageSize: 1024, slabCapacity: 64 * 1024 },
    );
    // Level 1's closed form is 18.185; the tile's own pitch is 0.4.
    sink.attach(0, decoded(1000) as never, 0.4, 1);
    expect(parent.children).toHaveLength(1);
    expect(meshesOf(parent)[0]!.userData['spacingWorld']).toBeCloseTo(0.4, 12);
    sink.dispose();
  });

  it("PerNodeSink stamps the pitch it was handed", () => {
    const parent = new Group();
    const sink = new PerNodeSink(parent, new MeshBasicMaterial());
    sink.attach(0, decoded(1000) as never, 0.4, 1);
    expect(meshesOf(parent)[0]!.userData['spacingWorld']).toBeCloseTo(0.4, 12);
    sink.dispose();
  });
});
