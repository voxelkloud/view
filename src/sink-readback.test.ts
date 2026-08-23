import type { DecodedPointData } from "@voxelkloud/format-potree";
import { Group, MeshBasicMaterial } from "three/webgpu";
import { describe, expect, it } from "vitest";
import { ArenaSink } from "./sink-arena.js";
import { PerNodeSink } from "./sink.js";

function makePointData(): DecodedPointData {
  return {
    nodeIndex: 12,
    nodeName: "r0",
    numPoints: 2,
    positions: new Float32Array([1, 2, 3, 4, 5, 6]),
    frame: {
      format: "float32",
      origin: [100, 200, 300],
      scale: [1, 1, 1],
      originPolicy: "cloud",
      maxPositionError: 0,
    },
    colors: {
      array: new Uint8Array([10, 11, 12, 255, 20, 21, 22, 255]),
      gpuFormat: "unorm8x4",
      maxValue: 255,
      declaredMax: 255,
      shift: 0,
    },
    attributes: [],
    attributesByName: new Map(),
    bounds: {
      min: [101, 202, 303],
      max: [104, 205, 306],
    },
  } as unknown as DecodedPointData;
}

describe("PerNodeSink readback", () => {
  it("returns the resident point arrays for a node", () => {
    const sink = new PerNodeSink(new Group(), new MeshBasicMaterial());
    sink.attach(12, makePointData(), 1, 3);

    const read = sink.readPoints(12);
    expect(read).toBeDefined();
    expect(read?.start).toBe(0);
    expect(read?.count).toBe(2);
    expect(Array.from(read!.positions)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(read!.colors!)).toEqual([
      10, 11, 12, 255, 20, 21, 22, 255,
    ]);
  });

  it("drops readback once the node is detached", () => {
    const sink = new PerNodeSink(new Group(), new MeshBasicMaterial());
    sink.attach(12, makePointData(), 1, 3);
    sink.detach(12);
    expect(sink.readPoints(12)).toBeUndefined();
  });
});

describe("ArenaSink readback", () => {
  it("returns the slab arrays for a resident block", () => {
    const sink = new ArenaSink(
      new Group(),
      new MeshBasicMaterial(),
      (level) => 36.37 / 2 ** level,
      (level) => 4031.7 / 2 ** level,
      false,
    );
    sink.attach(7, makePointData(), 1, 3);

    const read = sink.readPoints(7);
    expect(read).toBeDefined();
    expect(read?.start).toBe(0);
    expect(read?.count).toBe(2);
    expect(Array.from(read!.positions.slice(0, 6))).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(read!.colors!.slice(0, 8))).toEqual([
      10, 11, 12, 255, 20, 21, 22, 255,
    ]);
  });

  it("drops readback once the block is detached", () => {
    const sink = new ArenaSink(
      new Group(),
      new MeshBasicMaterial(),
      (level) => 36.37 / 2 ** level,
      (level) => 4031.7 / 2 ** level,
      false,
    );
    sink.attach(7, makePointData(), 1, 3);
    sink.detach(7);
    expect(sink.readPoints(7)).toBeUndefined();
  });
});
