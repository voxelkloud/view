import { describe, expect, it } from "vitest";
import {
  applySortOrder,
  buildSplatNodeGeometry,
  cameraPositionSignature,
  gatherVisibleGeometry,
  hasGaussianAttributes,
  rotationColumns,
  sortBackToFront,
  sourceSplatVectorToZUp,
  type SplatNodeGeometry,
} from "./sink-splat-geometry.js";

function makeAttr(array: Float32Array) {
  return { array };
}

function makeGaussianData(overrides: {
  readonly positions?: Float32Array;
  readonly scale?: readonly [number, number, number][];
  readonly rot?: readonly [number, number, number, number][];
  readonly opacity?: readonly number[];
  readonly colors?: Uint8Array;
} = {}) {
  const n = overrides.scale?.length ?? overrides.rot?.length ?? overrides.opacity?.length ?? 2;
  const positions = overrides.positions ?? new Float32Array(n * 3).fill(0);
  const scale = overrides.scale ?? Array.from({ length: n }, () => [0.1, 0.05, 0.02] as const);
  const rot = overrides.rot ?? Array.from({ length: n }, () => [1, 0, 0, 0] as const);
  const opacity = overrides.opacity ?? Array.from({ length: n }, () => 1);
  const attributesByName = new Map<string, { array: Float32Array }>([
    ["scale_0", makeAttr(Float32Array.from(scale.map((s) => s[0])))],
    ["scale_1", makeAttr(Float32Array.from(scale.map((s) => s[1])))],
    ["scale_2", makeAttr(Float32Array.from(scale.map((s) => s[2])))],
    ["rot_0", makeAttr(Float32Array.from(rot.map((r) => r[0])))],
    ["rot_1", makeAttr(Float32Array.from(rot.map((r) => r[1])))],
    ["rot_2", makeAttr(Float32Array.from(rot.map((r) => r[2])))],
    ["rot_3", makeAttr(Float32Array.from(rot.map((r) => r[3])))],
    ["opacity", makeAttr(Float32Array.from(opacity))],
  ]);
  return {
    numPoints: n,
    positions,
    colors:
      overrides.colors === undefined ? undefined : { array: overrides.colors, itemSize: 4 },
    attributesByName,
  } as unknown as Parameters<typeof buildSplatNodeGeometry>[0];
}

describe("hasGaussianAttributes", () => {
  it("accepts a cloud with every required extra", () => {
    expect(hasGaussianAttributes(makeGaussianData())).toBe(true);
  });

  it("rejects a cloud missing even one extra", () => {
    const data = makeGaussianData();
    (data.attributesByName as Map<string, unknown>).delete("rot_3");
    expect(hasGaussianAttributes(data)).toBe(false);
  });

  it("rejects a plain LiDAR cloud with no extras at all", () => {
    const data = {
      attributesByName: new Map(),
    } as unknown as Parameters<typeof hasGaussianAttributes>[0];
    expect(hasGaussianAttributes(data)).toBe(false);
  });
});

describe("rotationColumns", () => {
  it("the identity quaternion gives the identity basis", () => {
    const [u, v, w] = rotationColumns(1, 0, 0, 0);
    expect(u).toEqual([1, 0, 0]);
    expect(v).toEqual([0, 1, 0]);
    expect(w).toEqual([0, 0, 1]);
  });

  it("normalizes an un-normalized quaternion instead of trusting the input", () => {
    // (2,0,0,0) has the same rotation as (1,0,0,0) once normalized.
    const [u, v, w] = rotationColumns(2, 0, 0, 0);
    expect(u[0]).toBeCloseTo(1);
    expect(v[1]).toBeCloseTo(1);
    expect(w[2]).toBeCloseTo(1);
  });

  it("a 90° rotation about Z sends X to Y", () => {
    const s = Math.SQRT1_2;
    const [u] = rotationColumns(s, 0, 0, s); // w=cos(45°), z=sin(45°)
    expect(u[0]).toBeCloseTo(0, 5);
    expect(u[1]).toBeCloseTo(1, 5);
  });
});

describe("sourceSplatVectorToZUp", () => {
  it("maps (x,y,z) to (x,z,-y) — Y-up-visual to Z-up", () => {
    expect(sourceSplatVectorToZUp([1, 2, 3])).toEqual([1, 3, -2]);
  });
});

describe("buildSplatNodeGeometry", () => {
  it("picks the two LARGEST scale axes as the billboard basis", () => {
    // scale_1 is the smallest here, so axisU/axisV should come from the
    // identity rotation's X and Z columns (order 0, 2), not Y.
    const data = makeGaussianData({ scale: [[0.3, 0.01, 0.2]], rot: [[1, 0, 0, 0]] });
    const g = buildSplatNodeGeometry(data, 1);
    // X column [1,0,0] → Z-up (1,0,0); scaled by 0.3.
    expect(g.axisU[0]).toBeCloseTo(0.3);
    expect(g.axisU[1]).toBeCloseTo(0);
    expect(g.axisU[2]).toBeCloseTo(0);
    // Z column [0,0,1] → Z-up (0,1,0) via sourceSplatVectorToZUp; scaled by 0.2.
    expect(g.axisV[0]).toBeCloseTo(0);
    expect(g.axisV[1]).toBeCloseTo(0.2);
    expect(g.axisV[2]).toBeCloseTo(0);
  });

  it("floors scale at a small positive epsilon so a degenerate axis never zeroes the billboard", () => {
    const data = makeGaussianData({ scale: [[0, 0, 0]], rot: [[1, 0, 0, 0]] });
    const g = buildSplatNodeGeometry(data, 1);
    expect(Math.hypot(g.axisU[0]!, g.axisU[1]!, g.axisU[2]!)).toBeGreaterThan(0);
  });

  it("clamps opacity into 0..1 rather than trusting the source", () => {
    const data = makeGaussianData({ opacity: [1.5, -0.2] });
    const g = buildSplatNodeGeometry(data, 1);
    expect(g.opacities[0]).toBe(1);
    expect(g.opacities[1]).toBe(0);
  });

  it("falls back to a flat grey when the cloud carries no colour attribute", () => {
    const data = makeGaussianData({ opacity: [1] });
    const g = buildSplatNodeGeometry(data, 1);
    expect(g.colors[0]).toBe(200);
    expect(g.colors[3]).toBe(255); // alpha stays opaque; opacity lives apart
  });

  it("rejects non-float32 positions instead of drawing quantized noise", () => {
    const data = makeGaussianData();
    (data as { positions: unknown }).positions = new Int32Array(6);
    expect(() => buildSplatNodeGeometry(data, 1)).toThrow(/float32/);
  });

  it("throws instead of drawing an all-zero axis when an extra dimension is missing", () => {
    const data = makeGaussianData();
    (data.attributesByName as Map<string, unknown>).delete("scale_2");
    expect(() => buildSplatNodeGeometry(data, 1)).toThrow(/scale_2/);
  });
});

describe("gatherVisibleGeometry / sortBackToFront / applySortOrder", () => {
  function block(count: number, zStart: number): { geometry: SplatNodeGeometry } {
    const centers = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) centers[i * 3 + 2] = zStart + i;
    return {
      geometry: {
        count,
        centers,
        axisU: new Float32Array(count * 3),
        axisV: new Float32Array(count * 3),
        colors: new Uint8Array(count * 4).fill(1),
        opacities: new Float32Array(count).fill(1),
        bytes: 0,
      },
    };
  }

  it("gathers multiple blocks into one flat, concatenated set", () => {
    const g = gatherVisibleGeometry([block(2, 0), block(3, 10)]);
    expect(g.count).toBe(5);
    expect(Array.from(g.centers.filter((_, i) => i % 3 === 2))).toEqual([0, 1, 10, 11, 12]);
  });

  it("sorts back-to-front: the farthest point from the eye comes first", () => {
    // Eye at origin looking down +z (forward = [0,0,1]); farther z is farther away.
    const centers = new Float32Array([0, 0, 0, 0, 0, 5, 0, 0, 2]);
    const order = sortBackToFront(centers, 3, [0, 0, 0], [0, 0, 1]);
    expect(Array.from(order)).toEqual([1, 2, 0]); // z=5 farthest, then z=2, then z=0
  });

  it("applySortOrder reorders every field consistently, not just one", () => {
    const g = gatherVisibleGeometry([block(3, 0)]);
    g.colors.set([9, 9, 9, 9], 0); // tag index 0's colour so we can trace it
    const order = Uint32Array.from([2, 0, 1]);
    const sorted = applySortOrder(g, order);
    // dst 1 came from src 0 — the tagged colour should have moved with it.
    expect(sorted.colors[4]).toBe(9);
    expect(sorted.centers[3 + 2]).toBe(0); // src 0's z was 0
  });
});

describe("cameraPositionSignature", () => {
  it("is stable for the same rounded position", () => {
    expect(cameraPositionSignature([1.001, 2.001, 3.001], 100)).toBe(
      cameraPositionSignature([1.004, 2.004, 3.004], 100),
    );
  });

  it("changes once the position moves past the rounding step", () => {
    expect(cameraPositionSignature([1, 2, 3], 100)).not.toBe(
      cameraPositionSignature([1.02, 2, 3], 100),
    );
  });
});
