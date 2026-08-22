import { describe, expect, it } from "vitest";
import { resolveEdlOptions } from "./edl.js";

/**
 * There is no GPU here, so the shader itself is verified in a browser. What IS
 * checkable offline is the policy around it: the defaults a caller inherits,
 * and the sampling ring, where a duplicated or zero offset would silently
 * weaken the effect rather than fail.
 */
describe("resolveEdlOptions", () => {
  it("defaults to Potree's strength", () => {
    // 1.0 against the same 300x constant, so an `edlStrength` transfers.
    expect(resolveEdlOptions()).toEqual({
      strength: 1,
      radius: 1.4,
      opacity: 1,
    });
  });

  it("keeps a caller's zero rather than treating it as absent", () => {
    // `?? 1` and `|| 1` differ exactly here, and strength 0 is the meaningful
    // "shading off, pass still wired" state used to isolate the composite.
    expect(resolveEdlOptions({ strength: 0 }).strength).toBe(0);
    expect(resolveEdlOptions({ opacity: 0 }).opacity).toBe(0);
  });

  it("takes each field independently", () => {
    expect(resolveEdlOptions({ radius: 2.5 })).toEqual({
      strength: 1,
      radius: 2.5,
      opacity: 1,
    });
  });
});

/**
 * The response formula, as a reference. The shader mirrors it; this pins the
 * PROPERTIES that make it eye-dome lighting rather than an edge filter, so a
 * later "simplification" of the shader has something to contradict.
 */
function response(centre: number, neighbours: readonly number[]): number {
  let sum = 0;
  for (const n of neighbours) sum += Math.max(Math.log2(centre) - Math.log2(n), 0);
  return sum / neighbours.length;
}
const shade = (r: number, strength: number) => Math.exp(-r * 300 * strength);

describe("the EDL response", () => {
  it("is zero on a surface parallel to the image plane", () => {
    expect(response(100, [100, 100, 100, 100])).toBe(0);
    expect(shade(0, 1)).toBe(1);
  });

  it("darkens a pixel whose neighbours are CLOSER, and only that one", () => {
    // A pixel sitting in a crease, with the surface rising around it: shaded.
    // This is the "eye dome" — concavities go dark, ridges stay bright, which
    // is what makes it read as ambient occlusion rather than as an outline.
    expect(response(110, [100, 100, 100, 100])).toBeGreaterThan(0);
    // The pixel on the near side of the same edge: NOT shaded. Taking the
    // absolute difference instead would darken both sides and draw the cloud
    // as a wireframe of its own silhouettes.
    expect(response(90, [100, 100, 100, 100])).toBe(0);
  });

  it("is scale-invariant, which is why the metric is logarithmic", () => {
    // The same 10% step 100 m away and 10 km away must shade identically, or a
    // cloud with real extent only shades near the near plane.
    const near = response(110, [100, 100, 100, 100]);
    const far = response(11000, [10000, 10000, 10000, 10000]);
    expect(far).toBeCloseTo(near, 12);
  });

  it("cannot brighten: shade is bounded by 1", () => {
    for (const r of [0, 0.001, 0.01, 0.5]) {
      expect(shade(r, 1)).toBeLessThanOrEqual(1);
      expect(shade(r, 1)).toBeGreaterThan(0);
    }
  });

  it("saturates fast, which is what `radius` is really for", () => {
    // The scale the 300 sets, measured rather than assumed: with the WHOLE ring
    // a step behind the centre, a 0.2% depth difference is already mid-shade
    // and 1% is effectively black.
    const ring = (pct: number) => shade(response(100 * pct, new Array(8).fill(100)), 1);
    expect(ring(1.002)).toBeCloseTo(0.42, 2);
    expect(ring(1.005)).toBeCloseTo(0.12, 2);
    expect(ring(1.01)).toBeLessThan(0.02);

    // So on a frontier whose splats are smaller than the sampling ring, every
    // pixel sees a real depth discontinuity and the render goes dark. That is
    // Potree's behaviour too, not a defect — the knobs to reach for are a
    // smaller `radius` or a lower `strength`, and both are exposed.
    expect(shade(response(101, new Array(8).fill(100)), 0.15)).toBeGreaterThan(0.4);
  });
});
