import { readFileSync } from "node:fs";
import { parsePointCloudSource } from "@voxelkloud/format-potree";
import { describe, expect, it } from "vitest";
import { cloudRelativeElevationRange } from "./view.js";

const URLS = {
  base: "https://x/",
  metadata: "https://x/metadata.json",
  hierarchy: "https://x/hierarchy.bin",
  octree: "https://x/octree.bin",
};

function sourceFrom(path: string) {
  return parsePointCloudSource(
    JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>,
    URLS,
  );
}

describe("cloudRelativeElevationRange", () => {
  it("rebases the tight Z extent onto the cloud origin", () => {
    const source = sourceFrom(
      new URL(
        "./__fixtures__/autzen.metadata.json",
        import.meta.url,
      ).pathname,
    );
    const [lo, hi] = cloudRelativeElevationRange(source);
    const originZ = source.metadata.boundingBox.min[2];
    const tight = source.tightBoundingBox;

    // The frame pointOffset.z is in, not the absolute CRS one.
    expect(lo).toBeCloseTo(tight.min[2] - originZ, 6);
    expect(hi).toBeCloseTo(tight.max[2] - originZ, 6);
    expect(lo).toBeCloseTo(0, 6);
    expect(hi).toBeGreaterThan(0);
    // The bug this pins: passing the ABSOLUTE range makes every relative z fall
    // below the domain, so t clamps to 0 and the cloud renders flat.
    expect(hi).toBeLessThan(tight.min[2]);
  });

  it("spans the ramp instead of collapsing it", () => {
    const source = sourceFrom(
      new URL(
        "./__fixtures__/autzen.metadata.json",
        import.meta.url,
      ).pathname,
    );
    const [lo, hi] = cloudRelativeElevationRange(source);
    const t = (z: number, a: number, b: number) =>
      Math.max(0, Math.min(1, (z - a) / Math.max(b - a, 1e-9)));
    const mid = (lo + hi) / 2;

    expect(t(lo, lo, hi)).toBeCloseTo(0, 6);
    expect(t(mid, lo, hi)).toBeCloseTo(0.5, 6);
    expect(t(hi, lo, hi)).toBeCloseTo(1, 6);

    // The old behaviour, for contrast: every point pinned to the first stop.
    const tight = source.tightBoundingBox;
    expect(t(mid, tight.min[2], tight.max[2])).toBe(0);
  });
});
