import { describe, expect, it } from "vitest";
import {
  AUTO_CEILING_INDEX,
  DEFAULT_QUALITY_INDEX,
  QUALITY_LEVELS,
  QualityController,
  ceilingForOptions,
  clampIndex,
  initialQualityIndex,
} from "./quality.js";

/**
 * Feed the controller a steady frame rate and report where it settled.
 *
 * `intervalMs` is what the display actually delivered, so 16.7 is a renderer
 * making vsync on a 60 Hz panel and 33.3 is one missing every other one.
 */
function drive(
  controller: QualityController,
  intervalMs: number | ((index: number) => number),
  frames: number,
  startAt = 0,
): number {
  let now = startAt;
  for (let i = 0; i < frames; i++) {
    const dt =
      typeof intervalMs === "number" ? intervalMs : intervalMs(controller.current);
    now += dt;
    controller.sample(dt, now);
  }
  return now;
}

describe("QUALITY_LEVELS", () => {
  it("is ordered, so a step is always a step in one direction", () => {
    // The controller only ever moves by one and never compares two rungs, so
    // an out-of-order ladder would make "down" cost more than "up" somewhere
    // in the middle and the loop would climb a hill it meant to descend.
    for (let i = 1; i < QUALITY_LEVELS.length; i++) {
      const lower = QUALITY_LEVELS[i - 1]!;
      const higher = QUALITY_LEVELS[i]!;
      expect(higher.pointBudget).toBeGreaterThan(lower.pointBudget);
      expect(higher.targetScreenError).toBeLessThan(lower.targetScreenError);
      expect(higher.renderScale).toBeGreaterThanOrEqual(lower.renderScale);
    }
  });

  it("puts the library's own defaults on the default rung", () => {
    // If these drift apart, a caller who never asked for auto and a caller who
    // did would get different pictures on the same machine for no stated
    // reason. The defaults live in `lod/select.ts`.
    const level = QUALITY_LEVELS[DEFAULT_QUALITY_INDEX]!;
    expect(level.pointBudget).toBe(3_000_000);
    expect(level.targetScreenError).toBe(1.35);
    expect(level.renderScale).toBe(1);
  });
});

describe("initialQualityIndex", () => {
  const base = { rasterizer: "compute" as const, pixelRatio: 1, pixels: 2_000_000 };

  it("starts an unremarkable machine on the default rung", () => {
    expect(initialQualityIndex({ ...base, cores: 4 })).toBe(DEFAULT_QUALITY_INDEX);
  });

  it("puts the instanced fallback at the bottom whatever else is true", () => {
    // That arm was measured at 7.5 fps and 656 ms of INP. Sixteen cores do not
    // change what it is.
    expect(
      initialQualityIndex({ ...base, rasterizer: "instanced", cores: 16, memoryGb: 8 }),
    ).toBe(0);
  });

  it("treats a software rasteriser as the bottom, not as a slow GPU", () => {
    expect(
      initialQualityIndex({ ...base, cores: 16, vendor: "google swiftshader" }),
    ).toBe(0);
  });

  it("steps down for a big canvas and up for a small one", () => {
    const big = initialQualityIndex({ ...base, cores: 4, pixels: 12_000_000 });
    const small = initialQualityIndex({ ...base, cores: 4, pixels: 900_000 });
    expect(big).toBeLessThan(DEFAULT_QUALITY_INDEX);
    expect(small).toBeGreaterThan(DEFAULT_QUALITY_INDEX);
  });

  it("never leaves the ladder however the signals pile up", () => {
    const worst = initialQualityIndex({
      ...base,
      cores: 1,
      memoryGb: 0.5,
      pixels: 30_000_000,
      architecture: "integrated",
    });
    const best = initialQualityIndex({ ...base, cores: 32, memoryGb: 8, pixels: 500_000 });
    expect(worst).toBe(0);
    expect(best).toBeLessThanOrEqual(QUALITY_LEVELS.length - 1);
    expect(best).toBeGreaterThanOrEqual(0);
  });
});

describe("ceilingForOptions", () => {
  it("stops one rung above the defaults when nothing was asked for", () => {
    expect(ceilingForOptions()).toBe(AUTO_CEILING_INDEX);
    expect(ceilingForOptions({})).toBe(AUTO_CEILING_INDEX);
  });

  it("never exceeds a budget the caller named", () => {
    const ceiling = ceilingForOptions({ pointBudget: 1_500_000 });
    expect(QUALITY_LEVELS[ceiling]!.pointBudget).toBeLessThanOrEqual(1_500_000);
    expect(QUALITY_LEVELS[ceiling + 1]!.pointBudget).toBeGreaterThan(1_500_000);
  });

  it("never exceeds a screen error the caller named", () => {
    // A LOWER target is finer, so the ceiling is the last rung whose target is
    // still at or above what was asked for.
    const ceiling = ceilingForOptions({ targetScreenError: 2.0 });
    expect(QUALITY_LEVELS[ceiling]!.targetScreenError).toBeGreaterThanOrEqual(2.0);
  });

  it("honours the deprecated spelling of the target", () => {
    expect(ceilingForOptions({ targetPixelSpacing: 2.0 })).toBe(
      ceilingForOptions({ targetScreenError: 2.0 }),
    );
  });
});

describe("QualityController", () => {
  it("holds while the renderer is making vsync", () => {
    const c = new QualityController({ startIndex: 2, maxIndex: 2 });
    drive(c, 16.7, 400);
    expect(c.current).toBe(2);
  });

  it("steps down when the typical frame misses the display", () => {
    const c = new QualityController({ startIndex: 3 });
    // 33 ms on a 60 Hz panel is every frame going to the second vsync.
    drive(c, (i) => (i >= 3 ? 33.4 : 16.7), 600);
    expect(c.current).toBeLessThan(3);
  });

  it("keeps stepping down until the frames come back", () => {
    const c = new QualityController({ startIndex: 4 });
    // Nothing helps: the device cannot hold the rate at any rung. It must end
    // at the bottom rather than stall halfway or run off the end.
    drive(c, 50, 4000);
    expect(c.current).toBe(0);
  });

  it("climbs when there is headroom, but not before the delay", () => {
    const c = new QualityController({ startIndex: 1, maxIndex: 3 });
    // A clean second is not enough — the upshift delay is five.
    drive(c, 16.7, 60);
    expect(c.current).toBe(1);
    drive(c, 16.7, 2000, 1_000);
    expect(c.current).toBe(3);
  });

  it("never climbs past the ceiling", () => {
    const c = new QualityController({ startIndex: 0, maxIndex: 2 });
    drive(c, 8, 6000);
    expect(c.current).toBe(2);
  });

  it("settles instead of oscillating across a marginal rung", () => {
    // THE FAILURE THIS CLASS EXISTS TO AVOID. The device holds 60 Hz at rung 2
    // and misses at rung 3. A controller with symmetric patience walks up,
    // walks down, and repeats for ever — and a picture that changes resolution
    // every few seconds reads as broken in a way a permanently soft one does
    // not.
    const c = new QualityController({ startIndex: 2, maxIndex: 4 });
    let now = 0;
    const changes: number[] = [];
    for (let i = 0; i < 40_000; i++) {
      const dt = c.current >= 3 ? 33.4 : 16.7;
      now += dt;
      const d = c.sample(dt, now);
      if (d.changed) changes.push(now);
    }
    // It may try the bad rung a few times — that is the price of ever
    // improving — but the attempts must get rarer, not keep their pace.
    const firstHalf = changes.filter((t) => t < now / 2).length;
    const secondHalf = changes.length - firstHalf;
    expect(secondHalf).toBeLessThan(firstHalf);
    expect(c.current).toBe(2);
    // And over eleven minutes of this it must not have thrashed.
    expect(changes.length).toBeLessThan(20);
  });

  it("ignores a gap, because a paused tab is not a slow frame", () => {
    const c = new QualityController({ startIndex: 3 });
    let now = 0;
    for (let i = 0; i < 400; i++) {
      // One frame in twenty is a two-second stall: a backgrounded tab, a
      // breakpoint, a full collection. Downshifting for those would punish a
      // viewer for being paused.
      const dt = i % 20 === 0 ? 2_000 : 16.7;
      now += dt;
      c.sample(dt, now);
    }
    // At or above where it started. Not EQUAL to it: with the stalls discarded
    // every remaining frame made vsync, so climbing is the right answer and
    // pinning this to 3 would be asserting that the gaps were counted after
    // all.
    expect(c.current).toBeGreaterThanOrEqual(3);
  });

  it("ignores a nonsense interval", () => {
    const c = new QualityController({ startIndex: 2 });
    for (let i = 0; i < 500; i++) c.sample(0, i);
    for (let i = 0; i < 500; i++) c.sample(Number.NaN, i);
    for (let i = 0; i < 500; i++) c.sample(-5, i);
    expect(c.current).toBe(2);
  });

  it("aims at the display it actually has, not at 60 Hz", () => {
    // A 120 Hz panel presents every 8.3 ms. A renderer holding 16.7 there is
    // missing half the frames, and a controller hard-coded to 60 would call
    // that perfect and keep climbing.
    const c = new QualityController({ startIndex: 3 });
    drive(c, 8.3, 200);
    expect(c.estimatedRefreshMs).toBeLessThan(10);
    drive(c, (i) => (i >= 3 ? 25 : 8.3), 600);
    expect(c.current).toBeLessThan(3);
  });

  it("starts measuring afresh after a manual pick", () => {
    const c = new QualityController({ startIndex: 4 });
    drive(c, 40, 500);
    expect(c.current).toBeLessThan(4);
    c.set(4);
    expect(c.current).toBe(4);
    // And the old history must not immediately undo the choice: the next
    // decision has to be earned by new samples.
    const d = c.sample(40, 1_000_000);
    expect(d.changed).toBe(false);
  });

  it("clamps a forced pick to the ceiling and to the ladder", () => {
    const c = new QualityController({ startIndex: 0, maxIndex: 2 });
    c.set(99);
    expect(c.current).toBe(2);
    c.set(-4);
    expect(c.current).toBe(0);
  });
});

describe("clampIndex", () => {
  it("keeps a bad number on the ladder", () => {
    expect(clampIndex(Number.NaN)).toBe(DEFAULT_QUALITY_INDEX);
    expect(clampIndex(-1)).toBe(0);
    expect(clampIndex(99)).toBe(QUALITY_LEVELS.length - 1);
    expect(clampIndex(1.4)).toBe(1);
  });
});
