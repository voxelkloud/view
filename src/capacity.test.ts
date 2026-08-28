import { describe, expect, it } from "vitest";
import { initialCapacity } from "./capacity.js";

describe("initialCapacity", () => {
  it("takes the budget when the cloud's size is not known", () => {
    expect(initialCapacity(3_000_000, 1.25)).toBe(3_750_000);
    expect(initialCapacity(3_000_000, 1.6)).toBe(4_800_000);
  });

  it("clamps to the cloud when the cloud is smaller than the budget", () => {
    // The panel that started this: 512,359 points opened at a 4M budget, which
    // reserved 5M points of staging — 80 MB — for a cloud that has half a
    // million. Twelve of those on one page came to 1.03 GB.
    expect(initialCapacity(4_000_000, 1.25, 512_359)).toBe(512_359);
    expect(initialCapacity(4_000_000, 1.6, 34_720)).toBe(34_720);
  });

  it("leaves the budget alone when the cloud is bigger than it", () => {
    // The case the reservation was designed for, and it must not regress: a
    // survey larger than any budget still gets the whole budget with its slack,
    // because every point of it is reachable.
    expect(initialCapacity(3_000_000, 1.25, 100_000_000)).toBe(3_750_000);
  });

  it("keeps the slack when the cloud sits just above the budget", () => {
    // 3.5M points against a 3M budget: the slack is what absorbs a camera move
    // pulling in the frontier before the nodes it left go cold, so the clamp
    // must not eat it while there is still cloud above the budget to reach.
    expect(initialCapacity(3_000_000, 1.25, 3_500_000)).toBe(3_500_000);
  });

  it("ignores a count of zero rather than clamping to nothing", () => {
    // A driver that cannot count up front reports 0. Clamped literally, the
    // sink would have no capacity and would refuse every node it was handed —
    // a blank canvas with every counter claiming success.
    expect(initialCapacity(3_000_000, 1.25, 0)).toBe(3_750_000);
    expect(initialCapacity(3_000_000, 1.25, -1)).toBe(3_750_000);
  });
});
