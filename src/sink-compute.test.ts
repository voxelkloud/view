import { describe, expect, it } from "vitest";
import { BlockAllocator } from "./sink-compute.js";

/**
 * The allocator behind the compute sink. These are the cases that corrupt
 * silently rather than throw: a free run that fails to merge fragments the
 * arena until an allocation that SHOULD fit does not, and a wrong `start` hands
 * out a range overlapping a live node — neither of which any frame counter
 * would report.
 */
describe("BlockAllocator", () => {
  it("bumps while there is room and refuses past the cap", () => {
    const a = new BlockAllocator(100);
    expect(a.allocate(40)).toBe(0);
    expect(a.allocate(40)).toBe(40);
    expect(a.used).toBe(80);
    expect(a.allocate(40)).toBe(-1);
    expect(a.used).toBe(80);
  });

  it("reuses a freed run rather than growing", () => {
    const a = new BlockAllocator(100);
    a.allocate(30);
    const mid = a.allocate(30);
    a.allocate(30);
    a.release(mid, 30);
    expect(a.freePoints).toBe(30);
    expect(a.allocate(30)).toBe(mid);
    expect(a.used).toBe(90);
    expect(a.freeRunCount).toBe(0);
  });

  it("splits a run larger than the request and keeps the remainder", () => {
    const a = new BlockAllocator(100);
    a.allocate(10);
    const mid = a.allocate(50);
    a.allocate(10);
    a.release(mid, 50);
    expect(a.allocate(20)).toBe(mid);
    expect(a.freePoints).toBe(30);
    expect(a.allocate(30)).toBe(mid + 20);
    expect(a.freeRunCount).toBe(0);
  });

  it("coalesces with the run before, the run after, and both at once", () => {
    const a = new BlockAllocator(100);
    for (let i = 0; i < 4; i++) a.allocate(10);
    a.release(0, 10);
    a.release(10, 10); // merges FORWARD into the one before
    expect(a.freeRunCount).toBe(1);
    expect(a.freePoints).toBe(20);

    const b = new BlockAllocator(100);
    for (let i = 0; i < 4; i++) b.allocate(10);
    b.release(30, 10);
    b.release(20, 10); // merges BACKWARD into the one after
    expect(b.freeRunCount).toBe(1);
    expect(b.freePoints).toBe(20);

    const c = new BlockAllocator(100);
    for (let i = 0; i < 5; i++) c.allocate(10);
    c.release(0, 10);
    c.release(20, 10);
    expect(c.freeRunCount).toBe(2);
    c.release(10, 10); // fills the hole between them: three runs become one
    expect(c.freeRunCount).toBe(1);
    expect(c.freePoints).toBe(30);
    // And the merged run is usable AS ONE, which is the whole point of merging.
    expect(c.allocate(30)).toBe(0);
    expect(c.freeRunCount).toBe(0);
  });

  it("never hands out a range that overlaps a live one", () => {
    // Deterministic churn: allocate a varying size each round, retire the
    // oldest every third round, and assert after every allocation that the new
    // range intersects no live one. Overlap is the failure that would silently
    // draw one node's points with another node's colour.
    const a = new BlockAllocator(10_000);
    const live: { start: number; end: number }[] = [];
    const sizes = [7, 13, 5, 21, 3, 11, 17, 2];
    for (let round = 0; round < 200; round++) {
      const n = sizes[round % sizes.length]!;
      const start = a.allocate(n);
      expect(start).toBeGreaterThanOrEqual(0);
      for (const r of live) {
        expect(start >= r.end || start + n <= r.start).toBe(true);
      }
      live.push({ start, end: start + n });
      // Steady state after a short warm-up: retire one per round, so the live
      // set stops growing and the high-water mark is a statement about REUSE
      // rather than about how fast the test allocates.
      if (live.length > 20) {
        const victim = live.shift()!;
        a.release(victim.start, victim.end - victim.start);
      }
    }
    // At most 20 live blocks of at most 21 points is 420 live. A free list that
    // reuses keeps the high-water near that; one that leaks marches toward the
    // ~2000 points these 200 rounds allocated in total. The bound is 3x live,
    // which is loose enough for first-fit fragmentation and tight enough that a
    // leak fails it.
    expect(a.used).toBeLessThan(3 * 20 * 21);
  });

  it("raises the ceiling only upward", () => {
    const a = new BlockAllocator(50);
    expect(a.allocate(60)).toBe(-1);
    a.setCapacity(100);
    expect(a.allocate(60)).toBe(0);
    a.setCapacity(10);
    expect(a.capacity).toBe(100);
  });
});
