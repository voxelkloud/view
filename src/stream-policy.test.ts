import { describe, expect, it } from "vitest";
import {
  ABORT_OUTSIDE_FRAMES,
  ABORT_STALE_FRAMES,
  MAX_LOAD_ATTEMPTS,
  nextRetryFrame,
  retryDelayFrames,
  shouldAbortFetch,
} from "./stream-policy.js";
import type { AbortPolicy } from "./stream-policy.js";

const BOTH: AbortPolicy = {
  abortOutsideFrustum: true,
  abortSuperseded: true,
  saturated: true,
};
const FRUSTUM_ONLY: AbortPolicy = { ...BOTH, abortSuperseded: false };
const OFF: AbortPolicy = { ...BOTH, abortOutsideFrustum: false, abortSuperseded: false };

describe("shouldAbortFetch", () => {
  it("never cancels a node selected this frame", () => {
    expect(shouldAbortFetch(true, 0, BOTH)).toBe(false);
    expect(shouldAbortFetch(false, 0, BOTH)).toBe(false);
  });

  it("cancels an off-screen node after the hysteresis, unsaturated", () => {
    const quiet = { ...FRUSTUM_ONLY, saturated: false };
    expect(shouldAbortFetch(true, ABORT_OUTSIDE_FRAMES - 1, quiet)).toBe(false);
    expect(shouldAbortFetch(true, ABORT_OUTSIDE_FRAMES, quiet)).toBe(true);
  });

  it("leaves an on-screen node alone however stale, with only the frustum tier", () => {
    expect(shouldAbortFetch(false, 1000, FRUSTUM_ONLY)).toBe(false);
  });

  it("cancels a superseded on-screen node only when saturated and stale", () => {
    const sup: AbortPolicy = { ...BOTH, abortOutsideFrustum: false };
    expect(shouldAbortFetch(false, ABORT_STALE_FRAMES, { ...sup, saturated: false })).toBe(false);
    expect(shouldAbortFetch(false, ABORT_STALE_FRAMES - 1, sup)).toBe(false);
    expect(shouldAbortFetch(false, ABORT_STALE_FRAMES, sup)).toBe(true);
  });

  it("cancels nothing with both tiers off", () => {
    expect(shouldAbortFetch(true, 1000, OFF)).toBe(false);
    expect(shouldAbortFetch(false, 1000, OFF)).toBe(false);
  });

  it("frees the slot sooner off-screen than on", () => {
    expect(ABORT_OUTSIDE_FRAMES).toBeLessThan(ABORT_STALE_FRAMES);
  });
});

describe("retryDelayFrames", () => {
  it("backs off exponentially from half a second", () => {
    expect(retryDelayFrames(1)).toBe(30);
    expect(retryDelayFrames(2)).toBe(120);
    expect(retryDelayFrames(3)).toBe(480);
  });

  it("never returns a sub-frame delay, so a retry can never be a per-frame storm", () => {
    for (let a = 0; a <= MAX_LOAD_ATTEMPTS; a++) {
      expect(retryDelayFrames(a)).toBeGreaterThan(1);
    }
  });

  it("schedules forward from the current frame", () => {
    expect(nextRetryFrame(100, 1)).toBe(130);
    expect(nextRetryFrame(100, 2)).toBe(220);
  });
});
