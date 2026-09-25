// Automatic quality, the way a video player does it.
//
// THE PROBLEM THIS SOLVES is that one set of defaults cannot serve both ends of
// the hardware this library runs on. The numbers in `plano-performance-vs-potree-core.md`
// were all taken on a machine that holds 59.9 fps at a 3M budget; the same
// settings on an integrated GPU at a 2x device pixel ratio are drawing four
// times the fragments with a fraction of the fill rate. Shipping for the weak
// device wastes the strong one, and shipping for the strong one makes the weak
// one unusable. A video player settled this argument years ago: measure, and
// move.
//
// WHY A LADDER AND NOT FOUR INDEPENDENT KNOBS. The knobs interact, and two of
// the interactions are already measured in this repo:
//
//   - `targetScreenError` is INERT until the budget has room. On autzen at a
//     0.25 framing the 3M default binds at a target of 1.0 and below, so every
//     target from 1.0 to 0.25 selects the same 2,999,551 points. A controller
//     that lowered the target first would move a knob that does nothing, see
//     no improvement, and lower it again.
//   - Shrinking the SELECTION during motion changed INP not at all, while
//     shrinking the whole budget to 1M moved it from 320 ms to 112. The lever
//     is what is RESIDENT and drawn, not what is selected.
//
// So the levels below are coherent settings, ordered, and the controller moves
// one step at a time. That also makes the behaviour explicable in a UI: a user
// can be told which level they are on, which is the whole reason anyone trusts
// an automatic setting.
//
// WHAT IS MEASURED, and why it is not frame time. Wall clock is pinned to the
// display: the Task 10 baseline reports 16.7 ms at every quality stop, because
// every stop made vsync. A renderer with headroom and one with none report the
// same number, so frame time cannot tell them apart. What CAN is the interval
// between presented frames — once the renderer stops making vsync the interval
// steps to the next multiple of the refresh period, and that step is visible in
// the distribution long before it is visible to a user.

/** One rung. Absolute settings, not multipliers: see {@link QUALITY_LEVELS}. */
export interface QualityLevel {
  /** Shown to a user. Short, because it goes in a menu. */
  readonly name: string;
  /**
   * Fraction of the device pixel ratio the caller asked for.
   *
   * THE FRAGMENT LEVER, and the first one to move on an integrated GPU: the
   * compute rasteriser writes up to `(2 * maxPixelSize + 1)^2` atomics per
   * point per pass, so cost scales with pixels and not with the canvas's CSS
   * box. Dropping a 2x ratio to 1.4x is half the fragments for a difference
   * most people do not see on a point cloud, where there are no text edges to
   * soften.
   */
  readonly renderScale: number;
  /** Points drawn per frame. The vertex and fill lever together. */
  readonly pointBudget: number;
  /** Projected geometric error in device pixels. Higher is coarser. */
  readonly targetScreenError: number;
}

// EYE-DOME LIGHTING IS NOT ON THIS LADDER, and leaving it off was a decision
// rather than an omission.
//
// It looks like an obvious thing to drop — it is fragment work, and fragment
// work is what the render scale exists to cut. But it is also a SHADING MODE
// the user turned on, sitting behind a switch in the inspector labelled
// "Sombreamento (EDL)". An automatic setting that reaches over and turns that
// switch off has stopped adjusting quality and started overruling the person.
// A video player lowers the resolution; it does not remove the subtitles.
//
// The engineering agrees with the manners. On the compute path EDL is folded
// into the resolve pass that runs either way, so switching it off saves a
// handful of texture reads per pixel and not a pass; the render scale cuts the
// same fragments and every other fragment with them.

/**
 * The ladder, coarsest first.
 *
 * The budgets are not round numbers for their own sake. 3,000,000 is the
 * library default and the point every other level is placed against; 1,000,000
 * is the budget measured to bring INP from 320 ms to 112 on a scripted orbit;
 * 6,000,000 is roughly where autzen stops having more detail to give at the
 * default target, so above it the budget is spending on nothing.
 *
 * The screen errors go the other way for the same reason they exist: 1.35 px is
 * the calibrated point-octree default, and a level that raises the budget
 * without lowering the target would be buying points the selector will not ask
 * for.
 */
export const QUALITY_LEVELS: readonly QualityLevel[] = [
  { name: "minimum", renderScale: 0.6, pointBudget: 750_000, targetScreenError: 2.7 },
  { name: "low", renderScale: 0.75, pointBudget: 1_500_000, targetScreenError: 2.0 },
  { name: "medium", renderScale: 1.0, pointBudget: 3_000_000, targetScreenError: 1.35 },
  { name: "high", renderScale: 1.0, pointBudget: 6_000_000, targetScreenError: 1.0 },
  { name: "ultra", renderScale: 1.0, pointBudget: 12_000_000, targetScreenError: 0.7 },
];

/** The rung a device with nothing remarkable about it starts on. */
export const DEFAULT_QUALITY_INDEX = 2;

/** What the probe can learn about a machine before the first frame is drawn. */
export interface DeviceProfile {
  /** Which rasteriser won. Compute is WebGPU; instanced is the slow fallback. */
  readonly rasterizer: "compute" | "points" | "instanced";
  /** `navigator.hardwareConcurrency`, or undefined where it is not exposed. */
  readonly cores?: number | undefined;
  /** `navigator.deviceMemory` in GiB. Chrome only, and coarse (0.25..8). */
  readonly memoryGb?: number | undefined;
  /** The ratio the page asked for, before any scaling. */
  readonly pixelRatio: number;
  /** Drawing buffer size in device pixels at that ratio. */
  readonly pixels: number;
  /** `adapter.info.architecture`, lowercased. Chrome exposes it; others do not. */
  readonly architecture?: string | undefined;
  /** `adapter.info.vendor`, lowercased. */
  readonly vendor?: string | undefined;
}

/**
 * Where to START, from what can be known before anything is drawn.
 *
 * A GUESS, and deliberately a conservative one. Everything this function reads
 * is a proxy: core count is not fill rate, `deviceMemory` is quantised to
 * powers of two and capped at 8 on every machine above it, and the adapter
 * string is absent outside Chrome. Being wrong upward costs a user several
 * seconds of a janky first impression before the loop pulls it back; being
 * wrong downward costs them a slightly soft picture for four seconds. So this
 * leans down, and lets the measurement earn the way up.
 *
 * The one signal here that is not a proxy is the rasteriser. `"instanced"`
 * means neither the compute path nor the WebGL 2 points path was reachable,
 * and that arm was measured at 656 ms of INP and 7.5 fps against 72 ms and
 * 59.9 for the other two. That is not a device to be optimistic about.
 */
export function initialQualityIndex(device: DeviceProfile): number {
  if (device.rasterizer === "instanced") return 0;

  let index = DEFAULT_QUALITY_INDEX;

  // Integrated graphics, by the only name a browser will tell us. Chrome
  // reports "integrated" here for Intel HD/Iris/UHD and for Apple silicon's
  // shared-memory GPU alike — which lumps a very fast GPU in with a very slow
  // one, so it is worth exactly one step and not two.
  if (device.architecture !== undefined && /integrated|swiftshader|llvmpipe/.test(device.architecture)) {
    index -= 1;
  }
  // A software rasteriser is not a GPU. It reports itself through the vendor
  // string on the browsers that expose one, and nothing it does will hold a
  // frame rate.
  if (device.vendor !== undefined && /software|swiftshader|microsoft basic/.test(device.vendor)) {
    return 0;
  }
  // Two cores is a phone or a throttled laptop. Eight or more is not proof of
  // a good GPU, but with everything else neutral it is the best evidence
  // available that this is a workstation.
  if (device.cores !== undefined) {
    if (device.cores <= 2) index -= 1;
    else if (device.cores >= 8) index += 1;
  }
  // Chrome caps this at 8 however much is installed, so `>= 8` is "lots" and
  // not "exactly eight". Below 4 GiB the resident cache is the constraint
  // before the GPU is.
  if (device.memoryGb !== undefined && device.memoryGb < 4) index -= 1;

  // THE FRAGMENT COUNT, which is the one thing here measured in the units that
  // actually cost. A 4K canvas at a 2x ratio is 33 million fragments a pass;
  // the same page on a 1440x900 laptop at 1x is 1.3 million. No core count
  // compensates for a 25x difference in the stage that dominates.
  if (device.pixels > 8_000_000) index -= 1;
  else if (device.pixels < 1_500_000) index += 1;

  return clampIndex(index);
}

export function clampIndex(index: number): number {
  if (!Number.isFinite(index)) return DEFAULT_QUALITY_INDEX;
  return Math.max(0, Math.min(QUALITY_LEVELS.length - 1, Math.round(index)));
}

export interface QualityControllerOptions {
  /**
   * The highest rung auto may choose.
   *
   * It is the caller's explicit `lod` settings, resolved to a rung by
   * {@link ceilingForOptions}. A caller who asked for a 3M budget did so for a
   * reason, and an automatic setting that quietly doubles it is not automatic,
   * it is disobedient.
   */
  readonly maxIndex?: number;
  /** Where to start. Usually {@link initialQualityIndex}. */
  readonly startIndex?: number;
  /** Frame intervals kept for the decision. Default 90, about 1.5 s at 60 Hz. */
  readonly window?: number;
}

/** What one {@link QualityController.sample} decided. */
export interface QualityDecision {
  /** The rung now in force. */
  readonly index: number;
  /** True only on the sample that moved, so the caller can apply it once. */
  readonly changed: boolean;
  /** Why it moved, for the diagnostics panel. */
  readonly reason: "down" | "up" | "hold";
}

/**
 * Longest interval a sample may have and still count, in milliseconds.
 *
 * Above this it is not a slow frame, it is a gap: a backgrounded tab, a
 * breakpoint, a garbage collection of the whole heap, an alert. Feeding those
 * to the controller would downshift a viewer for being paused.
 */
const MAX_VALID_INTERVAL_MS = 400;

/** Frames discarded after a change, so the settle is never measured. */
const SETTLE_FRAMES = 20;

/**
 * Drives the ladder from measured frame intervals.
 *
 * ASYMMETRIC ON PURPOSE, and this is the part every naive version gets wrong.
 * Going down is cheap and must be quick: the user is already suffering, and a
 * step down ends it. Going up is a gamble with the thing that was just fixed,
 * so it is slow, it needs a longer clean run than a step down needs a dirty
 * one, and it gets harder every time it fails. Without that asymmetry the
 * controller finds the rung where the device is marginal and then oscillates
 * across it for ever, which is worse than either neighbour — a picture that
 * changes resolution every two seconds reads as broken in a way that a
 * permanently soft one does not.
 */
export class QualityController {
  private readonly intervals: number[] = [];
  private readonly maxIndex: number;
  private readonly windowSize: number;
  private index: number;
  private settle = SETTLE_FRAMES;
  private lastChangeAt = 0;
  /**
   * How long a clean run the next upshift needs, in milliseconds.
   *
   * Doubles every time an upshift is followed by a downshift, so a device that
   * is marginal at the next rung stops being asked about it. This is the
   * memory that turns oscillation into a single overshoot.
   */
  private upshiftDelay = 5_000;
  private lastUpshiftAt = 0;
  private lastUpshiftIndex = -1;
  /** Smallest interval seen, the estimate of the display's refresh period. */
  private refreshMs = 16.7;

  constructor(options: QualityControllerOptions = {}) {
    this.maxIndex = clampIndex(options.maxIndex ?? QUALITY_LEVELS.length - 1);
    this.index = Math.min(
      clampIndex(options.startIndex ?? DEFAULT_QUALITY_INDEX),
      this.maxIndex,
    );
    this.windowSize = Math.max(20, options.window ?? 90);
  }

  get current(): number {
    return this.index;
  }

  get level(): QualityLevel {
    return QUALITY_LEVELS[this.index]!;
  }

  /** The refresh period the controller believes it is aiming at. */
  get estimatedRefreshMs(): number {
    return this.refreshMs;
  }

  /** Force a rung and stop adapting from the old history. */
  set(index: number): void {
    this.index = Math.min(clampIndex(index), this.maxIndex);
    this.intervals.length = 0;
    this.settle = SETTLE_FRAMES;
  }

  /**
   * Offer one presented-frame interval.
   *
   * `now` is passed rather than read so the whole controller is testable
   * without a clock, which matters more here than usual: every rule in it is
   * about time, and a test that cannot control time can only assert the easy
   * half.
   */
  sample(intervalMs: number, now: number): QualityDecision {
    const hold: QualityDecision = { index: this.index, changed: false, reason: "hold" };

    if (!(intervalMs > 0) || intervalMs > MAX_VALID_INTERVAL_MS) return hold;

    // The refresh estimate tracks the FASTEST frames, because those are the
    // ones the display allowed rather than the ones the renderer managed. It
    // relaxes slowly upward so a viewer that never once makes vsync eventually
    // stops comparing itself against a rate it has no evidence for.
    if (intervalMs < this.refreshMs) this.refreshMs += (intervalMs - this.refreshMs) * 0.25;
    else this.refreshMs += (intervalMs - this.refreshMs) * 0.001;
    this.refreshMs = Math.min(Math.max(this.refreshMs, 6), 34);

    if (this.settle > 0) {
      this.settle--;
      return hold;
    }

    this.intervals.push(intervalMs);
    if (this.intervals.length > this.windowSize) this.intervals.shift();
    if (this.intervals.length < this.windowSize) return hold;

    const sorted = [...this.intervals].sort((a, b) => a - b);
    const p50 = sorted[Math.floor(sorted.length * 0.5)]!;
    const p95 = sorted[Math.floor(sorted.length * 0.95)]!;

    // DOWN: the typical frame is missing the display. At 1.35x the refresh
    // period, more than a third of frames have gone to the next vsync — which
    // on a 60 Hz panel is the difference between 60 fps and 45, and is the
    // point at which a drag stops feeling attached to the pointer.
    if (p50 > this.refreshMs * 1.35 && this.index > 0) {
      // An upshift that turned bad is the strongest evidence there is about
      // this device, so it costs the next one twice the patience.
      if (this.lastUpshiftIndex === this.index) {
        this.upshiftDelay = Math.min(this.upshiftDelay * 2, 120_000);
      }
      return this.move(this.index - 1, now, "down");
    }

    // UP: nearly every frame made the display, for long enough that it was not
    // just a still camera. p95 rather than p50, because the question is whether
    // there is room for MORE work and an occasional miss says there is not.
    if (
      this.index < this.maxIndex &&
      p95 < this.refreshMs * 1.1 &&
      now - this.lastChangeAt > this.upshiftDelay
    ) {
      this.lastUpshiftAt = now;
      this.lastUpshiftIndex = this.index + 1;
      return this.move(this.index + 1, now, "up");
    }

    return hold;
  }

  private move(index: number, now: number, reason: "down" | "up"): QualityDecision {
    this.index = index;
    this.lastChangeAt = now;
    this.intervals.length = 0;
    this.settle = SETTLE_FRAMES;
    return { index, changed: true, reason };
  }
}

/**
 * How high auto may climb when the caller named no `lod` at all.
 *
 * One rung above the library defaults, not the top of the ladder. The rung
 * above this one is a 12M budget, and a budget is not free the way a render
 * scale is: points have to be FETCHED. On the uplinks this library actually
 * runs over, letting an automatic setting quadruple the bytes a page pulls —
 * on the strength of a frame-rate measurement, which says nothing about the
 * network — would be trading a problem the user can see for one they cannot.
 *
 * `ultra` stays reachable by asking for it, from the ladder or from the menu.
 * Nobody gets there by accident.
 */
export const AUTO_CEILING_INDEX = 3;

/**
 * The highest rung that does not exceed what the caller explicitly asked for.
 *
 * Only the settings they NAMED count, and naming one is taken as the whole
 * answer: a caller who passed `pointBudget: 3_000_000` did so for a reason,
 * and an automatic setting that quietly doubles it is not automatic, it is
 * disobedient. Auto still moves DOWN from there freely, which is the half of
 * the behaviour nobody objects to.
 */
export function ceilingForOptions(lod?: {
  readonly pointBudget?: number | undefined;
  readonly targetScreenError?: number | undefined;
  readonly targetPixelSpacing?: number | undefined;
}): number {
  const budget = lod?.pointBudget;
  const error = lod?.targetScreenError ?? lod?.targetPixelSpacing;
  if (budget === undefined && error === undefined) return AUTO_CEILING_INDEX;

  let ceiling = 0;
  for (let i = 0; i < QUALITY_LEVELS.length; i++) {
    const level = QUALITY_LEVELS[i]!;
    if (budget !== undefined && level.pointBudget > budget) break;
    // A LOWER target is finer, so a level whose target is below what was asked
    // for is spending more than the caller wanted.
    if (error !== undefined && level.targetScreenError < error) break;
    ceiling = i;
  }
  return ceiling;
}

/**
 * Turn a level name or index into an index on the ladder.
 *
 * A name that is not on the ladder falls back to the default rung rather than
 * throwing: this is reached from a URL, a saved view and a `localStorage` key,
 * and none of those are worth a blank canvas when an old build wrote a name a
 * new one no longer has.
 */
export function resolveQualityIndex(quality: string | number): number {
  if (typeof quality === "number") return clampIndex(quality);
  const found = QUALITY_LEVELS.findIndex((l) => l.name === quality);
  return found < 0 ? DEFAULT_QUALITY_INDEX : found;
}
