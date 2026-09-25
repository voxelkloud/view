// How long the GPU actually spent, which wall clock cannot tell you.
//
// THE MEASUREMENT PROBLEM THIS SOLVES, stated once so nobody re-derives it.
// Every frame-time number this project has ever taken is pinned to the display:
// the Task 10 baseline reads 16.7 ms at five different quality stops, and a
// splat-size sweep taken while writing this file read 16.7 ms at caps of 2, 4,
// 8 and 12 pixels — a 40-fold difference in covered pixels that wall clock
// reported as identical, because every one of them made vsync with room to
// spare. A renderer with headroom and one with none present at the same rate.
//
// So an optimisation inside the frame is unmeasurable from outside it, and the
// usual escape — pile on load until vsync breaks — measures a machine at its
// thermal limit. On the host this was written on, the same configuration read
// 16.7 ms cold and 24.2 ms hot within one run. Timestamp queries sidestep both:
// they time the passes on the GPU's own clock, whether or not the frame had
// room, and they do not care what the fans are doing.
//
// WHY IT IS OFF BY DEFAULT. Reading the results back means mapping a buffer,
// which means waiting for the GPU. Doing that inside the loop that produces the
// numbers is how Fase 0 of the performance plan nearly published a figure that
// included a forced GPU-to-CPU sync and a whole vsync. Here the resolve is
// queued every frame — cheap, no sync — and the readback happens on a cadence,
// never more than one in flight, with the result landing a few frames late.
// Late is fine: nothing steers on it.

/** One pass's timing, in milliseconds. */
export interface GpuPassTiming {
  /** Most recent completed reading. */
  readonly last: number;
  /**
   * Median over the samples kept. QUOTE THIS ONE.
   *
   * Not the mean, and the difference is not pedantry: a handful of frames a
   * second are several times the cost of the rest — a compositor hiccup, a
   * page doing something else, the driver deciding to reclock — and a mean
   * over a hundred samples moves further from those outliers than the effect
   * of most optimisations. Measuring two builds of this shader, the means
   * disagreed by 10% in both directions on scenes whose medians were within
   * 2%.
   */
  readonly p50: number;
  /** Mean, kept because it is what a total-throughput question wants. */
  readonly mean: number;
  /** Smallest seen — the uncontended cost, and an optimistic one. */
  readonly best: number;
  readonly samples: number;
}

export interface GpuTimings {
  /** Clearing depth. The accumulator is cleared by the copy queue, untimed. */
  readonly clear: GpuPassTiming;
  /**
   * The depth and colour passes together, for every cloud in the frame.
   *
   * THE NUMBER THAT MATTERS for anything about the rasteriser: it is the two
   * dispatches that walk every resident point twice, project it, and write its
   * splat into the depth and accumulation buffers.
   */
  readonly points: GpuPassTiming;
  /** The full-screen resolve, EDL included. */
  readonly resolve: GpuPassTiming;
}

/** Slots in the query set. Two per pass: beginning and end. */
export const TIMING_SLOTS = 6;
const CLEAR_AT = 0;
const POINTS_AT = 2;
const RESOLVE_AT = 4;

/** Samples kept per pass. A ring, so a long session cannot grow without end. */
const KEEP = 512;

interface Accumulator {
  last: number;
  best: number;
  sum: number;
  n: number;
  /** The ring. `n` may exceed its length; `at` is where the next one goes. */
  ring: Float64Array;
  at: number;
}

function empty(): Accumulator {
  return { last: 0, best: Infinity, sum: 0, n: 0, ring: new Float64Array(KEEP), at: 0 };
}

function report(a: Accumulator): GpuPassTiming {
  const held = Math.min(a.n, KEEP);
  let p50 = 0;
  if (held > 0) {
    const sorted = Array.from(a.ring.subarray(0, held)).sort((x, y) => x - y);
    p50 = sorted[Math.floor(held * 0.5)] ?? 0;
  }
  return {
    last: a.last,
    p50,
    mean: a.n === 0 ? 0 : a.sum / a.n,
    best: a.best === Infinity ? 0 : a.best,
    samples: a.n,
  };
}

/**
 * Owns the query set and the readback, for one rasteriser.
 *
 * Constructed only when the device granted `timestamp-query`. Every method is
 * safe to call when it was not, because the caller holds `undefined` and the
 * optional chaining does the work — there is no disabled mode to get wrong.
 */
export class GpuTimer {
  private readonly querySet: GPUQuerySet;
  /** Where `resolveQuerySet` writes. Never mapped; it is not mappable. */
  private readonly resolveBuf: GPUBuffer;
  /** The mappable copy. One only, which is what bounds the readbacks. */
  private readonly readBuf: GPUBuffer;
  /** True between asking for a map and getting it. */
  private reading = false;
  private frame = 0;
  private disposed = false;

  private readonly clear = empty();
  private readonly points = empty();
  private readonly resolve = empty();

  /**
   * @param every Frames between readbacks. Thirty by default: the passes do
   *   not change shape from one frame to the next, and each readback is a
   *   buffer map. A bench comparing two builds wants 1, where the real rate
   *   is set by how fast the maps complete rather than by this number — only
   *   one is ever in flight.
   */
  constructor(
    private readonly device: GPUDevice,
    private readonly every = 30,
  ) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: TIMING_SLOTS });
    const bytes = TIMING_SLOTS * 8;
    this.resolveBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /** Timestamp writes for the clear pass, or undefined to leave it untimed. */
  clearWrites(): GPUComputePassTimestampWrites {
    return this.writesAt(CLEAR_AT);
  }
  pointsWrites(): GPUComputePassTimestampWrites {
    return this.writesAt(POINTS_AT);
  }
  resolveWrites(): GPURenderPassTimestampWrites {
    return this.writesAt(RESOLVE_AT);
  }

  private writesAt(at: number): GPUComputePassTimestampWrites {
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: at,
      endOfPassWriteIndex: at + 1,
    };
  }

  /**
   * Queue the resolve, and every `every` frames the copy that will be read.
   *
   * Called with the frame's encoder, after every pass has been recorded and
   * before it is submitted. The resolve itself is a GPU-side command with no
   * synchronisation; only the copy leads anywhere that has to be waited for.
   */
  record(enc: GPUCommandEncoder): void {
    if (this.disposed) return;
    enc.resolveQuerySet(this.querySet, 0, TIMING_SLOTS, this.resolveBuf, 0);
    this.frame++;
    if (this.reading || this.frame % this.every !== 0) return;
    this.reading = true;
    enc.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, TIMING_SLOTS * 8);
  }

  /**
   * Take the readback if one was queued. Call after `submit`.
   *
   * Fire and forget: the map resolves when the GPU gets there, and the numbers
   * land a few frames after the work they describe. Nothing steers on them, so
   * nothing waits.
   */
  poll(): void {
    if (!this.reading || this.disposed) return;
    void this.readBuf
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        if (this.disposed) return;
        // BigInt64Array because a timestamp is nanoseconds since an arbitrary
        // origin and overflows a double's integer range in about 104 days of
        // uptime. The DIFFERENCE is small, so it converts safely; the absolute
        // values do not.
        const raw = new BigInt64Array(this.readBuf.getMappedRange().slice(0));
        this.readBuf.unmap();
        this.reading = false;
        add(this.clear, span(raw, CLEAR_AT));
        add(this.points, span(raw, POINTS_AT));
        add(this.resolve, span(raw, RESOLVE_AT));
      })
      .catch(() => {
        // A device that went away mid-map, or a buffer destroyed under us.
        // Losing a diagnostic is not worth an unhandled rejection.
        this.reading = false;
      });
  }

  get timings(): GpuTimings {
    return {
      clear: report(this.clear),
      points: report(this.points),
      resolve: report(this.resolve),
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.querySet.destroy();
    this.resolveBuf.destroy();
    // Destroying a buffer with a map in flight is legal and rejects the map,
    // which the catch above absorbs.
    this.readBuf.destroy();
  }
}

/** Milliseconds between a begin/end pair, or 0 when the pair is unwritten. */
function span(raw: BigInt64Array, at: number): number {
  const a = raw[at] ?? 0n;
  const b = raw[at + 1] ?? 0n;
  if (b <= a) return 0;
  return Number(b - a) / 1e6;
}

function add(a: Accumulator, ms: number): void {
  // Zero means the pass did not run this frame — a cloud with nothing drawn,
  // or a query the implementation chose not to write. Counting it would drag
  // every mean toward zero and quietly flatter whatever is being measured.
  if (ms <= 0) return;
  a.last = ms;
  a.sum += ms;
  a.ring[a.at] = ms;
  a.at = (a.at + 1) % KEEP;
  a.n++;
  if (ms < a.best) a.best = ms;
}
