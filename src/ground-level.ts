/**
 * Where the ground is UNDER THE WHOLE CLOUD, in a survey that ships with noise.
 *
 * `tightBoundingBox.min[2]` is the lowest point in the FILE, and on an airborne
 * survey that point is routinely garbage. A NOAA topobathy tile of Morro Bay
 * declares a floor of −333.3 m where the sea bed is at −8.4 m and the water
 * surface at −0.5 m: 32,332 class-7 returns (0.4% of the tile) sit under the
 * water column, and two of them define the header. Anything that plants
 * geometry at "the cloud's floor" — a basemap plane most of all — therefore
 * lands a third of a kilometre below the survey, and reads as a georeferencing
 * bug rather than as the bad returns it actually is.
 *
 * So: a coarse histogram of Z, filled as nodes stream in, read as a low
 * PERCENTILE instead of a minimum. The tail that DEFINES a minimum is exactly
 * what a percentile shrugs off — on the tile above, p0.2% is −6.35 m against a
 * true bathymetric floor of −8.4 m, close enough that imagery placed there
 * sits under the survey rather than through it.
 *
 * A histogram and not a sorted sample because the input is a stream: nodes
 * arrive over seconds, the answer is read at any moment, and a node may arrive
 * again after an eviction. Buckets absorb all three for a fixed 4 KB per cloud.
 *
 * And it counts THE SURVEY, not the file: a return the file itself labels
 * class 7 or 18 is excluded before it reaches a bucket. That costs nothing —
 * the class is already decoded whenever the cloud has one — and it is what
 * lets the upper tail mean something. On the Morro Bay tile the high-noise
 * band is 1.45% of the points, seven times the tail this drops and CONTINUOUS
 * with the terrain below it, so no percentile and no gap-detection can find
 * its underside; excluding the labelled noise moves the p99.8 ceiling from
 * 302.2 m to 116.1 m, which is Morro Rock's shoulder. The floor does not move.
 *
 * NOT {@link GroundIndex}, which answers "how high is the ground HERE" from the
 * resident cut for a caller driving over it. This is one number for the cloud,
 * and it is deliberately cheap enough to keep up to date for every cloud open.
 */

/**
 * How much of the cloud is allowed to be below the ground — and, read from the
 * other end, above the sky. The same number serves both, because the two tails
 * are the same kind of thing: the returns that define a header extent.
 *
 * 0.2% clears the sub-ground noise measured on the NOAA tiles (class 7 is 0.44%
 * of the points and only part of it falls below the sea bed) without climbing
 * into real terrain: raising it to 0.5% already lifts the estimate by 3.6 m,
 * and lowering it to 0.05% puts it back in the tail on the sparser tile. It is
 * deliberately an UNDER-estimate — imagery a metre low still reads as ground,
 * imagery a metre high cuts through the survey.
 */
export const GROUND_TAIL = 0.002;

/**
 * How much of the cloud has to have arrived before the answer is given out.
 *
 * The estimate is not merely rough while the cloud streams — it is WRONG, and
 * wrong in the direction that matters. A COPC root is a uniform subsample, so
 * it carries the noise at the cloud's own rate: the root of the Morro Bay tile
 * is 41,002 points, of which ~160 sit under the sea bed, and the 0.2% cut over
 * it lands at −181 m. It only reaches −7.6 m once the first level has arrived
 * and the sample is a few hundred thousand.
 *
 * So the answer is WITHHELD until then rather than served and corrected. A
 * caller that placed geometry on an early estimate would have to move it a
 * second later, and a basemap that lands in the air and then slides down is a
 * worse bug than the one this class exists to fix — it looks like the viewer
 * cannot decide where the survey is.
 *
 * Capped by the cloud's own size, so a survey smaller than this still answers:
 * once every point is in there is nothing further to wait for.
 */
const SETTLE_SAMPLE = 200_000;

/** ASPRS "low point (noise)" and "high noise" — the two codes a file uses to
 *  say, itself, that a return is not part of the survey. */
const LOW_NOISE = 7;
const HIGH_NOISE = 18;

/** How finely the Z range is divided. 1024 buckets is 4 KB per cloud, and
 *  0.65 m across Morro Bay's 670 m range — finer than an imagery pixel. */
const BUCKETS = 1024;

/**
 * A streaming Z histogram over one cloud, in the frame its positions are in.
 *
 * CLOUD-RELATIVE, never absolute CRS: that is the frame a decoded node's
 * `positions` carry under the default `origin: "cloud"` policy, and converting
 * every point to absolute just to bucket it would spend a subtraction per point
 * to arrive in the same bucket.
 */
export class GroundLevel {
  private readonly counts = new Uint32Array(BUCKETS);
  private readonly min: number;
  /** Buckets per unit. `0` for a flat cloud, which short-circuits both calls. */
  private readonly scale: number;
  private total = 0;

  /** How many samples make this cloud's answer final. See {@link SETTLE_SAMPLE}. */
  private readonly enough: number;

  /**
   * `min` and `max` are the cloud-relative Z extent — which is exactly what
   * {@link cloudRelativeElevationRange} already computes for the ramp.
   * `pointCount` is the cloud's declared size, and only caps the wait.
   */
  constructor(min: number, max: number, pointCount: number) {
    this.min = min;
    const span = max - min;
    this.scale = span > 0 ? BUCKETS / span : 0;
    this.enough = Math.max(1, Math.min(SETTLE_SAMPLE, pointCount));
  }

  /**
   * Whether {@link get} will answer.
   *
   * Exposed separately so a caller can tell "not yet" from "this cloud has no
   * ground" without calling twice.
   */
  get settled(): boolean {
    return this.total >= this.enough;
  }

  /**
   * Fold one decoded node in. `positions` is `3 * numPoints` float32, Z last.
   *
   * `classes`, when the cloud has any, is one ASPRS code per point in whatever
   * width the reader emitted — read numerically, so a `Uint8Array` and the
   * widened `Float32Array` lane both work. Codes 7 and 18 are DROPPED, and
   * dropped from `total` too: a point that is not counted must not dilute the
   * fraction either, or the tail it defines drifts with how much noise the file
   * happens to carry.
   *
   * A node evicted and streamed again is counted twice. Deliberate: tracking
   * which indices are already folded in would cost a set per cloud to shift a
   * percentile by the weight of one node.
   */
  add(positions: Float32Array, numPoints: number, classes?: ArrayLike<number>): void {
    if (this.scale === 0) {
      this.total += numPoints;
      return;
    }
    const { counts, min, scale } = this;
    const last = BUCKETS - 1;
    for (let i = 0; i < numPoints; i++) {
      if (classes !== undefined) {
        const code = classes[i]!;
        if (code === LOW_NOISE || code === HIGH_NOISE) continue;
      }
      const b = Math.floor((positions[i * 3 + 2]! - min) * scale);
      counts[b < 0 ? 0 : b > last ? last : b]!++;
      this.total++;
    }
  }

  /**
   * The ground, cloud-relative — or `undefined` while the cloud has not
   * streamed in far enough for the answer to be FINAL. Not "rough": see
   * {@link SETTLE_SAMPLE} for why an early answer is worse than none.
   *
   * The bucket's LOWER edge, not its centre: half a bucket of deliberate
   * under-estimate, for the reason {@link GROUND_TAIL} gives.
   */
  get(fraction = GROUND_TAIL): number | undefined {
    if (!this.settled) return undefined;
    if (this.scale === 0) return this.min;
    let budget = this.total * fraction;
    for (let b = 0; b < BUCKETS; b++) {
      budget -= this.counts[b]!;
      if (budget < 0) return this.min + b / this.scale;
    }
    return undefined;
  }
}
