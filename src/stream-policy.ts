// WHICH FETCH TO CANCEL, AND WHEN TO TRY AGAIN.
//
// PURE. No three, no DOM, no GPU, no clock — the frame counter is passed in.
// The streaming loop in `view.ts` needs a device and a canvas and is therefore
// untestable; these two decisions are the whole of its policy, so they live
// here where a test can reach them.
//
// The two cancellation rules answer different questions and must not be
// collapsed into one:
//
//  - OUTSIDE THE FRUSTUM is a FACT. The node is not on screen, so its bytes
//    cannot change a pixel of this frame or of any frame until the camera
//    turns back. Cancelling costs nothing but the bytes already spent, and the
//    slot it frees goes to a node that IS on screen.
//  - SUPERSEDED is a GUESS. The node is still in front of the camera; the
//    scheduler merely declined it this frame, usually because it sits at the
//    error threshold and the budget is tight. Those nodes come back. See
//    `abortSuperseded` on the view for the measurement that keeps this tier
//    off by default and gated on saturation.

/**
 * Frames a node must stay out of the SELECTION before a superseded-tier cancel.
 * Eight is ~130 ms at 60 Hz — longer than damping's settle, shorter than any
 * deliberate camera move.
 */
export const ABORT_STALE_FRAMES = 8;

/**
 * Frames a node must stay outside the FRUSTUM before its fetch is cancelled.
 *
 * Two, not zero. A box that straddles a frustum plane flips containment on
 * sub-pixel camera motion, and a cancel-on-first-exit thrashes it: abort,
 * re-enter, re-dispatch, abort — each cycle throwing away a partly paid
 * response. Two frames is below anything a viewer can perceive and above the
 * jitter.
 *
 * Not eight, which is what the superseded tier waits: this one is answering a
 * fact rather than a guess, and the whole point is to free the slot while the
 * node on screen still needs it.
 */
export const ABORT_OUTSIDE_FRAMES = 2;

/**
 * Times a node's payload is re-attempted before it is given up on.
 *
 * Three, because the failures worth retrying are transient by nature — a 503
 * from a CDN edge, a dropped connection, a sink with no room until the next
 * eviction — and the ones that are not (404, a corrupt payload, a missing
 * decompressor) fail identically all three times and cost two extra requests
 * once, not once per frame.
 */
export const MAX_LOAD_ATTEMPTS = 3;

export interface AbortPolicy {
  /** Cancel fetches for nodes the frustum no longer contains. */
  readonly abortOutsideFrustum: boolean;
  /** Cancel fetches the scheduler has stopped selecting. */
  readonly abortSuperseded: boolean;
  /**
   * Whether the fetch queue is full.
   *
   * Gates the SUPERSEDED tier only. A fetch nothing is competing with is
   * cheaper to finish than to redo — its bytes are already partly paid for and
   * the node may well be selected again — so a guess is only worth acting on
   * when the slot it holds is a slot something else is waiting for. The
   * frustum tier ignores this: an off-screen node is wasted bandwidth whether
   * or not anything is queued behind it, and on a thin uplink the bandwidth is
   * the scarce thing, not the slot.
   */
  readonly saturated: boolean;
}

/**
 * Whether to cancel one in-flight fetch.
 *
 * @param outside the node's box is fully outside the frustum THIS frame.
 * @param staleFrames frames since the node was last in the selection. 0 means
 *   it was selected this frame, which never cancels.
 */
export function shouldAbortFetch(
  outside: boolean,
  staleFrames: number,
  p: AbortPolicy,
): boolean {
  if (staleFrames <= 0) return false;
  if (p.abortOutsideFrustum && outside && staleFrames >= ABORT_OUTSIDE_FRAMES) {
    return true;
  }
  return p.abortSuperseded && p.saturated && staleFrames >= ABORT_STALE_FRAMES;
}

/**
 * Frames to wait before attempt number `attempts + 1`: 30, 120, 480 — half a
 * second, two seconds, eight, at 60 Hz.
 *
 * Exponential because the two failure modes want opposite things. A CDN edge
 * that just 503'd wants to be left alone for longer than a frame; a sink that
 * refused for want of room is usually fixed by the very next eviction. Starting
 * at half a second serves the second and quadrupling serves the first, and
 * neither becomes the per-frame retry storm this replaced — the reference
 * re-requests a failed node on EVERY frame it stays selected, which is 60
 * requests a second for a URL that is going to 404 all day.
 */
export function retryDelayFrames(attempts: number): number {
  return 30 * 4 ** Math.max(0, attempts - 1);
}

/** Frame at which attempt `attempts + 1` becomes eligible. */
export function nextRetryFrame(frame: number, attempts: number): number {
  return frame + retryDelayFrames(attempts);
}
