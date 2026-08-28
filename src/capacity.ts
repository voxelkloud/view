/**
 * How much point staging a sink reserves before it has seen a single node.
 *
 * Shared by the compute sink and the WebGL 2 one because they got the same
 * thing wrong in the same place, with different constants: each reserved
 * `pointBudget * slack` points of GPU buffer AND of CPU mirror — 16 bytes a
 * point, 20 with a scalar lane — at construction, from a budget the caller sets
 * for the whole view rather than for this cloud.
 *
 * That is right when the budget is the only thing known. It stops being right
 * the moment the cloud's own point count is in hand, which at `addCloud` it
 * always is: a sink serves ONE cloud, residency cannot exceed what that cloud
 * contains, and capacity above it can never be filled by anything at all.
 *
 * The measurement that produced this: twelve panels of a point cloud paper page
 * (`demo/litept`), holding between 34,720 and 512,359 points each, opened at
 * the 4M-per-canvas budget that would be unremarkable for a single viewer. The
 * page drew 3,065,580 points and held 1.03 GB of JS heap, against 92 MB for the
 * three.js page it was rebuilt from. Clamped here, the same page holds 178 MB
 * and no other metric moves — the ceiling was reserving, never binding.
 *
 * NOT a replacement for the growth path. Both sinks double on demand and must
 * keep doing so: `cloudPoints` bounds what a cloud can ever need, but an octree
 * whose nodes overlap in residency, a re-attach after eviction, or a caller who
 * passes nothing at all can still arrive at a full allocator. What this removes
 * is the reservation nobody asked for, not the safety net.
 */
export function initialCapacity(
  pointBudget: number,
  slack: number,
  cloudPoints?: number,
): number {
  const wanted = Math.ceil(pointBudget * slack);
  // `> 0` rather than `!== undefined`: a driver that cannot count its points up
  // front reports 0, and 0 would otherwise clamp the sink to nothing and refuse
  // every node it was ever handed.
  if (cloudPoints === undefined || cloudPoints <= 0) return wanted;
  return Math.min(wanted, cloudPoints);
}
