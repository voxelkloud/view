// The `@voxelkloud/view/lod` subpath.
//
// This module graph imports NOTHING at runtime — no three, no DOM, no GPU. That
// is asserted by a test, and it is what makes the scheduler importable under
// SSR, inside a worker and in vitest, and what makes it the exact seam a wasm
// kernel cuts along.

export {
  Containment,
  FRUSTUM_PLANE_FLOATS,
  classifyAabb,
  extractFrustumPlanes,
  intersectsAabb,
} from "./frustum.js";
export type { DepthRange } from "./frustum.js";

export {
  estimateFragments,
  projectionFactorOrthographic,
  projectionFactorPerspective,
  screenPixelRadius,
  screenErrorPx,
  screenSpacingPx,
  suggestNearFar,
} from "./metric.js";
export type { NearFar, NearFarOptions } from "./metric.js";

export { heapPop, heapPush } from "./heap.js";
export type { HeapArrays } from "./heap.js";

export {
  DEFAULT_SCREEN_ERROR,
  createLodScratch,
  createLodSelection,
  ensureLodCapacity,
  resolveExpansions,
  resolveLodOptions,
  selectVisible,
} from "./select.js";
export type {
  LodCameraState,
  LodKernels,
  LodOptions,
  LodScratch,
  LodSelection,
  LodTreeView,
  ResolvedLodOptions,
} from "./select.js";
