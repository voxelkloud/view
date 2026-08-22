// @voxelkloud/view — WebGPU point cloud rendering on three.js.
//
// The LOD scheduler is re-exported here for convenience, but it also has its own
// subpath, `@voxelkloud/view/lod`, whose module graph pulls in no three, no DOM
// and no GPU.
export * from "./lod/index.js";
export * from "./profile/index.js";

export {
  createPointMaterial,
  resolvePointMaterialOptions,
  scalarAttributeFor,
} from "./material.js";
export type {
  ColorMode,
  PointCloudMaterial,
  PointMaterialOptions,
  ResolvedPointMaterialOptions,
} from "./material.js";

export { createEdlPipeline, resolveEdlOptions } from "./edl.js";
export type { EdlOptions, EdlPipeline, ResolvedEdlOptions } from "./edl.js";

export { PointCloudObject3D } from "./object.js";
export { pickPoint } from "./pick.js";
export type { PickPointOptions, PickResult } from "./pick.js";
export { PerNodeSink } from "./sink.js";
export type { PointReadback, PointSink } from "./sink.js";
export { ArenaSink } from "./sink-arena.js";
export { PointArena } from "./arena.js";
export type { ArenaBlock, ArenaOptions } from "./arena.js";

export {
  PointCloudView,
  cloudRelativeElevationRange,
  createPointCloudView,
  scalarRangeFor,
} from "./view.js";
export type { PointCloudViewOptions, ViewProfileOptions, ViewStats } from "./view.js";

/** `true` when this environment can create a WebGPU device. Safe in Node. */
export function isWebGPUAvailable(): boolean {
  return (
    typeof navigator !== "undefined" &&
    (navigator as { gpu?: unknown }).gpu !== undefined
  );
}

export const VOXELKLOUD_VIEW_VERSION = "0.0.0";
