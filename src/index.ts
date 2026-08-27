// @voxelkloud/view — WebGPU point cloud rendering on three.js.
//
// The LOD scheduler is re-exported here for convenience, but it also has its own
// subpath, `@voxelkloud/view/lod`, whose module graph pulls in no three, no DOM
// and no GPU.
export * from "./lod/index.js";
export * from "./profile/index.js";

// The option helpers come from `material-options.js`, whose module graph is
// plain arithmetic. `createPointMaterial` and `createEdlPipeline` do NOT: both
// build on `three/webgpu`, three's WebGPU build, which is 357 kB gzipped
// against 115 kB for the core. Re-exported here as VALUES they put a static
// edge from this entry point into that build, and every consumer downloaded it
// — including the ones rendering through raw WebGL 2, which is most of them.
// `sideEffects: false` did not save them: three itself makes no such promise,
// so the bundler kept the whole graph.
//
// They live on subpaths now: `@voxelkloud/view/material` and
// `@voxelkloud/view/edl`. Types stay here, because types cost nothing.
export { resolvePointMaterialOptions, scalarAttributeFor } from "./material-options.js";
export type {
  ColorMode,
  PointMaterialOptions,
  ResolvedPointMaterialOptions,
} from "./material-options.js";
export type { PointCloudMaterial } from "./material.js";
export type { EdlOptions, EdlPipeline, ResolvedEdlOptions } from "./edl.js";

export { OctreeCut } from "./cut.js";
export { BlockAllocator, ComputeRasterizer, ComputeSink } from "./sink-compute.js";
export type { ComputeSinkOptions } from "./sink-compute.js";
export { PointCloudObject3D } from "./object.js";
export { loadModelLayer } from "./model.js";
export {
  BVH_NODE_STRIDE,
  BVH_TRI_STRIDE,
  buildTriangleBvh,
  clusterDeviation,
  nearestOnBvh,
  raycastBvh,
  solveAlignment,
  distanceToBvh,
  pointBoxDistanceSq,
  pointTriangleDistanceSq,
  trianglesFromObject,
} from "./deviation.js";
export type { Alignment, DeviationCluster, TriangleBvh } from "./deviation.js";
export type { ModelLayer } from "./model.js";
export { MESH_LOC, quantisedMeshLayout } from "./mesh-layout.js";
export type { MeshVertexLayout } from "./mesh-layout.js";
export { pickPoint } from "./pick.js";
export type { PickPointOptions, PickResult } from "./pick.js";
export { PerNodeSink } from "./sink.js";
export type { PointReadback, PointSink } from "./sink.js";
export { ArenaSink } from "./sink-arena.js";
export { GroundIndex } from "./ground.js";
export type { GroundIndexOptions, GroundNormal, GroundSample } from "./ground.js";
export { GroundLevel } from "./ground-level.js";
export { PointArena } from "./arena.js";
export { createReplaceScratch, filterReplacedParents } from "./replace.js";
export type { ReplaceScratch, ReplaceTreeView } from "./replace.js";
export type { ArenaBlock, ArenaOptions } from "./arena.js";

export {
  PointCloudView,
  cloudRelativeElevationRange,
  createPointCloudView,
  scalarRangeFor,
} from "./view.js";
export type {
  ClipTarget,
  DeviceLostInfo,
  PointCloudViewOptions,
  ViewProfileOptions,
  ViewStats,
} from "./view.js";

/** `true` when this environment can create a WebGPU device. Safe in Node. */
export function isWebGPUAvailable(): boolean {
  return (
    typeof navigator !== "undefined" &&
    (navigator as { gpu?: unknown }).gpu !== undefined
  );
}

export const VOXELKLOUD_VIEW_VERSION = "0.0.0";
