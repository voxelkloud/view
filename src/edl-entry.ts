// `@voxelkloud/view/edl` — the eye-dome-lighting pass.
//
// A subpath for the same reason as the material: `PostProcessing` lives in
// three's WebGPU build. The compute and points rasterisers have their own EDL
// and never come here.
export { createEdlPipeline, resolveEdlOptions } from "./edl.js";
export type { EdlOptions, EdlPipeline, ResolvedEdlOptions } from "./edl.js";
