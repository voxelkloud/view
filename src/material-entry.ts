// `@voxelkloud/view/material` — the point material itself.
//
// A subpath rather than a root export, because building it needs `three/webgpu`
// and a root export made that edge static for everyone. Import from here if you
// construct materials directly; the option helpers and every type stay on the
// package root.
export { createPointMaterial } from "./material.js";
export type { PointCloudMaterial } from "./material.js";
