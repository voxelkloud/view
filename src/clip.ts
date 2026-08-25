import type { Matrix4 } from "three/webgpu";

/**
 * Force a clip matrix to WebGPU's depth range, in place.
 *
 * three's `PerspectiveCamera.updateProjectionMatrix` emits the WebGL form
 * (near maps to -1) in this setup even when the renderer and the camera both
 * report the WebGPU coordinate system — measured, not assumed. Everything
 * downstream here expects [0, 1]: the compute shader rejects `ndc.z < 0`, and
 * WebGPU's rasteriser clips it outright.
 *
 * Left unfixed, that silently drops geometry nearer than roughly twice the
 * near plane — invisible while orbiting a cloud from outside, constant the
 * moment a camera moves through it.
 *
 * The remap is exact and needs no near/far: z' = (z + w) / 2.
 */
export function toZeroToOneDepth(m: Matrix4): Matrix4 {
  const e = m.elements;
  for (let c = 0; c < 4; c++) {
    const z = c * 4 + 2;
    const w = c * 4 + 3;
    e[z] = 0.5 * (e[z]! + e[w]!);
  }
  return m;
}
