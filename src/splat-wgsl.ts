/**
 * The real alpha-blended Gaussian splat pass. Prototyped and visually
 * verified in `demo/app/src/gaussian-splat-shader.ts` first (see §Chunk 1 of
 * `docs/superpowers/plans/2026-08-31-live-gaussian-splat-capture.md`); this
 * copy adds the one thing a standalone demo canvas doesn't need — reading the
 * point cloud's own depth buffer to occlude against it, the same contract
 * `overlay.ts`'s `WGSL` const uses (`pointDepth`, `eyeDepth = clip.w`,
 * bit-pattern comparison because atomicMin only exists for integers).
 *
 * No existing WGSL in this package does BOTH order-dependent alpha blending
 * AND reads `compute-wgsl.ts`'s point depth: `compute-wgsl.ts` itself is an
 * atomics-based nearest-surface accumulator (order-independent by design),
 * and `overlay.ts` writes its own depth rather than only testing against it.
 */
export const SPLAT_WGSL = /* wgsl */ `
struct Uniforms {
  viewProj : mat4x4<f32>,
  screenW  : f32,
  screenH  : f32,
  ignoreDepth : f32,
  _pad     : f32,
  // Mirrors compute-wgsl.ts's zRange/clip fields in spirit (not offset --
  // this struct is smaller): cloud-local, tested against center before any
  // billboard offset, same as project()'s pre-projection clip test there.
  zLo      : f32,
  zHi      : f32,
  clipCount : f32,
  _pad2    : f32,
  clip     : array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read> pointDepth : array<u32>;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  // The clip w IS the eye depth — same convention overlay.ts uses, so a
  // splat compares against compute-wgsl.ts's stored depth on equal terms.
  @location(0) eyeDepth : f32,
  @location(1) corner : vec2<f32>,
  @location(2) color : vec4<f32>,
};

// Cloud-local, pre-billboard — same semantics as compute-wgsl.ts's zOff()
// and the clip-plane loop in project(), just evaluated per-splat instead of
// per-point.
fn culled(p : vec3<f32>) -> bool {
  if (p.z < u.zLo || p.z > u.zHi) {
    return true;
  }
  let nClip = u32(u.clipCount);
  for (var ci : u32 = 0u; ci < nClip; ci = ci + 1u) {
    let pl = u.clip[ci];
    if (dot(pl.xyz, p) + pl.w < 0.0) {
      return true;
    }
  }
  return false;
}

@vertex
fn vs_main(
  @location(0) corner : vec2<f32>,
  @location(1) center : vec3<f32>,
  @location(2) axisU : vec3<f32>,
  @location(3) axisV : vec3<f32>,
  @location(4) color : vec4<f32>,
  @location(5) opacity : f32,
) -> VertexOutput {
  let worldPosition = center + axisU * corner.x + axisV * corner.y;
  var clip = u.viewProj * vec4<f32>(worldPosition, 1.0);
  if (culled(center)) {
    // Outside the [-w,w] clip volume on every axis with w > 0: hardware
    // clipping drops it before rasterization, same effect as a discard but
    // legal from a vertex stage.
    clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  }
  var out : VertexOutput;
  out.position = clip;
  out.eyeDepth = clip.w;
  out.corner = corner;
  out.color = vec4<f32>(color.rgb, color.a * opacity);
  return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
  let x = u32(clamp(in.position.x, 0.0, u.screenW - 1.0));
  let y = u32(clamp(in.position.y, 0.0, u.screenH - 1.0));
  let stored = pointDepth[y * u32(u.screenW) + x];
  // Both are bit patterns of non-negative floats, monotonic in the value —
  // the same trick compute-wgsl.ts's atomicMin depends on, reused here to
  // compare without decoding either side.
  if (u.ignoreDepth < 0.5 && bitcast<u32>(in.eyeDepth) > stored) {
    discard;
  }

  let r2 = dot(in.corner, in.corner);
  if (r2 > 1.0) {
    discard;
  }
  let alpha = in.color.a * exp(-2.0 * r2);
  if (alpha < 0.01) {
    discard;
  }
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;
