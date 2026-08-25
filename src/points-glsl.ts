// The GLSL for the WebGL 2 points rasteriser, in its own file.
//
// Not organisation for its own sake: a shader written inline in a .ts template
// literal cannot contain a backtick, and every prose comment inside one wants to
// quote an identifier. That cost four broken builds in one sitting, and the
// sweep I wrote to fix the first three silently ate three TypeScript
// interpolations — including the dataset path, which turned into a 404 that
// looked like a server problem. compute-wgsl.ts already lives apart for exactly
// this reason.
//
// It also puts the part a graphics reviewer reads in one place, rather than
// interleaved with allocator bookkeeping.

export const VS = `#version 300 es
precision highp float;
in vec3 aPos;
in vec4 aColor;
in float aPitch;
// LEVEL and SLOT in one attribute: 'slot * 32 + level'. Both are small
// integers and float32 is exact to 2^24, so this costs no precision and saves
// an attribute — which matters at three million vertices.
in float aMeta;

uniform mat4 uClipFromCloud;
uniform mat4 uViewFromCloud;
uniform float uProjScale;   // 0.5 * drawingBufferHeight * projection[1][1]
uniform float uSizeMul;
uniform float uMinPx;
uniform float uMaxPx;
uniform highp usampler2D uCut;   // RGBA8UI: mask, then first-child slot big-endian
uniform vec3 uRootMin;
uniform vec3 uRootSize;
uniform int uCutDepth;
uniform int uUseCut;
uniform highp usampler2D uLive;   // R8UI per node slot: 1 when this frame draws it
uniform int uUseMask;
uniform int uMode;          // rgb flat elevation level intensity classification
uniform vec3 uFlatColor;
uniform vec2 uElevRange;
uniform vec2 uScalarRange;
uniform float uMaxLevel;

// The same five stops the instanced material ramps through, and the same ASPRS
// palette, as constants rather than a texture: 5 and 19 entries of compile-time
// data do not need a bind point.
// CONST and global, not local. A 'vec3 R[5];' declared inside the function is
// built per invocation in GLSL — three million times a frame — which is enough
// to drop this shader from 60 fps to exactly 30, one missed vsync per frame.
const vec3 RAMP[5] = vec3[5](
  vec3(0.19, 0.07, 0.23), vec3(0.21, 0.36, 0.55), vec3(0.13, 0.57, 0.55),
  vec3(0.48, 0.74, 0.32), vec3(0.99, 0.91, 0.15));

vec3 rampAt(float t) {
  float x = clamp(t, 0.0, 1.0) * 4.0;
  int i = min(int(x), 3);
  return mix(RAMP[i], RAMP[i + 1], x - float(i));
}

const vec3 CLASSES[19] = vec3[19](
  vec3(0.42,0.42,0.46), vec3(0.62,0.62,0.66), vec3(0.55,0.42,0.28),
  vec3(0.35,0.60,0.30), vec3(0.28,0.68,0.32), vec3(0.18,0.50,0.24),
  vec3(0.85,0.45,0.35), vec3(0.90,0.20,0.35), vec3(0.75,0.70,0.30),
  vec3(0.24,0.50,0.85), vec3(0.50,0.35,0.55), vec3(0.34,0.34,0.40),
  vec3(0.70,0.70,0.55), vec3(0.95,0.75,0.25), vec3(0.95,0.60,0.15),
  vec3(0.80,0.50,0.60), vec3(0.95,0.85,0.40), vec3(0.60,0.55,0.70),
  vec3(1.00,0.10,0.55));

vec3 classColor(int code) {
  // Deliberately a colour no standard class uses, so "no colour for this" never
  // reads as "unclassified".
  if (code < 0 || code > 18) return vec3(0.0, 0.85, 0.8);
  return CLASSES[code];
}

out vec4 vColor;
// The point's index, for picking. One buffer means one draw, so 'gl_VertexID'
// IS the global index — no attribute, no per-draw base.
flat out uint vId;

void main() {
  float aLevel = mod(aMeta, 32.0);
  int slotId = int((aMeta - aLevel) / 32.0);

  // Rejected BEFORE any projection work. A point whose node the scheduler did
  // not pick this frame is pushed outside clip space, which is the cheapest
  // cull available in a vertex shader — 'discard' would still pay for the
  // fragment.
  if (uUseMask == 1 && texelFetch(uLive, ivec2(slotId & 1023, slotId >> 10), 0).r == 0u) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    vColor = vec4(0.0);
    vId = 0u;
    return;
  }

  vec4 view = uViewFromCloud * vec4(aPos, 1.0);
  gl_Position = uClipFromCloud * vec4(aPos, 1.0);

  // 'aColor' carries whatever THIS mode needs, because only one is live per
  // cloud: RGBA bytes, or a scalar packed into the first two so a 16-bit
  // intensity survives a normalised byte attribute. Same trade the compute
  // rasteriser makes, for the same reason — a second per-point buffer to carry
  // a value only one mode reads is a buffer four modes pay for.
  vec3 rgb;
  if (uMode == 1) {
    rgb = uFlatColor;
  } else if (uMode == 2) {
    rgb = rampAt((aPos.z - uElevRange.x) / max(uElevRange.y - uElevRange.x, 1e-9));
  } else if (uMode == 3) {
    rgb = rampAt(aLevel / max(uMaxLevel, 1.0));
  } else if (uMode == 4) {
    float v = aColor.r * 255.0 + aColor.g * 255.0 * 256.0;
    rgb = rampAt((v - uScalarRange.x) / max(uScalarRange.y - uScalarRange.x, 1e-9));
  } else if (uMode == 5) {
    rgb = classColor(int(aColor.r * 255.0 + 0.5));
  } else {
    rgb = aColor.rgb;
  }
  vColor = vec4(rgb, 1.0);
  vId = uint(gl_VertexID);

  // LOCAL DEPTH: descend the selected cut to the deepest node containing this
  // point and size by THAT pitch. A point sized by its own node instead draws
  // 2**(D-L) too wide everywhere behind the frontier, painting over finer data
  // already on the GPU.
  //
  // 'usampler2D' and 'texelFetch' read the bytes as integers, where the
  // instanced material multiplies floats by 255 and rounds — the same encoding
  // through a conversion this shader does not need.
  float shrink = 0.0;
  if (uUseCut == 1) {
    int slot = 0;
    vec3 bMin = uRootMin;
    vec3 bSize = uRootSize;
    int depth = 0;
    for (int i = 0; i < 32; i++) {
      if (i >= uCutDepth) break;
      uvec4 t = texelFetch(uCut, ivec2(slot & 1023, slot >> 10), 0);
      int mask = int(t.r);
      vec3 half_ = bSize * 0.5;
      vec3 c = step(bMin + half_, aPos);
      int idx = int(c.x) * 4 + int(c.y) * 2 + int(c.z);
      if ((mask & (1 << idx)) == 0) break;
      int first = int(t.g) * 65536 + int(t.b) * 256 + int(t.a);
      int below = 0;
      for (int b = 0; b < 7; b++) {
        if (b >= idx) break;
        below += (mask >> b) & 1;
      }
      slot = first + below;
      bMin += c * half_;
      bSize = half_;
      depth += 1;
    }
    // Capped at one level, for the reason the shared cut is capped everywhere:
    // shrinking by the full level difference assumes the levels in between
    // deliver enough points to refill the area given up, and they do not
    // always — coverage FALLS as a cloud loads.
    shrink = clamp(float(depth) - aLevel, 0.0, 1.0);
  }

  // A world DIAMETER of 'pitch * 2 * sizeMultiplier', projected, clamped in
  // pixels. '-view.z' is the eye distance for a right-handed view matrix.
  float px = (aPitch / exp2(shrink)) * 2.0 * uSizeMul * uProjScale / max(-view.z, 1e-6);
  gl_PointSize = clamp(px, uMinPx, uMaxPx);
}`;

export const FS_PICK = `#version 300 es
precision highp float;
flat in uint vId;
uniform int uRound;
out uvec4 outId;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (uRound == 1 && dot(d, d) > 0.25) discard;
  // +1 so zero means "nothing here" — the framebuffer is cleared to zero and a
  // real point 0 would otherwise read as empty.
  uint id = vId + 1u;
  outId = uvec4(id & 255u, (id >> 8) & 255u, (id >> 16) & 255u, 255u);
}`;

export const FS = `#version 300 es
precision highp float;
in vec4 vColor;
uniform int uRound;
out vec4 outColor;
void main() {
  // A round splat, not the square gl.POINTS gives for free. Potree offers both;
  // the disc is what our instanced material draws, so this keeps the two
  // rasterisers comparable rather than flattering this one.
  vec2 d = gl_PointCoord - vec2(0.5);
  if (uRound == 1 && dot(d, d) > 0.25) discard;
  outColor = vColor;
}`;

export const VS_POST = `#version 300 es
void main() {
  // One oversized triangle, not a quad: fewer vertices and no seam.
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const FS_EDL = `#version 300 es
precision highp float;
uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform vec2 uTexel;
uniform float uNear;
uniform float uFar;
uniform float uStrength;
uniform float uRadius;
out vec4 outColor;

// Window depth back to EYE depth. The metric has to be a RATIO of distances or
// a cloud with real extent shades only near the near plane, which is why the
// response below takes log2 of this and not of the stored value.
float viewDepth(vec2 uv) {
  float d = texture(uDepth, uv).r;
  return uNear * uFar / (uFar - (uFar - uNear) * d);
}

void main() {
  vec2 uv = gl_FragCoord.xy * uTexel;
  vec4 src = texture(uColor, uv);
  float raw = texture(uDepth, uv).r;

  // A background pixel is at the far plane and is not geometry, so it must not
  // be shaded: without this every silhouette against the sky grows a dark halo
  // on the OUTSIDE. A threshold, not an equality — that value came out of a
  // projection divide.
  if (raw >= 0.999999 || uStrength <= 0.0) { outColor = src; return; }

  const vec2 RING[8] = vec2[8](
    vec2(1.0, 0.0), vec2(0.7071, 0.7071), vec2(0.0, 1.0), vec2(-0.7071, 0.7071),
    vec2(-1.0, 0.0), vec2(-0.7071, -0.7071), vec2(0.0, -1.0), vec2(0.7071, -0.7071));

  float centre = log2(viewDepth(uv));
  float sum = 0.0;
  for (int i = 0; i < 8; i++) {
    // Centre MINUS neighbour, darkening where the centre is farther. Potree's
    // formulation on purpose, so an 'edlStrength' transfers without retuning.
    sum += max(centre - log2(viewDepth(uv + RING[i] * uRadius * uTexel)), 0.0);
  }
  float shade = exp(-(sum / 8.0) * 300.0 * uStrength);
  outColor = vec4(src.rgb * shade, src.a);
}`;
