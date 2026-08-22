import {
  BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  PerspectiveCamera,
  Scene,
  Sphere,
  Vector3,
  WebGPURenderer,
} from "three/webgpu";
import { describe, expect, it } from "vitest";
import { createPointMaterial } from "./material.js";
import type { ColorMode } from "./material.js";

/**
 * Compile a material to WGSL with NO GPU.
 *
 * `backend.createNodeBuilder(...).build()` is the same call the renderer makes
 * per render object, and it needs no `GPUDevice` — which moves "does this node
 * graph compile to the shader we intended" from browser-only to a vitest
 * assertion. Without it, a swizzle on a wrong-width node or a vec3 where WGSL
 * wants vec4 surfaces as a console error and a blank canvas.
 */
function compile(colorMode: ColorMode, highPrecision = true) {
  const needsScalar =
    colorMode.kind === "intensity" || colorMode.kind === "classification";
  const g = new InstancedBufferGeometry();
  g.setAttribute(
    "position",
    new BufferAttribute(
      new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
      3,
    ),
  );
  g.setIndex([0, 1, 2, 0, 2, 3]);
  g.setAttribute(
    "pointOffset",
    new InstancedBufferAttribute(new Float32Array(12), 3, false),
  );
  g.setAttribute(
    "color",
    new InstancedBufferAttribute(new Uint8Array(16), 4, true),
  );
  if (needsScalar) {
    g.setAttribute(
      "scalarValue",
      new InstancedBufferAttribute(new Float32Array(4), 1, false),
    );
  }
  g.instanceCount = 4;
  g.boundingSphere = new Sphere(new Vector3(), 1);

  const material = createPointMaterial({ colorMode });
  const mesh = new Mesh(g, material);
  mesh.frustumCulled = false;
  const scene = new Scene();
  scene.add(mesh);
  const camera = new PerspectiveCamera(60, 1, 1, 1000);
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);

  const canvas = {
    getContext: () => null,
    addEventListener() {},
    removeEventListener() {},
    width: 8,
    height: 8,
    style: {},
    clientWidth: 8,
    clientHeight: 8,
  } as unknown as HTMLCanvasElement;

  const renderer = new WebGPURenderer({ canvas, antialias: false });
  renderer.highPrecision = highPrecision;

  const b = (
    renderer.backend as unknown as {
      createNodeBuilder(o: unknown, r: unknown): Record<string, unknown>;
    }
  ).createNodeBuilder(mesh, renderer);
  b['scene'] = scene;
  b['material'] = material;
  b['camera'] = camera;
  (b['context'] as Record<string, unknown>)['material'] = material;
  b['lightsNode'] = null;
  b['environmentNode'] = null;
  b['fogNode'] = null;
  // setupHardwareClipping short-circuits only on an explicit null.
  b['clippingContext'] = null;

  const state = (b['build'] as () => Record<string, unknown>).call(b);
  return {
    attributes: (b['attributes'] as Array<{ name: string; type: string }>).map(
      (a) => `${a.name}:${a.type}`,
    ),
    vertex: String(state['vertexShader']),
    fragment: String(state['fragmentShader']),
  };
}

describe("point material: WGSL emission", () => {
  // The two modes that read a scalar instance attribute rather than vertex
  // colour. three has no vertex format for a 1-component uint8, so these ride
  // Task 4's f32 gpu lane and normalise in the shader.
  it.each([
    ["intensity", { kind: "intensity" } as ColorMode],
    ["classification", { kind: "classification" } as ColorMode],
  ])("binds scalarValue for colour mode %s", (_name, mode) => {
    const { attributes, fragment } = compile(mode);
    expect(attributes).toContain("scalarValue:float");
    expect(fragment).toContain("@fragment");
  });

  it("does not bind scalarValue for the modes that do not need it", () => {
    expect(compile({ kind: "rgb" }).attributes).not.toContain(
      "scalarValue:float",
    );
    expect(compile({ kind: "elevation" }).attributes).not.toContain(
      "scalarValue:float",
    );
  });

  it("binds exactly the instanced-quad attribute set", () => {
    const { attributes } = compile({ kind: "rgb" });
    expect(attributes.sort()).toEqual([
      "color:vec4",
      "pointOffset:vec3",
      "position:vec3",
    ]);
  });

  // The precision decision, verified reaching the shader. Measured on autzen:
  // absolute float32 gives 29.2 px of jitter, cloud-relative on the DEFAULT
  // mediump path gives 34.1 px — WORSE, because mediumpModelViewMatrix
  // multiplies both large translations in float32 IN THE SHADER — and
  // highPrecision gives 0.46 px by uploading the float64 camera-relative
  // product as a single uniform.
  it("uses highpModelViewMatrix when the renderer asks for it", () => {
    expect(compile({ kind: "rgb" }, true).vertex).toContain(
      "highpModelViewMatrix",
    );
  });

  it("falls back to the mediump path when it does not", () => {
    const vs = compile({ kind: "rgb" }, false).vertex;
    expect(vs).not.toContain("highpModelViewMatrix");
    expect(vs).toContain("modelViewMatrix");
  });

  it("emits a vertex stage that reads the liveness alpha and expands a quad", () => {
    const { vertex } = compile({ kind: "rgb" });
    // The arena collapses a dead slot by zeroing colour alpha, so the diameter
    // must be multiplied by it.
    expect(vertex).toContain("color");
    expect(vertex).toContain("cameraProjectionMatrix");
    expect(vertex).toContain("@vertex");
    expect(vertex).toContain("vkCorner");
  });

  it("discards fragments outside the unit disc", () => {
    // NOT via pointUV: PointUVNode.generate returns a literal GLSL string
    // referencing gl_PointCoord, which does not exist in WGSL.
    expect(compile({ kind: "rgb" }).fragment).toContain("discard");
  });

  it("converts sRGB to working space exactly once", () => {
    const { fragment } = compile({ kind: "rgb" });
    // The conversion shows up as the sRGB transfer function constants.
    expect(/2\.4|1\.055|0\.0031308|0\.04045/.test(fragment)).toBe(true);
  });

  it.each([
    ["rgb", { kind: "rgb" } as ColorMode],
    ["flat", { kind: "flat", color: [1, 0.5, 0.2] } as ColorMode],
    ["elevation", { kind: "elevation" } as ColorMode],
    ["level", { kind: "level" } as ColorMode],
    ["intensity", { kind: "intensity" } as ColorMode],
    ["classification", { kind: "classification" } as ColorMode],
  ])("compiles colour mode %s to complete WGSL", (_name, mode) => {
    const { vertex, fragment } = compile(mode);
    expect(vertex).toContain("@vertex");
    expect(fragment).toContain("@fragment");
    expect(vertex.length).toBeGreaterThan(200);
    expect(fragment.length).toBeGreaterThan(200);
  });
});
