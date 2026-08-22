import { PostProcessing } from "three/webgpu";
import type { PerspectiveCamera, Scene, WebGPURenderer } from "three/webgpu";
import { Fn, float, log2, max, pass, screenSize, uniform, uv, vec2, vec4 } from "three/tsl";

export interface EdlOptions {
  /**
   * How hard the shading bites. Potree's default is 1.0 against the same 300x
   * constant this uses, so an `edlStrength` carries over unchanged.
   */
  readonly strength?: number;
  /**
   * Neighbour sampling radius in DEVICE pixels. Below ~1 the ring collapses
   * into the centre texel and the effect vanishes; much above ~3 it reads as an
   * outline rather than as shading.
   */
  readonly radius?: number;
  /**
   * Opacity of the shading, 0..1. Mixing the unshaded colour back in is the
   * honest way to soften the effect: turning `strength` down instead also
   * changes WHICH edges are found, not just how dark they get.
   */
  readonly opacity?: number;
}

export interface ResolvedEdlOptions {
  readonly strength: number;
  readonly radius: number;
  readonly opacity: number;
}

export function resolveEdlOptions(o: EdlOptions = {}): ResolvedEdlOptions {
  return {
    strength: o.strength ?? 1,
    radius: o.radius ?? 1.4,
    opacity: o.opacity ?? 1,
  };
}

/**
 * The 8-neighbour ring, as unit offsets.
 *
 * Unrolled at build time rather than looped: the count is a compile-time
 * constant, so this is one straight-line shader with no dynamic indexing, and
 * TSL's `Loop` would buy nothing.
 */
const RING: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [0.7071, 0.7071],
  [0, 1],
  [-0.7071, 0.7071],
  [-1, 0],
  [-0.7071, -0.7071],
  [0, -1],
  [0.7071, -0.7071],
];

/**
 * Potree's shading constant.
 *
 * Not derivable from anything: it is the number that makes `strength: 1` look
 * right against a mean log-depth response, and matching it is what lets a
 * Potree user's `edlStrength` transfer without retuning.
 */
const POTREE_SHADE_SCALE = 300;

export interface EdlPipeline {
  /** Draw the scene through the EDL pass. */
  render(): void;
  setStrength(v: number): void;
  setRadius(v: number): void;
  setOpacity(v: number): void;
  dispose(): void;
}

/**
 * Eye-dome lighting, as a two-pass post-process.
 *
 * A point cloud has no normals, so nothing shades it: a bare splat render is a
 * flat wash in which depth reads only from occlusion. EDL recovers the missing
 * shape from the DEPTH BUFFER — a pixel is darkened in proportion to how much
 * closer it is than its neighbours, so silhouettes and creases pick up a dark
 * edge and a surface starts reading as a surface.
 *
 * `response` is the mean POSITIVE log-depth difference over an 8-neighbour
 * ring and `shade = exp(-response * 300 * strength)`, which is Potree's
 * formulation on purpose.
 *
 * LOG depth, not linear: the difference between two depths has to mean the same
 * thing 10 m from the camera and 10 km from it, or a cloud with real extent
 * shades only near the near plane. `log2` makes the metric a RATIO.
 *
 * Only CLOSER neighbours darken. An absolute difference would outline both
 * sides of every edge and turn the cloud into a wireframe of its own
 * silhouettes.
 */
export function createEdlPipeline(
  renderer: WebGPURenderer,
  scene: Scene,
  camera: PerspectiveCamera,
  options: EdlOptions = {},
): EdlPipeline {
  const o = resolveEdlOptions(options);
  const uStrength = uniform(o.strength);
  const uRadius = uniform(o.radius);
  const uOpacity = uniform(o.opacity);
  // The pass's own near/far, NOT three's `cameraNear`/`cameraFar` nodes: during
  // the post pass those resolve against the full-screen quad's camera, not the
  // one the depth buffer was rendered with, and the linearisation comes out
  // silently wrong. `PassNode` keeps private uniforms for exactly this reason;
  // these are the public equivalent, refreshed each frame in `render`.
  const uNear = uniform(camera.near);
  const uFar = uniform(camera.far);

  const scenePass = pass(scene, camera);
  const color = scenePass.getTextureNode();
  const depth = scenePass.getTextureNode("depth");

  /** Positive distance along the view axis, in world units. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewDepthAt = (p: any): any => {
    const d = depth.sample(p).r;
    // perspectiveDepthToViewZ, negated: near*far / (far - (far-near)*d).
    return uNear.mul(uFar).div(uFar.sub(uFar.sub(uNear).mul(d)).max(1e-6));
  };

  const edl = Fn(() => {
    const p = uv();
    // Device pixels, so the ring is the same visual size at any DPR. The pass
    // renders at the drawing-buffer size, which is what `screenSize` reports.
    const step = uRadius.div(screenSize);

    const centre = log2(viewDepthAt(p));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let sum: any = float(0);
    for (const [dx, dy] of RING) {
      sum = sum.add(
        max(centre.sub(log2(viewDepthAt(p.add(vec2(dx, dy).mul(step))))), 0),
      );
    }
    const response = sum.div(RING.length);
    const shade = response
      .mul(uStrength)
      .mul(POTREE_SHADE_SCALE)
      .negate()
      .exp();

    const src = color.sample(p);
    // A background pixel is at the far plane and is not geometry, so it must
    // not be shaded: without this every silhouette against the sky grows a dark
    // halo on the OUTSIDE, which is Potree's `neighbourDepth == 0` case. The
    // test is a threshold, not `== 1`: that value came out of a projection
    // divide, and an equality test against it never fires.
    const isBackground = depth.sample(p).r.greaterThanEqual(0.999999);
    const lit = src.rgb.mix(src.rgb.mul(shade), uOpacity);
    return vec4(isBackground.select(src.rgb, lit), src.a);
  });

  const post = new PostProcessing(renderer);
  post.outputNode = edl();

  return {
    render: () => {
      // The camera's near/far are recomputed per frame from the visible set, so
      // reading them once at construction would linearise against a stale
      // range as soon as the camera moves.
      uNear.value = camera.near;
      uFar.value = camera.far;
      post.render();
    },
    setStrength: (v) => {
      uStrength.value = v;
    },
    setRadius: (v) => {
      uRadius.value = v;
    },
    setOpacity: (v) => {
      uOpacity.value = Math.min(Math.max(v, 0), 1);
    },
    dispose: () => {
      post.dispose();
    },
  };
}
