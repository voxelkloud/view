import type { Vec3 } from "@voxelkloud/core";
import { Group, Vector3 } from "three";

/** O eixo do yaw. Módulo-nível: um por processo, não um por colocação. */
const UP = new Vector3(0, 0, 1);

/**
 * Where a cloud sits in the scene when its coordinates are NOT the project's.
 *
 * A project has one coordinate system — the anchor layer's. A cloud delivered
 * in another one cannot simply be offset into it: two projections put the same
 * ground in numerically unrelated places, so the subtraction that should give
 * tens of metres gives thousands of kilometres. What relates them, over the
 * span of a single survey, is a SIMILARITY: rotate about the vertical, scale
 * uniformly, translate.
 *
 * MEASURED, fitting each form by least squares over an 81-point control grid
 * and reporting the worst residual over a 1 km cloud:
 *
 *   pair                                translation   rigid    similarity
 *   same UTM zone, different datum         0.09 cm   0.09 cm     0.00 cm
 *   RD New -> UTM31N (Rotterdam)          23.20 m   15.17 cm     0.08 cm
 *   adjacent UTM zones (22S -> 23S)       33.75 m    8.04 cm     0.37 cm
 *   State Plane ftUS -> UTM11N           491.65 m  491.64 m      0.01 cm
 *
 * So translation alone is unusable the moment the PROJECTION changes, and a
 * rigid fit still misses by 15 cm per kilometre and collapses entirely when the
 * units differ — the last row is feet against metres, which no rotation can
 * absorb. The scale is what earns the last two digits, and it is why this is a
 * similarity and not a rigid motion.
 *
 * The map is `at + scale * Rz(yaw) * (abs - pivot)`, with `abs` in the cloud's
 * OWN system and the result in the project's. Anchoring it on a pivot rather
 * than on the origin is what keeps the fit meaningful: a similarity fitted at
 * the cloud is exact at the cloud and drifts away from it, and the pivot names
 * the place where it was fitted.
 */
export interface CloudPlacement {
  /** Radians, right-handed about +Z. */
  readonly yaw: number;
  /** Uniform. Also scales Z — see {@link CloudFrame} on why that is right. */
  readonly scale: number;
  /** The point the fit was anchored on, in the cloud's own coordinates. */
  readonly pivot: Vec3;
  /** Where that point lands, in the project's coordinates. */
  readonly at: Vec3;
}

/** Mutable 3-vector destination, so the hot loops allocate nothing. */
export interface Vec3Out {
  x: number;
  y: number;
  z: number;
}

/**
 * One cloud's placement, folded into the two conversions everything needs.
 *
 * Before placements existed, every caller did the same subtraction inline —
 * `local + (cloudOrigin - sceneOrigin)` to reach the scene, `+ sceneOrigin` to
 * get back to absolute — in four files. That was correct exactly while the
 * relationship was a translation, and there was nothing in the code to say so.
 * This type is that assumption made explicit and given ONE implementation: a
 * general affine map `scene = A*abs + b`, of which the old subtraction is the
 * case `A = I`.
 *
 * `A` is a similarity about the vertical, so it is stored as three numbers
 * rather than nine, and its inverse is exact rather than solved.
 *
 * ON SCALING Z. The scale is uniform in all three axes, including height. That
 * is what the unit-conversion case demands — a cloud in US survey feet has its
 * elevations in feet too, and leaving Z alone would flatten a building by a
 * factor of three. Between two systems in the same units the scale sits within
 * 1e-4 of one, where applying it to a 100 m height is a 1 cm error: far below
 * the noise of any survey, and the wrong direction to chase when the
 * alternative is being wrong by a factor of 3.28 in the case that matters.
 */
export class CloudFrame {
  /** `scale * cos(yaw)` and `scale * sin(yaw)`: the whole of `A`, in two numbers. */
  private readonly ac: number;
  private readonly as: number;
  private readonly scale: number;
  /** `b`, the translation of the abs -> scene map. */
  private readonly bx: number;
  private readonly by: number;
  private readonly bz: number;
  /** `A * cloudOrigin + b`, so the local -> scene loop hoists one add. */
  private readonly ox: number;
  private readonly oy: number;
  private readonly oz: number;
  /**
   * True when `A` is the identity — the overwhelmingly common case, and the
   * one the old code handled. Callers that can take a cheaper path check this
   * rather than comparing floats themselves.
   */
  readonly isTranslationOnly: boolean;

  constructor(
    readonly cloudOrigin: Vec3,
    readonly sceneOrigin: Vec3,
    readonly placement?: CloudPlacement | undefined,
  ) {
    const scale = placement?.scale ?? 1;
    const yaw = placement?.yaw ?? 0;
    this.scale = scale;
    this.ac = scale * Math.cos(yaw);
    this.as = scale * Math.sin(yaw);
    if (placement === undefined) {
      // `A = I`, so `b` is just the old `-sceneOrigin`.
      this.bx = -sceneOrigin[0];
      this.by = -sceneOrigin[1];
      this.bz = -sceneOrigin[2];
    } else {
      // b = at - A*pivot - sceneOrigin
      const { pivot, at } = placement;
      this.bx = at[0] - (this.ac * pivot[0] - this.as * pivot[1]) - sceneOrigin[0];
      this.by = at[1] - (this.as * pivot[0] + this.ac * pivot[1]) - sceneOrigin[1];
      this.bz = at[2] - scale * pivot[2] - sceneOrigin[2];
    }
    this.ox = this.ac * cloudOrigin[0] - this.as * cloudOrigin[1] + this.bx;
    this.oy = this.as * cloudOrigin[0] + this.ac * cloudOrigin[1] + this.by;
    this.oz = scale * cloudOrigin[2] + this.bz;
    this.isTranslationOnly = placement === undefined || (yaw === 0 && scale === 1);
  }

  /** Cloud-local (the frame `pointOffset` is in) to scene. */
  localToScene(x: number, y: number, z: number, out: Vec3Out): void {
    out.x = this.ac * x - this.as * y + this.ox;
    out.y = this.as * x + this.ac * y + this.oy;
    out.z = this.scale * z + this.oz;
  }

  /** Absolute, in the cloud's OWN system, to scene. */
  absToScene(x: number, y: number, z: number, out: Vec3Out): void {
    out.x = this.ac * x - this.as * y + this.bx;
    out.y = this.as * x + this.ac * y + this.by;
    out.z = this.scale * z + this.bz;
  }

  /**
   * Scene back to absolute, in the cloud's OWN system.
   *
   * The cloud's own and not the project's, deliberately: a measurement taken on
   * a reprojected layer belongs in the coordinates that layer was delivered in,
   * which are the ones its owner can check against their own survey. Handing
   * back project coordinates would silently re-express someone's data in a
   * system they never chose.
   */
  sceneToAbs(x: number, y: number, z: number, out: Vec3Out): void {
    const dx = x - this.bx;
    const dy = y - this.by;
    // A^-1 = (1/scale) * Rz(-yaw), and `ac/as` already carry the scale, so
    // dividing by `scale^2` is what undoes it once in each factor.
    const k = this.scale * this.scale;
    out.x = (this.ac * dx + this.as * dy) / k;
    out.y = (-this.as * dx + this.ac * dy) / k;
    out.z = (z - this.bz) / this.scale;
  }

  /**
   * The scene-space AABB of an absolute-space AABB.
   *
   * A rotated box is no longer axis-aligned, so this returns the TIGHT
   * axis-aligned bound of the rotated one — bigger than the original, never
   * smaller. Every caller uses these boxes to decide whether to look closer
   * (ray-box rejection, grid overlap), so growing them costs a few extra nodes
   * read and shrinking them would silently drop geometry.
   */
  sceneBox(
    minX: number,
    minY: number,
    minZ: number,
    maxX: number,
    maxY: number,
    maxZ: number,
    out: Float64Array,
  ): void {
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;
    const hx = (maxX - minX) * 0.5;
    const hy = (maxY - minY) * 0.5;
    const hz = (maxZ - minZ) * 0.5;
    const sx = this.ac * cx - this.as * cy + this.bx;
    const sy = this.as * cx + this.ac * cy + this.by;
    const sz = this.scale * cz + this.bz;
    const ex = Math.abs(this.ac) * hx + Math.abs(this.as) * hy;
    const ey = Math.abs(this.as) * hx + Math.abs(this.ac) * hy;
    const ez = this.scale * hz;
    out[0] = sx - ex;
    out[1] = sy - ey;
    out[2] = sz - ez;
    out[3] = sx + ex;
    out[4] = sy + ey;
    out[5] = sz + ez;
  }

  /**
   * How much a length in the cloud's own units grows in the scene.
   *
   * Point spacing, node radii and z-ranges are all lengths, and under a scale
   * they are not the same number on both sides.
   */
  get lengthScale(): number {
    return this.scale;
  }

  /**
   * Height alone, both ways.
   *
   * Z is the one axis the yaw leaves alone, so the height filter, the ground
   * level and the elevation ramp can convert without touching X or Y — which is
   * what let those three keep working through a single `dz` before placements
   * existed. Under a scale that `dz` is no longer the whole story, and these
   * two are.
   */
  localZToScene(z: number): number {
    return this.scale * z + this.oz;
  }

  sceneZToLocal(z: number): number {
    return (z - this.oz) / this.scale;
  }
}

/**
 * The scene-graph root for one cloud, and the ONE place the coordinate offset
 * lives.
 *
 * Task 4 emits positions as float32 RELATIVE to `metadata.boundingBox.min`, the
 * same origin for every node, so all node buffers share one frame and one model
 * matrix. This Group's translation carries `cloudOrigin - sceneOrigin`, computed
 * in JS float64 and written once.
 *
 * Measured at autzen scale (935 px per world metre at 1080p / 60 deg / 1 m):
 * - absolute float32 vertices: 0.03125 m error = 29.2 px of jitter;
 * - cloud-relative on three's DEFAULT mediump path: 0.03649 m = 34.1 px, i.e.
 *   WORSE than doing nothing, because `mediumpModelViewMatrix` multiplies both
 *   large translations IN THE SHADER in float32;
 * - cloud-relative with `renderer.highPrecision = true`: 4.88e-4 m = 0.46 px.
 *
 * So the offset here is only half the story — the view sets `highPrecision`
 * unconditionally, and without it this arrangement is actively harmful.
 *
 * With a {@link CloudPlacement} the translation gains a rotation and a scale,
 * and the same float64-once rule applies to all three: three composes
 * `T * R * S` into the matrix, which for a UNIFORM scale is exactly the
 * `A*local + t` this class means. A non-uniform scale would not commute and is
 * not expressible here, which is the point — a survey is not stretched.
 */
export class PointCloudObject3D extends Group {
  /** `metadata.boundingBox.min`, absolute CRS, float64. */
  readonly cloudOrigin: Vec3;
  private sceneOrigin: Vec3;
  private placement: CloudPlacement | undefined;
  private frameCache: CloudFrame;

  constructor(cloudOrigin: Vec3, sceneOrigin: Vec3 = cloudOrigin) {
    super();
    this.cloudOrigin = cloudOrigin;
    this.sceneOrigin = sceneOrigin;
    // The matrix is written once, in float64, and never recomputed from
    // position/quaternion/scale by the renderer.
    this.matrixAutoUpdate = false;
    // Node visibility is the scheduler's decision, made against the same
    // frustum; letting three re-cull the root by a stale bounding sphere would
    // only be able to disagree.
    this.frustumCulled = false;
    this.frameCache = new CloudFrame(cloudOrigin, sceneOrigin, undefined);
    this.applyOrigin();
  }

  /**
   * Re-base the cloud against a different scene origin.
   *
   * Useful when several clouds share a scene: they must agree on one origin, or
   * their float32 offsets are in different frames.
   */
  setSceneOrigin(origin: Vec3): void {
    this.sceneOrigin = origin;
    this.applyOrigin();
  }

  getSceneOrigin(): Vec3 {
    return this.sceneOrigin;
  }

  /** `undefined` puts the cloud back in its own coordinates. */
  setPlacement(placement: CloudPlacement | undefined): void {
    this.placement = placement;
    this.applyOrigin();
  }

  getPlacement(): CloudPlacement | undefined {
    return this.placement;
  }

  /**
   * The conversions this cloud's placement implies.
   *
   * Rebuilt only when the placement or the scene origin changes, and handed out
   * by reference: the pick and profile loops call into it per point, and
   * allocating one per frame would put the garbage collector in the hot path.
   */
  frame(): CloudFrame {
    return this.frameCache;
  }

  private applyOrigin(): void {
    this.frameCache = new CloudFrame(this.cloudOrigin, this.sceneOrigin, this.placement);
    const p = this.placement;
    if (p === undefined) {
      this.position.set(
        this.cloudOrigin[0] - this.sceneOrigin[0],
        this.cloudOrigin[1] - this.sceneOrigin[1],
        this.cloudOrigin[2] - this.sceneOrigin[2],
      );
      this.quaternion.identity();
      this.scale.set(1, 1, 1);
    } else {
      // three composes `T * R * S`, which applied to a cloud-local point is
      // `scale * R * local + position` — so `position` is where the cloud's OWN
      // origin lands, which is exactly what `CloudFrame` hoisted.
      const o = { x: 0, y: 0, z: 0 };
      this.frameCache.localToScene(0, 0, 0, o);
      this.position.set(o.x, o.y, o.z);
      this.quaternion.setFromAxisAngle(UP, p.yaw);
      this.scale.set(p.scale, p.scale, p.scale);
    }
    this.updateMatrix();
    this.updateMatrixWorld(true);
  }
}
