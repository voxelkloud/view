import type { Vec3 } from "@voxelkloud/core";
import { Group } from "three";

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
 */
export class PointCloudObject3D extends Group {
  /** `metadata.boundingBox.min`, absolute CRS, float64. */
  readonly cloudOrigin: Vec3;
  private sceneOrigin: Vec3;

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

  private applyOrigin(): void {
    this.position.set(
      this.cloudOrigin[0] - this.sceneOrigin[0],
      this.cloudOrigin[1] - this.sceneOrigin[1],
      this.cloudOrigin[2] - this.sceneOrigin[2],
    );
    this.updateMatrix();
    this.updateMatrixWorld(true);
  }
}
