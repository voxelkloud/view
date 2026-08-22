import {
  Box3,
  Frustum,
  Matrix4,
  PerspectiveCamera,
  Vector3,
  WebGLCoordinateSystem,
  WebGPUCoordinateSystem,
} from "three";
import type { CoordinateSystem } from "three";
import { describe, expect, it } from "vitest";
import {
  Containment,
  classifyAabb,
  extractFrustumPlanes,
  intersectsAabb,
} from "./frustum.js";

/**
 * three's own Frustum is the oracle. Its module graph is only constants plus
 * Vector3/Sphere/Plane, so it imports cleanly in bare Node with no GPU.
 */
function oracle(m: Matrix4, coordinateSystem: CoordinateSystem): Frustum {
  return new Frustum().setFromProjectionMatrix(m, coordinateSystem);
}

function clipFromWorld(camera: PerspectiveCamera): Matrix4 {
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();
  return new Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse,
  );
}

const CASES: Array<[string, () => PerspectiveCamera]> = [
  [
    "origin, looking down -Z",
    () => new PerspectiveCamera(60, 16 / 9, 0.1, 1000),
  ],
  [
    "translated and rotated",
    () => {
      const c = new PerspectiveCamera(45, 1, 1, 5000);
      c.position.set(120, -40, 33);
      c.lookAt(new Vector3(5, 5, 0));
      return c;
    },
  ],
  [
    "at autzen CRS scale",
    () => {
      const c = new PerspectiveCamera(70, 1.6, 5.68, 20_000);
      c.position.set(637_000, 851_000, 900);
      c.lookAt(new Vector3(637_905, 851_209, 510));
      return c;
    },
  ],
  [
    "Z-up, steeply inclined",
    () => {
      const c = new PerspectiveCamera(35, 2, 0.5, 900);
      c.up.set(0, 0, 1);
      c.position.set(-30, -30, 80);
      c.lookAt(new Vector3(0, 0, 0));
      return c;
    },
  ],
];

describe("extractFrustumPlanes", () => {
  it.each(CASES)(
    "matches three's Frustum under WebGPU depth (%s)",
    (_label, make) => {
      const m = clipFromWorld(make());
      const planes = extractFrustumPlanes(
        m.elements,
        new Float64Array(24),
        "zero-to-one",
      );
      const want = oracle(m, WebGPUCoordinateSystem);

      for (let p = 0; p < 6; p++) {
        const ref = want.planes[p]!.clone().normalize();
        const o = p * 4;
        expect(planes[o]).toBeCloseTo(ref.normal.x, 10);
        expect(planes[o + 1]).toBeCloseTo(ref.normal.y, 10);
        expect(planes[o + 2]).toBeCloseTo(ref.normal.z, 10);
        expect(planes[o + 3]).toBeCloseTo(ref.constant, 6);
      }
    },
  );

  it.each(CASES)(
    "matches three's Frustum under WebGL depth (%s)",
    (_label, make) => {
      const m = clipFromWorld(make());
      const planes = extractFrustumPlanes(
        m.elements,
        new Float64Array(24),
        "minus-one-to-one",
      );
      const want = oracle(m, WebGLCoordinateSystem);
      for (let p = 0; p < 6; p++) {
        const ref = want.planes[p]!.clone().normalize();
        const o = p * 4;
        expect(planes[o]).toBeCloseTo(ref.normal.x, 10);
        expect(planes[o + 3]).toBeCloseTo(ref.constant, 6);
      }
    },
  );

  // The whole reason DepthRange has no default: using the WebGL near plane
  // against a WebGPU matrix does not fail loudly, it just misplaces one plane.
  it("puts the near plane in a different place per convention", () => {
    const m = clipFromWorld(new PerspectiveCamera(60, 1, 2, 100));
    const gpu = extractFrustumPlanes(m.elements, new Float64Array(24), "zero-to-one");
    const gl = extractFrustumPlanes(
      m.elements,
      new Float64Array(24),
      "minus-one-to-one",
    );
    // Planes 0..3 are shared; only the near plane (5) differs.
    for (let p = 0; p <= 3; p++) {
      expect(gpu[p * 4 + 3]).toBeCloseTo(gl[p * 4 + 3]!, 10);
    }
    expect(Math.abs(gpu[23]! - gl[23]!)).toBeGreaterThan(0.5);
  });

  it("normalises every plane so n·p + d is a distance in metres", () => {
    const camera = new PerspectiveCamera(60, 1, 1, 100);
    camera.position.set(0, 0, 10);
    // three builds a PerspectiveCamera's projection matrix with
    // WebGLCoordinateSystem by default, so the matching convention must be used
    // here. Passing "zero-to-one" against a WebGL-built matrix misplaces the
    // near plane by exactly the amount this assertion would catch — which is
    // precisely why DepthRange has no default and must be read from
    // renderer.coordinateSystem after init.
    const planes = extractFrustumPlanes(
      clipFromWorld(camera).elements,
      new Float64Array(24),
      "minus-one-to-one",
    );
    for (let p = 0; p < 6; p++) {
      const o = p * 4;
      expect(Math.hypot(planes[o]!, planes[o + 1]!, planes[o + 2]!)).toBeCloseTo(
        1,
        12,
      );
    }
    // The near plane sits 1 m in front of an eye at z = 10 looking down -Z, so
    // the eye is 1 m on the OUTSIDE of it: a signed distance of -1.
    const camPos = new Vector3(0, 0, 10);
    const o = 5 * 4;
    const signed =
      planes[o]! * camPos.x +
      planes[o + 1]! * camPos.y +
      planes[o + 2]! * camPos.z +
      planes[o + 3]!;
    expect(signed).toBeCloseTo(-1, 6);
  });
});

describe("intersectsAabb", () => {
  it.each(CASES)("agrees with three's intersectsBox (%s)", (_label, make) => {
    const camera = make();
    const m = clipFromWorld(camera);
    const planes = extractFrustumPlanes(
      m.elements,
      new Float64Array(24),
      "zero-to-one",
    );
    const want = oracle(m, WebGPUCoordinateSystem);

    // Deterministic pseudo-random boxes around the camera target.
    const target = new Vector3();
    camera.getWorldDirection(target).multiplyScalar(200).add(camera.position);
    let seed = 12345;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    let agreed = 0;
    for (let t = 0; t < 400; t++) {
      const cx = target.x + (rnd() - 0.5) * 900;
      const cy = target.y + (rnd() - 0.5) * 900;
      const cz = target.z + (rnd() - 0.5) * 900;
      const h = 1 + rnd() * 120;
      const box = new Box3(
        new Vector3(cx - h, cy - h, cz - h),
        new Vector3(cx + h, cy + h, cz + h),
      );
      const ours = intersectsAabb(
        planes,
        box.min.x,
        box.min.y,
        box.min.z,
        box.max.x,
        box.max.y,
        box.max.z,
      );
      expect(ours).toBe(want.intersectsBox(box));
      agreed++;
    }
    expect(agreed).toBe(400);
  });
});

describe("classifyAabb", () => {
  const camera = new PerspectiveCamera(90, 1, 1, 100);
  camera.position.set(0, 0, 0);
  camera.lookAt(new Vector3(0, 0, -1));
  const planes = extractFrustumPlanes(
    clipFromWorld(camera).elements,
    new Float64Array(24),
    "zero-to-one",
  );

  it("reports Inside for a small box well within the frustum", () => {
    expect(classifyAabb(planes, -1, -1, -20, 1, 1, -18)).toBe(
      Containment.Inside,
    );
  });

  it("reports Outside for a box behind the camera", () => {
    expect(classifyAabb(planes, -1, -1, 10, 1, 1, 20)).toBe(
      Containment.Outside,
    );
  });

  it("reports Intersecting for a box straddling the near plane", () => {
    expect(classifyAabb(planes, -1, -1, -2, 1, 1, 2)).toBe(
      Containment.Intersecting,
    );
  });

  it("never reports Outside where intersectsAabb reports true", () => {
    let seed = 999;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    for (let t = 0; t < 600; t++) {
      const cx = (rnd() - 0.5) * 200;
      const cy = (rnd() - 0.5) * 200;
      const cz = -rnd() * 150;
      const h = 0.5 + rnd() * 40;
      const c = classifyAabb(planes, cx - h, cy - h, cz - h, cx + h, cy + h, cz + h);
      const i = intersectsAabb(planes, cx - h, cy - h, cz - h, cx + h, cy + h, cz + h);
      expect(c === Containment.Outside).toBe(!i);
    }
  });

  // Inside must be sound, because the selector propagates it and skips the
  // plane test for an entire subtree on the strength of it.
  it("only reports Inside when every corner is inside", () => {
    let seed = 4242;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    for (let t = 0; t < 400; t++) {
      const cx = (rnd() - 0.5) * 120;
      const cy = (rnd() - 0.5) * 120;
      const cz = -1 - rnd() * 90;
      const h = 0.5 + rnd() * 25;
      if (classifyAabb(planes, cx - h, cy - h, cz - h, cx + h, cy + h, cz + h) !==
        Containment.Inside) continue;
      for (let corner = 0; corner < 8; corner++) {
        const x = cx + (corner & 1 ? h : -h);
        const y = cy + (corner & 2 ? h : -h);
        const z = cz + (corner & 4 ? h : -h);
        for (let p = 0; p < 6; p++) {
          const o = p * 4;
          const dist =
            planes[o]! * x + planes[o + 1]! * y + planes[o + 2]! * z + planes[o + 3]!;
          expect(dist).toBeGreaterThanOrEqual(-1e-9);
        }
      }
    }
  });
});
