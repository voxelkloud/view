import type { DecodedPointData } from "@voxelkloud/format-potree";
import {
  BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  Sphere,
  Vector3,
} from "three";
import type { Group, Material } from "three";

/**
 * A view-aligned unit quad, shared in VALUE but never in IDENTITY.
 *
 * three's `Geometries.initGeometry` binds a dispose listener to the first render
 * object that used a geometry, so sharing one `BufferAttribute` instance across
 * geometries makes disposal order load-bearing. Each geometry gets its own copy.
 */
const CORNERS = new Float32Array([
  -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0,
]);
const CORNER_INDEX = [0, 1, 2, 0, 2, 3];

/**
 * Where decoded nodes go once they exist.
 *
 * Deliberately three methods, so a slab arena can satisfy it unchanged: nothing
 * above this interface may depend on "one object per node".
 */
export interface PointSink {
  /** Upload a decoded node. Returns the bytes it now holds on the GPU. */
  attach(index: number, data: DecodedPointData, spacingWorld: number, level: number): number;
  /** Release a node. */
  detach(index: number): void;
  /** Show exactly this set; everything else is hidden. */
  /** Points held on the GPU, live or not. See `PerNodeSink.residentPoints`. */
  readonly residentPoints: number;
  setVisible(indices: Int32Array, count: number): void;
  /** Flush pending GPU writes. Called once per frame after `setVisible`. */
  commit(): void;
  /**
   * Read back the resident CPU mirror for one node.
   *
   * Picking uses this rather than any GPU path so it works identically in the
   * WebGPU renderer and in the WebGL fallback.
   */
  readPoints(index: number): PointReadback | undefined;
  dispose(): void;
}

export interface PointReadback {
  readonly positions: Float32Array | Int32Array;
  readonly start: number;
  readonly count: number;
  readonly colors?: Uint8Array | Uint16Array;
  readonly scalars?: ArrayLike<number>;
}

interface NodeEntry {
  readonly mesh: Mesh;
  readonly geometry: InstancedBufferGeometry;
  readonly bytes: number;
  readonly pointCount: number;
}

/**
 * One `Mesh` + `InstancedBufferGeometry` per node.
 *
 * The straightforward sink, and the one that reaches pixels first. It has a
 * known ceiling: autzen's median node is 1446 points, so a 3M budget is ~1200
 * draw calls per frame, and three's `Bindings.delete` is `DataMap.delete`, which
 * never destroys the per-object uniform buffer — so a long streaming session
 * leaks one bind-group buffer per node ever attached. A slab arena fixes both
 * and satisfies this same interface; this stays as the fallback and as the
 * offline geometry-contract fixture.
 */
export class PerNodeSink implements PointSink {
  private readonly entries = new Map<number, NodeEntry>();
  private readonly visible = new Set<number>();
  private readonly shownEpoch = new Map<number, number>();
  private epoch = 0;
  private bytes = 0;
  private points = 0;

  constructor(
    private readonly parent: Group,
    private readonly material: Material,
    /**
     * The attribute the material's colour mode reads per point, if any —
     * `"intensity"` or `"classification"`. Decided once per cloud by the view,
     * which also selects it in the layout with a `f32` gpu lane.
     */
    private readonly scalarAttribute: string | undefined = undefined,
  ) {}

  /**
   * Points currently held on the GPU, live or not.
   *
   * Separate from `residentBytes` because the ratio against the LIVE count is
   * what says whether the draw list is worth compacting: the arena masks rather
   * than compacts, so every resident point costs vertex work whether or not it
   * is selected. Measured on autzen at a 0.25 framing, that ratio is 1.79 at
   * rest and 5.13 after a drag — 80% of the vertex work on points nobody is
   * looking at. Derived from bytes it was an estimate; here it is the number.
   */
  get residentPoints(): number {
    return this.points;
  }

  get residentBytes(): number {
    return this.bytes;
  }
  get nodeCount(): number {
    return this.entries.size;
  }

  attach(
    index: number,
    data: DecodedPointData,
    spacingWorld: number,
    level: number,
  ): number {
    if (this.entries.has(index)) return 0;
    if (data.numPoints === 0) return 0;
    // A cloud with no colour attribute is normal — LAS point format 1 carries
    // intensity and classification and no RGB. The colour attribute is still
    // built, because the material multiplies the splat diameter by its alpha.
    const srcColors =
      data.colors?.array instanceof Uint8Array ? data.colors.array : undefined;
    const colors =
      srcColors ?? new Uint8Array(4 * data.numPoints).fill(255);

    const g = new InstancedBufferGeometry();
    // A fresh corner attribute per geometry — see CORNERS.
    g.setAttribute("position", new BufferAttribute(CORNERS.slice(), 3));
    g.setIndex(CORNER_INDEX.slice());
    g.setAttribute(
      "pointOffset",
      new InstancedBufferAttribute(data.positions as Float32Array, 3, false),
    );
    // `normalized: true` is load-bearing: it yields unorm8x4 at 4 B/pt and a
    // float vec4 in the shader. With it false, three widens the array to
    // Uint32Array IN PLACE — 4x the GPU memory and an integer-typed shader
    // value — and it still renders, just wrong.
    g.setAttribute("color", new InstancedBufferAttribute(colors, 4, true));
    // Task 4's `f32` gpu lane. A raw uint8 classification would be a
    // 1-component Uint8Array, for which three's `_getVertexFormat` has no
    // entry: the pipeline descriptor gets `undefined` and the device rejects
    // the draw. The shader normalises it from the attribute's declared range.
    const scalars =
      this.scalarAttribute === undefined
        ? undefined
        : data.attributesByName.get(this.scalarAttribute)?.array;
    if (scalars instanceof Float32Array) {
      g.setAttribute(
        "scalarValue",
        new InstancedBufferAttribute(scalars, 1, false),
      );
    }
    g.instanceCount = data.numPoints;

    // Explicit, because the quad corners are the only thing three could infer a
    // bounding volume from and they are a unit square at the origin.
    const c = new Vector3();
    let radius = 1;
    if (data.bounds !== undefined) {
      const { min, max } = data.bounds;
      const o = data.frame.origin;
      c.set(
        (min[0] + max[0]) / 2 - o[0],
        (min[1] + max[1]) / 2 - o[1],
        (min[2] + max[2]) / 2 - o[2],
      );
      radius =
        0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    }
    g.boundingSphere = new Sphere(c, Math.max(radius, 1e-3));

    const mesh = new Mesh(g, this.material);
    mesh.frustumCulled = false;
    mesh.visible = false;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    // Read by the material's per-object uniforms, which land in the OBJECT bind
    // group and so never touch the pipeline cache key.
    mesh.userData['spacingWorld'] = spacingWorld;
    mesh.userData['level'] = level;
    this.parent.add(mesh);

    const bytes =
      data.positions.byteLength +
      colors.byteLength +
      (scalars instanceof Float32Array ? scalars.byteLength : 0);
    this.entries.set(index, { mesh, geometry: g, bytes, pointCount: data.numPoints });
    this.bytes += bytes;
    this.points += data.numPoints;
    return bytes;
  }

  detach(index: number): void {
    const e = this.entries.get(index);
    if (e === undefined) return;
    this.parent.remove(e.mesh);
    e.geometry.dispose();
    this.entries.delete(index);
    this.visible.delete(index);
    this.shownEpoch.delete(index);
    this.bytes -= e.bytes;
    this.points -= e.pointCount;
  }

  /**
   * Show exactly `indices[0..count)`.
   *
   * Epoch-stamped rather than set-differenced: marking the incoming set and then
   * sweeping the previously-visible one is O(shown + visible), where the obvious
   * "is each visible node still in the list" check is O(shown x visible) — 1200
   * x 1200 per frame at a 3M budget.
   */
  setVisible(indices: Int32Array, count: number): void {
    const epoch = ++this.epoch;
    for (let k = 0; k < count; k++) {
      const i = indices[k]!;
      const e = this.entries.get(i);
      if (e === undefined) continue;
      this.shownEpoch.set(i, epoch);
      if (!this.visible.has(i)) {
        e.mesh.visible = true;
        this.visible.add(i);
      }
    }
    for (const i of this.visible) {
      if (this.shownEpoch.get(i) === epoch) continue;
      const e = this.entries.get(i);
      if (e !== undefined) e.mesh.visible = false;
      this.visible.delete(i);
    }
  }

  commit(): void {
    // Nothing to flush: each geometry owns its buffers and three uploads them on
    // first render. The arena sink is where this becomes a partial writeBuffer.
  }

  readPoints(index: number): PointReadback | undefined {
    const entry = this.entries.get(index);
    if (entry === undefined) return undefined;
    const positions = entry.geometry.attributes["pointOffset"]?.array;
    if (!(positions instanceof Float32Array)) return undefined;
    const colors = entry.geometry.attributes["color"]?.array;
    const scalars = entry.geometry.attributes["scalarValue"]?.array;
    return {
      positions,
      start: 0,
      count: positions.length / 3,
      ...(colors instanceof Uint8Array || colors instanceof Uint16Array
        ? { colors }
        : {}),
      ...(scalars instanceof Float32Array ||
      scalars instanceof Int32Array ||
      scalars instanceof Uint32Array ||
      scalars instanceof Uint16Array ||
      scalars instanceof Uint8Array ||
      scalars instanceof Int16Array ||
      scalars instanceof Int8Array
        ? { scalars }
        : {}),
    };
  }

  dispose(): void {
    for (const [, e] of this.entries) {
      this.parent.remove(e.mesh);
      e.geometry.dispose();
    }
    this.entries.clear();
    this.visible.clear();
    this.shownEpoch.clear();
    this.bytes = 0;
    this.points = 0;
  }

  /** Test/diagnostic access. */
  meshFor(index: number): Mesh | undefined {
    return this.entries.get(index)?.mesh;
  }
}
