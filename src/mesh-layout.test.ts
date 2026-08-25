import { BufferAttribute, BufferGeometry, InterleavedBuffer, InterleavedBufferAttribute } from "three/webgpu";
import { describe, expect, it } from "vitest";
import { MESH_LOC, quantisedMeshLayout } from "./mesh-layout.js";

/**
 * The 16-byte vertex `apps/runner/ifc.mjs` writes, in the exact shape three's
 * GLTFLoader hands back for it — verified against a real converted GLB:
 *
 *   position       Uint16Array  stride 8   offset 0    -> bytes  0..5  (+2 pad)
 *   normal         Int8Array    stride 16  offset 8    -> bytes  8..10 (+1 pad)
 *   _feature_id_0  Uint16Array  stride 8   offset 6    -> byte  12..13 (+2 pad)
 *
 * Two typed arrays over ONE ArrayBuffer, so `stride` differs between them while
 * the vertex does not. That is the trap this function exists to not fall into.
 */
function quantisedGeometry(vertexCount = 4): BufferGeometry {
  const bytes = new ArrayBuffer(vertexCount * 16);
  const u16 = new InterleavedBuffer(new Uint16Array(bytes), 8);
  const i8 = new InterleavedBuffer(new Int8Array(bytes), 16);
  const geo = new BufferGeometry();
  geo.setAttribute("position", new InterleavedBufferAttribute(u16, 3, 0, false));
  geo.setAttribute("normal", new InterleavedBufferAttribute(i8, 3, 8, true));
  geo.setAttribute("_feature_id_0", new InterleavedBufferAttribute(u16, 1, 6, false));
  return geo;
}

describe("quantisedMeshLayout", () => {
  it("reads three's element strides as WebGPU byte strides", () => {
    const layout = quantisedMeshLayout(quantisedGeometry())!;
    expect(layout).toBeDefined();
    // 8 Uint16 elements and 16 Int8 elements are the SAME 16 bytes.
    expect(layout.arrayStride).toBe(16);
    expect(layout.attributes).toEqual([
      { shaderLocation: MESH_LOC.position, offset: 0, format: "uint16x4" },
      { shaderLocation: MESH_LOC.normal, offset: 8, format: "snorm8x4" },
      { shaderLocation: MESH_LOC.featureId, offset: 12, format: "uint16x2" },
    ]);
  });

  it("covers the whole vertex and never reads into the next one", () => {
    const layout = quantisedMeshLayout(quantisedGeometry())!;
    const width = { uint16x4: 8, snorm8x4: 4, uint16x2: 4 };
    for (const a of layout.attributes) {
      expect(a.offset % 4).toBe(0);
      expect(a.offset + width[a.format]).toBeLessThanOrEqual(layout.arrayStride);
    }
  });

  it("points at the one buffer the three attributes share", () => {
    const geo = quantisedGeometry(10);
    const layout = quantisedMeshLayout(geo)!;
    expect(layout.byteLength).toBe(160);
    expect(layout.byteOffset).toBe(0);
  });

  it("declines a plain float geometry rather than reading it as uint16", () => {
    // The vehicle and the measurement gizmos look like this, and the overlay
    // has always drawn them. Misreading one as quantised would smear it across
    // the scene instead of failing.
    const geo = new BufferGeometry();
    geo.setAttribute("position", new BufferAttribute(new Float32Array(9), 3));
    geo.setAttribute("normal", new BufferAttribute(new Float32Array(9), 3));
    expect(quantisedMeshLayout(geo)).toBeUndefined();
  });

  it("declines when the feature id is missing", () => {
    const geo = quantisedGeometry();
    geo.deleteAttribute("_feature_id_0");
    expect(quantisedMeshLayout(geo)).toBeUndefined();
  });

  it("declines when the attributes do not share one buffer", () => {
    const geo = quantisedGeometry();
    const other = new InterleavedBuffer(new Int8Array(64), 16);
    geo.setAttribute("normal", new InterleavedBufferAttribute(other, 3, 8, true));
    expect(quantisedMeshLayout(geo)).toBeUndefined();
  });

  it("declines a stride WebGPU cannot bind", () => {
    const bytes = new ArrayBuffer(4 * 14);
    const u16 = new InterleavedBuffer(new Uint16Array(bytes), 7); // 14 bytes
    const i8 = new InterleavedBuffer(new Int8Array(bytes), 14);
    const geo = new BufferGeometry();
    geo.setAttribute("position", new InterleavedBufferAttribute(u16, 3, 0, false));
    geo.setAttribute("normal", new InterleavedBufferAttribute(i8, 3, 8, true));
    geo.setAttribute("_feature_id_0", new InterleavedBufferAttribute(u16, 1, 6, false));
    expect(quantisedMeshLayout(geo)).toBeUndefined();
  });
});
