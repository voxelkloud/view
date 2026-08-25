// The vertex layout for a BIM model's geometry, derived from what three's
// GLTFLoader hands back rather than assumed.
//
// A model written by `apps/runner/ifc.mjs` is quantised and interleaved into a
// 16-byte vertex: position as three uint16 over the model's box, normal as
// three int8, and the element's feature index as one uint16. That is not a
// space optimisation the renderer may ignore — it is the ONLY thing that keeps
// a hospital under a quarter of a gigabyte, so the renderer has to consume it
// where it lies.
//
// TWO UNIT SYSTEMS MEET HERE, and confusing them is the whole risk. three
// reports `stride` and `offset` in ARRAY ELEMENTS of the attribute's own typed
// array; WebGPU wants BYTES. The same 16-byte vertex is therefore `stride: 8`
// on a Uint16Array attribute and `stride: 16` on an Int8Array one. Multiply by
// BYTES_PER_ELEMENT and the two agree.
//
// WebGPU also has NO three-component 8- or 16-bit vertex format — only x2 and
// x4. The converter's padding is what makes that a non-issue: position's three
// uint16 are followed by two spare bytes, so `uint16x4` reads them in one go
// and the shader drops `.w`. Same for the normal's `snorm8x4`.

import type { BufferGeometry, InterleavedBufferAttribute } from "three/webgpu";

/** Where the shader expects each attribute. Mirrored in the WGSL. */
export const MESH_LOC = { position: 0, normal: 1, featureId: 2 } as const;

export interface MeshVertexLayout {
  /** Bytes between one vertex and the next. */
  readonly arrayStride: number;
  readonly attributes: readonly {
    readonly shaderLocation: number;
    readonly offset: number;
    readonly format: "uint16x4" | "snorm8x4" | "uint16x2";
  }[];
  /** The single interleaved buffer, as a byte range inside its ArrayBuffer. */
  readonly source: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
}

interface Interleavable {
  readonly isInterleavedBufferAttribute?: boolean;
  readonly offset?: number;
  readonly itemSize: number;
  readonly array: ArrayBufferView & { readonly BYTES_PER_ELEMENT: number };
  readonly data?: {
    readonly stride: number;
    readonly array: ArrayBufferView & { readonly BYTES_PER_ELEMENT: number };
  };
}

const byteStrideOf = (a: Interleavable): number =>
  (a.data?.stride ?? 0) * a.array.BYTES_PER_ELEMENT;

const byteOffsetOf = (a: Interleavable): number => (a.offset ?? 0) * a.array.BYTES_PER_ELEMENT;

/**
 * The layout for a quantised BIM geometry, or `undefined` when this is some
 * other mesh — a gizmo, the vehicle, anything three built in floats. Returning
 * `undefined` rather than throwing is deliberate: the overlay has always drawn
 * plain meshes and must keep doing so.
 */
export function quantisedMeshLayout(geo: BufferGeometry): MeshVertexLayout | undefined {
  const position = geo.attributes.position as unknown as Interleavable | undefined;
  const normal = geo.attributes.normal as unknown as Interleavable | undefined;
  const feature = geo.attributes._feature_id_0 as unknown as Interleavable | undefined;
  if (position === undefined || normal === undefined || feature === undefined) return undefined;
  if (position.isInterleavedBufferAttribute !== true) return undefined;
  if (normal.isInterleavedBufferAttribute !== true) return undefined;
  if (feature.isInterleavedBufferAttribute !== true) return undefined;

  // All three must read the same buffer, or they are not one vertex.
  const src = position.data?.array;
  if (src === undefined) return undefined;
  if (normal.data?.array.buffer !== src.buffer) return undefined;
  if (feature.data?.array.buffer !== src.buffer) return undefined;

  const arrayStride = byteStrideOf(position);
  if (arrayStride === 0 || arrayStride % 4 !== 0) return undefined;
  if (byteStrideOf(normal) !== arrayStride || byteStrideOf(feature) !== arrayStride)
    return undefined;

  // The formats the converter writes, checked rather than trusted: a model that
  // arrives as float32 positions would otherwise be read as uint16 noise.
  if (position.array.BYTES_PER_ELEMENT !== 2 || position.itemSize !== 3) return undefined;
  if (normal.array.BYTES_PER_ELEMENT !== 1 || normal.itemSize !== 3) return undefined;
  if (feature.array.BYTES_PER_ELEMENT !== 2 || feature.itemSize !== 1) return undefined;

  const attributes = [
    { shaderLocation: MESH_LOC.position, offset: byteOffsetOf(position), format: "uint16x4" as const },
    { shaderLocation: MESH_LOC.normal, offset: byteOffsetOf(normal), format: "snorm8x4" as const },
    { shaderLocation: MESH_LOC.featureId, offset: byteOffsetOf(feature), format: "uint16x2" as const },
  ];

  // A format that reads past the stride would sample the NEXT vertex. The
  // padding the converter leaves is what makes x4 legal here; assert it rather
  // than discover it as a diagonal smear on screen.
  const width = { uint16x4: 8, snorm8x4: 4, uint16x2: 4 };
  for (const a of attributes) {
    if (a.offset % 4 !== 0) return undefined;
    if (a.offset + width[a.format] > arrayStride) return undefined;
  }

  return {
    arrayStride,
    attributes,
    source: src.buffer as ArrayBuffer,
    byteOffset: src.byteOffset,
    byteLength: src.byteLength,
  };
}
