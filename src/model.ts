// Loading a BIM model layer into the scene the overlay already draws.
//
// This is the whole of what a caller has to do: `view.scene.add(await
// loadModelLayer(url))`. The overlay picks it up from there — `collect` walks
// the scene, `quantisedMeshLayout` recognises the geometry, and the mesh
// pipeline draws it against the points' depth.
//
// THE LOADER STAYS BEHIND A DYNAMIC IMPORT, like every other `three/addons`
// use in this repo: GLTFLoader is ~100 kB and an app that never opens a model
// must not pay for it. `packages/react` and `packages/vue` each have a test
// asserting the same thing about OrbitControls.

import { Group } from "three";
import type { Object3D } from "three";

export interface ModelLayer {
  /**
   * What to add to `view.scene` — the model in IFC coordinates, Z-up.
   *
   * NOT the glTF root. The GLB carries a `z-up-to-y-up` node so the file is
   * conformant in any viewer, and THIS scene is Z-up: adding the root would lay
   * the building on its side. Dropping that one node is exactly why the
   * converter kept the rotation separate from the vertices (see `ifc.mjs`), and
   * what remains still carries the dequantisation, so the numbers here are
   * metres in the model's own CRS — which is what the alignment of B4 and the
   * deviation field of B5 both need to work in.
   */
  readonly object: Object3D;
  /** The conformant glTF root, Y-up, for anything that wants the file's own frame. */
  readonly yUpRoot: Object3D;
  /** Elements with geometry, from `EXT_mesh_features`. */
  readonly featureCount: number;
  /** Draw calls this model will cost per frame: one per material batch. */
  readonly drawCalls: number;
}

interface GltfPrimitive {
  readonly extensions?: {
    readonly EXT_mesh_features?: { readonly featureIds?: readonly { featureCount?: number }[] };
  };
}

/**
 * Parse a `.glb` written by `apps/runner/ifc.mjs`.
 *
 * `bytes` rather than a URL: the caller already knows how to fetch under its
 * own auth, and a loader that fetches is a loader that has to be told about
 * cookies, tokens and CORS.
 */
export async function loadModelLayer(bytes: ArrayBuffer): Promise<ModelLayer> {
  const [{ GLTFLoader }, { MeshoptDecoder }] = await Promise.all([
    import(/* @vite-ignore */ "three/addons/loaders/GLTFLoader.js"),
    // O decodificador vem DENTRO do three, não de um pacote novo: um `.glb`
    // escrito por `ifc.mjs` usa `EXT_meshopt_compression` e sem isto o loader
    // recusa o arquivo. ~30 kB, e só carrega quando um modelo abre.
    import(/* @vite-ignore */ "three/addons/libs/meshopt_decoder.module.js"),
  ]);
  const loader = new GLTFLoader();
  await MeshoptDecoder.ready;
  loader.setMeshoptDecoder(MeshoptDecoder);
  const gltf = await new Promise<{
    scene: Object3D;
    parser: { json: { meshes?: { primitives: GltfPrimitive[] }[] } };
  }>((resolve, reject) => {
    loader.parse(bytes, "", resolve as (g: unknown) => void, reject);
  });

  // three splits a glTF mesh into one Object3D per PRIMITIVE, which is one per
  // material batch — so counting them counts the draw calls the converter
  // promised, from the file rather than from the report beside it.
  let drawCalls = 0;
  gltf.scene.traverse((o: Object3D & { isMesh?: boolean }) => {
    if (o.isMesh === true) drawCalls++;
  });

  const first = gltf.parser.json.meshes?.[0]?.primitives?.[0];
  const featureCount = first?.extensions?.EXT_mesh_features?.featureIds?.[0]?.featureCount ?? 0;

  // scene -> "z-up-to-y-up" -> the model. Verified against a converted GLB;
  // if the shape ever changes, falling back to the root is wrong-side-up and
  // obvious, which is better than silently drawing nothing.
  const rotation = gltf.scene.children[0];
  const zUp = rotation?.name === "z-up-to-y-up" ? rotation.children[0] : undefined;

  // WRAPPED, and this is not ceremony. The glTF node carries the
  // dequantisation as its own translation and scale, so a caller that writes
  // `object.position.set(...)` to place the model in the cloud's CRS silently
  // ERASES that translation and shifts the building by the corner of its own
  // bounding box. Handing back a fresh parent makes `position` mean what the
  // caller thinks it means, and leaves the alignment of B4 somewhere it can
  // live without fighting the file.
  const placed = new Group();
  placed.name = "model-placement";
  placed.add(zUp ?? gltf.scene);

  return { object: placed, yUpRoot: gltf.scene, featureCount, drawCalls };
}
