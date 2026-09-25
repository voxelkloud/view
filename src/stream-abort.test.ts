// O CANCELAMENTO, contra a hierarquia real em vez de contra booleanos.
//
// `stream-policy.test.ts` tranca a regra; isto tranca o SINAL. A pergunta que
// nenhum teste de política responde é se o frustum chega sequer a declarar um
// nó fora: se `intersectsAabb` dissesse "dentro" para a nuvem toda, a política
// estaria correcta e não cancelaria um único byte, e a única forma de descobrir
// seria voar sobre um dataset a olhar para o HUD.
//
// A montagem é a de `lod/select.test.ts` — o mesmo autzen vendorizado, a mesma
// câmara — porque o que está em causa é a forma de uma árvore a sério.
import { readFileSync } from "node:fs";
import { createHierarchy, parsePointCloudSource } from "@voxelkloud/format-potree";
import type { PointCloudHierarchy } from "@voxelkloud/format-potree";
import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { beforeAll, describe, expect, it } from "vitest";
import { extractFrustumPlanes, intersectsAabb } from "./lod/frustum.js";
import {
  createLodScratch,
  createLodSelection,
  resolveLodOptions,
  selectVisible,
} from "./lod/select.js";
import type { LodCameraState, LodScratch } from "./lod/select.js";
import { ABORT_OUTSIDE_FRAMES, shouldAbortFetch } from "./stream-policy.js";

const LOADER_FIXTURES = new URL("./__fixtures__/", import.meta.url);
const FAKE_URLS = {
  base: "https://example.test/cloud/",
  metadata: "https://example.test/cloud/metadata.json",
  hierarchy: "https://example.test/cloud/hierarchy.bin",
  octree: "https://example.test/cloud/octree.bin",
};

let autzen: PointCloudHierarchy;

beforeAll(async () => {
  const json = JSON.parse(
    readFileSync(new URL("autzen.metadata.json", LOADER_FIXTURES), "utf8"),
  ) as Record<string, unknown>;
  const source = parsePointCloudSource(json, FAKE_URLS);
  const buf = readFileSync(new URL("autzen.hierarchy.bin", LOADER_FIXTURES));
  autzen = createHierarchy(source, {
    buffer: buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  });
  await autzen.expandAll();
});

/** A câmara a olhar para o centro a partir de um ângulo, como a view a monta. */
function cameraAt(
  tree: PointCloudHierarchy,
  offset: readonly [number, number, number],
  scratch: LodScratch,
): LodCameraState {
  const box = tree.source.metadata.boundingBox;
  const centre = new Vector3(
    (box.min[0] + box.max[0]) / 2,
    (box.min[1] + box.max[1]) / 2,
    (box.min[2] + box.max[2]) / 2,
  );
  const camera = new PerspectiveCamera(60, 16 / 9, 1, 50_000);
  camera.up.set(0, 0, 1);
  camera.position.set(centre.x + offset[0], centre.y + offset[1], centre.z + offset[2]);
  camera.lookAt(centre);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();
  const clip = new Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse,
  );
  const cam: LodCameraState = {
    clipFromAbs: new Float64Array(clip.elements),
    camX: camera.position.x,
    camY: camera.position.y,
    camZ: camera.position.z,
    slope: Math.tan(((camera.fov * Math.PI) / 180) / 2),
    viewportHeightPx: 1080,
    orthographic: false,
    orthoProjFactor: 0,
    nearFloor: camera.near,
    depthRange: "minus-one-to-one",
    reversedDepth: false,
  };
  extractFrustumPlanes(cam.clipFromAbs, scratch.planes, cam.depthRange, false);
  return cam;
}

/** Quantos dos nós de `indices` o frustum de `scratch` deixa de fora. */
function outsideCount(indices: readonly number[], scratch: LodScratch): number {
  let n = 0;
  for (const i of indices) {
    const node = autzen.node(i)!;
    if (
      !intersectsAabb(
        scratch.planes,
        node.minX,
        node.minY,
        node.minZ,
        node.maxX,
        node.maxY,
        node.maxZ,
      )
    ) {
      n++;
    }
  }
  return n;
}

describe("o sinal de cancelamento sobre o autzen real", () => {
  it("põe fora do frustum quase toda a selecção quando a câmara se vira", () => {
    const s = createLodScratch();
    const out = createLodSelection(4096);
    const opts = resolveLodOptions({});

    // Quadro 1: a câmara perto, de um lado. A selecção em ordem de pop É a
    // ordem por que o `stream` despacha.
    const near = cameraAt(autzen, [300, 300, 300], s);
    selectVisible(autzen, near, opts, s, out);
    const selected = Array.from(out.indices.subarray(0, out.count));
    expect(selected.length).toBeGreaterThan(200);
    // Nada seleccionado está fora enquanto a câmara não se mexe — senão o resto
    // do teste não estaria a medir a viragem.
    expect(outsideCount(selected, s)).toBe(0);

    // Quadro 2: mesma distância, lado oposto.
    const turned = cameraAt(autzen, [-300, -300, 300], s);
    selectVisible(autzen, turned, opts, s, out);

    // 264 de 272 medidos. É a dimensão do desperdício: sem cancelamento, cada
    // um destes que estivesse em voo acabava de descarregar para não pintar
    // pixel nenhum, com a frente visível na fila atrás dele.
    const gone = outsideCount(selected, s);
    expect(gone / selected.length).toBeGreaterThan(0.95);

    // A JANELA DE DESPACHO, que é o que interessa na prática. Os doze primeiros
    // são os mais grosseiros — caixas enormes que continuam a intersectar do
    // outro lado, e só 5 dos 12 saem. Mas esses doze estão residentes ao fim do
    // primeiro segundo, e a partir daí a janela vive fundo na lista, onde a
    // viragem põe fora TODOS.
    const frontier = selected.slice(100, 112);
    expect(outsideCount(frontier, s)).toBe(frontier.length);

    // E que a política os cancele mesmo, passada a histerese e SEM saturação —
    // que é a diferença que faz isto funcionar numa ligação lenta e vazia.
    const policy = {
      abortOutsideFrustum: true,
      abortSuperseded: false,
      saturated: false,
    };
    let cancelled = 0;
    for (const i of frontier) {
      const node = autzen.node(i)!;
      const outside = !intersectsAabb(
        s.planes,
        node.minX, node.minY, node.minZ,
        node.maxX, node.maxY, node.maxZ,
      );
      if (shouldAbortFetch(outside, ABORT_OUTSIDE_FRAMES, policy)) cancelled++;
    }
    expect(cancelled).toBe(frontier.length);
  });

  it("uma viragem de 90 graus chega, não é preciso dar meia volta", () => {
    const s = createLodScratch();
    const out = createLodSelection(4096);
    const opts = resolveLodOptions({});
    const near = cameraAt(autzen, [300, 300, 300], s);
    selectVisible(autzen, near, opts, s, out);
    const selected = Array.from(out.indices.subarray(0, out.count));

    // 252 de 272. O caso comum de uma órbita, não o extremo.
    const quarter = cameraAt(autzen, [-424, 0, 300], s);
    selectVisible(autzen, quarter, opts, s, out);
    expect(outsideCount(selected, s) / selected.length).toBeGreaterThan(0.9);
  });

  it("não cancela nada num quadro em que a câmara não se mexeu", () => {
    const s = createLodScratch();
    const out = createLodSelection(4096);
    const cam = cameraAt(autzen, [300, 300, 300], s);
    selectVisible(autzen, cam, resolveLodOptions({}), s, out);

    const policy = {
      abortOutsideFrustum: true,
      abortSuperseded: true,
      saturated: true,
    };
    for (let k = 0; k < out.count; k++) {
      // `staleFrames` 0 é "seleccionado neste quadro", que é o estado de tudo o
      // que acabou de sair do `selectVisible`.
      expect(shouldAbortFetch(false, 0, policy)).toBe(false);
    }
  });
});
