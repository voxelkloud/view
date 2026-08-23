import { readFileSync } from "node:fs";
import { openPotreePoints, parsePointCloudSource } from "@voxelkloud/format-potree";
import type { DecodedPointData, PointCloudSource } from "@voxelkloud/format-potree";
import { Group, MeshBasicMaterial } from "three/webgpu";
import { describe, expect, it } from "vitest";
import { PointArena } from "./arena.js";
import { scalarAttributeFor } from "./material.js";
import { ArenaSink } from "./sink-arena.js";
import { PerNodeSink } from "./sink.js";
import { scalarRangeFor } from "./view.js";

const URLS = {
  base: "https://x/",
  metadata: "https://x/metadata.json",
  hierarchy: "https://x/hierarchy.bin",
  octree: "https://x/octree.bin",
};

function autzen(): PointCloudSource {
  return parsePointCloudSource(
    JSON.parse(
      readFileSync(
        new URL(
          "../../format-potree/src/__fixtures__/autzen.metadata.json",
          import.meta.url,
        ).pathname,
        "utf8",
      ),
    ) as Record<string, unknown>,
    URLS,
  );
}

// `scalarRangeFor` reads the packing off the READER now, not off a layout: the
// transform has to be whatever the driver's decoder actually applied, and there
// is no layout object in a neutral view any more.
function readerFor(source: PointCloudSource, name: string) {
  return openPotreePoints(source, {
    attributes: [name],
    scalarFormat: "gpu",
    lanes: { [name]: "f32" },
  });
}

describe("scalarRangeFor", () => {
  it("uses the attribute's declared range when the lane is unpacked", () => {
    const source = autzen();
    // intensity is uint16 0..254 here, and 2 bytes wide, so Task 2 attaches no
    // normalization and the decoded f32 carries raw counts.
    expect(scalarRangeFor(source, readerFor(source, "intensity"), "intensity"))
      .toEqual([0, 254]);
  });

  it("applies the reader's packing transform rather than recomputing it", () => {
    const source = autzen();
    // gps-time is a double, so it is normalised to 0..1 and the range must
    // follow the lane, not the manifest. Handing the shader 245369..249783
    // against a 0..1 value pins every point to the ramp's first stop.
    const reader = readerFor(source, "gps-time");
    expect(reader.packingFor("gps-time")).toBeDefined();
    expect(scalarRangeFor(source, reader, "gps-time")).toEqual([0, 1]);
  });

  it("falls back to the 16-bit LAS range on a degenerate declared range", () => {
    const source = autzen();
    // A manifest that never filled min/max in would otherwise collapse the ramp
    // to its first stop for every point.
    const flat = {
      ...source,
      attributesByName: new Map(source.attributesByName).set("intensity", {
        ...source.attributesByName.get("intensity")!,
        min: [7],
        max: [7],
      }),
    } as PointCloudSource;
    expect(scalarRangeFor(flat, readerFor(source, "intensity"), "intensity"))
      .toEqual([0, 65535]);
  });

  it("is unaffected by an attribute the reader does not carry", () => {
    const source = autzen();
    expect(
      scalarRangeFor(source, readerFor(source, "intensity"), "classification"),
    ).toEqual([1, 2]);
  });
});

describe("scalarAttributeFor", () => {
  it("names an attribute only for the modes that read one", () => {
    expect(scalarAttributeFor({ kind: "intensity" })).toBe("intensity");
    expect(scalarAttributeFor({ kind: "classification" })).toBe(
      "classification",
    );
    for (const kind of ["rgb", "elevation", "level"] as const) {
      expect(scalarAttributeFor({ kind })).toBeUndefined();
    }
    expect(scalarAttributeFor({ kind: "flat", color: [1, 0, 0] })).toBeUndefined();
  });
});

function decoded(n: number, scalar?: Float32Array): DecodedPointData {
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) positions[i] = i;
  const attributes =
    scalar === undefined
      ? []
      : [
          {
            name: "intensity",
            role: undefined,
            itemSize: 1,
            array: scalar,
            gpuFormat: "float32",
            normalized: false,
          },
        ];
  return {
    nodeIndex: 0,
    nodeName: "r",
    numPoints: n,
    positions,
    frame: { kind: "cloud-relative", origin: [0, 0, 0] },
    colors: undefined,
    attributes,
    attributesByName: new Map(attributes.map((a) => [a.name, a])),
    bounds: undefined,
    transferList: [],
    byteLength: positions.byteLength + (scalar?.byteLength ?? 0),
  } as unknown as DecodedPointData;
}

describe("sinks: the scalar lane", () => {
  it("PerNodeSink binds scalarValue only when a scalar mode asked for one", () => {
    const parent = new Group();
    const plain = new PerNodeSink(parent, new MeshBasicMaterial());
    plain.attach(0, decoded(8, new Float32Array(8).fill(3)), 1, 0);
    const g0 = parent.children[0]!;
    expect(
      (g0 as unknown as { geometry: { attributes: Record<string, unknown> } })
        .geometry.attributes["scalarValue"],
    ).toBeUndefined();

    const host = new Group();
    const sink = new PerNodeSink(host, new MeshBasicMaterial(), "intensity");
    const values = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const bytes = sink.attach(0, decoded(8, values), 1, 0);
    const attr = (
      host.children[0] as unknown as {
        geometry: { attributes: Record<string, { array: Float32Array }> };
      }
    ).geometry.attributes["scalarValue"]!;
    expect(Array.from(attr.array)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    // The lane is real memory and must show up in the residency accounting, or
    // the point budget silently overshoots.
    expect(bytes).toBe(8 * 3 * 4 + 8 * 4 + 8 * 4);
  });

  it("ArenaSink writes the scalar into the slab at the block's offset", () => {
    const parent = new Group();
    const sink = new ArenaSink(
      parent,
      new MeshBasicMaterial(),
      () => 1,
      () => 1,
      false,
      { pageSize: 4, slabCapacity: 64 },
      "intensity",
    );
    const bytes = sink.attach(0, decoded(4, new Float32Array([9, 8, 7, 6])), 1, 0);
    sink.attach(1, decoded(4, new Float32Array([5, 4, 3, 2])), 1, 0);
    sink.commit();

    // Staged bytes, which is what the view's per-frame upload budget spends.
    // Positions 12 B/pt, the white colour the arena fills in is not staged from
    // here, and the scalar lane is 4 B/pt.
    expect(bytes).toBe(4 * 12 + 4 * 4);

    const slab = parent.children[0] as unknown as {
      geometry: { attributes: Record<string, { array: Float32Array }> };
    };
    const values = slab.geometry.attributes["scalarValue"]!.array;
    // Page-aligned, so the second node starts at slot 4, not slot 8.
    expect(Array.from(values.subarray(0, 8))).toEqual([9, 8, 7, 6, 5, 4, 3, 2]);
  });

  it("leaves the slab layout alone when no mode reads a scalar", () => {
    const parent = new Group();
    const arena = new PointArena(
      parent,
      new MeshBasicMaterial(),
      () => 1,
      () => 1,
      { pageSize: 4, slabCapacity: 64 },
    );
    const block = arena.allocate(0, 4)!;
    arena.stage(block, new Float32Array(12), undefined, false);
    arena.commit();
    const slab = parent.children[0] as unknown as {
      geometry: { attributes: Record<string, unknown> };
    };
    // 4 bytes per point that nothing reads is 200 MB on a 50M-point cloud.
    expect(slab.geometry.attributes["scalarValue"]).toBeUndefined();
  });
});

describe("the classification palette", () => {
  it("covers every ASPRS class and nothing else", async () => {
    const src = readFileSync(new URL("./material.ts", import.meta.url).pathname, "utf8");
    const codes = [...src.matchAll(/^\s{2}\[(\d+), \[/gm)].map((m) => Number(m[1]));
    // 0..18 is the LAS 1.4 standard set. Listing all of them is what lets the
    // fallback colour mean "outside the standard" rather than "not common".
    expect(codes).toEqual(Array.from({ length: 19 }, (_, i) => i));
  });

  it("gives unknown codes a colour no class uses", () => {
    const src = readFileSync(new URL("./material.ts", import.meta.url).pathname, "utf8");
    const unknown = /UNKNOWN_CLASS: readonly \[number, number, number\] = \[([^\]]+)\]/
      .exec(src)?.[1]
      ?.split(",")
      .map((s) => Number(s.trim()));
    expect(unknown).toHaveLength(3);
    const classes = [...src.matchAll(/^\s{2}\[\d+, \[([^\]]+)\]\]/gm)].map((m) =>
      m[1]!.split(",").map((s) => Number(s.trim())),
    );
    // Not merely different: FAR. A grey one shade off "unclassified" is what
    // this replaced, and on a survey that is 80% class 1 the two were the same
    // picture.
    for (const c of classes) {
      const d = Math.hypot(c[0]! - unknown![0]!, c[1]! - unknown![1]!, c[2]! - unknown![2]!);
      expect(d, `class colour ${c.join(",")} is too close to the unknown colour`)
        .toBeGreaterThan(0.35);
    }
  });
});
