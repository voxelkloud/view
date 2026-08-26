import { readFileSync } from "node:fs";
import { openPotreePoints, parsePointCloudSource } from "@voxelkloud/format-potree";
import type { PointCloudSource } from "@voxelkloud/format-potree";
import { describe, expect, it } from "vitest";
import { pointDecodeSelection } from "./view.js";

const URLS = {
  base: "https://x/",
  metadata: "https://x/metadata.json",
  hierarchy: "https://x/hierarchy.bin",
  octree: "https://x/octree.bin",
};

/** Carries BOTH `rgb` and `classification`, which is the case that matters. */
function autzen(): PointCloudSource {
  return parsePointCloudSource(
    JSON.parse(
      readFileSync(
        new URL("./__fixtures__/autzen.metadata.json", import.meta.url).pathname,
        "utf8",
      ),
    ) as Record<string, unknown>,
    URLS,
  );
}

/** The same cloud with an attribute dropped, to reach the branches autzen cannot. */
function without(name: string): PointCloudSource {
  const source = autzen();
  const attributes = source.attributes.filter((a) => a.name !== name);
  const attributesByName = new Map(attributes.map((a) => [a.name, a]));
  return { ...source, attributes, attributesByName } as PointCloudSource;
}

describe("pointDecodeSelection", () => {
  it("keeps the colour when it adds the class to an RGB cloud", () => {
    // THE regression this guards: naming an attribute replaces the default
    // selection rather than adding to it, so a class asked for without naming
    // the colour back would leave an RGB cloud grey.
    const sel = pointDecodeSelection(autzen(), undefined);
    expect(sel.attributes).toEqual(["rgb", "classification"]);
    expect(sel.lanes).toEqual({ classification: "f32" });
    expect(sel.scalarFormat).toBe("gpu");
  });

  it("names the class once when the colour mode already paints by it", () => {
    const sel = pointDecodeSelection(autzen(), "classification");
    expect(sel.attributes).toEqual(["classification"]);
    expect(sel.lanes).toEqual({ classification: "f32" });
  });

  it("carries the class alongside another scalar, and still drops colour", () => {
    // Intensity deselects RGB as it always did; the class rides along because
    // hiding one has to work in the intensity mode too.
    const sel = pointDecodeSelection(autzen(), "intensity");
    expect(sel.attributes).toEqual(["intensity", "classification"]);
    expect(sel.lanes).toEqual({ intensity: "f32", classification: "f32" });
  });

  it("asks for nothing on a cloud with no class and no scalar mode", () => {
    // Saying nothing IS the default selection of position + colour, so this is
    // byte for byte what the RGB path decoded before the class existed.
    expect(pointDecodeSelection(without("classification"), undefined)).toEqual({});
  });

  it("names only the scalar when the cloud has no class", () => {
    const sel = pointDecodeSelection(without("classification"), "intensity");
    expect(sel.attributes).toEqual(["intensity"]);
    expect(sel.lanes).toEqual({ intensity: "f32" });
  });

  it("takes the class alone on a cloud with no colour", () => {
    // LAS point format 1: intensity and classification, no RGB. There is no
    // colour to name back, so the list must not carry an undefined hole.
    const sel = pointDecodeSelection(without("rgb"), undefined);
    expect(sel.attributes).toEqual(["classification"]);
  });

  it("names attributes the driver actually accepts", () => {
    // The selection is verbatim SOURCE names, and this cloud spells its colour
    // "rgb" — a plausible guess like "color" would throw rather than be
    // ignored. Opening the reader for real is what catches that; the reader
    // exposes no list of what it selected, so this proves the names are legal
    // and not that the lanes were filled. That needs a decoded node.
    const source = autzen();
    for (const scalarAttribute of [undefined, "intensity", "classification"]) {
      expect(() =>
        openPotreePoints(source, pointDecodeSelection(source, scalarAttribute)),
      ).not.toThrow();
    }
    // The anchor: without it, "does not throw" would pass on a driver that
    // ignored the selection entirely.
    expect(() => openPotreePoints(source, { attributes: ["color"] })).toThrow();
  });
});
