import { describe, expect, it } from "vitest";
import { extractProfile } from "./index.js";
import type { ProfileCloudContext, ProfileQuery } from "./index.js";

/**
 * The smallest context the option resolution and the first traversal step will
 * accept: one root node the corridor misses, so the generator resolves options,
 * finds nothing, and completes. That is all these tests need — they are about
 * what `extractProfile` does with its OPTIONS, before any point is read.
 */
function stubContext(): ProfileCloudContext {
  const root = {
    index: 0,
    level: 0,
    childMask: 0,
    numPoints: 0,
    minX: 0,
    minY: 0,
    minZ: 0,
    maxX: 1,
    maxY: 1,
    maxZ: 1,
  };
  return {
    source: {
      pointCount: 0,
      attributes: [],
      tightBoundingBox: { min: [0, 0, 0], max: [1, 1, 1] },
    },
    hierarchy: {
      root,
      node: () => root,
      tryExpandSync: () => true,
      expand: async () => undefined,
    },
    openPoints: () => ({
      hasPayload: () => false,
      read: async () => {
        throw new Error("no node should be read by these tests");
      },
      close: () => undefined,
    }),
  } as unknown as ProfileCloudContext;
}

// Far from the stub root's unit box, so traversal rejects it immediately.
const QUERY: ProfileQuery = {
  kind: "vertical",
  points: [
    [500, 500],
    [500, 600],
  ],
  width: 1,
};

async function drain(options: Parameters<typeof extractProfile>[2]): Promise<number> {
  let batches = 0;
  for await (const _ of extractProfile(stubContext(), QUERY, options)) batches++;
  return batches;
}

describe("extractProfile options", () => {
  it("treats an omitted maxDepth as unlimited", async () => {
    // REGRESSION: the call site read `options.maxDepth ?? POSITIVE_INFINITY`
    // while the validator accepted only `undefined` as the unlimited sentinel,
    // so omitting the documented default threw RangeError every time. Nothing
    // covered this module, which is how it survived.
    await expect(drain({})).resolves.toBeGreaterThanOrEqual(0);
    await expect(drain(undefined)).resolves.toBeGreaterThanOrEqual(0);
  });

  it("accepts an explicit infinite maxDepth", async () => {
    await expect(drain({ maxDepth: Number.POSITIVE_INFINITY })).resolves.toBeGreaterThanOrEqual(0);
  });

  it("accepts a finite maxDepth", async () => {
    await expect(drain({ maxDepth: 0 })).resolves.toBeGreaterThanOrEqual(0);
    await expect(drain({ maxDepth: 5 })).resolves.toBeGreaterThanOrEqual(0);
  });

  it("rejects a negative or NaN maxDepth", async () => {
    await expect(drain({ maxDepth: -1 })).rejects.toThrow(RangeError);
    await expect(drain({ maxDepth: Number.NaN })).rejects.toThrow(RangeError);
  });
});
