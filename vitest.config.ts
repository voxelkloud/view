import { defineConfig } from "vitest/config";

export default defineConfig({
  // The default 5s is not a latency budget anyone chose — it is vitest's, and
  // two files sit right under it: `lod/select` runs 1000 real selection frames
  // over the Autzen octree (~4.8s here) and `cut` walks the same tree. Neither
  // asserts on wall clock, so a timeout firing means a loaded machine, not a
  // regression — and on CI, where the publish workflow runs these before it
  // pushes a tarball, a flaky timeout would fail a release for nothing.
  test: { testTimeout: 30_000 },
  resolve: {
    alias: {
      // `#brotli-native` is a subpath import declared in
      // @voxelkloud/format-potree's package.json. Node resolves it; Vite does
      // not, once the importing file is reached across a workspace link from
      // THIS package. Nothing here decompresses, so the no-op branch is the
      // honest stand-in — and pointing it at the Node branch instead would make
      // these tests depend on a build artefact of another package.
      "#brotli-native": new URL(
        "../format-potree/src/point-data-brotli-none.ts",
        import.meta.url,
      ).pathname,
    },
  },
});
