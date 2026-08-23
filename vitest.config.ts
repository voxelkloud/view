import { defineConfig } from "vitest/config";

export default defineConfig({
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
