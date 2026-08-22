import { defineConfig } from "tsup";

export default defineConfig({
  // `lod` is a separate entry, not bundled into the main one: its module graph
  // imports no three, no DOM and no GPU, and the `@voxelkloud/view/lod` subpath
  // is what makes that guarantee usable from a worker or under SSR.
  entry: ["src/index.ts", "src/lod/index.ts", "src/profile/index.ts"],
  format: ["esm"],
  dts: true,
  sourcemap: true,
  clean: true,
  external: ["three", "three/webgpu", "three/tsl"],
});
