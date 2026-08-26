import { defineConfig } from "tsup";

export default defineConfig({
  // `lod` is a separate entry, not bundled into the main one: its module graph
  // imports no three, no DOM and no GPU, and the `@voxelkloud/view/lod` subpath
  // is what makes that guarantee usable from a worker or under SSR.
  // `material` and `edl` are separate entries for the OPPOSITE reason to `lod`:
  // their graphs pull three's WebGPU build, and as root exports they made that
  // edge static for every consumer.
  entry: [
    "src/index.ts",
    "src/lod/index.ts",
    "src/profile/index.ts",
    "src/material-entry.ts",
    "src/edl-entry.ts",
  ],
  format: ["esm"],
  dts: true,
  sourcemap: true,
  clean: true,
  external: ["three", "three/webgpu", "three/tsl"],
});
