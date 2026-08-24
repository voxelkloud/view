# @voxelkloud/view

WebGPU point cloud renderer on three.js, consuming
[@voxelkloud/loader](../loader/README.md) streams.

```sh
npm install @voxelkloud/view @voxelkloud/loader three
```

```ts
import { loadHierarchy, loadPointCloudSource } from "@voxelkloud/loader";
import { createPointCloudView } from "@voxelkloud/view";

const source = await loadPointCloudSource(url);
const hierarchy = await loadHierarchy(source);
await hierarchy.expandAll();

const view = createPointCloudView({ canvas });
await view.init();
view.addCloud(source, hierarchy);
view.frameCloud();
view.setSize(canvas.clientWidth, canvas.clientHeight, devicePixelRatio);

const tick = () => {
  requestAnimationFrame(tick);
  view.renderFrame();
};
tick();
```

`view.camera` and `view.scene` are the three objects, so OrbitControls and every
other add-on attach normally.

Instanced quads, NOT `Points`. three's WebGPU backend maps `object.isPoints` to
`point-list` topology, and WGSL has no point-size builtin, so `sizeNode` is
silently ignored there and every point rasterises as one pixel with no
attenuation. Instanced quads are the only path to a sized splat.

Colour modes: `rgb`, `elevation`, `level`, `intensity`, `classification`,
`flat`. The two scalar modes select a different decode layout, so they are set
when the view is built rather than toggled.

Eye-dome lighting via `edl: { strength, radius, opacity }`, using Potree's
constants so an `edlStrength` transfers unchanged. Off by default.

Splat size comes from the LOCAL depth of the selection, not from the level of
the node a point came from. Each point walks the selected octree cut in the
vertex stage down to the finest node that has landed at its own position, and
sizes itself to that pitch — so the coarse layers that sit under a refined
region draw at the refined pitch instead of `2**(D-L)` too wide and painting
over data that is already on the GPU.

The point budget is spent in two tiers. Refinement up to `targetScreenError` is
what the caller asked for and may spend the whole `pointBudget`; refinement past
it is opportunistic, runs at most `BONUS_LEVELS` (2) octree levels deeper, and
may only spend down to `pointBudget * (1 - budgetHeadroom)` — 15% by default,
left free so the next camera move can grow into it without evicting.

`stats.limitedBy` is the field the reference viewer lacks. `"error"` means the
target was met and the bonus tier ran out of detail to add; `"headroom"` means
it was met and the bonus tier hit the withheld slice; `"budget"` and `"nodes"`
mean the target was NOT met and quality is being left on the table. Read it with
`stats.achievedScreenError`, the worst projected error still on screen.

### `@voxelkloud/view/lod`

The scheduler on its own — frustum extraction, AABB classification, the
screen-space-error metric, best-first selection — behind a subpath whose module
graph pulls in no three, no DOM and no GPU. Allocation-free in steady state.

Full documentation: [voxelkloud](../../README.md).

MIT.
