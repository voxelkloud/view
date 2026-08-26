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

### A cloud you can point it at right now

[**Open it running →**](https://voxelkloud.github.io/example/)


```ts
const url = "https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz";
```

One COPC file on a bucket that is not ours, read by HTTP Range with no
conversion step and nothing downloaded whole. Verified from a browser rather
than assumed: it answers the CORS preflight for a ranged `GET` with
`Access-Control-Allow-Headers: range`, which is the part most public buckets
get wrong.

USGS 3DEP is the obvious bigger example — the whole United States, as EPT, and
genuinely public. It does not work from a page: plain `GET`s succeed, and the
preflight for a ranged one returns **403**. `curl` will tell you the data is
fine and the browser will still refuse it, so check the preflight before
promising anyone a dataset.

## Three rasterisers

`sinkMode` picks how points reach the screen. The default is `"auto"`, which
resolves in order: **compute** on WebGPU, **points** on WebGL 2, and the
instanced arena only if neither is reachable. `view.rasterizer` reports which
won, so a caller can see a fallback instead of inferring it from a frame time.

```ts
createPointCloudView({ canvas, sinkMode: "arena" }); // pin the instanced path
```

**Compute** software-rasterises in three passes — `atomicMin` the depth,
`atomicAdd` the colour and a weight, then a fullscreen resolve that averages.
One invocation per point: no instancing, no quad envelope, no per-instance
attribute step.

**Points** draws `gl.POINTS` with a real `gl_PointSize` on WebGL 2 — what Potree
does, and what WGSL cannot express. It is unreachable through three's node
system, whose GLSL builder writes `gl_PointSize = 1.0` after our code, so this
path owns its draw too.

**Instanced** draws a view-aligned quad per point through three. NOT `Points`:
three's WebGPU backend maps `object.isPoints` to `point-list` topology and WGSL
has no point-size builtin, so `sizeNode` is silently ignored there and every
point rasterises as one pixel with no attenuation.

### The instanced path is for COMPOSITION now, not performance

It is the slowest of the three and it is not going away, because it is the only
one that draws through three's scene graph. Compute and points both own their
draw, which is what makes them fast and also what stops them composing: a gizmo,
a mesh, an overlay that has to occlude and be occluded by the cloud only works
on the instanced path. Pick it deliberately with `sinkMode: "arena"` when a
scene needs that; the other two are for when the cloud IS the scene.

### Why compute is the default

Measured on autzen at a 3M budget, same camera, same selected points, runs with
a contaminated main thread discarded:

| | backend | INP | idle fps |
| --- | --- | --- | --- |
| **compute** | WebGPU | **72 ms** | 59.9 |
| **points** | WebGL 2 | **72 ms** | 59.9 |
| Potree 1.8, for scale | WebGL | 88 ms | 59.9 |
| instanced | WebGPU | 272–320 ms | 59.9 |
| instanced | WebGL 2 | 656 ms | 7.5 |

The cost the instanced path cannot shed is per INSTANCE — the attribute step for
`pointOffset`, `color` and `scalarValue`, once per splat — and that was found by
elimination rather than guessed: pinning the splat to 1 px left INP unchanged
(so not fill rate), a 3-vertex envelope left it unchanged (so not per-vertex),
and a sweep showed INP linear in instance count.

### Watch them load, frame by frame

[**Convergence race →**](https://voxelkloud.github.io/#measurements)

[![Four rasterisers at 1.9 s after load](docs/bench.png)](https://voxelkloud.github.io/#measurements)

One slider, four renderers, the same millisecond. Drag it and every panel jumps
together, so what you compare is the picture each one had at that moment rather
than four frames chosen independently.

Captured at 480x320 over a throttled 2.5 MB/s link with the HTTP cache off, all
four given a byte-identical camera — eye and direction agree to the last decimal
place, which is checked before a run rather than assumed:

| | first ink | half way | settled | fps |
| --- | --- | --- | --- | --- |
| **compute** (WebGPU) | **436 ms** | 881 ms | 10.4 s | 59.9 |
| potree-core | 479 ms | 920 ms | 13.4 s | 41.9 |
| **points** (WebGL 2) | 586 ms | **843 ms** | **9.9 s** | 59.9 |
| Potree 1.8 | 1286 ms | 1655 ms | 13.7 s | 59.9 |

Half way — when a renderer has closed half the distance to its own final frame,
which is roughly when the scene stops looking wrong — separates them more than
settling does, and settling separates them least of all: that column is a
sanity check, not a podium.

Distance is measured against each arm's OWN final frame, never against another's.
Comparing one renderer's pixels to another's would measure visual character
rather than convergence — compute is smooth by construction and points is
grainy, and neither is lateness.

The page carries its own caveats, including why the Potree arms report no
resident point count and why a frame rate that lands on exactly 30.0 or 50.0 is
the machine rather than the renderer.

Both paths share the LOD scheduler, the octree cut, the six colour modes, EDL
and `pickPoint`, and both size a splat with the same numbers. What differs is
what happens after the point is chosen.

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

### When it goes wrong, it does not throw

The failures that matter here are silent. A GPU that goes away, a shader that
fails validation, a pipeline the driver rejects — none of them raise an
exception. The device keeps accepting calls, every submit is dropped, the canvas
stops changing, and the console stays empty. From outside that is
indistinguishable from a camera pointed at nothing, which is what makes "it went
black" such a hard report to act on.

```ts
const view = createPointCloudView({
  canvas,
  onDeviceLost: (info) => {
    // `reason` is the browser's own, verbatim. There is nothing to recover —
    // every buffer went with the device — but there is everything to say.
    console.error(`GPU lost after ${info.afterSeconds}s: ${info.reason}`);
  },
});
await view.init();

view.adapterInfo;   // { vendor: "amd", architecture: "rdna-1", … } or undefined
view.gpuErrors;     // uncaptured validation errors, in order
view.gpuWarnings;   // shader messages that were NOT errors
view.deviceLost;    // set once, never cleared
```

`adapterInfo` is what turns a report into something actionable: the same code is
fine on one vendor's driver and blank on another, and without it every report
reads identically.

`gpuWarnings` exists because a module that compiles with warnings still runs.
`createShaderModule` never rejects and `getCompilationInfo` is async, so a
module that warned looks exactly like one that compiled clean — right up until a
driver treats the warning as fatal. On the WebGL 2 path the same array collects
`getShaderInfoLog` from programs that linked *successfully*, which is the only
warning channel that API has.

`renderFrame()` returns `false` once the device is gone, rather than reporting
sixty successful frames a second onto a dead canvas.

### Subpath exports

Two values live off the package root, because building them needs
`three/webgpu` — three's WebGPU bundle, 357 kB gzipped against 115 kB for the
core — and a root export made that edge static for every consumer, including
the ones that render through raw WebGL 2 and never touch it:

```ts
import { createPointMaterial } from "@voxelkloud/view/material";
import { createEdlPipeline, resolveEdlOptions } from "@voxelkloud/view/edl";
```

Every type stays on the root, and so do `resolvePointMaterialOptions` and
`scalarAttributeFor`, which are arithmetic and always were.

### What a dataset costs, and why there is a budget at all

An octree is self-similar by construction: one box becomes eight, each with half
the edge, and every LOD quantity is the root's divided by `2 ** level` —
`boundingRadiusAt(L) === boundingRadiusAt(0) / 2 ** L`, and the same for point
spacing. That is why the scheduler can work from the level alone on every octree
format, and why `nodeGeometricError` exists for the formats where it cannot.

What is NOT self-similar is which of those boxes have anything in them, and that
is the number that decides what a cloud costs you. Counting occupied nodes per
level IS box counting, so the growth ratio between levels is a
Minkowski–Bouligand dimension of whatever was scanned. Measured by walking
`hierarchy.bin` over every Potree v2 set in `demo/data`, each walk summing to the
declared point count:

| dataset | points | occupied nodes per level | D |
| --- | --- | --- | --- |
| autzen | 10.6M | 4.00 3.00 4.00 4.00 4.22 4.03 | **1.95** |
| large-20m | 20.0M | 4.00 5.50 3.55 5.12 3.84 3.26 | **2.05** |
| large-50m | 50.7M | 4.00 6.75 4.00 3.76 4.03 3.98 | **1.97** |
| large-100m | 100.5M | 4.00 4.00 3.62 2.97 4.40 4.19 3.80 | **1.84** |

Every one of them is a surface. `D = 2` means a level down quadruples the
occupied nodes; a filled volume would be `D = 3` and 8x, and vegetation sits
between the two because foliage really does carry detail at every scale. Each
tree's last level is excluded from `D` — it is the converter's truncation tail,
not a scaling regime. autzen's is 40 nodes against level 6's 3269.

Two consequences a caller feels:

**A level down costs ~4x the points and buys 2x the sharpness.** Spacing halves,
occupancy quadruples. Quadratic cost, linear gain — which is why
`targetScreenError` is denominated in device pixels rather than in levels, and
why `pointBudget` is a ceiling rather than a suggestion.

**Three quarters of the tree is its deepest full level.** That is what a 4x ratio
means, and autzen measures it: 3269 nodes of 4377. Almost every refinement
decision the scheduler makes is happening on the frontier, which is also why
`budgetHeadroom` withholds its slice there rather than anywhere else.

The stress sets (`rotterdam`, `dublin`) exist because they are NOT this — 240 m
of verticality across the Wilhelminapier towers gives an octree that is genuinely
3D rather than a draped sheet. They are COPC, so they are not in the table above,
and `D > 2` for them is an expectation, not a measurement.

#### The depth cap is float32, not taste

`MAX_CUT_DEPTH` is 20. The vertex-stage walk halves a cloud-local box `depth`
times and compares a point offset against its centre, so the comparison stops
meaning anything once a cell is the size of a float32 ULP. At autzen's 4655 m
root extent the ULP near the far corner is 4.9e-4 m and a level-20 cell is
4.4e-3 m — 9x of margin. A level-23 cell is 5.5e-4 m, and there is none left.

No stock converter comes close. autzen reaches level 7 and the 100M set reaches
9, so the cap sits about a thousandfold in linear extent away from binding. It
binds for a cloud whose root extent is very large against its finest spacing — a
continental tile at millimetre pitch — and it fails as splat sizing going wrong
in the deepest nodes, not as an error.

### `@voxelkloud/view/lod`

The scheduler on its own — frustum extraction, AABB classification, the
screen-space-error metric, best-first selection — behind a subpath whose module
graph pulls in no three, no DOM and no GPU. Allocation-free in steady state.

Full documentation: [voxelkloud](../../README.md).

MIT.
