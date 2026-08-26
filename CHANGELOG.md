# @voxelkloud/view

## 0.6.0

### WebGL 2 stops being unusable

Without WebGPU, 0.5.2 rendered at **7.5 fps** with 656 ms of INP at a 3M budget.
It also rendered nothing at all before 0.5.2, because its vertex shader failed to
compile — nobody noticed, since WebGPU is available on every machine we develop
on. A new `points` rasteriser replaces it:

| | backend | INP | idle fps |
| --- | --- | --- | --- |
| compute | WebGPU | 56 ms | 60.2 |
| **points** | WebGL 2 | **56 ms** | 60.2 |
| Potree 1.8, for scale | WebGL | 104 ms | 59.5 |
| instanced | WebGL 2 | 656 ms | 7.5 |

`sinkMode: "auto"` now resolves compute on WebGPU, points on WebGL 2, instanced
only if neither is reachable. `view.rasterizer` says which one ran.

It works by drawing `gl.POINTS` with a real `gl_PointSize` — what Potree does,
and what WGSL has no way to express. The cost the instanced path could not shed
was ~40 ns per point in three's WebGL-compat instanced draw: pinning the splat to
1 px changed nothing, and disabling the octree cut walk entirely changed nothing.
What makes it faster than Potree is draw calls, not per-point cost — Potree
creates a `THREE.Points` per octree NODE, about two thousand for a 3M frame,
where this keeps every point in one buffer and issues one.

### three's WebGPU build stopped being everyone's problem

`three/webgpu` is three's WebGPU bundle: **357 kB gzipped against 115 kB** for
the core. Every value imported from it pulls the whole thing, and this package
imported from it in fifteen files — so a page rendering through raw WebGL 2
downloaded the entire WebGPU node system in order not to use it. `sideEffects:
false` did not save anyone, because three makes no such promise and the bundler
kept the graph.

Measured on the benchmark page: the main chunk went from **279.7 kB gzipped to
118.7 kB**, with `three/webgpu` split into a chunk only the instanced path ever
requests. First ink fell from 743 ms to 522, first contentful paint from 554 to
324, and decoded JavaScript from 0.91 MB to 0.39.

Two changes made it possible. `createPointCloudView` no longer builds a
`WebGPURenderer` at all on the compute and points paths — it opens the device
or the GL 2 context itself, which is also **27 ms of `renderer.init()` replaced
by 3 ms** of `requestAdapter` + `requestDevice` + `configure`. Neither path ever
rendered through three (both draw straight to the swapchain), so the renderer
was only ever holding the canvas.

**Breaking, and the reason this is a minor rather than a patch:** two value
exports move off the package root, because building them needs `three/webgpu`
and a root export made that edge static for everyone.

```diff
- import { createPointMaterial } from '@voxelkloud/view'
+ import { createPointMaterial } from '@voxelkloud/view/material'

- import { createEdlPipeline, resolveEdlOptions } from '@voxelkloud/view'
+ import { createEdlPipeline, resolveEdlOptions } from '@voxelkloud/view/edl'
```

Every type stays on the root, along with `resolvePointMaterialOptions` and
`scalarAttributeFor`, which are arithmetic and always were. `view.renderer` is
now `WebGPURenderer | undefined` — it exists only on the instanced path.

### The WebGL 2 path stopped waiting for its own shader

`PointsSink` linked its program in its constructor, which runs inside
`addCloud`, which runs only after both the manifest and the hierarchy have
landed. The shader source is constant and never depended on the cloud, so it
moves to `PointsRasterizer`, built during `init` — the compile now overlaps the
network instead of following it, and every cloud shares one program.

It is also warmed there with a one-vertex draw behind a colour mask, because
`linkProgram` returning success does not mean the driver has generated machine
code: most defer that to the first draw, and the first draw is the frame the
user is waiting for. `addCloud` on this path went from **17–19 ms to 6.8**.

### A lost GPU says so

Nothing here listened for `device.lost`, and a lost device throws nothing: the
library keeps submitting, every submit is dropped in silence, the canvas stops
updating, and the console stays empty. "It went black and nothing appeared" is
what that looks like from outside, and it was indistinguishable from a camera
pointed at nothing.

`onDeviceLost` reports the browser's own reason and how long the view had been
alive. `view.deviceLost` holds the same, `view.gpuErrors` collects the first
uncaptured validation errors, and `renderFrame()` returns `false` afterwards
rather than reporting sixty successful frames a second onto a dead canvas.

### The instanced path changed roles

It is now the COMPOSITION path, not the performance one. Compute and points both
own their draw, which is what makes them fast and also what stops them composing
with other three content — a gizmo or a mesh that must occlude and be occluded
by the cloud needs the scene graph. Pick it with `sinkMode: "arena"` when a scene
needs that.

### `forceWebGL`

New option, and it exists because a fallback nobody can reach on purpose is a
fallback nobody finds out is broken — which is exactly what happened to this one.

## 0.5.2

### The compute rasteriser became the default in 0.5.1, and this says so

0.5.1 shipped a different renderer by default and no note explained it. If it
regresses on your hardware, `sinkMode: "arena"` is the way back to exactly what
0.5.0 did:

```ts
createPointCloudView({ canvas, sinkMode: "arena" });
```

`view.rasterizer` reports which one actually ran — `"compute"` or `"instanced"`
— so a fallback is visible rather than inferred.

Why it changed: at a 3M budget, on the same camera and the same selected points,
INP is 272–320 ms through the instanced path and 56–72 ms through compute, with
the worst CPU frame falling from 70.7 ms to 9.5 ms. Potree 1.8 measures 136 ms
on the same page. The cost the instanced path cannot shed is per instance, and
that was established by elimination — see the README.

Both paths share the LOD scheduler, the octree cut, the six colour modes, EDL
and `pickPoint`, and size a splat with the same numbers.

### Fixed: coverage fell as more data arrived

The octree cut shrinks a coarse point once finer data is resident under it, and
it was shrinking by the full level difference — which assumes the levels in
between deliver enough points to refill the area given up. They do not always.
Measured on autzen, 5.4% of the frame went from painted at 13 s to background at
25 s: the picture got HOLES as it loaded. Potree loses 1.2% over the same window
because it never shrinks.

Capping the shrink at one level takes that to 0.7%, and took the Speed Index
from 5083 to 1918 with visual completeness arriving at 21 s instead of 25 s.
Applied to both rasterisers, which share the cut.

### Sharper without opening holes

The colour pass now weights each point by a reconstruction filter with a fixed
0.7 px width, rather than averaging everything inside the splat radius equally.
A point centred on a pixel outweighs one 2 px away by ~50x, so overlapping
coarse splats stop blending into a wash; a floor keeps an isolated splat
painting its whole disc, which is what shrinking the splat could not do.

The depth tolerance moved from a fixed 1% to the pitch the cut resolved, in
world units. One percent at a kilometre is ten metres, which is how a tree came
to average with the ground behind it.

## 0.5.1

First release built and published from CI with provenance.
