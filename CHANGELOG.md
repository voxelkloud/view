# @voxelkloud/view

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
