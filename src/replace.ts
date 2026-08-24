// REPLACE refinement: which admitted nodes must NOT be drawn.
//
// Every format this repo read until now is ADDITIVE — a node's points and its
// children's points are different points, and drawing both is the picture. 3D
// Tiles has a second mode, `refine: "REPLACE"`, where a tile's content is a
// COARSE STAND-IN for its children and drawing both draws the same ground twice.
//
// This is deliberately NOT in the scheduler, and the reason is the one fact the
// scheduler does not have: RESIDENCY. Deciding to hide a parent needs to know
// its children are already on the GPU, and dropping it before they are opens a
// hole exactly where the picture was about to improve. So the selection stays
// whatever the scheduler chose, and this runs between it and `setVisible`.
//
// PURE and allocation-free after the first call: an epoch-stamped array marks
// the selection instead of a Set, so nothing is cleared per frame.

/** The slice of a tree this needs. Declared structurally, like `LodTreeView`. */
export interface ReplaceTreeView {
  readonly nodeCount: number;
  node(index: number): { readonly children: readonly (ReplaceNode | undefined)[] } | undefined;
  /** 1 where a node's children REPLACE it. `undefined` on every additive format. */
  readonly nodeReplaces?: Uint8Array | undefined;
}

interface ReplaceNode {
  readonly index: number;
}

export interface ReplaceScratch {
  /** Epoch-stamped: `stamp[i] === epoch` means "in this frame's selection". */
  stamp: Int32Array;
  epoch: number;
}

export function createReplaceScratch(capacity = 4096): ReplaceScratch {
  return { stamp: new Int32Array(capacity), epoch: 0 };
}

/**
 * Copy `indices[0..count)` into `out`, dropping every node whose children have
 * taken over. Returns the number written.
 *
 * A node is dropped when ALL of these hold:
 *
 *  - it is marked REPLACE,
 *  - it has children,
 *  - every one of them is in this frame's selection, and
 *  - every one of them is resident.
 *
 * The last two are conservative on purpose. A child the scheduler declined —
 * because the budget ran out, or because it was culled — is a region the parent
 * is still the only cover for, so hiding the parent would leave a hole there
 * rather than a coarser picture. One frame of overdraw is the cheaper mistake,
 * and it is the one Cesium makes too.
 *
 * A tree with no `nodeReplaces` array — every octree format — is copied
 * through, and the caller can skip the copy entirely by checking for the array
 * first. This still handles it so the call site needs no branch.
 */
export function filterReplacedParents(
  tree: ReplaceTreeView,
  indices: Int32Array,
  count: number,
  isResident: (index: number) => boolean,
  scratch: ReplaceScratch,
  out: Int32Array,
): number {
  const replaces = tree.nodeReplaces;
  if (replaces === undefined) {
    if (out !== indices) out.set(indices.subarray(0, count));
    return count;
  }

  if (scratch.stamp.length < tree.nodeCount) {
    let cap = Math.max(scratch.stamp.length, 16);
    while (cap < tree.nodeCount) cap *= 2;
    scratch.stamp = new Int32Array(cap);
    scratch.epoch = 0;
  }
  const stamp = scratch.stamp;
  // Wrap safely rather than overflow into a stale match after 2^31 frames.
  if (++scratch.epoch === 0x7fffffff) {
    stamp.fill(0);
    scratch.epoch = 1;
  }
  const epoch = scratch.epoch;

  for (let k = 0; k < count; k++) {
    const i = indices[k]!;
    if (i >= 0 && i < stamp.length) stamp[i] = epoch;
  }

  let w = 0;
  for (let k = 0; k < count; k++) {
    const i = indices[k]!;
    if (replaces[i] !== 1) {
      out[w++] = i;
      continue;
    }
    const node = tree.node(i);
    const children = node?.children;
    if (children === undefined || children.length === 0) {
      out[w++] = i;
      continue;
    }
    let covered = true;
    for (let c = 0; c < children.length; c++) {
      const child = children[c];
      if (child === undefined) continue;
      const ci = child.index;
      if (stamp[ci] !== epoch || !isResident(ci)) {
        covered = false;
        break;
      }
    }
    if (!covered) out[w++] = i;
  }
  return w;
}
