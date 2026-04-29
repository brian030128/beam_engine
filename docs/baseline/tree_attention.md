# Baseline: Tree Attention (arXiv:2502.00085)

## Overview

Tree attention is a beam-search KV layout that fuses all beams into a **single
logical sequence** and uses a custom attention mask to keep each beam's
computation isolated. It builds on FlashAttention: a standard FlashAttention
kernel processes the fused sequence in a single launch, with the mask
implementing the tree topology.

The motivation is the same problem paged attention struggles with — the shared
prefix KV is read once per beam in paged attention. Tree attention reads it
**once total** by stuffing every beam's queries into one attention call against
one packed KV tensor.

Reference: https://arxiv.org/pdf/2502.00085

## How It Is Implemented for Beam Search

### 1. Fused sequence layout

The beam tree is linearized into a single token sequence stored contiguously in
the KV cache. Layout:

```
[ shared prefix tokens | beam 1 suffix | beam 2 suffix | ... | beam B suffix ]
```

There is exactly **one copy** of every KV vector — the prefix is stored once,
and each beam's diverging suffix is appended after it. No block tables, no
copy-on-write: everything lives in one packed KV tensor per layer.

A small auxiliary structure records, for each token position, which beam it
belongs to and what its parent in the tree is. This is what feeds the mask.

### 2. Tree-shaped attention mask

The attention mask is built so that for any query at position `q` belonging to
beam `b`, the allowed keys are exactly the tokens on the path from the root to
`q` in the tree:

- Tokens in the shared prefix are visible to all beams.
- Tokens in beam `b`'s own suffix are visible only to queries in beam `b`.
- Tokens in any sibling beam's suffix are masked out.

This is a generalization of the causal mask: it is causal *along each
root-to-leaf path*, but blocks cross-beam visibility.

### 3. Single FlashAttention launch

At each decode step, the kernel is launched once over the fused sequence. All
`B` beams' queries attend to the packed KV in one pass. Because FlashAttention
streams KV tiles through SRAM, the **shared prefix KV is loaded from HBM
exactly once** and reused across every beam's queries while it is resident.
This is the central win over paged attention's `O(B * L_p)` prefix traffic.

### 4. Garbage collection on pruning

When a beam is eliminated, its suffix tokens are now dead weight inside the
fused KV tensor. Because the layout is contiguous, freeing those slots requires
**reordering the KV cache**: surviving beams' suffixes must be compacted so the
fused sequence stays packed (or, equivalently, the position metadata and mask
must be rewritten to skip the holes, which still costs a memory rewrite at
some point to keep the sequence length from growing unbounded).

This GC pass runs every step (or every few steps) and incurs:
- A read+write of the affected KV regions.
- An update of the per-token beam/parent metadata.
- Cache invalidation of the prior layout.

### 5. Decode step

Per step:
1. Compute Q for each live beam.
2. Pack Qs into the fused query tensor.
3. Launch FlashAttention with the tree mask over the fused KV.
4. Sample / score top-k continuations, choose surviving beams.
5. Append new KV entries; run GC if pruning occurred.

## The Flaws

### Flaw 1: GC requires reordering the KV cache

Because tree attention insists on a contiguous fused layout, every prune step
forces a compaction of the KV cache. This is pure overhead — a memory shuffle
that produces no useful computation. The cost is proportional to the live
suffix length per layer, and it must be paid on the critical path of each
generation step that prunes a beam.

Paged attention sidesteps this entirely: pruning is a refcount decrement.

### Flaw 2: Quadratic blow-up for long generation

Attention is `O(L^2)` in the fused sequence length. With `B` beams generating
for `L_g` tokens each, the fused suffix region grows to `B * L_g`. Every beam's
query attends across **all** suffix tokens (modulo the mask, which doesn't
reduce FLOPs in a standard FlashAttention kernel — masked positions are still
loaded and multiplied, just zeroed out).

So as generation length grows:
- FLOPs per step scale as `O((L_p + B * L_g) * B)` — the suffix term grows
  quadratically in the joint dimension.
- Memory traffic per step grows similarly because all beams' suffix KVs must
  be streamed even though each query only needs its own beam's suffix.

For long-output workloads this can erase the prefix-read savings and make tree
attention slower than paged attention. The break-even point depends on `L_p`,
`L_g`, and `B`: tree attention dominates when the prefix is long and the
generation is short; paged attention dominates the opposite regime.

## Summary

| Aspect | Tree Attention |
| --- | --- |
| KV layout | Single fused sequence, packed contiguously |
| Prefix KV memory | Stored once |
| Prefix HBM reads per step | `O(L_p)` total (shared across all beams in one kernel) |
| Suffix HBM reads per step | `O(B * L_g)` per beam → `O(B^2 * L_g)` aggregate |
| Pruning cost | KV compaction / reorder on the critical path |
| Attention kernel | Single FlashAttention launch with tree mask |
| Best regime | Long shared prefix, short generation, wide beam |
| Worst regime | Long generation (suffix-dominated, quadratic blow-up) |
