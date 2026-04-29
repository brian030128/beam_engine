# Baseline: Paged Attention for Beam Search

## Overview

Paged attention is a KV cache management technique (originally introduced in vLLM)
that decouples the logical sequence layout from the physical memory layout. The
KV cache is broken into fixed-size **pages** (blocks of contiguous tokens, e.g.
16 or 32 tokens per page), and a per-sequence **block table** maps logical
positions to physical pages. Because pages do not need to be contiguous in
memory, multiple sequences can reference the same physical page, enabling KV
sharing without copying.

For beam search, paged attention is used as the primary mechanism for sharing
the KV cache of a common prefix across multiple beams.

## How It Is Implemented for Beam Search

### 1. Prefix sharing via shared pages

When a beam expands into multiple candidate continuations, all child beams
**share the physical pages** that hold the prefix KV cache. Each child beam has
its own block table, but the entries that point to prefix pages are identical
across siblings. A reference count is kept per page so that pages are only freed
when no beam references them.

Concretely:
- During prompt prefill, the prompt KV is written into a chain of pages owned
  by the root beam.
- When the root expands to `B` beams, each of the `B` beams gets a block table
  that initially aliases the prefix pages of the root.
- No KV data is duplicated at this point — only the block-table metadata is
  copied.

### 2. Copy-on-write at divergence

A divergence occurs when a beam is about to write a token into a page that is
shared (refcount > 1). In that case the runtime performs a **copy-on-write**:

1. Allocate a new page.
2. Copy the populated portion of the shared page into the new page.
3. Update only the diverging beam's block table to point to the new page.
4. Decrement the refcount of the original shared page.

In practice, the copy only happens at most once per beam per page boundary —
once a beam owns its tail page, subsequent writes go directly into it. So the
copy cost is bounded by `num_beams * page_size` per layer at the divergence
point, not by the prefix length.

### 3. Eviction when a beam is pruned

Beam search prunes branches that fall out of the top-k. When a beam is
eliminated:

1. Walk its block table.
2. Decrement the refcount of every referenced page.
3. Free any page whose refcount drops to zero.

Pages still referenced by surviving beams remain live. This is what gives
paged attention its memory advantage over a dense layout: pruned suffix pages
are reclaimed immediately while the shared prefix stays resident exactly once.

### 4. Attention computation

At each decoding step, the attention kernel is invoked once per beam. The
kernel takes the beam's query, its block table, and the global KV pool, and
gathers the KV pages indicated by the block table. From the kernel's
perspective, beams are independent sequences — there is no cross-beam fusion.

## The Flaw: Repeated Prefix Reads

The memory benefit of paged attention does not translate into an I/O benefit on
the attention compute. Each beam launches its own attention call, and that
call **reads the full prefix KV from HBM into SRAM independently**, even though
the same physical pages back every beam.

For `B` beams and a prefix of length `L_p`:
- Memory footprint: `O(L_p)` (shared once).
- HBM → SRAM traffic per decode step: `O(B * L_p)` for the prefix portion alone.

When `L_p` is long (long context, long-form generation, RAG-style prompts),
attention becomes **memory-bandwidth bound** and the prefix dominates the read
volume. Decode latency scales linearly with `B`, even though the underlying KV
data is identical across beams.

This is the specific weakness that tree-attention-style approaches (the second
baseline) target: read the shared prefix KV from HBM **once** and reuse it for
all beams' queries within a single fused kernel.

## Summary

| Aspect | Paged Attention for Beam Search |
| --- | --- |
| Prefix KV memory | Stored once, shared via block tables |
| Divergence cost | Copy-on-write per page boundary |
| Pruning cost | Refcount decrement + page free |
| Attention kernel | One launch per beam, beams independent |
| Prefix HBM reads per step | `O(B * L_p)` — repeated reads dominate when `L_p` is large |
| Best regime | Short prefixes, small beam width |
| Worst regime | Long shared prefix, wide beam (memory-bandwidth bound) |
