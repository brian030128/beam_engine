# beam_engine

Research project — building the fastest LLM beam search engine.

The repo ships **six** beam-search methods, all driven by a unified
`page_driver.beam_search` (cross-prompt batched: B prompts × K beams in
one launch):

- **paged** (`baselines/paged.py`) — FlashInfer paged decode wrapper;
  refcounted prefix sharing, copy-on-write at divergence.
- **tree** (`baselines/tree.py`, arXiv:2502.00085) — fused-sequence
  layout, single FlashAttention call with a tree-shaped mask.
- **mlca** (`baselines/mlca.py`) — FlashInfer
  `MultiLevelCascadeAttentionWrapper` (1 launch, 2 levels: shared
  prefix + per-beam tail).
- **fasttree** (`baselines/fasttree.py`, MLSys'25) — Triton
  two-stage radix-tree decode kernel; rewritten on top of
  `page_driver` with page-level radix + LCA-prefix-skip + numpy-bridged
  metadata. See `docs/baseline/fasttree_analysis.md`.
- **adaptive_pool** (`methods/adaptive_pool.py`) — adaptive 2-pool
  routing on top of FlashInfer's `FusedMultiLevelCascadeAttentionWrapper`;
  picks pool sizes per step.
- **bs_kernel** (`methods/bs_kernel/`, our proposed method) — unified
  cost-model picker over (cascade depth × pool count × tail-kernel),
  with online-calibrated coefficients and plan-state caching keyed by
  page-level LCA. See `docs/bs_kernel_design.md`. Tail-kernel choice
  includes `DEC_TAIL` (CTA_Q=1 decode kernel + online softmax merge),
  the natural CTA_Q=1 case of the two-pool concept in
  `docs/cta_tile_q_design.md`.

All methods share the same Llama model code (`models/`) via the
`AttentionContext` dispatch layer.

## Environment

- `uv` for env management. Python 3.11. Dependencies in
  `pyproject.toml` (torch 2.9, flashinfer-python at the
  `brian030128/flashinfer@beam_engine` fork in `3rdparty/flashinfer`,
  transformers 4.57.3).
- `uv sync` to install. `uv run python ...` to run.

## GPU usage

GPU access varies by host. Check `docs/h100_usage.md` for cluster
specifics.

- **nano5 (current default)** — SLURM cluster. H100s on
  `hgpn[01-06,17-21]`; the login node has no usable GPU. Submit work
  with `sbatch slurm/<job>.sbatch`. Account `MST114554`, partition
  `dev`. SLURM sets `CUDA_VISIBLE_DEVICES`. Do NOT pick a GPU manually
  via `nvidia-smi` here.
- **Shared-workstation hosts** — run `nvidia-smi` first, pick a GPU at
  ~0% util / ≤2 MiB used, pin via `CUDA_VISIBLE_DEVICES=<id>`.

## Layout

```
src/beam_engine/
  models/                       # standalone Llama (no vLLM deps)
    attention.py                # AttentionContext protocol + thin Attention call site
    modeling_llama.py           # LlamaForCausalLM, weight remap from HF safetensors
    rmsnorm.py, rotary_embedding.py, configuration_llama.py
  page_table.py                 # paged KV cache; layout
                                #   [2, max_pages, page_size, num_kv_heads, head_dim]
                                # (kv[0] = K slab, kv[1] = V slab; flat slot view
                                # = kv[i].view(max_pages*page_size, ...) is no-copy,
                                # consumed by fasttree's slot-indexed kernel.)
  page_driver.py                # unified beam_search(model, ..., backend=...).
                                # Backend protocol (init_wrappers + plan_decode_step)
                                # is what each method implements.
  decoding.py                   # standard / DBS top-K selectors (batched + per-prompt)
  baselines/
    paged.py, tree.py, mlca.py, fasttree.py
  methods/
    adaptive_pool.py            # PageDecodeBackend; 2-pool fused-cascade picker
    bs_kernel/
      cost_model.py             # Strategy enum, pick_strategy_batch, fallbacks
      driver.py                 # _BsKernelWrappers + dispatch (PER_BEAM/SHARED_*L/DEC_TAIL)
      calibrate.py              # per-tile cost probes (prefill + decode); cached
      autotune.py               # fits share_extra_us / dual_pool_extra_us
      plan_update.py            # Phase-1 GPU-resident plan-state Triton kernels
      decode_tail_context.py    # AttentionContext for SHARED_*L_DEC_TAIL
benchmarks/
  bs_kernel/                    # see benchmarks/bs_kernel/README.md
slurm/                          # *.sbatch entry points (nano5 cluster)
docs/
  bs_kernel_design.md           # design plan for bs_kernel (cost model + kernel)
  cta_tile_q_design.md          # two-pool CTA_TILE_Q justification
  baseline/
    paged_attention.md          # paged-attention baseline
    tree_attention.md           # tree-attention baseline
    fasttree_analysis.md        # FastTree weaknesses + bs_kernel novelty
tests/
  test_baselines.py             # cross-baseline equality + greedy match
```

## Cascade attention references

- https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout
- https://docs.flashinfer.ai/api/attention.html#batch-prefill-append-attention
- https://docs.flashinfer.ai/generated/flashinfer.decode.cudnn_batch_decode_with_kv_cache.html
