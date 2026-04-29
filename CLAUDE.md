# beam_engine

Research project — building the fastest LLM beam search engine. Two baselines
under `docs/baseline/`:

- **paged attention** (`src/beam_engine/baselines/paged.py`) — FlashInfer paged
  decode wrapper, refcounted page sharing for prefix, copy-on-write at
  divergence.
- **tree attention** (`src/beam_engine/baselines/tree.py`, arXiv:2502.00085) —
  fused-sequence layout, single FlashAttention call with a tree-shaped custom
  mask.

Both share the same Llama model code (`src/beam_engine/models/`) via the
`AttentionContext` dispatch layer.

## Environment

- `uv` for env management. Python 3.11. Dependencies pinned in
  `pyproject.toml` (torch 2.9, flashinfer-python, transformers 4.57.3).
- `uv sync` to install. `uv run python ...` to run.
- Conda env `flashtree` is the host shell environment (`test.sh` activates it
  before invoking `uv`).

## GPU usage — ALWAYS check before running

The host is a shared multi-GPU machine. Before running any command that uses
the GPU:

1. Run `nvidia-smi` and pick a **fully idle** GPU. Required: ~0% utilization
   AND only minimal memory in use (effectively no other user's process).
   "Mostly idle" or "lots of memory free but high util" is NOT acceptable —
   running there will steal cycles from another user and produce noisy
   benchmarks. If no GPU is fully idle, **do not run**; wait or ask the user.
2. Pin every CUDA-using command to that GPU with `CUDA_VISIBLE_DEVICES=<id>`.

Examples:

```bash
nvidia-smi   # check first

CUDA_VISIBLE_DEVICES=2 uv run python tests/test_baselines.py

# Quick imports / sanity checks: still pin a GPU even if you think it won't allocate.
CUDA_VISIBLE_DEVICES=2 uv run python -c "from beam_engine.baselines import paged, tree; print('ok')"
```

The `test.sh` wrapper accepts a GPU id as its second arg and forwards it via
`CUDA_VISIBLE_DEVICES`:

```bash
./test.sh tests/test_baselines.py 2
```

Never launch a GPU job without picking a free GPU first — taking a GPU another
user is on will OOM both jobs.

## Layout

```
src/beam_engine/
  models/                    # standalone Llama (no vLLM deps)
    attention.py             # AttentionContext protocol + thin Attention call site
    modeling_llama.py        # LlamaForCausalLM, weight remap from HF safetensors
    rmsnorm.py, rotary_embedding.py, configuration_llama.py
  page_table.py              # paged KV cache (FlashInfer 5D NHD layout)
  baselines/
    paged.py                 # baseline 1 — paged-attention beam search
    tree.py                  # baseline 2 — tree-attention beam search
  logger.py
tests/
  test_baselines.py          # cross-baseline equality + greedy match
docs/baseline/
  paged_attention.md
  tree_attention.md
```
