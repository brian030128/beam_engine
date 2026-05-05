# bs_kernel benchmark suite

End-to-end and kernel-level benchmarks for the beam-search-specialized
attention dispatcher (`src/beam_engine/methods/bs_kernel/`).

## Quick start

```bash
nvidia-smi                       # pick a fully-idle GPU (per CLAUDE.md)
./run_all.sh <gpu_id>            # full grid (~10-15 min on RTX-class)
./run_all.sh <gpu_id> quick      # ~3-5 min smoke run
```

Results are written to `results/run-<gpu>-<timestamp>/`. Each step also
emits a `.log` next to its CSV.

## What `run_all.sh` runs

| Step | Script | What it measures |
|------|--------|------------------|
| 1 | `calibrate.py` | Physics coefficients (B_hbm, launch_us, sync_us, merge_us, per_tile_us, num_sms) |
| 2 | `autotune.py` | Fits `share_extra_us` / `dual_pool_extra_us` to minimize oracle regret |
| 3 | `tests/test_bs_kernel.py` | Correctness vs tree.py reference |
| 4 | `sweep.py` | End-to-end Llama-3.1-8B beam search on 5 methods × workload grid |
| 5 | `oracle_vs_model.py` | Cost-model regret distribution per grid cell |
| 6 | `tree_shapes.py` | Kernel-level comparison across 100+ tree shapes (vs paged / tree-attention / cascade) |
| 7 | `demo_e2e.py` | Real text generation with timing per method (showcase) |

## Individual scripts

```bash
# physics calibration (per-GPU coefficients, cached)
CUDA_VISIBLE_DEVICES=<id> uv run python -m beam_engine.methods.bs_kernel.calibrate --force

# auto-tune (fits 2 free parameters by minimizing regret)
CUDA_VISIBLE_DEVICES=<id> uv run python -m beam_engine.methods.bs_kernel.autotune --save

# correctness — bs_kernel beams must match tree.py
CUDA_VISIBLE_DEVICES=<id> uv run python tests/test_bs_kernel.py

# end-to-end sweep (5 methods, full grid by default)
CUDA_VISIBLE_DEVICES=<id> uv run python benchmarks/bs_kernel/sweep.py --full

# oracle regret on the full grid
CUDA_VISIBLE_DEVICES=<id> uv run python benchmarks/bs_kernel/oracle_vs_model.py --full

# kernel-level tree-shape benchmark
CUDA_VISIBLE_DEVICES=<id> uv run python benchmarks/bs_kernel/tree_shapes.py

# end-to-end demo (real beam search, prints generated text)
CUDA_VISIBLE_DEVICES=<id> uv run python benchmarks/bs_kernel/demo_e2e.py \
    --K 64 --L_p 32768 --max_new 32
```

## GPU policy

The host is shared. Per `CLAUDE.md`:

* Run `nvidia-smi` first.
* Pick a GPU with `~0% utilization` AND `≤2 MiB` memory used.
* Pin every CUDA-using command via `CUDA_VISIBLE_DEVICES=<id>`.
* `run_all.sh` takes the GPU id as its first argument and pins it.

## Workload grids

* **`sweep.py` / `oracle_vs_model.py`** — `K ∈ {16, 32, 64}` × `L_p ∈ {2048, 8192, 32768, 65536}`, max_new=16.
* **`tree_shapes.py`** — 8 shape families × 100+ shapes (long-prefix flat, uniform fork G=2/4/8, imbalanced 1+singletons, all-unique G=K, high-G small-intermediate, boundary-K, short-prefix, tiny-L_p, k=1 greedy).
* **`autotune.py`** — same K/L_p as sweep, used only at calibration time.

## What the result files mean

* `01-calibrate.log` — measured device coefficients.
* `02-autotune.log` — fitted overhead values (`share_extra_us`, `dual_pool_extra_us`) and per-cell regret.
* `03-correctness.log` — pass/fail per (K, L_p, max_new) cell.
* `04-sweep-<method>.csv` — `(method, K, L_p, max_new, prefill_ms, p50, p90, p99, mean, count)` per cell, one method per file.
* `05-oracle-vs-model.csv` — `(K, L_p, model_pick, oracle_pick, regret_pct, t_<strategy>_ms ...)`.
* `06-tree-shapes.csv` — `(category, K, L_p, G, ..., paged_us, tree_us, two_level_us, three_level_us, best_kernel, pick_kernel, regret_pct)`.
* `07-demo-*.log` — text output + headline timing for the demo.

## Calibration cache

Per-GPU coefficients live at `~/.cache/beam_engine/coeffs-<gpu_name>.json`.
Auto-tune overwrites this file with the fitted overhead values. The driver
auto-loads from cache when `coefficients=None` is passed to
`bs_kernel.beam_search` (the default).

To reset:

```bash
rm ~/.cache/beam_engine/coeffs-<gpu_name>.json
```
