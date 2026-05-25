# bs_kernel benchmark suite

End-to-end and kernel-level benchmarks for the beam-search-specialized
attention dispatcher (`src/beam_engine/methods/bs_kernel/`) and its
baselines (paged, mlca, adaptive_pool, fasttree, tree).

## Quick start (nano5 / SLURM)

```bash
sbatch slurm/smoke_batched_kernels.sbatch                    # cross-method correctness
sbatch slurm/profile_fasttree_vs_bs.sbatch                   # phase-timed perf
sbatch slurm/bench_b32_k64_lp8k.sbatch                       # canonical perf cell
sbatch slurm/bench_dbs_single.sbatch                         # DBS variants
```

Logs land in `slurm/logs/`; results CSVs in `benchmarks/bs_kernel/results/`.

## Quick start (workstation)

```bash
nvidia-smi                       # pick a fully-idle GPU (per CLAUDE.md)
./run_all.sh <gpu_id>            # full grid (~10-15 min on RTX-class)
./run_all.sh <gpu_id> quick      # ~3-5 min smoke run
```

## Methods covered (7 total)

| Method            | Source                              | Role                           |
|-------------------|-------------------------------------|--------------------------------|
| paged             | `baselines/paged.py`                | baseline                       |
| tree              | `baselines/tree.py`                 | baseline (single-beam reference) |
| mlca              | `baselines/mlca.py`                 | baseline                       |
| adaptive_pool     | `methods/adaptive_pool.py`          | first baseline we beat         |
| fasttree          | `baselines/fasttree.py`             | baseline (post page_driver rewrite) |
| **bs_kernel**     | `methods/bs_kernel/`                | proposed                       |
| **bs_kernel_2L_dec / 3L_dec** | `methods/bs_kernel/` (forced strategies) | DEC_TAIL ablation forced modes |

DBS variants are the same 7 methods with `dbs_` aliases (Hamming-diversity
top-K). Toggle via `bench_batched.py --methods dbs_<name>`.

## Key sbatch entry points

| Sbatch                                       | What it measures                                                |
|----------------------------------------------|-----------------------------------------------------------------|
| `smoke_batched_kernels.sbatch`              | top-token equivalence across all 7 methods                      |
| `profile_fasttree_vs_bs.sbatch`             | phase timings (cow / plan / forward / topk / fork) per method   |
| `profile_fasttree_plan.sbatch`              | substep breakdown of fasttree plan (radix / metadata / tensors) |
| `bench_b32_k64_lp8k.sbatch`                 | canonical cell K=64/L_p=8K/B=32/max_new=256                     |
| `bench_b32_k64_lp8k_m256_distinct.sbatch`   | same cell with distinct prompts                                  |
| `bench_long_decode_k64.sbatch`              | max_new=512 sweep                                                |
| `bench_modes.sbatch` / `bench_modes_single*.sbatch` | forced-strategy ablation (per-cell)                      |
| `bench_dbs_single.sbatch`                   | DBS variants at the canonical cell                               |
| `bench_fasttree_check.sbatch`               | fasttree only, used to track integration optimization wins      |
| `bench_tree_kernel.sbatch`                  | tree-structure taxonomy: single-step kernel competition across families (paper) |
| `dump_picks_*.sbatch`                       | picker decision histograms (depth / pool / tail-kernel)         |
| `autotune_h100.sbatch`                      | re-fit cost-model coefficients on H100                          |

## Individual scripts

```bash
# physics calibration (per-GPU coefficients, cached)
uv run python -m beam_engine.methods.bs_kernel.calibrate --force

# auto-tune (fits 2 free parameters by minimizing regret)
uv run python -m beam_engine.methods.bs_kernel.autotune --save

# correctness — bs_kernel beams must match tree.py
uv run python tests/test_bs_kernel.py

# tree-structure taxonomy — single-step kernel competition (paper figure).
# Families span prefix-len / K / B / tail-len / depth / cross-prompt sys;
# competes paged, mlca, fasttree, picker-dispatched bs_kernel, and the
# forced 2l1p / 2dt / 3dt variants. Reports per-shape oracle + picker regret.
uv run python benchmarks/bs_kernel/bench_tree_kernel.py            # all families
uv run python benchmarks/bs_kernel/bench_tree_kernel.py --family cross_prompt_sys wide_fanout
uv run python benchmarks/bs_kernel/bench_tree_kernel.py --kernels paged bs_kernel bs_2dt bs_3dt

# cross-method bench (foreground)
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 64 --L_p 8192 --B 32 --max_new 256 \
    --methods paged mlca adaptive_pool fasttree bs_kernel

# phase profile (which stage of each method's step is slow)
uv run python benchmarks/bs_kernel/profile_fasttree_vs_bs.py

# forced-mode ablation
uv run python benchmarks/bs_kernel/bench_modes.py
```

## Workload grids

- **Canonical perf cell:** K=64, L_p=8192, B=32, max_new=256.
- **Sweep:** K ∈ {16, 32, 64} × L_p ∈ {2048, 8192, 32768, 65536}.
- **Long decode:** max_new ∈ {128, 256, 512}.

## Calibration cache

Per-GPU coefficients live at `~/.cache/beam_engine/coeffs-<gpu_name>.json`.
Auto-tune overwrites this file with fitted overhead values. The driver
auto-loads from cache when `coefficients=None` is passed to
`bs_kernel.beam_search` (the default).

To reset:

```bash
rm ~/.cache/beam_engine/coeffs-<gpu_name>.json
sbatch slurm/autotune_h100.sbatch
```

## GPU policy

See project root `CLAUDE.md`. On nano5 the SLURM scheduler pins
`CUDA_VISIBLE_DEVICES`; do not pick GPUs manually. On shared-workstation
hosts: `nvidia-smi` first, pin via `CUDA_VISIBLE_DEVICES=<id>`.
