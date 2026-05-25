# bs_kernel benchmark suite

The benchmark harnesses behind the paper's results. They drive the
beam-search attention dispatcher (`src/beam_engine/methods/bs_kernel/`)
and its baselines (`paged`, `fasttree`, `deft`, `mlca`) through the
unified `page_driver.beam_search`. See the top-level `REPRODUCE.md` for
the artifact → command map.

## Harnesses

| File | Used for |
|---|---|
| `bench_sglang_e2e.py` | end-to-end F1 (multi-doc-qa), F2 (multi-few-shot), F4 (self-consistency); builds GSM8K/HotpotQA workloads via `sglang_workloads.py` |
| `bench_batched.py` | end-to-end F3 (beam-search, dynamic tree); also the forced-depth ablation and the plan-time breakdown |
| `bench_tree_kernel.py` | single-step kernel competition over the tree-shape taxonomy (Appendix table) |
| `sglang_workloads.py` | workload builder (shared prefixes + branch tails from real GSM8K/HotpotQA text) |

## Methods

| Method | Source | Role |
|---|---|---|
| paged | `baselines/paged.py` | PagedAttention baseline |
| fasttree | `baselines/fasttree.py` | FastTree baseline (MLSys'25) |
| deft | `baselines/deft.py` | DeFT baseline (vendored kernel) |
| mlca | `baselines/mlca.py` | FlashInfer multi-level cascade (MLCA) |
| **bs_kernel** | `methods/bs_kernel/` | **Ours** — cost-model dispatcher |

`bs_kernel`'s two runtime strategies can be forced for the dispatcher
diagnosis / depth ablation via the `bs_kernel_2l1p` (s₂f) and
`bs_kernel_2ldt` (s₂d) method aliases (`bench_batched.py`), or
`bsk_2l_1p` / `bsk_2l_dt` (`bench_sglang_e2e.py`); deeper forced variants
`bs_kernel_{3,4,5}l{1p,dt}` exist for the ablation only.

## Entry points (SLURM)

```bash
# end-to-end (per model): see scripts/paper-exp/exp_final_*.sbatch
sbatch slurm/bench_tree_kernel.sbatch            # single-step taxonomy (Appendix)
sbatch slurm/plan_breakdown_8b_beamsearch.sbatch # timing breakdown (8B beam-search)
```

## Cost-model calibration

```bash
# per-GPU cost-model coefficients (cached; ~3 min on an H100)
uv run python -m beam_engine.methods.bs_kernel.calibrate --force
```

Results land in `benchmarks/bs_kernel/results/paper-exp/`; the paper-data
subdirs (`final_paper/`, `exp2_divlevels_8b_tp2/`,
`exp3_plan_breakdown/`) are tracked, other run outputs are gitignored.
