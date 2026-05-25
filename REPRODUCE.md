# Reproducing the paper

This artifact reproduces every table and figure in
*"Q-Bimodal Tree Attention: Fixed-Cardinality Dispatch for Efficient
Tree-Structured LLM Decoding."*

The repository has been trimmed to **only** the code, scripts, and result
files behind the paper's reported numbers. The five compared methods are
**paged**, **fasttree** ([MLSys'25]), **deft**, **mlca** (FlashInfer
multi-level cascade), and **bs_kernel** (our method, "Ours"; its two
runtime strategies are `s_2f` = `SHARED_2L_1POOL` and `s_2d` =
`SHARED_2L_DEC_TAIL`).

## Environment

```bash
uv sync                       # Python 3.11, torch 2.9, the flashinfer fork
git submodule update --init   # 3rdparty/flashinfer, FastTree-Artifact, DeFT
```

Hardware: H100. 1B/8B run on 1×H100 (TP=1); 70B runs on 4×H100 (TP=4).
Models are the public instruction-tuned Llama-3 checkpoints
(1B, 8B, 70B), served bf16 weights + bf16 KV. Set `HF_TOKEN` /
`HF_HOME`. The full suite takes ≈2 days of H100 time.

## Map: paper artifact → command → output

| Paper artifact | Run | Output |
|---|---|---|
| **Fig. (speedup bars)** + **Table: Absolute Results** (F1–F4 × 1B/8B/70B) | `sbatch scripts/paper-exp/exp_final_paper_{1b,8b,70b_bf16}.sbatch` (F1–F3) and `sbatch scripts/paper-exp/exp_final_f4_{1b,8b,70b_bf16}.sbatch` (F4) | `benchmarks/bs_kernel/results/paper-exp/final_paper/<model>-bf16/*.csv` |
| **Table: Dispatcher diagnosis** (s₂f / s₂d / dispatcher) | *(same F1–F4 runs above — the forced `bsk_2l_1p` / `bsk_2l_dt` methods)* | same dir; columns `1p` / `dt` / `bs_kernel` |
| **Table: Timing breakdown** (8B beam-search, build / decision / forward) | `sbatch slurm/plan_breakdown_8b_beamsearch.sbatch` | `…/paper-exp/exp3_plan_breakdown/all_8b_beamsearch/plan_breakdown.csv` |
| **Appendix: Depth ablation** (s₂d / s₃d / s₄d / s₅d, 8B TP=2) | `sbatch scripts/paper-exp/exp2_divlevels_8b_tp2.sbatch` | `…/paper-exp/exp2_divlevels_8b_tp2/merged.csv` |
| **Appendix: Single-step strategy validation** (16 tree shapes) | `sbatch slurm/bench_tree_kernel.sbatch` | `…/results/bench_tree_kernel-<ts>.csv` |
| **Appendix: Padding fraction** (1−G/T) | analytical (Eq. 9); the 75% empirical value is the F1 / `s_2f` instrumentation in the same e2e runs | — |

After the e2e runs, regenerate the collated tables file:

```bash
uv run python scripts/paper-exp/build_final_results.py   # -> final_paper_results.md
```

`final_paper_results.md` is the canonical numbers file: its decode/plan
tables are the source for the paper's Absolute-Results and
Dispatcher-diagnosis tables and the speedup figure, and its
"Tree-kernel microbenchmark" section is the Single-step appendix table.

## Workload data

Built automatically by the e2e libs from **GSM8K** + **HotpotQA**
(`scripts/data/build_hotpotqa_{multi_q,prompts}.py`). The depth-ablation
and timing-breakdown jobs additionally read `data/gov_report.jsonl` and
`data/hotpotqa.jsonl`. Multi-draw (resampled text at fixed token-length
shape) is gated by `SEEDS=...`; `verify_shape_parity.py` checks shape
conformance. See the per-scenario builders and `final_paper_lib.sh`.

## Cost-model calibration

`bs_kernel`'s cost-model constants are recalibrated once per model at
engine init (the e2e libs call
`python -m beam_engine.methods.bs_kernel.calibrate --force`; ≈3 min on an
H100). See Appendix "Calibration".

## Correctness

`tests/test_bs_kernel.py` asserts `bs_kernel` matches the PagedAttention
reference token-for-token (within fp16 attention drift) across K / L_p /
max_new. Other `tests/test_*.py` cover the plan cache, the radix
fast-path, and TP.
