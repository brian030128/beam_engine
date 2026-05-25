# Paper experiments — reproduction scripts

One job per paper artifact. Each sbatch is self-contained: it loads
`cuda/12.4`, activates `.venv/`, recalibrates the `bs_kernel` cost model,
runs a bench harness in `benchmarks/bs_kernel/`, and writes to a stable
dir under `benchmarks/bs_kernel/results/paper-exp/`. See the top-level
`REPRODUCE.md` for the full artifact → command → output map.

All Llama-3 checkpoints are served **bf16 weights + bf16 KV**:
1B / 8B on 1×H100 (TP=1), 70B on 4×H100 (TP=4).

## End-to-end F1–F4 (Fig. speedup bars, Absolute-Results & Dispatcher-diagnosis tables)

| sbatch | model | cells |
|---|---|---|
| `exp_final_paper_1b.sbatch`      | Llama-3.2-1B  | F1 multi_doc_qa, F2 multi_few_shot, F3 beam_search |
| `exp_final_paper_8b.sbatch`      | Llama-3.1-8B  | F1, F2, F3 |
| `exp_final_paper_70b_bf16.sbatch`| Llama-3-70B   | F1, F2, F3 |
| `exp_final_f4_1b.sbatch`         | Llama-3.2-1B  | F4 self_consistency (long-decode, mn=12000) |
| `exp_final_f4_8b.sbatch`         | Llama-3.1-8B  | F4 |
| `exp_final_f4_70b_bf16.sbatch`   | Llama-3-70B   | F4 |

Shared drivers: `final_paper_lib.sh` (F1–F3), `final_f4_lib.sh` (F4).
`cell_complete.py` is the max-fit completeness gate (walks a B ladder
largest-first and accepts the first config where every method fits — the
accepted (K,B) is encoded in the CSV name). Datasets are built on first
use by `scripts/data/build_hotpotqa_{multi_q,prompts}.py`.

Collate into the canonical numbers file:

```bash
uv run python build_final_results.py   # -> ../../final_paper_results.md
```

Multi-draw spread (resampled GSM8K/HotpotQA text at a fixed token-length
shape): pass `SEEDS=0,1,2,...` to the sbatch; `verify_shape_parity.py`
checks that every draw conforms to the seed-0 shape.

## Ablation (Appendix: Depth ablation)

| sbatch | what |
|---|---|
| `exp2_divlevels_8b_tp2.sbatch` | forced cascade depth s₂d/s₃d/s₄d/s₅d (and 1POOL variants), Llama-3.1-8B TP=2, K=32 L_p=8192 B=8, forward measured over decode steps 1500–2000 → `exp2_divlevels_8b_tp2/merged.csv` |

The **timing-breakdown** table (8B beam-search build/decision/forward) and
the **single-step** appendix table are driven from `slurm/`
(`plan_breakdown_8b_beamsearch.sbatch`, `bench_tree_kernel.sbatch`).

## DeFT baseline

DeFT's Triton split-by-node kernel is vendored at
`src/beam_engine/baselines/_deft_kernel/` (copied from `3rdparty/DeFT`;
DeFT's package pins torch==2.5.1 / py≥3.12 so it can't co-install in
`.venv/`). The bench wrapper `baselines/deft.py:DeftBackend` builds
DeFT-format metadata from the page-radix tree and splits per-node readers
into BLOCK_M=32 chunks. It is included directly in the F1–F4 method lists.
