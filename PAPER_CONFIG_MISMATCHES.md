# Paper-text vs. experiment-data config mismatches

This report lists every place in `paper.tex` where a **stated workload
configuration disagrees with the experiment that actually produced the
reported numbers**. In every case the *numbers* in the paper's tables and
figure are correct (they come straight from `final_paper_results.md` /
`benchmarks/bs_kernel/results/paper-exp/final_paper/`); only the *prose
description of the setting* is wrong. The "Correct value" column is the
setting that matches the data and therefore the published numbers.

I did **not** edit `paper.tex`. Fix these by hand.

---

## 1. F3 beam-search — prefix length (40K → 16K)

- **Location:** `paper.tex` line 316 (§Scenarios, `\textbf{beam-search}`).
- **Paper says:** *"We use a **40K-token prefix**, a moderate batch size of
  $B=16$, $K=64$ branches, and an output length of 256 tokens."*
- **Correct value:** **L_p = 16K (16000 tokens)**.
- **Evidence:** every F3 result file is
  `F3_beam_search_K64_B*_Lp16000_mn256.csv`; `final_paper_lib.sh` runs
  `ladder_batched F3 16000 256 ...` and its own comment reads
  *"16K prefix (down from 40K)"*. The beam_search numbers in Table
  "Absolute Results", the speedup figure, and the "Dispatcher diagnosis"
  table are all the 16K numbers (e.g. 1B paged 28447, ours 9016).

## 2. F3 beam-search — batch size (B=16 is 1B-only)

- **Location:** `paper.tex` line 316 (same sentence as #1).
- **Paper says:** *"a moderate batch size of $B=16$"* (implies all models).
- **Correct value:** **B = 16 on Llama-3.2-1B only; B = 8 on Llama-3.1-8B
  and Llama-3-70B** (the larger models OOM at B=16, so the max-fit ladder
  settled at B=8).
- **Evidence:** the settled F3 cells are
  `Llama-3.2-1B-bf16/F3_..._B16_Lp16000_...`,
  `Llama-3.1-8B-bf16/F3_..._B8_Lp16000_...`,
  `Llama-3-70B-bf16/F3_..._B8_Lp16000_...`. `final_paper_results.md`
  labels them "K=64 B=16" (1B) and "K=64 B=8" (8B, 70B).
- **Suggested fix:** *"a batch size of $B=16$ (1B) / $B=8$ (8B, 70B)"* or
  state the max-fit-per-model ladder.

## 3. F4 self-consistency — prefix length (uniform 16K → per-model)

- **Location:** `paper.tex` line 313 (§Scenarios, `\textbf{self-consistency}`).
- **Paper says:** *"This setting has a **moderate prefix of 16K tokens**, a
  small batch size of $B=1$, and $K=16$ branches."* (implies all models).
- **Correct value:** **L_p is per-model — 1B = 32K, 8B = 16K, 70B = 64K.**
  Only the 8B number is 16K; the 1B run used 32768 and the 70B run used
  65536 tokens.
- **Evidence:** the F4 cells are
  `Llama-3.2-1B-bf16/F4_self_consistency_K16_B1_Lp32768_mn12000.csv`,
  `Llama-3.1-8B-bf16/...Lp16384...`,
  `Llama-3-70B-bf16/...Lp65536...`; `build_final_results.py` hard-codes
  `F4_LP_BY_MODEL = {1B: 32768, 8B: 16384, 70B: 65536}`, and
  `final_paper_results.md` labels them "L_p=32K", "L_p=16K", "L_p=64K".
  The per-model prefix is deliberate: it places each model in the
  1POOL-vs-DEC-TAIL crossover regime its forward/attention balance needs.
- **Suggested fix:** *"a per-model prefix (32K on 1B, 16K on 8B, 64K on
  70B), a batch size of $B=1$, and $K=16$ branches."*

---

## Settings that DO match (verified, no change needed)

- **F1 multi-doc-qa:** ~80K prefix, B=1, K=128, mn=256. ✔ (`paper.tex` line ~300)
- **F2 multi-few-shot:** 4K prefix, K=128, mn=256, B=16 (1B) / B=4 (8B, 70B).
  ✔ matches `final_paper_results.md` (1B B=16, 8B B=4, 70B B=4) — line 304.
- **F4 output length:** 12,000 tokens (long-decode). ✔
- **Models / precision / TP:** Llama-3.2-1B & Llama-3.1-8B bf16 TP=1;
  Llama-3-70B-Instruct bf16 TP=4. ✔ (`exp_final_paper_70b_bf16.sbatch`
  `NPROC=4`, `BE_DTYPE=bf16`, `BE_KV_DTYPE=bf16`).

## Minor (non-config) wording note

- `paper.tex` line 316 says beam-search is *"Unlike the previous **two**
  scenarios"* — but three scenarios are introduced before it (F1, F2, and
  self-consistency F4). Should read "previous **three** scenarios" (or
  reorder so beam-search follows multi-few-shot). Not a config error;
  flagged for completeness.
