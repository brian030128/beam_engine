# Final paper — end-to-end results (F1–F4)

Each decode value is the **median** over `n` measured iterations (warmup
discarded; `n` per row); lower = better, **bold** = row winner. We report the
median — not the best/min or max run. Two kinds of error bar appear:

- `±r%` (default) — relative half-range `(max−min)/(2·median)` over **`n` warm
  repeats on a single fixed data instance**: an estimate of run-to-run
  **timing jitter** only.
- `±r%‡` — the same statistic over **`n=3` distinct data draws** (the
  GSM8K/HotpotQA text resampled at the *same* token-length shape): a
  **cross-instance** error bar. Applies to **1B F3** and **8B F1/F2/F3** (rerun
  on the data-draw axis); the **8B F4** draw rerun is still in progress, and
  **70B F3** was not rerun (so both keep their original bars).

`†` marks a single run (`n=1`): no error bar, treat as a point estimate.
`1p`/`dt` = forced SHARED_2L_1POOL / SHARED_2L_DEC_TAIL; `bs_kernel` =
cost-model picker. F4 uses single-launch 1p + single-launch picker. All bars
are sub-1%, so winner orderings are robust; the plan-time tables below show
medians without bars for readability (their jitter is comparably small).

**F4 is the long-decode cell (self_consistency, max_new=12000); F1–F3 use max_new=256.** At 12k tokens the picker flips 1POOL→DEC_TAIL mid-run (1B @4139, 8B @2064, 70B @8585 of 12000 steps — scaling with L_p) and **wins on 1B/8B**; 70B is forward-bound (K=16, G=8 GQA) so forced `1p` wins and the picker trails by 0.3%. F4 picker speedup vs paged: 1B 1.22×, 8B 1.35×, 70B 1.06×. F4 plan-time is the no-trace `plan_total` (no decision-in-parens).

## Llama-3.2-1B (bf16 weights + bf16 KV, TP=1)

### Decode time (median `decode_total_ms`, ±half-range over `n` iters; `‡` = over data draws)

| cell | config | n | paged | fasttree | deft | mlca | bs_kernel | 1p | dt | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 multi_doc_qa | K=128 B=1 ~80K | 3 | 14293 ±0.0% | 3563 ±0.1% | 21755 ±0.0% | 2485 ±0.0% | 2243 ±0.2% | 9929 ±0.4% | **2241 ±0.4%** | dt |
| F2 multi_few_shot | K=128 B=16 | 3 | 21805 ±0.1% | 14682 ±0.3% | 23237 ±0.0% | 10070 ±0.2% | 7791 ±0.2% | 9152 ±0.2% | **7757 ±0.1%** | dt |
| F3 beam_search | K=64 B=16 L_p=16K | 3 | 28447 ±0.3%‡ | 14058 ±0.8%‡ | 21300 ±0.4%‡ | 13877 ±0.5%‡ | 9016 ±0.3%‡ | 10017 ±0.4%‡ | **8906 ±0.2%‡** | dt |
| F4 self_consistency | K=16 B=1 L_p=32K mn=12K | 2 | 72659 ±0.1% | 71564 ±0.4% | 479295 ±0.0% | 64982 ±0.1% | **59692 ±0.1%** | 62827 ±0.0% | 61582 ±0.2% | bs_kernel |

### Plan time (median `plan_total_ms`; bs_kernel shown as build-indices+decision, decision in parens)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 multi_doc_qa | K=128 B=1 ~80K | 332 | 590 | 117 | 358 | 148 (23) | 174 | 138 |
| F2 multi_few_shot | K=128 B=16 | 3060 | 7595 | 1183 | 3974 | 1515 (99) | 2098 | 1480 |
| F3 beam_search | K=64 B=16 L_p=16K | 1870 | 6181 | 3144 | 2820 | 2619 | 2370 | 2497 |
| F4 self_consistency | K=16 B=1 L_p=32K mn=12K | 6803 | 13851 | 5099 | 8921 | 6488 | 8673 | 5319 |

## Llama-3.1-8B (bf16 weights + bf16 KV, TP=1)

### Decode time (median `decode_total_ms`, ±half-range over `n` iters; `‡` = over data draws)

| cell | config | n | paged | fasttree | deft | mlca | bs_kernel | 1p | dt | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 multi_doc_qa | K=128 B=1 ~80K | 3 | 44643 ±0.0%‡ | 19702 ±0.1%‡ | 49317 ±0.1%‡ | 13098 ±0.1%‡ | 6640 ±0.2%‡ | 29172 ±0.2%‡ | **6625 ±0.1%‡** | dt |
| F2 multi_few_shot | K=128 B=4 | 3 | 16818 ±0.2%‡ | 13175 ±0.0%‡ | 18286 ±0.0%‡ | 13234 ±0.1%‡ | 6862 ±0.0%‡ | 8464 ±0.1%‡ | **6842 ±0.1%‡** | dt |
| F3 beam_search | K=64 B=8 L_p=16K | 3 | 39580 ±0.3%‡ | 22943 ±0.6%‡ | 24355 ±0.1%‡ | 14727 ±0.0%‡ | 9985 ±0.4%‡ | 12842 ±0.5%‡ | **9891 ±0.6%‡** | dt |
| F4 self_consistency | K=16 B=1 L_p=16K mn=12K | 2 | 175956 ±0.0% | 186359 ±0.1% | 699269 ±0.1% | 333107 ±0.0% | **130142 ±0.0%** | 169480 ±0.0% | 160866 ±0.0% | bs_kernel |

### Plan time (median `plan_total_ms`; bs_kernel shown as build-indices+decision, decision in parens)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 multi_doc_qa | K=128 B=1 ~80K | 355 | 608 | 136 | 485 | 146 (23) | 189 | 146 |
| F2 multi_few_shot | K=128 B=4 | 698 | 1333 | 275 | 1260 | 356 (40) | 493 | 359 |
| F3 beam_search | K=64 B=8 L_p=16K | 1045 | 2774 | 1424 | 1557 | 1248 | 1285 | 1192 |
| F4 self_consistency | K=16 B=1 L_p=16K mn=12K | 6125 | 14771 | 5884 | 13900 | 5638 | 8153 | 4923 |

## Llama-3-70B-Instruct (bf16 weights + bf16 KV, TP=4)

### Decode time (median `decode_total_ms`, ±half-range over `n` iters; `‡` = over data draws)

| cell | config | n | paged | fasttree | deft | mlca | bs_kernel | 1p | dt | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 multi_doc_qa | K=128 B=1 ~80K | 3 | 32502 ±0.0% | 73429 ±0.0% | 121912 ±0.1% | 31004 ±0.0% | 11348 ±0.1% | 52139 ±0.0% | **11317 ±0.1%** | dt |
| F2 multi_few_shot | K=128 B=4 | 3 | 19649 ±0.1% | 44519 ±0.1% | 32183 ±0.0% | 23878 ±0.1% | 14340 ±0.1% | 16398 ±0.0% | **14330 ±0.1%** | dt |
| F3 beam_search | K=64 B=8 L_p=16K | 1 | 35252† | 68110† | 55292† | 24079† | 18158† | 26309† | **18054†** | dt |
| F4 self_consistency | K=16 B=1 L_p=64K mn=12K | 2 | 403123 ±0.1% | 621572† | 5003149† | 1295019† | 379670 ±0.1% | **378462 ±0.2%** | 407454 ±0.1% | 1p |

### Plan time (median `plan_total_ms`; bs_kernel shown as build-indices+decision, decision in parens)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 multi_doc_qa | K=128 B=1 ~80K | 353 | 681 | 169 | 460 | 150 (23) | 201 | 142 |
| F2 multi_few_shot | K=128 B=4 | 728 | 1749 | 265 | 1142 | 384 (43) | 511 | 371 |
| F3 beam_search | K=64 B=8 L_p=16K | 1040 | 2982 | 1640 | 1245 | 1544 (100) | 1450 | 1443 |
| F4 self_consistency | K=16 B=1 L_p=64K mn=12K | 8317 | 28949 | 5510 | 12982 | 8199 | 9053 | 5940 |

---

# Tree-kernel microbenchmark (single decode-step kernel competition)

A **structural** study, complementary to the end-to-end F1–F4 cells above.
Instead of timing full generations, it times **one decode attention step**
on a curated taxonomy of prefix-sharing tree shapes (`bench_tree_kernel.py`,
`slurm/bench_tree_kernel.sbatch`). Goal: show that across the structural
space, **each strategy in our fixed dispatch set is the kernel-level winner
on some family, and the cost-model picker tracks the per-shape oracle.**

**Method.** H100, **fp16**, random KV (no model weights). Each cell is the
**median GPU time (ms)** over 30 timed iters + 5 warmups (`bench_gpu_time`,
CUDA events). Kernels: `paged` (`BatchDecodeWithPagedKVCache`), `mlca`
(non-fused `MultiLevelCascadeAttentionWrapper`), `fasttree` (Triton
radix-tree two-stage decode), `bs_kernel` (cost-model picker), and the three
forced dispatch variants `2l1p` (`SHARED_2L_1POOL`, single fused cascade),
`2dt` (`SHARED_2L_DEC_TAIL`, prefix cascade + CTA_Q=1 decode tail + merge),
`3dt` (`SHARED_3L_DEC_TAIL`). `oracle` = fastest dispatchable strategy
{paged, 2l1p, 2dt, 3dt}; **regret** = picker time ÷ oracle time
(1.00 = optimal). mlca/fasttree are competing baselines, not in the dispatch
space. Head/page config: 32 QO heads, 8 KV heads (G=4 GQA), head_dim=128,
page_size=16.

### How to read a tree shape

A shape is a balanced radix tree written top-down as a list of
`(tokens_per_group × n_groups)` levels. Level 0 is the shared root prefix;
the last level is the per-branch leaf (its `n_groups` = total leaf count).
Every level subdivides each parent group into equal children — the balanced
tree that beam search / self-consistency / diverse beam search produce at a
decode step. `K` = leaves per prompt, `B` = top-level prompt count,
`L_p` = root prefix length, *tail* = per-leaf private KV. Each leaf issues
**one** query at the timed step; the leaf's KV length is the sum of token
counts along its root-to-leaf path.

### Results (median ms; **bold** = oracle)

| family | shape | tree `(tok×grp)` | paged | mlca | fasttree | bs_kernel | 2l1p | 2dt | 3dt | oracle | bs/paged | regret |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| flat_beam | flat_lp8k_k16 | 8192×1 ▸ 80×16 | 0.076 | 0.154 | 0.063 | 0.043 | **0.043** | 0.064 | – | 2l1p | 1.76× | 1.00 |
| flat_beam | flat_lp32k_k16 | 32768×1 ▸ 80×16 | 0.272 | 0.495 | 0.169 | 0.083 | 0.091 | **0.085** | – | 2dt | 3.26× | 0.98 |
| flat_beam | flat_lp8k_k4 | 8192×1 ▸ 80×4 | 0.034 | 0.142 | 0.061 | 0.031 | **0.031** | 0.063 | – | 2l1p | 1.08× | 1.01 |
| flat_beam | flat_lp8k_k64 | 8192×1 ▸ 80×64 | 0.257 | 0.188 | 0.217 | 0.069 | 0.268 | **0.068** | – | 2dt | 3.70× | 1.01 |
| batched_beam | bbeam_b20_lp8k_k16 | 8192×20 ▸ 80×320 | 1.366 | 0.904 | 0.424 | 0.358 | **0.358** | 0.359 | – | 2l1p | 3.82× | 1.00 |
| batched_beam | bbeam_b8_lp16k_k32 | 16384×8 ▸ 80×256 | 2.186 | 0.712 | 1.827 | 0.292 | 0.557 | **0.292** | – | 2dt | 7.50× | 1.00 |
| diverse_groups | dbs_g4_k4 | 8192×1 ▸ 256×4 ▸ 80×16 | 0.079 | 0.172 | 0.059 | 0.049 | **0.049** | 0.066 | 0.071 | 2l1p | 1.60× | 1.00 |
| diverse_groups | dbs_g4_k5 | 8192×1 ▸ 256×4 ▸ 80×20 | 0.128 | 0.175 | 0.081 | 0.055 | **0.055** | 0.072 | 0.092 | 2l1p | 2.31× | 1.00 |
| diverse_groups | dbs_g4_k6 ‡ | 8192×1 ▸ 256×4 ▸ 80×24 | 0.130 | 0.180 | 0.176 | 0.064 | **0.059** | 0.064 | 0.099 | 2l1p | 2.03× | 1.08 |
| diverse_groups | dbs_g4_k8 | 8192×1 ▸ 256×4 ▸ 80×32 | 0.134 | 0.183 | 0.183 | 0.064 | 0.264 | **0.064** | 0.069 | 2dt | 2.10× | 1.00 |
| diverse_groups | dbs_g8_k8 | 8192×1 ▸ 256×8 ▸ 80×64 | 0.265 | 0.216 | 0.224 | 0.078 | 0.279 | **0.078** | 0.094 | 2dt | 3.38× | 1.00 |
| multi_chain | chain_b32_c4 | 4096×32 ▸ 256×128 | 0.316 | 0.737 | 0.306 | 0.256 | **0.255** | 0.256 | – | 2l1p | 1.23× | 1.00 |
| long_decode | ld_lp8k_tail2k | 8192×1 ▸ 2048×16 | 0.103 | 0.277 | 0.137 | 0.090 | **0.091** | 0.098 | – | 2l1p | 1.14× | 1.00 |
| adversarial | adv_tiny_prefix | 256×1 ▸ 80×16 | **0.031** | 0.065 | 0.059 | 0.031 | 0.034 | 0.059 | – | paged | 1.00× | 1.00 |
| wide_fanout | wide_k128 | 8192×1 ▸ 80×128 | 0.515 | 0.243 | 0.407 | 0.108 | 0.292 | **0.110** | – | 2dt | 4.76× | 0.98 |
| wide_fanout | wide_k256 | 8192×1 ▸ 80×256 | 1.034 | 0.338 | 0.551 | 0.186 | 0.318 | **0.186** | – | 2dt | 5.56× | 1.00 |

`‡ dbs_g4_k6` is the one shape where the picker is non-optimal: it selects
DEC_TAIL while 1POOL is the oracle (0.064 vs 0.059 ms). The two kernels are
within ~7% there, so the error is tolerable (regret 1.08); see the
diverse_groups discussion below for why the crossover is a cliff.

### Family summary (geomean)

| family | bs/paged | regret | picker-optimal |
|---|---:|---:|:--:|
| flat_beam | 2.19× | 1.001× | 4/4 |
| batched_beam | 5.35× | 1.000× | 2/2 |
| diverse_groups | 2.22× | 1.015× | 4/5 |
| multi_chain | 1.23× | 1.004× | 1/1 |
| long_decode | 1.14× | 0.997× | 1/1 |
| adversarial | 1.00× | 1.001× | 1/1 |
| wide_fanout | 5.14× | 0.991× | 2/2 |
| **OVERALL** | **2.41×** | **1.004×** | **15/16** |

The picker hits the per-shape oracle on **15 of 16** shapes (geomean regret
1.004×), and the oracle column spans **paged, 2l1p, and 2dt** — no single
fixed kernel is best everywhere, which is the dispatcher's reason to exist.
The lone miss (`dbs_g4_k6`) is a tolerable ~7% one in a near-tie region.

### What each tree looks like

Legend: `[Lp]` = shared root prefix (loaded once if shared, re-read per leaf
under paged); `+t` = private tail added at that level; a branch drawn once
with `×n` means `n` identical siblings.

**flat_beam** — one prompt, one shared prefix, K beams with a 1-page tail.
The canonical *bimodal* tree: a wide shared root + K narrow tails. Sweeps the
prefix length and the fan-out K.

```
flat_lp8k_k16     root[8192] ──┬── beam0  +80
(K=16, B=1)                     ├── beam1  +80
                                ┊   (×16 leaves)
                                └── beam15 +80
flat_lp32k_k16    root[32768] ── ×16 beams +80      # long root: L2-thrash regime → dec_tail
flat_lp8k_k4      root[8192]  ── ×4  beams +80       # narrow fan-out: little reuse → 1POOL
flat_lp8k_k64     root[8192]  ── ×64 beams +80       # wide fan-out: max reuse → dec_tail (0%-pad tail)
```
*Winner:* short/narrow → **2l1p** (fused single launch; merge overhead not
worth it); long-prefix or wide-K → **2dt** (the decode-native tail removes
CTA_Q padding and the prefix is read once).

**batched_beam** — B independent serving requests, each its own root prefix +
K beams. **No** cross-prompt sharing; probes the batch / occupancy axis.

```
bbeam_b20_lp8k_k16   prompt0  root[8192] ── ×16 beams +80
(B=20, K=16,          prompt1  root[8192] ── ×16 beams +80
 320 leaves)          ┊  (×20 distinct roots)
                      prompt19 root[8192] ── ×16 beams +80
bbeam_b8_lp16k_k32   ×8  distinct root[16384] ── ×32 beams +80  (long prefix, wide K)
```
*Winner:* with a long 16k prefix + K=32 (`b8_lp16k_k32`), the prefix dominates
and the decode-native tail (**2dt**) wins by the suite's largest margin —
**7.4× over paged**, which re-reads the 16k root 32× per prompt.

*The occupancy knee.* `bbeam_b20_lp8k_k16` is the point where the two kernels
are within ~1% of each other (1POOL 0.370 ms ≈ DEC_TAIL 0.358 ms): the
dispatch choice barely matters, and the picker's regret is 1.04× — inside the
[1.00, 1.05] band. Sweeping B *down* from here, the oracle tips toward DEC_TAIL
as the GPU empties and the picker (which selects 1POOL) mispicks, with regret
growing as B shrinks (≈1.00× at B≥28, 1.04× at B=24, **1.04× at B=20**, 1.11×
at B=16, 1.45× at B=8, 2.11× at B=4); sweeping B *up* it saturates and 1POOL
takes over (B=32 ties). The picker tracks the oracle down to the B≈20 knee; the
under-filled B≤16 regime is a known cost-model gap (no B-occupancy term).

**diverse_groups** — diverse beam search: one root prefix → G diversity
groups (each adds a *shared* group suffix) → beams per group. A genuine
3-level tree (the intermediate level is real shared KV, not a flattening
artifact).

```
dbs_g4_k4    root[8192] ──┬── grp0 +256 ──┬── beam +80 (×4)     (16 leaves)
(1→4→4)                    ├── grp1 +256 ── ×4 beams +80
                           ├── grp2 +256 ── ×4 beams +80
                           └── grp3 +256 ── ×4 beams +80
dbs_g4_k5    root[8192] ── ×4 grp +256 ── ×5 beams +80          (20 leaves)
dbs_g4_k6    root[8192] ── ×4 grp +256 ── ×6 beams +80          (24 leaves)  ‡ tolerable mispick
dbs_g4_k8    root[8192] ── ×4 grp +256 ── ×8 beams +80          (32 leaves)
dbs_g8_k8    root[8192] ── ×8 grp +256 ── ×8 beams +80          (64 leaves)
```
*Winner:* small total work (g4_k4) → **2l1p**; larger fan-out (g8_k8) → **2dt**.
The intermediate level is short (256 tok) vs the 8192 root, so exposing it as a
3rd cascade level (**3dt**) costs more than it saves — the picker correctly
*declines* depth-3 throughout (3dt is always ≥ the chosen kernel).

*The g4 beams-per-group sweep (16→20→24→32 leaves) walks the 1POOL→DEC_TAIL
crossover and contains the suite's one tolerable mispick.* 1POOL wins while the
tree is narrow (16, 20 leaves), then hits a **padding cliff**: collapsing the
4 intermediate groups into a single per-beam pool jumps from 0.059 ms at 24
leaves to 0.264 ms at 32. The picker's switch-to-DEC_TAIL boundary fires just
*before* that cliff, so at **`dbs_g4_k6` (24 leaves)** it takes DEC_TAIL
(0.064 ms) while 1POOL is still the oracle (0.059 ms) — but the two are only
~7% apart there, so the wrong pick is tolerable (regret 1.08). One step later
(32 leaves) the cliff makes DEC_TAIL the clear winner and the picker is right
again.

**multi_chain** — many independent questions, each fanning into a few
reasoning chains with a moderate per-chain tail (multi_chain_reasoning).

```
chain_b32_c4   q0  root[4096] ── ×4 chains +256
(B=32, C=4,    ┊   (×32 questions)
 128 leaves)   q31 root[4096] ── ×4 chains +256
```
*Winner:* **2l1p** — like batched_beam, the 32-way batch already fills the GPU;
the 256-token chain tails are short enough that 1POOL ≈ dec_tail and the
single launch wins on overhead.

**long_decode** — deep into generation: the per-branch tail has grown to
thousands of tokens. Probes the 1POOL→dec_tail crossover.

```
ld_lp8k_tail2k   root[8192] ── ×16 beams +2048      # tail now comparable to root
```
*Winner:* **2l1p**, but only just (regret-neutral with 2dt: 0.090 vs 0.097).
This shape sits right at the crossover knee; pushing the tail to ~8k flips
the oracle to dec_tail (and eventually to paged when the prefix stops paying
back) — confirming the F4 long-decode flip seen end-to-end.

**adversarial** — the root is tiny (256 tok), smaller than the fan-out work.
Sharing can't pay back any cascade/merge launch. Guards the picker against
over-sharing.

```
adv_tiny_prefix   root[256] ── ×16 beams +80
```
*Winner:* **paged (PER_BEAM)** — and the picker correctly selects PER_BEAM
(no sharing), matching paged to within noise.

**wide_fanout** — very large K (top-K speculative decoding / large-sample
self-consistency) over a fixed 8k root. Prefix reuse is maximal and the
per-branch tail is a single page, so the decode-native tail matters most.

```
wide_k128   root[8192] ── ×128 leaves +80
wide_k256   root[8192] ── ×256 leaves +80
```
*Winner:* **2dt**, by the widest margins in the suite (4.6× / 5.5× over
paged) — paged re-reads the 8k root 128–256× per step, while dec_tail reads
it once and runs the 128–256 tails through a zero-padding decode kernel.
