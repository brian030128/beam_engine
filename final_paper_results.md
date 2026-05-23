# Final paper — end-to-end results (F1–F4)

Median over 3 warm repeats; lower = better. `1p`/`dt` = forced SHARED_2L_1POOL / SHARED_2L_DEC_TAIL; `bs_kernel` = cost-model picker. **bold** = row winner. F4 uses single-launch 1p + single-launch picker.

## Llama-3.2-1B (bf16 weights + bf16 KV, TP=1)

### Decode time (median `decode_total_ms`)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F1 multi_doc_qa | K=128 B=1 ~80K | 16893 | 3507 | 21947 | 6062 | 2641 | 3266 | **2630** | dt |
| F2 multi_few_shot | K=128 B=16 | 24812 | 15136 | 23360 | 22317 | **7762** | 9039 | 7808 | bs_kernel |
| F3 beam_search | K=64 B=16 L_p=16K | 32859 | 14333 | 21775 | 14112 | 9042 | 10032 | **8912** | dt |
| F4 self_consistency | K=16 B=1 L_p=16K | 1158 | 1413 | 5129 | 1750 | 1134 | **1118** | 1247 | 1p |

### Plan time (median `plan_total_ms`; bs_kernel shown as build-indices+decision, decision in parens)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 multi_doc_qa | K=128 B=1 ~80K | 3050 | 572 | 121 | 464 | 143 (22) | 170 | 136 |
| F2 multi_few_shot | K=128 B=16 | 6059 | 8075 | 1345 | 5311 | 1524 (92) | 2022 | 1557 |
| F3 beam_search | K=64 B=16 L_p=16K | 6274 | 6721 | 3688 | 3094 | 2758 | 2421 | 2625 |
| F4 self_consistency | K=16 B=1 L_p=16K | 167 | 191 | 57 | 199 | 117 (20) | 111 | 83 |

## Llama-3.1-8B (bf16 weights + bf16 KV, TP=1)

### Decode time (median `decode_total_ms`)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F1 multi_doc_qa | K=128 B=1 ~80K | 47384 | 19940 | 49050 | 13206 | 7425 | 9586 | **7420** | dt |
| F2 multi_few_shot | K=128 B=4 | 17533 | 13369 | 18439 | 13306 | **7107** | 8524 | 7109 | bs_kernel |
| F3 beam_search | K=64 B=8 L_p=16K | 42046 | 23131 | 24662 | 15241 | 10720 | 13111 | **10646** | dt |
| F4 self_consistency | K=16 B=1 L_p=16K | 3137 | 2849 | 11896 | 4191 | 2396 | **2361** | 2593 | 1p |

### Plan time (median `plan_total_ms`; bs_kernel shown as build-indices+decision, decision in parens)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 multi_doc_qa | K=128 B=1 ~80K | 3136 | 730 | 170 | 560 | 191 (27) | 205 | 166 |
| F2 multi_few_shot | K=128 B=4 | 1516 | 1499 | 344 | 1347 | 412 (41) | 531 | 398 |
| F3 beam_search | K=64 B=8 L_p=16K | 3257 | 3020 | 1588 | 1887 | 1399 | 1355 | 1312 |
| F4 self_consistency | K=16 B=1 L_p=16K | 167 | 191 | 57 | 209 | 126 (22) | 119 | 95 |

## Llama-3-70B-Instruct-FP8 (fp8 weights + fp8_e4m3 KV, TP=2)

### Decode time (median `decode_total_ms`)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F1 multi_doc_qa | K=128 B=1 ~80K | 84171 | — | — | **31354** | 39527 | 47578 | 39382 | mlca |
| F2 multi_few_shot | K=128 B=4 | 41937 | — | — | **28134** | 29361 | 35265 | 29348 | mlca |
| F3 beam_search | K=64 B=4 L_p=16K | 44159 | — | — | **22837** | 35014 | 39314 | 34902 | mlca |
| F4 self_consistency | K=16 B=1 L_p=16K | **13024** | — | — | 14227 | 13239 | 13159 | 14324 | paged |

### Plan time (median `plan_total_ms`; bs_kernel shown as build-indices+decision, decision in parens)

| cell | config | paged | fasttree | deft | mlca | bs_kernel | 1p | dt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 multi_doc_qa | K=128 B=1 ~80K | 3457 | — | — | 367 | 159 (23) | 190 | 150 |
| F2 multi_few_shot | K=128 B=4 | 1571 | — | — | 946 | 379 (39) | 492 | 364 |
| F3 beam_search | K=64 B=4 L_p=16K | 1507 | — | — | 529 | 633 | 631 | 587 |
| F4 self_consistency | K=16 B=1 L_p=16K | 172 | — | — | 166 | 123 (21) | 117 | 88 |

## Notes

- **Scenarios.** F1 multi_doc_qa (~80K shared prefix, B=1, K=128); F2 multi_few_shot (~4K prefix, large B, K=128); F3 beam search (multi-level dynamic tree, L_p=16K, K=64); F4 self_consistency / best-of-N (static 2-level: shared prefix → K independent tails, B=1, K=16, L_p=16K).
- **Max-fit batch size.** F2/F3 batch sizes are the largest where all methods fit one node; they differ by model (footnoted in the config column).
- **fp8 baselines on 70B.** `fasttree`/`deft` have no fp8-KV support (no `kv_dtype` path) → shown as “—” on all 70B cells. `mlca` *does* support fp8 once `MlcaBackend.plan` is given the page-table store dtype (fixed in baselines/mlca.py).
- **Plan time.** Sum of per-step index/metadata build over all decode steps. For `bs_kernel` it also includes the cost-model dispatch decision (shown in parens); the decision is a small fraction of bs_kernel's already-small plan cost, and bs_kernel's total plan is far below paged's index-build on the long-prefix cells.
