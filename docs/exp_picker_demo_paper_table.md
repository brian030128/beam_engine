# Picker-demo paper table (combined across all models)

Each row is one cell. Times are median-across-repeats of `decode_p50_ms` (per-decode-step latency). Speedup columns are `bs_kernel / baseline` — values < 1.0 mean bs_kernel wins.

| Model | ID | Scenario | K | B | mn | paged | fasttree | deft | bs_kernel | 1POOL | DT | bs/paged | bs/ft | bs/deft | 1p/dt | picker | match |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Llama-3.2-1B-FP8 | **A1** | beam_search | 4 | 1 | 64 | 550.66 | 623.94 | 3257.25 | 599.69 | 599.43 | 634.81 | 1.09 | 0.96 | 0.18 | 0.944 | 1POOL | ✓ |
| Llama-3.2-1B-FP8 | **A2** | multi_chain_reasoning | 4 | 1 | 32 | 8.17 | 9.09 | 10.59 | 8.84 | 8.86 | 9.42 | 1.08 | 0.97 | 0.83 | 0.940 | 1POOL | ✓ |
| Llama-3.2-1B-FP8 | **A3** | multi_doc_qa | 32 | 1 | 256 | 23.33 | 11.04 | 105.75 | 9.41 | 8.71 | 9.33 | 0.40 | 0.85 | 0.09 | 0.934 | DT | ✗ |
| Llama-3.2-1B-FP8 | **A4** | multi_few_shot | 16 | 8 | 256 | 10.72 | 10.20 | 16.92 | 9.41 | 8.93 | 9.43 | 0.88 | 0.92 | 0.56 | 0.947 | DT | ✗ |
| Llama-3.2-1B-FP8 | **A5** | multi_level_system | 32 | 4 | 256 | 8.78 | 10.45 | 10.68 | 8.83 | 8.78 | 9.26 | 1.01 | 0.84 | 0.83 | 0.948 | 1POOL | ✓ |
| Llama-3.2-1B-FP8 | **D1** | multi_chain_reasoning | 32 | 32 | 256 | 33.06 | 30.90 | 36.85 | 17.37 | 18.85 | 16.34 | 0.53 | 0.56 | 0.47 | 1.153 | DT | ✗ |
| Llama-3.2-1B-FP8 | **D2** | multi_few_shot | 16 | 32 | 256 | 27.92 | 18.27 | 36.42 | 14.17 | 16.24 | 13.88 | 0.51 | 0.78 | 0.39 | 1.170 | DT | ✗ |
| Llama-3.2-1B-FP8 | **D3** | multi_level_system | 32 | 16 | 256 | 17.17 | 14.75 | 20.63 | 10.56 | 10.71 | 10.28 | 0.61 | 0.72 | 0.51 | 1.042 | 1POOL | ✗ |
| Llama-3.2-1B-FP8 | **D4** | multi_doc_qa | 16 | 4 | 256 | 41.59 | 11.80 | 123.27 | 15.94 | 45.82 | 15.82 | 0.38 | 1.35 | 0.13 | 2.897 | DT | ✓ |
| Llama-3.1-8B-FP8 | **B1** | beam_search | 4 | 1 | 128 | 2128.09 | 2354.62 | 13130.60 | 2163.45 | 2174.16 | 2301.06 | 1.02 | 0.92 | 0.16 | 0.945 | 1POOL | ✓ |
| Llama-3.1-8B-FP8 | **B2** | multi_chain_reasoning | 4 | 1 | 32 | 15.89 | 17.56 | 20.85 | 16.61 | 16.40 | 17.52 | 1.05 | 0.95 | 0.80 | 0.936 | 1POOL | ✗ |
| Llama-3.1-8B-FP8 | **B3** | multi_doc_qa | 64 | 1 | 256 | 108.66 | 52.02 | 211.29 | 21.13 | 26.91 | 21.02 | 0.19 | 0.41 | 0.10 | 1.280 | DT | ✓ |
| Llama-3.1-8B-FP8 | **B4** | multi_few_shot | 32 | 4 | 256 | 23.32 | 25.84 | 34.84 | 18.23 | 18.14 | 18.43 | 0.78 | 0.71 | 0.52 | 0.984 | 1POOL | ✓ |
| Llama-3.1-8B-FP8 | **B5** | multi_level_system | 64 | 4 | 256 | 24.52 | 25.38 | 32.78 | 16.78 | 16.74 | 17.07 | 0.68 | 0.66 | 0.51 | 0.981 | 1POOL | ✓ |
| Llama-3.1-8B-FP8 | **B6** | multi_chain_reasoning | 64 | 32 | 256 | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama-3.1-8B-FP8 | **D1** | multi_chain_reasoning | 32 | 32 | 256 | 80.41 | 94.92 | 95.82 | 48.90 | 48.74 | 45.68 | 0.61 | 0.52 | 0.51 | 1.067 | 1POOL | ✗ |
| Llama-3.1-8B-FP8 | **D2** | multi_few_shot | 16 | 32 | 256 | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama-3.1-8B-FP8 | **D3** | multi_level_system | 32 | 16 | 256 | 42.84 | 52.08 | 57.59 | 27.30 | 27.18 | 26.32 | 0.64 | 0.52 | 0.47 | 1.033 | 1POOL | ✗ |
| Llama-3.1-8B-FP8 | **D4** | multi_doc_qa | 16 | 4 | 256 | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama-3-70B-Instruct-FP8 | **C1** | beam_search | 4 | 1 | 64 | 3250.99 | 3470.88 | 18383.37 | 3346.34 | 3382.71 | 3695.76 | 1.03 | 0.96 | 0.18 | 0.915 | 1POOL | ✓ |
| Llama-3-70B-Instruct-FP8 | **C2** | multi_chain_reasoning | 4 | 1 | 32 | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama-3-70B-Instruct-FP8 | **C3** | multi_doc_qa | 32 | 1 | 256 | 123.18 | — | — | 63.32 | 74.48 | 63.28 | 0.51 | — | — | 1.177 | DT | ✓ |
| Llama-3-70B-Instruct-FP8 | **C4** | multi_few_shot | 16 | 4 | 256 | 50.06 | — | — | 64.55 | 64.23 | 64.66 | 1.29 | — | — | 0.993 | DT | ✓ |
| Llama-3-70B-Instruct-FP8 | **C5** | multi_level_system | 32 | 4 | 256 | 50.00 | — | — | 49.52 | 49.54 | 53.21 | 0.99 | — | — | 0.931 | 1POOL | ✓ |
| Llama-3-70B-Instruct-FP8 | **C6** | multi_chain_reasoning | 64 | 4 | 256 | 74.78 | — | — | 64.27 | 67.84 | 64.04 | 0.86 | — | — | 1.059 | DT | ✓ |
| Llama-3-70B-Instruct-FP8 | **D1** | multi_chain_reasoning | 32 | 32 | 256 | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama-3-70B-Instruct-FP8 | **D2** | multi_few_shot | 16 | 32 | 256 | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama-3-70B-Instruct-FP8 | **D3** | multi_level_system | 32 | 16 | 256 | — | — | — | — | — | — | — | — | — | — | — | _missing_ |
| Llama-3-70B-Instruct-FP8 | **D4** | multi_doc_qa | 16 | 4 | 256 | — | — | — | — | — | — | — | — | — | — | — | _missing_ |

## Aggregate

- **bs/paged**: 21 cells, 14 bs_kernel wins, geomean=0.705
- **bs/fasttree**: 17 cells, 16 bs_kernel wins, geomean=0.768
- **bs/deft**: 17 cells, 17 bs_kernel wins, geomean=0.342
- **picker match**: 13/21 cells within 1% of oracle
  - worst miss: Llama-3.2-1B-FP8 / A3 (multi_doc_qa) at 1.080× oracle
