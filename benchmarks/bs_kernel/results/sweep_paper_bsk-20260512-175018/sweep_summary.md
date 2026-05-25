# Cross-method sweep — paper results

Model: `meta-llama/Llama-3.2-1B` (fp16). H100 80GB. Same prompt repeated B times.

OOM cells omitted.


## L_p=8192, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.60 | 8.75 | 12.11 | 19.60 | 34.21 |
| K=16 | 7.02 | 11.38 | 18.41 | 33.73 | 65.35 |
| K=32 | 7.75 | 14.18 | 24.27 | 46.08 | 89.82 |
| K=64 | 8.66 | 17.56 | 32.30 | 63.77 | 128.31 |

## L_p=8192, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.73 | 7.13 | 8.27 | 11.63 | 18.57 |
| K=16 | 6.27 | 7.67 | 10.93 | 18.11 | 34.83 |
| K=32 | 6.46 | 8.21 | 12.51 | 22.85 | 43.82 |
| K=64 | 6.76 | 9.87 | 17.20 | 32.02 | 64.70 |

## L_p=30000, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.19 | 9.48 | 15.52 | 27.45 |
| K=16 | 7.17 | 13.32 | 26.32 | 51.84 |
| K=32 | 7.83 | 17.77 | 32.81 | 64.96 |
| K=64 | 10.69 | 21.95 | 41.37 | 82.02 |

## L_p=30000, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.02 | 7.71 | 11.69 | 19.80 |
| K=16 | 6.33 | 10.08 | 19.11 | 36.35 |
| K=32 | 6.55 | 12.05 | 21.73 | 42.08 |
| K=64 | 8.74 | 14.76 | 26.69 | 52.16 |


## bs_kernel dominant pick per cell

Format: ``strategy (depth, pool, t_large) — N/255 steps``


### L_p=8192, bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | PER_BEAM 255/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 124/255 | SHARED_2L_DEC_TAIL (2,1,64) 197/255 | SHARED_2L_DEC_TAIL (2,1,64) 225/255 | SHARED_2L_DEC_TAIL (2,1,64) 246/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 197/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 |
| K=64 | SHARED_2L_DEC_TAIL (2,1,64) 126/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 | SHARED_2L_DEC_TAIL (2,1,64) 255/255 |

### L_p=8192, dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | PER_BEAM 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 145/255 | SHARED_2L_DEC_TAIL (2,1,64) 201/255 | SHARED_2L_DEC_TAIL (2,1,64) 226/255 | SHARED_2L_DEC_TAIL (2,1,64) 246/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 224/255 | SHARED_2L_DEC_TAIL (2,1,64) 198/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 143/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 | SHARED_2L_DEC_TAIL (2,1,64) 255/255 |

### L_p=30000, bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 125/255 | SHARED_2L_DEC_TAIL (2,1,64) 197/255 | SHARED_2L_DEC_TAIL (2,1,64) 228/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 198/255 | SHARED_2L_DEC_TAIL (2,1,64) 225/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 |
| K=64 | SHARED_2L_DEC_TAIL (2,1,64) 124/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 |

### L_p=30000, dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 168/255 | SHARED_2L_DEC_TAIL (2,1,64) 195/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 196/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 |
| K=64 | SHARED_2L_DEC_TAIL (2,1,64) 132/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 |
