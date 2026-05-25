# Cross-method sweep — paper results

Model: `meta-llama/Llama-3.2-1B` (fp16). H100 80GB. Same prompt repeated B times.

OOM cells omitted.


## L_p=8192, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.72 | 8.96 | 12.42 | 19.94 | 35.04 |
| K=16 | 7.20 | 11.71 | 18.95 | 35.04 | 68.03 |
| K=32 | 7.95 | 14.77 | 25.25 | 47.84 | 94.35 |
| K=64 | 8.95 | 18.35 | 33.87 | 67.91 | 137.61 |

## L_p=8192, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.78 | 7.39 | 8.58 | 12.08 | 19.62 |
| K=16 | 6.44 | 7.98 | 11.32 | 18.74 | 36.43 |
| K=32 | 6.62 | 8.50 | 12.89 | 23.37 | 46.45 |
| K=64 | 6.94 | 10.12 | 18.05 | 33.95 | 70.26 |

## L_p=30000, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.17 | 9.57 | 15.71 | 27.93 |
| K=16 | 7.20 | 13.57 | 26.76 | 52.90 |
| K=32 | 7.94 | 18.12 | 34.12 | 66.65 |
| K=64 | 10.81 | 22.39 | 42.75 | 85.69 |

## L_p=30000, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.07 | 7.82 | 12.01 | 20.31 |
| K=16 | 6.35 | 10.23 | 19.61 | 36.95 |
| K=32 | 6.60 | 12.21 | 22.24 | 43.33 |
| K=64 | 8.82 | 15.23 | 27.36 | 53.98 |


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
