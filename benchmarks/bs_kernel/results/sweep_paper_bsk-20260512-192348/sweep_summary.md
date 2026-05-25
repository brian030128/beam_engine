# Cross-method sweep — paper results

Model: `meta-llama/Llama-3.2-1B` (fp16). H100 80GB. Same prompt repeated B times.

OOM cells omitted.


## L_p=8192, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.77 | 9.04 | 12.82 | 20.03 | 34.65 |
| K=16 | 7.14 | 11.60 | 18.58 | 33.45 | 64.96 |
| K=32 | 7.92 | 14.03 | 24.27 | 45.90 | 89.31 |
| K=64 | 8.81 | 17.54 | 32.55 | 63.21 | 127.39 |

## L_p=8192, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.97 | 46.37 | 8.45 | 11.78 | 18.62 |
| K=16 | 6.43 | 7.81 | 10.68 | 17.92 | 34.14 |
| K=32 | 6.64 | 8.30 | 12.35 | 22.05 | 42.94 |
| K=64 | 6.93 | 9.77 | 17.06 | 32.02 | 63.78 |

## L_p=30000, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.37 | 9.60 | 15.69 | 27.92 |
| K=16 | 7.23 | 12.52 | 25.23 | 51.25 |
| K=32 | 7.96 | 17.63 | 32.73 | 64.43 |
| K=64 | 10.39 | 21.92 | 41.82 | 82.39 |

## L_p=30000, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.22 | 7.78 | 11.81 | 20.52 |
| K=16 | 6.45 | 9.72 | 18.30 | 36.57 |
| K=32 | 6.69 | 11.79 | 21.35 | 41.92 |
| K=64 | 8.54 | 14.66 | 26.23 | 51.66 |


## bs_kernel dominant pick per cell

Format: ``strategy (depth, pool, t_large) — N/255 steps``


### L_p=8192, bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | PER_BEAM 255/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 141/255 | SHARED_2L_DEC_TAIL (2,1,64) 173/255 | SHARED_2L_DEC_TAIL (2,1,64) 225/255 | SHARED_2L_DEC_TAIL (2,1,64) 246/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 180/255 | SHARED_2L_DEC_TAIL (2,1,64) 221/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 131/255 | SHARED_2L_DEC_TAIL (2,1,64) 219/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 | SHARED_2L_DEC_TAIL (2,1,64) 255/255 |

### L_p=8192, dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | PER_BEAM 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 148/255 | SHARED_2L_DEC_TAIL (2,1,64) 182/255 | SHARED_2L_DEC_TAIL (2,1,64) 226/255 | SHARED_2L_DEC_TAIL (2,1,64) 246/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 181/255 | SHARED_2L_DEC_TAIL (2,1,64) 220/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 181/255 | SHARED_2L_DEC_TAIL (2,1,64) 220/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 | SHARED_2L_DEC_TAIL (2,1,64) 255/255 |

### L_p=30000, bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 141/255 | SHARED_2L_DEC_TAIL (2,1,64) 184/255 | SHARED_2L_DEC_TAIL (2,1,64) 228/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 181/255 | SHARED_2L_DEC_TAIL (2,1,64) 223/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 132/255 | SHARED_2L_DEC_TAIL (2,1,64) 221/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 |

### L_p=30000, dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 222/255 | SHARED_2L_DEC_TAIL (2,1,64) 163/255 | SHARED_2L_DEC_TAIL (2,1,64) 221/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_2L_DEC_TAIL (2,1,64) 178/255 | SHARED_2L_DEC_TAIL (2,1,64) 221/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 133/255 | SHARED_2L_DEC_TAIL (2,1,64) 219/255 | SHARED_2L_DEC_TAIL (2,1,64) 244/255 | SHARED_2L_DEC_TAIL (2,1,64) 253/255 |
