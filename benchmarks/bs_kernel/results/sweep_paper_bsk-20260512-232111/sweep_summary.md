# Cross-method sweep — paper results

Model: `meta-llama/Llama-3.2-1B` (fp16). H100 80GB. Same prompt repeated B times.

OOM cells omitted.


## L_p=8192, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.68 | 8.82 | 12.12 | 19.35 | 33.58 |
| K=16 | 7.03 | 11.33 | 16.96 | 30.19 | 55.43 |
| K=32 | 7.78 | 13.85 | 23.64 | 44.33 | 86.53 |
| K=64 | 8.69 | 17.62 | 32.20 | 63.30 | 127.13 |

## L_p=8192, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.63 | 7.10 | 8.15 | 11.15 | 17.57 |
| K=16 | 6.28 | 7.52 | 9.70 | 14.34 | 25.51 |
| K=32 | 6.49 | 8.19 | 11.82 | 20.81 | 39.17 |
| K=64 | 6.82 | 9.89 | 17.35 | 30.86 | 64.17 |

## L_p=30000, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.13 | 9.13 | 14.84 | 26.11 |
| K=16 | 7.04 | 11.89 | 19.80 | 36.89 |
| K=32 | 7.71 | 16.12 | 29.40 | 57.29 |
| K=64 | 9.43 | 21.72 | 41.30 | 81.28 |

## L_p=30000, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 5.98 | 7.35 | 11.00 | 18.88 |
| K=16 | 6.27 | 8.08 | 12.41 | 21.59 |
| K=32 | 6.47 | 10.57 | 18.50 | 35.09 |
| K=64 | 7.71 | 14.52 | 25.70 | 49.18 |


## bs_kernel dominant pick per cell

Format: ``strategy (depth, pool, t_large) — N/255 steps``


### L_p=8192, bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | PER_BEAM 255/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 | SHARED_3L_1POOL (3,1,64) 236/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 223/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 173/255 | SHARED_3L_1POOL (3,1,64) 139/255 | SHARED_3L_1POOL (3,1,64) 123/255 | SHARED_2L_DEC_TAIL (2,1,64) 124/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 183/255 | SHARED_2L_DEC_TAIL (2,1,64) 161/255 | SHARED_2L_DEC_TAIL (2,1,64) 178/255 | SHARED_2L_DEC_TAIL (2,1,64) 187/255 | SHARED_2L_DEC_TAIL (2,1,64) 192/255 |

### L_p=8192, dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | PER_BEAM 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 169/255 | SHARED_3L_1POOL (3,1,64) 148/255 | SHARED_3L_1POOL (3,1,64) 130/255 | SHARED_2L_DEC_TAIL (2,1,64) 130/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 234/255 | SHARED_2L_DEC_TAIL (2,1,64) 154/255 | SHARED_2L_DEC_TAIL (2,1,64) 177/255 | SHARED_2L_DEC_TAIL (2,1,64) 187/255 | SHARED_2L_DEC_TAIL (2,1,64) 195/255 |

### L_p=30000, bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 | SHARED_3L_1POOL (3,1,64) 172/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 159/255 | SHARED_3L_1POOL (3,1,64) 125/255 | SHARED_2L_DEC_TAIL (2,1,64) 129/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 181/255 | SHARED_2L_DEC_TAIL (2,1,64) 162/255 | SHARED_2L_DEC_TAIL (2,1,64) 178/255 | SHARED_2L_DEC_TAIL (2,1,64) 190/255 |

### L_p=30000, dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 | SHARED_2L_1POOL (2,1,64) 255/255 |
| K=16 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 239/255 |
| K=32 | SHARED_3L_1POOL (3,1,64) 239/255 | SHARED_3L_1POOL (3,1,64) 192/255 | SHARED_3L_1POOL (3,1,64) 141/255 | SHARED_2L_DEC_TAIL (2,1,64) 120/255 |
| K=64 | SHARED_3L_1POOL (3,1,64) 185/255 | SHARED_2L_DEC_TAIL (2,1,64) 163/255 | SHARED_2L_DEC_TAIL (2,1,64) 181/255 | SHARED_2L_DEC_TAIL (2,1,64) 189/255 |
