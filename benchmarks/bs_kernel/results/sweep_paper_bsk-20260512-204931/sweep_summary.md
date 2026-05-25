# Cross-method sweep — paper results

Model: `meta-llama/Llama-3.2-1B` (fp16). H100 80GB. Same prompt repeated B times.

OOM cells omitted.


## L_p=8192, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.71 | 8.97 | 12.34 | 19.76 | 34.38 |
| K=16 | 7.16 | 11.59 | 17.30 | 30.30 | 56.19 |
| K=32 | 7.99 | 14.10 | 23.21 | 44.01 | 85.72 |
| K=64 | 8.93 | 18.12 | 32.39 | 62.94 | 125.89 |

## L_p=8192, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | 5.85 | 7.35 | 8.51 | 11.66 | 18.53 |
| K=16 | 6.47 | 7.80 | 10.07 | 14.96 | 26.49 |
| K=32 | 6.69 | 8.79 | 12.00 | 21.45 | 39.87 |
| K=64 | 7.01 | 10.02 | 16.95 | 31.63 | 62.96 |

## L_p=30000, mode=dbs — ms/step


### dbs_bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.39 | 9.51 | 15.58 | 27.69 |
| K=16 | 7.28 | 12.29 | 20.55 | 37.89 |
| K=32 | 7.94 | 16.10 | 29.88 | 58.10 |
| K=64 | 9.67 | 21.94 | 41.55 | 81.26 |

## L_p=30000, mode=std — ms/step


### bs_kernel

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | 6.26 | 7.75 | 11.75 | 20.32 |
| K=16 | 6.50 | 8.54 | 13.24 | 22.98 |
| K=32 | 6.72 | 10.79 | 18.68 | 36.08 |
| K=64 | 7.89 | 14.69 | 26.05 | 49.64 |


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
