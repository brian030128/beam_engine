# Fastest method per cell (with speedup over runner-up)

Model: meta-llama/Llama-3.2-1B (fp16). H100 80GB. ms/step.


## L_p=8192, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 5.09 (1.13×) | **paged** 4.73 (1.32×) | **paged** 5.01 (1.27×) | **paged** 6.76 (1.10×) | **paged** 11.08 (1.01×) |
| K=16 | **paged** 5.04 (1.23×) | **paged** 6.72 (1.06×) | **fasttree** 8.75 (1.24×) | **fasttree** 12.75 (1.42×) | **fasttree** 23.98 (1.45×) |
| K=32 | **paged** 5.36 (1.21×) | **bs_kernel** 8.21 (1.17×) | **bs_kernel** 12.51 (1.17×) | **bs_kernel** 22.85 (1.18×) | **bs_kernel** 43.82 (1.25×) |
| K=64 | **bs_kernel** 6.76 (1.03×) | **bs_kernel** 9.87 (1.30×) | **bs_kernel** 17.20 (1.38×) | **bs_kernel** 32.02 (1.48×) | **bs_kernel** 64.70 (1.48×) |


## L_p=8192, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 4.98 (1.12×) | **paged** 6.84 (1.21×) | **paged** 9.26 (1.14×) | **paged** 15.32 (1.03×) | **fasttree** 27.88 (1.01×) |
| K=16 | **paged** 5.75 (1.22×) | **paged** 10.37 (1.09×) | **fasttree** 16.50 (1.10×) | **fasttree** 28.17 (1.20×) | **fasttree** 54.82 (1.19×) |
| K=32 | **paged** 6.63 (1.17×) | **bs_kernel** 14.18 (1.08×) | **bs_kernel** 24.27 (1.06×) | **bs_kernel** 46.08 (1.08×) | **bs_kernel** 89.82 (1.13×) |
| K=64 | **bs_kernel** 8.66 (1.01×) | **bs_kernel** 17.56 (1.16×) | **bs_kernel** 32.30 (1.21×) | **bs_kernel** 63.77 (1.23×) | **bs_kernel** 128.31 (1.24×) |


## L_p=30000, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.57 (1.25×) | **paged** 5.97 (1.03×) | **fasttree** 7.80 (1.18×) | **fasttree** 12.74 (1.29×) |
| K=16 | **paged** 5.98 (1.06×) | **fasttree** 7.45 (1.35×) | **fasttree** 10.96 (1.74×) | **fasttree** 19.58 (1.86×) |
| K=32 | **bs_kernel** 6.55 (1.08×) | **bs_kernel** 12.05 (1.18×) | **bs_kernel** 21.73 (1.16×) | **bs_kernel** 42.08 (1.11×) |
| K=64 | **fasttree** 7.68 (1.14×) | **bs_kernel** 14.76 (1.25×) | **bs_kernel** 26.69 (1.32×) | **bs_kernel** 52.16 (1.35×) |


## L_p=30000, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.94 (1.23×) | **paged** 7.93 (1.02×) | **fasttree** 11.84 (1.12×) | **fasttree** 20.83 (1.17×) |
| K=16 | **paged** 6.82 (1.05×) | **fasttree** 11.57 (1.15×) | **fasttree** 18.84 (1.40×) | **fasttree** 34.17 (1.52×) |
| K=32 | **bs_kernel** 7.83 (1.07×) | **bs_kernel** 17.77 (1.10×) | **bs_kernel** 32.81 (1.11×) | **bs_kernel** 64.96 (1.08×) |
| K=64 | **fasttree** 9.63 (1.11×) | **bs_kernel** 21.95 (1.23×) | **bs_kernel** 41.37 (1.24×) | **bs_kernel** 82.02 (1.23×) |

