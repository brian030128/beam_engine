# Fastest method per cell (with speedup over runner-up)

Model: meta-llama/Llama-3.2-1B (fp16). H100 80GB. ms/step.


## L_p=8192, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 5.09 (1.15×) | **paged** 4.73 (1.32×) | **paged** 5.01 (1.27×) | **paged** 6.76 (1.10×) | **paged** 11.08 (1.01×) |
| K=16 | **paged** 5.04 (1.23×) | **paged** 6.72 (1.06×) | **fasttree** 8.75 (1.15×) | **fasttree** 12.75 (1.17×) | **fasttree** 23.98 (1.10×) |
| K=32 | **paged** 5.36 (1.25×) | **bs_kernel** 8.79 (1.09×) | **bs_kernel** 12.00 (1.22×) | **bs_kernel** 21.45 (1.25×) | **bs_kernel** 39.87 (1.37×) |
| K=64 | **paged** 6.95 (1.01×) | **bs_kernel** 10.02 (1.28×) | **bs_kernel** 16.95 (1.40×) | **bs_kernel** 31.63 (1.50×) | **bs_kernel** 62.96 (1.52×) |


## L_p=8192, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 4.98 (1.15×) | **paged** 6.84 (1.21×) | **paged** 9.26 (1.14×) | **paged** 15.32 (1.03×) | **fasttree** 27.88 (1.01×) |
| K=16 | **paged** 5.75 (1.23×) | **paged** 10.37 (1.09×) | **fasttree** 16.50 (1.05×) | **fasttree** 28.17 (1.08×) | **fasttree** 54.82 (1.03×) |
| K=32 | **paged** 6.63 (1.20×) | **bs_kernel** 14.10 (1.09×) | **bs_kernel** 23.21 (1.11×) | **bs_kernel** 44.01 (1.13×) | **bs_kernel** 85.72 (1.19×) |
| K=64 | **paged** 8.73 (1.02×) | **bs_kernel** 18.12 (1.12×) | **bs_kernel** 32.39 (1.21×) | **bs_kernel** 62.94 (1.25×) | **bs_kernel** 125.89 (1.27×) |


## L_p=30000, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.57 (1.25×) | **paged** 5.97 (1.03×) | **fasttree** 7.80 (1.18×) | **fasttree** 12.74 (1.29×) |
| K=16 | **paged** 5.98 (1.09×) | **fasttree** 7.45 (1.15×) | **fasttree** 10.96 (1.21×) | **fasttree** 19.58 (1.17×) |
| K=32 | **bs_kernel** 6.72 (1.05×) | **bs_kernel** 10.79 (1.32×) | **bs_kernel** 18.68 (1.35×) | **bs_kernel** 36.08 (1.29×) |
| K=64 | **fasttree** 7.68 (1.03×) | **bs_kernel** 14.69 (1.26×) | **bs_kernel** 26.05 (1.35×) | **bs_kernel** 49.64 (1.41×) |


## L_p=30000, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.94 (1.23×) | **paged** 7.93 (1.02×) | **fasttree** 11.84 (1.12×) | **fasttree** 20.83 (1.17×) |
| K=16 | **paged** 6.82 (1.07×) | **fasttree** 11.57 (1.06×) | **fasttree** 18.84 (1.09×) | **fasttree** 34.17 (1.11×) |
| K=32 | **bs_kernel** 7.94 (1.05×) | **bs_kernel** 16.10 (1.21×) | **bs_kernel** 29.88 (1.22×) | **bs_kernel** 58.10 (1.21×) |
| K=64 | **fasttree** 9.63 (1.00×) | **bs_kernel** 21.94 (1.23×) | **bs_kernel** 41.55 (1.23×) | **bs_kernel** 81.26 (1.24×) |

