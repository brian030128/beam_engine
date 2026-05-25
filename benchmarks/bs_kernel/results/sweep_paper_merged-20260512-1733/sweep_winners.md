# Fastest method per cell (with speedup over runner-up)

Model: meta-llama/Llama-3.2-1B (fp16). H100 80GB. ms/step.


## L_p=8192, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 5.09 (1.14×) | **paged** 4.73 (1.32×) | **paged** 5.01 (1.27×) | **paged** 6.76 (1.10×) | **paged** 11.08 (1.01×) |
| K=16 | **paged** 5.04 (1.23×) | **paged** 6.72 (1.06×) | **fasttree** 8.75 (1.24×) | **fasttree** 12.75 (1.47×) | **fasttree** 23.98 (1.49×) |
| K=32 | **paged** 5.36 (1.23×) | **bs_kernel** 8.50 (1.13×) | **bs_kernel** 12.89 (1.14×) | **bs_kernel** 23.37 (1.15×) | **bs_kernel** 46.45 (1.18×) |
| K=64 | **bs_kernel** 6.94 (1.00×) | **bs_kernel** 10.12 (1.27×) | **bs_kernel** 18.05 (1.31×) | **bs_kernel** 33.95 (1.40×) | **bs_kernel** 70.26 (1.37×) |


## L_p=8192, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 4.98 (1.15×) | **paged** 6.84 (1.21×) | **paged** 9.26 (1.14×) | **paged** 15.32 (1.03×) | **fasttree** 27.88 (1.01×) |
| K=16 | **paged** 5.75 (1.23×) | **paged** 10.37 (1.09×) | **fasttree** 16.50 (1.10×) | **fasttree** 28.17 (1.21×) | **fasttree** 54.82 (1.19×) |
| K=32 | **paged** 6.63 (1.20×) | **bs_kernel** 14.77 (1.04×) | **bs_kernel** 25.25 (1.02×) | **bs_kernel** 47.84 (1.04×) | **bs_kernel** 94.35 (1.08×) |
| K=64 | **paged** 8.73 (1.02×) | **bs_kernel** 18.35 (1.11×) | **bs_kernel** 33.87 (1.16×) | **bs_kernel** 67.91 (1.16×) | **bs_kernel** 137.61 (1.16×) |


## L_p=30000, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.57 (1.25×) | **paged** 5.97 (1.03×) | **fasttree** 7.80 (1.18×) | **fasttree** 12.74 (1.29×) |
| K=16 | **paged** 5.98 (1.06×) | **fasttree** 7.45 (1.37×) | **fasttree** 10.96 (1.79×) | **fasttree** 19.58 (1.89×) |
| K=32 | **bs_kernel** 6.60 (1.07×) | **bs_kernel** 12.21 (1.17×) | **bs_kernel** 22.24 (1.14×) | **bs_kernel** 43.33 (1.07×) |
| K=64 | **fasttree** 7.68 (1.15×) | **bs_kernel** 15.23 (1.21×) | **bs_kernel** 27.36 (1.28×) | **bs_kernel** 53.98 (1.30×) |


## L_p=30000, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.94 (1.23×) | **paged** 7.93 (1.02×) | **fasttree** 11.84 (1.12×) | **fasttree** 20.83 (1.17×) |
| K=16 | **paged** 6.82 (1.06×) | **fasttree** 11.57 (1.17×) | **fasttree** 18.84 (1.42×) | **fasttree** 34.17 (1.55×) |
| K=32 | **bs_kernel** 7.94 (1.06×) | **bs_kernel** 18.12 (1.08×) | **bs_kernel** 34.12 (1.07×) | **bs_kernel** 66.65 (1.05×) |
| K=64 | **fasttree** 9.63 (1.12×) | **bs_kernel** 22.39 (1.20×) | **bs_kernel** 42.75 (1.20×) | **bs_kernel** 85.69 (1.18×) |

