# Fastest method per cell (with speedup over runner-up)

Model: meta-llama/Llama-3.2-1B (fp16). H100 80GB. ms/step.

## L_p=8192, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 5.09 (1.04×) | **paged** 4.73 (1.32×) | **paged** 5.01 (1.27×) | **paged** 6.76 (1.10×) | **paged** 11.08 (1.01×) |
| K=16 | **paged** 5.04 (1.23×) | **paged** 6.72 (1.06×) | **fasttree** 8.75 (1.22×) | **fasttree** 12.75 (1.47×) | **fasttree** 23.98 (1.46×) |
| K=32 | **paged** 5.36 (1.21×) | **bs_kernel** 8.36 (1.15×) | **bs_kernel** 12.73 (1.15×) | **bs_kernel** 23.29 (1.15×) | **bs_kernel** 45.03 (1.21×) |
| K=64 | **bs_kernel** 6.70 (1.04×) | **bs_kernel** 9.96 (1.29×) | **bs_kernel** 17.07 (1.39×) | **bs_kernel** 32.62 (1.45×) | **bs_kernel** 65.63 (1.46×) |

## L_p=8192, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 4.98 (1.13×) | **paged** 6.84 (1.21×) | **paged** 9.26 (1.14×) | **paged** 15.32 (1.03×) | **fasttree** 27.88 (1.01×) |
| K=16 | **paged** 5.75 (1.23×) | **paged** 10.37 (1.09×) | **fasttree** 16.50 (1.10×) | **fasttree** 28.17 (1.20×) | **fasttree** 54.82 (1.19×) |
| K=32 | **paged** 6.63 (1.18×) | **bs_kernel** 13.90 (1.11×) | **bs_kernel** 24.55 (1.05×) | **bs_kernel** 46.09 (1.08×) | **bs_kernel** 91.02 (1.12×) |
| K=64 | **bs_kernel** 8.62 (1.01×) | **bs_kernel** 18.10 (1.12×) | **bs_kernel** 33.12 (1.18×) | **bs_kernel** 64.38 (1.22×) | **bs_kernel** 129.21 (1.24×) |

## L_p=30000, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.57 (1.25×) | **paged** 5.97 (1.03×) | **fasttree** 7.80 (1.18×) | **fasttree** 12.74 (1.29×) |
| K=16 | **paged** 5.98 (1.02×) | **fasttree** 7.45 (1.36×) | **fasttree** 10.96 (1.72×) | **fasttree** 19.58 (1.86×) |
| K=32 | **bs_kernel** 6.35 (1.11×) | **bs_kernel** 12.02 (1.18×) | **bs_kernel** 21.41 (1.18×) | **bs_kernel** 42.32 (1.10×) |
| K=64 | **fasttree** 7.68 (1.12×) | **bs_kernel** 14.83 (1.25×) | **bs_kernel** 26.38 (1.33×) | **bs_kernel** 52.35 (1.34×) |

## L_p=30000, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.94 (1.22×) | **paged** 7.93 (1.02×) | **fasttree** 11.84 (1.12×) | **fasttree** 20.83 (1.17×) |
| K=16 | **paged** 6.82 (1.02×) | **fasttree** 11.57 (1.15×) | **fasttree** 18.84 (1.40×) | **fasttree** 34.17 (1.51×) |
| K=32 | **bs_kernel** 7.66 (1.09×) | **bs_kernel** 18.14 (1.08×) | **bs_kernel** 33.17 (1.10×) | **bs_kernel** 65.48 (1.07×) |
| K=64 | **fasttree** 9.63 (1.10×) | **bs_kernel** 22.33 (1.20×) | **bs_kernel** 41.82 (1.22×) | **bs_kernel** 82.29 (1.23×) |

