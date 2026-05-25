# Fastest method per cell (with speedup over runner-up)

Model: meta-llama/Llama-3.2-1B (fp16). H100 80GB. ms/step.


## L_p=8192, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 4.94 (1.14×) | **paged** 4.69 (1.32×) | **paged** 4.96 (1.27×) | **paged** 6.79 (1.09×) | **paged** 11.14 (1.01×) |
| K=16 | **paged** 4.94 (1.24×) | **paged** 6.81 (1.03×) | **fasttree** 8.68 (1.12×) | **fasttree** 12.74 (1.13×) | **fasttree** 24.50 (1.04×) |
| K=32 | **paged** 5.33 (1.22×) | **bs_kernel** 8.19 (1.12×) | **bs_kernel** 11.82 (1.26×) | **bs_kernel** 20.81 (1.32×) | **bs_kernel** 39.17 (1.42×) |
| K=64 | **bs_kernel** 6.82 (1.02×) | **bs_kernel** 9.89 (1.30×) | **bs_kernel** 17.35 (1.39×) | **bs_kernel** 30.86 (1.61×) | **bs_kernel** 64.17 (1.56×) |


## L_p=8192, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| K=4 | **paged** 4.92 (1.15×) | **paged** 6.90 (1.20×) | **paged** 9.34 (1.13×) | **paged** 15.44 (1.03×) | **fasttree** 28.36 (1.01×) |
| K=16 | **paged** 5.69 (1.24×) | **paged** 10.47 (1.04×) | **fasttree** 16.91 (1.00×) | **fasttree** 28.48 (1.06×) | **fasttree** 55.41 (1.00×) |
| K=32 | **paged** 6.64 (1.17×) | **bs_kernel** 13.85 (1.12×) | **bs_kernel** 23.64 (1.10×) | **bs_kernel** 44.33 (1.14×) | **bs_kernel** 86.53 (1.21×) |
| K=64 | **bs_kernel** 8.69 (1.01×) | **bs_kernel** 17.62 (1.16×) | **bs_kernel** 32.20 (1.23×) | **bs_kernel** 63.30 (1.29×) | **bs_kernel** 127.13 (1.30×) |


## L_p=30000, mode=std

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 4.67 (1.26×) | **paged** 6.01 (1.05×) | **fasttree** 7.83 (1.18×) | **fasttree** 12.79 (1.29×) |
| K=16 | **paged** 6.04 (1.04×) | **fasttree** 7.60 (1.06×) | **fasttree** 11.55 (1.07×) | **fasttree** 19.97 (1.08×) |
| K=32 | **bs_kernel** 6.47 (1.12×) | **bs_kernel** 10.57 (1.32×) | **bs_kernel** 18.50 (1.36×) | **bs_kernel** 35.09 (1.37×) |
| K=64 | **bs_kernel** 7.71 (1.02×) | **bs_kernel** 14.52 (1.29×) | **bs_kernel** 25.70 (1.40×) | **bs_kernel** 49.18 (1.47×) |


## L_p=30000, mode=dbs

| K\B | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| K=4 | **paged** 5.04 (1.22×) | **paged** 8.02 (1.03×) | **fasttree** 12.00 (1.12×) | **fasttree** 21.10 (1.19×) |
| K=16 | **paged** 6.91 (1.02×) | **fasttree** 11.49 (1.04×) | **fasttree** 18.80 (1.05×) | **fasttree** 35.44 (1.04×) |
| K=32 | **bs_kernel** 7.71 (1.11×) | **bs_kernel** 16.12 (1.23×) | **bs_kernel** 29.40 (1.26×) | **bs_kernel** 57.29 (1.26×) |
| K=64 | **bs_kernel** 9.43 (1.04×) | **bs_kernel** 21.72 (1.26×) | **bs_kernel** 41.30 (1.26×) | **bs_kernel** 81.28 (1.27×) |

