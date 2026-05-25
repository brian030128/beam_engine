#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=70b_fp8_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/70b_fp8_smoke_%j.out
#SBATCH --error=slurm/logs/70b_fp8_smoke_%j.err

# Smoke test: load RedHatAI/Meta-Llama-3-70B-Instruct-FP8 across 2x H100
# (TP=2), allocate fp8 KV cache pages, and run a short beam-search
# through the `paged` and `bs_kernel` backends. The model name lives
# only as a --model CLI arg below; nothing on the login node ever
# resolves it.
#
# Node-local ephemeral HF cache: each job redownloads ~70 GB of fp8
# weights into $SLURM_TMPDIR and the trap below wipes them at job
# end. No bytes persist into /work/u4320956.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# Ephemeral HF cache on the compute node's local NVMe. Use SLURM_TMPDIR
# when set, otherwise /tmp; either way, scoped to this job's id.
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

# HF_TOKEN must be in the user's environment if the upstream
# meta-llama gated config files are accessed; RedHatAI's FP8 repo is
# ungated but it still pulls a base config.
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== fp8 loader unit test (synthetic fixture, no model) ==="
# pytest isn't in the project venv; the test module ships an
# __main__ block that runs all four assertions and prints "OK".
uv run python tests/test_fp8_load.py

echo
echo "=== TP=2 smoke on 70B fp8 weights + bf16 KV (paged, bs_kernel) ==="
# NOTE: fp8 KV cache is deferred to a follow-up PR. FlashInfer's
# append_paged_kv_cache currently can't dispatch fp8 / uint8 cache
# dtypes through tvm_ffi (dlpack has no fp8 entry; uint8 isn't in the
# kernel's dispatch table). Once that lands, swap --kv-dtype bf16 →
# fp8_e4m3 to enable the full pipeline.
uv run torchrun --nproc_per_node=2 scripts/run_tp_beam_search.py \
    --model RedHatAI/Meta-Llama-3-70B-Instruct-FP8 \
    --dtype bf16 \
    --kv-dtype bf16 \
    --methods paged bs_kernel \
    --num-prompts 2 \
    --beam-width 4 \
    --max-new-tokens 16 \
    --prompt-len 256

echo
echo "=== smoke complete ==="
