#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_70b_draws
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/f4_70b_draws_%j.out
#SBATCH --error=slurm/logs/f4_70b_draws_%j.err

# F4 self_consistency error bar — Llama-3-70B-Instruct (bf16/bf16, TP=4),
# L_p=64K K=16 B=1 mn=12000. THREE data draws (seeds 0,1,2) -> n=3 per method
# -> mean + half-range error bar. F4_REPEAT=1 (one measured iter per draw;
# long-decode timing is content-near-independent so draws are the replication).
#
# deft is DROPPED from the draw set: it is ~83 min/run (5e6 ms, a 12x
# catastrophic-baseline loser) -> 3 draws would add ~5.5 h for an error bar on
# a number no one disputes. Its original n=1 point stays in the (untouched)
# .baselines.csv. All 6 retained methods all run in ONE process per the lib, so
# the 140 GiB model loads ONCE for all 3 draws.
#
# CAVEAT: the lib recalibrates (calibrate --force) with the CURRENT B_hbm
# method (attn-probe, ~2.4x the old memcpy value) — the same change implicated
# in the 8B F4 picker regression. The forced strategies (1p/dt) and the
# baselines are calibration-independent, so their bars are clean; the bs_kernel
# (picker) median may differ from the paper's 379352. Step-timing dump is on so
# the 70B flip behaviour can be checked tomorrow.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"; trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_WORKSPACE_MB=2048   # 64K single-launch prefill scratch
export BE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

MODEL_TAG="Llama-3-70B-bf16"
export BE_DUMP_STEP_TIMINGS="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}/step_timings_mn12000_draws"
mkdir -p "$BE_DUMP_STEP_TIMINGS"

export F4_SEEDS="0,1,2"
export F4_REPEAT=1
export F4_LP=65536

NPROC=4
SC_CORE="paged fasttree mlca bs_kernel bsk_2l_1p_single bsk_2l_dt"
SC_EXTRA=""
source scripts/paper-exp/final_f4_lib.sh
