#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=tp_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/tp_smoke_%j.out
#SBATCH --error=slurm/logs/tp_smoke_%j.err

set -euo pipefail

# Worktree path (multi-gpu branch). The worktree has no .venv of its own,
# so we activate the parent repo's venv and run with the worktree's src on
# PYTHONPATH so beam_engine.distributed (added in this worktree) resolves.
WORKTREE="${BEAM_ENGINE_WORKTREE:-/home/u4320956/beam_engine-v3/.claude/worktrees/multi-gpu}"
REPO="/home/u4320956/beam_engine-v3"

cd "$WORKTREE"
source /etc/profile.d/lmod.sh
module load cuda/12.4
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$WORKTREE/src:${PYTHONPATH:-}"

export BE_SMOKE_BACKENDS="${BE_SMOKE_BACKENDS:-paged,adaptive_pool,bs_kernel}"

echo "============================================================"
echo "TP=1 sanity (single-process)  backends=$BE_SMOKE_BACKENDS"
echo "============================================================"
python tests/test_tp_smoke.py

echo
echo "============================================================"
echo "TP=2 cross-rank smoke  backends=$BE_SMOKE_BACKENDS"
echo "============================================================"
torchrun --standalone --nproc_per_node=2 tests/test_tp_smoke.py
