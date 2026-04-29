#!/usr/bin/env bash
# Usage: ./test.sh <test_file> [gpu_ids]
# Activates the flashtree conda env, syncs uv deps, and runs the given test
# under uv. Mirrors the old beam_engine harness.
set -euo pipefail

TEST_FILE="${1:-tests/test_baselines.py}"
GPU_IDS="${2:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flashtree

cd "$(dirname "$0")"
uv sync

CUDA_VISIBLE_DEVICES="$GPU_IDS" uv run python "$TEST_FILE"
