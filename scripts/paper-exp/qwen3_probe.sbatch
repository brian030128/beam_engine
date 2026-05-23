#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=qwen3_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/qwen3_probe_%j.out
#SBATCH --error=slurm/logs/qwen3_probe_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

uv run python - <<'PY'
import torch
from transformers import AutoTokenizer
from beam_engine.models import load_model_for_causal_lm
from beam_engine.baselines.paged import PagedBackend
from beam_engine.page_driver import beam_search

DEVICE = "cuda"
DTYPE = torch.float16
torch.set_printoptions(precision=4, sci_mode=False)

print("[probe] loading Qwen/Qwen3-4B ...", flush=True)
model = load_model_for_causal_lm("Qwen/Qwen3-4B", dtype=DTYPE, device=DEVICE)
print(f"[probe] loaded: {type(model).__name__}", flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
prompts = [
    tok("The capital of France is", add_special_tokens=False).input_ids,
    tok("Quantum mechanics describes", add_special_tokens=False).input_ids,
]
print(f"[probe] prompt lengths: {[len(p) for p in prompts]}", flush=True)

backend = PagedBackend()
print("[probe] calling beam_search ...", flush=True)
out = beam_search(
    model=model,
    config=model.config,
    prompt_ids=prompts,
    max_new_tokens=4,
    beam_width=2,
    backend=backend,
)
print("[probe] OK:", out, flush=True)
PY
