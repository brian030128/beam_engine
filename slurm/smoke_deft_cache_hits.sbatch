#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=deft_cache_hits
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/deft_cache_hits_%j.out
#SBATCH --error=slurm/logs/deft_cache_hits_%j.err

# Measures the DeftBackend plan-cache hit rate. Runs three cells of
# different (K, B, max_new) and prints the post-run hits/misses count
# that DeftBackend tracks on the backend object.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python -c "
import torch
from transformers import AutoTokenizer
from beam_engine.baselines.deft import DeftBackend
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.page_driver import beam_search

MODEL = 'meta-llama/Llama-3.2-1B'
tok = AutoTokenizer.from_pretrained(MODEL)
model = LlamaForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device='cuda')
config = model.config

def run(K, L_p, B, max_new, tag):
    prompt = tok.encode('The quick brown fox ' * (L_p // 4))[:L_p]
    prompts = [prompt] * B
    backend = DeftBackend()
    backend._plan_hits = 0
    backend._plan_misses = 0
    # warmup
    beam_search(model, config, prompts, max_new, K,
                backend=backend, max_num_pages=8192, return_timings=False)
    backend._plan_hits = 0
    backend._plan_misses = 0
    # measured
    _, timings = beam_search(model, config, prompts, max_new, K,
                              backend=backend, max_num_pages=8192,
                              return_timings=True)
    h, m = backend._plan_hits, backend._plan_misses
    rate = h / max(1, h + m) * 100
    n_steps = len(timings['decode_step_ms'])
    print(f'  {tag:<32} hits={h:5d} misses={m:5d} hit_rate={rate:5.1f}% steps={n_steps}')

print('cell                              hits/misses')
run(K=4,  L_p=128,  B=2, max_new=64,  tag='K=4  L_p=128  B=2  mn=64')
run(K=32, L_p=1024, B=4, max_new=128, tag='K=32 L_p=1024 B=4  mn=128')
run(K=64, L_p=512,  B=2, max_new=64,  tag='K=64 L_p=512  B=2  mn=64')
"
