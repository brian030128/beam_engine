#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=dbg2stg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/debug_two_stage_%j.out
#SBATCH --error=slurm/logs/debug_two_stage_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python <<'PY'
import sys, traceback
sys.path.insert(0, 'benchmarks/bs_kernel')
import torch
from transformers import AutoTokenizer
from beam_engine.baselines.fasttree import FastTreeBackend
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.tree_driver import tree_batch_decode
from sglang_workloads import SCENARIO_BUILDERS, build_multi_chain_reasoning_stage2

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", dtype=torch.float16, device="cuda")
config = model.config

print("STAGE 1: build")
spec = SCENARIO_BUILDERS["multi_chain_reasoning"](tok, paper_exact=True)
spec.validate()
print(f"  B={spec.B} K={spec.K} L_shared={len(spec.groups[0].shared_prefix_ids)} L_priv={len(spec.groups[0].private_prefix_ids_per_leaf[0])} n_leaves={spec.n_leaves}")
print("STAGE 1: run")
try:
    backend = FastTreeBackend()
    result = tree_batch_decode(model, config, spec, 16, backend=backend, return_timings=True, return_phase_timings=True)
    print(f"  stage1 OK: decode_total = {sum(result.timings['decode_step_ms']):.1f} ms over {len(result.timings['decode_step_ms'])} steps")
except Exception as e:
    print(f"  STAGE 1 FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    raise

print("STAGE 2: build")
s2 = build_multi_chain_reasoning_stage2(tok, stage1_chain_len=16)
s2.validate()
print(f"  B={s2.B} K={s2.K} L_shared={len(s2.groups[0].shared_prefix_ids)} L_priv={len(s2.groups[0].private_prefix_ids_per_leaf[0])} n_leaves={s2.n_leaves}")
print("STAGE 2: run")
try:
    backend2 = FastTreeBackend()
    result2 = tree_batch_decode(model, config, s2, 16, backend=backend2, return_timings=True, return_phase_timings=True)
    print(f"  stage2 OK: decode_total = {sum(result2.timings['decode_step_ms']):.1f} ms over {len(result2.timings['decode_step_ms'])} steps")
except Exception as e:
    print(f"  STAGE 2 FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    raise

print("DONE")
PY
