#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/diag_xprompt_k128_%j.out
#SBATCH --error=slurm/logs/diag_xprompt_k128_%j.err

# Why did the K=128 sys=20k smoke pick SHARED_2L_DEC_TAIL instead of
# XPROMPT_DEC_TAIL? Load the REAL calibrated 1B coefficients and dump the
# picker's candidate costs for the realistic shape with cross_prompt_lca
# forced to 20000 (1250 pages). If it picks XPROMPT here, the real bench's
# lca was 0 (workload). If it picks 2L_dt, the calibrated coeffs flip it.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}

uv run python - <<'PY'
import torch
from beam_engine.methods.bs_kernel.cost_model import (
    WorkloadShape, pick_strategy_batch, Strategy)
from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
dev = torch.device("cuda")
c = load_or_defaults(dev, "meta-llama/Llama-3.2-1B-Instruct", tp_size=1)
print("coeffs: B_hbm=%.4g num_sms=%s cta_tile_kv=%s merge_launch_us=%s decode_launch_us=%s"
      % (c.B_hbm, c.num_sms, c.cta_tile_kv, getattr(c,'merge_launch_us','?'),
         getattr(c,'decode_launch_us','?')))
bpkv = 2*8*64*2
def mk(Lp,K,suf):
    return WorkloadShape(K=K, L_p=Lp, suffix_lens=[suf]*K, num_kv_heads=8,
        head_dim=64, bytes_per_kv=bpkv, num_qo_heads=32, kv_dtype="bf16")
for K in (16, 128):
  for B in (8,):
    for suf in (1, 32, 64):
      w=mk(40000 if K==128 else 40000, K, suf)
      p=pick_strategy_batch([w]*B, c, cross_prompt_lca=20000)
      d=p.debug
      xp=min((v for k,v in d.items() if k.startswith("xprompt")),default=None)
      dt2=min((v for k,v in d.items() if k.startswith("shared_d2_dec_tail")),default=None)
      pb=d.get("per_beam")
      print(f"\nK={K} B={B} L_p=40000 suffix={suf} lca=20000 -> PICK={p.strategy.name}")
      print(f"   xprompt={xp:.1f}  2l_dt={dt2:.1f}  per_beam={pb:.1f}")
      # full debug for K=128 suf=32
      if K==128 and suf==32:
          for k in sorted(d): print(f"      {k:32s} {d[k]:.1f}")
PY
echo "diag done"
