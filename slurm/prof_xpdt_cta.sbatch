#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=prof_xp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/prof_xpdt_cta_%j.out
#SBATCH --error=slurm/logs/prof_xpdt_cta_%j.err

# Post-fix: is XPROMPT (3l_dt, cross-prompt L0 sys + L1 bundle, fused) using the
# right CTA_TILE_Q on its levels? Capture kernel names (2nd KernelTraits arg =
# CTA_TILE_Q) + grid + per-kernel time for bsk_xp_dt and bsk_2l_dt (fixed ref).
# 1B FA2, K=16 B=16 sys=20k, max-new=4.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/prof_xpdt_cta-${TS}"; mkdir -p "$RD"; LOG="$RD/bench.log"
NCU="ncu --csv --metrics launch__grid_size,launch__block_size,gpu__time_duration.sum --target-processes all --launch-count 1400"
COMMON="--no-stage2 --tree-root --max-new 4 --repeat 1 --scenarios multi_few_shot --k-override 16 --b-override 16"
echo "=== XPROMPT CTA_Q profile (post-fix, 1B FA2) $(date -Iseconds) ===" | tee "$LOG"

for M in bsk_xp_dt bsk_2l_dt; do
  echo "===== $M =====" | tee -a "$LOG"
  env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 \
  $NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
      --methods $M $COMMON > "$RD/$M.csv" 2> "$RD/$M.prof.log" || echo "$M ncu rc=$?" | tee -a "$LOG"
done

python3 - "$RD" <<'PY' 2>&1 | tee -a "$LOG"
import csv, os, sys
from collections import defaultdict
RD=sys.argv[1]
def load(m):
    p=os.path.join(RD,f"{m}.csv")
    if not os.path.exists(p): return None
    lines=open(p,errors="replace").read().splitlines()
    hi=next((i for i,l in enumerate(lines) if l.startswith('"ID"') or '"Kernel Name"' in l),None)
    if hi is None: return None
    rows=list(csv.DictReader(lines[hi:]))
    agg=defaultdict(lambda:{"cnt":0,"ns":0.0,"grid":set()})
    U={"":1,"usecond":1e3,"msecond":1e6,"second":1e9,"nsecond":1}
    for r in rows:
        name=r.get("Kernel Name") or "?"; mn=r.get("Metric Name","")
        mv=r.get("Metric Value","").replace(",",""); mu=r.get("Metric Unit","")
        if mn=="gpu__time_duration.sum":
            try: agg[name]["ns"]+=float(mv)*U.get(mu,1); agg[name]["cnt"]+=1
            except: pass
        if mn=="launch__grid_size":
            try: agg[name]["grid"].add(int(float(mv)))
            except: pass
    return agg
def cta(name):
    # 2nd KernelTraits template arg = CTA_TILE_Q
    import re
    m=re.search(r"KernelTraits<\(flashinfer::MaskMode\)\d+,\s*\(unsigned int\)(\d+)", name)
    return m.group(1) if m else "?"
for m in ("bsk_xp_dt","bsk_2l_dt"):
    agg=load(m)
    if not agg: print(f"\n{m}: no data"); continue
    print(f"\n=== {m}: attention kernels (CTA_TILE_Q, grid, time) ===")
    for name,d in sorted(agg.items(), key=lambda kv:-kv[1]["ns"]):
        if any(s in name for s in ("Prefill","Decode","Cascade")):
            print(f"  {d['ns']/1e6:8.2f} ms  x{d['cnt']:3d}  CTA_TILE_Q={cta(name):>3}  grid={sorted(d['grid'])[:5]}  {name[:54]}")
PY
echo "Done. $RD" | tee -a "$LOG"
