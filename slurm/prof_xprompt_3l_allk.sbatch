#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_allk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/prof_xprompt_3l_allk_%j.out
#SBATCH --error=slurm/logs/prof_xprompt_3l_allk_%j.err

# Filter-free DRAM profiling (the -k KVCache filter missed the 3L fused-cascade
# kernel). Capture ALL kernels; isolate decode via mn1 (prefill only) vs mn3
# (prefill + 2 decode steps). decode_X = total(mn3) - total(mn1). Compare 2L vs
# cross-prompt-3L (with force_cta_tile_q=128). Also dump per-kernel top list to
# identify the 3L cascade kernel and its read volume.
#   sys=8192, prompt=2560, B=8, K=32

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BE_SYS_LEN=8192
export BE_FEWSHOT_LEN=2560

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/prof_xprompt_3l_allk-${TS}"; mkdir -p "$RD"
LOG="$RD/run.log"
NCU="ncu --csv --metrics dram__bytes_read.sum --target-processes all --launch-count 1500"
C="--scenarios multi_few_shot --tree-root --k-override 32 --b-override 8 --repeat 1"

run() { echo "=== $1 ===" | tee -a "$LOG"
    env $2 $NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        $C --methods $3 --max-new $4 > "$RD/$1.csv" 2> "$RD/$1.prof.log" || echo "$1 rc=$?" | tee -a "$LOG"; }

run 2l_mn1    ""                                                  bsk_2l_dt 1
run 2l_mn3    ""                                                  bsk_2l_dt 3
run 3l_mn1    "BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_XP_TILE_Q=128" bs_kernel 1
run 3l_mn3    "BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_XP_TILE_Q=128" bs_kernel 3

echo "=== analysis ===" | tee -a "$LOG"
python3 - "$RD" <<'PY' 2>&1 | tee -a "$LOG"
import csv, os, sys
RD=sys.argv[1]; U={"byte":1,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9}
def load(fn):
    p=os.path.join(RD,fn+".csv")
    if not os.path.exists(p): return None,None
    lines=open(p,errors="replace").read().splitlines()
    hi=next((i for i,l in enumerate(lines) if '"Kernel Name"' in l),None)
    if hi is None: return None,None
    tot=0.0; per={}
    for r in csv.DictReader(lines[hi:]):
        if r.get("Metric Name")!="dram__bytes_read.sum": continue
        try: v=float(r["Metric Value"].replace(",",""))*U.get(r.get("Metric Unit",""),1)
        except: continue
        tot+=v; k=r["Kernel Name"][:55]; per[k]=per.get(k,0.0)+v
    return tot,per
T={}
for k in ["2l_mn1","2l_mn3","3l_mn1","3l_mn3"]:
    t,per=load(k); T[k]=t
    print(f"\n{k}: total DRAM read = {('%.3f GB'%(t/1e9)) if t else 'NA'}")
    if per:
        for n,v in sorted(per.items(),key=lambda x:-x[1])[:6]:
            print(f"    {v/1e9:7.3f} GB  {n}")
if all(T.get(k) is not None for k in T):
    d2=(T["2l_mn3"]-T["2l_mn1"])/1e9; d3=(T["3l_mn3"]-T["3l_mn1"])/1e9
    print(f"\n  decode_2L  (2 steps) = {d2:.3f} GB")
    print(f"  decode_3L  (2 steps) = {d3:.3f} GB   [cross-prompt, tile=128]")
    print(f"  ratio 3L/2L = {d3/d2:.3f}" if d2 else "")
PY
echo "Done. $RD" | tee -a "$LOG"
