#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_dec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/prof_xprompt_3l_decode_%j.out
#SBATCH --error=slurm/logs/prof_xprompt_3l_decode_%j.err

# ISOLATE decode attention DRAM reads (prefill is backend-independent and was
# masking the signal in the earlier total). Profile ONLY attention kernels
# (-k regex:KVCache). prefill-only baseline = max_new=1 (0 decode steps);
# max_new=5 = prefill + 4 decode steps. decode_attn = mn5 - mn1.
#   decode_2L      = 2l_mn5  - 2l_mn1
#   decode_3L@128  = 3l128_mn5 - 2l_mn1   (prefill identical across backends)
#   decode_3L@auto = 3lauto_mn5 - 2l_mn1
# If the forced large CTA_TILE_Q is the fix, decode_3L@128 << decode_2L while
# decode_3L@auto ~= decode_2L.
#   sys=8192 (16 MB/layer KV), prompt=2560, B=8, K=32

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
RD="benchmarks/bs_kernel/results/prof_xprompt_3l_decode-${TS}"; mkdir -p "$RD"
LOG="$RD/run.log"
NCU="ncu --csv --metrics dram__bytes_read.sum --target-processes all -k regex:KVCache --launch-count 2000"
C="--scenarios multi_few_shot --tree-root --k-override 32 --b-override 8 --repeat 1"

run() { # $1=tag $2=extra-env $3=method $4=max_new
    echo "=== $1 ===" | tee -a "$LOG"
    env $2 $NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        $C --methods $3 --max-new $4 > "$RD/$1.csv" 2> "$RD/$1.prof.log" || echo "$1 rc=$?" | tee -a "$LOG"
}

run 2l_mn1     ""                          bsk_2l_dt 1
run 2l_mn5     ""                          bsk_2l_dt 5
run 3l128_mn5  "BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_XP_TILE_Q=128" bs_kernel 5
run 3lauto_mn5 "BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_XP_TILE_Q=0"   bs_kernel 5

echo "=== analysis (attention-only DRAM reads, GB) ===" | tee -a "$LOG"
python3 - "$RD" <<'PY' 2>&1 | tee -a "$LOG"
import csv, os, sys
RD=sys.argv[1]; U={"byte":1,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9}
def total(fn):
    p=os.path.join(RD,fn)
    if not os.path.exists(p): return None
    lines=open(p,errors="replace").read().splitlines()
    hi=next((i for i,l in enumerate(lines) if '"Kernel Name"' in l),None)
    if hi is None: return None
    t=0.0
    for r in csv.DictReader(lines[hi:]):
        if r.get("Metric Name")!="dram__bytes_read.sum": continue
        try: t+=float(r["Metric Value"].replace(",",""))*U.get(r.get("Metric Unit",""),1)
        except: pass
    return t
v={k:total(k+".csv") for k in ["2l_mn1","2l_mn5","3l128_mn5","3lauto_mn5"]}
for k,x in v.items(): print(f"  {k:12s} attn DRAM = {('%.3f GB'%(x/1e9)) if x is not None else 'NA'}")
base=v["2l_mn1"]
if base is not None:
    def dec(x): return None if (x is None) else (x-base)/1e9
    print(f"\n  decode_2L      = {dec(v['2l_mn5']):.3f} GB")
    print(f"  decode_3L@128  = {dec(v['3l128_mn5']):.3f} GB")
    print(f"  decode_3L@auto = {dec(v['3lauto_mn5']):.3f} GB")
PY
echo "Done. $RD" | tee -a "$LOG"
