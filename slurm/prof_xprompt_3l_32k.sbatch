#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_32k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/prof_xprompt_3l_32k_%j.out
#SBATCH --error=slurm/logs/prof_xprompt_3l_32k_%j.err

# sys=32768 -> per-layer sys KV = 64 MB > 50 MB L2. Now L2 CANNOT absorb the
# 2L per-group sys reads, so 2L must re-read sys from HBM ~B times/step, while
# cross-prompt 3L reads it once. This is the regime where the shared read
# should finally cut DRAM. Test = (1) timing forward 3L vs 2L, (2) ncu total
# DRAM @ mn3 (prefill+weights identical -> total diff = decode read diff).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BE_SYS_LEN=32768
export BE_FEWSHOT_LEN=2560

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/prof_xprompt_3l_32k-${TS}"; mkdir -p "$RD"
LOG="$RD/run.log"
echo "=== sys=32768 (64 MB/layer > L2) $(date -Iseconds) ===" | tee "$LOG"

echo "--- timing (no ncu), B=8, K=32, mn=64 ---" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --tree-root --methods bsk_2l_dt \
    --k-override 32 --b-override 8 --max-new 64 --repeat 2 --warmup \
    --out "$RD/time_2l.csv" 2>&1 | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_XP_TILE_Q=128 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --tree-root --methods bs_kernel \
    --k-override 32 --b-override 8 --max-new 64 --repeat 2 --warmup \
    --out "$RD/time_3l.csv" 2>&1 | tee -a "$LOG"

echo "--- ncu total DRAM, B=4, mn=3 (2 decode steps) ---" | tee -a "$LOG"
NCU="ncu --csv --metrics dram__bytes_read.sum --target-processes all --launch-count 1500"
C="--scenarios multi_few_shot --tree-root --k-override 32 --b-override 4 --repeat 1 --max-new 3"
$NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py $C --methods bsk_2l_dt \
    > "$RD/2l_mn3.csv" 2> "$RD/2l.prof.log" || echo "2L rc=$?" | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_XP_TILE_Q=128 $NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py $C --methods bs_kernel \
    > "$RD/3l_mn3.csv" 2> "$RD/3l.prof.log" || echo "3L rc=$?" | tee -a "$LOG"

echo "=== analysis ===" | tee -a "$LOG"
python3 - "$RD" <<'PY' 2>&1 | tee -a "$LOG"
import csv, os, sys
RD=sys.argv[1]; U={"byte":1,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9}
def tot(fn):
    p=os.path.join(RD,fn)
    if not os.path.exists(p): return None
    L=open(p,errors="replace").read().splitlines()
    hi=next((i for i,l in enumerate(L) if '"Kernel Name"' in l),None)
    if hi is None: return None
    t=0.0; per={}
    for r in csv.DictReader(L[hi:]):
        if r.get("Metric Name")!="dram__bytes_read.sum": continue
        try: v=float(r["Metric Value"].replace(",",""))*U.get(r.get("Metric Unit",""),1)
        except: continue
        t+=v; k=r["Kernel Name"][:55]; per[k]=per.get(k,0.0)+v
    return t,per
for tag in ["time_2l","time_3l"]:
    p=os.path.join(RD,tag+".csv")
    if os.path.exists(p):
        r=min(csv.DictReader(open(p)),key=lambda x:float(x['decode_total_ms']))
        print(f"  {tag}: fwd_p50={float(r['forward_p50_ms']):.3f}  dec_p50={float(r['decode_p50_ms']):.3f}")
a=tot("2l_mn3.csv"); b=tot("3l_mn3.csv")
if a and b:
    print(f"\n  2L mn3 total DRAM = {a[0]/1e9:.3f} GB")
    print(f"  3L mn3 total DRAM = {b[0]/1e9:.3f} GB")
    print(f"  2L - 3L = {(a[0]-b[0])/1e9:.3f} GB  (= decode read saved by cross-prompt, prefill cancels)")
    print("  -- 2L top kernels --")
    for n,v in sorted(a[1].items(),key=lambda x:-x[1])[:6]: print(f"    {v/1e9:7.3f} GB  {n}")
    print("  -- 3L top kernels --")
    for n,v in sorted(b[1].items(),key=lambda x:-x[1])[:6]: print(f"    {v/1e9:7.3f} GB  {n}")
PY
echo "Done. $RD" | tee -a "$LOG"
