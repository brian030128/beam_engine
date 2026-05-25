#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/prof_xprompt_3l_dram_%j.out
#SBATCH --error=slurm/logs/prof_xprompt_3l_dram_%j.err

# VERIFY the shared read with ncu DRAM-read profiling. The multilevel prefill
# + model weights are IDENTICAL for the 2L and the cross-prompt-3L runs (only
# the decode backend differs), so total(2L) - total(3L) DRAM read isolates the
# decode sys-read difference. If cross-prompt 3L reads sys ONCE (vs B× in 2L),
# 3L total should be ~ (B-1) * sys_KV_per_layer * n_layers * n_decode_steps
# LOWER. If equal -> the kernel re-reads sys per query-tile and the structure
# isn't actually saving reads.
#   sys=8192 (16 MB/layer KV), prompt=2560, B=8, K=32, max_new=5 (4 decode steps)

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
RD="benchmarks/bs_kernel/results/prof_xprompt_3l_dram-${TS}"
mkdir -p "$RD"
LOG="$RD/run.log"
NCU="ncu --csv --metrics dram__bytes_read.sum,dram__bytes_write.sum --target-processes all --launch-count 1200"
COMMON="--scenarios multi_few_shot --tree-root --k-override 32 --b-override 8 --max-new 5 --repeat 1"

echo "=== profiling 2L (bsk_2l_dt) ===" | tee "$LOG"
$NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bsk_2l_dt > "$RD/2l.csv" 2> "$RD/2l.prof.log" || echo "2L ncu rc=$?" | tee -a "$LOG"

echo "=== profiling cross-prompt 3L (bs_kernel + knob) ===" | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 $NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bs_kernel > "$RD/3l.csv" 2> "$RD/3l.prof.log" || echo "3L ncu rc=$?" | tee -a "$LOG"

echo "=== analysis ===" | tee -a "$LOG"
python3 - "$RD" <<'PY' 2>&1 | tee -a "$LOG"
import csv, os, sys, re
RD = sys.argv[1]
U = {"byte":1,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9}
def load(path):
    if not os.path.exists(path): return None
    # ncu --csv emits some preamble lines; find the header row.
    lines = open(path, errors="replace").read().splitlines()
    hi = next((i for i,l in enumerate(lines) if l.startswith('"ID"') or '"Kernel Name"' in l), None)
    if hi is None: return None
    rdr = csv.DictReader(lines[hi:])
    perk = {}; total = 0.0
    for r in rdr:
        name = r.get("Kernel Name") or r.get("Function Name") or "?"
        mn = r.get("Metric Name",""); mu = r.get("Metric Unit",""); mv = r.get("Metric Value","")
        if mn != "dram__bytes_read.sum": continue
        try: v = float(mv.replace(",","")) * U.get(mu,1)
        except: continue
        perk[name] = perk.get(name,0.0)+v; total += v
    return total, perk
for tag,fn in [("2L","2l.csv"),("3L","3l.csv")]:
    r = load(os.path.join(RD,fn))
    if r is None: print(f"{tag}: no data"); continue
    total, perk = r
    print(f"\n{tag} total DRAM read = {total/1e9:.3f} GB  (across all profiled kernels)")
    for n,v in sorted(perk.items(), key=lambda x:-x[1])[:8]:
        print(f"   {v/1e9:8.3f} GB  {n[:80]}")
PY
echo "Done. $RD" | tee -a "$LOG"
