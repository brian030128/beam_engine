#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_20k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/bench_xprompt_3l_20k_%j.out
#SBATCH --error=slurm/logs/bench_xprompt_3l_20k_%j.err

# User-requested cell: B=16 prompts x K=16 beams = 256 queries, sys=20k,
# prompt(fewshot)=20k. Per-layer sys KV = 20000 tok * 2 KB = 40 MB (still
# < 50 MB H100 L2, crossover ~= 25k tok), but sys+bundle co-resident = 80 MB
# per group > L2, so 2L's per-group shared-prefix pass may now evict sys
# between groups -> first config where cross-prompt sys-dedup *could* cut DRAM.
#
# Two measurements:
#   (1) timing: cross-prompt 3L vs good-2L baseline (decode ms, fwd/step).
#   (2) ncu DRAM-read: total(2L)-total(3L) isolates the decode sys-read delta
#       (prefill + weights identical across the two runs). max-new=5 so ncu
#       replay stays tractable. all-kernel capture (NOT -k regex:KVCache, which
#       drops the FusedMultiLevelCascade num_levels=3 kernel by name).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BE_SYS_LEN=20000
export BE_FEWSHOT_LEN=20000

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/xprompt_3l_20k-${TS}"
mkdir -p "$RD"
LOG="$RD/bench.log"
echo "=== xprompt 3L 20k-sys + 20k-prompt B=16 K=16 $(date -Iseconds) ; $BE_MODEL ===" | tee "$LOG"
echo "per-layer sys KV = 40 MB (L2=50 MB); sys+bundle/group = 80 MB > L2" | tee -a "$LOG"

COMMON="--scenarios multi_few_shot --tree-root --k-override 16 --b-override 16"

# ---- (1) timing ----
echo "" | tee -a "$LOG"
echo "===== (1) timing: good-2L baseline (bsk_2l_dt) =====" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bsk_2l_dt --max-new 128 --repeat 2 --warmup \
    --out "$RD/2l_timing.csv" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== (1) timing: cross-prompt 3L (bs_kernel + CROSS_PROMPT_3L) =====" | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_TRACE_PLAN=1 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bs_kernel --max-new 128 --repeat 2 --warmup \
    --out "$RD/3l_timing.csv" 2>&1 | tee -a "$LOG"

# ---- (2) ncu DRAM-read ----
NCU="ncu --csv --metrics dram__bytes_read.sum,dram__bytes_write.sum --target-processes all --launch-count 2000"
PROF="$COMMON --max-new 5 --repeat 1"

echo "" | tee -a "$LOG"
echo "===== (2) DRAM profile: 2L (bsk_2l_dt) =====" | tee -a "$LOG"
$NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $PROF --methods bsk_2l_dt > "$RD/2l.csv" 2> "$RD/2l.prof.log" \
    || echo "2L ncu rc=$?" | tee -a "$LOG"

echo "===== (2) DRAM profile: cross-prompt 3L (bs_kernel + knob) =====" | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 $NCU uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $PROF --methods bs_kernel > "$RD/3l.csv" 2> "$RD/3l.prof.log" \
    || echo "3L ncu rc=$?" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== DRAM analysis (total + top kernels) =====" | tee -a "$LOG"
python3 - "$RD" <<'PY' 2>&1 | tee -a "$LOG"
import csv, os, sys
RD = sys.argv[1]
U = {"byte":1,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9}
def load(path):
    if not os.path.exists(path): return None
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
res = {}
for tag,fn in [("2L","2l.csv"),("3L","3l.csv")]:
    r = load(os.path.join(RD,fn))
    if r is None: print(f"{tag}: no data"); continue
    total, perk = r; res[tag]=(total,perk)
    print(f"\n{tag} total DRAM read = {total/1e9:.3f} GB  (all profiled kernels)")
    for n,v in sorted(perk.items(), key=lambda x:-x[1])[:8]:
        print(f"   {v/1e9:8.3f} GB  {n[:80]}")
if "2L" in res and "3L" in res:
    d = (res["2L"][0]-res["3L"][0])/1e9
    print(f"\n2L - 3L total DRAM read = {d:+.3f} GB  "
          f"(positive => 3L reads less, i.e. sys-dedup materialized)")
PY
echo "Done. $RD" | tee -a "$LOG"
