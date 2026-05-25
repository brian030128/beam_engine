#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=plan_bd_bs8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/plan_bd_bs8b_%j.out
#SBATCH --error=slurm/logs/plan_bd_bs8b_%j.err

# 8B / beam_search plan-time breakdown on the F3 cell:
#   K=64, B=8, L_p=16000, max_new=256, prompts=data/hotpotqa.jsonl.
# Real dynamic beam search (bench_batched.py). Trace flags ON so
# bs_kernel (pick_ms) + fasttree (dispatch_ms) split plan into
# build-indices vs dispatch-decision (new dispatch_total_ms column).
# Each method in its own process to isolate OOM/crash.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export HF_HOME=/work/u4320956/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FT_TRACE_PLAN=1
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL=meta-llama/Llama-3.1-8B-Instruct

OUT_DIR=benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/all_8b_beamsearch
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"
CSV="$OUT_DIR/plan_breakdown.csv"
rm -f "$CSV"

{
    echo "=== 8B/beam_search plan-time breakdown (F3: K=64 B=8 L_p=16000) $(date -Iseconds) ==="
    echo "model: $BE_MODEL"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

first=1
for m in paged fasttree mlca bs_kernel deft; do
    echo "--- method: $m ---" | tee -a "$LOG"
    tmp="$OUT_DIR/${m}.tmp.csv"
    rm -f "$tmp"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 64 --L_p 16000 --B 8 --max_new 256 \
        --prompts-file data/hotpotqa.jsonl \
        --warmup --methods "$m" --out "$tmp" \
        2>&1 | tee -a "$LOG" || echo "  ($m crashed/partial)" | tee -a "$LOG"
    if [[ -f "$tmp" ]]; then
        if [[ $first -eq 1 ]]; then cp "$tmp" "$CSV"; first=0
        else tail -n +2 "$tmp" >> "$CSV"; fi
        rm -f "$tmp"
    fi
done

echo
echo "=== beam_search plan-time breakdown table ===" | tee -a "$LOG"
python3 -c "
import csv
order = ['deft','mlca','bs_kernel','fasttree','paged']
rows = {r['method']: r for r in csv.DictReader(open('$CSV'))}
hdr = f'{\"method\":<10}{\"build_idx\":>11}{\"decision\":>10}{\"forward\":>11}{\"decode\":>11}{\"plan%dec\":>9}'
print(hdr)
for m in order:
    r = rows.get(m)
    if not r: continue
    plan = float(r['plan_total_ms']); disp = float(r['dispatch_total_ms'])
    build = plan - disp
    fwd = float(r['forward_total_ms']); dec = float(r['decode_total_ms'])
    pdec = plan/dec*100 if dec>0 else 0
    print(f'{m:<10}{build:>11.1f}{disp:>10.1f}{fwd:>11.1f}{dec:>11.1f}{pdec:>8.1f}%')
" | tee -a "$LOG"
echo "Done. CSV: $CSV" | tee -a "$LOG"
