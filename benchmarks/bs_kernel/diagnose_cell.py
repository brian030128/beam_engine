"""Diagnose why bs_kernel picks depth=3 / 2-pool at the failing cell.

Workload (fixed): K=64, L_p=8192, B=32, max_new=256, 32 pairwise-distinct
prompts, Llama-3.2-1B, fp16.

Two subcommands:

  run    --mode {auto,per_beam,2l1p,2l2p,3l1p,3l2p} --out <json>
         Run one beam_search invocation with the corresponding
         `available_strategies` filter; dump per-step decode_step_ms
         and (only for `auto`) the per-step Pick.debug dict.

  merge  --indir <dir-with-mode-jsons> --out <csv>
         Combine per-mode JSONs into a per-step CSV with predicted
         cost (from auto's pick.debug) and measured ms (from each
         forced run) side-by-side.

Each subcommand exits when done so SLURM can isolate the model load /
KV-page lifetime per mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from beam_engine.methods.bs_kernel import beam_search as bs_kernel_search
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"
DTYPE = torch.float16

K = 64
L_P = 8192
B = 32
MAX_NEW = 256
PAGE_SIZE = 16


_DISTINCT_SEEDS = [
    "Once upon a time, in a kingdom far away, there lived a curious scholar who studied the stars. ",
    "The bustling marketplace of Constantinople was filled with merchants from every corner of the known world. ",
    "Deep in the Amazon rainforest, biologists discovered a new species of luminescent frog. ",
    "On the dusty plains of the American Midwest, a young farmer dreamed of becoming a railroad engineer. ",
    "In Edo-period Japan, a master swordsmith forged blades that were said to sing when drawn. ",
    "The Antarctic research station crackled with radio static as the long polar night began. ",
    "Aboard the steamship bound for Liverpool, the elderly diplomat reread the encrypted message. ",
    "High in the Andes, a llama herder noticed strange patterns carved into the volcanic rock. ",
    "The detective lit her pipe and stared at the rain streaking down the office window. ",
    "Long before the first cities rose along the Tigris, hunter-gatherers followed seasonal herds. ",
    "When the great library of Alexandria still stood, scholars argued about the shape of the heavens. ",
    "The Viking longship cut through the cold North Sea waters under a sky heavy with gulls. ",
    "In a sleepy New England fishing village, the lighthouse keeper recorded the day's tide. ",
    "Beneath the towering canopy of redwoods, a paleobotanist sifted through compressed peat. ",
    "The neon-lit streets of Shibuya pulsed with commuters hurrying through the spring drizzle. ",
    "Aboard the International Space Station, the flight engineer floated past the Cupola windows. ",
    "In the dim lamplight of the medieval scriptorium, the monk dipped his quill once more. ",
    "The desert caravan paused at the oasis as the sun began its descent toward the dunes. ",
    "On a quiet research vessel in the Pacific, the marine biologist tagged her hundredth manta ray. ",
    "Through the crowded bazaar of old Damascus, the spice merchant called out his prices. ",
    "The astronaut adjusted her helmet visor and stepped onto the regolith of the lunar far side. ",
    "Inside the cathedral, the organist practiced a fugue while the masons repaired the buttress. ",
    "The Mongolian steppes stretched endlessly under a sky so wide it felt like another ocean. ",
    "Far below the surface of Europa, autonomous probes mapped the hydrothermal vents. ",
    "The Parisian café hummed with conversation about the latest exhibition at the Salon. ",
    "On a rocky coast of Cornwall, the lighthouse keeper recounted tales of shipwrecks past. ",
    "The Silk Road merchant unloaded his bolts of silk at the gates of Samarkand. ",
    "In the Brazilian favela, a young girl practiced her violin on the rooftop each evening. ",
    "The cartographer unrolled his maps and pointed to a coastline no European had yet seen. ",
    "Deep in the Carpathian mountains, the wolf packs moved silently through the winter snow. ",
    "Beneath the Antarctic ice shelf, the autonomous submarine recorded a never-before-heard call. ",
    "The royal astronomer of the Mughal court adjusted his sextant and watched Jupiter rise. ",
]


def _make_distinct_prompts(tok, target_len: int, B: int) -> list[list[int]]:
    prompts: list[list[int]] = []
    for i in range(B):
        seed = f"Document {i:03d}. " + _DISTINCT_SEEDS[i % len(_DISTINCT_SEEDS)]
        repeats = max(8, target_len // 50)
        while True:
            ids = tok.encode(seed * repeats, add_special_tokens=False)
            if len(ids) >= target_len:
                break
            repeats *= 2
        prompts.append(ids[:target_len])
    return prompts


MODE_TO_FILTER: dict[str, set[Strategy] | None] = {
    "auto":     None,
    "per_beam": {Strategy.PER_BEAM},
    "2l1p":     {Strategy.SHARED_2L_1POOL},
    "2l2p":     {Strategy.SHARED_2L_2POOL},
    # depth=3 modes use 2L fallback for steps where no prompt produces an
    # intermediate group structure (cost_model filters d=3 there).
    "3l1p":     {Strategy.SHARED_3L_1POOL, Strategy.SHARED_2L_1POOL},
    "3l2p":     {Strategy.SHARED_3L_2POOL, Strategy.SHARED_2L_2POOL},
}

MODES_IN_ORDER = ["auto", "per_beam", "2l1p", "2l2p", "3l1p", "3l2p"]


def cmd_run(args) -> None:
    model_name = getattr(args, "model", MODEL_NAME)
    print(f"loading model {model_name}...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, dtype=DTYPE, device=DEVICE)
    config = model.config
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"building {B} distinct prompts of length {L_P}...", flush=True)
    prompts = _make_distinct_prompts(tok, L_P, B)

    needed_pages = (
        sum((len(p) + PAGE_SIZE - 1) // PAGE_SIZE for p in prompts)
        + B * K * ((MAX_NEW + PAGE_SIZE) // PAGE_SIZE + 2)
        + 256
    )

    available = MODE_TO_FILTER[args.mode]
    return_picks = (args.mode == "auto")

    print(f"running mode={args.mode!r} filter={available}", flush=True)
    t0 = time.perf_counter()
    out: dict = {
        "mode": args.mode,
        "K": K, "L_p": L_P, "B": B, "max_new": MAX_NEW,
        "model_name": model_name,
    }
    try:
        if return_picks:
            beams, timings, picks = bs_kernel_search(
                model, config, prompts, MAX_NEW, K,
                return_timings=True, return_picks=True,
                available_strategies=available,
                max_num_pages=needed_pages,
            )
        else:
            beams, timings = bs_kernel_search(
                model, config, prompts, MAX_NEW, K,
                return_timings=True,
                available_strategies=available,
                max_num_pages=needed_pages,
            )
            picks = None
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)
        out["failed"] = True
        out["error"] = f"{type(e).__name__}: {e}"
        Path(args.out).write_text(json.dumps(out, indent=2))
        return

    wall = time.perf_counter() - t0
    decode = timings["decode_step_ms"]
    out["prefill_ms"] = float(timings["prefill_ms"])
    out["decode_step_ms"] = [float(x) for x in decode]
    out["decode_total_ms"] = float(sum(decode))
    out["decode_median_ms"] = float(statistics.median(decode)) if decode else float("nan")
    out["wall_s"] = wall

    # Per-phase breakdown (driver instrumentation). Each key is a list
    # parallel to decode_step_ms — one entry per decode step.
    out["breakdown"] = {
        k: [float(x) for x in v]
        for k, v in timings.items()
        if k.startswith("breakdown_")
    }

    if picks is not None:
        # All B prompts share the same Pick per step (batched picker).
        # Take prompt 0's pick log.
        steps = picks[0]
        out["picks"] = []
        for p in steps:
            out["picks"].append({
                "strategy": p.strategy.value,
                "estimated_us": float(p.estimated_us),
                "t_large": int(p.t_large),
                "depth": int(p.depth),
                "pool_count": int(p.pool_count),
                "debug": {k: float(v) for k, v in p.debug.items()},
            })

    Path(args.out).write_text(json.dumps(out))
    print(
        f"  prefill={out['prefill_ms']:.1f}ms  decode_total={out['decode_total_ms']:.1f}ms  "
        f"median_step={out['decode_median_ms']:.3f}ms  wall={wall:.1f}s",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


def cmd_merge(args) -> None:
    indir = Path(args.indir)
    out_path = Path(args.out)

    data: dict[str, dict] = {}
    for m in MODES_IN_ORDER:
        f = indir / f"{m}.json"
        if not f.exists():
            print(f"WARN: {f} missing", flush=True)
            continue
        data[m] = json.loads(f.read_text())

    if "auto" not in data or data["auto"].get("failed"):
        print("ERROR: auto run is missing or failed; cannot build CSV", flush=True)
        sys.exit(2)

    auto = data["auto"]
    n_steps = len(auto["decode_step_ms"])

    # Union of debug keys across all steps (depth=3 keys may be missing on
    # some steps; depth=2 + per_beam are always present).
    debug_keys: set[str] = set()
    for p in auto["picks"]:
        debug_keys.update(p["debug"].keys())
    debug_keys_sorted = sorted(debug_keys)

    header = [
        "step",
        "auto_pick_strategy", "auto_pick_t_large", "auto_pick_pool", "auto_pick_depth",
        "auto_pick_estimated_us",
    ]
    for k in debug_keys_sorted:
        header.append(f"pred_us_{k}")
    for m in MODES_IN_ORDER:
        header.append(f"meas_ms_{m}")

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in range(n_steps):
            pick = auto["picks"][s]
            row = [
                s,
                pick["strategy"], pick["t_large"], pick["pool_count"], pick["depth"],
                f"{pick['estimated_us']:.3f}",
            ]
            for k in debug_keys_sorted:
                v = pick["debug"].get(k)
                row.append(f"{v:.3f}" if v is not None else "")
            for m in MODES_IN_ORDER:
                d = data.get(m)
                if d is None or d.get("failed"):
                    row.append("")
                    continue
                ms_arr = d["decode_step_ms"]
                row.append(f"{ms_arr[s]:.4f}" if s < len(ms_arr) else "")
            w.writerow(row)

    print(f"wrote {out_path} ({n_steps} rows, {len(header)} cols)", flush=True)
    print()
    print("=== per-mode totals ===")
    print(f"  {'mode':<10} {'prefill_ms':>12} {'decode_total_ms':>17} {'decode_median_ms':>18} {'wall_s':>8}")
    for m in MODES_IN_ORDER:
        d = data.get(m)
        if d is None:
            continue
        if d.get("failed"):
            print(f"  {m:<10}  FAILED: {d.get('error')}")
            continue
        print(
            f"  {m:<10} {d['prefill_ms']:>12.1f} {d['decode_total_ms']:>17.1f} "
            f"{d['decode_median_ms']:>18.4f} {d['wall_s']:>8.1f}"
        )

    # Per-strategy step-level optimum analysis.
    print()
    print("=== per-step argmin analysis (excluding auto) ===")
    forced_modes = [m for m in MODES_IN_ORDER if m != "auto" and m in data and not data[m].get("failed")]
    if forced_modes:
        wins = {m: 0 for m in forced_modes}
        auto_regrets = []
        for s in range(n_steps):
            ms_per_mode = {}
            for m in forced_modes:
                arr = data[m]["decode_step_ms"]
                if s < len(arr):
                    ms_per_mode[m] = arr[s]
            if not ms_per_mode:
                continue
            best_mode = min(ms_per_mode, key=ms_per_mode.get)
            wins[best_mode] += 1
            best_ms = ms_per_mode[best_mode]
            # Regret = auto_step_ms / best_step_ms - 1 (if auto run available).
            auto_ms = auto["decode_step_ms"][s]
            if best_ms > 0:
                auto_regrets.append(auto_ms / best_ms - 1.0)

        print(f"  step-level wins (out of {n_steps} steps):")
        for m in forced_modes:
            print(f"    {m:<10} {wins[m]:>5}  ({100.0 * wins[m] / max(1,n_steps):.1f}%)")
        if auto_regrets:
            auto_regrets.sort()
            mean = sum(auto_regrets) / len(auto_regrets)
            p50 = auto_regrets[len(auto_regrets) // 2]
            p90 = auto_regrets[int(0.9 * len(auto_regrets))]
            p99 = auto_regrets[int(0.99 * len(auto_regrets))]
            mx = max(auto_regrets)
            print(f"  auto-vs-best regret: mean={mean:.3%} p50={p50:.3%} p90={p90:.3%} p99={p99:.3%} max={mx:.3%}")


def cmd_phase_summary(args) -> None:
    """Print per-phase totals + medians from a single mode's JSON."""
    d = json.loads(Path(args.input).read_text())
    if d.get("failed"):
        print(f"FAILED: {d.get('error')}")
        return
    print(f"mode={d['mode']!r}  K={d['K']} L_p={d['L_p']} B={d['B']} max_new={d['max_new']}")
    print(f"  prefill={d['prefill_ms']:.1f}ms  decode_total={d['decode_total_ms']:.1f}ms  "
          f"median_step={d['decode_median_ms']:.3f}ms  wall={d['wall_s']:.1f}s")
    bd = d.get("breakdown") or {}
    if not bd:
        print("  (no breakdown captured)")
        return
    print()
    print(f"  {'phase':<35} {'sum_ms':>12} {'mean_ms':>10} {'median_ms':>11} "
          f"{'p99_ms':>10} {'%_of_total':>11}")
    total = d["decode_total_ms"]
    rows = []
    for k in sorted(bd.keys()):
        vs = bd[k]
        if not vs:
            continue
        s = sum(vs)
        m = s / len(vs)
        ss = sorted(vs)
        med = ss[len(ss) // 2]
        p99 = ss[max(0, int(0.99 * (len(ss) - 1)))]
        rows.append((k, s, m, med, p99, s / total * 100.0))
    rows.sort(key=lambda r: -r[1])
    for k, s, m, med, p99, pct in rows:
        label = k.replace("breakdown_", "").replace("_ms", "")
        print(f"  {label:<35} {s:>12.1f} {m:>10.3f} {med:>11.3f} {p99:>10.3f} {pct:>10.2f}%")
    accounted = sum(r[1] for r in rows)
    print(f"  {'(sum of phases)':<35} {accounted:>12.1f}  ({accounted / total * 100.0:.2f}% of decode_total)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--mode", required=True, choices=list(MODE_TO_FILTER.keys()))
    run.add_argument("--out", required=True)
    run.add_argument("--model", default=MODEL_NAME,
                     help=f"HF model name (default: {MODEL_NAME})")
    run.set_defaults(func=cmd_run)

    merge = sub.add_parser("merge")
    merge.add_argument("--indir", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge)

    summ = sub.add_parser("phase_summary")
    summ.add_argument("--input", required=True)
    summ.set_defaults(func=cmd_phase_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
