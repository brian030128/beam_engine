"""Paper sweep: phase timings + bs_kernel pick distribution across
(K, L_p, B, mode) for paged / mlca / fasttree / bs_kernel and their
diverse-beam-search variants.

Writes to ``<out_dir>``:
  * ``sweep_main.csv``     — one row per (method, K, L_p, B, mode)
  * ``sweep_picks.csv``    — bs_kernel pick histogram per cell
  * ``sweep_summary.md``   — paper-ready markdown tables

Cells that OOM are recorded as ``status=OOM`` rather than crashing the
whole sweep.

Usage:
    uv run python benchmarks/bs_kernel/sweep_paper.py \\
        --K 4 16 32 64 \\
        --L_p 8192 30000 \\
        --B 1 4 8 16 32 \\
        --max_new 256 \\
        --out_dir benchmarks/bs_kernel/results/sweep_paper
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import dbs, fasttree, mlca, paged
from beam_engine.methods import bs_kernel
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16


METHODS_STD = {
    "paged":     paged.beam_search,
    "mlca":      mlca.beam_search,
    "fasttree":  fasttree.beam_search,
    "bs_kernel": bs_kernel.beam_search,
}
METHODS_DBS = {
    "dbs_paged":     dbs.dbs_paged,
    "dbs_mlca":      dbs.dbs_mlca,
    "dbs_fasttree":  dbs.dbs_fasttree,
    "dbs_bs_kernel": dbs.dbs_bs_kernel,
}


def _make_prompt(tok, target_len: int) -> list[int]:
    seed = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
    )
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    return ids[:target_len]


def _phase_mean(timings: dict, key: str) -> float:
    xs = timings.get(key, [])
    return (sum(xs) / len(xs)) if xs else 0.0


def _run_method(
    name, fn, model, config, prompts, K, B, L_p, max_new, needed_pages,
    wpicks,
):
    """Run one (method, cell) combo. Returns a row dict (status OK or
    OOM/FAILED). On bs_kernel/dbs_bs_kernel also dumps pick histogram to
    wpicks.
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t_wall0 = time.perf_counter()

    kwargs = dict(
        max_num_pages=needed_pages,
        return_timings=True,
        return_phase_timings=True,
    )
    try:
        if name in ("bs_kernel", "dbs_bs_kernel"):
            kwargs["return_picks"] = True
            beams, timings, picks = fn(
                model, config, prompts, max_new, K, **kwargs,
            )
        else:
            beams, timings = fn(
                model, config, prompts, max_new, K, **kwargs,
            )
            picks = None
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {
            "method": name, "K": K, "L_p": L_p, "B": B,
            "max_new": max_new, "status": "OOM",
            "prefill_ms": 0.0, "decode_total_ms": 0.0,
            "decode_per_token_ms": 0.0,
            "plan_mean_ms": 0.0, "forward_mean_ms": 0.0,
            "topk_mean_ms": 0.0, "cow_mean_ms": 0.0, "fork_mean_ms": 0.0,
            "wall_s": time.perf_counter() - t_wall0,
        }
    except Exception as e:
        print(f"    [{name}] FAILED: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
        return {
            "method": name, "K": K, "L_p": L_p, "B": B,
            "max_new": max_new, "status": f"FAIL_{type(e).__name__}",
            "prefill_ms": 0.0, "decode_total_ms": 0.0,
            "decode_per_token_ms": 0.0,
            "plan_mean_ms": 0.0, "forward_mean_ms": 0.0,
            "topk_mean_ms": 0.0, "cow_mean_ms": 0.0, "fork_mean_ms": 0.0,
            "wall_s": time.perf_counter() - t_wall0,
        }

    torch.cuda.synchronize()
    wall = time.perf_counter() - t_wall0
    steps = timings["decode_step_ms"]
    n = max(1, len(steps))
    decode_total = sum(steps)

    # Dump picks for bs_kernel variants.
    if picks is not None and picks and picks[0]:
        cnt: Counter = Counter()
        for p in picks[0]:
            if p.strategy.name == "PER_BEAM":
                cnt[("PER_BEAM", -1, -1, -1)] += 1
            else:
                cnt[(p.strategy.name, p.depth, p.pool_count, p.t_large)] += 1
        for (strat, depth, pool, tlarge), count in cnt.most_common():
            wpicks.writerow([
                name, K, L_p, B, max_new,
                strat,
                depth if depth >= 0 else "-",
                pool if pool >= 0 else "-",
                tlarge if tlarge >= 0 else "-",
                count, n,
            ])

    return {
        "method": name, "K": K, "L_p": L_p, "B": B,
        "max_new": max_new, "status": "OK",
        "prefill_ms": timings["prefill_ms"],
        "decode_total_ms": decode_total,
        "decode_per_token_ms": decode_total / n,
        "plan_mean_ms":    _phase_mean(timings, "plan_ms"),
        "forward_mean_ms": _phase_mean(timings, "forward_ms"),
        "topk_mean_ms":    _phase_mean(timings, "topk_ms"),
        "cow_mean_ms":     _phase_mean(timings, "cow_ms"),
        "fork_mean_ms":    _phase_mean(timings, "fork_ms"),
        "wall_s": wall,
    }


def _emit_markdown(main_rows, picks_rows, out_md: Path):
    """Per (L_p, mode), one ms/step table; per cell, the bs_kernel
    dominant pick."""
    # Group by (L_p, mode) where mode in {std, dbs}
    by_lp_mode: dict = defaultdict(list)
    for r in main_rows:
        if r["status"] != "OK":
            continue
        mode = "dbs" if r["method"].startswith("dbs_") else "std"
        by_lp_mode[(r["L_p"], mode)].append(r)

    # Picks: pick the (most-common strategy, count) per (method, K, L_p, B)
    picks_dominant: dict = {}
    by_cell: dict = defaultdict(list)
    for p in picks_rows:
        method, K, L_p, B, max_new, strat, depth, pool, tlarge, count, n_steps = p
        by_cell[(method, K, L_p, B)].append((strat, depth, pool, tlarge,
                                              int(count), int(n_steps)))
    for key, entries in by_cell.items():
        entries.sort(key=lambda e: -e[4])
        top = entries[0]
        picks_dominant[key] = top  # (strat, depth, pool, tlarge, count, n_steps)

    lines: list[str] = []
    lines.append("# Cross-method sweep — paper results\n")
    lines.append(
        f"Model: `{MODEL_NAME}` (fp16). H100 80GB. "
        "Same prompt repeated B times.\n"
    )
    lines.append("OOM cells omitted.\n")

    # ms/step tables per (L_p, mode)
    for (L_p, mode), rows in sorted(by_lp_mode.items()):
        method_set = sorted({r["method"] for r in rows})
        Ks = sorted({r["K"] for r in rows})
        Bs = sorted({r["B"] for r in rows})
        lines.append(f"\n## L_p={L_p}, mode={mode} — ms/step\n")
        # Header
        header_cells = ["K\\B"] + [f"B={B}" for B in Bs]
        for method in method_set:
            lines.append(f"\n### {method}\n")
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
            for K in Ks:
                row = [f"K={K}"]
                for B in Bs:
                    cell = next(
                        (r for r in rows
                         if r["method"] == method and r["K"] == K
                         and r["B"] == B),
                        None,
                    )
                    row.append(
                        f"{cell['decode_per_token_ms']:.2f}"
                        if cell else "—"
                    )
                lines.append("| " + " | ".join(row) + " |")

    # bs_kernel pick tables per (L_p, mode)
    lines.append("\n\n## bs_kernel dominant pick per cell\n")
    lines.append("Format: ``strategy (depth, pool, t_large) — N/255 steps``\n")
    for L_p in sorted({r["L_p"] for r in main_rows}):
        for mode_name, bs_method in [("std", "bs_kernel"), ("dbs", "dbs_bs_kernel")]:
            lines.append(f"\n### L_p={L_p}, {bs_method}\n")
            Ks = sorted({k for (m, k, lp, b) in picks_dominant
                         if m == bs_method and lp == L_p})
            Bs = sorted({b for (m, k, lp, b) in picks_dominant
                         if m == bs_method and lp == L_p})
            if not Ks:
                lines.append("(no successful cells)")
                continue
            header_cells = ["K\\B"] + [f"B={B}" for B in Bs]
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
            for K in Ks:
                row = [f"K={K}"]
                for B in Bs:
                    entry = picks_dominant.get((bs_method, K, L_p, B))
                    if entry is None:
                        row.append("—")
                    else:
                        strat, depth, pool, tlarge, count, n_steps = entry
                        if strat == "PER_BEAM":
                            label = f"PER_BEAM {count}/{n_steps}"
                        else:
                            label = f"{strat} ({depth},{pool},{tlarge}) {count}/{n_steps}"
                        row.append(label)
                lines.append("| " + " | ".join(row) + " |")

    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[4, 16, 32, 64])
    ap.add_argument("--L_p", nargs="+", type=int, default=[8192, 30000])
    ap.add_argument("--B", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument(
        "--out_dir",
        default="benchmarks/bs_kernel/results/sweep_paper",
    )
    ap.add_argument("--no_dbs", action="store_true")
    ap.add_argument(
        "--methods", nargs="+", default=None,
        help="Restrict to a subset of methods (names from METHODS_STD/DBS). "
             "Default: all (or all-std if --no_dbs).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = dict(METHODS_STD)
    if not args.no_dbs:
        methods.update(METHODS_DBS)
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in set(args.methods)}
        if not methods:
            raise SystemExit(
                f"no methods match --methods={args.methods}; "
                f"available={list(METHODS_STD)} + {list(METHODS_DBS)}"
            )

    grid_size = len(args.K) * len(args.L_p) * len(args.B)
    print(
        f"grid: {grid_size} cells × {len(methods)} methods = "
        f"{grid_size * len(methods)} runs"
    )
    print(f"K={args.K}  L_p={args.L_p}  B={args.B}  max_new={args.max_new}")
    print(f"methods: {list(methods.keys())}\n")

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device=DEVICE,
    )
    config = model.config
    print("Model loaded.\n")

    main_csv = out_dir / "sweep_main.csv"
    picks_csv = out_dir / "sweep_picks.csv"
    summary_md = out_dir / "sweep_summary.md"

    main_fields = [
        "method", "K", "L_p", "B", "max_new", "status",
        "prefill_ms", "decode_total_ms", "decode_per_token_ms",
        "plan_mean_ms", "forward_mean_ms", "topk_mean_ms",
        "cow_mean_ms", "fork_mean_ms", "wall_s",
    ]
    main_rows: list[dict] = []
    picks_rows: list[list] = []

    with main_csv.open("w", newline="") as fmain, \
            picks_csv.open("w", newline="") as fpicks:
        wmain = csv.DictWriter(fmain, fieldnames=main_fields)
        wmain.writeheader()
        wpicks = csv.writer(fpicks)
        wpicks.writerow([
            "method", "K", "L_p", "B", "max_new",
            "strategy", "depth", "pool", "t_large", "count", "n_steps",
        ])

        for L_p in args.L_p:
            base = _make_prompt(tok, L_p)
            for K in args.K:
                for B in args.B:
                    prompts = [list(base) for _ in range(B)]
                    page_size = 16
                    unique_prompts = {tuple(p) for p in prompts}
                    unique_prompt_pages = sum(
                        (len(p) + page_size - 1) // page_size
                        for p in unique_prompts
                    )
                    needed_pages = (
                        unique_prompt_pages
                        + B * K * ((args.max_new + page_size) // page_size + 2)
                        + 256
                    )
                    print(
                        f"\n=== K={K}  L_p={L_p}  B={B}  "
                        f"(max_num_pages={needed_pages}) ===", flush=True,
                    )

                    # Wrapping picks into a per-cell buffer so we only
                    # write rows on OK status.
                    cell_picks_buffer: list = []
                    class _PicksProxy:
                        def writerow(self, row):
                            cell_picks_buffer.append(row)
                    proxy = _PicksProxy()

                    cell_rows = []
                    for name, fn in methods.items():
                        row = _run_method(
                            name, fn, model, config, prompts,
                            K, B, L_p, args.max_new, needed_pages, proxy,
                        )
                        status = row["status"]
                        if status == "OK":
                            print(
                                f"  [{name:<14}] {row['decode_per_token_ms']:7.2f} ms/step "
                                f"(plan {row['plan_mean_ms']:6.2f}, "
                                f"fwd {row['forward_mean_ms']:6.2f}, "
                                f"wall {row['wall_s']:.1f}s)",
                                flush=True,
                            )
                        else:
                            print(f"  [{name:<14}] {status}", flush=True)
                        cell_rows.append(row)

                    for r in cell_rows:
                        wmain.writerow(r)
                        main_rows.append(r)
                    for row in cell_picks_buffer:
                        wpicks.writerow(row)
                        picks_rows.append(row)
                    fmain.flush()
                    fpicks.flush()

    print(f"\nwrote {main_csv}")
    print(f"wrote {picks_csv}")
    _emit_markdown(main_rows, picks_rows, summary_md)
    print(f"wrote {summary_md}")


if __name__ == "__main__":
    main()
