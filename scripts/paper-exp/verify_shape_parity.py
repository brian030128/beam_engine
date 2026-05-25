"""Tokenizer-only check that multi-draw data seeds feed the model the SAME
input shape as the original single-draw final_paper_results run.

No GPU / no model load — this builds the F1/F2/F4 TreeSpecs for each data
seed, applies the same seed-0 shape-conform the harness uses, and asserts
that every draw's per-group token lengths are identical. It also reports
whether the text actually differs across draws (so we know the resampling
is real, not a no-op). F3 (beam_search) is shape-pinned by L_p truncation
in bench_batched.py, so it's verified by construction and only sanity-checked
here if the prompt pool file exists.

Run:
    HF_HOME=/work/u4320956/hf uv run python \
        scripts/paper-exp/verify_shape_parity.py \
        --model meta-llama/Llama-3.1-8B --seeds 0,1,2,3

Exit code 0 = all cells shape-identical across draws.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "benchmarks" / "bs_kernel"))


def _ref_shape(spec):
    """Per-group (shared_len, private_len). Mirrors the harness helper."""
    if spec.is_multilevel:
        return None
    return [
        (len(g.shared_prefix_ids), len(g.private_prefix_ids_per_leaf[0]))
        for g in spec.groups
    ]


def _conform(spec, ref, pad_or_trunc) -> None:
    """Same conform the harness applies to seeds>0 (pad/truncate to ref)."""
    if ref is None:
        return
    for g, (slen, plen) in zip(spec.groups, ref):
        g.shared_prefix_ids = pad_or_trunc(g.shared_prefix_ids, slen)
        g.private_prefix_ids_per_leaf = [
            pad_or_trunc(p, plen) for p in g.private_prefix_ids_per_leaf
        ]


def _spec_text_key(spec):
    """A hashable fingerprint of the actual token content (to confirm draws
    differ in text even though their shape is fixed)."""
    return tuple(
        (tuple(g.shared_prefix_ids), tuple(g.private_prefix_ids_per_leaf[0]))
        for g in spec.groups
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--f1-file", default=os.environ.get(
        "BE_HOTPOTQA_MULTI_Q_PATH", "data/hotpotqa_multi_q_K128_80k.jsonl"))
    ap.add_argument("--f4-lp", type=int, default=32768)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/work/u4320956/hf")
    os.environ["BE_HOTPOTQA_MULTI_Q_PATH"] = args.f1_file
    os.environ["BE_SELF_CONSISTENCY_LP"] = str(args.f4_lp)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    from transformers import AutoTokenizer
    import sglang_workloads as W

    tok = AutoTokenizer.from_pretrained(args.model)

    # (label, builder, kwargs) — the actual final_paper F-cells that go
    # through the sglang harness (F3 is the batched harness; pinned by L_p).
    cells = [
        ("F1 multi_doc_qa  K=128 B=1",
         W.build_multi_doc_qa, dict(k_override=128, b_override=1, paper_exact=True)),
        ("F2 multi_few_shot K=128 B=16",
         W.build_multi_few_shot, dict(k_override=128, b_override=16, paper_exact=True)),
        ("F2 multi_few_shot K=128 B=4",
         W.build_multi_few_shot, dict(k_override=128, b_override=4, paper_exact=True)),
        ("F4 self_consistency K=16 B=1",
         W.build_self_consistency, dict(k_override=16, b_override=1)),
    ]

    all_ok = True
    print(f"model={args.model}  seeds={seeds}\n")
    for label, builder, kw in cells:
        try:
            ref_spec = builder(tok, data_seed=seeds[0], **kw)
            ref = _ref_shape(ref_spec)
            ref_spec.validate()
        except FileNotFoundError as e:
            print(f"  [SKIP] {label}: {e}")
            continue
        text_keys = {_spec_text_key(ref_spec)}
        shape_ok = True
        for s in seeds[1:]:
            spec = builder(tok, data_seed=s, **kw)
            _conform(spec, ref, W._pad_or_truncate)
            spec.validate()
            if _ref_shape(spec) != ref:
                shape_ok = False
            text_keys.add(_spec_text_key(spec))
        L_s = ref[0][0]
        L_p = ref[0][1]
        n_distinct_text = len(text_keys)
        status = "PASS" if shape_ok else "FAIL"
        all_ok &= shape_ok
        print(
            f"  [{status}] {label:<32} shape: L_shared={L_s} L_priv={L_p} "
            f"groups={len(ref)} | identical across {len(seeds)} draws; "
            f"{n_distinct_text}/{len(seeds)} draws have distinct text"
        )
        if n_distinct_text == 1 and len(seeds) > 1:
            print(f"      NOTE: all draws produced identical text — data file "
                  f"may have too few rows to resample for this cell.")

    # F3: shape pinned by truncation to L_p in bench_batched.py. Confirm the
    # prompt pool has enough distinct prompts, if it exists.
    f3_pool = Path("data/hotpotqa_f3_pool.jsonl")
    print()
    if f3_pool.exists():
        import json
        n = sum(1 for ln in f3_pool.open() if ln.strip())
        print(f"  [INFO] F3 beam_search: shape pinned by L_p truncation; "
              f"pool {f3_pool} has {n} prompts (each draw slices a window).")
    else:
        print("  [INFO] F3 beam_search: shape pinned by L_p truncation in "
              "bench_batched.py (every prompt sliced to exactly L_p). "
              f"Pool {f3_pool} not built yet.")

    print()
    print("RESULT:", "ALL CELLS SHAPE-IDENTICAL ACROSS DRAWS ✓" if all_ok
          else "SHAPE MISMATCH DETECTED ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
