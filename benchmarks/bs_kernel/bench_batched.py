"""Cross-kernel batched-decode benchmark.

For each method × (K, L_p, B), run beam search on B prompts (use
``--distinct`` for pairwise-distinct prompts; default repeats one prompt
B times) and report total decode wall time per (prompt, token). All
methods (paged, tree, fasttree, mlca, adaptive_pool, bs_kernel) are
batched: one decode-loop iteration covers all B prompts, with a single
attention kernel launch per layer covering B·K query rows. The
per-prompt Python loop only builds the layout arrays before they're
concatenated into the batched dispatch.

Output CSV columns:
    method, K, L_p, B, max_new, prefill_ms, decode_total_ms,
    decode_per_token_ms, decode_per_prompt_per_token_ms

Usage:
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 16 --L_p 8192 --B 1 2 4 8 --max_new 16 \
        --methods paged tree fasttree mlca adaptive_pool bs_kernel \
        --out benchmarks/bs_kernel/results/bench_batched.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import dbs, deft, fasttree, mlca, paged, tree
from beam_engine.distributed import (
    destroy_tp,
    get_tp_rank,
    get_tp_world_size,
    init_tp,
)
from beam_engine.methods import adaptive_pool, bs_kernel
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.models import load_model_for_causal_lm
from beam_engine.models.modeling_llama import LlamaForCausalLM


import os as _os
MODEL_NAME = _os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
# Bound to ``cuda:LOCAL_RANK`` inside ``main`` once init_tp() runs. Plain
# ``cuda`` is fine for single-GPU runs (no torchrun → LOCAL_RANK unset).
DEVICE = "cuda"
_BE_DTYPE = _os.environ.get("BE_DTYPE", "fp16").lower()
DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}[_BE_DTYPE]
# Optional fp8-KV storage (Llama-3-70B-FP8 with fp8 KV path).
_BE_KV_DTYPE = _os.environ.get("BE_KV_DTYPE", "").lower()
KV_DTYPE: torch.dtype | None = {
    "": None, "fp16": torch.float16, "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn, "fp8_e4m3": torch.float8_e4m3fn,
}[_BE_KV_DTYPE]


def _bs_kernel_force_2l1p(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel pinned to SHARED_2L_1POOL — adaptive_pool's strategy
    dispatched through bs_kernel's driver path."""
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        dtype=dtype,
        kv_dtype=kv_dtype,
        available_strategies={Strategy.SHARED_2L_1POOL},
    )


def _bs_kernel_force_2l1p_single(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel pinned to SHARED_2L_1POOL but with the FusedMultiLevelCascade
    wrapper configured for ``single_launch=True`` — every (req, qo_tile) goes
    into ONE pool whose CTA_TILE_Q is derived from the prefix-level packed_qo
    (typically T=128 for K≥8). Drops the per-step cascade dispatch from two
    ``fused_paged_run`` launches (small-Q + large-Q pools) to one."""
    import flashinfer
    _orig_init = flashinfer.FusedMultiLevelCascadeAttentionWrapper.__init__

    def _patched_init(self, *a, **kw):
        kw.setdefault("single_launch", True)
        return _orig_init(self, *a, **kw)

    flashinfer.FusedMultiLevelCascadeAttentionWrapper.__init__ = _patched_init
    try:
        return bs_kernel.beam_search(
            model, config, prompts, max_new_tokens, beam_width,
            return_timings=return_timings,
            return_phase_timings=return_phase_timings,
            max_num_pages=max_num_pages,
            dtype=dtype,
            kv_dtype=kv_dtype,
            available_strategies={Strategy.SHARED_2L_1POOL},
        )
    finally:
        flashinfer.FusedMultiLevelCascadeAttentionWrapper.__init__ = _orig_init


def _bs_kernel_force_2l_dec_tail(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel pinned to SHARED_2L_DEC_TAIL — shared prefix prefill +
    CTA_Q=1 decode kernel for per-beam tail + merge_state_in_place.
    Baseline depth for the DECTAIL family in dispatch-space ablations."""
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        dtype=dtype,
        kv_dtype=kv_dtype,
        available_strategies={Strategy.SHARED_2L_DEC_TAIL},
    )


def _bs_kernel_force_3l1p(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel pinned to SHARED_3L_1POOL — fused 3-level cascade prefill
    (prefix + intermediate + per-beam tail) on the prefill kernel path
    (CTA_Q=K). When a step's workload has no intermediate-level structure,
    the picker's fallback path collapses to SHARED_2L_1POOL."""
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        dtype=dtype,
        kv_dtype=kv_dtype,
        available_strategies={Strategy.SHARED_3L_1POOL},
    )


def _bs_kernel_force_3l_dec_tail(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel pinned to SHARED_3L_DEC_TAIL. When a step's workload
    doesn't support depth=3 (no intermediate-level structure), the
    picker's fallback path collapses to the deepest supported family
    (SHARED_2L_DEC_TAIL)."""
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        dtype=dtype,
        kv_dtype=kv_dtype,
        available_strategies={Strategy.SHARED_3L_DEC_TAIL},
    )


def _bs_kernel_force_4l1p(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel pinned to SHARED_4L_1POOL. Requires
    ``max_cascade_levels=4`` (wrapper allocation) and
    ``max_dispatch_depth=4`` in the coefficients (picker enumeration).
    Hard-forced to the 4L family only: when a step's beam tree supports
    fewer than 4 qualifying levels the cost model auto-falls-back to the
    deepest supported 1POOL depth (so the strategy-mix trace reports the
    real per-step depth instead of letting a cheaper shallow variant win
    the cost comparison)."""
    from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
    from dataclasses import replace
    coeffs = replace(load_or_defaults("cuda"), max_dispatch_depth=4)
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        max_cascade_levels=4,
        coefficients=coeffs,
        dtype=dtype,
        kv_dtype=kv_dtype,
        available_strategies={Strategy.SHARED_4L_1POOL},
    )


def _bs_kernel_force_4l_dec_tail(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel hard-forced to SHARED_4L_DEC_TAIL. Auto-falls-back to the
    deepest supported DEC_TAIL depth only when the beam tree lacks 4
    qualifying levels (see ``_bs_kernel_force_4l1p``)."""
    from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
    from dataclasses import replace
    coeffs = replace(load_or_defaults("cuda"), max_dispatch_depth=4)
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        max_cascade_levels=4,
        coefficients=coeffs,
        dtype=dtype,
        kv_dtype=kv_dtype,
        available_strategies={Strategy.SHARED_4L_DEC_TAIL},
    )


def _bs_kernel_force_5l1p(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel hard-forced to SHARED_5L_1POOL (max_cascade_levels=5,
    max_dispatch_depth=5). Auto-falls-back to the deepest supported 1POOL
    depth when the beam tree has fewer than 5 qualifying levels."""
    from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
    from dataclasses import replace
    coeffs = replace(load_or_defaults("cuda"), max_dispatch_depth=5)
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        max_cascade_levels=5,
        coefficients=coeffs,
        dtype=dtype,
        kv_dtype=kv_dtype,
        available_strategies={Strategy.SHARED_5L_1POOL},
    )


def _bs_kernel_force_5l_dec_tail(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
):
    """bs_kernel hard-forced to SHARED_5L_DEC_TAIL (max_cascade_levels=5,
    max_dispatch_depth=5). Auto-falls-back to the deepest supported
    DEC_TAIL depth when the beam tree has fewer than 5 qualifying levels."""
    from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
    from dataclasses import replace
    coeffs = replace(load_or_defaults("cuda"), max_dispatch_depth=5)
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        max_cascade_levels=5,
        coefficients=coeffs,
        dtype=dtype,
        kv_dtype=kv_dtype,
        available_strategies={Strategy.SHARED_5L_DEC_TAIL},
    )


METHODS: dict[str, Callable] = {
    "paged":            paged.beam_search,
    "tree":             tree.beam_search,
    "fasttree":         fasttree.beam_search,
    "deft":             deft.beam_search,
    "mlca":             mlca.beam_search,
    "adaptive_pool":    adaptive_pool.beam_search,
    "bs_kernel":        bs_kernel.beam_search,
    "bs_kernel_2l1p":   _bs_kernel_force_2l1p,
    "bs_kernel_2l1p_single": _bs_kernel_force_2l1p_single,
    "bs_kernel_2ldt":   _bs_kernel_force_2l_dec_tail,
    "bs_kernel_3l1p":   _bs_kernel_force_3l1p,
    "bs_kernel_3ldt":   _bs_kernel_force_3l_dec_tail,
    "bs_kernel_4l1p":   _bs_kernel_force_4l1p,
    "bs_kernel_4ldt":   _bs_kernel_force_4l_dec_tail,
    "bs_kernel_5l1p":   _bs_kernel_force_5l1p,
    "bs_kernel_5ldt":   _bs_kernel_force_5l_dec_tail,
    # DBS variants — same kernel, diversity-penalized top-K
    # (num_groups=4, λ=0.5 default).
    "dbs_paged":         dbs.dbs_paged,
    "dbs_tree":          dbs.dbs_tree,
    "dbs_fasttree":      dbs.dbs_fasttree,
    "dbs_mlca":          dbs.dbs_mlca,
    "dbs_adaptive_pool": dbs.dbs_adaptive_pool,
    "dbs_bs_kernel":     dbs.dbs_bs_kernel,
}


@dataclass
class Row:
    method: str
    K: int
    L_p: int
    B: int
    max_new: int
    data_seed: int
    prefill_ms: float
    decode_total_ms: float
    decode_per_token_ms: float
    decode_per_prompt_per_token_ms: float
    # Per-phase decode-step breakdowns (from return_phase_timings=True).
    plan_mean_ms: float = 0.0
    plan_total_ms: float = 0.0
    forward_mean_ms: float = 0.0
    forward_total_ms: float = 0.0
    cow_mean_ms: float = 0.0
    cow_total_ms: float = 0.0
    topk_mean_ms: float = 0.0
    topk_total_ms: float = 0.0
    fork_mean_ms: float = 0.0
    fork_total_ms: float = 0.0
    # Dispatch-decision sub-total of plan time (build_indices = plan_total -
    # dispatch_total). Populated only under FT_TRACE_PLAN / BS_KERNEL_TRACE_PLAN.
    dispatch_total_ms: float = 0.0


def _phase_mean_total(xs):
    if not xs:
        return 0.0, 0.0
    total = sum(xs)
    return total / len(xs), total


def _make_prompt(tokenizer, target_len: int) -> list[int]:
    snippet = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
        "She traveled across mountains and rivers, recording her findings "
        "in a leather-bound journal that grew thicker with every passing "
        "season. "
    )
    repeats = max(8, target_len // 50)
    while True:
        ids = tokenizer.encode(snippet * repeats, add_special_tokens=False)
        if len(ids) >= target_len:
            break
        repeats *= 2
    return ids[:target_len]


# Distinct seed openings — keep B prompts pairwise prefix-disjoint so
# paged-style methods cannot share prefix pages across prompts.
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


def _make_distinct_prompts(tokenizer, target_len: int, B: int) -> list[list[int]]:
    """Build B prompts that are pairwise prefix-disjoint.

    Each prompt starts with a unique opening sentence + an index tag, then
    grows to ``target_len`` tokens by repeating its own seed. Different
    prompts therefore diverge from token 0.
    """
    prompts: list[list[int]] = []
    for i in range(B):
        seed = f"Document {i:03d}. " + _DISTINCT_SEEDS[i % len(_DISTINCT_SEEDS)]
        repeats = max(8, target_len // 50)
        while True:
            ids = tokenizer.encode(seed * repeats, add_special_tokens=False)
            if len(ids) >= target_len:
                break
            repeats *= 2
        prompts.append(ids[:target_len])
    return prompts


def run_one(
    method_name: str,
    method_fn: Callable,
    model,
    config,
    prompts: list[list[int]],
    K: int,
    max_new: int,
    max_pages_override: int | None = None,
    measure_start: int = 0,
    measure_end: int | None = None,
) -> Row | None:
    B = len(prompts)
    L_p = len(prompts[0])
    page_size = 16
    needed_pages = (
        sum((len(p) + page_size - 1) // page_size for p in prompts)
        + B * K * ((max_new + page_size) // page_size + 2)
        + 256
    )
    if max_pages_override is not None:
        needed_pages = max_pages_override
    extra_kwargs: dict = {}
    sig = inspect.signature(method_fn)
    if "max_num_pages" in sig.parameters:
        extra_kwargs["max_num_pages"] = needed_pages

    if "return_phase_timings" in sig.parameters:
        extra_kwargs["return_phase_timings"] = True

    if KV_DTYPE is not None and "kv_dtype" in sig.parameters:
        extra_kwargs["kv_dtype"] = KV_DTYPE

    # Compute dtype: backends default to fp16; if BE_DTYPE picks something
    # else (bf16 for Llama-3-70B-FP8), the prefill_wrapper.plan was given
    # q_data_type=fp16 but the model emits q at bf16 → dtype mismatch at
    # run(). Pass dtype explicitly when the backend signature accepts it.
    if "dtype" in sig.parameters:
        extra_kwargs["dtype"] = DTYPE

    try:
        beams, timings = method_fn(
            model, config, prompts, max_new, K,
            return_timings=True, **extra_kwargs,
        )
    except Exception as e:
        # OOM ends up here too; aggressive empty_cache to release any partial
        # buffers the failed driver allocated before the exception bubbled up.
        print(f"  [{method_name}] FAILED: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        return None

    # Measurement window: restrict all per-step aggregates to decode steps
    # [measure_start, measure_end). Used by the dispatch-space depth
    # ablation to time forward over the deep-decode tail (e.g. steps
    # 1500-2000), where the beam tree has accumulated long, stable shared
    # runs that deeper cascade levels can exploit. Default (0, None) = all.
    def _win(xs: list) -> list:
        return xs[measure_start:measure_end]

    decode_steps = _win(timings["decode_step_ms"])
    decode_total = sum(decode_steps)
    n_steps = len(decode_steps)
    # All methods are batched now: n_steps == max_new - 1 regardless of B.
    # Per-(prompt,token) latency normalizes by total tokens decoded over the
    # measured window (B * n_steps).
    total_tokens = B * n_steps
    per_token = decode_total / max(1, total_tokens)
    plan_mean, plan_total = _phase_mean_total(_win(timings.get("plan_ms", [])))
    fwd_mean,  fwd_total  = _phase_mean_total(_win(timings.get("forward_ms", [])))
    cow_mean,  cow_total  = _phase_mean_total(_win(timings.get("cow_ms", [])))
    topk_mean, topk_total = _phase_mean_total(_win(timings.get("topk_ms", [])))
    fork_mean, fork_total = _phase_mean_total(_win(timings.get("fork_ms", [])))
    # Dispatch-decision total across decode steps (same convention as
    # bench_sglang_e2e): fasttree logs per-step 'dispatch_ms', bs_kernel
    # logs 'pick_ms', into the backend plan trace. paged/mlca/deft have no
    # dispatch decision → 0. Requires the per-method TRACE_PLAN flag.
    dispatch_total = 0.0
    for e in _win(timings.get("plan_trace", [])):
        dispatch_total += float(e.get("dispatch_ms", 0.0)) + float(e.get("pick_ms", 0.0))
    return Row(
        method=method_name,
        K=K, L_p=L_p, B=B, max_new=max_new,
        data_seed=0,  # overwritten by the caller with the active draw seed
        prefill_ms=timings["prefill_ms"],
        decode_total_ms=decode_total,
        decode_per_token_ms=decode_total / max(1, n_steps),
        decode_per_prompt_per_token_ms=per_token,
        plan_mean_ms=plan_mean, plan_total_ms=plan_total,
        forward_mean_ms=fwd_mean, forward_total_ms=fwd_total,
        cow_mean_ms=cow_mean, cow_total_ms=cow_total,
        topk_mean_ms=topk_mean, topk_total_ms=topk_total,
        fork_mean_ms=fork_mean, fork_total_ms=fork_total,
        dispatch_total_ms=dispatch_total,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--L_p", nargs="+", type=int, default=[8192])
    ap.add_argument("--B", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--out", default=None)
    ap.add_argument("--distinct", action="store_true",
                    help="use B pairwise-distinct prompts instead of B copies "
                         "of one prompt (disables cross-prompt page sharing)")
    ap.add_argument("--max_pages", type=int, default=None,
                    help="override the computed max_num_pages cap (slab "
                         "allocation size). Use for shared-prompt cells where "
                         "the worst-case bound overallocates.")
    ap.add_argument("--warmup", action="store_true",
                    help="run each (method, cell) twice and discard the "
                         "first iter (absorbs Triton-JIT autotune + first-"
                         "call CUDA-graph capture).")
    ap.add_argument("--prompts-file", default=None,
                    help="JSONL file (one record per line, fields "
                         "{token_len, prompt}) supplying real-world prompts. "
                         "First B records are taken (skipping any with "
                         "token_len < L_p) and truncated to L_p tokens. "
                         "Overrides --distinct.")
    ap.add_argument("--data-seed", type=int, default=0,
                    help="data-draw index for --prompts-file: take the "
                         "window usable[data_seed*B : data_seed*B+B] instead "
                         "of the first B prompts, so distinct draws use "
                         "distinct HotpotQA prompts. L_p truncation already "
                         "fixes the shape, so every draw feeds the model the "
                         "same shape. data_seed=0 = original first-B behaviour.")
    ap.add_argument("--measure_start", type=int, default=0,
                    help="first decode-step index to include in the timing "
                         "aggregates (default 0). Use with --measure_end to "
                         "time the deep-decode tail only (dispatch ablation).")
    ap.add_argument("--measure_end", type=int, default=None,
                    help="one-past-last decode-step index for the timing "
                         "window (default: all steps).")
    args = ap.parse_args()

    methods = {k: METHODS[k] for k in args.methods if k in METHODS}
    if not methods:
        print(f"no valid methods in {args.methods}; valid: {list(METHODS.keys())}", file=sys.stderr)
        sys.exit(2)

    # Bind to the per-rank device under torchrun; a no-op for single-GPU
    # invocation (init_tp short-circuits when WORLD_SIZE=1). Mirrors the
    # bench_sglang_e2e.py setup so TP=2 cells (70B-fp8, B=8 beam_search)
    # actually use 2 GPUs instead of stacking both ranks on cuda:0.
    init_tp()
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()
    local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
    global DEVICE
    DEVICE = f"cuda:{local_rank}" if tp_size > 1 else "cuda"
    is_rank0 = tp_rank == 0

    def _say(*a, **kw):
        if is_rank0:
            print(*a, **kw)

    grid = [(K, L_p, B) for K in args.K for L_p in args.L_p for B in args.B]
    _say(f"grid: {len(grid)} (K,L_p,B) cells × {len(methods)} methods = {len(grid) * len(methods)} runs")
    _say(f"methods: {list(methods.keys())}")
    _say(f"max_new: {args.max_new}")
    _say(f"tp_size: {tp_size}, device: {DEVICE}")

    _say("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model_for_causal_lm(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    _say("Model loaded.\n")

    # Pre-load real-world prompts once if --prompts-file is set; we
    # re-truncate per (K, L_p, B) cell below.
    file_prompts_raw: list[list[int]] | None = None
    if args.prompts_file:
        import json as _json
        file_prompts_raw = []
        with open(args.prompts_file) as _pf:
            for line in _pf:
                rec = _json.loads(line)
                # Tokenize once; we'll truncate per cell.
                file_prompts_raw.append(tok.encode(rec["prompt"]))
        print(f"loaded {len(file_prompts_raw)} prompts from {args.prompts_file}")

    rows: list[Row] = []
    for (K, L_p, B) in grid:
        if file_prompts_raw is not None:
            usable = [ids for ids in file_prompts_raw if len(ids) >= L_p]
            need = B if args.data_seed == 0 else args.data_seed * B + B
            if len(usable) < B:
                _say(f"  WARNING: only {len(usable)} prompts ≥ {L_p} tokens; "
                     f"need {B}. Skipping (K={K}, L_p={L_p}, B={B}).")
                continue
            if len(usable) < need:
                _say(f"  NOTE: only {len(usable)} prompts ≥ {L_p}; data_seed="
                     f"{args.data_seed} window wraps (draws may overlap).")
            # Take a per-seed window so distinct draws use distinct prompts;
            # wrap if the file is short. data_seed=0 -> first B (original).
            off = args.data_seed * B
            prompts = [usable[(off + i) % len(usable)][:L_p] for i in range(B)]
        elif args.distinct:
            prompts = _make_distinct_prompts(tok, L_p, B)
        else:
            base = _make_prompt(tok, L_p)
            prompts = [list(base) for _ in range(B)]
        for name, fn in methods.items():
            iters = 2 if args.warmup else 1
            for it in range(iters):
                tag = "warmup " if (args.warmup and it == 0) else "       "
                _say(f"  {name:<14} {tag}K={K:<3d} L_p={L_p:<6d} B={B:<2d}", end=" ", flush=True)
                t0 = time.perf_counter()
                row = run_one(name, fn, model, config, prompts, K, args.max_new,
                              max_pages_override=args.max_pages,
                              measure_start=args.measure_start,
                              measure_end=args.measure_end)
                t1 = time.perf_counter()
                if row is None:
                    torch.cuda.empty_cache()
                    break  # OOM / failure — don't bother retrying
                # Discard warmup iteration's row.
                if args.warmup and it == 0:
                    _say(
                        f"discard          decode_total={row.decode_total_ms:7.1f}ms  "
                        f"({t1 - t0:5.1f}s wall)"
                    )
                else:
                    row.data_seed = args.data_seed
                    rows.append(row)
                    _say(
                        f"prefill={row.prefill_ms:7.1f}ms  "
                        f"decode_total={row.decode_total_ms:7.1f}ms  "
                        f"per_pt_token={row.decode_per_prompt_per_token_ms:6.2f}ms  "
                        f"({t1 - t0:5.1f}s wall)"
                    )
                torch.cuda.empty_cache()

    # CSV is identical across ranks (same timing on each), so only rank 0
    # writes it. Other ranks just clean up and exit.
    if not is_rank0:
        destroy_tp()
        return
    out = args.out
    if out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = results_dir / f"bench_batched-{gpu}-{ts}.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "K", "L_p", "B", "max_new", "data_seed",
            "prefill_ms", "decode_total_ms",
            "decode_per_token_ms", "decode_per_prompt_per_token_ms",
            "plan_mean_ms", "plan_total_ms",
            "forward_mean_ms", "forward_total_ms",
            "cow_mean_ms", "cow_total_ms",
            "topk_mean_ms", "topk_total_ms",
            "fork_mean_ms", "fork_total_ms",
            "dispatch_total_ms",
        ])
        for r in rows:
            w.writerow([
                r.method, r.K, r.L_p, r.B, r.max_new, r.data_seed,
                f"{r.prefill_ms:.4f}",
                f"{r.decode_total_ms:.4f}",
                f"{r.decode_per_token_ms:.4f}",
                f"{r.decode_per_prompt_per_token_ms:.4f}",
                f"{r.plan_mean_ms:.4f}",    f"{r.plan_total_ms:.4f}",
                f"{r.forward_mean_ms:.4f}", f"{r.forward_total_ms:.4f}",
                f"{r.cow_mean_ms:.4f}",     f"{r.cow_total_ms:.4f}",
                f"{r.topk_mean_ms:.4f}",    f"{r.topk_total_ms:.4f}",
                f"{r.fork_mean_ms:.4f}",    f"{r.fork_total_ms:.4f}",
                f"{r.dispatch_total_ms:.4f}",
            ])
    print(f"\nwrote {len(rows)} rows to {out}")
    destroy_tp()


if __name__ == "__main__":
    main()
