"""Baseline — ``unop_fasttree`` ("un-optimized FastTree").

Variant of the ``fasttree`` backend that adds a per-step *probe* calling the
upstream artifact's ``fasttree_preparation`` end-to-end (no port of its
algorithm, no integration optimizations). The probe receives our LCA-skipped
page-level tree converted to token-level seqlens plus synthetic contiguous
``KV_ptrs``; its returned metadata is discarded. The actual kernel is driven
by our validated metadata path, so end-to-end forward + token outputs match
``fasttree``.

The reported ``plan_ms`` is replaced after the harness returns with the
upstream probe time per step. This gives a wall-clock measurement of
"upstream as published + LCA-skipped tree handoff" against the same workload
the other methods are benchmarked on.

Reading the timings:

  * ``plan_total_ms`` (substituted) — upstream ``fasttree_preparation``
    wall-clock per step, summed. This is the headline number to cite as
    "FastTree plan-time as published". The radix walk and page→slot
    expansion are excluded — they are integration glue any FastTree consumer
    must pay, not part of upstream planner cost.
  * ``forward_*`` — unchanged from ``fasttree`` (same kernel, same KV
    append).
  * ``decode_total_ms`` — wall-clock of decode, which includes BOTH the
    upstream probe AND our validated planner running per step. NOT directly
    comparable to ``fasttree``'s ``decode_total_ms``; subtract the probe
    overhead (≈ ``plan_total_ms`` for ``unop_fasttree``) for a meaningful
    end-to-end number, or just compare ``forward_*`` + the substituted
    ``plan_total_ms`` against ``fasttree``'s.
  * ``unop_fasttree_probe_ms`` (in timings dict) — raw per-step probe
    times, same as the substituted ``plan_ms``.

Other notes:

  * ``KV_ptrs`` slot values are placeholders; the kernel is not driven by
    the probe's output, only timed by its execution.
  * ``fasttree_preparation`` mutates ``FastTreeParams`` (calls
    ``set_q_tile_sizes`` / ``set_kv_tile_sizes`` inside its refinement
    loop); we pass a fresh instance per probe so our validated planner is
    unaffected.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..page_driver import StepPlan, beam_search as _shared_beam_search
from ..page_table import PageTable
from ..methods.adaptive_pool import Beam
from ..decoding import (
    DecodeSelect,
    PrefillSelect,
    standard_decode_select,
    standard_prefill_select,
)
from .fasttree import (
    FastTreeAttentionContext,
    FastTreeBackend,
    _FastTreeWrappers,
    _build_combined_radix_tree_pages,
    _build_metadata,
    _expand_pages_to_slots,
)

_FT_DIR = (
    Path(__file__).resolve().parents[3]
    / "3rdparty"
    / "FastTree-Artifact"
    / "kernel_bench"
)
if str(_FT_DIR) not in sys.path:
    sys.path.insert(0, str(_FT_DIR))

from fasttree import FastTreeParams, fasttree_preparation  # noqa: E402
from kv_tree_simple import KVTreeNode  # noqa: E402


def _clone_tree_with_token_seqlens(
    tree_info: list[KVTreeNode],
) -> list[KVTreeNode]:
    """Build a new KVTreeNode list with the same shape but seqlens already in
    tokens (``tree_info[i].seqlen`` is mutated to tokens by
    ``_expand_pages_to_slots``; we copy to avoid coupling our planner state
    to the probe call which mutates ``num_children`` for the BFS queue).
    """
    out: list[KVTreeNode] = []
    for n in tree_info:
        n2 = KVTreeNode()
        n2.parent = n.parent
        n2.id = n.id
        n2.seqlen = n.seqlen
        n2.num_children = n.num_children
        n2.requests = list(n.requests)
        out.append(n2)
    return out


@dataclass
class UnopFastTreeBackend(FastTreeBackend):
    """FastTree backend with an upstream-direct planner probe.

    Extends ``FastTreeBackend`` with a per-step call to
    ``3rdparty/FastTree-Artifact/kernel_bench/fasttree.py:fasttree_preparation``
    on shape-matched synthetic inputs. The probe time is recorded on the
    backend and substituted into the harness's ``plan_ms`` timing after the
    run finishes.
    """

    name: str = "unop_fasttree"
    # Per-step list of probe wall-clock (ms). Populated inside
    # ``plan_decode_step`` and consumed by ``beam_search`` to replace
    # ``timings["plan_ms"]`` after the harness returns.
    _probe_ms_per_step: list[float] = field(
        default_factory=list, init=False, repr=False,
    )

    def plan_decode_step(
        self,
        *,
        wrappers: _FastTreeWrappers,
        beams_per_prompt: list[list[Beam]],
        current_pos: list[int],
        page_table: PageTable,
        K: int,
        B: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        last_lca_per_prompt: list[int],
    ) -> StepPlan:
        ps = page_size

        # ---- Gather + tree build (same as parent). ----
        shared_prefix_per_prompt: list[list[int]] = []
        tails_per_beam_per_prompt: list[list[list[int]]] = []
        for b in range(B):
            bp_b = beams_per_prompt[b]
            shared_prefix_per_prompt.append(bp_b[0].pages_prefix)
            tails_per_beam_per_prompt.append([beam.pages_tail for beam in bp_b])

        tree_info, node_pages = _build_combined_radix_tree_pages(
            shared_prefix_per_prompt, tails_per_beam_per_prompt, K,
        )

        leaf_partial_last: dict[int, int] = {}
        for i, n in enumerate(tree_info):
            if n.num_children == 0 and len(n.requests) == 1 and node_pages[i]:
                rid = n.requests[0]
                b_idx = rid // K
                pos = current_pos[b_idx]
                leaf_partial_last[i] = pos % ps + 1
        node_slots = _expand_pages_to_slots(
            tree_info, node_pages, ps, leaf_partial_last,
        )

        # ---- Upstream probe: call fasttree_preparation end-to-end. ----
        # tree_info.seqlen is now token-level (mutated by _expand_pages_to_slots).
        # Synthetic KV_ptrs: cumulative seqlen. Upstream consumes these via
        # ``list(range(KV_ptrs[node], KV_ptrs[node + 1]))``; values are
        # placeholders since we discard the returned metadata.
        probe_tree = _clone_tree_with_token_seqlens(tree_info)
        KV_ptrs: list[int] = [0]
        for n in probe_tree:
            KV_ptrs.append(KV_ptrs[-1] + max(int(n.seqlen), 0))

        # Fresh FastTreeParams — upstream mutates TSQs/TSKs via
        # set_q_tile_sizes / set_kv_tile_sizes inside the refinement loop.
        probe_params = FastTreeParams()
        probe_params.set_kv_group_num(num_qo_heads // num_kv_heads)

        _stdout_buf = io.StringIO()
        torch.cuda.synchronize()
        _t_probe_start = _time.perf_counter()
        with contextlib.redirect_stdout(_stdout_buf):
            try:
                fasttree_preparation(
                    probe_tree,
                    KV_ptrs,
                    B * K,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    list(self.KV_SPLIT_SIZES),
                    list(self.para_threshs1),
                    list(self.para_threshs2),
                    probe_params,
                )
            except Exception as e:  # noqa: BLE001
                # Probe must not break the run; record 0 and continue with
                # our validated planner. Surface the failure to stderr so
                # it isn't silent.
                print(
                    f"[unop_fasttree] upstream probe failed: {e!r}",
                    file=sys.stderr,
                )
        torch.cuda.synchronize()
        probe_ms = (_time.perf_counter() - _t_probe_start) * 1000.0
        self._probe_ms_per_step.append(probe_ms)

        # ---- Build real metadata for the kernel (our validated path). ----
        total_pages = 0
        for b in range(B):
            total_pages += len(shared_prefix_per_prompt[b])
            for tl in tails_per_beam_per_prompt[b]:
                total_pages += len(tl)
        cache_key = (len(tree_info), B, K, total_pages)
        meta = _build_metadata(
            tree_info=tree_info,
            node_slots=node_slots,
            batch_size=B * K,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            KV_SPLIT_SIZES=list(self.KV_SPLIT_SIZES),
            para_threshs1=list(self.para_threshs1),
            para_threshs2=list(self.para_threshs2),
            params=wrappers.ft_params,
            device=device,
            wrappers=wrappers,
            cache_key=cache_key,
        )

        # ---- write_slots + output buffer (same as parent). ----
        write_slots: list[int] = []
        for b in range(B):
            pos = current_pos[b]
            off = pos % ps
            pli = pos // ps
            bp_b = beams_per_prompt[b]
            prefix_len = len(bp_b[0].pages_prefix)
            tail_idx = pli - prefix_len
            for beam in bp_b:
                page = beam.pages_tail[tail_idx]
                write_slots.append(page * ps + off)
        write_slots_t = torch.tensor(
            write_slots, dtype=torch.int64, device=device,
        )

        if wrappers.out_buf is None or wrappers.out_buf.shape[0] < B * K:
            wrappers.out_buf = torch.empty(
                (B * K, num_qo_heads, head_dim),
                dtype=dtype, device=device,
            )
        out_buf = wrappers.out_buf[: B * K]

        sm_scale = 1.0 / (head_dim ** 0.5)
        ctx = FastTreeAttentionContext(
            page_table=page_table,
            write_slots=write_slots_t,
            sm_scale=sm_scale,
            meta=meta,
            out=out_buf,
        )
        return StepPlan(ctx=ctx, beam_order_per_prompt=None, pick=None)


def beam_search(
    model,
    config,
    prompt_ids: list[list[int]],
    max_new_tokens: int,
    beam_width: int,
    *,
    page_size: int = 16,
    max_num_pages: int = 2048,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
    return_timings: bool = False,
    return_phase_timings: bool = False,
    fasttree_params: FastTreeParams | None = None,
    KV_SPLIT_SIZES: tuple[int, int] = (1024, 128),
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """``unop_fasttree`` — FastTree with upstream-direct planner probe.

    Identical kernel + KV-append path to ``fasttree``. Per decode step, also
    calls upstream ``fasttree_preparation`` on a shape-matched synthetic
    input and times the call; that probe time is substituted into the
    harness's ``plan_ms`` so the wall-clock comparison reflects
    "upstream-as-published + LCA-skip handoff" rather than our optimized
    integration.
    """
    if kv_dtype is not None and kv_dtype != dtype:
        raise RuntimeError(
            "unop_fasttree fp8 KV not supported (inherits from fasttree)"
        )

    backend = UnopFastTreeBackend(
        fasttree_params=fasttree_params,
        KV_SPLIT_SIZES=KV_SPLIT_SIZES,
    )

    # Force phase timings so we can substitute the per-step ``plan_ms`` list
    # with the upstream probe times. Callers that asked only for
    # ``return_timings=True`` (without phase timings) will still receive the
    # phase-level dict — this is intentional, because the whole point of
    # ``unop_fasttree`` is the substituted plan_ms.
    force_phase = return_timings or return_phase_timings

    result = _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=backend,
        page_size=page_size,
        max_num_pages=max_num_pages,
        device=device,
        dtype=dtype,
        kv_dtype=kv_dtype,
        return_timings=return_timings,
        return_phase_timings=force_phase,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )

    if not return_timings:
        return result

    beams, timings = result
    probes = list(backend._probe_ms_per_step)
    timings["unop_fasttree_probe_ms"] = list(probes)
    if (
        "plan_ms" in timings
        and isinstance(timings["plan_ms"], list)
        and len(timings["plan_ms"]) == len(probes)
    ):
        timings["plan_ms"] = probes
    return beams, timings
