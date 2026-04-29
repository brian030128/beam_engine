"""Baseline 1 — Paged-attention beam search.

Each beam is treated as an independent sequence by FlashInfer's paged decode
wrapper. Prefix KV pages are shared across beams via a refcount; copy-on-write
runs at divergence; pruned beams release page refs. The shared prefix lives in
HBM exactly once but is read once per beam at decode time
(``O(B * L_p)`` traffic per step).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import flashinfer.page
import torch
import torch.nn.functional as F
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from ..models.attention import AttentionContext
from ..page_table import PageTable


# ---------------------------------------------------------------------------
# Attention context — dispatches q/k/v through FlashInfer paged wrappers
# ---------------------------------------------------------------------------


@dataclass
class PagedAttentionContext(AttentionContext):
    """Per-forward state for FlashInfer paged attention.

    The driver builds this once before each forward, then the model's per-layer
    attention call routes through ``attend``. Set ``is_prefill=True`` to use
    the prefill wrapper (ragged-batch, causal) or False for the decode wrapper.
    """

    is_prefill: bool
    page_table: PageTable
    kv_page_indices: torch.Tensor      # [nnz] int32 — page idx per token to write
    kv_page_offsets: torch.Tensor      # [nnz] int32 — offset within page per token
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None
    _write_helper_indptr: torch.Tensor | None = field(default=None, repr=False)

    def _get_kv_write_helpers(self, nnz: int, device: torch.device):
        buf = self._write_helper_indptr
        if buf is None or buf.shape[0] < nnz + 1:
            buf = torch.arange(nnz + 1, dtype=torch.int32, device=device)
            self._write_helper_indptr = buf
        return buf[:nnz], buf[: nnz + 1]

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim

        kv_cache = self.page_table.kv_cache_at_layer[layer_idx]

        k_3d = k.view(-1, num_kv_heads, head_dim)
        v_3d = v.view(-1, num_kv_heads, head_dim)
        batch_indices, kv_indptr = self._get_kv_write_helpers(k_3d.shape[0], k_3d.device)
        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_indices,
            positions=self.kv_page_offsets,
            paged_kv_cache=kv_cache,
            kv_indices=self.kv_page_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.kv_page_offsets,
            kv_layout="NHD",
        )

        q_3d = q.view(-1, num_heads, head_dim)
        if self.is_prefill:
            output = self.prefill_wrapper.run(q_3d, kv_cache)
        else:
            output = self.decode_wrapper.run(q_3d, kv_cache)
        return output.reshape(*q.shape[:-1], num_heads * head_dim)


# ---------------------------------------------------------------------------
# Beam state
# ---------------------------------------------------------------------------


@dataclass
class Beam:
    token_ids: list[int]       # generated tokens (excluding prompt)
    cum_log_prob: float
    pages: list[int]           # ordered physical page indices for this beam


def _add_ref(rc: dict[int, int], page: int, count: int = 1) -> None:
    rc[page] = rc.get(page, 0) + count


def _remove_ref(rc: dict[int, int], page: int, page_table: PageTable, count: int = 1) -> None:
    rc[page] -= count
    assert rc[page] >= 0
    if rc[page] == 0:
        page_table.free_block(page)
        del rc[page]


# ---------------------------------------------------------------------------
# Beam search driver
# ---------------------------------------------------------------------------


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
    return_timings: bool = False,
):
    """Batched beam search with copy-on-write paged KV cache.

    Returns ``list[list[Beam]]`` — outer index per prompt, inner sorted by
    ``cum_log_prob`` (best first). When ``return_timings=True``, returns
    ``(beams, timings)`` where timings is ``{prefill_ms, decode_step_ms: list}``.
    """
    timings = {"prefill_ms": 0.0, "decode_step_ms": []}
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    B = len(prompt_ids)
    K = beam_width
    prompt_lens = [len(p) for p in prompt_ids]

    page_table = PageTable(
        layer_num=num_layers,
        page_size=page_size,
        max_num_pages=max_num_pages,
        head_num=num_kv_heads,
        head_dim=head_dim,
        device=torch.device(device),
        store_dtype=dtype,
    )
    ps = page_table.page_size

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", use_tensor_cores=True
    )

    # ----- Prefill (ragged batch over B prompts) -----
    prompt_pages_list: list[list[int]] = []
    for plen in prompt_lens:
        num_pages = (plen + ps - 1) // ps
        prompt_pages_list.append([page_table.allocate_block() for _ in range(num_pages)])

    all_kv_pi: list[int] = []
    all_kv_po: list[int] = []
    for b in range(B):
        pages = prompt_pages_list[b]
        for i in range(prompt_lens[b]):
            all_kv_pi.append(pages[i // ps])
            all_kv_po.append(i % ps)

    kv_page_indices = torch.tensor(all_kv_pi, dtype=torch.int32, device=device)
    kv_page_offsets = torch.tensor(all_kv_po, dtype=torch.int32, device=device)

    qo_indptr_list = [0]
    for plen in prompt_lens:
        qo_indptr_list.append(qo_indptr_list[-1] + plen)

    paged_kv_indptr_list = [0]
    all_paged_kv_indices: list[int] = []
    paged_kv_lpl_list: list[int] = []
    for b in range(B):
        pages = prompt_pages_list[b]
        all_paged_kv_indices.extend(pages)
        paged_kv_indptr_list.append(paged_kv_indptr_list[-1] + len(pages))
        paged_kv_lpl_list.append(prompt_lens[b] - (len(pages) - 1) * ps)

    prefill_wrapper.plan(
        qo_indptr=torch.tensor(qo_indptr_list, dtype=torch.int32, device=device),
        paged_kv_indptr=torch.tensor(paged_kv_indptr_list, dtype=torch.int32, device=device),
        paged_kv_indices=torch.tensor(all_paged_kv_indices, dtype=torch.int32, device=device),
        paged_kv_last_page_len=torch.tensor(paged_kv_lpl_list, dtype=torch.int32, device=device),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=ps,
        causal=True,
    )

    all_token_ids: list[int] = []
    all_positions: list[int] = []
    for b in range(B):
        all_token_ids.extend(prompt_ids[b])
        all_positions.extend(range(prompt_lens[b]))

    input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    positions = torch.tensor(all_positions, device=device).unsqueeze(0)

    ctx = PagedAttentionContext(
        is_prefill=True,
        page_table=page_table,
        kv_page_indices=kv_page_indices,
        kv_page_offsets=kv_page_offsets,
        prefill_wrapper=prefill_wrapper,
    )

    if return_timings:
        torch.cuda.synchronize()
        t_pre = time.perf_counter()

    with torch.no_grad():
        hidden_states = model.forward(input_ids=input_ids, positions=positions, ctx=ctx)
        last_indices = [qo_indptr_list[b + 1] - 1 for b in range(B)]
        last_hidden = hidden_states[0, last_indices, :]
        logits = model.compute_logits(last_hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        topk_log_probs, topk_ids = log_probs.topk(K, dim=-1)

    if return_timings:
        torch.cuda.synchronize()
        timings["prefill_ms"] = (time.perf_counter() - t_pre) * 1000.0

    # All K beams of prompt b initially share that prompt's pages — refcount = K each.
    beams_per_prompt: list[list[Beam]] = []
    page_ref_counts_list: list[dict[int, int]] = []
    for b in range(B):
        beams = [
            Beam(
                token_ids=[topk_ids[b, i].item()],
                cum_log_prob=topk_log_probs[b, i].item(),
                pages=list(prompt_pages_list[b]),
            )
            for i in range(K)
        ]
        rc: dict[int, int] = {}
        for p in prompt_pages_list[b]:
            _add_ref(rc, p, K)
        beams_per_prompt.append(beams)
        page_ref_counts_list.append(rc)

    # ----- Decode loop -----
    current_positions = list(prompt_lens)

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if return_timings:
                torch.cuda.synchronize()
                t_step = time.perf_counter()
            # 1) Copy-on-write: ensure each beam owns its write page.
            for b in range(B):
                pos = current_positions[b]
                pli = pos // ps
                off = pos % ps
                for beam in beams_per_prompt[b]:
                    if off == 0:
                        new_page = page_table.allocate_block()
                        beam.pages.append(new_page)
                        _add_ref(page_ref_counts_list[b], new_page, 1)
                    else:
                        write_page = beam.pages[pli]
                        if page_ref_counts_list[b][write_page] > 1:
                            new_page = page_table.copy_block(write_page, off)
                            _remove_ref(page_ref_counts_list[b], write_page, page_table)
                            beam.pages[pli] = new_page
                            _add_ref(page_ref_counts_list[b], new_page, 1)

            # 2) Build per-beam decode metadata (B*K independent sequences).
            flat_indptr = [0]
            flat_indices: list[int] = []
            flat_lpl: list[int] = []
            all_write_pi: list[int] = []
            all_write_po: list[int] = []
            all_input: list[list[int]] = []
            all_pos: list[list[int]] = []

            for b in range(B):
                pos = current_positions[b]
                off = pos % ps
                pli = pos // ps
                for beam in beams_per_prompt[b]:
                    flat_indices.extend(beam.pages)
                    flat_indptr.append(len(flat_indices))
                    flat_lpl.append(off + 1)
                    all_write_pi.append(beam.pages[pli])
                    all_write_po.append(off)
                    all_input.append([beam.token_ids[-1]])
                    all_pos.append([pos])

            decode_wrapper.plan(
                indptr=torch.tensor(flat_indptr, dtype=torch.int32, device=device),
                indices=torch.tensor(flat_indices, dtype=torch.int32, device=device),
                last_page_len=torch.tensor(flat_lpl, dtype=torch.int32, device=device),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=ps,
            )

            ctx = PagedAttentionContext(
                is_prefill=False,
                page_table=page_table,
                kv_page_indices=torch.tensor(all_write_pi, dtype=torch.int32, device=device),
                kv_page_offsets=torch.tensor(all_write_po, dtype=torch.int32, device=device),
                decode_wrapper=decode_wrapper,
            )

            beam_input_ids = torch.tensor(all_input, dtype=torch.long, device=device)
            beam_positions = torch.tensor(all_pos, device=device)

            hidden_states = model.forward(
                input_ids=beam_input_ids,
                positions=beam_positions,
                ctx=ctx,
            )
            logits = model.compute_logits(hidden_states[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            # 3) Score and pick top-K independently per prompt.
            vocab_size = logits.shape[-1]
            log_probs_bkv = log_probs.view(B, K, vocab_size)
            cum_probs = torch.tensor(
                [[beam.cum_log_prob for beam in beams_per_prompt[b]] for b in range(B)],
                device=device,
                dtype=torch.float32,
            )
            scores = cum_probs[:, :, None] + log_probs_bkv.float()
            flat_scores = scores.reshape(B, -1)
            topk_scores, topk_flat_ids = flat_scores.topk(K, dim=-1)

            parent_beam_ids = topk_flat_ids // vocab_size
            new_token_ids = topk_flat_ids % vocab_size

            # 4) Resolve fork / eliminate: refcount surgery + new beam list.
            new_beams_per_prompt: list[list[Beam]] = []
            for b in range(B):
                parent_usage = [0] * K
                for pid in parent_beam_ids[b].tolist():
                    parent_usage[pid] += 1

                for old_idx, usage in enumerate(parent_usage):
                    if usage == 0:
                        for p in beams_per_prompt[b][old_idx].pages:
                            _remove_ref(page_ref_counts_list[b], p, page_table)
                    elif usage > 1:
                        for p in beams_per_prompt[b][old_idx].pages:
                            _add_ref(page_ref_counts_list[b], p, usage - 1)

                new_beams: list[Beam] = []
                for i in range(K):
                    pid = parent_beam_ids[b, i].item()
                    new_beams.append(
                        Beam(
                            token_ids=beams_per_prompt[b][pid].token_ids
                            + [new_token_ids[b, i].item()],
                            cum_log_prob=topk_scores[b, i].item(),
                            pages=list(beams_per_prompt[b][pid].pages),
                        )
                    )
                new_beams_per_prompt.append(new_beams)

            beams_per_prompt = new_beams_per_prompt
            for b in range(B):
                current_positions[b] += 1

            if return_timings:
                torch.cuda.synchronize()
                timings["decode_step_ms"].append((time.perf_counter() - t_step) * 1000.0)

    for b in range(B):
        beams_per_prompt[b].sort(key=lambda beam: beam.cum_log_prob, reverse=True)
    if return_timings:
        return beams_per_prompt, timings
    return beams_per_prompt
