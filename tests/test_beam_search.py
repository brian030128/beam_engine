"""
Beam search with copy-on-write paged KV cache.

Maintains K candidate beams per prompt, selecting the top-K
accumulated-probability sequences at each step. Beams share
prefix KV cache pages; copy-on-write when beams diverge,
free pages when beams are eliminated.

Verification:
  1. beam_width=1 must match greedy exactly (token-level)
  2. beam_width=4 best beam score >= greedy score
  3. Print all beam texts for manual coherence inspection
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from transformers import AutoTokenizer

from beam_engine.models.attention import AttentionMetadata
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.page_table import PageTable

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Beam:
    token_ids: list[int]       # generated tokens (excluding prompt)
    cum_log_prob: float        # cumulative log probability
    pages: list[int]           # ordered physical page indices for this beam's KV cache


# ---------------------------------------------------------------------------
# Page reference counting helpers
# ---------------------------------------------------------------------------

def add_ref(page_ref_counts: dict[int, int], page_idx: int, count: int = 1):
    page_ref_counts[page_idx] = page_ref_counts.get(page_idx, 0) + count


def remove_ref(
    page_ref_counts: dict[int, int],
    page_idx: int,
    page_table: PageTable,
    count: int = 1,
):
    page_ref_counts[page_idx] -= count
    assert page_ref_counts[page_idx] >= 0
    if page_ref_counts[page_idx] == 0:
        page_table.free_block(page_idx)
        del page_ref_counts[page_idx]


# ---------------------------------------------------------------------------
# Greedy decode (with log-prob tracking, for verification baseline)
# ---------------------------------------------------------------------------

def greedy_decode(
    model,
    config,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> tuple[list[int], float]:
    """Greedy decode with cumulative log-prob tracking. Returns (tokens, cum_log_prob)."""
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers
    prompt_len = len(prompt_ids)

    page_table = PageTable(
        layer_num=num_layers,
        page_size=PAGE_SIZE,
        max_num_pages=1024,
        head_num=num_kv_heads,
        head_dim=head_dim,
        device=torch.device(DEVICE),
        store_dtype=DTYPE,
    )
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", use_tensor_cores=True,
    )

    # Allocate prompt pages
    num_prompt_pages = (prompt_len + PAGE_SIZE - 1) // PAGE_SIZE
    pages = [page_table.allocate_block() for _ in range(num_prompt_pages)]

    # Per-token KV write positions
    kv_page_indices = torch.tensor(
        [pages[i // PAGE_SIZE] for i in range(prompt_len)], dtype=torch.int32, device=DEVICE
    )
    kv_page_offsets = torch.tensor(
        [i % PAGE_SIZE for i in range(prompt_len)], dtype=torch.int32, device=DEVICE
    )

    # Plan prefill
    qo_indptr = torch.tensor([0, prompt_len], dtype=torch.int32, device=DEVICE)
    paged_kv_indptr = torch.tensor([0, len(pages)], dtype=torch.int32, device=DEVICE)
    paged_kv_indices = torch.tensor(pages, dtype=torch.int32, device=DEVICE)
    last_page_len = prompt_len - (num_prompt_pages - 1) * PAGE_SIZE
    paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=DEVICE)

    prefill_wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=PAGE_SIZE,
        causal=True,
    )

    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    positions = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)

    attn_metadata = AttentionMetadata(
        is_prefill=True,
        prefill_wrapper=prefill_wrapper,
        page_table=page_table,
        kv_page_indices=kv_page_indices,
        kv_page_offsets=kv_page_offsets,
    )

    generated_tokens: list[int] = []
    cum_log_prob = 0.0

    with torch.no_grad():
        # Prefill
        hidden_states = model.forward(input_ids=input_ids, positions=positions, attn_metadata=attn_metadata)
        logits = model.compute_logits(hidden_states[:, -1, :])  # [1, vocab]
        log_probs = F.log_softmax(logits, dim=-1)               # [1, vocab]
        next_token = log_probs.argmax(dim=-1)                    # [1]
        cum_log_prob += log_probs[0, next_token[0]].item()
        generated_tokens.append(next_token[0].item())
        next_token_input = next_token.unsqueeze(0)               # [1, 1]

        current_pos = prompt_len

        # Decode loop
        for _ in range(max_new_tokens - 1):
            if current_pos % PAGE_SIZE == 0:
                pages.append(page_table.allocate_block())

            write_page_idx = pages[current_pos // PAGE_SIZE]
            write_page_offset = current_pos % PAGE_SIZE

            decode_indptr = torch.tensor([0, len(pages)], dtype=torch.int32, device=DEVICE)
            decode_indices = torch.tensor(pages, dtype=torch.int32, device=DEVICE)
            decode_last_page_len = torch.tensor(
                [write_page_offset + 1], dtype=torch.int32, device=DEVICE
            )

            decode_wrapper.plan(
                indptr=decode_indptr,
                indices=decode_indices,
                last_page_len=decode_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=PAGE_SIZE,
            )

            kv_pi = torch.tensor([write_page_idx], dtype=torch.int32, device=DEVICE)
            kv_po = torch.tensor([write_page_offset], dtype=torch.int32, device=DEVICE)

            attn_metadata = AttentionMetadata(
                is_prefill=False,
                decode_wrapper=decode_wrapper,
                page_table=page_table,
                kv_page_indices=kv_pi,
                kv_page_offsets=kv_po,
            )

            decode_positions = torch.tensor([[current_pos]], device=DEVICE)
            hidden_states = model.forward(
                input_ids=next_token_input,
                positions=decode_positions,
                attn_metadata=attn_metadata,
            )
            logits = model.compute_logits(hidden_states[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)
            next_token = log_probs.argmax(dim=-1)
            cum_log_prob += log_probs[0, next_token[0]].item()
            generated_tokens.append(next_token[0].item())
            next_token_input = next_token.unsqueeze(0)

            current_pos += 1

    return generated_tokens, cum_log_prob


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def beam_search(
    model,
    config,
    prompt_ids: list[list[int]],
    max_new_tokens: int,
    beam_width: int,
    *,
    use_cascade: bool = True,
    page_table=None,
    workspace_buffer=None,
    prefill_wrapper=None,
    decode_wrapper=None,
) -> list[list[Beam]]:
    """
    Batched beam search with copy-on-write paged KV cache.

    Accepts B prompts and runs all B×K beams in a single batched forward pass.

    Position tracking follows the greedy decode pattern exactly:
      - After prefill, current_pos[b] = prompt_lens[b]
      - Each decode step feeds the last generated token, writes its KV at current_pos,
        then produces logits for the next token at current_pos + 1
      - current_pos increments by 1 each step

    Returns list[list[Beam]] — outer list per prompt, inner sorted by cum_log_prob (best first).
    """
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers
    B = len(prompt_ids)
    K = beam_width
    prompt_lens = [len(p) for p in prompt_ids]

    if page_table is not None:
        page_table.reset()
    else:
        page_table = PageTable(
            layer_num=num_layers,
            page_size=PAGE_SIZE,
            max_num_pages=2048,
            head_num=num_kv_heads,
            head_dim=head_dim,
            device=torch.device(DEVICE),
            store_dtype=DTYPE,
        )
    if workspace_buffer is None:
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    if prefill_wrapper is None:
        prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")
    if decode_wrapper is None:
        if use_cascade:
            decode_wrapper = MultiLevelCascadeAttentionWrapper(
                num_levels=2, float_workspace_buffer=workspace_buffer, kv_layout="NHD",
            )
        else:
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, kv_layout="NHD", use_tensor_cores=True,
            )

    # -----------------------------------------------------------------------
    # Prefill (batched — all B prompts in one ragged forward pass)
    # -----------------------------------------------------------------------

    # Allocate pages per prompt
    prompt_pages_list: list[list[int]] = []
    for plen in prompt_lens:
        num_pages = (plen + PAGE_SIZE - 1) // PAGE_SIZE
        pages = [page_table.allocate_block() for _ in range(num_pages)]
        prompt_pages_list.append(pages)

    # KV write indices (concatenated across all prompts)
    all_kv_pi: list[int] = []
    all_kv_po: list[int] = []
    for b in range(B):
        pages = prompt_pages_list[b]
        plen = prompt_lens[b]
        for i in range(plen):
            all_kv_pi.append(pages[i // PAGE_SIZE])
            all_kv_po.append(i % PAGE_SIZE)

    kv_page_indices = torch.tensor(all_kv_pi, dtype=torch.int32, device=DEVICE)
    kv_page_offsets = torch.tensor(all_kv_po, dtype=torch.int32, device=DEVICE)

    # Ragged batch metadata for FlashInfer prefill
    qo_indptr_list = [0]
    for plen in prompt_lens:
        qo_indptr_list.append(qo_indptr_list[-1] + plen)
    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device=DEVICE)

    paged_kv_indptr_list = [0]
    all_paged_kv_indices: list[int] = []
    paged_kv_lpl_list: list[int] = []
    for b in range(B):
        pages = prompt_pages_list[b]
        all_paged_kv_indices.extend(pages)
        paged_kv_indptr_list.append(paged_kv_indptr_list[-1] + len(pages))
        last_page_len = prompt_lens[b] - (len(pages) - 1) * PAGE_SIZE
        paged_kv_lpl_list.append(last_page_len)

    paged_kv_indptr = torch.tensor(paged_kv_indptr_list, dtype=torch.int32, device=DEVICE)
    paged_kv_indices = torch.tensor(all_paged_kv_indices, dtype=torch.int32, device=DEVICE)
    paged_kv_last_page_len = torch.tensor(paged_kv_lpl_list, dtype=torch.int32, device=DEVICE)

    prefill_wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=PAGE_SIZE,
        causal=True,
    )

    # Concatenate all prompt tokens and positions (ragged)
    all_token_ids: list[int] = []
    all_positions: list[int] = []
    for b in range(B):
        all_token_ids.extend(prompt_ids[b])
        all_positions.extend(range(prompt_lens[b]))

    input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    positions = torch.tensor(all_positions, device=DEVICE).unsqueeze(0)

    attn_metadata = AttentionMetadata(
        is_prefill=True,
        prefill_wrapper=prefill_wrapper,
        page_table=page_table,
        kv_page_indices=kv_page_indices,
        kv_page_offsets=kv_page_offsets,
    )

    with torch.no_grad():
        hidden_states = model.forward(input_ids=input_ids, positions=positions, attn_metadata=attn_metadata)
        # Extract last token of each prompt from concatenated sequence
        last_indices = [qo_indptr_list[b + 1] - 1 for b in range(B)]
        last_hidden = hidden_states[0, last_indices, :]          # [B, hidden_dim]
        logits = model.compute_logits(last_hidden)               # [B, vocab]
        log_probs = F.log_softmax(logits, dim=-1)                # [B, vocab]
        topk_log_probs, topk_ids = log_probs.topk(K, dim=-1)    # [B, K], [B, K]

    # Initialize beams per prompt — all K beams share prompt pages
    beams_per_prompt: list[list[Beam]] = []
    page_ref_counts_list: list[dict[int, int]] = []
    for b in range(B):
        beams: list[Beam] = []
        page_ref_counts: dict[int, int] = {}
        for i in range(K):
            beams.append(Beam(
                token_ids=[topk_ids[b, i].item()],
                cum_log_prob=topk_log_probs[b, i].item(),
                pages=list(prompt_pages_list[b]),
            ))
        for p in prompt_pages_list[b]:
            add_ref(page_ref_counts, p, K)
        beams_per_prompt.append(beams)
        page_ref_counts_list.append(page_ref_counts)

    # -----------------------------------------------------------------------
    # Decode loop (batched — all B×K beams in one forward pass)
    #
    # After prefill, the KV cache holds entries for positions [0, prompt_len_b).
    # The first generated token (from prefill logits) has NOT had its KV written.
    # In the decode loop, we feed that token, write its KV at current_pos,
    # and get logits for the next position.
    # -----------------------------------------------------------------------
    current_positions = list(prompt_lens)

    with torch.no_grad():
        for step in range(max_new_tokens - 1):

            # Step 1 — Ensure unique write pages (COW), per prompt
            for b in range(B):
                pos = current_positions[b]
                pli = pos // PAGE_SIZE
                off = pos % PAGE_SIZE
                for beam in beams_per_prompt[b]:
                    if off == 0:
                        new_page = page_table.allocate_block()
                        beam.pages.append(new_page)
                        add_ref(page_ref_counts_list[b], new_page, 1)
                    else:
                        write_page = beam.pages[pli]
                        if page_ref_counts_list[b][write_page] > 1:
                            new_page = page_table.copy_block(write_page, off)
                            remove_ref(page_ref_counts_list[b], write_page, page_table)
                            beam.pages[pli] = new_page
                            add_ref(page_ref_counts_list[b], new_page, 1)

            # Step 2 — Build decode metadata (all B×K beams)
            all_write_pi: list[int] = []
            all_write_po: list[int] = []
            all_input: list[list[int]] = []
            all_pos: list[list[int]] = []

            tensor = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEVICE)

            if use_cascade:
                # Cascade: Level 0 (shared prompt full pages), Level 1 (unique suffix)
                full_prompt_pages = [prompt_lens[b] // PAGE_SIZE for b in range(B)]

                l0_qo_indptr = [b * K for b in range(B + 1)]
                l0_kv_indptr = [0]
                l0_kv_indices: list[int] = []
                l0_kv_lpl: list[int] = []
                for b in range(B):
                    shared_pages = prompt_pages_list[b][:full_prompt_pages[b]]
                    l0_kv_indices.extend(shared_pages)
                    l0_kv_indptr.append(len(l0_kv_indices))
                    l0_kv_lpl.append(PAGE_SIZE if full_prompt_pages[b] > 0 else 0)

                l1_qo_indptr = list(range(B * K + 1))
                l1_kv_indptr = [0]
                l1_kv_indices: list[int] = []
                l1_kv_lpl: list[int] = []
                for b in range(B):
                    pos = current_positions[b]
                    off = pos % PAGE_SIZE
                    pli = pos // PAGE_SIZE
                    n_shared = full_prompt_pages[b]
                    for beam in beams_per_prompt[b]:
                        suffix_pages = beam.pages[n_shared:]
                        l1_kv_indices.extend(suffix_pages)
                        l1_kv_indptr.append(len(l1_kv_indices))
                        l1_kv_lpl.append(off + 1)
                        all_write_pi.append(beam.pages[pli])
                        all_write_po.append(off)
                        all_input.append([beam.token_ids[-1]])
                        all_pos.append([pos])

                decode_wrapper.plan(
                    qo_indptr_arr=[tensor(l0_qo_indptr), tensor(l1_qo_indptr)],
                    paged_kv_indptr_arr=[tensor(l0_kv_indptr), tensor(l1_kv_indptr)],
                    paged_kv_indices_arr=[tensor(l0_kv_indices), tensor(l1_kv_indices)],
                    paged_kv_last_page_len=[tensor(l0_kv_lpl), tensor(l1_kv_lpl)],
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=PAGE_SIZE,
                )
            else:
                # Flat: each beam is an independent sequence with all its pages
                flat_indptr = [0]
                flat_indices: list[int] = []
                flat_lpl: list[int] = []
                for b in range(B):
                    pos = current_positions[b]
                    off = pos % PAGE_SIZE
                    pli = pos // PAGE_SIZE
                    for beam in beams_per_prompt[b]:
                        flat_indices.extend(beam.pages)
                        flat_indptr.append(len(flat_indices))
                        flat_lpl.append(off + 1)
                        all_write_pi.append(beam.pages[pli])
                        all_write_po.append(off)
                        all_input.append([beam.token_ids[-1]])
                        all_pos.append([pos])

                decode_wrapper.plan(
                    indptr=tensor(flat_indptr),
                    indices=tensor(flat_indices),
                    last_page_len=tensor(flat_lpl),
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=PAGE_SIZE,
                )

            write_page_indices = torch.tensor(all_write_pi, dtype=torch.int32, device=DEVICE)
            write_page_offsets = torch.tensor(all_write_po, dtype=torch.int32, device=DEVICE)

            attn_metadata = AttentionMetadata(
                is_prefill=False,
                decode_wrapper=decode_wrapper,
                page_table=page_table,
                kv_page_indices=write_page_indices,
                kv_page_offsets=write_page_offsets,
            )

            # Step 3 — Forward pass (B*K sequences)
            beam_input_ids = torch.tensor(
                all_input, dtype=torch.long, device=DEVICE
            )  # [B*K, 1]
            beam_positions = torch.tensor(
                all_pos, device=DEVICE
            )  # [B*K, 1]

            hidden_states = model.forward(
                input_ids=beam_input_ids,
                positions=beam_positions,
                attn_metadata=attn_metadata,
            )
            logits = model.compute_logits(hidden_states[:, -1, :])  # [B*K, vocab]
            log_probs = F.log_softmax(logits, dim=-1)               # [B*K, vocab]

            # Step 4 — Score and select top-K independently per prompt
            vocab_size = logits.shape[-1]
            log_probs_bkv = log_probs.view(B, K, vocab_size)       # [B, K, vocab]

            cum_probs = torch.tensor(
                [[beam.cum_log_prob for beam in beams_per_prompt[b]] for b in range(B)],
                device=DEVICE, dtype=torch.float32,
            )  # [B, K]
            scores = cum_probs[:, :, None] + log_probs_bkv.float()  # [B, K, vocab]
            flat_scores = scores.reshape(B, -1)                      # [B, K*vocab]
            topk_scores, topk_flat_ids = flat_scores.topk(K, dim=-1) # [B, K]

            parent_beam_ids = topk_flat_ids // vocab_size   # [B, K]
            new_token_ids = topk_flat_ids % vocab_size      # [B, K]

            # Step 5 — Rearrange beams per prompt
            new_beams_per_prompt: list[list[Beam]] = []
            for b in range(B):
                parent_usage = [0] * K
                for pid in parent_beam_ids[b].tolist():
                    parent_usage[pid] += 1

                for old_idx, usage in enumerate(parent_usage):
                    if usage == 0:
                        for p in beams_per_prompt[b][old_idx].pages:
                            remove_ref(page_ref_counts_list[b], p, page_table)
                    elif usage > 1:
                        for p in beams_per_prompt[b][old_idx].pages:
                            add_ref(page_ref_counts_list[b], p, usage - 1)

                new_beams: list[Beam] = []
                for i in range(K):
                    pid = parent_beam_ids[b, i].item()
                    new_beams.append(Beam(
                        token_ids=beams_per_prompt[b][pid].token_ids + [new_token_ids[b, i].item()],
                        cum_log_prob=topk_scores[b, i].item(),
                        pages=list(beams_per_prompt[b][pid].pages),
                    ))
                new_beams_per_prompt.append(new_beams)

            beams_per_prompt = new_beams_per_prompt
            for b in range(B):
                current_positions[b] += 1

    # Sort per prompt and return
    for b in range(B):
        beams_per_prompt[b].sort(key=lambda beam: beam.cum_log_prob, reverse=True)
    return beams_per_prompt


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded!\n")

    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)
    max_new_tokens = 20

    # --- Test 1: Greedy decode baseline ---
    print("=" * 60)
    print("Test 1: Greedy decode (baseline)")
    print("=" * 60)
    greedy_tokens, greedy_score = greedy_decode(model, config, prompt_ids, max_new_tokens)
    greedy_text = tokenizer.decode(greedy_tokens, skip_special_tokens=True)
    print(f"  Tokens: {greedy_tokens}")
    print(f"  Score:  {greedy_score:.4f}")
    print(f"  Text:   {prompt}{greedy_text}")
    print()

    # --- Test 2: beam_width=1 must match greedy exactly ---
    print("=" * 60)
    print("Test 2: Beam search (width=1) — must match greedy")
    print("=" * 60)
    beams_w1 = beam_search(model, config, [prompt_ids], max_new_tokens, beam_width=1)[0]
    b1 = beams_w1[0]
    b1_text = tokenizer.decode(b1.token_ids, skip_special_tokens=True)
    print(f"  Tokens: {b1.token_ids}")
    print(f"  Score:  {b1.cum_log_prob:.4f}")
    print(f"  Text:   {prompt}{b1_text}")

    match = (b1.token_ids == greedy_tokens)
    print(f"\n  Token match with greedy: {'PASS' if match else 'FAIL'}")
    if not match:
        for i, (gt, bt) in enumerate(zip(greedy_tokens, b1.token_ids)):
            if gt != bt:
                print(f"    First mismatch at position {i}: greedy={gt} beam={bt}")
                break
    print()

    # --- Test 3: beam_width=4, best beam >= greedy ---
    print("=" * 60)
    print("Test 3: Beam search (width=4)")
    print("=" * 60)
    beams_w4 = beam_search(model, config, [prompt_ids], max_new_tokens, beam_width=4)[0]

    for i, b in enumerate(beams_w4):
        text = tokenizer.decode(b.token_ids, skip_special_tokens=True)
        print(f"  Beam {i}: score={b.cum_log_prob:.4f}  text=\"{prompt}{text}\"")

    best_beam = beams_w4[0]
    score_check = best_beam.cum_log_prob >= greedy_score
    print(f"\n  Best beam score ({best_beam.cum_log_prob:.4f}) >= greedy ({greedy_score:.4f}): "
          f"{'PASS' if score_check else 'FAIL'}")
    print()

    # --- Test 4: Batched multi-prompt vs individual ---
    print("=" * 60)
    print("Test 4: Batched multi-prompt beam search consistency")
    print("=" * 60)

    test_prompts = [
        "The capital of France is",
        "The meaning of life is",
        "In the year 2025, artificial intelligence",
    ]
    test_prompt_ids = [tokenizer.encode(p) for p in test_prompts]
    print(f"  Prompt lengths: {[len(p) for p in test_prompt_ids]}")

    # Run each prompt individually (B=1)
    individual_results: list[list[Beam]] = []
    for i, pids in enumerate(test_prompt_ids):
        result = beam_search(model, config, [pids], max_new_tokens, beam_width=4)[0]
        individual_results.append(result)
        text = tokenizer.decode(result[0].token_ids, skip_special_tokens=True)
        print(f"  Individual prompt {i}: \"{test_prompts[i]}{text}\"")

    # Run all prompts batched (B=3)
    batched_results = beam_search(model, config, test_prompt_ids, max_new_tokens, beam_width=4)
    for i in range(len(test_prompts)):
        text = tokenizer.decode(batched_results[i][0].token_ids, skip_special_tokens=True)
        print(f"  Batched   prompt {i}: \"{test_prompts[i]}{text}\"")

    # Compare best beam token_ids for each prompt
    batch_matches = []
    for i in range(len(test_prompts)):
        ind_tokens = individual_results[i][0].token_ids
        bat_tokens = batched_results[i][0].token_ids
        m = (ind_tokens == bat_tokens)
        batch_matches.append(m)
        status = "PASS" if m else "FAIL"
        print(f"  Prompt {i} batch==individual: {status}")
        if not m:
            for j, (a, b) in enumerate(zip(ind_tokens, bat_tokens)):
                if a != b:
                    print(f"    First mismatch at position {j}: individual={a} batched={b}")
                    break
    batch_all_match = all(batch_matches)
    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = match and score_check and batch_all_match
    print(f"  beam_width=1 matches greedy:   {'PASS' if match else 'FAIL'}")
    print(f"  beam_width=4 best >= greedy:   {'PASS' if score_check else 'FAIL'}")
    print(f"  batch==individual consistency: {'PASS' if batch_all_match else 'FAIL'}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
