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
    prompt_ids: list[int],
    max_new_tokens: int,
    beam_width: int,
) -> list[Beam]:
    """
    Beam search with copy-on-write paged KV cache.

    Position tracking follows the greedy decode pattern exactly:
      - After prefill, current_pos = prompt_len (position of first generated token)
      - Each decode step feeds the last generated token, writes its KV at current_pos,
        then produces logits for the next token at current_pos + 1
      - current_pos increments by 1 each step

    Returns list of Beam sorted by cum_log_prob (best first).
    """
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers
    prompt_len = len(prompt_ids)
    K = beam_width

    page_table = PageTable(
        layer_num=num_layers,
        page_size=PAGE_SIZE,
        max_num_pages=2048,
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

    page_ref_counts: dict[int, int] = {}

    # -----------------------------------------------------------------------
    # Prefill
    # -----------------------------------------------------------------------
    num_prompt_pages = (prompt_len + PAGE_SIZE - 1) // PAGE_SIZE
    prompt_pages = [page_table.allocate_block() for _ in range(num_prompt_pages)]

    kv_page_indices = torch.tensor(
        [prompt_pages[i // PAGE_SIZE] for i in range(prompt_len)],
        dtype=torch.int32, device=DEVICE,
    )
    kv_page_offsets = torch.tensor(
        [i % PAGE_SIZE for i in range(prompt_len)],
        dtype=torch.int32, device=DEVICE,
    )

    qo_indptr = torch.tensor([0, prompt_len], dtype=torch.int32, device=DEVICE)
    paged_kv_indptr = torch.tensor([0, len(prompt_pages)], dtype=torch.int32, device=DEVICE)
    paged_kv_indices = torch.tensor(prompt_pages, dtype=torch.int32, device=DEVICE)
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

    with torch.no_grad():
        hidden_states = model.forward(input_ids=input_ids, positions=positions, attn_metadata=attn_metadata)
        logits = model.compute_logits(hidden_states[:, -1, :])   # [1, vocab]
        log_probs = F.log_softmax(logits, dim=-1)                # [1, vocab]
        topk_log_probs, topk_ids = log_probs[0].topk(K)         # [K], [K]

    # Initialize beams — all share the same prompt pages
    beams: list[Beam] = []
    for i in range(K):
        beams.append(Beam(
            token_ids=[topk_ids[i].item()],
            cum_log_prob=topk_log_probs[i].item(),
            pages=list(prompt_pages),
        ))
    for p in prompt_pages:
        add_ref(page_ref_counts, p, K)

    # -----------------------------------------------------------------------
    # Decode loop
    #
    # After prefill, the KV cache holds entries for positions [0, prompt_len).
    # The first generated token (from prefill logits) has NOT had its KV written.
    # In the decode loop, we feed that token, write its KV at current_pos,
    # and get logits for the next position.
    # -----------------------------------------------------------------------
    current_pos = prompt_len

    with torch.no_grad():
        for step in range(max_new_tokens - 1):
            page_list_idx = current_pos // PAGE_SIZE
            offset = current_pos % PAGE_SIZE

            # Step 1 — Ensure unique write pages (COW)
            for b in beams:
                if offset == 0:
                    # New page boundary: allocate fresh page
                    new_page = page_table.allocate_block()
                    b.pages.append(new_page)
                    add_ref(page_ref_counts, new_page, 1)
                else:
                    write_page = b.pages[page_list_idx]
                    if page_ref_counts[write_page] > 1:
                        new_page = page_table.copy_block(write_page, offset)
                        remove_ref(page_ref_counts, write_page, page_table)
                        b.pages[page_list_idx] = new_page
                        add_ref(page_ref_counts, new_page, 1)

            # Step 2 — Build decode wrapper metadata
            all_page_indices: list[int] = []
            indptr = [0]
            for b in beams:
                all_page_indices.extend(b.pages)
                indptr.append(len(all_page_indices))

            decode_indptr = torch.tensor(indptr, dtype=torch.int32, device=DEVICE)
            decode_indices = torch.tensor(all_page_indices, dtype=torch.int32, device=DEVICE)
            decode_last_page_len = torch.tensor(
                [offset + 1] * K, dtype=torch.int32, device=DEVICE
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

            # Per-beam write positions
            write_page_indices = torch.tensor(
                [b.pages[page_list_idx] for b in beams], dtype=torch.int32, device=DEVICE
            )
            write_page_offsets = torch.tensor(
                [offset] * K, dtype=torch.int32, device=DEVICE
            )

            attn_metadata = AttentionMetadata(
                is_prefill=False,
                decode_wrapper=decode_wrapper,
                page_table=page_table,
                kv_page_indices=write_page_indices,
                kv_page_offsets=write_page_offsets,
            )

            # Step 3 — Forward pass
            beam_input_ids = torch.tensor(
                [[b.token_ids[-1]] for b in beams], dtype=torch.long, device=DEVICE
            )  # [K, 1]
            beam_positions = torch.tensor(
                [[current_pos]] * K, device=DEVICE
            )  # [K, 1]

            hidden_states = model.forward(
                input_ids=beam_input_ids,
                positions=beam_positions,
                attn_metadata=attn_metadata,
            )
            logits = model.compute_logits(hidden_states[:, -1, :])  # [K, vocab]
            log_probs = F.log_softmax(logits, dim=-1)               # [K, vocab]

            # Step 4 — Score and select top-K
            cum_probs = torch.tensor(
                [b.cum_log_prob for b in beams], device=DEVICE, dtype=torch.float32
            )  # [K]
            scores = cum_probs[:, None] + log_probs.float()  # [K, vocab]
            flat_scores = scores.reshape(-1)                  # [K * vocab]
            topk_scores, topk_flat_ids = flat_scores.topk(K)  # [K], [K]

            vocab_size = logits.shape[-1]
            parent_beam_ids = topk_flat_ids // vocab_size      # [K]
            new_token_ids = topk_flat_ids % vocab_size         # [K]

            # Step 5 — Rearrange beams based on parent selection
            parent_usage = [0] * K
            for pid in parent_beam_ids.tolist():
                parent_usage[pid] += 1

            # Update ref counts based on parent usage
            for old_idx, usage in enumerate(parent_usage):
                if usage == 0:
                    # Eliminated: decrement refs on all its pages, free at 0
                    for p in beams[old_idx].pages:
                        remove_ref(page_ref_counts, p, page_table)
                elif usage > 1:
                    # Forked: add (usage-1) refs per page
                    for p in beams[old_idx].pages:
                        add_ref(page_ref_counts, p, usage - 1)
                # usage == 1: no ref change

            # Build new beams
            new_beams: list[Beam] = []
            for i in range(K):
                pid = parent_beam_ids[i].item()
                new_beams.append(Beam(
                    token_ids=beams[pid].token_ids + [new_token_ids[i].item()],
                    cum_log_prob=topk_scores[i].item(),
                    pages=list(beams[pid].pages),
                ))

            beams = new_beams
            current_pos += 1

    beams.sort(key=lambda b: b.cum_log_prob, reverse=True)
    return beams


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
    beams_w1 = beam_search(model, config, prompt_ids, max_new_tokens, beam_width=1)
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
    beams_w4 = beam_search(model, config, prompt_ids, max_new_tokens, beam_width=4)

    for i, b in enumerate(beams_w4):
        text = tokenizer.decode(b.token_ids, skip_special_tokens=True)
        print(f"  Beam {i}: score={b.cum_log_prob:.4f}  text=\"{prompt}{text}\"")

    best_beam = beams_w4[0]
    score_check = best_beam.cum_log_prob >= greedy_score
    print(f"\n  Best beam score ({best_beam.cum_log_prob:.4f}) >= greedy ({greedy_score:.4f}): "
          f"{'PASS' if score_check else 'FAIL'}")
    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = match and score_check
    print(f"  beam_width=1 matches greedy: {'PASS' if match else 'FAIL'}")
    print(f"  beam_width=4 best >= greedy: {'PASS' if score_check else 'FAIL'}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
