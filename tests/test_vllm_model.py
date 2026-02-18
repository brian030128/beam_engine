"""
Test script for token-by-token generation with greedy decoding using LlamaForCausalLM.
Uses FlashInfer paged attention with prefill + decode loop.
"""

import torch
from transformers import AutoTokenizer
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
)

from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.models.attention import AttentionMetadata
from beam_engine.page_table import PageTable


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 16


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading LlamaForCausalLM model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded!")

    num_qo_heads = config.num_attention_heads       # 32
    num_kv_heads = config.num_key_value_heads        # 8
    head_dim = config.head_dim                       # 128
    num_layers = config.num_hidden_layers             # 32

    # Create page table
    page_table = PageTable(
        layer_num=num_layers,
        page_size=PAGE_SIZE,
        max_num_pages=1024,
        head_num=num_kv_heads,
        head_dim=head_dim,
        device=torch.device(DEVICE),
        store_dtype=DTYPE,
    )

    # Shared workspace buffer for FlashInfer wrappers (128 MB)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")

    # Input prompt
    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = input_ids.shape[1]

    # --- Allocate pages for prompt tokens ---
    pages = []  # list of physical page indices in order
    num_prompt_pages = (prompt_len + PAGE_SIZE - 1) // PAGE_SIZE
    for _ in range(num_prompt_pages):
        pages.append(page_table.allocate_block())

    # Compute per-token (page_idx, offset) for prefill
    token_positions = torch.arange(prompt_len, device=DEVICE)
    kv_page_indices = torch.tensor(
        [pages[i // PAGE_SIZE] for i in range(prompt_len)], dtype=torch.int32, device=DEVICE
    )
    kv_page_offsets = (token_positions % PAGE_SIZE).to(torch.int32)

    # --- Plan prefill wrapper ---
    paged_kv_indices = torch.tensor(pages, dtype=torch.int32, device=DEVICE)
    paged_kv_indptr = torch.tensor([0, len(pages)], dtype=torch.int32, device=DEVICE)
    last_page_len = prompt_len - (num_prompt_pages - 1) * PAGE_SIZE
    paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=DEVICE)
    qo_indptr = torch.tensor([0, prompt_len], dtype=torch.int32, device=DEVICE)

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

    # --- Prefill forward pass ---
    positions = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)
    attn_metadata = AttentionMetadata(
        is_prefill=True,
        prefill_wrapper=prefill_wrapper,
        decode_wrapper=None,
        page_table=page_table,
        kv_page_indices=kv_page_indices,
        kv_page_offsets=kv_page_offsets,
    )

    max_new_tokens = 20
    generated_tokens = []

    print("\nGenerating tokens (prefill + decode, greedy decoding):")
    print("-" * 50)

    with torch.no_grad():
        # Prefill: forward all prompt tokens at once
        hidden_states = model.forward(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
        )
        logits = model.compute_logits(hidden_states[:, -1:, :])
        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_token_id.item()
        generated_tokens.append(token_id)

        token_text = tokenizer.decode([token_id])
        print(f"Step 1: token_id={token_id}, text='{token_text}'")

        current_pos = prompt_len  # position of the token we just generated

        # Decode loop: one token at a time
        for i in range(1, max_new_tokens):
            if token_id == tokenizer.eos_token_id:
                print("  [EOS reached]")
                break

            # Allocate new page if needed
            if current_pos % PAGE_SIZE == 0:
                pages.append(page_table.allocate_block())

            # Compute (page_idx, offset) for the new token
            page_idx = pages[current_pos // PAGE_SIZE]
            page_offset = current_pos % PAGE_SIZE

            kv_page_indices_dec = torch.tensor([page_idx], dtype=torch.int32, device=DEVICE)
            kv_page_offsets_dec = torch.tensor([page_offset], dtype=torch.int32, device=DEVICE)

            # Plan decode wrapper
            # last_page_len is state AFTER writing the new token
            decode_last_page_len = (current_pos % PAGE_SIZE) + 1
            paged_kv_indices_dec = torch.tensor(pages, dtype=torch.int32, device=DEVICE)
            paged_kv_indptr_dec = torch.tensor([0, len(pages)], dtype=torch.int32, device=DEVICE)
            paged_kv_last_page_len_dec = torch.tensor(
                [decode_last_page_len], dtype=torch.int32, device=DEVICE
            )

            decode_wrapper.plan(
                indptr=paged_kv_indptr_dec,
                indices=paged_kv_indices_dec,
                last_page_len=paged_kv_last_page_len_dec,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=PAGE_SIZE,
            )

            # Decode forward pass: single token
            decode_positions = torch.tensor([[current_pos]], device=DEVICE)
            attn_metadata = AttentionMetadata(
                is_prefill=False,
                prefill_wrapper=None,
                decode_wrapper=decode_wrapper,
                page_table=page_table,
                kv_page_indices=kv_page_indices_dec,
                kv_page_offsets=kv_page_offsets_dec,
            )

            hidden_states = model.forward(
                input_ids=next_token_id,
                positions=decode_positions,
                attn_metadata=attn_metadata,
            )
            logits = model.compute_logits(hidden_states[:, -1:, :])
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token_id.item()
            generated_tokens.append(token_id)

            token_text = tokenizer.decode([token_id])
            print(f"Step {i+1}: token_id={token_id}, text='{token_text}'")

            current_pos += 1

    print("-" * 50)

    # Decode full generated sequence
    all_ids = tokenizer.encode(prompt) + generated_tokens
    full_text = tokenizer.decode(all_ids, skip_special_tokens=True)
    print(f"\nFinal generated text:\n{full_text}")
    print(f"\nGenerated token ids: {generated_tokens}")


if __name__ == "__main__":
    main()
