# beam_engine

Standalone LLaMA inference engine. Originally forked from vLLM, rewritten to remove all vLLM dependencies. Targets single-GPU inference with PyTorch + FlashInfer kernels.

## Project Structure

```
src/beam_engine/
  models/
    modeling_llama.py   # LlamaForCausalLM — main model (no vLLM deps)
    attention.py        # FlashInfer paged attention + AttentionMetadata
    rotary_embedding.py # FlashInfer RoPE kernels (supports llama3 scaling)
    rmsnorm.py          # RMSNorm with fused residual add
    configuration_llama.py # HF-compatible LlamaConfig
  page_table.py         # Paged KV cache (FlashInfer 5D format)
  logger.py
tests/
  test_vllm_model.py    # Greedy decoding test on Llama-3.1-8B
```

## Key Architecture Decisions

- **Fused projections**: q/k/v → single `qkv_proj`, gate/up → single `gate_up_proj`
- **Weight loading**: HF safetensors are remapped at load time (concat q+k+v, gate+up)
- **Paged attention**: FlashInfer paged KV cache with prefill + decode wrappers (`attn_metadata` required)
- **No quantization, torch.compile, or CUDA graphs** for now

## Development Workflow

Development happens locally (Windows). Testing runs on a remote Linux GPU machine.

### Remote test machine
- Host: `brain_l@140.113.24.210`
- Test command: `ssh brain_l@140.113.24.210 "bash -i ./test.sh <test_file> <gpu_ids>"`
- `test.sh` on the remote: activates conda env `flashtree`, cd's to `~/flashtree/base/beam_engine`, runs `git pull`, then `uv run python tests/test_vllm_model.py`
- `test.sh` arguments:
  - First arg: test file (e.g. `tests/test_vllm_model.py`)
  - Second arg: comma-separated GPU IDs to use (e.g. `0,1` to use GPUs 0 and 1)
  - Example: `ssh brain_l@140.113.24.210 "bash -i ./test.sh tests/test_vllm_model.py 0,1"`
- Model: `meta-llama/Llama-3.1-8B` (needs HF access token on remote)
- Python environment: conda `flashtree` + uv virtualenv at `.venv/`
- To inspect the remote environment (e.g. check installed package versions or function signatures): `ssh brain_l@140.113.24.210 "bash -i -c 'conda activate flashtree && cd ~/flashtree/base/beam_engine && uv run python -c \"...\"'"`

### GPU availability — check before running
The remote machine is shared with other users. Before running any test, check which GPUs are free:
```bash
ssh brain_l@140.113.24.210 "bash -i -c nvidia-smi"
```
Pick GPUs with no active processes and pass them as the second argument to `test.sh`.

### Typical deploy + test cycle
```bash
git add <files>
git commit -m "message"
git push
ssh brain_l@140.113.24.210 "bash -i -c nvidia-smi"   # check free GPUs first
ssh brain_l@140.113.24.210 "bash -i ./test.sh tests/test_vllm_model.py <free_gpu_ids>"
```

## Beam Search

### Algorithm

1. **Prefill** (1 forward, batch size 1): Run the full prompt. Take logits at the last position, compute log-softmax, pick top-K tokens. Each becomes a beam with `cum_log_prob = log_prob(token)`.

2. **Decode loop** (`max_new_tokens - 1` forwards, batch size K): Each step:
   - Feed the last token of each beam through the model as a batch of K ([K, 1] input).
   - Compute scores: `cum_log_prob[beam] + log_prob[beam][token]` → shape [K, vocab].
   - Flatten to [K × vocab], pick top-K `(parent_beam_id, new_token)` pairs.
   - Beams can fork (multiple children from same parent) or be eliminated (usage=0).

3. **Return**: K beams sorted best-first by `cum_log_prob`.

**Total forwards with OUTPUT_LEN=10**: 1 prefill + 9 decode = 10 (all K beams batched per forward).

### Our Implementation (`tests/test_beam_search.py`)

**Data structure**: `Beam(token_ids, cum_log_prob, pages)` — `token_ids` holds generated tokens only (excludes prompt), `pages` is the ordered list of physical KV-cache page indices for this beam.

**Paged KV cache with reference counting + copy-on-write (COW)**:
- All K beams start sharing the same `prompt_pages` (each page gets ref_count = K).
- Before each decode write, COW is enforced:
  - At a page boundary (`offset == 0`): allocate a fresh page per beam (no sharing).
  - Mid-page (`offset > 0`): if `page_ref_counts[write_page] > 1`, copy the page up to `offset`, decrement ref on the old page, point the beam to the new page.
- When beams are eliminated (usage=0 after selection), all their page refs are decremented; pages with ref_count=0 are freed.
- When a beam is forked (usage>1), add `usage-1` extra refs to each of its pages.

**Batched decode forward**: All K beams run in one `model.forward()` call with input shape `[K, 1]`. The `decode_wrapper` describes K independent KV sequences via `indptr` / `indices` / `last_page_len`.

**Position tracking**: After prefill the KV cache holds positions `[0, prompt_len)`. The first generated token has NOT had its KV written yet. The decode loop starts at `current_pos = prompt_len`, writes KV for the fed token, produces logits for the next position, then increments `current_pos`.

**Reuse across calls**: `page_table`, `workspace_buffer`, `prefill_wrapper`, and `decode_wrapper` can be passed in and reused. `page_table.reset()` recycles bookkeeping without reallocating the ~4 GiB GPU tensor.

## Branch

- `mini-vllm` — main development branch
