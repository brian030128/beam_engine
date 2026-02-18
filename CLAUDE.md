# beam_engine

Standalone LLaMA inference engine. Originally forked from vLLM, rewritten to remove all vLLM dependencies. Targets single-GPU inference with PyTorch + FlashInfer kernels.

## Project Structure

```
src/beam_engine/
  models/
    modeling_llama.py   # LlamaForCausalLM — main model (no vLLM deps)
    attention.py        # FlashInferAttention + AttentionMetadata
    rotary_embedding.py # Pure PyTorch RoPE (supports llama3 scaling)
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
- **Two-phase attention**: Phase 1 uses naive `F.scaled_dot_product_attention` (no KV cache). Phase 2 adds FlashInfer paged attention.
- **No quantization, torch.compile, or CUDA graphs** for now

## Development Workflow

Development happens locally (Windows). Testing runs on a remote Linux GPU machine.

### Remote test machine
- Host: `brain_l@140.113.24.210`
- Test command: `ssh brain_l@140.113.24.210 "bash -i ./test.sh"`
- `test.sh` on the remote pulls latest code and runs `tests/test_vllm_model.py`
- Model: `meta-llama/Llama-3.1-8B` (needs HF access token on remote)

### Typical deploy + test cycle
```bash
git add <files>
git commit -m "message"
git push
ssh brain_l@140.113.24.210 "bash -i ./test.sh"
```

## Branch

- `mini-vllm` — main development branch
