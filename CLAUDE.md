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

## Branch

- `mini-vllm` — main development branch
