"""End-to-end correctness for variable-kv-len split-K in the fused cascade.

2-level cascade with TWO level-0 groups of DIFFERENT kv-len (so the split-K
gets variable num_kv_chunks per request) and a small grid (so split-K
engages). Compares the fused wrapper against:
  (a) MultiLevelCascadeAttentionWrapper  (independent reference), and
  (b) the same fused wrapper with split-K forced off.
All three must agree.
"""
import os
import torch
import flashinfer


def build_inputs(K, s0_pages, s1_pages, uniq_pages, page_size, num_kv_heads, head_dim, dev):
    total_uniq = 2 * K * uniq_pages
    total_pages = s0_pages + s1_pages + total_uniq
    kv = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim,
                     dtype=torch.float16, device=dev)
    # Level 0: 2 groups (K queries each) over different shared lengths.
    l0_indices = torch.arange(0, s0_pages + s1_pages, dtype=torch.int32, device=dev)
    l0_indptr = torch.tensor([0, s0_pages, s0_pages + s1_pages], dtype=torch.int32, device=dev)
    l0_last = torch.tensor([page_size, page_size], dtype=torch.int32, device=dev)
    qo_top = torch.tensor([0, K, 2 * K], dtype=torch.int32, device=dev)
    # Level 1: per-query unique tails.
    l1_indices = torch.arange(0, total_uniq, dtype=torch.int32, device=dev) + (s0_pages + s1_pages)
    l1_indptr = torch.arange(0, 2 * K + 1, dtype=torch.int32, device=dev) * uniq_pages
    l1_last = torch.full((2 * K,), page_size, dtype=torch.int32, device=dev)
    qo_bot = torch.arange(0, 2 * K + 1, dtype=torch.int32, device=dev)
    return {
        "qo_indptr_arr": [qo_top, qo_bot],
        "kv_indptr_arr": [l0_indptr, l1_indptr],
        "kv_indices_arr": [l0_indices, l1_indices],
        "kv_last_page_arr": [l0_last, l1_last],
        "kv_data": kv,
    }


def run_mlca(inp, q, nqo, nkv, hd, ps):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    w = flashinfer.MultiLevelCascadeAttentionWrapper(2, ws, "NHD")
    w.plan(inp["qo_indptr_arr"], inp["kv_indptr_arr"], inp["kv_indices_arr"],
           inp["kv_last_page_arr"], nqo, nkv, hd, ps)
    return w.run(q, inp["kv_data"])


def run_fused(inp, q, nqo, nkv, hd, ps):
    w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(2, kv_layout="NHD")
    w.plan(inp["qo_indptr_arr"], inp["kv_indptr_arr"], inp["kv_indices_arr"],
           inp["kv_last_page_arr"], nqo, nkv, hd, ps)
    # report whether the variable-split path engaged
    print(f"   has_variable_split={getattr(w, '_has_variable_split', '?')} "
          f"num_chunks_per_level={getattr(w, '_num_chunks_per_level', '?')}")
    return w.run(q, inp["kv_data"])


def main():
    dev = "cuda"
    torch.manual_seed(0)
    K, ps, nkv, hd = 4, 16, 8, 128
    nqo = 8
    inp = build_inputs(K, s0_pages=128, s1_pages=96, uniq_pages=2,
                       page_size=ps, num_kv_heads=nkv, head_dim=hd, dev=dev)
    q = torch.randn(2 * K, nqo, hd, dtype=torch.float16, device=dev)

    ref = run_mlca(inp, q, nqo, nkv, hd, ps)
    print("split-K ON (variable):")
    out = run_fused(inp, q, nqo, nkv, hd, ps)
    os.environ["FLASHINFER_FORCE_KV_CHUNKS"] = "0:1,1:1"
    print("split-K forced OFF:")
    out_ns = run_fused(inp, q, nqo, nkv, hd, ps)
    os.environ.pop("FLASHINFER_FORCE_KV_CHUNKS", None)

    e_ref = (out.float() - ref.float()).abs().max().item()
    e_ns = (out.float() - out_ns.float()).abs().max().item()
    print(f"max |fused_split - mlca|   = {e_ref:.4e}")
    print(f"max |fused_split - nosplit|= {e_ns:.4e}")
    print("RESULT:", "PASS" if e_ref < 3e-2 and e_ns < 3e-2 else "FAIL")


if __name__ == "__main__":
    main()
