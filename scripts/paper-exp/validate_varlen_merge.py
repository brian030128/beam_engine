"""Standalone correctness check for the newly-bound
flashinfer.cascade.variable_length_merge_states kernel.

Builds ragged split-K partials (variable #partials per output row), folds
them with the kernel, and compares against a reference online-softmax
merge computed in float64. No model needed — just flashinfer on a GPU.
"""
import torch
from flashinfer.cascade import variable_length_merge_states


def reference_merge(v, s, indptr):
    """v:[P,H,D] s:[P,H] indptr:[R+1] -> v_merged:[R,H,D], s_merged:[R,H]."""
    R = indptr.numel() - 1
    H, D = v.shape[1], v.shape[2]
    vm = torch.zeros(R, H, D, dtype=torch.float64, device=v.device)
    sm = torch.full((R, H), float("-inf"), dtype=torch.float64, device=v.device)
    vf, sf = v.double(), s.double()
    for r in range(R):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        if hi <= lo:
            continue
        seg_s = sf[lo:hi]                         # [n, H]
        m = seg_s.max(dim=0).values               # [H]
        # FlashInfer stores lse in log2 units and merges via exp2 (base-2).
        w = torch.exp2(seg_s - m)                 # [n, H]
        denom = w.sum(dim=0)                      # [H]
        vm[r] = (w.unsqueeze(-1) * vf[lo:hi]).sum(dim=0) / denom.unsqueeze(-1)
        sm[r] = m + torch.log2(denom)
    return vm, sm


def main():
    torch.manual_seed(0)
    dev = "cuda"
    H, D = 8, 128
    # Ragged: rows with 4,1,8,3,5,2 partials (variable num_kv_chunks).
    counts = [4, 1, 8, 3, 5, 2, 6, 6]
    indptr = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()],
                          dtype=torch.int32, device=dev)
    P = int(indptr[-1])
    for dt in (torch.float16, torch.bfloat16):
        v = torch.randn(P, H, D, dtype=dt, device=dev)
        s = torch.randn(P, H, dtype=torch.float32, device=dev) * 3.0
        vm, sm = variable_length_merge_states(v, s, indptr)
        vref, sref = reference_merge(v, s, indptr)
        v_err = (vm.double() - vref).abs().max().item()
        s_err = (sm.double() - sref).abs().max().item()
        print(f"[{dt}] max |Δv|={v_err:.4e}  max |Δs|={s_err:.4e}  "
              f"{'PASS' if v_err < 5e-2 and s_err < 5e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
