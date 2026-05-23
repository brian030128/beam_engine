"""CPU pick-verification for the cost model. For each (model, cell) it
builds the WorkloadShape (per-rank heads), sweeps decode progress, and
reports which strategy family the picker chooses vs the empirically-best
family. Used to validate cost-model tweaks without GPU runs.
"""
import json, os
from collections import Counter
from beam_engine.methods.bs_kernel import cost_model as cm
from beam_engine.methods.bs_kernel.calibrate import _from_payload

c = _from_payload(json.load(open(os.path.expanduser(
    "~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json"))))

# model -> (num_kv_heads_per_rank, num_qo_heads_per_rank, head_dim)
MODELS = {
    "1B  TP1": (8, 32, 64),
    "8B  TP1": (8, 32, 128),
    "70B TP4": (2, 16, 128),
}
# cell -> (K, B, L_p, priv_tail, expected_family)
def cells_for(model):
    f2b = 16 if model.startswith("1B") else 4
    return {
        "F1 multi_doc":      (128, 1, 80000, 51, "dt"),
        "F2 multi_few_shot": (128, f2b, 6656, 80, "dt"),
        "F4 self_consist":   (16, 1, 16384, 11, "1p"),
    }

def fam(s):
    v = s.value
    if "dec_tail" in v: return "dt"
    if "1pool" in v: return "1p"
    if "2pool" in v: return "2p"
    if "per_beam" in v: return "pb"
    return v

def run():
    bad = 0
    for mname, (kvh, qoh, hd) in MODELS.items():
        for cname, (K, B, Lp, priv, exp) in cells_for(mname).items():
            picks = Counter()
            for dec in range(0, 256, 8):
                tail = priv + dec
                w = cm.WorkloadShape(
                    K=K, L_p=Lp, suffix_lens=[tail]*K,
                    num_kv_heads=kvh, head_dim=hd, bytes_per_kv=2*kvh*hd*2,
                    num_qo_heads=qoh, kv_dtype="bf16")
                picks[fam(cm.pick_strategy_batch([w]*B, c).strategy)] += 1
            dom = picks.most_common(1)[0][0]
            ok = "OK " if dom == exp else "MISPICK"
            if dom != exp: bad += 1
            print(f"  {mname}  {cname:18s} picks={dict(picks)}  dom={dom} exp={exp}  {ok}")
    print(f"  >>> mispicks: {bad}")

if __name__ == "__main__":
    run()
