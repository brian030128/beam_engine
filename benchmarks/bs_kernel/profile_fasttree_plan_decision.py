"""Dump FastTree's planning decisions for a fixed (K, B, L_p) shape.

CPU-only — runs the radix-tree builder, _tree_heuristic, _compute_parallelism,
and the 3-iteration refinement loop with the same defaults the FastTree
backend uses, then prints what got decided. No GPU, no model.

Lets us answer: at K=16/B=8/L_p=30000, what does FastTree's planner
actually output? Specifically:
  * node_assignments (split-K vs split-Q per node)
  * Q_TILE_SIZE_PER_PHASE, KV_SPLIT_SIZE_PER_PHASE after refinement
  * parallelism per phase and how it evolved across iters
  * per-vnode (q_len, kv_len) histogram + phase reordering
  * total CTAs in stage 1 grid
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Hook FastTree artifact into sys.path the same way baselines/fasttree.py does.
_FT_DIR = Path(__file__).resolve().parents[2] / "3rdparty" / "FastTree-Artifact" / "kernel_bench"
if str(_FT_DIR) not in sys.path:
    sys.path.insert(0, str(_FT_DIR))

from fasttree import FastTreeParams, _tree_heuristic  # noqa: E402
from kv_tree_simple import KVTreeNode  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic tree matching the fastpath in baselines/fasttree.py:
#   virtual_root → B prompt-roots (seqlen = L_p_tokens, K children) → B*K leaves.
# Each leaf has seqlen = current_pos % page_size + 1 (here: 1 token, step 0).
# ---------------------------------------------------------------------------

def build_synthetic_tree(K: int, B: int, L_p_tokens: int, leaf_seqlen: int):
    nodes: list[KVTreeNode] = []
    virtual = KVTreeNode()
    virtual.parent = -1
    virtual.id = 0
    virtual.seqlen = 0
    virtual.num_children = B
    virtual.requests = []
    nodes.append(virtual)

    for b in range(B):
        root_id = len(nodes)
        root = KVTreeNode()
        root.parent = 0
        root.id = root_id
        root.seqlen = L_p_tokens
        root.num_children = K
        root.requests = list(range(b * K, (b + 1) * K))
        nodes.append(root)
        for k in range(K):
            leaf = KVTreeNode()
            leaf.parent = root_id
            leaf.id = len(nodes)
            leaf.seqlen = leaf_seqlen
            leaf.num_children = 0
            leaf.requests = [b * K + k]
            nodes.append(leaf)
    return nodes


# ---------------------------------------------------------------------------
# Manual port of _compute_parallelism (kernel_bench/fasttree.py:126), exposed
# so we can print intermediate state across the 3-iter refinement loop.
# ---------------------------------------------------------------------------

def compute_parallelism(tree_info, node_assignments, Q_TILE_PER_PHASE,
                        KV_SPLIT_PER_PHASE, num_kv_heads):
    import queue as _queue
    node_num = len(tree_info)
    node_to_reqs = [[] for _ in range(node_num)]
    q = _queue.Queue()
    for i in range(node_num):
        if tree_info[i].num_children == 0:
            q.put(i)
            node_to_reqs[i] = list(tree_info[i].requests)
    virtual_children = [tree_info[n].num_children for n in range(node_num)]
    while not q.empty():
        node = q.get()
        if node_assignments[node] == 0 and node != 0:
            node_to_reqs[tree_info[node].parent] += tree_info[node].requests
        if tree_info[node].parent != -1:
            virtual_children[tree_info[node].parent] -= 1
            if virtual_children[tree_info[node].parent] == 0:
                q.put(tree_info[node].parent)
    parallelisms = [0, 0]
    per_phase_vnodes = [0, 0]
    for i in range(node_num):
        req_num = len(node_to_reqs[i])
        if req_num == 0:
            continue
        node = i
        kv_len = tree_info[i].seqlen
        while node_assignments[node] == 1:
            node = tree_info[node].parent
            kv_len += tree_info[node].seqlen
        phase = 0 if req_num > Q_TILE_PER_PHASE[1] else 1
        q_vnode_count = (req_num - 1) // Q_TILE_PER_PHASE[phase] + 1
        kv_vnode_count = (kv_len - 1) // KV_SPLIT_PER_PHASE[phase] + 1
        parallelisms[phase] += kv_vnode_count * q_vnode_count
        per_phase_vnodes[phase] += kv_vnode_count * q_vnode_count
    parallelisms = [p * num_kv_heads for p in parallelisms]
    return parallelisms, per_phase_vnodes, node_to_reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--L_p", type=int, default=30000)
    ap.add_argument("--leaf_seqlen", type=int, default=1,
                    help="tail tokens at first decode step")
    ap.add_argument("--num_qo_heads", type=int, default=32)
    ap.add_argument("--num_kv_heads", type=int, default=8,
                    help="Llama-3.2-1B has 8 kv_heads; gqa_group=4")
    ap.add_argument("--KV_SPLIT_SIZES", nargs=2, type=int, default=[1024, 128])
    ap.add_argument("--para_threshs1", nargs=2, type=int, default=[132, 528])
    ap.add_argument("--para_threshs2", nargs=2, type=int, default=[132, 132])
    args = ap.parse_args()

    K, B, L_p = args.K, args.B, args.L_p
    print(f"=== FastTree plan decisions for K={K} B={B} L_p={L_p} ===")
    print(f"GQA group size = {args.num_qo_heads // args.num_kv_heads} "
          f"(num_qo={args.num_qo_heads}, num_kv={args.num_kv_heads})")
    print(f"KV_SPLIT_SIZES = {args.KV_SPLIT_SIZES} "
          f"(large fallback / small fallback)")
    print(f"para_threshs1  = {args.para_threshs1} "
          f"(target CTA count per phase to trigger small KV split)")
    print(f"para_threshs2  = {args.para_threshs2}\n")

    tree = build_synthetic_tree(K, B, L_p, args.leaf_seqlen)
    print(f"Tree shape: 1 virtual root + {B} prompt-roots + {B*K} leaves "
          f"= {len(tree)} nodes")
    print(f"  prompt-root seqlen = {L_p} tokens (shared prefix)")
    print(f"  leaf seqlen        = {args.leaf_seqlen} token(s) (per-beam tail)\n")

    # ----- Run the 3-iter refinement loop manually, printing each step. -----
    params = FastTreeParams()
    params.set_kv_group_num(args.num_qo_heads // args.num_kv_heads)
    Q_TILE = list(params.TSQs)
    KV_SPLIT = [args.KV_SPLIT_SIZES[0], args.KV_SPLIT_SIZES[0]]
    print(f"Initial Q_TILE_SIZE_PER_PHASE = {Q_TILE} (TSQs default)")
    print(f"Initial KV_SPLIT_SIZE_PER_PHASE = {KV_SPLIT} "
          f"(both start at large_split={args.KV_SPLIT_SIZES[0]})\n")

    node_assignments = None
    final_phase_vnodes = (0, 0)
    for it in range(3):
        node_assignments = _tree_heuristic(tree, params)
        para, per_phase_vnodes, node_to_reqs = compute_parallelism(
            tree, node_assignments, Q_TILE, KV_SPLIT, args.num_kv_heads,
        )
        final_phase_vnodes = tuple(per_phase_vnodes)
        print(f"--- refinement iter {it} ---")
        print(f"  Q_TILE_SIZE_PER_PHASE  = {Q_TILE}")
        print(f"  KV_SPLIT_SIZE_PER_PHASE = {KV_SPLIT}")
        print(f"  parallelism (CTAs)     = {para}    "
              f"(phase 0 = nQ>{Q_TILE[1]}, phase 1 = nQ<={Q_TILE[1]})")
        print(f"  vnodes per phase       = {per_phase_vnodes}")
        if it == 0:
            done = True
            for phase in range(2):
                if 0 < para[phase] < args.para_threshs1[phase]:
                    KV_SPLIT[phase] = args.KV_SPLIT_SIZES[1]
                    done = False
                    print(f"  -> phase {phase} below threshs1 "
                          f"({para[phase]} < {args.para_threshs1[phase]}): "
                          f"shrink KV_SPLIT[{phase}] -> {args.KV_SPLIT_SIZES[1]}")
            if done:
                print("  -> both phases above threshs1: STOP")
                break
        elif it == 1:
            if 0 < para[0] < args.para_threshs2[0]:
                Q_TILE = [Q_TILE[1]] * 2
                params.set_q_tile_sizes(Q_TILE)
                params.set_kv_tile_sizes([params.TSKs[1]] * 2)
                print(f"  -> phase 0 below threshs2: fold both phases to "
                      f"small Q tile = {Q_TILE[0]}")
            elif 0 < para[1] < args.para_threshs2[1]:
                Q_TILE = [Q_TILE[0]] * 2
                params.set_q_tile_sizes(Q_TILE)
                params.set_kv_tile_sizes([params.TSKs[0]] * 2)
                print(f"  -> phase 1 below threshs2: fold both phases to "
                      f"large Q tile = {Q_TILE[0]}")
            else:
                print("  -> both phases above threshs2: STOP")
                break

    print()
    print(f"FINAL Q_TILE_PER_PHASE  = {Q_TILE}")
    print(f"FINAL KV_SPLIT_PER_PHASE = {KV_SPLIT}")
    print(f"FINAL params.TSKs       = {params.TSKs}")
    print()

    # ----- Per-node split-K / split-Q decision -----
    counts = Counter()
    for i, asg in enumerate(node_assignments):
        n = tree[i]
        kind = "ROOT" if n.parent == -1 else (
            "PROMPT_ROOT" if n.parent == 0 else "LEAF"
        )
        counts[(kind, asg)] += 1
    print(f"node_assignments (0 = split-K, indep CTA; 1 = split-Q, merge with parent):")
    for (kind, asg), c in sorted(counts.items()):
        print(f"  {kind:<12} assignment={asg}: {c} nodes")
    print()

    # ----- Stage 1 grid layout: per-vnode (q_len, kv_len) and totals -----
    Q_TILE_FINAL = Q_TILE[0]
    print("Stage 1 CTA layout (after final tile/split choice):")
    print(f"  per CTA: Q_BLOCK_SIZE = Q_TILE × gqa_group = "
          f"{Q_TILE_FINAL} × {args.num_qo_heads // args.num_kv_heads} = "
          f"{Q_TILE_FINAL * (args.num_qo_heads // args.num_kv_heads)} q-rows")
    print()

    # Walk the final assignment and emit per-vnode shape buckets.
    vnode_shapes = []
    for i in range(len(tree)):
        req_num = len(node_to_reqs[i])
        if req_num == 0:
            continue
        phase = 0 if req_num > Q_TILE[1] else 1
        kv_split = KV_SPLIT[phase]
        node = i
        kv_len = tree[i].seqlen
        while node_assignments[node] == 1:
            node = tree[node].parent
            kv_len += tree[node].seqlen
        kv_vnode_count = (kv_len - 1) // kv_split + 1
        q_vnode_count = (req_num - 1) // Q_TILE[phase] + 1
        vnode_shapes.append({
            "node": i, "kind": "PROMPT_ROOT" if tree[i].parent == 0
                      else ("LEAF" if tree[i].num_children == 0 else "OTHER"),
            "phase": phase, "req_num": req_num, "kv_len": kv_len,
            "q_vnode_count": q_vnode_count,
            "kv_vnode_count": kv_vnode_count,
            "total_vnodes": q_vnode_count * kv_vnode_count,
        })

    # Aggregate by (kind, phase).
    agg: dict = {}
    for v in vnode_shapes:
        key = (v["kind"], v["phase"], v["req_num"], v["kv_len"],
               v["q_vnode_count"], v["kv_vnode_count"])
        agg[key] = agg.get(key, 0) + 1
    print(f"vnode bucket summary "
          f"(kind, phase, req_num, kv_len, q_vnodes, kv_vnodes) -> n_nodes:")
    for key, count in sorted(agg.items()):
        kind, phase, rn, kvl, qv, kvv = key
        tot = qv * kvv * count
        print(f"  {kind:<12} phase={phase} req={rn:<3} kv={kvl:<6} "
              f"q_vnodes={qv} kv_vnodes={kvv:<3} -> "
              f"{count} such nodes, {tot} vnodes")

    # Total stage-1 CTAs.
    total_vnodes = sum(v["total_vnodes"] for v in vnode_shapes)
    stage1_ctas = total_vnodes * args.num_kv_heads
    print()
    print(f"Total vnodes (stage 1 vnode count)  = {total_vnodes}")
    print(f"Stage 1 grid CTAs = vnodes × num_kv_heads "
          f"= {total_vnodes} × {args.num_kv_heads} = {stage1_ctas}")
    print(f"  ({stage1_ctas} CTAs / 132 SMs on H100 "
          f"= {stage1_ctas / 132:.2f} waves)")

    # Phase split (mirrors metadata reorder above/below threshold).
    threshold = Q_TILE[1]
    above = sum(1 for v in vnode_shapes for _ in range(v["total_vnodes"])
                if v["req_num"] > threshold)
    # Approx: vnode_to_q_lens[i] is the q size of each emitted vnode
    # (req_num for whole node, capped at Q_TILE for split). Phase split
    # is by q_len > threshold, so q_vnode_count==1 with req_num<=threshold
    # goes below. Easier: rely on req_num as proxy (all q_vnodes inherit it).
    print(f"\nStage 1 phase ordering: above-threshold ({threshold}) vnodes "
          f"go first")
    above_vnodes = sum(v["total_vnodes"] for v in vnode_shapes
                       if v["req_num"] > threshold)
    below_vnodes = sum(v["total_vnodes"] for v in vnode_shapes
                       if v["req_num"] <= threshold)
    print(f"  phase_node_nums = [{above_vnodes}, {below_vnodes}]  "
          f"(launched as 2 stage-1 kernel grids)")


if __name__ == "__main__":
    main()
