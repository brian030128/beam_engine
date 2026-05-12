"""Differential test for the fast-path in _build_radix_tree_pages_single.

Calls the builder with BE_FT_FASTPATH_RADIX=0 (slow recursive walk) and
again with =1 (fast path enabled when all-K-distinct at d=0), and
asserts the outputs are structurally identical across random inputs.

Covers:
  * Various K (1, 2, 4, 16, 64)
  * Tails with all-distinct first pages (fast path triggers)
  * Tails with first-page sharing (must fall back to slow path)
  * Beams with shared tail prefix beyond d=0 (fall back)
  * K=1 (must NOT take fast path — semantics differ for singleton)
"""

from __future__ import annotations

import os
import random
import sys


def _call(shared_prefix, tails, fastpath):
    os.environ["BE_FT_FASTPATH_RADIX"] = "1" if fastpath else "0"
    from beam_engine.baselines.fasttree import _build_radix_tree_pages_single
    return _build_radix_tree_pages_single(shared_prefix, tails)


def _node_to_tuple(n):
    return (n.parent, n.id, n.seqlen, n.num_children, tuple(n.requests))


def _equal(a, b) -> bool:
    nodes_a, pages_a = a
    nodes_b, pages_b = b
    if len(nodes_a) != len(nodes_b):
        return False
    if len(pages_a) != len(pages_b):
        return False
    for na, nb in zip(nodes_a, nodes_b):
        if _node_to_tuple(na) != _node_to_tuple(nb):
            return False
    for pa, pb in zip(pages_a, pages_b):
        if list(pa) != list(pb):
            return False
    return True


def _gen_case(rng, scenario: str):
    K = rng.choice([1, 2, 4, 16, 64])
    pl = rng.randint(0, 8)
    shared_prefix = list(range(100, 100 + pl))
    tail_len = rng.randint(1, 6)
    next_pid = 1000
    tails: list[list[int]] = []
    if scenario == "all_distinct":
        for _ in range(K):
            t = [next_pid + j for j in range(tail_len)]
            next_pid += tail_len
            tails.append(t)
    elif scenario == "shared_first_page":
        # All beams share the first 1-2 tail pages.
        share = rng.randint(1, min(2, tail_len))
        head = [next_pid + j for j in range(share)]
        next_pid += share
        for _ in range(K):
            t = list(head) + [next_pid + j for j in range(tail_len - share)]
            next_pid += (tail_len - share)
            tails.append(t)
    elif scenario == "partial_groups":
        if K >= 2:
            half = K // 2
            grp0 = [next_pid]; next_pid += 1
            grp1 = [next_pid]; next_pid += 1
            for i in range(K):
                head = grp0 if i < half else grp1
                t = list(head) + [next_pid + j for j in range(tail_len - 1)]
                next_pid += (tail_len - 1)
                tails.append(t)
        else:
            tails = [[next_pid]]
    elif scenario == "single_beam":
        K = 1
        tails = [[next_pid + j for j in range(tail_len)]]
        next_pid += tail_len
    else:
        raise ValueError(scenario)
    return shared_prefix, tails, K


def main():
    rng = random.Random(42)
    scenarios = ["all_distinct", "shared_first_page", "partial_groups", "single_beam"]
    per_scenario = 50

    n_total = 0
    n_fail = 0
    fast_eligible = 0
    fail_examples = []

    for sc in scenarios:
        for trial in range(per_scenario):
            shared_prefix, tails, K = _gen_case(rng, sc)
            try:
                slow = _call(shared_prefix, tails, fastpath=False)
                fast = _call(shared_prefix, tails, fastpath=True)
            except Exception as e:
                n_fail += 1
                fail_examples.append((sc, trial, "exception", str(e),
                                      shared_prefix, tails))
                continue
            n_total += 1
            # Fast path triggers iff K>=2 and slow output is flat (1 root + K leaves)
            if len(slow[0]) == K + 1 and K >= 2:
                fast_eligible += 1
            if not _equal(slow, fast):
                n_fail += 1
                if len(fail_examples) < 5:
                    fail_examples.append((sc, trial, "mismatch", "",
                                          shared_prefix, tails, slow, fast))

    print(f"scenarios: {scenarios}")
    print(f"trials per scenario: {per_scenario}")
    print(f"total compared: {n_total}")
    print(f"  fast-path-would-trigger: {fast_eligible}")
    print(f"  slow-path-required:      {n_total - fast_eligible}")
    print(f"failures: {n_fail}")
    if fail_examples:
        for ex in fail_examples:
            print("\nFAIL example:")
            print(f"  scenario={ex[0]} trial={ex[1]} kind={ex[2]} detail={ex[3]}")
            print(f"  shared_prefix={ex[4]}")
            print(f"  tails={ex[5]}")
            if len(ex) >= 8:
                print(f"  slow nodes:")
                for n in ex[6][0]:
                    print(f"    {_node_to_tuple(n)}")
                print(f"  fast nodes:")
                for n in ex[7][0]:
                    print(f"    {_node_to_tuple(n)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
