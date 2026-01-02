"""
Test file for BeamState.get_cascade_input() using the exact example from cascade.md
"""

import torch
import sys
import time
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src" / "beam_engine"
sys.path.insert(0, str(src_path.parent))

from beam_engine.page_table import PageTable
from beam_engine.beam_state import BeamState, BeamCandidate, TrieNode


def build_cascade_example_tree(beam_state: BeamState):
    """
    Manually build the exact tree structure from cascade.md

    The structure has:
    - Level 0: Shared root (pages 0-3) with 9 leaf sequences
    - Level 1: Three branches
      - Branch 1 (pages 4-5): 4 leaf sequences
      - Branch 2 (pages 6-7): 2 leaf sequences
      - Branch 3 (pages 8-9): 3 leaf sequences
    - Level 2: Individual endings for each sequence (pages 10-15)

    8 total input sequences producing 9 output tokens (one sequence ends with 2 tokens)
    """

    # Allocate pages manually to match the expected page IDs
    # Level 0: Root shared prefix (pages 0-3)
    # "<s> As you come into this world, something else is also born"

    # Create root node chain (Level 0)
    root_0 = TrieNode(tokens=[1, 2, 3, 4], page_id=0, parent=None)  # <s> As you come
    root_1 = TrieNode(tokens=[5, 6, 7, 8], page_id=1, parent=root_0)  # into this world,
    root_2 = TrieNode(tokens=[9, 10, 11, 12], page_id=2, parent=root_1)  # something else is also
    root_3 = TrieNode(tokens=[13], page_id=3, parent=root_2)  # born

    beam_state.root = root_0

    # Level 1: Three branches
    # Branch 1: "-- a connection to the" (pages 4-5)

    branch1_0 = TrieNode(tokens=[14, 15, 16, 17], page_id=4, parent=root_3)  # -- a connection to
    branch1_1 = TrieNode(tokens=[18], page_id=5, parent=branch1_0)  # the

    # Branch 2: "-- an echo of your" (pages 6-7)

    branch2_0 = TrieNode(tokens=[19, 20, 21,22], page_id=6, parent=root_3)  # -- an echo of
    branch2_1 = TrieNode(tokens=[23], page_id=7, parent=branch2_0)  # your

    # Branch 3: ". You begin your life," (pages 8-9)

    branch3_0 = TrieNode(tokens=[24, 25, 26, 27], page_id=8, parent=root_3)  # . You begin your
    branch3_1 = TrieNode(tokens=[28], page_id=9, parent=branch3_0)  # life,

    # Level 2: Individual endings
    # Under branch 1:
    leaf_0 = TrieNode(tokens=[29, 51], page_id=10, parent=branch1_1)  # unseen forces -> "forces"

    leaf_1_0 = TrieNode(tokens=[30, 31, 32, 33], page_id=11, parent=branch1_1)  # people who will shape
    leaf_1_1 = TrieNode(tokens=[34, 52], page_id=12, parent=leaf_1_0)  # your journey -> "journey"

    # "infinite possibilities" - appends to branch1_1 (no new pages)
    leaf_2 = TrieNode(tokens=[35, 36], page_id=13, parent=branch1_1)  # infinite possibilities -> "possibilities"

    # Under branch 2:
    leaf_3 = TrieNode(tokens=[37], page_id=14, parent=branch2_1)  # existence

    leaf_4 = TrieNode(tokens=[38], page_id=15, parent=branch2_1)  # essence
    # Also reuses page_7 - this creates a problem since they can't both reuse the same page
    # Let me reconsider the structure...

    # Actually, looking at cascade.md more carefully, the kv_page_indices for level 2 are [10, 11, 12, 13, 14, 15]
    # So there should be pages 13, 14, 15 as well

    leaf_5 = TrieNode(tokens=[39, 40, 41, 42], page_id=16, parent=branch3_1)  # it begins a journey

    leaf_6 = TrieNode(tokens=[43, 44, 45, 46], page_id=17, parent=branch3_1)  # and in that moment

    leaf_7 = TrieNode(tokens=[47, 48, 49, 50], page_id=18, parent=branch3_1)  # and the world around you

    # Now create beam candidates for each leaf
    # Based on cascade.md, there are 8 sequences (0-7) but 9 output tokens
    # Sequence 2 produces 2 tokens (infinite, possibilities)

    beam_state.candidates = [
        BeamCandidate(trie_node=leaf_0, score=0.0),  # Seq 0: forces
        BeamCandidate(trie_node=leaf_1_1, score=0.0),  # Seq 1: journey
        BeamCandidate(trie_node=leaf_2, score=0.0),  # Seq 2: possibilities
        BeamCandidate(trie_node=leaf_3, score=0.0),  # Seq 3: existence
        BeamCandidate(trie_node=leaf_4, score=0.0),  # Seq 4: essence
        BeamCandidate(trie_node=leaf_5, score=0.0),  # Seq 5: it begins a journey
        BeamCandidate(trie_node=leaf_6, score=0.0),  # Seq 6: and in that moment
        BeamCandidate(trie_node=leaf_7, score=0.0),  # Seq 7: and the world around you
    ]


def test_cascade_input():
    """Test get_cascade_input with the exact example from cascade.md"""

    print("=" * 80)
    print("Testing BeamState.get_cascade_input() with cascade.md example")
    print("=" * 80)

    # Create page table with page_size=4 to match cascade.md
    page_table = PageTable(
        layer_num=1,
        page_size=4,
        max_num_pages=20,
        head_num=32,
        head_dim=128,
        device=torch.device('cpu'),
        store_dtype=torch.float16
    )

    # Create beam state
    beam_state = BeamState(beam_size=8, page_table=page_table)

    # Build the example tree structure
    build_cascade_example_tree(beam_state)

    print(f"\nCreated tree with {len(beam_state.candidates)} candidates")
    print(f"Root: {beam_state.root}")

    # Call get_cascade_input with timing
    start_time = time.perf_counter()
    (qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr,
     paged_kv_last_page_len, q) = beam_state.get_cascade_input()
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"\nget_cascade_input() took {elapsed_ms:.4f} ms")
    print(f"Number of cascade levels: {len(qo_indptr_arr)}")

    # Print raw outputs for inspection
    print("\n" + "=" * 80)
    print("RAW OUTPUTS:")
    print("=" * 80)
    for level_idx in range(len(qo_indptr_arr)):
        print(f"\n--- Level {level_idx} ---")
        print(f"qo_indptr:             {qo_indptr_arr[level_idx].tolist()}")
        print(f"paged_kv_indptr:       {paged_kv_indptr_arr[level_idx].tolist()}")
        print(f"paged_kv_indices:      {paged_kv_indices_arr[level_idx].tolist()}")
        print(f"paged_kv_last_page_len: {paged_kv_last_page_len[level_idx].tolist()}")

    # Verify the outputs
    verify_cascade_output(beam_state, qo_indptr_arr, paged_kv_indptr_arr,
                         paged_kv_indices_arr, paged_kv_last_page_len, q)


def build_cascade_swapped_tree(beam_state: BeamState):
    """
    Build tree with swapped branch construction order.

    Swaps Branch 1 and Branch 3 construction order:
    - Original: Branch1 (pages 4-5), Branch2 (pages 6-7), Branch3 (pages 8-9)
    - Swapped:  Branch3 (pages 4-5), Branch2 (pages 6-7), Branch1 (pages 8-9)
    """

    # Level 0: Root shared prefix (pages 0-3) - SAME
    root_0 = TrieNode(tokens=[1, 2, 3, 4], page_id=0, parent=None)
    root_1 = TrieNode(tokens=[5, 6, 7, 8], page_id=1, parent=root_0)
    root_2 = TrieNode(tokens=[9, 10, 11, 12], page_id=2, parent=root_1)
    root_3 = TrieNode(tokens=[13], page_id=3, parent=root_2)

    beam_state.root = root_0

    # Level 1: Build Branch 3 FIRST (now gets pages 4-5)
    # Branch 3: ". You begin your life," (pages 4-5)
    branch3_0 = TrieNode(tokens=[24, 25, 26, 27], page_id=4, parent=root_3)
    branch3_1 = TrieNode(tokens=[28], page_id=5, parent=branch3_0)

    # Branch 2: "-- an echo of your" (pages 6-7) - SAME
    branch2_0 = TrieNode(tokens=[19, 20, 21, 22], page_id=6, parent=root_3)
    branch2_1 = TrieNode(tokens=[23], page_id=7, parent=branch2_0)

    # Branch 1: "-- a connection to the" (now gets pages 8-9)
    branch1_0 = TrieNode(tokens=[14, 15, 16, 17], page_id=8, parent=root_3)
    branch1_1 = TrieNode(tokens=[18], page_id=9, parent=branch1_0)

    # Level 2: Individual endings
    # Under branch 3 (now pages 4-5):
    leaf_5 = TrieNode(tokens=[39, 40, 41, 42], page_id=10, parent=branch3_1)
    leaf_6 = TrieNode(tokens=[43, 44, 45, 46], page_id=11, parent=branch3_1)
    leaf_7 = TrieNode(tokens=[47, 48, 49, 50], page_id=12, parent=branch3_1)

    # Under branch 2 (still pages 6-7):
    leaf_3 = TrieNode(tokens=[37], page_id=13, parent=branch2_1)
    leaf_4 = TrieNode(tokens=[38], page_id=14, parent=branch2_1)

    # Under branch 1 (now pages 8-9):
    leaf_0 = TrieNode(tokens=[29, 51], page_id=15, parent=branch1_1)
    leaf_1_0 = TrieNode(tokens=[30, 31, 32, 33], page_id=16, parent=branch1_1)
    leaf_1_1 = TrieNode(tokens=[34, 52], page_id=17, parent=leaf_1_0)
    leaf_2 = TrieNode(tokens=[35, 36], page_id=18, parent=branch1_1)

    # Candidates in same order as original (Seq 0-7)
    beam_state.candidates = [
        BeamCandidate(trie_node=leaf_0, score=0.0),   # Seq 0: forces
        BeamCandidate(trie_node=leaf_1_1, score=0.0), # Seq 1: journey
        BeamCandidate(trie_node=leaf_2, score=0.0),   # Seq 2: possibilities
        BeamCandidate(trie_node=leaf_3, score=0.0),   # Seq 3: existence
        BeamCandidate(trie_node=leaf_4, score=0.0),   # Seq 4: essence
        BeamCandidate(trie_node=leaf_5, score=0.0),   # Seq 5: it begins a journey
        BeamCandidate(trie_node=leaf_6, score=0.0),   # Seq 6: and in that moment
        BeamCandidate(trie_node=leaf_7, score=0.0),   # Seq 7: and the world around you
    ]


def verify_cascade_output(beam_state, qo_indptr_arr, paged_kv_indptr_arr,
                         paged_kv_indices_arr, paged_kv_last_page_len, q):
    """
    Verify that cascade outputs correctly represent each candidate's page sequence and query token.

    For each candidate:
    1. Get expected page sequence by walking from leaf to root
    2. Reconstruct page sequence from cascade outputs
    3. Verify they match
    4. Verify query token matches the last token in the candidate's leaf node
    """
    print("\n" + "=" * 80)
    print("VERIFICATION: Reconstructing page sequences for each candidate")
    print("=" * 80)

    all_match = True

    for cand_idx, candidate in enumerate(beam_state.candidates):
        # Get expected page sequence by walking up the trie
        expected_pages = []
        node = candidate.trie_node
        while node is not None:
            expected_pages.insert(0, node.page_id)
            node = node.parent

        # Get expected query token (last token in leaf node)
        expected_query_token = candidate.trie_node.tokens[-1]

        # Reconstruct page sequence from cascade outputs
        reconstructed_pages = []

        for level_idx in range(len(qo_indptr_arr)):
            qo_indptr = qo_indptr_arr[level_idx].tolist()
            kv_indptr = paged_kv_indptr_arr[level_idx].tolist()
            kv_indices = paged_kv_indices_arr[level_idx].tolist()

            # Find which group this candidate belongs to at this level
            group_idx = None
            for i in range(len(qo_indptr) - 1):
                if qo_indptr[i] <= cand_idx < qo_indptr[i + 1]:
                    group_idx = i
                    break

            if group_idx is None:
                print(f"ERROR: Could not find group for candidate {cand_idx} at level {level_idx}")
                all_match = False
                continue

            # Get pages for this group
            page_start = kv_indptr[group_idx]
            page_end = kv_indptr[group_idx + 1]
            group_pages = kv_indices[page_start:page_end]

            # Add to reconstructed pages
            reconstructed_pages.extend(group_pages)

        # Compare
        pages_match = reconstructed_pages == expected_pages
        query_match = q[cand_idx].item() == expected_query_token

        status = "✓" if (pages_match and query_match) else "✗"

        print(f"\n{status} Candidate {cand_idx}:")
        print(f"  Expected pages:      {expected_pages}")
        print(f"  Reconstructed pages: {reconstructed_pages}")
        print(f"  Pages match: {'YES' if pages_match else 'NO'}")
        print(f"  Expected query token:  {expected_query_token}")
        print(f"  Actual query token:    {q[cand_idx].item()}")
        print(f"  Query match: {'YES' if query_match else 'NO'}")

        if not pages_match or not query_match:
            all_match = False

    print("\n" + "=" * 80)
    if all_match:
        print("✓ ALL CANDIDATES VERIFIED SUCCESSFULLY!")
    else:
        print("✗ VERIFICATION FAILED FOR SOME CANDIDATES")
    print("=" * 80)

    return all_match


def test_cascade_input_swapped():
    """Test get_cascade_input with swapped branch construction order"""

    print("=" * 80)
    print("Testing BeamState.get_cascade_input() with SWAPPED branch order")
    print("=" * 80)

    # Create page table with page_size=4 to match cascade.md
    page_table = PageTable(
        layer_num=1,
        page_size=4,
        max_num_pages=20,
        head_num=32,
        head_dim=128,
        device=torch.device('cpu'),
        store_dtype=torch.float16
    )

    # Create beam state
    beam_state = BeamState(beam_size=8, page_table=page_table)

    # Build the swapped tree structure
    build_cascade_swapped_tree(beam_state)

    print(f"\nCreated tree with {len(beam_state.candidates)} candidates")
    print(f"Root: {beam_state.root}")

    # Call get_cascade_input with timing
    start_time = time.perf_counter()
    (qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr,
     paged_kv_last_page_len, q) = beam_state.get_cascade_input()
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"\nget_cascade_input() took {elapsed_ms:.4f} ms")
    print(f"Number of cascade levels: {len(qo_indptr_arr)}")

    # Print raw outputs for inspection
    print("\n" + "=" * 80)
    print("RAW OUTPUTS:")
    print("=" * 80)
    for level_idx in range(len(qo_indptr_arr)):
        print(f"\n--- Level {level_idx} ---")
        print(f"qo_indptr:             {qo_indptr_arr[level_idx].tolist()}")
        print(f"paged_kv_indptr:       {paged_kv_indptr_arr[level_idx].tolist()}")
        print(f"paged_kv_indices:      {paged_kv_indices_arr[level_idx].tolist()}")
        print(f"paged_kv_last_page_len: {paged_kv_last_page_len[level_idx].tolist()}")

    # Verify the outputs
    verify_cascade_output(beam_state, qo_indptr_arr, paged_kv_indptr_arr,
                         paged_kv_indices_arr, paged_kv_last_page_len, q)


if __name__ == "__main__":
    test_cascade_input()
    print("\n\n")
    test_cascade_input_swapped()
