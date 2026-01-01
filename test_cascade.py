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

    # Expected values from cascade.md
    expected_qo_indptr = [
        [0, 8],                      # level 0
        [0, 3, 5, 8],               # level 1
        [0, 1, 2, 3, 4, 5, 6, 7, 8] # level 2
    ]

    expected_kv_indptr = [
        [0, 4],                     # level 0 
        [0, 2, 4, 6],               # level 1
        [0, 1, 3, 4, 5, 6, 7, 8, 9] # level 2
    ]

    expected_kv_indices = [
        [0, 1, 2, 3],               # level 0
        [4, 5, 6, 7, 8, 9],         # level 1
        [10, 11, 12, 13, 14, 15, 16, 17, 18]    # level 2
    ]

    expected_kv_last_page_len = [
        [1],                        # level 0
        [1, 1, 1],                  # level 1
        [1, 1, 1, 0, 0, 3, 3, 3]    # level 2
    ]

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    for level_idx in range(len(qo_indptr_arr)):
        print(f"\n--- Level {level_idx} ---")

        print(f"qo_indptr:             {qo_indptr_arr[level_idx].tolist()}")
        print(f"Expected qo_indptr:    {expected_qo_indptr[level_idx]}")
        qo_match = qo_indptr_arr[level_idx].tolist() == expected_qo_indptr[level_idx]
        print(f"Match: {'YES' if qo_match else 'NO'}")

        print(f"\npaged_kv_indptr:       {paged_kv_indptr_arr[level_idx].tolist()}")
        print(f"Expected kv_indptr:    {expected_kv_indptr[level_idx]}")
        kv_indptr_match = paged_kv_indptr_arr[level_idx].tolist() == expected_kv_indptr[level_idx]
        print(f"Match: {'YES' if kv_indptr_match else 'NO'}")

        print(f"\npaged_kv_indices:      {paged_kv_indices_arr[level_idx].tolist()}")
        print(f"Expected kv_indices:   {expected_kv_indices[level_idx]}")
        kv_indices_match = paged_kv_indices_arr[level_idx].tolist() == expected_kv_indices[level_idx]
        print(f"Match: {'YES' if kv_indices_match else 'NO'}")

        print(f"\npaged_kv_last_page_len: {paged_kv_last_page_len[level_idx].tolist()}")
        print(f"Expected last_page_len: {expected_kv_last_page_len[level_idx]}")
        last_len_match = paged_kv_last_page_len[level_idx].tolist() == expected_kv_last_page_len[level_idx]
        print(f"Match: {'YES' if last_len_match else 'NO'}")

    print(f"\n--- Query Token IDs ---")
    print(f"Shape: {q.shape}")
    print(f"Token IDs: {q.tolist()}")

    # Expected query token IDs in cascade order (last token from each candidate's leaf node)
    # Based on the tree structure:
    # - Seq 0 (leaf_0): last token = 51
    # - Seq 1 (leaf_1_1): last token = 52
    # - Seq 2 (leaf_2): last token = 36
    # - Seq 3 (leaf_3): last token = 37
    # - Seq 4 (leaf_4): last token = 38
    # - Seq 5 (leaf_5): last token = 42
    # - Seq 6 (leaf_6): last token = 46
    # - Seq 7 (leaf_7): last token = 50
    expected_query_tokens = [51, 52, 36, 37, 38, 42, 46, 50]

    print(f"Expected token IDs: {expected_query_tokens}")

    # Check shape
    expected_shape = [8]
    shape_match = list(q.shape) == expected_shape
    print(f"Shape match: {'YES' if shape_match else 'NO'}")

    # Check token IDs match expected positions
    tokens_match = q.tolist() == expected_query_tokens
    print(f"Token IDs match: {'YES' if tokens_match else 'NO'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_cascade_input()
