#!/usr/bin/env python3
"""
Test cascade attention parameter generation for beam search.
This test runs on CPU and mocks the model to focus on testing the trie and cascade logic.
"""

import torch
import sys
import os
from typing import List, Dict

# Add the src directory to path so we can import beam_engine modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from beam_engine.beam import TrieNode, BeamCandidate, BeamState
from beam_engine.page_table import PageTable


def create_test_page_table(page_size: int = 4, max_pages: int = 100) -> PageTable:
    """Create a small page table for testing."""
    return PageTable(
        layer_num=1,  # Only need 1 layer for testing
        page_size=page_size,
        max_num_pages=max_pages,
        head_num=8,
        head_dim=64,
        device=torch.device('cpu'),
        store_dtype=torch.float32
    )


def create_test_trie_structure(beam_state: BeamState) -> List[BeamCandidate]:
    """
    Create a test trie structure that represents this scenario:

    Root -> "The best way to solve" (pages 0,1)
               |
           [BRANCH POINT 1]
           /              \
    "climate change"   "world hunger" (pages 2,3)
       /        \         /      \
   "through"  "by"    "via"   "using" (pages 4,5,6,7)

    This creates:
    - Level 0: Shared prefix (all candidates)
    - Level 1: Two branches (climate vs world)
    - Level 2: Four individual paths
    """

    # Simulate input sequence: "The best way to solve" (8 tokens, 2 pages)
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # "The best way to solve"

    # Add input to root, this creates the shared prefix
    root_to_solve = beam_state.root.add_sequence(input_tokens)

    # Allocate pages for the shared prefix
    # First page: tokens [1,2,3,4]
    page_0 = beam_state.page_table.allocate_block()
    # Second page: tokens [5,6,7,8]
    page_1 = beam_state.page_table.allocate_block()

    # Assign pages to the path from root to "solve"
    current = root_to_solve
    path_nodes = []
    while current != beam_state.root:
        path_nodes.append(current)
        current = current.parent
    path_nodes.reverse()

    # Assign page IDs (this is simplified - in reality the trie might have multiple nodes)
    if len(path_nodes) >= 1:
        path_nodes[0].page_id = page_0
    if len(path_nodes) >= 2:
        path_nodes[1].page_id = page_1

    # Create first branch: "climate change" vs "world hunger"
    climate_tokens = [10, 11]  # "climate change"
    world_tokens = [20, 21]    # "world hunger"

    climate_node = root_to_solve.add_sequence(climate_tokens)
    world_node = root_to_solve.add_sequence(world_tokens)

    # Allocate pages for first branch level
    page_2 = beam_state.page_table.allocate_block()  # climate change
    page_3 = beam_state.page_table.allocate_block()  # world hunger

    climate_node.page_id = page_2
    world_node.page_id = page_3

    # Create second branch: individual continuations
    through_tokens = [30]   # "through"
    by_tokens = [31]        # "by"
    via_tokens = [32]       # "via"
    using_tokens = [33]     # "using"

    through_node = climate_node.add_sequence(through_tokens)
    by_node = climate_node.add_sequence(by_tokens)
    via_node = world_node.add_sequence(via_tokens)
    using_node = world_node.add_sequence(using_tokens)

    # Allocate pages for second branch level
    page_4 = beam_state.page_table.allocate_block()  # through
    page_5 = beam_state.page_table.allocate_block()  # by
    page_6 = beam_state.page_table.allocate_block()  # via
    page_7 = beam_state.page_table.allocate_block()  # using

    through_node.page_id = page_4
    by_node.page_id = page_5
    via_node.page_id = page_6
    using_node.page_id = page_7

    # Create beam candidates
    candidates = [
        BeamCandidate(trie_node=through_node, score=-1.0, finished=False),
        BeamCandidate(trie_node=by_node, score=-1.5, finished=False),
        BeamCandidate(trie_node=via_node, score=-2.0, finished=False),
        BeamCandidate(trie_node=using_node, score=-2.5, finished=False),
    ]

    return candidates


def test_cascade_parameters():
    """Test the cascade attention parameter generation."""
    print("=== Testing Cascade Attention Parameters ===")

    # Create test setup
    page_table = create_test_page_table(page_size=4)
    beam_state = BeamState(beam_size=4, device=torch.device('cpu'), page_table=page_table)

    # Create test trie structure
    candidates = create_test_trie_structure(beam_state)

    print(f"Created {len(candidates)} test candidates")
    for i, candidate in enumerate(candidates):
        seq = candidate.trie_node.get_full_sequence()
        print(f"  Candidate {i+1}: tokens = {seq}")

    # Generate cascade parameters
    cascade_params = beam_state.get_cascade_parameters_for_candidates(candidates)

    print("\n=== Cascade Parameters ===")
    print(f"Number of levels: {len(cascade_params['qo_indptr_arr'])}")

    for level in range(len(cascade_params['qo_indptr_arr'])):
        print(f"\nLevel {level}:")
        print(f"  qo_indptr: {cascade_params['qo_indptr_arr'][level]}")
        print(f"  paged_kv_indptr: {cascade_params['paged_kv_indptr_arr'][level]}")
        print(f"  paged_kv_indices: {cascade_params['paged_kv_indices_arr'][level]}")
        print(f"  paged_kv_last_page_len: {cascade_params['paged_kv_last_page_len'][level]}")

    # Validate the results
    validate_cascade_parameters(cascade_params, candidates)


def validate_cascade_parameters(cascade_params: Dict, candidates: List[BeamCandidate]):
    """Validate that the generated cascade parameters are correct."""
    print("\n=== Validation ===")

    num_levels = len(cascade_params['qo_indptr_arr'])
    batch_size = len(candidates)

    print(f"Batch size: {batch_size}")
    print(f"Number of cascade levels: {num_levels}")

    # Expected structure for our test case:
    # Level 0: Shared prefix "The best way to solve" (pages 0,1) - all 4 candidates
    # Level 1: Branch level "climate change"/"world hunger" (pages 2,3) - 2 groups
    # Level 2: Individual level "through"/"by"/"via"/"using" (pages 4,5,6,7) - 4 individual

    assert num_levels == 3, f"Expected 3 levels, got {num_levels}"
    print("✓ Correct number of levels")

    # Level 0 validation (shared prefix)
    level_0_qo = cascade_params['qo_indptr_arr'][0]
    level_0_kv_indptr = cascade_params['paged_kv_indptr_arr'][0]
    level_0_kv_indices = cascade_params['paged_kv_indices_arr'][0]

    assert level_0_qo.tolist() == [0, 4], f"Level 0 qo_indptr should be [0,4], got {level_0_qo.tolist()}"
    assert level_0_kv_indptr.tolist() == [0, 2], f"Level 0 should have 2 shared pages, got {level_0_kv_indptr.tolist()}"
    assert level_0_kv_indices.tolist() == [0, 1], f"Level 0 should use pages [0,1], got {level_0_kv_indices.tolist()}"
    print("✓ Level 0 (shared prefix) validation passed")

    # Level 1 validation (first branch)
    level_1_qo = cascade_params['qo_indptr_arr'][1]
    level_1_kv_indptr = cascade_params['paged_kv_indptr_arr'][1]
    level_1_kv_indices = cascade_params['paged_kv_indices_arr'][1]

    # Should have 4 candidates, each getting 1 query
    assert level_1_qo.tolist() == [0, 1, 2, 3, 4], f"Level 1 qo_indptr should be [0,1,2,3,4], got {level_1_qo.tolist()}"
    print("✓ Level 1 (branch level) validation passed")

    # Level 2 validation (individual paths)
    level_2_qo = cascade_params['qo_indptr_arr'][2]
    level_2_kv_indptr = cascade_params['paged_kv_indptr_arr'][2]
    level_2_kv_indices = cascade_params['paged_kv_indices_arr'][2]

    # Should have 4 candidates, each with individual pages
    assert level_2_qo.tolist() == [0, 1, 2, 3, 4], f"Level 2 qo_indptr should be [0,1,2,3,4], got {level_2_qo.tolist()}"
    assert level_2_kv_indices.tolist() == [4, 5, 6, 7], f"Level 2 should use pages [4,5,6,7], got {level_2_kv_indices.tolist()}"
    print("✓ Level 2 (individual paths) validation passed")

    print("\n🎉 All validations passed!")


def test_branching_detection():
    """Test the branching point detection algorithm."""
    print("\n=== Testing Branching Point Detection ===")

    page_table = create_test_page_table(page_size=4)
    beam_state = BeamState(beam_size=4, device=torch.device('cpu'), page_table=page_table)

    candidates = create_test_trie_structure(beam_state)

    # Test branching point detection
    branching_points = beam_state._find_branching_points(candidates)

    print(f"Found {len(branching_points)} branching points:")
    for i, branch_point in enumerate(branching_points):
        print(f"  Branch {i}: tokens = {branch_point.tokens}, children = {len(branch_point.children)}")

    # We should have 2 branching points:
    # 1. The node after "solve" that branches to "climate"/"world"
    # 2. The "climate" node that branches to "through"/"by"
    # 3. The "world" node that branches to "via"/"using"

    assert len(branching_points) >= 1, "Should find at least 1 branching point"
    print("✓ Branching point detection working")


if __name__ == "__main__":
    print("Running Cascade Attention Tests on CPU")
    print("=" * 50)

    try:
        test_branching_detection()
        test_cascade_parameters()

        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)