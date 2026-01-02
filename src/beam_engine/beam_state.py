from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import torch
from collections import defaultdict

from page_table import PageTable
class TrieNode:
    """
    Pure data structure for storing beam search sequence tree.

    Contains only data, no logic. All operations are handled by BeamState.
    Each node represents one page in the page table.
    """

    def __init__(self, tokens: List[int], page_id: int, parent: Optional['TrieNode'] = None):
        # Pure data - no logic, no page_size dependency
        self.tokens = tokens  # List of tokens stored in this node
        self.parent = parent  # Parent node reference
        self.children: List['TrieNode'] = [] # Key is first token of child sequence
        self.page_id = page_id  # Page ID in the page table (managed externally)
        if parent:
            parent.children.append(self)

    def __repr__(self):
        return f"TrieNode(tokens={self.tokens}, page_id={self.page_id}, children={len(self.children)})"


@dataclass
class BeamCandidate:
    """Represents a single beam candidate with its sequence and score."""
    trie_node: TrieNode              # Reference to trie node leaf
    score: float                     # Log probability score
    finished: bool = False           # Whether sequence has ended (EOS token)

@dataclass
class BeamTokenCandidate:
    """Input: represents a potential token with its raw log probability"""
    token_id: int
    log_prob: float

@dataclass
class BeamToken:
    """Output: represents a selected token with accumulated score after penalties"""
    token_id: int
    accumulated_score: float

@dataclass
class BeamGenerateInput:
    candidate: BeamCandidate
    children: List[BeamTokenCandidate]

@dataclass
class BeamGenerateResult:
    candidate: BeamCandidate
    children: List[BeamToken]

class BeamState:
    """Manages the current state of all beam candidates using a trie structure."""

    def __init__(self, beam_size: int, page_table: PageTable):
        self.beam_size = beam_size
        self.page_table = page_table
        self.root = None
        self.candidates: List[BeamCandidate] = []
        self.finished_candidates: List[BeamCandidate] = []

    def add_root_sequence(self, sequence: List[int]) -> List[TrieNode]:
        """ returns created nodes as their creation order"""
        ptr = 0
        current_node = None
        created = []
        while ptr < len(sequence):
            # the higher bound of this page, use min to ensure in bound
            he = min(len(sequence), ptr + self.page_table.page_size)
            # allocate a new block for this node
            page_id = self.page_table.allocate_block()
           
            new_node = TrieNode(sequence[ptr:he], page_id, current_node)
            created.append(new_node)
            if current_node is None:
                self.root = new_node
            current_node = new_node
            ptr += self.page_table.page_size
        
        self.candidates.append(BeamCandidate(current_node, 0, False))
        return created


    def add_filtered_results(self, results: List[BeamGenerateResult]):
        """
        Update beam candidates and trie structure after token filtering.
        """

        def free_dead_branch(node: TrieNode):
            """Free page blocks upward until a node with children is found."""
            while node and not node.children:
                self.page_table.free_block(node.page_id)
                parent = node.parent
                if parent:
                    parent.children.remove(node)
                node = parent

        new_candidates: List[BeamCandidate] = []

        # 1. Cleanup dead branches
        for result in results:
            if not result.children:
                free_dead_branch(result.candidate.trie_node)

        # 2. Expand surviving candidates
        for result in results:
            children = result.children
            if not children:
                continue

            beam = result.candidate
            node = beam.trie_node
            tokens = node.tokens
            page_size = self.page_table.page_size

            # ── Case 1: single child ───────────────────────────────
            if len(children) == 1:
                child = children[0]

                if len(tokens) < page_size:
                    tokens.append(child.token_id)
                    new_candidates.append(
                        BeamCandidate(node, child.accumulated_score)
                    )
                else:
                    new_page = self.page_table.allocate_block()
                    new_node = TrieNode(
                        tokens=[child.token_id],
                        page_id=new_page,
                        parent=node
                    )
                    new_candidates.append(
                        BeamCandidate(new_node, child.accumulated_score)
                    )
                continue

            # ── Case 2: multiple children ──────────────────────────
            if len(tokens) == page_size:
                # Full page → allocate N new blocks
                for child in children:
                    new_page = self.page_table.allocate_block()
                    new_node = TrieNode(
                        tokens=[child.token_id],
                        page_id=new_page,
                        parent=node
                    )
                    new_candidates.append(
                        BeamCandidate(new_node, child.accumulated_score)
                    )
            else:
                # Not full → reuse current node for first child
                first, rest = children[0], children[1:]

                for child in rest:
                    new_page = self.page_table.copy_block(node.page_id, len(node.tokens))
                    new_node = TrieNode(
                        tokens=[*tokens, child.token_id],
                        page_id=new_page,
                        parent=node.parent
                    )
                    new_candidates.append(
                        BeamCandidate(new_node, child.accumulated_score)
                    )

                tokens.append(first.token_id)
                new_candidates.append(
                    BeamCandidate(node, first.accumulated_score)
                )

        self.candidates = new_candidates

    def get_cascade_input(self):
        """
        Prepare cascade attention input data for multi-level KV cache access.
        Optimized implementation: O(N*L) complexity.

        Returns:
            Tuple containing:
            - qo_indptr_arr: List[torch.Tensor]
            - paged_kv_indptr_arr: List[torch.Tensor]
            - paged_kv_indices_arr: List[torch.Tensor]
            - paged_kv_last_page_len: List[torch.Tensor]
            - q: torch.Tensor
        """


        if not self.candidates:
            return ([], [], [], [], torch.empty(0, 0, 0))

        # ------------------------------------------------------------------
        # Phase 1: Compute Branching Levels (Memoized)
        # ------------------------------------------------------------------
        # node_id -> branching_level
        memo_level = {}

        def get_node_level(node: TrieNode) -> int:
            """Recursive memoized level computation."""
            nid = id(node)
            if nid in memo_level:
                return memo_level[nid]
            
            if node.parent is None:
                # Root is always level 0
                memo_level[nid] = 0
                return 0
            
            parent_level = get_node_level(node.parent)
            
            # Check if parent is a branching point
            # Note: This logic matches the original implementation where
            # ANY multiple children (even dead ones) trigger a level increment.
            is_branching = len(node.parent.children) > 1
            
            level = parent_level + (1 if is_branching else 0)
            memo_level[nid] = level
            return level

        # ------------------------------------------------------------------
        # Phase 2: Build Paths and Group by Level
        # ------------------------------------------------------------------
        # Structure: groups_by_level[level][rep_node_id] = list of (cand_idx, cand, nodes_at_level)
        groups_by_level = defaultdict(lambda: defaultdict(list))
        max_cascade_level = 0
        
        # Store paths for query token generation step
        candidate_paths = []

        for cand_idx, candidate in enumerate(self.candidates):
            # Build path from leaf to root
            path = []
            current = candidate.trie_node
            while current is not None:
                path.append(current)
                current = current.parent
            path.reverse() # root to leaf
            
            candidate_paths.append((cand_idx, candidate, path))

            # Segment path nodes by their computed level
            nodes_in_current_path_by_level = defaultdict(list)
            for node in path:
                lvl = get_node_level(node)
                nodes_in_current_path_by_level[lvl].append(node)
                if lvl > max_cascade_level:
                    max_cascade_level = lvl
            
            # Add to global grouping
            for lvl, nodes in nodes_in_current_path_by_level.items():
                # The "representative" node for this candidate at this level 
                # is the last node (deepest) in the chain for this level.
                rep_node = nodes[-1]
                groups_by_level[lvl][id(rep_node)].append((cand_idx, candidate, nodes))

        # ------------------------------------------------------------------
        # Phase 3: Construct Output Tensors
        # ------------------------------------------------------------------
        qo_indptr_arr = []
        paged_kv_indptr_arr = []
        paged_kv_indices_arr = []
        paged_kv_last_page_len = []

        for cascade_level in range(max_cascade_level + 1):
            level_groups = groups_by_level[cascade_level]
            
            # Sort groups by the first candidate index to ensure deterministic order
            # item structure: (rep_node_id, list_of_tuples)
            # list_of_tuples[0] is the first candidate added to this group
            # tuple[0] is cand_idx
            sorted_groups = sorted(level_groups.items(), key=lambda x: x[1][0][0])

            qo_indptr = [0]
            kv_indptr = [0]
            kv_indices = []
            kv_last_page_lens = []

            for _, group_candidates in sorted_groups:
                # All candidates in this group share the same nodes at this level
                # We pick the first one to extract Page IDs
                _, example_candidate, example_nodes = group_candidates[0]

                # Collect unique pages (maintaining order)
                seen_pages = set()
                for node in example_nodes:
                    if node.page_id not in seen_pages:
                        kv_indices.append(node.page_id)
                        seen_pages.add(node.page_id)

                # Determine last page length
                if example_nodes:
                    last_node = example_nodes[-1]
                    token_count = len(last_node.tokens)
                    
                    # Logic: subtract 1 only if this node is the absolute leaf of the candidate
                    # (meaning the query token is in this node and hasn't been KV-cached yet)
                    is_leaf = (last_node == example_candidate.trie_node)
                    
                    if is_leaf:
                        kv_last_page_lens.append(max(0, token_count - 1))
                    else:
                        kv_last_page_lens.append(token_count)
                else:
                    kv_last_page_lens.append(0)

                # Update indptrs
                kv_indptr.append(len(kv_indices))
                
                # Each candidate contributes 1 query
                total_group_outputs = len(group_candidates)
                qo_indptr.append(qo_indptr[-1] + total_group_outputs)

            # Convert to Tensors
            qo_indptr_arr.append(torch.tensor(qo_indptr, dtype=torch.int32))
            paged_kv_indptr_arr.append(torch.tensor(kv_indptr, dtype=torch.int32))
            paged_kv_indices_arr.append(torch.tensor(kv_indices, dtype=torch.int32))
            paged_kv_last_page_len.append(torch.tensor(kv_last_page_lens, dtype=torch.int32))

        # ------------------------------------------------------------------
        # Phase 4: Construct Query Token IDs
        # ------------------------------------------------------------------
        # We need to order query tokens based on the grouping at the FINAL cascade level
        query_token_ids = []
        
        if max_cascade_level in groups_by_level:
            final_level_groups = groups_by_level[max_cascade_level]
            sorted_final_groups = sorted(final_level_groups.items(), key=lambda x: x[1][0][0])
            
            for _, group_candidates in sorted_final_groups:
                for _, candidate, _ in group_candidates:
                    leaf = candidate.trie_node
                    if leaf.tokens:
                        query_token_ids.append(leaf.tokens[-1])
                    else:
                        query_token_ids.append(0)
        
        query_token_ids_tensor = torch.tensor(query_token_ids, dtype=torch.long, 
                                            device=self.page_table.device)

        return (qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr, 
                paged_kv_last_page_len, query_token_ids_tensor)

    def get_best_finished(self, num_return: int) -> List[BeamCandidate]:
        """Get the best finished candidates, normalized by length."""
        if not self.finished_candidates:
            # If no finished candidates, return best active ones
            sorted_active = sorted(self.candidates, key=lambda x: x.score / x.length, reverse=True)
            return sorted_active[:num_return]

        # Sort by normalized score (score / length) to avoid length bias
        sorted_finished = sorted(self.finished_candidates, key=lambda x: x.score / x.length, reverse=True)
        return sorted_finished[:num_return]

