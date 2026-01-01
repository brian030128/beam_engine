from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .page_table import PageTable

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
                    new_page = self.page_table.copy_block(node)
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

        In cascade attention, we organize the trie structure into multiple levels,
        where each level corresponds to layers between branching points in the tree.
        The same output queries are reorganized differently at each cascade level.

        Returns:
            Tuple containing:
            - qo_indptr_arr: List[torch.Tensor] - qo indptr for each cascade level
            - paged_kv_indptr_arr: List[torch.Tensor] - KV indptr for each cascade level
            - paged_kv_indices_arr: List[torch.Tensor] - KV page indices for each cascade level
            - paged_kv_last_page_len: List[torch.Tensor] - Last page lengths for each cascade level
            - q: torch.Tensor - Query tensor [total_outputs, num_qo_heads, head_dim]
        """
        import torch
        from collections import defaultdict

        if not self.candidates:
            return ([], [], [], [], torch.empty(0, 0, 0))

        # Helper function to count branching ancestors of a node
        def count_branching_ancestors(node):
            """Count how many ancestors of this node have multiple children."""
            count = 0
            current = node.parent
            while current is not None:
                if len(current.children) > 1:
                    count += 1
                current = current.parent
            return count

        # Step 1: Build candidate info
        # In beam search, each candidate generates exactly 1 output token
        candidate_info = []  # List of (cand_idx, candidate, path)
        total_outputs = len(self.candidates)  # One output per candidate

        for cand_idx, candidate in enumerate(self.candidates):
            # Build path from leaf to root
            path = []
            current = candidate.trie_node
            while current is not None:
                path.append(current)
                current = current.parent
            path.reverse()  # root to leaf

            candidate_info.append((cand_idx, candidate, path))

        # Step 2: Determine max cascade level
        max_cascade_level = 0
        for _, _, path in candidate_info:
            for node in path:
                level = count_branching_ancestors(node)
                max_cascade_level = max(max_cascade_level, level)

        # Step 3: For each cascade level, organize candidates into groups
        qo_indptr_arr = []
        paged_kv_indptr_arr = []
        paged_kv_indices_arr = []
        paged_kv_last_page_len = []

        for cascade_level in range(max_cascade_level + 1):
            # Group candidates by their last node at this cascade level
            node_groups = defaultdict(list)  # node_id -> list of (cand_idx, candidate, path, nodes_at_level)

            for cand_idx, candidate, path in candidate_info:
                # Find nodes in this path that belong to this cascade level
                nodes_at_level = [node for node in path if count_branching_ancestors(node) == cascade_level]

                if nodes_at_level:
                    # Group by the last node at this level (furthest from root)
                    grouping_node = nodes_at_level[-1]
                    node_groups[id(grouping_node)].append((cand_idx, candidate, path, nodes_at_level))

            # Step 4: Build indptr and indices for this level
            qo_indptr = [0]
            kv_indptr = [0]
            kv_indices = []
            kv_last_page_lens = []

            # Sort groups by the first candidate index for deterministic ordering
            sorted_groups = sorted(node_groups.items(), key=lambda x: x[1][0][0])  # Sort by first cand_idx

            for node_id, group_candidates in sorted_groups:
                # Extract nodes at this level for this group
                example_cand_idx, example_candidate, example_path, example_nodes_at_level = group_candidates[0]

                # At each cascade level, we need to include pages for the KV cache at this level
                # This includes both newly allocated pages AND pages reused from parents
                # (because current tokens' KV will be stored in those reused pages)

                # For each candidate in this group, we need to determine which pages to include
                # Since all candidates in a group share the same nodes at this level, we use the example

                # Collect all unique pages from nodes at this level
                pages_at_level = []
                seen_pages = set()

                for node in example_nodes_at_level:
                    if node.page_id not in seen_pages:
                        pages_at_level.append(node.page_id)
                        seen_pages.add(node.page_id)

                # Add these pages for this group
                kv_indices.extend(pages_at_level)

                # Last page length: number of tokens in the last page's KV cache
                # At the leaf level (final node of the candidate), exclude the current query token
                # since it hasn't been added to the KV cache yet
                if example_nodes_at_level:
                    last_node = example_nodes_at_level[-1]
                    token_count = len(last_node.tokens)

                    # Check if this is the leaf node (final level for this candidate)
                    is_leaf = (last_node == example_candidate.trie_node)

                    # Only subtract 1 at the leaf level (where query tokens are)
                    if is_leaf:
                        kv_last_page_lens.append(max(0, token_count - 1))
                    else:
                        kv_last_page_lens.append(token_count)
                else:
                    kv_last_page_lens.append(0)

                kv_indptr.append(len(kv_indices))

                # Count total outputs for this group (1 output per candidate)
                total_group_outputs = len(group_candidates)
                qo_indptr.append(qo_indptr[-1] + total_group_outputs)

            # Convert to tensors
            qo_indptr_arr.append(torch.tensor(qo_indptr, dtype=torch.int32))
            paged_kv_indptr_arr.append(torch.tensor(kv_indptr, dtype=torch.int32))
            paged_kv_indices_arr.append(torch.tensor(kv_indices, dtype=torch.int32))
            paged_kv_last_page_len.append(torch.tensor(kv_last_page_lens, dtype=torch.int32))

        # Step 5: Build query tensor (one query per output token)
        num_qo_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim

        # Create placeholder queries (one per output token)
        q = torch.zeros(total_outputs, num_qo_heads, head_dim,
                       dtype=self.page_table.store_dtype,
                       device=self.page_table.device)

        return (qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr, paged_kv_last_page_len, q)

    def get_best_finished(self, num_return: int) -> List[BeamCandidate]:
        """Get the best finished candidates, normalized by length."""
        if not self.finished_candidates:
            # If no finished candidates, return best active ones
            sorted_active = sorted(self.candidates, key=lambda x: x.score / x.length, reverse=True)
            return sorted_active[:num_return]

        # Sort by normalized score (score / length) to avoid length bias
        sorted_finished = sorted(self.finished_candidates, key=lambda x: x.score / x.length, reverse=True)
        return sorted_finished[:num_return]

