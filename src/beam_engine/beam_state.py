from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
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
        where each level corresponds to different depths in the tree (shared prefixes).
        This allows efficient reuse of KV cache across beam candidates that share
        common prefixes.

        The query tensor is constructed from the last tokens of each beam candidate,
        which will be used for the next token prediction.

        Returns:
            Tuple containing:
            - qo_indptr_arr: List[torch.Tensor] - qo indptr for each cascade level
            - paged_kv_indptr_arr: List[torch.Tensor] - KV indptr for each cascade level
            - paged_kv_indices_arr: List[torch.Tensor] - KV page indices for each cascade level
            - paged_kv_last_page_len: List[torch.Tensor] - Last page lengths for each cascade level
            - q: torch.Tensor - Query tensor [beam_width, num_qo_heads, head_dim]
        """
        import torch

        if not self.candidates:
            # Return empty structures if no candidates
            return ([], [], [], [], torch.empty(0, 0, 0))

        beam_width = len(self.candidates)

        # For cascade attention, organize by branching points (nodes with >1 child)
        # Each level corresponds to a set of nodes that branch (have multiple children)
        levels_data = {}  # level -> list of (candidate_idx, candidate, branching_path)

        # First, identify all branching nodes in the trie
        branching_nodes = set()

        def find_branching_nodes(node):
            if node is None:
                return
            if len(node.children) > 1:
                branching_nodes.add(id(node))
            for child in node.children:
                find_branching_nodes(child)

        # Start from root to find all branching points
        if self.root:
            find_branching_nodes(self.root)

        # For each candidate, find its path through branching points
        for cand_idx, candidate in enumerate(self.candidates):
            # Build path from leaf to root
            full_path = []
            current = candidate.trie_node
            while current is not None:
                full_path.append(current)
                current = current.parent

            # Reverse to get root-to-leaf path
            full_path.reverse()

            # Extract branching path - nodes that are branching points
            branching_path = []
            for node in full_path:
                if id(node) in branching_nodes:
                    branching_path.append(node)

            # Assign candidate to levels based on how many branching points it has encountered
            # Level 0: before first branch, Level 1: after first branch, etc.
            level = len(branching_path)

            if level not in levels_data:
                levels_data[level] = []
            levels_data[level].append((cand_idx, candidate, full_path, branching_path))

        # Sort levels by number of branching points encountered
        sorted_levels = sorted(levels_data.keys())

        qo_indptr_arr = []
        paged_kv_indptr_arr = []
        paged_kv_indices_arr = []
        paged_kv_last_page_len = []

        for level_idx, level in enumerate(sorted_levels):
            level_candidates = levels_data[level]

            # For query/output indptr: cumulative count of queries at this level
            # Each candidate contributes 1 query
            qo_indptr = [0]
            for i in range(len(level_candidates)):
                qo_indptr.append(qo_indptr[-1] + 1)
            qo_indptr_arr.append(torch.tensor(qo_indptr, dtype=torch.int32))

            # For KV cache: build page indices and last page lengths for this level
            kv_indptr = [0]
            kv_indices = []
            kv_last_page_lens = []

            for cand_idx, candidate, full_path, branching_path in level_candidates:
                # For each candidate, include all pages in its full path
                # This represents all the KV cache needed for this candidate
                candidate_pages = []

                for path_node in full_path:
                    candidate_pages.append(path_node.page_id)

                # Add pages for this candidate
                kv_indices.extend(candidate_pages)

                # Last page length is the number of tokens in the final node of the full path
                if full_path:
                    kv_last_page_lens.append(len(full_path[-1].tokens))
                else:
                    kv_last_page_lens.append(0)

                # Update indptr - points to start of next candidate's pages
                kv_indptr.append(len(kv_indices))

            # Convert to tensors
            paged_kv_indptr_arr.append(torch.tensor(kv_indptr, dtype=torch.int32))
            paged_kv_indices_arr.append(torch.tensor(kv_indices, dtype=torch.int32))
            paged_kv_last_page_len.append(torch.tensor(kv_last_page_lens, dtype=torch.int32))

        # Build query tensor organized by cascade levels
        # The queries must be in the same order as referenced by qo_indptr arrays
        num_qo_heads = self.page_table.head_num  # assuming qo_heads = kv_heads for simplicity
        head_dim = self.page_table.head_dim

        # Collect queries in cascade level order
        ordered_queries = []

        for level_idx, level in enumerate(sorted_levels):
            level_candidates = levels_data[level]

            for cand_idx, candidate, full_path, branching_path in level_candidates:
                # Extract the last token from this candidate's full sequence
                full_sequence = []
                for path_node in full_path:
                    full_sequence.extend(path_node.tokens)

                if full_sequence:
                    last_token = full_sequence[-1]
                else:
                    last_token = 0  # fallback

                # In practice, this would be:
                # 1. Embed the last token: embedding(last_token)
                # 2. Project through query layer: query_proj(embedded_token)
                # For now, create a placeholder query vector
                query_vector = torch.zeros(num_qo_heads, head_dim,
                                         dtype=self.page_table.store_dtype,
                                         device=self.page_table.device)
                ordered_queries.append(query_vector)

        # Stack all queries into final tensor [total_queries, num_qo_heads, head_dim]
        if ordered_queries:
            q = torch.stack(ordered_queries, dim=0)
        else:
            q = torch.empty(0, num_qo_heads, head_dim,
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

