from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import torch
from collections import defaultdict

from page_table import PageTable
from .logger import init_logger

logger = init_logger(__name__)
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
        logger.debug(f"\n[BeamState] Adding root sequence with {len(sequence)} tokens")
        ptr = 0
        current_node = None
        created = []
        while ptr < len(sequence):
            # the higher bound of this page, use min to ensure in bound
            he = min(len(sequence), ptr + self.page_table.page_size)
            # allocate a new block for this node
            page_id = self.page_table.allocate_block()
            logger.debug(f"  [PAGE ALLOC] Page {page_id} allocated for root tokens {ptr}:{he}")

            new_node = TrieNode(sequence[ptr:he], page_id, current_node)
            created.append(new_node)
            if current_node is None:
                self.root = new_node
            current_node = new_node
            ptr += self.page_table.page_size

        self.candidates.append(BeamCandidate(current_node, 0, False))
        logger.debug(f"  [BeamState] Root candidate created: {len(created)} nodes/pages")
        return created


    def add_filtered_results(self, results: List[BeamGenerateResult]):
        """
        Update beam candidates and trie structure after token filtering.
        """
        logger.debug(f"\n[BeamState] add_filtered_results: {len(results)} results, {len(self.candidates)} current candidates")

        def free_dead_branch(node: TrieNode):
            """Free page blocks upward until a node with children is found."""
            freed_pages = []
            while node and not node.children:
                freed_pages.append(node.page_id)
                self.page_table.free_block(node.page_id)
                parent = node.parent
                if parent:
                    parent.children.remove(node)
                node = parent
            if freed_pages:
                logger.debug(f"  [PAGE FREE] Pages freed (dead branch): {freed_pages}")

        new_candidates: List[BeamCandidate] = []

        # 1. Cleanup dead branches
        logger.debug(f"  [BRANCH CLEANUP] Checking for branches to eliminate...")
        for idx, result in enumerate(results):
            if not result.children:
                logger.debug(f"    [BRANCH ELIMINATED] Candidate {idx} (score={result.candidate.score:.4f}) has no children - eliminating")
                free_dead_branch(result.candidate.trie_node)
            else:
                logger.debug(f"    [BRANCH KEPT] Candidate {idx} (score={result.candidate.score:.4f}) has {len(result.children)} children")

        # 2. Expand surviving candidates
        logger.debug(f"\n  [EXPAND] Expanding surviving candidates...")
        for result_idx, result in enumerate(results):
            children = result.children
            if not children:
                continue

            beam = result.candidate
            node = beam.trie_node
            tokens = node.tokens
            page_size = self.page_table.page_size

            logger.debug(f"    [EXPAND {result_idx}] Node page_id={node.page_id}, tokens={len(tokens)}/{page_size}, children={len(children)}")

            # ── Case 1: single child ───────────────────────────────
            if len(children) == 1:
                child = children[0]
                logger.debug(f"      [SINGLE CHILD] token_id={child.token_id}, score={child.accumulated_score:.4f}")

                if len(tokens) < page_size:
                    tokens.append(child.token_id)
                    logger.debug(f"        [APPEND] Appended to existing page {node.page_id} (now {len(tokens)}/{page_size} tokens)")
                    new_candidates.append(
                        BeamCandidate(node, child.accumulated_score)
                    )
                else:
                    new_page = self.page_table.allocate_block()
                    logger.debug(f"        [PAGE ALLOC] Page {new_page} allocated (parent page {node.page_id} is full)")
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
            logger.debug(f"      [MULTIPLE CHILDREN] {len(children)} branches")
            if len(tokens) == page_size:
                # Full page → allocate N new blocks
                logger.debug(f"        [FULL PAGE] Allocating {len(children)} new pages")
                for child_idx, child in enumerate(children):
                    new_page = self.page_table.allocate_block()
                    logger.debug(f"          [PAGE ALLOC] Page {new_page} for child {child_idx}: token_id={child.token_id}, score={child.accumulated_score:.4f}")
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
                logger.debug(f"        [PARTIAL PAGE] Reusing page {node.page_id} for first child, copying for {len(rest)} others")

                for child_idx, child in enumerate(rest):
                    new_page = self.page_table.copy_block(node.page_id, len(node.tokens))
                    logger.debug(f"          [PAGE COPY] Page {new_page} copied from {node.page_id} ({len(node.tokens)} tokens) for child {child_idx+1}: token_id={child.token_id}, score={child.accumulated_score:.4f}")
                    new_node = TrieNode(
                        tokens=[*tokens, child.token_id],
                        page_id=new_page,
                        parent=node.parent
                    )
                    new_candidates.append(
                        BeamCandidate(new_node, child.accumulated_score)
                    )

                tokens.append(first.token_id)
                logger.debug(f"          [APPEND] First child token_id={first.token_id} appended to page {node.page_id}, score={first.accumulated_score:.4f}")
                new_candidates.append(
                    BeamCandidate(node, first.accumulated_score)
                )

        self.candidates = new_candidates
        logger.debug(f"\n  [RESULT] {len(new_candidates)} new candidates created")

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

        logger.debug(f"\n[CASCADE] Building cascade input for {len(self.candidates)} candidates")

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

        # First pass: build candidate info and determine max cascade level
        candidate_info = []  # List of (cand_idx, candidate, path, nodes_by_level)
        max_cascade_level = 0

        for cand_idx, candidate in enumerate(self.candidates):
            # Build path from leaf to root
            path = []
            current = candidate.trie_node
            while current is not None:
                path.append(current)
                current = current.parent
            path.reverse() # root to leaf

            # Segment path nodes by their computed level
            nodes_in_current_path_by_level = defaultdict(list)
            for node in path:
                lvl = get_node_level(node)
                nodes_in_current_path_by_level[lvl].append(node)
                if lvl > max_cascade_level:
                    max_cascade_level = lvl

            candidate_info.append((cand_idx, candidate, path, nodes_in_current_path_by_level))

        # Second pass: build groups ensuring EVERY candidate appears at EVERY level
        # This is critical when different branches have different max cascade levels
        groups_by_level = defaultdict(lambda: defaultdict(list))

        for cand_idx, candidate, path, nodes_by_level in candidate_info:
            # Find this candidate's maximum level
            candidate_max_level = max(nodes_by_level.keys()) if nodes_by_level else 0

            # Add this candidate to ALL cascade levels from 0 to max_cascade_level
            for cascade_level in range(max_cascade_level + 1):
                if cascade_level in nodes_by_level:
                    # Candidate has nodes at this specific level
                    nodes_at_level = nodes_by_level[cascade_level]
                else:
                    # Candidate doesn't have nodes at this level (its path ended earlier)
                    # Use nodes from its deepest level
                    nodes_at_level = nodes_by_level[candidate_max_level]

                # The "representative" node for this candidate at this level
                rep_node = nodes_at_level[-1]
                groups_by_level[cascade_level][id(rep_node)].append((cand_idx, candidate, nodes_at_level))

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
        # Extract query token from each candidate's leaf node
        # This ensures we get a query token for ALL candidates, regardless of
        # which cascade level they're at (fixes bug where different branches
        # have different max cascade levels)
        query_token_ids = []

        for candidate in self.candidates:
            leaf = candidate.trie_node
            if leaf.tokens:
                query_token_ids.append(leaf.tokens[-1])
            else:
                query_token_ids.append(0)

        query_token_ids_tensor = torch.tensor(query_token_ids, dtype=torch.long,
                                            device=self.page_table.device)

        # Debug output
        logger.debug(f"  [CASCADE] Built {max_cascade_level + 1} cascade levels")
        for lvl in range(max_cascade_level + 1):
            num_groups = len(groups_by_level[lvl])
            total_candidates = sum(len(group) for group in groups_by_level[lvl].values())
            logger.debug(f"    Level {lvl}: {num_groups} groups, {total_candidates} candidates")
            logger.debug(f"      qo_indptr: {qo_indptr_arr[lvl].tolist()}")
            logger.debug(f"      kv_indptr: {paged_kv_indptr_arr[lvl].tolist()}")
            logger.debug(f"      kv_indices: {paged_kv_indices_arr[lvl].tolist()}")
            logger.debug(f"      kv_last_page_len: {paged_kv_last_page_len[lvl].tolist()}")
        logger.debug(f"  [CASCADE] Query token IDs: {query_token_ids_tensor.tolist()}")

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

    def get_final_sequences(self, num_return: int, length_penalty: float = 1.0) -> List[Tuple[List[int], float]]:
        """
        Get final sequences from both alive and finished candidates.

        Args:
            num_return: Number of sequences to return
            length_penalty: Penalty for sequence length (score = log_prob / length^penalty)

        Returns:
            List of (sequence, score) tuples sorted by score
        """
        # Combine alive and finished candidates
        all_candidates = self.finished_candidates + self.candidates

        if not all_candidates:
            return []

        # Extract sequences and apply length penalty
        results: List[Tuple[List[int], float]] = []
        for candidate in all_candidates:
            # Reconstruct sequence from trie
            sequence = []
            node = candidate.trie_node
            while node is not None:
                sequence = node.tokens + sequence
                node = node.parent

            # Apply length penalty
            score = candidate.score
            if length_penalty != 0 and len(sequence) > 1:
                score = score / (len(sequence) ** length_penalty)

            results.append((sequence, score))

        # Sort by score (descending) and return top num_return
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_return]

