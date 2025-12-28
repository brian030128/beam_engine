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
        self.children: Dict[int, 'TrieNode'] = {}  # Key is first token of child sequence
        self.page_id = page_id  # Page ID in the page table (managed externally)

    def add_children(self, node: "TrieNode"):
        self.children[node.tokens[0]] = node
        node.parent = self

    def __repr__(self):
        return f"TrieNode(tokens={self.tokens}, page_id={self.page_id}, children={len(self.children)})"


@dataclass
class BeamCandidate:
    """Represents a single beam candidate with its sequence and score."""
    trie_node: TrieNode              # Reference to trie node leaf
    score: float                     # Log probability score
    finished: bool = False           # Whether sequence has ended (EOS token)

@dataclass
class BeamScoreItem:
    """Represents a single beam candidate with its sequence and score."""
    candidate: BeamCandidate
    token: int
    score: float


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
            if current_node is not None:
                current_node.add_children(new_node)
            current_node = new_node
            ptr += self.page_table.page_size
        return created

    
    def create_diverge(self, parent: BeamCandidate, token: int, score: int):
        page_id = self.page_table.allocate_block()
        new_node = TrieNode([token], page_id, parent.trie_node)
        parent.trie_node.add_children(new_node)
        if parent in self.candidates:
            self.candidates.remove(parent)
        self.candidates.append(BeamCandidate(new_node, score, False))




    def add_candidate(self, candidate: BeamCandidate):
        """Add a new beam candidate."""
        if candidate.finished:
            self.finished_candidates.append(candidate)
        else:
            self.candidates.append(candidate)

    def get_active_candidates(self) -> List[BeamCandidate]:
        """Get all active (non-finished) candidates."""
        return [c for c in self.candidates if not c.finished]

    def get_best_finished(self, num_return: int) -> List[BeamCandidate]:
        """Get the best finished candidates, normalized by length."""
        if not self.finished_candidates:
            # If no finished candidates, return best active ones
            sorted_active = sorted(self.candidates, key=lambda x: x.score / x.length, reverse=True)
            return sorted_active[:num_return]

        # Sort by normalized score (score / length) to avoid length bias
        sorted_finished = sorted(self.finished_candidates, key=lambda x: x.score / x.length, reverse=True)
        return sorted_finished[:num_return]

