from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List, Set
from beam_state import BeamState, BeamCandidate, BeamScoreItem, TrieNode



class BeamStrategy(ABC):
    """Abstract base class for beam search strategies."""

    @abstractmethod
    def select_candidates(self, beam_state: BeamState, new_candidates: List[BeamScoreItem],
                         step: int) -> List[BeamScoreItem]:
        """
        Select which candidates to keep for the next step.

        Args:
            beam_state: Current beam state
            new_candidates: All newly generated candidates
            step: Current generation step

        Returns:
            List of candidates to keep (should be <= beam_size)
        """
        pass

    @abstractmethod
    def should_stop(self, beam_state: BeamState, max_length: int, step: int) -> bool:
        """
        Decide whether to stop generation.

        Args:
            beam_state: Current beam state
            max_length: Maximum sequence length
            step: Current generation step

        Returns:
            True if generation should stop
        """
        pass

class DiverseBeamSearchStrategy(BeamStrategy):
    """
    Implements Diverse Beam Search (DBS) strategy.
    
    Diverse beam search encourages diversity by:
    1. Grouping beams and ensuring diversity within groups
    2. Applying diversity penalties to similar sequences
    3. Balancing exploration vs exploitation
    
    Based on Vijayakumar et al. "Diverse Beam Search: Decoding Diverse 
    Solutions from Neural Sequence Models" (AAAI 2018)
    """

    def __init__(
        self,
        num_groups: int = 4,
        diversity_penalty: float = 0.5,
        diversity_type: str = "hamming",
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ):
        """
        Initialize DBS strategy.

        Args:
            num_groups: Number of diverse beam groups. Total beams = beam_size,
                       beams per group = beam_size // num_groups
            diversity_penalty: Lambda parameter controlling diversity strength.
                              Higher values encourage more diverse outputs.
            diversity_type: Type of diversity function ("hamming" or "cumulative")
            length_penalty: Penalty applied to sequence length for scoring
            early_stopping: Whether to stop when all beams have finished
        """
        self.num_groups = num_groups
        self.diversity_penalty = diversity_penalty
        self.diversity_type = diversity_type
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

    def _get_sequence_from_node(self, node: TrieNode) -> List[int]:
        """Reconstruct full token sequence from trie node by traversing to root."""
        tokens = []
        current = node
        while current is not None and current.tokens:
            tokens = current.tokens + tokens
            current = current.parent
        return tokens

    def _compute_hamming_diversity(
        self,
        candidate_token: int,
        selected_tokens: Set[int],
    ) -> float:
        """
        Compute Hamming diversity penalty.
        
        Penalizes tokens that have already been selected in previous groups
        at the same position.

        Args:
            candidate_token: Token being evaluated
            selected_tokens: Tokens already selected by previous groups

        Returns:
            Diversity penalty (0 if token is diverse, penalty otherwise)
        """
        if candidate_token in selected_tokens:
            return self.diversity_penalty
        return 0.0

    def _compute_cumulative_diversity(
        self,
        candidate_sequence: List[int],
        selected_sequences: List[List[int]],
    ) -> float:
        """
        Compute cumulative diversity penalty based on sequence overlap.

        Args:
            candidate_sequence: Full sequence of candidate
            selected_sequences: Sequences already selected by previous groups

        Returns:
            Cumulative diversity penalty
        """
        if not selected_sequences:
            return 0.0

        total_penalty = 0.0
        for selected_seq in selected_sequences:
            overlap = sum(
                1 for t1, t2 in zip(candidate_sequence, selected_seq) if t1 == t2
            )
            total_penalty += overlap * self.diversity_penalty

        return total_penalty / len(selected_sequences)

    def _apply_length_penalty(self, score: float, length: int) -> float:
        """
        Apply length normalization to score.

        Args:
            score: Raw log probability score
            length: Sequence length

        Returns:
            Length-normalized score
        """
        if self.length_penalty == 0:
            return score
        return score / (length ** self.length_penalty)

    def _group_candidates_by_parent(
        self,
        candidates: List[BeamScoreItem],
    ) -> Dict[int, List[BeamScoreItem]]:
        """
        Group candidates by their parent beam for efficient processing.

        Args:
            candidates: List of all candidates

        Returns:
            Dictionary mapping parent node id to candidates
        """
        grouped: Dict[int, List[BeamScoreItem]] = {}
        for cand in candidates:
            parent_id = id(cand.candidate.trie_node)
            if parent_id not in grouped:
                grouped[parent_id] = []
            grouped[parent_id].append(cand)
        return grouped

    def select_candidates(
        self,
        beam_state: BeamState,
        new_candidates: List[BeamScoreItem],
        step: int,
    ) -> List[BeamScoreItem]:
        """
        Select candidates using Diverse Beam Search algorithm.

        The algorithm processes groups sequentially. Each group selects its
        top candidates while being penalized for selecting tokens/sequences
        that previous groups have already chosen.

        Args:
            beam_state: Current beam state
            new_candidates: All newly generated candidates with scores
            step: Current generation step

        Returns:
            List of selected candidates (size <= beam_size)
        """
        if not new_candidates:
            return []

        beam_size = beam_state.beam_size
        beams_per_group = max(1, beam_size // self.num_groups)
        
        # Track what's been selected across all groups
        selected_tokens_at_step: Set[int] = set()
        selected_sequences: List[List[int]] = []
        all_selected: List[BeamScoreItem] = []

        # Sort candidates by score initially
        sorted_candidates = sorted(new_candidates, key=lambda x: x.score, reverse=True)

        for group_idx in range(self.num_groups):
            group_selected: List[BeamScoreItem] = []
            remaining_slots = beams_per_group

            # Create a copy of candidates with diversity-adjusted scores
            adjusted_candidates: List[Tuple[BeamScoreItem, float]] = []

            for cand in sorted_candidates:
                if cand in all_selected:
                    continue

                # Get the sequence for cumulative diversity
                parent_sequence = self._get_sequence_from_node(cand.candidate.trie_node)
                candidate_sequence = parent_sequence + [cand.token]
                sequence_length = len(candidate_sequence)

                # Calculate diversity penalty
                if self.diversity_type == "hamming":
                    diversity_penalty = self._compute_hamming_diversity(
                        cand.token, selected_tokens_at_step
                    )
                else:  # cumulative
                    diversity_penalty = self._compute_cumulative_diversity(
                        candidate_sequence, selected_sequences
                    )

                # Apply diversity penalty to score
                adjusted_score = cand.score - diversity_penalty

                # Apply length penalty
                adjusted_score = self._apply_length_penalty(adjusted_score, sequence_length)

                adjusted_candidates.append((cand, adjusted_score))

            # Sort by adjusted score
            adjusted_candidates.sort(key=lambda x: x[1], reverse=True)

            # Select top candidates for this group
            for cand, adj_score in adjusted_candidates:
                if remaining_slots <= 0:
                    break

                group_selected.append(cand)
                all_selected.append(cand)
                remaining_slots -= 1

                # Update tracking for subsequent groups
                selected_tokens_at_step.add(cand.token)
                parent_sequence = self._get_sequence_from_node(cand.candidate.trie_node)
                selected_sequences.append(parent_sequence + [cand.token])

        # Ensure we don't exceed beam_size
        return all_selected[:beam_size]

    def should_stop(
        self,
        beam_state: BeamState,
        max_length: int,
        step: int,
    ) -> bool:
        """
        Determine whether to stop generation.

        Stops if:
        1. Maximum length is reached
        2. All beams have finished (if early_stopping is enabled)
        3. Required number of finished sequences is reached

        Args:
            beam_state: Current beam state
            max_length: Maximum allowed sequence length
            step: Current generation step

        Returns:
            True if generation should stop
        """
        # Stop if max length reached
        if step >= max_length:
            return True

        # If early stopping, check if all beams are finished
        if self.early_stopping:
            if not beam_state.candidates:
                return True
            if all(c.finished for c in beam_state.candidates):
                return True

        # Check if we have enough finished candidates
        if len(beam_state.finished_candidates) >= beam_state.beam_size:
            return True

        return False

    def get_final_sequences(
        self,
        beam_state: BeamState,
        num_return: Optional[int] = None,
    ) -> List[Tuple[List[int], float]]:
        """
        Get the final diverse sequences after search completion.

        Args:
            beam_state: Final beam state
            num_return: Number of sequences to return (default: beam_size)

        Returns:
            List of (sequence, score) tuples, sorted by score
        """
        if num_return is None:
            num_return = beam_state.beam_size

        # Combine finished and active candidates
        all_candidates = beam_state.finished_candidates + beam_state.candidates

        # Extract sequences and scores
        results: List[Tuple[List[int], float]] = []
        for candidate in all_candidates:
            sequence = self._get_sequence_from_node(candidate.trie_node)
            score = self._apply_length_penalty(candidate.score, len(sequence))
            results.append((sequence, score))

        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_return]

    def get_diverse_groups(
        self,
        beam_state: BeamState,
    ) -> List[List[BeamCandidate]]:
        """
        Return candidates organized by their diversity groups.

        This is useful for applications that want to show diverse alternatives.

        Args:
            beam_state: Current beam state

        Returns:
            List of groups, where each group contains related candidates
        """
        beams_per_group = max(1, beam_state.beam_size // self.num_groups)
        groups: List[List[BeamCandidate]] = []

        all_candidates = beam_state.finished_candidates + beam_state.candidates
        all_candidates.sort(key=lambda x: x.score, reverse=True)

        for i in range(self.num_groups):
            start_idx = i * beams_per_group
            end_idx = min(start_idx + beams_per_group, len(all_candidates))
            if start_idx < len(all_candidates):
                groups.append(all_candidates[start_idx:end_idx])

        return groups