from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List, Set
from beam_state import BeamState, BeamCandidate, TrieNode, BeamGenerateResult, BeamToken, BeamGenerateInput, BeamTokenCandidate
from dataclasses import dataclass



class BeamStrategy(ABC):
    """Abstract base class for beam search strategies."""

    @abstractmethod
    def select_candidates(self, beam_state: BeamState, new_candidates: List[BeamGenerateInput],
                         step: int) -> List[BeamGenerateResult]:
        """
        Select which candidates to keep for the next step.

        Args:
            beam_state: Current beam state
            new_candidates: All newly generated candidates with their raw log probabilities
            step: Current generation step

        Returns:
            List of candidates with accumulated scores after penalties (same size as input)
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

    @abstractmethod
    def get_final_sequences(
        self,
        beam_state: BeamState,
        num_return: Optional[int] = None,
    ) -> List[Tuple[List[int], float]]:
        """
        Get the final sequences after search completion.

        Args:
            beam_state: Final beam state
            num_return: Number of sequences to return (default: beam_size)

        Returns:
            List of (sequence, score) tuples, sorted by score
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


    def select_candidates(
        self,
        beam_state: BeamState,
        new_candidates: List[BeamGenerateInput],
        step: int,
    ) -> List[BeamGenerateResult]:
        """
        Select candidates using Diverse Beam Search algorithm.

        The algorithm processes groups sequentially. Each group selects its
        top candidates while being penalized for selecting tokens/sequences
        that previous groups have already chosen.

        Args:
            beam_state: Current beam state
            new_candidates: List of BeamGenerateInput with candidate and raw log probabilities
            step: Current generation step

        Returns:
            List of BeamGenerateResult with accumulated scores after penalties (same size as input)
        """
        if not new_candidates:
            return []

        beam_size = beam_state.beam_size
        beams_per_group = max(1, beam_size // self.num_groups)

        # Track what's been selected across all groups
        selected_tokens_at_step: Set[int] = set()
        selected_sequences: List[List[int]] = []

        # Flatten all token choices from all candidates for processing
        # Each item is (BeamGenerateInput, BeamTokenCandidate, parent_sequence)
        all_token_choices: List[Tuple[BeamGenerateInput, BeamTokenCandidate, List[int]]] = []

        for beam_input in new_candidates:
            parent_sequence = self._get_sequence_from_node(beam_input.candidate.trie_node)
            for token_candidate in beam_input.children:
                all_token_choices.append((beam_input, token_candidate, parent_sequence))

        # Sort by log probability initially
        all_token_choices.sort(key=lambda x: x[1].log_prob, reverse=True)

        # Track selected token choices for each group
        selected_choices: List[Tuple[BeamGenerateInput, BeamTokenCandidate]] = []

        for group_idx in range(self.num_groups):
            remaining_slots = beams_per_group

            # Create candidates with diversity-adjusted scores
            adjusted_choices: List[Tuple[BeamGenerateInput, BeamTokenCandidate, float]] = []

            for beam_input, token_candidate, parent_sequence in all_token_choices:
                # Skip if this exact choice was already selected
                if any(bi == beam_input and tc.token_id == token_candidate.token_id for bi, tc in selected_choices):
                    continue

                candidate_sequence = parent_sequence + [token_candidate.token_id]
                sequence_length = len(candidate_sequence)

                # Calculate diversity penalty
                if self.diversity_type == "hamming":
                    diversity_penalty = self._compute_hamming_diversity(
                        token_candidate.token_id, selected_tokens_at_step
                    )
                else:  # cumulative
                    diversity_penalty = self._compute_cumulative_diversity(
                        candidate_sequence, selected_sequences
                    )

                # Apply diversity penalty to log probability
                adjusted_score = token_candidate.log_prob - diversity_penalty

                # Apply length penalty
                adjusted_score = self._apply_length_penalty(adjusted_score, sequence_length)

                adjusted_choices.append((beam_input, token_candidate, adjusted_score))

            # Sort by adjusted score
            adjusted_choices.sort(key=lambda x: x[2], reverse=True)

            # Select top choices for this group
            for beam_input, token_candidate, adj_score in adjusted_choices:
                if remaining_slots <= 0:
                    break

                selected_choices.append((beam_input, token_candidate))
                remaining_slots -= 1

                # Update tracking for subsequent groups
                selected_tokens_at_step.add(token_candidate.token_id)
                parent_sequence = self._get_sequence_from_node(beam_input.candidate.trie_node)
                selected_sequences.append(parent_sequence + [token_candidate.token_id])

        # Group selected choices back by their parent BeamGenerateInput
        result_map: Dict[id, List[BeamToken]] = {}

        # Initialize all candidates with empty children lists
        for beam_input in new_candidates:
            result_id = id(beam_input)
            result_map[result_id] = []

        # Add selected choices to their parent candidates with accumulated scores
        for beam_input, token_candidate in selected_choices:
            result_id = id(beam_input)

            # Calculate accumulated score: parent's score + token's log probability
            accumulated_score = beam_input.candidate.score + token_candidate.log_prob

            beam_token = BeamToken(
                token_id=token_candidate.token_id,
                accumulated_score=accumulated_score
            )
            result_map[result_id].append(beam_token)

        # Create new BeamGenerateResult objects with filtered children, maintaining original order
        final_results = []
        for beam_input in new_candidates:
            result_id = id(beam_input)
            filtered_children = result_map[result_id]

            new_result = BeamGenerateResult(
                candidate=beam_input.candidate,
                children=filtered_children  # May be empty if all children were filtered out
            )
            final_results.append(new_result)

        # Ensure we don't exceed beam_size total tokens across all candidates
        total_tokens = sum(len(result.children) for result in final_results)
        if total_tokens > beam_size:
            # Truncate by removing tokens from the end of each result
            tokens_to_remove = total_tokens - beam_size
            for result in reversed(final_results):
                while tokens_to_remove > 0 and result.children:
                    result.children.pop()
                    tokens_to_remove -= 1
                if tokens_to_remove <= 0:
                    break

        return final_results

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
        Get the final diverse sequences with length penalty applied.

        Args:
            beam_state: Final beam state
            num_return: Number of sequences to return (default: beam_size)

        Returns:
            List of (sequence, score) tuples, sorted by score
        """
        if num_return is None:
            num_return = beam_state.beam_size

        return beam_state.get_final_sequences(num_return, self.length_penalty)

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
    

class VanillaBeamSearchStrategy(BeamStrategy):
    """
    Implements standard (Vanilla) Beam Search strategy.
    
    This strategy selects the top-K candidates based purely on their 
    accumulated log-probabilities, with an optional length penalty.
    """

    def __init__(
        self,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ):
        """
        Initialize Vanilla Beam Search.

        Args:
            length_penalty: Exponential penalty for sequence length. 
                           score = log_prob / (length ** length_penalty)
            early_stopping: If True, stop when beam_size finished sequences are found.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

    def _apply_length_penalty(self, log_prob: float, length: int) -> float:
        """Applies standard length normalization."""
        if self.length_penalty == 0 or length <= 1:
            return log_prob
        return log_prob / (length ** self.length_penalty)

    def select_candidates(
        self,
        beam_state: BeamState,
        new_candidates: List[BeamGenerateInput],
        step: int
    ) -> List[BeamGenerateResult]:
        """
        Select the top-K candidates across all potential next tokens.
        """
        if not new_candidates:
            return []

        beam_size = beam_state.beam_size
        
        # 1. Flatten all potential next tokens into a single list for global comparison
        # Each entry: (parent_input_index, token_candidate, global_accumulated_score)
        all_potential_next: List[Tuple[int, 'BeamTokenCandidate', float]] = []

        for idx, beam_input in enumerate(new_candidates):
            parent_score = beam_input.candidate.score
            for token_choice in beam_input.children:
                # Accumulated log probability
                total_score = parent_score + token_choice.log_prob
                all_potential_next.append((idx, token_choice, total_score))

        # 2. Sort by accumulated score descending
        # Note: We sort by raw log-probs first; length penalty is usually used 
        # for the final ranking or during selection if preferred.
        all_potential_next.sort(key=lambda x: x[2], reverse=True)

        # 3. Take the top beam_size candidates
        top_selections = all_potential_next[:beam_size]

        # 4. Map these selections back to their parents to create BeamGenerateResults
        # Initialize results list with empty children
        results_map: Dict[int, List[BeamToken]] = {i: [] for i in range(len(new_candidates))}
        
        for parent_idx, token_choice, total_score in top_selections:
            results_map[parent_idx].append(
                BeamToken(
                    token_id=token_choice.token_id,
                    accumulated_score=total_score
                )
            )

        # 5. Construct final BeamGenerateResult objects in the original order
        final_results = []
        for i, beam_input in enumerate(new_candidates):
            final_results.append(
                BeamGenerateResult(
                    candidate=beam_input.candidate,
                    children=results_map[i]
                )
            )

        return final_results

    def should_stop(
        self,
        beam_state: BeamState,
        max_length: int,
        step: int
    ) -> bool:
        """
        Standard stopping logic for beam search.
        """
        # Condition 1: Max length reached
        if step >= max_length:
            return True

        # Condition 2: No active candidates left
        if not beam_state.candidates:
            return True

        # Condition 3: Early stopping 
        # (Stop if we have found 'beam_size' completed sequences)
        if self.early_stopping:
            if len(beam_state.finished_candidates) >= beam_state.beam_size:
                return True
        
        # Condition 4: All current beams are marked as finished
        if all(c.finished for c in beam_state.candidates):
            return True

        return False

    def get_final_sequences(
        self,
        beam_state: BeamState,
        num_return: Optional[int] = None,
    ) -> List[Tuple[List[int], float]]:
        """
        Get the final sequences with length penalty applied.

        Args:
            beam_state: Final beam state
            num_return: Number of sequences to return (default: beam_size)

        Returns:
            List of (sequence, score) tuples, sorted by score
        """
        if num_return is None:
            num_return = beam_state.beam_size

        return beam_state.get_final_sequences(num_return, self.length_penalty)

    def finalize_scores(self, candidates: List[BeamCandidate], step: int) -> List[Tuple[BeamCandidate, float]]:
        """
        Utility to rank candidates at the end of generation using length penalty.
        
        Returns:
            List of (candidate, normalized_score) sorted descending.
        """
        scored_candidates = []
        for c in candidates:
            # Reconstruct length from trie depth or step
            # Assuming step is provided or calculable
            norm_score = self._apply_length_penalty(c.score, step)
            scored_candidates.append((c, norm_score))
            
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)