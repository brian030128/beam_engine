import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer
from models.modeling_llama import LlamaForCausalLM
from page_table import PageTable
from attention_mode import AttentionMode


class TrieNode:
    """
    Node in the prefix trie for storing beam search sequences.

    Each node can store multiple tokens up to max_tokens_per_node.
    When max length is reached, a new child node is created even if sequences don't diverge.
    """

    def __init__(self, tokens: Optional[List[int]] = None, parent: Optional['TrieNode'] = None,
                 max_tokens_per_node: int = 32):
        self.tokens = tokens or []  # List of tokens stored in this node
        self.parent = parent
        self.children: Dict[tuple, 'TrieNode'] = {}  # Key is tuple of tokens leading to child
        self.max_tokens_per_node = max_tokens_per_node

        # Calculate total depth (total tokens from root to this node)
        self.total_tokens = (parent.total_tokens if parent else 0) + len(self.tokens)

    def add_sequence(self, new_tokens: List[int]) -> 'TrieNode':
        """
        Add a sequence of tokens, creating nodes as needed.

        Returns the final node containing the sequence.
        """
        if not new_tokens:
            return self

        current_node = self
        remaining_tokens = new_tokens[:]

        while remaining_tokens:
            # Determine how many tokens to store in the next node
            tokens_for_node = remaining_tokens[:self.max_tokens_per_node]
            remaining_tokens = remaining_tokens[self.max_tokens_per_node:]

            # Use tuple as key for children dict
            tokens_key = tuple(tokens_for_node)

            # Check if child with these tokens already exists
            if tokens_key not in current_node.children:
                current_node.children[tokens_key] = TrieNode(
                    tokens=list(tokens_for_node),
                    parent=current_node,
                    max_tokens_per_node=self.max_tokens_per_node
                )

            current_node = current_node.children[tokens_key]

        return current_node

    def get_full_sequence(self) -> List[int]:
        """Reconstruct the full sequence from root to this node."""
        sequence = []
        current = self
        path = []

        # Collect all nodes from this node back to root
        while current is not None:
            path.append(current)
            current = current.parent

        # Reverse to go from root to current node, and collect tokens
        path.reverse()
        for node in path:
            sequence.extend(node.tokens)

        return sequence

    def get_sequence_tensor(self) -> torch.Tensor:
        """Get the full sequence as a PyTorch tensor."""
        tokens = self.get_full_sequence()
        return torch.tensor(tokens) if tokens else torch.tensor([])

    def __repr__(self):
        return f"TrieNode(tokens={self.tokens}, total_tokens={self.total_tokens}, children={len(self.children)})"


@dataclass
class BeamCandidate:
    """Represents a single beam candidate with its sequence and score."""
    trie_node: TrieNode              # Reference to trie node containing the sequence
    score: float                     # Log probability score
    finished: bool = False           # Whether sequence has ended (EOS token)
    page_indices: Optional[List[int]] = None  # Page indices in the page table for this candidate
    last_page_len: Optional[int] = None       # Length of the last page for this candidate

    @property
    def sequence(self) -> torch.Tensor:
        """Get the full sequence as a tensor."""
        return self.trie_node.get_sequence_tensor()

    @property
    def length(self) -> int:
        """Get the total length of the sequence."""
        return self.trie_node.total_tokens

class BeamState:
    """Manages the current state of all beam candidates using a trie structure."""

    def __init__(self, beam_size: int, device: torch.device, max_tokens_per_node: int = 32):
        self.beam_size = beam_size
        self.device = device
        self.max_tokens_per_node = max_tokens_per_node
        self.root = TrieNode(max_tokens_per_node=max_tokens_per_node)
        self.candidates: List[BeamCandidate] = []
        self.finished_candidates: List[BeamCandidate] = []

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


class BeamStrategy(ABC):
    """Abstract base class for beam search strategies."""

    @abstractmethod
    def select_candidates(self, beam_state: BeamState, new_candidates: List[BeamCandidate],
                         step: int) -> List[BeamCandidate]:
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


class VanillaBeamSearchStrategy(BeamStrategy):
    """
    Standard beam search strategy that selects candidates purely by score.

    This is the classic beam search implementation that simply keeps the
    top-k candidates with the highest scores at each step.
    """

    def __init__(self, beam_size: int, length_penalty: float = 1.0):
        """
        Args:
            beam_size: Number of beams to maintain
            length_penalty: Length normalization penalty (>1 favors longer sequences)
        """
        self.beam_size = beam_size
        self.length_penalty = length_penalty

    def select_candidates(self, beam_state: BeamState, new_candidates: List[BeamCandidate],
                         step: int) -> List[BeamCandidate]:
        """Select top candidates by score."""
        if not new_candidates:
            return []

        # Sort by score (with optional length penalty)
        def get_score(candidate):
            if self.length_penalty == 1.0:
                return candidate.score
            else:
                return candidate.score / (candidate.length ** self.length_penalty)

        sorted_candidates = sorted(new_candidates, key=get_score, reverse=True)
        return sorted_candidates[:self.beam_size]

    def should_stop(self, beam_state: BeamState, max_length: int, step: int) -> bool:
        """Stop when max length reached or all beams finished."""
        active_candidates = beam_state.get_active_candidates()

        # Stop if no active candidates or max length reached
        if not active_candidates or step >= max_length:
            return True

        # Stop if we have enough finished candidates and they're better than active ones
        if len(beam_state.finished_candidates) >= self.beam_size:
            best_finished_score = max(c.score / c.length for c in beam_state.finished_candidates)
            best_active_score = max(c.score / c.length for c in active_candidates) if active_candidates else 0
            return best_finished_score > best_active_score

        return False




class DiverseBeamSearchStrategy(BeamStrategy):
    """
    Implements diverse beam search strategy.

    Diverse beam search encourages diversity by:
    1. Grouping beams and ensuring diversity within groups
    2. Applying diversity penalties to similar sequences
    3. Balancing exploration vs exploitation

    This strategy manages its own grouping logic internally without
    polluting the general BeamCandidate structure.
    """

    def __init__(self, beam_size: int, num_groups: int = 2, diversity_penalty: float = 0.5,
                 length_penalty: float = 1.0):
        """
        Args:
            beam_size: Total number of beams
            num_groups: Number of diverse groups to maintain
            diversity_penalty: Penalty for generating similar tokens to other groups
            length_penalty: Length normalization penalty (>1 favors longer sequences)
        """
        self.beam_size = beam_size
        self.num_groups = num_groups
        self.group_size = beam_size // num_groups
        self.diversity_penalty = diversity_penalty
        self.length_penalty = length_penalty

        # Internal state for tracking candidate groups
        self._candidate_to_group: Dict[int, int] = {}  # Maps candidate id to group id

        if beam_size % num_groups != 0:
            raise ValueError(f"beam_size ({beam_size}) must be divisible by num_groups ({num_groups})")

    def select_candidates(self, beam_state: BeamState, new_candidates: List[BeamCandidate],
                         step: int) -> List[BeamCandidate]:
        """Select diverse candidates using group-based selection."""
        if not new_candidates:
            return []

        # Sort all candidates by score
        all_candidates = sorted(new_candidates, key=lambda x: x.score, reverse=True)

        # Initialize groups for tracking diverse candidates
        selected_groups: List[List[BeamCandidate]] = [[] for _ in range(self.num_groups)]
        selected_tokens_by_group = [set() for _ in range(self.num_groups)]

        # Clear previous candidate to group mapping for new step
        self._candidate_to_group.clear()

        # Assign candidates to groups with diversity penalty
        for candidate in all_candidates:
            if sum(len(group) for group in selected_groups) >= self.beam_size:
                break

            # Get the last token of this candidate
            last_token = candidate.sequence[-1].item() if candidate.sequence.numel() > 0 else None

            # Find the best group for this candidate
            best_group_idx = self._find_best_group(
                candidate, selected_groups, selected_tokens_by_group, last_token
            )

            if best_group_idx is not None and len(selected_groups[best_group_idx]) < self.group_size:
                # Track which group this candidate belongs to
                self._candidate_to_group[id(candidate)] = best_group_idx
                selected_groups[best_group_idx].append(candidate)
                if last_token is not None:
                    selected_tokens_by_group[best_group_idx].add(last_token)

        # Flatten selected candidates
        selected_candidates = []
        for group in selected_groups:
            selected_candidates.extend(group)

        return selected_candidates[:self.beam_size]

    def _find_best_group(self, candidate: BeamCandidate, selected_groups: List[List[BeamCandidate]],
                        selected_tokens_by_group: List[set], last_token: Optional[int]) -> Optional[int]:
        """Find the best group for a candidate considering diversity."""
        best_group_idx = None
        best_score = float('-inf')

        for group_idx, group in enumerate(selected_groups):
            if len(group) >= self.group_size:
                continue

            # Calculate diversity penalty
            diversity_penalty = 0.0
            if last_token is not None:
                # Penalty for using tokens already used by other groups
                for other_group_idx, other_tokens in enumerate(selected_tokens_by_group):
                    if other_group_idx != group_idx and last_token in other_tokens:
                        diversity_penalty += self.diversity_penalty

            # Adjusted score with diversity penalty
            adjusted_score = candidate.score - diversity_penalty

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_group_idx = group_idx

        return best_group_idx

    def should_stop(self, beam_state: BeamState, max_length: int, step: int) -> bool:
        """Stop if all beams are finished or max length reached."""
        active_candidates = beam_state.get_active_candidates()

        # Stop if no active candidates or max length reached
        if not active_candidates or step >= max_length:
            return True

        # Stop if we have enough finished candidates and they're better than active ones
        if len(beam_state.finished_candidates) >= self.beam_size:
            best_finished_score = max(c.score / c.length for c in beam_state.finished_candidates)
            best_active_score = max(c.score / c.length for c in active_candidates) if active_candidates else 0
            return best_finished_score > best_active_score

        return False


class BeamSearchGenerator:
    """Main beam search generator with pluggable strategies."""

    def __init__(self, model, tokenizer, strategy: BeamStrategy, page_size: int = 64):
        """
        Args:
            model: The language model
            tokenizer: The tokenizer
            strategy: The beam search strategy to use
            page_size: Size of each page in the page table (tokens per page)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.device = next(model.parameters()).device
        self.page_size = page_size

        # Extract model configuration for page table
        config = self.model.config
        self.page_table = PageTable(
            layer_num=config.num_hidden_layers,
            page_size=page_size,
            max_num_pages=1024,  # Adjust based on memory constraints
            head_num=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            device=self.device,
            store_dtype=torch.float16  # Use half precision for memory efficiency
        )

    def generate(self, input_text: str, max_length: int = 50, num_return_sequences: int = 1,
                 temperature: float = 1.0, pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> List[str]:
        """
        Generate sequences using beam search with the configured strategy.

        Args:
            input_text: Input text to continue
            max_length: Maximum length of generated sequences
            num_return_sequences: Number of sequences to return
            temperature: Sampling temperature
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID

        Returns:
            List of generated text sequences
        """
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # Initialize beam state
        beam_state = BeamState(self.strategy.beam_size, self.device)

        # Create initial trie node with input sequence
        input_tokens = input_ids[0].tolist()
        initial_node = beam_state.root.add_sequence(input_tokens)

        # Create initial beam candidate (no page info yet, will be set during prefill)
        initial_candidate = BeamCandidate(
            trie_node=initial_node,
            score=0.0
        )
        beam_state.add_candidate(initial_candidate)

        # Generation loop
        step = 0
        is_prefill = True  # First step is prefill, subsequent steps are decode

        while not self.strategy.should_stop(beam_state, max_length, step):
            active_candidates = beam_state.get_active_candidates()
            if not active_candidates:
                break

            # For single-batch beam search, we process one candidate at a time
            # This is simpler for FlashInfer integration
            new_candidates = []

            for candidate in active_candidates:
                if is_prefill:
                    # First step: use prefill kernel for the full input sequence
                    sequence = candidate.sequence.unsqueeze(0).to(self.device)
                    
                    # PRE-ALLOCATE pages BEFORE calling the model
                    seq_len = sequence.shape[1]
                    pages_needed = (seq_len + self.page_size - 1) // self.page_size
                    page_indices = [self.page_table.allocate_block() for _ in range(pages_needed)]
                    last_page_len = seq_len % self.page_size
                    if last_page_len == 0 and seq_len > 0:
                        last_page_len = self.page_size

                    with torch.no_grad():
                        outputs = self.model(
                            sequence,
                            attention_mode=AttentionMode.PREFILL,
                            page_table=self.page_table,
                            page_indices=page_indices,        # <-- ADD THIS
                            last_page_len=last_page_len       # <-- ADD THIS
                        )
                        logits = outputs.logits

                    # Compute page info based on sequence length
                    seq_len = sequence.shape[1]
                    pages_needed = (seq_len + self.page_size - 1) // self.page_size
                    page_indices = list(range(len(self.page_table.allocated_pages) - pages_needed, len(self.page_table.allocated_pages)))
                    last_page_len = seq_len % self.page_size
                    if last_page_len == 0 and seq_len > 0:
                        last_page_len = self.page_size

                    # Update candidate with page info
                    candidate.page_indices = page_indices
                    candidate.last_page_len = last_page_len

                    # Get next token probabilities
                    next_token_logits = logits[0, -1, :]  # Last token logits

                    # Debug: Check what the model actually computed
                    print(f"Debug: Model outputs shape: {outputs.logits.shape}")
                    print(f"Debug: Hidden states after attention (sample): {outputs.logits[0, -1, :10]}")
                    print(f"Debug: Next token logits stats - mean: {next_token_logits.mean().item():.4f}, std: {next_token_logits.std().item():.4f}")
                    print(f"Debug: Logits range - min: {next_token_logits.min().item():.4f}, max: {next_token_logits.max().item():.4f}")

                    # Let's also check a few positions to see if it's always predicting start-of-sentence
                    for pos in [0, 3, 6]:  # Check beginning, middle, and end
                        pos_logits = logits[0, pos, :]
                        pos_probs = F.log_softmax(pos_logits, dim=-1)
                        top_prob, top_token = torch.topk(pos_probs, 1)
                        token_text = self.tokenizer.decode([top_token.item()], skip_special_tokens=False)
                        print(f"Debug: Position {pos} top token: {top_token.item()} '{token_text}' (prob: {top_prob.item():.4f})")
                else:
                    # Subsequent steps: use decode kernel for single token
                    # Get the last token of the sequence and add batch dimension
                    last_token = candidate.sequence[-1:].unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(
                            last_token,
                            attention_mode=AttentionMode.DECODE,
                            page_table=self.page_table,
                            page_indices=candidate.page_indices,
                            last_page_len=candidate.last_page_len
                        )
                        logits = outputs.logits

                    # Get next token probabilities
                    next_token_logits = logits[0, -1, :]  # Last token logits

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Get probabilities for next token
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)

                # Generate new candidates from top-k tokens
                top_k = min(self.strategy.beam_size * 2, next_token_probs.shape[0])
                top_probs, top_indices = torch.topk(next_token_probs, top_k)

                # Debug: Print first token details (only on first iteration)
                if is_prefill and len(active_candidates) == 1:
                    print(f"\n=== PREFILL TOKEN GENERATION DEBUG ===")
                    print(f"Input sequence: {candidate.sequence.tolist()}")
                    print(f"Input text: '{self.tokenizer.decode(candidate.sequence.tolist(), skip_special_tokens=False)}'")
                    print(f"Next token logits shape: {next_token_logits.shape}")
                    print(f"Top {min(10, len(top_indices))} tokens:")
                    for i, (prob, token_id) in enumerate(zip(top_probs[:10], top_indices[:10])):
                        token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)
                        print(f"  {i+1}. Token {token_id.item()}: '{token_text}' (prob: {prob.item():.4f})")
                    print(f"Selected top-{top_k} tokens for beam expansion")
                    print("=" * 50)

                for prob, token_id in zip(top_probs, top_indices):
                    # Create new trie node by adding the new token
                    new_node = candidate.trie_node.add_sequence([token_id.item()])
                    new_score = candidate.score + prob.item()

                    # Check if sequence is finished
                    is_finished = token_id.item() == eos_token_id

                    # Create new candidate with updated page info
                    new_last_page_len = candidate.last_page_len + 1 if candidate.last_page_len is not None else None

                    new_candidate = BeamCandidate(
                        trie_node=new_node,
                        score=new_score,
                        finished=is_finished,
                        page_indices=candidate.page_indices.copy() if candidate.page_indices else None,
                        last_page_len=new_last_page_len
                    )
                    new_candidates.append(new_candidate)

            # Let strategy select which candidates to keep
            selected_candidates = self.strategy.select_candidates(beam_state, new_candidates, step)

            # Debug: Show selected candidates after first step
            if is_prefill:
                print(f"\n=== SELECTED CANDIDATES AFTER PREFILL ===")
                print(f"Generated {len(new_candidates)} new candidates, selected {len(selected_candidates)}")
                for i, candidate in enumerate(selected_candidates):
                    sequence = candidate.trie_node.get_full_sequence()
                    text = self.tokenizer.decode(sequence, skip_special_tokens=False)
                    print(f"  Candidate {i+1}: '{text}' (score: {candidate.score:.4f})")
                print("=" * 50)

            # Update beam state
            beam_state.candidates = []
            for candidate in selected_candidates:
                beam_state.add_candidate(candidate)

            # After first step, switch to decode mode
            is_prefill = False
            step += 1

        # Get final results
        final_candidates = beam_state.get_best_finished(num_return_sequences)

        # Decode sequences
        results = []
        for candidate in final_candidates:
            # Get full sequence and remove input tokens
            full_sequence = candidate.trie_node.get_full_sequence()
            generated_tokens = full_sequence[len(input_tokens):]  # Remove input portion
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(generated_text)

        return results


# Model and tokenizer setup
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def demo_diverse_beam_search():
    """Demonstrate diverse beam search generation."""
    print("=== Diverse Beam Search Demo ===")

    # Create diverse beam search strategy
    strategy = DiverseBeamSearchStrategy(
        beam_size=4,           # Total number of beams
        num_groups=2,          # Divide beams into 2 groups for diversity
        diversity_penalty=0.5, # Penalty for generating similar tokens
        length_penalty=1.1     # Slight preference for longer sequences
    )

    # Create generator
    generator = BeamSearchGenerator(model, tokenizer, strategy)

    # Example prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a magical forest,",
        "The best way to solve climate change is"
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 50)

        # Generate diverse sequences
        generated_texts = generator.generate(
            input_text=prompt,
            max_length=30,
            num_return_sequences=4,
            temperature=0.8
        )

        for i, text in enumerate(generated_texts, 1):
            print(f"{i}. {text}")




if __name__ == "__main__":
    print("Loading model and tokenizer...")

    # Run demonstrations
    demo_diverse_beam_search()

    print("\n=== Strategy Comparison ===")
    print("Vanilla beam search: Selects candidates purely by score - simple and fast.")
    print("Diverse beam search: Promotes variety by grouping beams and penalizing similarity.")
    print("\nBoth strategies are now cleanly separated with no shared dependencies!")
    print("You can easily switch strategies by changing the BeamStrategy class.")
    print("Try implementing TopKBeamStrategy, NucleusBeamStrategy, or other custom approaches.")


