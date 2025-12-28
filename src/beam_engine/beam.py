import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

from transformers import AutoTokenizer
from models.modeling_llama import LlamaForCausalLM
from page_table import PageTable
from attention_mode import AttentionMode


from beam_state import BeamState, TrieNode, BeamScoreItem, BeamCandidate
from beam_strategy import BeamStrategy, DiverseBeamSearchStrategy


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

    def generate(self, input_text: str,beam_size: int = 4, max_length: int = 50, num_return_sequences: int = 1,
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
        input_tokens = input_ids[0].tolist()

        # Initialize beam state
        beam_state = BeamState(beam_size, self.page_table)

        # PRE-ALLOCATE pages BEFORE calling the model
        new_nodes = beam_state.add_root_sequence(input_tokens)
        page_indices = [ n.page_id for n in new_nodes]

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mode=AttentionMode.PREFILL,
                page_table=self.page_table,
                page_indices=page_indices,
                last_page_len=len(new_nodes[len(new_nodes) - 1].tokens)
            )
            logits = outputs.logits

        # Get next token probabilities from the last position
        next_token_logits = logits[0, -1, :]  # Last token logits

        # Debug: Check what the model actually computed
        print(f"Debug: Prefill - Model outputs shape: {outputs.logits.shape}")
        print(f"Debug: Prefill - Next token logits stats - mean: {next_token_logits.mean().item():.4f}, std: {next_token_logits.std().item():.4f}")
        print(f"Debug: Prefill - Logits range - min: {next_token_logits.min().item():.4f}, max: {next_token_logits.max().item():.4f}")

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Get probabilities for next token
        next_token_probs = F.log_softmax(next_token_logits, dim=-1)

        # Generate initial beam candidates from top-k tokens
        top_k = min(beam_size * 2, next_token_probs.shape[0])
        top_probs, top_indices = torch.topk(next_token_probs, top_k)

        # Debug: Print prefill token generation details
        print(f"\n=== PREFILL TOKEN GENERATION DEBUG ===")
        print(f"Input sequence: {input_ids[0].tolist()}")
        print(f"Input text: '{self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)}'")
        print(f"Next token logits shape: {next_token_logits.shape}")
        print(f"Top {min(10, len(top_indices))} tokens:")
        for i, (prob, token_id) in enumerate(zip(top_probs[:10], top_indices[:10])):
            token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)
            print(f"  {i+1}. Token {token_id.item()}: '{token_text}' (prob: {prob.item():.4f})")
        print("=" * 50)


        for prob, token_id in zip(top_probs, top_indices):
            # Create new trie node by adding the new token
            beam_state.create_diverge(beam_state.root, token_id, prob)
        
        return 
        ## prefilling complete, now decoding
        step = 1

        # DECODE PHASE: Continue generating tokens for each beam candidate
        while not self.strategy.should_stop(beam_state, max_length, step):
            active_candidates = beam_state.get_active_candidates()
            if not active_candidates:
                break

            print(f"\n=== DECODE STEP {step} ===")
            print(f"Processing {len(active_candidates)} active candidates")

            # Prepare batch input for all active candidates
            batch_size = len(active_candidates)

            # Get the last token from each candidate for batch processing
            last_tokens = []
            batch_page_indices = []
            batch_last_page_lens = []

            for candidate in active_candidates:
                sequence = candidate.trie_node.get_full_sequence()
                last_token = sequence[-1] if sequence else 0
                last_tokens.append(last_token)

                # Copy page state since decode modifies it in place
                batch_page_indices.append(candidate.page_indices.copy())
                batch_last_page_lens.append(candidate.last_page_len)

            # Create batch input tensor [batch_size, 1] for decode step
            batch_input = torch.tensor(last_tokens, device=self.device).unsqueeze(1)  # [batch_size, 1]

            print(f"Debug: Batch input shape: {batch_input.shape}")
            print(f"Debug: Batch input tokens: {batch_input.flatten().tolist()}")
            print(f"Debug: Batch page indices: {batch_page_indices}")
            print(f"Debug: Batch last page lens: {batch_last_page_lens}")

            # Call model with true batch decode
            with torch.no_grad():
                outputs = self.model(
                    batch_input,  # [batch_size, 1]
                    attention_mode=AttentionMode.DECODE,
                    page_table=self.page_table,
                    batch_page_indices=batch_page_indices,  # List of page_indices for each sequence
                    batch_last_page_lens=batch_last_page_lens  # List of last_page_lens for each sequence
                )
                logits = outputs.logits

            # Extract logits [batch_size, 1, vocab_size] -> [batch_size, vocab_size]
            batch_logits = logits[:, -1, :]

            print(f"Debug: Batch logits shape: {batch_logits.shape}")

            # Apply temperature
            if temperature != 1.0:
                batch_logits = batch_logits / temperature

            # Get probabilities for next tokens
            batch_probs = F.log_softmax(batch_logits, dim=-1)  # [batch_size, vocab_size]

            # Generate new candidates for each active candidate
            new_candidates = []

            for candidate_idx, (candidate, next_token_probs) in enumerate(zip(active_candidates, batch_probs)):
                # Get top-k tokens for this candidate
                top_k = min(beam_size, next_token_probs.shape[0])
                top_probs, top_indices = torch.topk(next_token_probs, top_k)

                print(f"Debug: Candidate {candidate_idx} top tokens:")
                for j, (prob, token_id) in enumerate(zip(top_probs[:5], top_indices[:5])):
                    token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)
                    print(f"  {j+1}. Token {token_id.item()}: '{token_text}' (prob: {prob.item():.4f})")

                # Create new candidates by extending current candidate
                for prob, token_id in zip(top_probs, top_indices):
                    # Create new trie node by adding the new token
                    new_node = candidate.trie_node.add_sequence([token_id.item()])

                    # Check if sequence is finished
                    is_finished = token_id.item() == eos_token_id

                    # Use updated page state from batch decode call
                    # The batch decode function updated batch_page_indices and batch_last_page_lens in place
                    new_page_indices = batch_page_indices[candidate_idx].copy()
                    new_last_page_len = batch_last_page_lens[candidate_idx]

                    # Create new candidate with combined score
                    new_score = candidate.score + prob.item()
                    new_candidate = BeamCandidate(
                        trie_node=new_node,
                        score=new_score,
                        finished=is_finished,
                        page_indices=new_page_indices,
                        last_page_len=new_last_page_len
                    )
                    new_candidates.append(new_candidate)

            print(f"Debug: Generated {len(new_candidates)} new candidates from {len(active_candidates)} active candidates")

            # Let strategy select which candidates to keep
            selected_candidates = self.strategy.select_candidates(beam_state, new_candidates, step)

            print(f"Debug: Selected {len(selected_candidates)} candidates for next step")
            for i, candidate in enumerate(selected_candidates[:3]):  # Show top 3
                sequence = candidate.trie_node.get_full_sequence()
                text = self.tokenizer.decode(sequence, skip_special_tokens=False)
                print(f"  Candidate {i+1}: '{text}' (score: {candidate.score:.4f}, finished: {candidate.finished})")

            # Update beam state - clear current candidates and add selected ones
            beam_state.candidates = []
            for candidate in selected_candidates:
                beam_state.add_candidate(candidate)

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





def demo_diverse_beam_search(model, tokenizer):
    """Demonstrate diverse beam search generation."""
    print("=== Diverse Beam Search Demo ===")

    # Create diverse beam search strategy
    strategy = DiverseBeamSearchStrategy(
        num_groups=2,          # Divide beams into 2 groups for diversity
        diversity_penalty=0.5, # Penalty for generating similar tokens
        length_penalty=1.1     # Slight preference for longer sequences
    )

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
# Model and tokenizer setup
    device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Run demonstrations
    demo_diverse_beam_search(model, tokenizer)

    print("\n=== Strategy Comparison ===")
    print("Vanilla beam search: Selects candidates purely by score - simple and fast.")
    print("Diverse beam search: Promotes variety by grouping beams and penalizing similarity.")
    print("\nBoth strategies are now cleanly separated with no shared dependencies!")
    print("You can easily switch strategies by changing the BeamStrategy class.")
    print("Try implementing TopKBeamStrategy, NucleusBeamStrategy, or other custom approaches.")


