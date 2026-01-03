import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modeling_llama import LlamaForCausalLM
from page_table import PageTable
from attention_mode import AttentionMode


from beam_state import BeamState, TrieNode, BeamCandidate, BeamGenerateResult, BeamToken, BeamGenerateInput, BeamTokenCandidate
from beam_strategy import BeamStrategy, DiverseBeamSearchStrategy, VanillaBeamSearchStrategy


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

        # Compute position IDs for prefilling (sequential from 0 to len-1)
        prefill_position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                position_ids=prefill_position_ids,
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

        # After add_root_sequence, beam_state has 1 root candidate
        if not beam_state.candidates:
            raise RuntimeError("Expected root candidate after add_root_sequence")

        root_candidate = beam_state.candidates[0]

        # Create initial beam branches from prefill top-k tokens
        initial_tokens = []
        for i in range(min(beam_size, len(top_indices))):
            token_id = top_indices[i].item()
            log_prob = top_probs[i].item()
            initial_tokens.append(BeamToken(token_id=token_id, accumulated_score=log_prob))

        initial_result = BeamGenerateResult(candidate=root_candidate, children=initial_tokens)
        beam_state.add_filtered_results([initial_result])
        print(f"\nExpanded root into {len(beam_state.candidates)} beam candidates")

        # Decode loop
        for step in range(max_length - len(input_tokens) - 1):
            print(f"\n{'='*80}")
            print(f"{'='*80}")
            print(f"=== DECODE STEP {step + 1} ===")
            print(f"{'='*80}")
            print(f"[STEP START] {len(beam_state.candidates)} active candidates before this step")

            # Show current state of all candidates
            for cand_idx, candidate in enumerate(beam_state.candidates):
                sequence = []
                node = candidate.trie_node
                while node:
                    sequence = node.tokens + sequence
                    node = node.parent
                text = self.tokenizer.decode(sequence, skip_special_tokens=False)
                print(f"  Candidate {cand_idx}: score={candidate.score:.4f}, len={len(sequence)}, page_id={candidate.trie_node.page_id}")
                print(f"    Current text: '{text}'")

            if self.strategy.should_stop(beam_state, max_length, step):
                print("Stopping condition met")
                break

            # Get cascade input including query token IDs
            (qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr,
             paged_kv_last_page_len_arr, query_token_ids) = beam_state.get_cascade_input()

            print(f"\n[DECODE INPUT] Cascade levels: {len(qo_indptr_arr)}, Candidates: {len(beam_state.candidates)}, Query tokens: {len(query_token_ids)}")
            query_texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in query_token_ids.tolist()]
            print(f"[DECODE INPUT] Query tokens: {query_token_ids.tolist()}")
            print(f"[DECODE INPUT] Query texts: {query_texts}")

            # Prepare write locations for K/V cache
            cascade_write_page_indices = [candidate.trie_node.page_id for candidate in beam_state.candidates]
            cascade_write_positions = [len(candidate.trie_node.tokens) - 1 for candidate in beam_state.candidates]

            # Compute position IDs for each candidate's query token
            # Position ID = total sequence length - 1 (0-indexed position of current query token)
            position_ids_list = []
            for candidate in beam_state.candidates:
                # Walk up the trie to count total tokens
                total_tokens = 0
                node = candidate.trie_node
                while node is not None:
                    total_tokens += len(node.tokens)
                    node = node.parent
                # Current query token is at position (total_tokens - 1)
                position_ids_list.append(total_tokens - 1)

            position_ids = torch.tensor(position_ids_list, dtype=torch.int, device=self.device)  # [1, num_candidates]

            print(f"[DECODE INPUT] Write page indices: {cascade_write_page_indices}")
            print(f"[DECODE INPUT] Write positions: {cascade_write_positions}")
            print(f"[DECODE INPUT] Position IDs: {position_ids_list}")

            # query_token_ids are already in correct cascade order, just reshape for model
            decode_input_ids = query_token_ids.unsqueeze(0)  # [1, num_candidates]

            # Run model with cascade decode
            with torch.no_grad():
                outputs = self.model(
                    decode_input_ids,
                    position_ids=position_ids,
                    attention_mode=AttentionMode.DECODE,
                    page_table=self.page_table,
                    cascade_qo_indptr_arr=qo_indptr_arr,
                    cascade_kv_indptr_arr=paged_kv_indptr_arr,
                    cascade_kv_indices_arr=paged_kv_indices_arr,
                    cascade_kv_last_page_len_arr=paged_kv_last_page_len_arr,
                    cascade_write_page_indices=cascade_write_page_indices,
                    cascade_write_positions=cascade_write_positions,
                )
                logits = outputs.logits  # [1, num_candidates, vocab_size]

            # Create BeamGenerateInput for each candidate
            generate_inputs = []
            print(f"\n[TOKEN SELECTION] Processing {len(beam_state.candidates)} candidates")
            for cand_idx, candidate in enumerate(beam_state.candidates):
                candidate_logits = logits[0, cand_idx, :]

                if temperature != 1.0:
                    candidate_logits = candidate_logits / temperature

                candidate_probs = F.log_softmax(candidate_logits, dim=-1)

                top_k = min(beam_size * 2, candidate_probs.shape[0])
                top_probs, top_indices = torch.topk(candidate_probs, top_k)

                # Debug: show top tokens for this candidate
                print(f"  [Candidate {cand_idx}] score={candidate.score:.4f}, page_id={candidate.trie_node.page_id}")
                print(f"    Top 5 token candidates:")
                for k in range(min(5, len(top_indices))):
                    token_id = top_indices[k].item()
                    log_prob = top_probs[k].item()
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    print(f"      {k+1}. Token {token_id}: '{token_text}' (log_prob: {log_prob:.4f})")

                token_candidates = []
                for k in range(len(top_indices)):
                    token_id = top_indices[k].item()
                    log_prob = top_probs[k].item()
                    token_candidates.append(BeamTokenCandidate(token_id=token_id, log_prob=log_prob))

                generate_inputs.append(BeamGenerateInput(candidate=candidate, children=token_candidates))

            # Use strategy to select candidates
            print(f"\n[STRATEGY] Selecting candidates using {self.strategy.__class__.__name__}")
            filtered_results = self.strategy.select_candidates(beam_state, generate_inputs, step)

            print(f"\n[STRATEGY RESULTS] {len(filtered_results)} results selected")
            for res_idx, result in enumerate(filtered_results):
                if result.children:
                    token_ids = [c.token_id for c in result.children]
                    token_texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
                    print(f"  Result {res_idx}: {len(result.children)} tokens selected: {token_ids} -> {token_texts}")
                else:
                    print(f"  Result {res_idx}: NO CHILDREN (will be eliminated)")

            beam_state.add_filtered_results(filtered_results)
            print(f"\n[BEAM STATE] After filtering: {len(beam_state.candidates)} active candidates")

            # Show current sequences for each active candidate
            for cand_idx, candidate in enumerate(beam_state.candidates):
                # Reconstruct sequence from trie
                sequence = []
                node = candidate.trie_node
                while node:
                    sequence = node.tokens + sequence
                    node = node.parent
                text = self.tokenizer.decode(sequence, skip_special_tokens=False)
                print(f"  Candidate {cand_idx} (score={candidate.score:.4f}): {len(sequence)} tokens")
                print(f"    Text: '{text}'")

        # Get final sequences
        print(f"\n{'='*80}")
        print(f"=== FINAL RESULTS ===")
        print(f"{'='*80}")
        final_sequences = self.strategy.get_final_sequences(beam_state, num_return_sequences)

        generated_texts = []
        for idx, (tokens, score) in enumerate(final_sequences):
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
            print(f"\n[FINAL {idx+1}] Score: {score:.4f}, Tokens: {len(tokens)}")
            print(f"  Full token sequence: {tokens}")
            print(f"  Text: {text}")

        # Show page table statistics
        print(f"\n[PAGE TABLE] Statistics:")
        print(f"  Total pages allocated: (tracked in page_table)")
        print(f"  Page size: {self.page_size}")

        return generated_texts





def run_huggingface_beam_search(hf_model, tokenizer, prompt: str, beam_size: int = 4,
                                max_length: int = 50, num_return_sequences: int = 1,
                                temperature: float = 1.0):
    """Run HuggingFace's native beam search for comparison."""
    print("\n" + "=" * 80)
    print("=== HUGGINGFACE BEAM SEARCH (REFERENCE) ===")
    print("=" * 80)

    inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=beam_size,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=False,  # Deterministic beam search
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            repetition_penalty = 1,
            length_penalty = 1,

        )

    generated_texts = []
    print(f"\n[HF RESULTS] Generated {len(outputs.sequences)} sequences:")
    for idx, sequence in enumerate(outputs.sequences):
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_texts.append(text)
        tokens = sequence.tolist()
        print(f"\n[HF {idx+1}] Tokens: {len(tokens)}")
        print(f"  Full token sequence: {tokens}")
        print(f"  Text: {text}")

    return generated_texts


def demo_diverse_beam_search(model, tokenizer, hf_model=None):
    """Demonstrate diverse beam search generation."""
    print("=== Diverse Beam Search Demo ===")

    # Create diverse beam search strategy
    # strategy = DiverseBeamSearchStrategy(
    #     num_groups=1,          # Divide beams into 2 groups for diversity
    #     diversity_penalty=0.5, # Penalty for generating similar tokens
    #     length_penalty=1.1     # Slight preference for longer sequences
    # )

    generator = BeamSearchGenerator(model, tokenizer, VanillaBeamSearchStrategy())

    # Example prompts
    prompts = [
        "The future of artificial intelligence is",
        #"Once upon a time in a magical forest,",
        #"The best way to solve climate change is"
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 50)

        # Run HuggingFace beam search first (if available)
        if hf_model is not None:
            hf_texts = run_huggingface_beam_search(
                hf_model=hf_model,
                tokenizer=tokenizer,
                prompt=prompt,
                beam_size=1,
                max_length=9,
                num_return_sequences=1,
                temperature=1.0
            )

        # Generate diverse sequences with custom implementation
        print("\n" + "=" * 80)
        print("=== CUSTOM BEAM SEARCH (CASCADE ATTENTION) ===")
        print("=" * 80)
        generated_texts = generator.generate(
            input_text=prompt,
            beam_size=1,
            max_length=9,
            num_return_sequences=1,
            temperature=1.0
        )

        print("\n" + "=" * 80)
        print("=== COMPARISON ===")
        print("=" * 80)
        if hf_model is not None:
            print("\nHuggingFace Results:")
            for i, text in enumerate(hf_texts, 1):
                print(f"  {i}. {text}")

        print("\nCustom Cascade Results:")
        for i, text in enumerate(generated_texts, 1):
            print(f"  {i}. {text}")




if __name__ == "__main__":
    print("Loading models and tokenizer...")

    # Model and tokenizer setup
    device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "meta-llama/Llama-3.2-1B"

    # Load custom model with cascade attention
    print(f"Loading custom model from {model_name}...")
    model = LlamaForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)

    # Load HuggingFace reference model
    print(f"Loading HuggingFace reference model from {model_name}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Models loaded successfully!\n")

    # Run demonstrations with comparison
    demo_diverse_beam_search(model, tokenizer, hf_model)

    print("\n=== Strategy Comparison ===")
    print("Vanilla beam search: Selects candidates purely by score - simple and fast.")
    print("Diverse beam search: Promotes variety by grouping beams and penalizing similarity.")
    print("\nBoth strategies are now cleanly separated with no shared dependencies!")
    print("You can easily switch strategies by changing the BeamStrategy class.")
    print("Try implementing TopKBeamStrategy, NucleusBeamStrategy, or other custom approaches.")


