import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.page_table import PageTable
from beam_engine.attention_mode import AttentionMode
from beam_engine.beam_state import BeamState, TrieNode, BeamCandidate, BeamGenerateResult, BeamToken, BeamGenerateInput, BeamTokenCandidate
from beam_engine.beam_strategy import BeamStrategy, DiverseBeamSearchStrategy, VanillaBeamSearchStrategy
from beam_engine.logger import init_logger

logger = init_logger(__name__)


class BeamSearchGenerator:
    """Main beam search generator with pluggable strategies."""

    def __init__(self, model, tokenizer, strategy: BeamStrategy, page_size: int = 8, kernel_type: str = "cascade"):
        """
        Args:
            model: The language model
            tokenizer: The tokenizer
            strategy: The beam search strategy to use
            page_size: Size of each page in the page table (tokens per page)
            kernel_type: Attention kernel type - "cascade" or "fasttree" (default: "cascade")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.device = next(model.parameters()).device
        self.page_size = page_size
        self.kernel_type = kernel_type

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
        logger.debug(f"Debug: Prefill - Model outputs shape: {outputs.logits.shape}")
        logger.debug(f"Debug: Prefill - Next token logits stats - mean: {next_token_logits.mean().item():.4f}, std: {next_token_logits.std().item():.4f}")
        logger.debug(f"Debug: Prefill - Logits range - min: {next_token_logits.min().item():.4f}, max: {next_token_logits.max().item():.4f}")

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Get probabilities for next token
        next_token_probs = F.log_softmax(next_token_logits, dim=-1)

        # Generate initial beam candidates from top-k tokens
        top_k = min(beam_size * 2, next_token_probs.shape[0])
        top_probs, top_indices = torch.topk(next_token_probs, top_k)

        # Debug: Print prefill token generation details
        logger.debug(f"\n=== PREFILL TOKEN GENERATION DEBUG ===")
        logger.debug(f"Input sequence: {input_ids[0].tolist()}")
        logger.debug(f"Input text: '{self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)}'")
        logger.debug(f"Next token logits shape: {next_token_logits.shape}")
        logger.debug(f"Top {min(10, len(top_indices))} tokens:")
        for i, (prob, token_id) in enumerate(zip(top_probs[:10], top_indices[:10])):
            token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)
            logger.debug(f"  {i+1}. Token {token_id.item()}: '{token_text}' (prob: {prob.item():.4f})")
        logger.debug("=" * 50)

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
        logger.debug(f"\nExpanded root into {len(beam_state.candidates)} beam candidates")

        # Decode loop
        for step in range(max_length - len(input_tokens) - 1):
            logger.debug(f"\n{'='*80}")
            logger.debug(f"{'='*80}")
            logger.debug(f"=== DECODE STEP {step + 1} ===")
            logger.debug(f"{'='*80}")
            logger.debug(f"[STEP START] {len(beam_state.candidates)} active candidates before this step")

            # Show current state of all candidates
            for cand_idx, candidate in enumerate(beam_state.candidates):
                sequence = []
                node = candidate.trie_node
                while node:
                    sequence = node.tokens + sequence
                    node = node.parent
                text = self.tokenizer.decode(sequence, skip_special_tokens=False)
                logger.debug(f"  Candidate {cand_idx}: score={candidate.score:.4f}, len={len(sequence)}, page_id={candidate.trie_node.page_id}")
                logger.debug(f"    Current text: '{text}'")

            if self.strategy.should_stop(beam_state, max_length, step):
                logger.debug("Stopping condition met")
                break

            # Get attention kernel input based on kernel type
            if self.kernel_type == "cascade":
                # Get cascade input including query token IDs and pre-allocated write tensors
                (qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr,
                 paged_kv_last_page_len_arr, query_token_ids,
                 write_batch_indices, write_kv_indptr) = beam_state.get_cascade_input()

                logger.debug(f"\n[DECODE INPUT] Cascade levels: {len(qo_indptr_arr)}, Candidates: {len(beam_state.candidates)}, Query tokens: {len(query_token_ids)}")
                query_texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in query_token_ids.tolist()]
                logger.debug(f"[DECODE INPUT] Query tokens: {query_token_ids.tolist()}")
                logger.debug(f"[DECODE INPUT] Query texts: {query_texts}")

                # Prepare write locations for K/V cache
                cascade_write_page_indices = torch.tensor(
                    [candidate.trie_node.page_id for candidate in beam_state.candidates],
                    dtype=torch.int32,
                    device=self.device
                )
                cascade_write_positions = torch.tensor(
                    [len(candidate.trie_node.tokens) - 1 for candidate in beam_state.candidates],
                    dtype=torch.int32,
                    device=self.device
                )

            elif self.kernel_type == "fasttree":
                # Get FastTree input including tree structure and req_to_token mapping
                (tree_nodes, req_to_token, query_token_ids,
                 cascade_write_page_indices, cascade_write_positions) = beam_state.get_fasttree_input()

                logger.debug(f"\n[DECODE INPUT] FastTree nodes: {len(tree_nodes)}, Candidates: {len(beam_state.candidates)}, Query tokens: {len(query_token_ids)}")
                query_texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in query_token_ids.tolist()]
                logger.debug(f"[DECODE INPUT] Query tokens: {query_token_ids.tolist()}")
                logger.debug(f"[DECODE INPUT] Query texts: {query_texts}")

                # For FastTree, we use the same write tensors that cascade uses
                write_batch_indices = torch.arange(len(beam_state.candidates), dtype=torch.int32, device=self.device)
                write_kv_indptr = torch.arange(len(beam_state.candidates) + 1, dtype=torch.int32, device=self.device)

            else:
                raise ValueError(f"Unknown kernel_type: {self.kernel_type}. Must be 'cascade' or 'fasttree'.")

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

            position_ids = torch.tensor([position_ids_list], dtype=torch.int, device=self.device)  # [1, num_candidates]

            logger.debug(f"[DECODE INPUT] Write page indices: {cascade_write_page_indices}")
            logger.debug(f"[DECODE INPUT] Write positions: {cascade_write_positions}")
            logger.debug(f"[DECODE INPUT] Position IDs: {position_ids_list}")

            # query_token_ids are already in correct cascade order, just reshape for model
            decode_input_ids = query_token_ids.unsqueeze(0)  # [1, num_candidates]

            # Run model with appropriate decode kernel
            with torch.no_grad():
                if self.kernel_type == "cascade":
                    outputs = self.model(
                        decode_input_ids,
                        position_ids=position_ids,
                        attention_mode=AttentionMode.DECODE,
                        kernel_type="cascade",
                        page_table=self.page_table,
                        cascade_qo_indptr_arr=qo_indptr_arr,
                        cascade_kv_indptr_arr=paged_kv_indptr_arr,
                        cascade_kv_indices_arr=paged_kv_indices_arr,
                        cascade_kv_last_page_len_arr=paged_kv_last_page_len_arr,
                        cascade_write_page_indices=cascade_write_page_indices,
                        cascade_write_positions=cascade_write_positions,
                        cascade_write_batch_indices=write_batch_indices,
                        cascade_write_kv_indptr=write_kv_indptr,
                    )

                elif self.kernel_type == "fasttree":
                    outputs = self.model(
                        decode_input_ids,
                        position_ids=position_ids,
                        attention_mode=AttentionMode.DECODE,
                        kernel_type="fasttree",
                        page_table=self.page_table,
                        fasttree_tree_nodes=tree_nodes,
                        fasttree_req_to_token=req_to_token,
                        cascade_write_page_indices=cascade_write_page_indices,
                        cascade_write_positions=cascade_write_positions,
                        cascade_write_batch_indices=write_batch_indices,
                        cascade_write_kv_indptr=write_kv_indptr,
                    )

                logits = outputs.logits  # [1, num_candidates, vocab_size]

            # Vectorized operations: process all candidates at once
            all_logits = logits[0]  # [num_candidates, vocab_size]

            if temperature != 1.0:
                all_logits = all_logits / temperature

            all_probs = F.log_softmax(all_logits, dim=-1)  # [num_candidates, vocab_size]

            # Compute topk for all candidates at once
            top_k = min(beam_size * 2, all_probs.shape[1]) if beam_size != 1 else 1
            all_top_probs, all_top_indices = torch.topk(all_probs, top_k, dim=-1)  # [num_candidates, top_k]

            # Create BeamGenerateInput for each candidate
            generate_inputs = []
            logger.debug(f"\n[TOKEN SELECTION] Processing {len(beam_state.candidates)} candidates")

            for cand_idx, candidate in enumerate(beam_state.candidates):
                # Get pre-computed topk results for this candidate
                top_probs = all_top_probs[cand_idx]
                top_indices = all_top_indices[cand_idx]

                # Debug: show top tokens for this candidate
                logger.debug(f"  [Candidate {cand_idx}] score={candidate.score:.4f}, page_id={candidate.trie_node.page_id}")
                logger.debug(f"    Top 5 token candidates:")
                for k in range(min(5, len(top_indices))):
                    token_id = top_indices[k].item()
                    log_prob = top_probs[k].item()
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    logger.debug(f"      {k+1}. Token {token_id}: '{token_text}' (log_prob: {log_prob:.4f})")

                token_candidates = []
                for k in range(len(top_indices)):
                    token_id = top_indices[k].item()
                    log_prob = top_probs[k].item()
                    token_candidates.append(BeamTokenCandidate(token_id=token_id, log_prob=log_prob))

                generate_inputs.append(BeamGenerateInput(candidate=candidate, children=token_candidates))

            # Use strategy to select candidates
            logger.debug(f"\n[STRATEGY] Selecting candidates using {self.strategy.__class__.__name__}")
            filtered_results = self.strategy.select_candidates(beam_state, generate_inputs, step)

            logger.debug(f"\n[STRATEGY RESULTS] {len(filtered_results)} results selected")
            for res_idx, result in enumerate(filtered_results):
                if result.children:
                    token_ids = [c.token_id for c in result.children]
                    token_texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
                    logger.debug(f"  Result {res_idx}: {len(result.children)} tokens selected: {token_ids} -> {token_texts}")
                else:
                    logger.debug(f"  Result {res_idx}: NO CHILDREN (will be eliminated)")

            beam_state.add_filtered_results(filtered_results)

            logger.debug(f"\n[BEAM STATE] After filtering: {len(beam_state.candidates)} active candidates")

            # Show current sequences for each active candidate
            for cand_idx, candidate in enumerate(beam_state.candidates):
                # Reconstruct sequence from trie
                sequence = []
                node = candidate.trie_node
                while node:
                    sequence = node.tokens + sequence
                    node = node.parent
                text = self.tokenizer.decode(sequence, skip_special_tokens=False)
                logger.debug(f"  Candidate {cand_idx} (score={candidate.score:.4f}): {len(sequence)} tokens")
                logger.debug(f"    Text: '{text}'")

        # Get final sequences
        logger.info(f"\n{'='*80}")
        logger.info(f"=== FINAL RESULTS ===")
        logger.info(f"{'='*80}")
        final_sequences = self.strategy.get_final_sequences(beam_state, num_return_sequences)

        generated_texts = []
        for idx, (tokens, score) in enumerate(final_sequences):
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
            logger.info(f"\n[FINAL {idx+1}] Score: {score:.4f}, Tokens: {len(tokens)}")
            logger.info(f"  Full token sequence: {tokens}")
            logger.info(f"  Text: {text}")

        # Show page table statistics
        logger.debug(f"\n[PAGE TABLE] Statistics:")
        logger.debug(f"  Total pages allocated: (tracked in page_table)")
        logger.debug(f"  Page size: {self.page_size}")

        return generated_texts

