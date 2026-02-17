# SPDX-License-Identifier: Apache-2.0
# Adapted from HuggingFace Transformers LLaMA implementation
# Modified to use FlashInfer attention kernels with PageTable KV cache
"""Inference-only LLaMA model with FlashInfer paged attention."""

from typing import Optional, List, Tuple
from collections.abc import Iterable

import torch
from torch import nn
from transformers import LlamaConfig
import flashinfer

from beam_engine.page_table import PageTable


class LlamaRMSNorm(nn.Module):
    """RMSNorm implementation."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    """Rotary positional embedding."""
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor):
        """Return cos and sin for the given positions."""
        return self.cos_cached[positions], self.sin_cached[positions]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to q and k."""
    # q, k: [batch, seq_len, num_heads, head_dim]
    # cos, sin: [batch, seq_len, head_dim]
    cos = cos.unsqueeze(2)  # [batch, seq_len, 1, head_dim]
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    """LLaMA MLP with SiLU activation."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """LLaMA attention with FlashInfer paged attention."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, 'rope_theta', 10000.0),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        page_table: Optional[PageTable] = None,
        page_indices: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_last_page_len: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        prefill_wrapper: Optional[flashinfer.BatchPrefillWithPagedKVCacheWrapper] = None,
        decode_wrapper: Optional[flashinfer.BatchDecodeWithPagedKVCacheWrapper] = None,
    ) -> torch.Tensor:
        """
        Forward pass with FlashInfer paged attention.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            positions: [batch_size, seq_len] position indices
            page_table: PageTable for KV cache
            page_indices: [total_pages] page indices for paged attention
            kv_indptr: [batch_size + 1] cumulative page counts
            kv_last_page_len: [batch_size] tokens in last page per sequence
            is_prefill: Whether this is prefill (True) or decode (False)
            prefill_wrapper: FlashInfer prefill wrapper
            decode_wrapper: FlashInfer decode wrapper
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(positions.flatten())
        cos = cos.view(batch_size, seq_len, -1)
        sin = sin.view(batch_size, seq_len, -1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # If no page_table, use standard attention (for testing without paging)
        if page_table is None:
            # Standard scaled dot-product attention
            q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # GQA: expand k, v to match q heads
            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            # Causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            attn_weights.masked_fill_(causal_mask, float('-inf'))

            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            # Write K, V to page table
            # k, v: [batch, seq, num_kv_heads, head_dim] -> [batch * seq, num_kv_heads, head_dim]
            k_flat = k.reshape(-1, self.num_kv_heads, self.head_dim)
            v_flat = v.reshape(-1, self.num_kv_heads, self.head_dim)

            # For simplicity, assume single sequence and write to pages
            # In production, you'd handle batching properly
            # page_table.write_block(self.layer_idx, page_idx, k_flat, v_flat, index)

            # Get paged KV cache for this layer
            paged_kv_cache = page_table.kv_cache_at_layer[self.layer_idx]

            # Flatten q for flashinfer: [total_tokens, num_heads, head_dim]
            q_flat = q.reshape(-1, self.num_heads, self.head_dim)

            if is_prefill:
                # Use prefill wrapper
                attn_output = prefill_wrapper.run(
                    q_flat,
                    paged_kv_cache,
                )
            else:
                # Use decode wrapper
                attn_output = decode_wrapper.run(
                    q_flat,
                    paged_kv_cache,
                )

            # Reshape back: [batch, seq, num_heads, head_dim]
            attn_output = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Output projection
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    """Single LLaMA decoder layer."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        page_table: Optional[PageTable] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, page_table, **kwargs)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    """LLaMA model (transformer body without LM head)."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        page_table: Optional[PageTable] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, page_table, **kwargs)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """LLaMA model with causal LM head."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if configured
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        page_table: Optional[PageTable] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass returning hidden states.

        Args:
            input_ids: [batch_size, seq_len] input token IDs
            positions: [batch_size, seq_len] position indices
            page_table: Optional PageTable for KV cache

        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.model(input_ids, positions, page_table, **kwargs)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from hidden states.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(cls, model_name: str, dtype: torch.dtype = torch.float16, device: str = "cuda"):
        """Load pretrained weights from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoConfig

        # Load config
        config = AutoConfig.from_pretrained(model_name)

        # Create our model
        model = cls(config)

        # Load HuggingFace model to get weights
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu",  # Load to CPU first
        )

        # Copy weights
        model.load_state_dict(hf_model.state_dict(), strict=False)

        # Move to device
        model = model.to(device).to(dtype)
        model.eval()

        del hf_model
        torch.cuda.empty_cache()

        return model
