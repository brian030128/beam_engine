# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig
from attention_mode import AttentionMode
from page_table import PageTable
import flashinfer


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def flashinfer_prefill_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    page_table: PageTable,
    page_indices: list,      # <-- ADD THIS PARAMETER
    last_page_len: int,      # <-- ADD THIS PARAMETER  
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    FlashInfer prefill attention using batch prefill with paged KV cache.
    page_indices must be pre-allocated BEFORE calling this function.
    """
    seq_len, num_heads, head_dim = query.shape


    # DON'T allocate pages here - use the pre-allocated page_indices
    # Write KV states to the pre-allocated pages for THIS LAYER
    remaining_tokens = seq_len
    current_pos = 0

    for i, page_idx in enumerate(page_indices):
        tokens_in_this_page = min(remaining_tokens, page_table.page_size)

        if tokens_in_this_page > 0:
            page_key = key[0, current_pos:current_pos + tokens_in_this_page, :, :]
            page_value = value[0, current_pos:current_pos + tokens_in_this_page, :, :]

            # Determine the write index for this page
            write_index = 0 if i < len(page_indices) - 1 else 0  # Always start at 0 for prefill
            
            page_table.write_block(
                layer=module.layer_idx,
                page_idx=page_idx,
                key=page_key,
                value=page_value,
                index=write_index
            )

        current_pos += tokens_in_this_page
        remaining_tokens -= tokens_in_this_page

    # Prepare FlashInfer - use the passed page_indices
    device = query.device
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor(page_indices, dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, len(page_indices)], dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)

    print(f"Debug: Created index tensors")

    # Create workspace buffer (128MB recommended)
    workspace_size = 128 * 1024 * 1024  # 128MB
    workspace_buffer = torch.empty(workspace_size, dtype=torch.uint8, device=device)

    # Initialize prefill wrapper
    print(f"Debug: Initializing prefill wrapper")
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD"
    )
    print(f"Debug: Prefill wrapper initialized")

    # Plan the attention computation
    print(f"Debug: Planning attention computation")
    # Extract correct head counts for Grouped Query Attention (GQA)
    num_qo_heads = num_heads  # Query heads from query tensor
    num_kv_heads = key.shape[2]  # KV heads from key tensor shape [batch, seq_len, num_kv_heads, head_dim]
    print(f"Debug: num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}")

    prefill_wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,  # num_qo_heads (query heads)
        num_kv_heads,  # num_kv_heads (key-value heads)
        head_dim,
        page_table.page_size,
        causal=True,
        pos_encoding_mode="None"
    )
    print(f"Debug: Attention computation planned")

    # FlashInfer expects query without batch dimension: [seq_len, num_heads, head_dim]
    print(f"Debug: Query shape for FlashInfer: {query.shape}")
    query_flashinfer = query[0]  # Remove batch dimension: [seq_len, num_heads, head_dim]
    print(f"Debug: Query ready for FlashInfer: {query_flashinfer.shape}")

    # Get paged KV cache for this layer
    paged_kv_cache = page_table.kv_cache_at_layer[module.layer_idx]
    print(f"Debug: Got paged KV cache shape {paged_kv_cache.shape}")

    # Run prefill attention
    print(f"Debug: Running prefill attention")
    print(f"Debug: query_flashinfer shape: {query_flashinfer.shape}, dtype: {query_flashinfer.dtype}")
    print(f"Debug: paged_kv_cache shape: {paged_kv_cache.shape}, dtype: {paged_kv_cache.dtype}")
    print(f"Debug: qo_indptr: {qo_indptr}")
    print(f"Debug: paged_kv_indices: {paged_kv_indices}")
    print(f"Debug: paged_kv_indptr: {paged_kv_indptr}")
    print(f"Debug: paged_kv_last_page_len: {paged_kv_last_page_len}")

    print(f"Debug: About to call prefill_wrapper.run")
    print(f"Debug: All tensors on same device? query: {query_flashinfer.device}, kv_cache: {paged_kv_cache.device}")
    print(f"Debug: All tensors same dtype? query: {query_flashinfer.dtype}, kv_cache: {paged_kv_cache.dtype}")

    try:
        print(f"Debug: Calling prefill_wrapper.run...")
        attn_output = prefill_wrapper.run(query_flashinfer, paged_kv_cache)
        print(f"Debug: prefill_wrapper.run completed successfully!")
    except Exception as e:
        print(f"Debug: ERROR in prefill_wrapper.run: {e}")
        print(f"Debug: Error type: {type(e)}")
        print(f"Debug: Full traceback:")
        import traceback
        traceback.print_exc()
        raise
    print(f"Debug: Prefill attention completed, output shape {attn_output.shape}")

    # Reshape output back to original format
    print(f"Debug: Reshaping output back to original format")
    # FlashInfer returns [seq_len, num_qo_heads, head_dim], we need [batch, num_heads, seq_len, head_dim]
    attn_output = attn_output.unsqueeze(0)
    print(f"Debug: Final output shape {attn_output.shape}")

    return attn_output, None  # FlashInfer doesn't return attention weights



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def _cascade_decode_attention(
        self,
        query: torch.Tensor,
        page_table: PageTable,
        qo_indptr_arr: list[torch.Tensor],
        paged_kv_indptr_arr: list[torch.Tensor],
        paged_kv_indices_arr: list[torch.Tensor],
        paged_kv_last_page_len_arr: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Cascade decode attention using FlashInfer's MultiLevelCascadeAttentionWrapper.

        Handles beam search decode where sequences share prefixes in a hierarchical structure.

        Note: kv_last_page_len excludes the current query token for the attention kernel,
        but we still store the query's K/V in the page table for the next iteration.

        Args:
            query: Query tensor [batch=1, seq_len, num_qo_heads, head_dim] (RoPE applied)
            page_table: PageTable with KV cache
            qo_indptr_arr: QO index pointers per cascade level
            paged_kv_indptr_arr: KV index pointers per cascade level
            paged_kv_indices_arr: Page indices per cascade level
            paged_kv_last_page_len_arr: Last page lengths per cascade level

        Returns:
            Attention output tensor
        """
        device = query.device
        num_qo_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.head_dim

        # TODO: Store current token's K/V in page table for next iteration

        # Create workspace buffer
        workspace_size = 128 * 1024 * 1024  # 128MB
        workspace_buffer = torch.empty(workspace_size, dtype=torch.uint8, device=device)

        # Initialize cascade wrapper
        cascade_wrapper = flashinfer.cascade.MultiLevelCascadeAttentionWrapper(
            len(qo_indptr_arr),
            workspace_buffer,
            kv_layout="NHD"
        )

        # Plan cascade attention
        cascade_wrapper.plan(
            qo_indptr_arr=qo_indptr_arr,
            paged_kv_indptr_arr=paged_kv_indptr_arr,
            paged_kv_indices_arr=paged_kv_indices_arr,
            paged_kv_last_page_len=paged_kv_last_page_len_arr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_table.page_size,
            causal=True,
            pos_encoding_mode='NONE',
            sm_scale=self.scaling,
            q_data_type=query.dtype
        )

        # Prepare query: [batch, seq_len, num_heads, head_dim] -> [seq_len, num_heads, head_dim]
        query_flashinfer = query

        # Get KV cache for this layer
        paged_kv_cache = page_table.kv_cache_at_layer[self.layer_idx]

        # Run cascade attention
        attn_output = cascade_wrapper.run(query_flashinfer, paged_kv_cache)

        # Add batch dimension back
        attn_output = attn_output.unsqueeze(0)

        return attn_output

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mode: Optional[AttentionMode] = None,
        page_table: Optional[PageTable] = None,
        page_indices: Optional[list] = None,
        last_page_len: Optional[int] = None,
        batch_page_indices: Optional[list] = None,  # List of page_indices for batch decode
        batch_last_page_lens: Optional[list] = None,  # List of last_page_lens for batch decode
        # Cascade decode parameters
        cascade_qo_indptr_arr: Optional[list[torch.Tensor]] = None,
        cascade_kv_indptr_arr: Optional[list[torch.Tensor]] = None,
        cascade_kv_indices_arr: Optional[list[torch.Tensor]] = None,
        cascade_kv_last_page_len_arr: Optional[list[torch.Tensor]] = None,
        cascade_write_page_indices: Optional[list] = None,  # Page IDs to write K/V for each candidate
        cascade_write_positions: Optional[list] = None,  # Positions within page to write K/V
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        key_states = key_states.squeeze(0)    # [1, num_candidates, num_kv_heads, head_dim] -> [num_candidates, num_kv_heads, head_dim]
        query_states = query_states.squeeze(0)  # [1, num_candidates, num_heads, head_dim] -> [num_candidates, num_heads, head_dim]
        value_states = value_states.squeeze(0)  # [1, num_candidates, num_kv_heads, head_dim] -> [num_candidates, num_kv_heads, head_dim]


        # Choose attention implementation based on mode
        if attention_mode == AttentionMode.PREFILL:
            if page_table is None or page_indices is None:
                raise ValueError("PageTable and page_indices must be provided for PREFILL attention mode")       
            flashinfer.rope.apply_rope_pos_ids_inplace(query_states, key_states, position_ids)
            attn_output, attn_weights = flashinfer_prefill_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                page_table,
                page_indices,      # <-- Pass through
                last_page_len,     # <-- Pass through
                dropout=0.0 if not self.training else self.attention_dropout,
                **kwargs,
            )

        elif attention_mode == AttentionMode.DECODE:
            

            # Validate cascade decode parameters
            if page_table is None:
                raise ValueError("PageTable must be provided for DECODE attention mode")
            if cascade_qo_indptr_arr is None or cascade_kv_indptr_arr is None or \
               cascade_kv_indices_arr is None or cascade_kv_last_page_len_arr is None:
                raise ValueError("Cascade parameters must be provided for DECODE attention mode")
            if cascade_write_page_indices is None or cascade_write_positions is None:
                raise ValueError("cascade_write_page_indices and cascade_write_positions must be provided for DECODE attention mode")
            

            # Write current token's K/V to page table for each candidate
            # key_states: [seq_len=num_candidates, num_kv_heads, head_dim]
            # value_states: [ seq_len=num_candidates, num_kv_heads, head_dim]
            num_candidates = key_states.shape[0]
            rope_params = self.config.rope_scaling
            flashinfer.rope.apply_llama31_rope_inplace(
                query_states,
                key_states,
                cascade_qo_indptr_arr[len(cascade_qo_indptr_arr) - 1].to(query_states.device),
                position_ids
            )
            # flashinfer.rope.apply_llama31_rope_pos_ids_inplace(
            #     query_states,
            #     key_states,
            #     position_ids,
            #     rope_scale=rope_params.get("factor", 8.0),
            #     rope_theta=self.config.rope_theta,
            #     low_freq_factor=rope_params.get("low_freq_factor", 1.0),
            #     high_freq_factor=rope_params.get("high_freq_factor", 4.0),
            #     old_context_len=rope_params.get("original_max_position_embeddings", 8192),
            #     interleave=False # Llama uses non-interleaved (rotate_half)
            # )

            for cand_idx in range(num_candidates):
                page_id = cascade_write_page_indices[cand_idx]
                write_pos = cascade_write_positions[cand_idx]

                # Extract K/V for this candidate [num_kv_heads, head_dim]
                key_to_write = key_states[cand_idx:cand_idx+1, :, :]
                value_to_write = value_states[cand_idx:cand_idx+1, :, :]

                # Write to page table at this layer
                page_table.write_block(
                    layer=self.layer_idx,
                    page_idx=page_id,
                    key=key_to_write,
                    value=value_to_write,
                    index=write_pos    
                )

            # Run cascade decode attention
            attn_output = self._cascade_decode_attention(
                query_states,
                page_table,
                cascade_qo_indptr_arr,
                cascade_kv_indptr_arr,
                cascade_kv_indices_arr,
                cascade_kv_last_page_len_arr,
            )

            attn_weights = None

        else:
            raise ValueError(f"Invalid attention mode: {attention_mode}. Must be PREFILL or DECODE")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Extract FlashInfer parameters from kwargs to avoid duplicate arguments
        attention_kwargs = kwargs.copy()
        attention_mode = attention_kwargs.pop('attention_mode', None)
        page_table = attention_kwargs.pop('page_table', None)
        page_indices = attention_kwargs.pop('page_indices', None)
        last_page_len = attention_kwargs.pop('last_page_len', None)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            attention_mode=attention_mode,
            page_table=page_table,
            page_indices=page_indices,
            last_page_len=last_page_len,
            **attention_kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        #position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                #position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]