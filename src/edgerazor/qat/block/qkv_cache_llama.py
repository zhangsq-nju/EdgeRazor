from collections.abc import Callable

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

from ..util.quant_config import QuantConfig


class QKVCacheLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper with KV Cache quantization"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        quant_config: QuantConfig | None = None,
    ):
        super().__init__(config, layer_idx)
        if quant_config is None:
            raise ValueError("quant_config must be provided for QKVCacheLlamaAttention")

        # Small value to prevent division by zero
        self.epsilon = quant_config.function.epsilon

        # Quantization configuration
        ## KV Cache (State)
        self.kv_cache_quant_function = quant_config.function.kv_cache_function
        self.kv_block_size = quant_config.function.kv_block_size
        self.kv_mixed_precision_prop = quant_config.function.kv_mixed_precision_prop
        self.kv_kwargs = {'epsilon': self.epsilon}
        if self.kv_block_size > 0:
            self.kv_kwargs['block_size'] = self.kv_block_size
        if self.kv_mixed_precision_prop > 0:
            self.kv_kwargs['mixed_precision_prop'] = self.kv_mixed_precision_prop

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --------------------------------------------------------------------------
        # After RoPE | Before KV Cache Storing: Apply KV Cache Quantization
        key_quant = self.kv_cache_quant_function(x=key_states, **self.kv_kwargs)
        value_quant = self.kv_cache_quant_function(x=value_states, **self.kv_kwargs)
        key_states = key_states + (key_quant - key_states).detach()
        value_states = value_states + (value_quant - value_states).detach()
        # --------------------------------------------------------------------------

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def copy_llamaattention_to_qkvcache_llamaattention(
    llama_attn: LlamaAttention,
    qkvcache_llamaattention_cls: QKVCacheLlamaAttention,
    quant_config: QuantConfig,
) -> QKVCacheLlamaAttention:
    """Copy LlamaAttention to QKVCacheLlamaAttention with quantization config"""
    if quant_config is None:
        raise ValueError("quant_config must be provided for QKVCacheLlamaAttention")

    qkvcache_llama_attn = qkvcache_llamaattention_cls(
        config=llama_attn.config,
        layer_idx=llama_attn.layer_idx,
        quant_config=quant_config,
    )

    # Copy all projection weights
    qkvcache_llama_attn.q_proj.weight.data = llama_attn.q_proj.weight.data.clone()
    qkvcache_llama_attn.k_proj.weight.data = llama_attn.k_proj.weight.data.clone()
    qkvcache_llama_attn.v_proj.weight.data = llama_attn.v_proj.weight.data.clone()
    qkvcache_llama_attn.o_proj.weight.data = llama_attn.o_proj.weight.data.clone()

    # Copy biases if they exist
    if llama_attn.q_proj.bias is not None:
        qkvcache_llama_attn.q_proj.bias.data = llama_attn.q_proj.bias.data.clone()
    if llama_attn.k_proj.bias is not None:
        qkvcache_llama_attn.k_proj.bias.data = llama_attn.k_proj.bias.data.clone()
    if llama_attn.v_proj.bias is not None:
        qkvcache_llama_attn.v_proj.bias.data = llama_attn.v_proj.bias.data.clone()
    if llama_attn.o_proj.bias is not None:
        qkvcache_llama_attn.o_proj.bias.data = llama_attn.o_proj.bias.data.clone()
    
    # Copy state
    qkvcache_llama_attn.training = llama_attn.training
    
    return qkvcache_llama_attn
