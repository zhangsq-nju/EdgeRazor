"""
Quantized Multi-Head Attention Module.

For transformers multi-head attention, there is no need to replace attn block. The reason is that
attn blocks like Qwen3Attention use nn.Linear layers for QKV and output projections, which can be
directly replaced by QLinear layers.

Otherwise, for pytorch multi-head attention, we need to replace the attention block with
a custom implementation that supports quantization.

Besides, current llm models usually use transformers customized Attention blocks which are
equipped with nn.Linear layers for projections, so no need to replace those Attention blocks either.

If weight, activation, and KV cache quantization are needed, we can apply quantized these blocks
with quantized nn.Linear layers as well.
"""
# ruff: noqa N812

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.activation import (
    _arg_requires_grad,
    _check_arg_device,
    _is_make_fx_tracing,
)

from ..util.quant_config import QuantConfig


class QMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        # Standard nn.MultiheadAttention parameters
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
        # Additional QAT hyperparameters
        quant_config: QuantConfig | None = None,   # Quantization configuration
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
            kdim, vdim, batch_first, device=device, dtype=dtype
        )
        
        if quant_config is None:
            raise ValueError("quant_config must be provided for QMultiheadAttention.")

        # Small value to prevent division by zero
        self.epsilon = quant_config.function.epsilon
        # Whether the weights are already quantized
        self.is_w_quantized = quant_config.function.is_w_quantized
        
        # Track quantization state for different weight groups
        self.is_qkv_quantized = quant_config.function.is_w_quantized
        self.is_in_proj_quantized = quant_config.function.is_w_quantized
        self.is_out_proj_quantized = quant_config.function.is_w_quantized

        # Quantization configuration
        ## Weight
        self.w_quant_function = quant_config.function.weight_function
        self.w_scale_factor = quant_config.function.w_scale_factor
        self.w_block_size = quant_config.function.w_block_size
        self.w_mixed_precision_prop = quant_config.function.w_mixed_precision_prop
        self.w_kwargs = {'epsilon': self.epsilon}
        if self.w_scale_factor > 0:
            self.w_kwargs['w_scale_factor'] = self.w_scale_factor
        if self.w_block_size > 0:
            self.w_kwargs['block_size'] = self.w_block_size
        if self.w_mixed_precision_prop > 0:
            self.w_kwargs['mixed_precision_prop'] = self.w_mixed_precision_prop
        
        ## Activation
        self.a_quant_function = quant_config.function.activation_function
        self.a_block_size = quant_config.function.a_block_size
        self.a_mixed_precision_prop = quant_config.function.a_mixed_precision_prop
        self.a_kwargs = {'epsilon': self.epsilon}
        if self.a_block_size > 0:
            self.a_kwargs['block_size'] = self.a_block_size
        if self.a_mixed_precision_prop > 0:
            self.a_kwargs['mixed_precision_prop'] = self.a_mixed_precision_prop

    def _qkv_proj_weight_quant(self, replace_self: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize QKV projection weights (when _qkv_same_embed_dim is False)"""
        # Quantize Q, K, V projection weights separately
        q_w = self.q_proj_weight.data.clone()
        k_w = self.k_proj_weight.data.clone()
        v_w = self.v_proj_weight.data.clone()
        
        # Apply quantization function
        q_proj_weight_quant = self.w_quant_function(w=q_w, **self.w_kwargs)
        k_proj_weight_quant = self.w_quant_function(w=k_w, **self.w_kwargs)
        v_proj_weight_quant = self.w_quant_function(w=v_w, **self.w_kwargs)

        if replace_self:
            if not self.is_qkv_quantized:
                self.q_proj_weight.data = q_proj_weight_quant.clone()
                self.k_proj_weight.data = k_proj_weight_quant.clone()
                self.v_proj_weight.data = v_proj_weight_quant.clone()
                self.is_qkv_quantized = True
            else:
                raise RuntimeError("Weights (qkv_proj) are already quantized. Cannot replace self again.")
        
        return q_proj_weight_quant, k_proj_weight_quant, v_proj_weight_quant
    
    def _in_proj_weight_quant(self, replace_self: bool = False) -> Tensor:
        """Quantize in_proj weight (when _qkv_same_embed_dim is True)"""
        W = self.in_proj_weight.data.clone()
        
        # Apply quantization function
        w_quant = self.w_quant_function(w=W, **self.w_kwargs)
        
        if replace_self:
            if not self.is_in_proj_quantized:
                self.in_proj_weight.data = w_quant.clone()
                self.is_in_proj_quantized = True
            else:
                raise RuntimeError("Weights (in_proj) are already quantized. Cannot replace self again.")
        
        return w_quant
    
    def _out_proj_weight_quant(self, replace_self: bool = False) -> Tensor:
        """Quantize out_proj weight"""
        W = self.out_proj.weight.data.clone()
        
        # Apply quantization function
        w_quant = self.w_quant_function(w=W, **self.w_kwargs)
        
        if replace_self:
            if not self.is_out_proj_quantized:
                self.out_proj.weight.data = w_quant.clone()
                self.is_out_proj_quantized = True
            else:
                raise RuntimeError("Weights (out_proj) are already quantized. Cannot replace self again.")
        
        return w_quant

    def _activation_quant(self, x: Tensor) -> Tensor:
        """Quantize activation"""
        x_quant = self.a_quant_function(x=x, **self.a_kwargs)
        return x_quant
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        
        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        
        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = (
                "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
            )
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(
                _arg_requires_grad(x) for x in tensor_args
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # Step 1: Quantize activations if configured
        if self.a_quant_function is not None:
            query_quant = self._activation_quant(query)
            key_quant = self._activation_quant(key)
            value_quant = self._activation_quant(value)
            
            # Straight-Through Estimator
            query_quant = query + (query_quant - query).detach()
            key_quant = key + (key_quant - key).detach()
            value_quant = value + (value_quant - value).detach()
        else:
            query_quant = query
            key_quant = key
            value_quant = value
        
        # Step 2: Handle weight quantization based on projection type
        if not self._qkv_same_embed_dim:
            # Separate Q, K, V projections
            if self.training:
                # Straight-Through Estimator for training
                q_proj_weight_quant, k_proj_weight_quant, v_proj_weight_quant = self._qkv_proj_weight_quant()
                out_proj_weight_quant = self._out_proj_weight_quant()
                
                q_proj_weight_quant = self.q_proj_weight + (q_proj_weight_quant - self.q_proj_weight).detach()
                k_proj_weight_quant = self.k_proj_weight + (k_proj_weight_quant - self.k_proj_weight).detach()
                v_proj_weight_quant = self.v_proj_weight + (v_proj_weight_quant - self.v_proj_weight).detach()
                out_proj_weight_quant = self.out_proj.weight + (out_proj_weight_quant - self.out_proj.weight).detach()
            else:  # eval, inference_mode
                if self.is_qkv_quantized and self.is_out_proj_quantized:
                    q_proj_weight_quant = self.q_proj_weight
                    k_proj_weight_quant = self.k_proj_weight
                    v_proj_weight_quant = self.v_proj_weight
                    out_proj_weight_quant = self.out_proj.weight
                else:
                    q_proj_weight_quant, k_proj_weight_quant, v_proj_weight_quant = self._qkv_proj_weight_quant()
                    out_proj_weight_quant = self._out_proj_weight_quant()
            
            # Call attention forward with quantized weights
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query_quant,
                key_quant,
                value_quant,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                out_proj_weight_quant,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=q_proj_weight_quant,
                k_proj_weight=k_proj_weight_quant,
                v_proj_weight=v_proj_weight_quant,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            # Combined in_proj weight
            if self.training:
                in_proj_weight_quant = self._in_proj_weight_quant()
                out_proj_weight_quant = self._out_proj_weight_quant()
                
                in_proj_weight_quant = self.in_proj_weight + (in_proj_weight_quant - self.in_proj_weight).detach()
                out_proj_weight_quant = self.out_proj.weight + (out_proj_weight_quant - self.out_proj.weight).detach()
            else:  # eval, inference_mode
                if self.is_in_proj_quantized and self.is_out_proj_quantized:
                    in_proj_weight_quant = self.in_proj_weight
                    out_proj_weight_quant = self.out_proj.weight
                else:
                    in_proj_weight_quant = self._in_proj_weight_quant()
                    out_proj_weight_quant = self._out_proj_weight_quant()
            
            # Call attention forward with quantized weights
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query_quant,
                key_quant,
                value_quant,
                self.embed_dim,
                self.num_heads,
                in_proj_weight_quant,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                out_proj_weight_quant,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def copy_multiheadattention_to_qmultiheadattention(
    mha: nn.MultiheadAttention,
    qmha_cls: type[nn.Module] = QMultiheadAttention,
    quant_config: QuantConfig | None = None
):
    """Copy MultiheadAttention to quantized MultiheadAttention"""
    if quant_config is None:
        raise ValueError("quant_config must be provided for QMultiheadAttention.")
    
    qmha = qmha_cls(
        embed_dim=mha.embed_dim,
        num_heads=mha.num_heads,
        dropout=mha.dropout,
        bias=mha.in_proj_bias is not None,
        add_bias_kv=mha.bias_k is not None and mha.bias_v is not None,
        add_zero_attn=mha.add_zero_attn,
        kdim=mha.kdim,
        vdim=mha.vdim,
        batch_first=mha.batch_first,
        device=mha.out_proj.weight.device,
        dtype=mha.out_proj.weight.dtype,
        quant_config=quant_config
    )
    
    # Copy weights
    if mha._qkv_same_embed_dim:
        # q_proj_weight, k_proj_weight, v_proj_weight ❌
        # in_proj_weight ✅
        qmha.in_proj_weight.data = mha.in_proj_weight.data.clone()
    else:
        # q_proj_weight, k_proj_weight, v_proj_weight ✅
        # in_proj_weight ❌
        qmha.q_proj_weight.data = mha.q_proj_weight.data.clone()
        qmha.k_proj_weight.data = mha.k_proj_weight.data.clone()
        qmha.v_proj_weight.data = mha.v_proj_weight.data.clone()
    
    qmha.out_proj.weight.data = mha.out_proj.weight.data.clone()

    # Copy biases
    if mha.in_proj_bias is not None:
        qmha.in_proj_bias.data = mha.in_proj_bias.data.clone()
    
    if mha.out_proj.bias is not None:
        qmha.out_proj.bias.data = mha.out_proj.bias.data.clone()
    
    if mha.bias_k is not None:
        qmha.bias_k.data = mha.bias_k.data.clone()
    if mha.bias_v is not None:
        qmha.bias_v.data = mha.bias_v.data.clone()
    
    # Copy state
    qmha.training = mha.training
    
    return qmha
