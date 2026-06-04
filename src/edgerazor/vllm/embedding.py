"""Quantized embedding method for EdgeRazor — W4 / W1.58.

Per-row dequant at lookup time.  Only the requested rows are unpacked
and dequantized; the packed weight table stays on GPU at all times.

Works with both the Marlin and pure-Python backends (embedding is
independent of the GEMM kernel choice).
"""

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs

from .quant_ops import (
    dequantize_weight,
    quantize_activation_per_block_int8,
    quantize_weight_per_block_int4,
    quantize_weight_per_block_int2,
)

logger = init_logger("vllm.edgerazor.embed")


class EdgeRazorEmbeddingMethod(QuantizeMethodBase):
    """Quantized W4 / W1.58 embedding via sparse per-row dequant."""

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.weight_bits = quant_config.weight_bits
        self.activation_bits = quant_config.activation_bits
        self.ie_block_size = quant_config._scale_block_size

    # ── create_weights ───────────────────────────────────────────

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.pop("weight_loader")

        from vllm.model_executor.parameter import ModelWeightParameter

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        layer._edgerazor_needs_pack = True

    # ── pack ──────────────────────────────────────────────────────

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_edgerazor_needs_pack", False):
            return

        w = layer.weight.data
        orig_bytes = w.numel() * w.element_size()

        if self.weight_bits == 4:
            qweight, qweight_scale = quantize_weight_per_block_int4(
                w, er_block_size=256, ie_block_size=self.ie_block_size,
            )
        elif self.weight_bits == 1.58:
            qweight, qweight_scale = quantize_weight_per_block_int2(
                w, er_block_size=256, ie_block_size=self.ie_block_size,
            )
        else:
            raise ValueError(f"Unsupported embedding weight_bits={self.weight_bits}")

        del layer.weight
        layer.register_parameter(
            "qweight",
            Parameter(qweight.contiguous(), requires_grad=False),
        )
        layer.register_parameter(
            "qweight_scale",
            Parameter(qweight_scale.contiguous(), requires_grad=False),
        )
        layer._edgerazor_needs_pack = False

        packed_bytes = qweight.numel() * 1 + qweight_scale.numel() * 2
        ratio = packed_bytes / orig_bytes * 100
        wbits_label = "1.58" if self.weight_bits == 1.58 else str(self.weight_bits)
        layer_name = getattr(layer, "_edgerazor_layer_name", "?")
        logger.info(
            "[EdgeRazor EMB] W%sA%d, packed %s %s → %s / %s  "
            "(%.1f%% of bf16, %.1f bits/el, IE=%d)",
            wbits_label,
            self.quant_config.activation_bits,
            layer_name,
            list(w.shape),
            list(qweight.shape),
            list(qweight_scale.shape),
            ratio,
            ratio * 16 / 100,
            self.ie_block_size,
        )

    # ── forward ──────────────────────────────────────────────────

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Linear projection (lm_head when tie_word_embeddings=True).

        vLLM sets ``self.lm_head = self.model.embed_tokens`` when weights
        are tied, so the same ``EdgeRazorEmbeddingMethod`` must handle
        both ``embedding()`` (lookup) and ``apply()`` (matmul).
        """
        w_deq = dequantize_weight(
            layer.qweight,
            layer.qweight_scale,
            block_size=self.ie_block_size,
            out_dtype=x.dtype,
            weight_bits=self.weight_bits,
        )

        if self.activation_bits == 8:
            x_int, x_scale = quantize_activation_per_block_int8(x)
            x = (x_int.float() * x_scale.repeat_interleave(
                x.shape[-1] // x_scale.shape[-1], dim=-1,
            ).float()).to(x.dtype)

        return torch.nn.functional.linear(x, w_deq, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return dequantize_weight(
            layer.qweight[input_],
            layer.qweight_scale[input_],
            block_size=self.ie_block_size,
            out_dtype=layer.qweight_scale.dtype,
            weight_bits=self.weight_bits,
        )
