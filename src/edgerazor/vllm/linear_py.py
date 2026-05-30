"""
EdgeRazor pure-Python linear method for vLLM.

W4A16: dequantize INT4 weights → bf16 matmul.
W4A8:  per-token INT8 quantize activation → dequantize back → bf16 matmul.

No custom CUDA kernels — works on any GPU / CPU.
"""

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from .quant_ops import (
    dequantize_weight,
    quantize_activation_per_block_int8,
    quantize_weight_per_block_int4,
)

logger = init_logger("vllm.edgerazor.py")


class EdgeRazorPyLinearMethod(LinearMethodBase):
    """Pure-Python W4A16 / W4A8 linear method."""

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.er_block_size = quant_config._quant_block_size
        self.ie_block_size = quant_config._scale_block_size
        self.needs_split = quant_config._needs_scale_split
        self.activation_bits = quant_config.activation_bits

    # ── create_weights (shared with Marlin) ──────────────────────

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

    # ── pack (EdgeRazor uint8 format) ────────────────────────────

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_edgerazor_needs_pack", False):
            return

        w = layer.weight.data
        orig_bytes = w.numel() * w.element_size()

        qweight, qweight_scale = quantize_weight_per_block_int4(
            w,
            er_block_size=self.er_block_size,
            ie_block_size=self.ie_block_size,
        )

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

        logger.info(
            "[EdgeRazor PY] W%dA%d, packed %s → %s / %s  "
            "(%.1f%% of bf16, %.1f bits/el, ER=%d→IE=%d)",
            self.quant_config.weight_bits,
            self.activation_bits,
            list(w.shape),
            list(qweight.shape),
            list(qweight_scale.shape),
            ratio,
            ratio * 16 / 100,
            self.er_block_size,
            self.ie_block_size,
        )

    # ── forward ──────────────────────────────────────────────────

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_deq = dequantize_weight(
            layer.qweight,
            layer.qweight_scale,
            block_size=self.ie_block_size,
            out_dtype=x.dtype,
        )

        if self.activation_bits == 8:
            # A8: per-block INT8 quantize activation
            x_int, _x_scale = quantize_activation_per_block_int8(x)
            x = x_int.to(x.dtype)

        return torch.nn.functional.linear(x, w_deq, bias)
