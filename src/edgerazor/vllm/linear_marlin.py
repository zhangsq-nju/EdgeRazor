"""
EdgeRazor Marlin-kernel linear method for vLLM.

Requires CUDA GPU with compute capability >= 7.5 (Turing).

W4A16:   fused INT4 dequant + bf16 matmul via Marlin GEMM.
W4A8:    fused INT4 dequant + INT8 matmul via Marlin GEMM.
W1.58A16: ternary {-1,0,1} → uint4b8 → Marlin W4 pipeline (upcast).
W1.58A8:  ternary → uint4b8 → Marlin W4A8 pipeline (upcast).

Weight packing (shared across all bit-widths):
  bf16 (N,K)
    → per-block quantize (ER block → split to IE=128)
    → GPTQ row-packed int32 (K/8, N)
    → gptq_marlin_repack → Marlin tile-interleaved
    → marlin_permute_scales
"""

import torch
from torch.nn.parameter import Parameter
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .quant_ops import INT1_58_MAX, INT4_MAX

logger = init_logger("vllm.edgerazor.marlin")

# Marlin requires these specific group sizes; we use 128 (max supported
# that evenly divides EdgeRazor's ER block of 256).
MARLIN_GROUP_SIZE = 128

# W1.58 ternary clip scale factor (mean-abs multiplier from EdgeRazor training).
W1_58_SCALE_FACTOR = 2.0


class EdgeRazorMarlinLinearMethod(LinearMethodBase):
    """Marlin-kernel W4A16 / W4A8 / W1.58A16 / W1.58A8 linear method."""

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.er_block_size = quant_config._quant_block_size
        self.activation_bits = quant_config.activation_bits
        self.weight_bits = quant_config.weight_bits
        self._forward_fn = None  # cached from first apply

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
        # Save dims for apply
        self.output_size_per_partition = sum(output_partition_sizes)
        self.input_size_per_partition = input_size_per_partition

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

    # ── pack (Marlin format) ─────────────────────────────────────

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_edgerazor_needs_pack", False):
            return

        w = layer.weight.data  # (N, K)  bf16
        N, K = w.shape
        orig_bytes = w.numel() * w.element_size()

        if K % MARLIN_GROUP_SIZE != 0:
            raise ValueError(
                f"Marlin requires in_features ({K}) divisible by "
                f"group_size ({MARLIN_GROUP_SIZE})"
            )

        # 1. Per-block quantize (ER block)
        er = self.er_block_size
        w_blocks = w.view(N, -1, er)

        if self.weight_bits == 4:
            w_scale_er = (
                w_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / INT4_MAX
            )
            int_max = INT4_MAX
        elif self.weight_bits == 1.58:
            # Ternary: clip-method scale = mean(|w|) * w_scale_factor
            w_scale_er = (
                w_blocks.abs().mean(dim=-1, keepdim=True)
                .mul_(W1_58_SCALE_FACTOR).clamp(min=1e-5)
            )
            int_max = INT1_58_MAX  # 1 → clamp to {-1, 0, 1}
        else:
            raise ValueError(f"Unsupported weight_bits={self.weight_bits}")

        w_int = (
            (w_blocks / w_scale_er).round().clamp(-int_max, int_max).to(torch.int8)
        )
        w_int_flat = w_int.view(N, K)  # (N, K)  int8

        # 2. Split scales: ER → IE (Marlin group_size)
        assert er % MARLIN_GROUP_SIZE == 0
        n_split = er // MARLIN_GROUP_SIZE
        w_scale = w_scale_er.squeeze(-1).repeat_interleave(n_split, dim=1)
        # w_scale: (N, K/128)  bf16

        # 3. Pack to GPTQ row-packed int32: (K/8, N)
        #    Both W4 [-7,7] and W1.58 {-1,0,1} → uint4b8 [1,15] via bias +8.
        #    Ternary values {7,8,9} are a proper subset of the valid uint4b8 range.
        w_uint = (w_int_flat + 8).to(torch.int32).view(N, -1, 8)  # (N, K/8, 8)
        gptq_qweight = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
        for i in range(8):
            gptq_qweight = gptq_qweight | (w_uint[:, :, i] << (4 * i))
        gptq_qweight = gptq_qweight.T.contiguous()  # (K/8, N)

        # 4. Repack to Marlin tile-interleaved format.
        is_a8 = self.activation_bits == 8
        perm = torch.empty(0, dtype=torch.int32, device=w.device)
        qweight_marlin = ops.gptq_marlin_repack(
            gptq_qweight,
            perm=perm,
            size_k=K,
            size_n=N,
            num_bits=4,
            is_a_8bit=is_a8,
        )

        # 5. Permute scales for Marlin: (N, K/128) → (K/128, N)
        scales_for_marlin = w_scale.T.contiguous()  # (K/128, N)
        scales_permuted = marlin_permute_scales(
            scales_for_marlin,
            size_k=K,
            size_n=N,
            group_size=MARLIN_GROUP_SIZE,
            is_a_8bit=is_a8,
        )

        # 6. Replace params
        del layer.weight
        layer.register_parameter(
            "qweight",
            Parameter(qweight_marlin.contiguous(), requires_grad=False),
        )
        layer.register_parameter(
            "scales",
            Parameter(scales_permuted.contiguous(), requires_grad=False),
        )
        layer.register_parameter(
            "weight_zp",
            Parameter(
                torch.empty(0, dtype=torch.int32, device=w.device),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "g_idx",
            Parameter(
                marlin_make_empty_g_idx(w.device),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "g_idx_sort_indices",
            Parameter(
                marlin_make_empty_g_idx(w.device),
                requires_grad=False,
            ),
        )
        layer.workspace = marlin_make_workspace_new(w.device)
        layer._edgerazor_needs_pack = False

        # W4A8: Marlin needs an input_global_scale. EdgeRazor uses dynamic
        # per-token quant without a pre-computed global scale → identity.
        layer.input_global_scale = torch.tensor(1.0, dtype=torch.float32, device=w.device)

        packed_bytes = (
            qweight_marlin.numel() * 4
            + scales_permuted.numel() * scales_permuted.element_size()
        )
        ratio = packed_bytes / orig_bytes * 100

        wbits_label = "1.58" if self.weight_bits == 1.58 else str(self.weight_bits)
        logger.info(
            "[EdgeRazor MARLIN] W%sA%d, packed %s → Marlin %s / %s  "
            "(%.1f%% of bf16, %.1f bits/el, ER=%d→IE=%d)",
            wbits_label,
            self.activation_bits,
            [N, K],
            list(qweight_marlin.shape),
            list(scales_permuted.shape),
            ratio,
            ratio * 16 / 100,
            self.er_block_size,
            MARLIN_GROUP_SIZE,
        )

    # ── forward ──────────────────────────────────────────────────

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_a8 = self.activation_bits == 8
        input_dtype = torch.int8 if use_a8 else None
        input_global_scale = getattr(layer, "input_global_scale", None) if use_a8 else None

        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.weight_zp,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            wtype=scalar_types.uint4b8,
            output_size_per_partition=self.output_size_per_partition,
            input_size_per_partition=self.input_size_per_partition,
            is_k_full=True,
            bias=bias,
            input_global_scale=input_global_scale,
            input_dtype=input_dtype,
        )


# ──────────────────────────────────────────────────────────
# Backend selection
# ──────────────────────────────────────────────────────────


def can_use_marlin(quant_config) -> bool:
    """Check whether the Marlin kernel can be used for this config."""
    if not current_platform.is_cuda():
        return False
    if not current_platform.has_device_capability(75):
        return False
    # ER block must be evenly splittable into Marlin group_size=128
    if quant_config._quant_block_size % MARLIN_GROUP_SIZE != 0:
        return False
    return True
