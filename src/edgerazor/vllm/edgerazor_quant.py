"""
EdgeRazor quantization plugin for vLLM.

Supports W4-A8-KV8 and W1.58-A8-KV8 (1.58-bit weights degraded to 4-bit packing).

Weight format (per-block INT4, symmetric absmax):
  - qweight:  packed uint8 tensor (shape: out × in//2), 2 INT4 values per byte
  - qweight_scale:  per-block bf16 scale (shape: out × nblocks, block_size=32)

Activation quant (per-token INT8, optional):
  - Runtime dynamic quantization along last dim before matmul.

KV-cache quant (per-token INT8, optional):
  - Delegated to vLLM's native ``kv_cache_dtype`` mechanism when
    ``--kv-cache-dtype fp8`` is passed alongside ``--quantization edgerazor``.

Reference: llama.cpp Q4_0 (weight) + Q8_0 (KV cache)
"""

from typing import Any

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.utils import set_weight_attrs

from .quant_ops import (
    ER_W4A8_BLOCK_SIZE,
    IE_W4A8_BLOCK_SIZE,
    dequantize_weight,
    quantize_weight_per_block_int4,
    resolve_quant_block,
)

logger = init_logger("vllm.edgerazor.quant")


# ──────────────────────────────────────────────
# EdgeRazor Linear Method
# ──────────────────────────────────────────────

class EdgeRazorLinearMethod(LinearMethodBase):
    """Linear method for EdgeRazor per-block INT4 weights.

    Weights are loaded as bf16 from the HF checkpoint, then packed to INT4
    in ``process_weights_after_loading()``. Forward pass dequantizes and
    applies standard matmul.
    """

    def __init__(self, quant_config: "EdgeRazorConfig"):
        self.quant_config = quant_config
        self.er_block_size = quant_config._quant_block_size   # quantization block
        self.ie_block_size = quant_config._scale_block_size   # scale / dequant block
        self.needs_split = quant_config._needs_scale_split
        self.use_activation_quant = quant_config.activation_bits > 0

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_edgerazor_needs_pack", False):
            return

        w = layer.weight.data
        out_dim, in_dim = w.shape

        # Quantize with ER block_size, pack with IE block_size
        qweight, qweight_scale = quantize_weight_per_block_int4(
            w,
            er_block_size=self.er_block_size,
            ie_block_size=self.ie_block_size,
        )

        # Replace temporary weight with packed params
        orig_bytes = w.numel() * w.element_size()
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
            "[EdgeRazor W4] Packed weight %s → qweight+scale: "
            "%s → %s  (%.1f%% of bf16, %.1f bits/el, ER=%d IE=%d split=%s)",
            list(w.shape), list(qweight.shape), list(qweight_scale.shape),
            ratio, ratio * 16 / 100,
            self.er_block_size, self.ie_block_size, self.needs_split,
        )

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
        return torch.nn.functional.linear(x, w_deq, bias)


# ──────────────────────────────────────────────
# EdgeRazor Quantization Config
# ──────────────────────────────────────────────

@register_quantization_config("edgerazor")
class EdgeRazorConfig(QuantizationConfig):
    """Quantization config for EdgeRazor W4-A8-KV8 / W1.58-A8-KV8 models.

    Auto-detects EdgeRazor models via ``quantization_config.quant_method``
    in the HF config.json, or by explicit ``--quantization edgerazor``.
    """

    def __init__(
        self,
        weight_bits: int = 4,
        weight_block_size: int | list[int] | None = None,
        er_block_size: int = ER_W4A8_BLOCK_SIZE,
        ie_block_size: int = IE_W4A8_BLOCK_SIZE,
        activation_bits: int = 8,
        kv_cache_bits: int = 8,
        quant_mode: str = "",
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.kv_cache_bits = kv_cache_bits
        self.quant_mode = quant_mode
        self.modules_to_not_convert = modules_to_not_convert or []

        # Resolve block sizes: weight_block_size (legacy) overrides IE
        if weight_block_size is None:
            weight_block_size = ie_block_size
        if isinstance(weight_block_size, list):
            weight_block_size = weight_block_size[0]
        if isinstance(weight_block_size, int):
            if weight_block_size != ie_block_size:
                ie_block_size = weight_block_size
        # vLLM's LinearBase calls len(quant_config.weight_block_size),
        # so expose ie as a list for compat.
        self.weight_block_size = [ie_block_size]
        self.er_block_size = er_block_size
        self.ie_block_size = ie_block_size

        # Validate ER / IE split
        resolved = resolve_quant_block(self.er_block_size, self.ie_block_size)
        self._quant_block_size = resolved[0]
        self._scale_block_size = resolved[1]
        self._needs_scale_split = resolved[2]

        logger.info(
            "[EdgeRazor] Quantization config created: W%dA%dKV%d, "
            "ER_block=%d IE_block=%d split=%s, quant_mode=%s",
            self.weight_bits,
            self.activation_bits,
            self.kv_cache_bits,
            self.er_block_size,
            self.ie_block_size,
            self._needs_scale_split,
            self.quant_mode or "default",
        )

    def __repr__(self) -> str:
        return (
            f"EdgeRazorConfig(weight_bits={self.weight_bits}, "
            f"weight_block_size={self.weight_block_size}, "
            f"activation_bits={self.activation_bits}, "
            f"kv_cache_bits={self.kv_cache_bits}, "
            f"quant_mode={self.quant_mode!r})"
        )

    def get_name(self) -> str:
        return "edgerazor"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # No custom CUDA kernels — works on any GPU (and CPU via vLLM CPU backend)
        return 50

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EdgeRazorConfig":
        """Create config from a model's ``quantization_config`` dict."""
        quant_mode = config.get("quant_mode", "")
        weight_bits = config.get("weight_bits", 4)
        er_block_size = config.get("er_block_size", ER_W4A8_BLOCK_SIZE)
        ie_block_size = config.get("ie_block_size", IE_W4A8_BLOCK_SIZE)
        activation_bits = config.get("activation_bits", 8)
        kv_cache_bits = config.get("kv_cache_bits", 8)
        modules_to_not_convert = config.get("modules_to_not_convert", [])
        return cls(
            weight_bits=weight_bits,
            er_block_size=er_block_size,
            ie_block_size=ie_block_size,
            activation_bits=activation_bits,
            kv_cache_bits=kv_cache_bits,
            quant_mode=quant_mode,
            modules_to_not_convert=modules_to_not_convert,
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        """Auto-detect EdgeRazor models from HF config."""
        if user_quant == "edgerazor":
            return "edgerazor"

        # Primary path: quantization_config.quant_method == "edgerazor"
        if hf_quant_cfg is not None and hf_quant_cfg.get("quant_method") == "edgerazor":
            logger.info(
                "Auto-detected EdgeRazor model (quantization_config.quant_method=edgerazor, "
                "quant_mode=%s)", hf_quant_cfg.get("quant_mode", "unknown"),
            )
            return "edgerazor"

        if hf_config is not None:
            # Secondary: hf_config.quantization_config.quant_method
            qc = getattr(hf_config, "quantization_config", None)
            if qc is not None and qc.get("quant_method") == "edgerazor":
                logger.info(
                    "Auto-detected EdgeRazor model (quantization_config.quant_method=edgerazor, "
                    "quant_mode=%s)", qc.get("quant_mode", "unknown"),
                )
                return "edgerazor"

            # Backward compat: edgerazor_qconfig top-level key
            if hasattr(hf_config, "edgerazor_qconfig"):
                logger.info(
                    "Auto-detected EdgeRazor model (edgerazor_qconfig=%s)",
                    hf_config.edgerazor_qconfig,
                )
                return "edgerazor"

            # Backward compat: quant_mode top-level key
            if hasattr(hf_config, "quant_mode") and hf_config.quant_mode:
                logger.info(
                    "Auto-detected EdgeRazor model (quant_mode=%s)",
                    hf_config.quant_mode,
                )
                return "edgerazor"

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if self._is_layer_skipped(prefix):
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod
                return UnquantizedLinearMethod()
            return EdgeRazorLinearMethod(self)
        if isinstance(layer, VocabParallelEmbedding):
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                UnquantizedEmbeddingMethod,
            )
            return UnquantizedEmbeddingMethod()
        return None

    def _is_layer_skipped(self, prefix: str) -> bool:
        return any(m in prefix for m in self.modules_to_not_convert)
