"""
EdgeRazor quantization plugin for vLLM.

Supports W4-A8 (both A16 and A8 activation quantization).

Uses two backends, auto-selected by GPU capability:
  - ``linear_marlin.EdgeRazorMarlinLinearMethod``  —  CUDA sm>=75
  - ``linear_py.EdgeRazorPyLinearMethod``           —  fallback (CPU / old GPU)

All layers go through one ``--quantization edgerazor`` entry point.
"""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from .quant_ops import (
    ER_W1_58A8_BLOCK_SIZE,
    ER_W4A8_BLOCK_SIZE,
    IE_W1_58A8_BLOCK_SIZE,
    IE_W4A8_BLOCK_SIZE,
    resolve_quant_block,
)

logger = init_logger("vllm.edgerazor.quant")

_BACKEND = None  # cached backend name for logging

SUPPRTED_W_BITS = (1.58, 4)
SUPPRTED_A_BITS = (8, 16)


@register_quantization_config("edgerazor")
class EdgeRazorConfig(QuantizationConfig):
    """Quantization config for EdgeRazor quantized models.

    Supports W1.58 (ternary) and W4 (INT4) weight quantization.

    Auto-detected via ``edgerazor_qconfig`` in config.json or explicit
    ``--quantization edgerazor``.
    """

    def __init__(
        self,
        weight_bits: float = 4,
        weight_block_size: int | list[int] | None = None,
        er_block_size: int | None = None,
        ie_block_size: int | None = None,
        activation_bits: int = 16,
        kv_cache_bits: int = 16,
        quant_mode: str = "",
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        if weight_bits not in SUPPRTED_W_BITS:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}. "
                f"EdgeRazor supports: {SUPPRTED_W_BITS}"
            )
        if activation_bits not in SUPPRTED_A_BITS:
            raise ValueError(
                f"Unsupported activation_bits={activation_bits}. "
                f"EdgeRazor supports: {SUPPRTED_A_BITS}"
            )
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.kv_cache_bits = kv_cache_bits
        self.quant_mode = quant_mode
        self.modules_to_not_convert = modules_to_not_convert or []

        # Default ER / IE block sizes per weight bit-width
        if er_block_size is None:
            er_block_size = (
                ER_W1_58A8_BLOCK_SIZE if weight_bits == 1.58 else ER_W4A8_BLOCK_SIZE
            )
        if ie_block_size is None:
            ie_block_size = (
                IE_W1_58A8_BLOCK_SIZE if weight_bits == 1.58 else IE_W4A8_BLOCK_SIZE
            )

        # Resolve block sizes
        if weight_block_size is None:
            weight_block_size = ie_block_size
        if isinstance(weight_block_size, list):
            weight_block_size = weight_block_size[0]
        if isinstance(weight_block_size, int):
            if weight_block_size != ie_block_size:
                ie_block_size = weight_block_size

        # vLLM compat: LinearBase expects weight_block_size to be a list
        self.weight_block_size = [ie_block_size]
        self.er_block_size = er_block_size
        self.ie_block_size = ie_block_size

        # Validate ER / IE split
        resolved = resolve_quant_block(self.er_block_size, self.ie_block_size)
        self._quant_block_size = resolved[0]
        self._scale_block_size = resolved[1]
        self._needs_scale_split = resolved[2]

        logger.info(
            "[EdgeRazor] Quantization config created: "
            "weight_bits=%d, activation_bits=%d, kv_cache_bits=%d, "
            "weight_block_size=%s, quant_mode=%s",
            self.weight_bits,
            self.activation_bits,
            self.kv_cache_bits,
            self.weight_block_size,
            self.quant_mode or "default",
        )

    def __repr__(self) -> str:
        return (
            f"EdgeRazorConfig(weight_bits={self.weight_bits}, "
            f"activation_bits={self.activation_bits}, "
            f"kv_cache_bits={self.kv_cache_bits}, "
            f"weight_block_size={self.weight_block_size}, "
            f"quant_mode={self.quant_mode!r})"
        )

    def get_name(self) -> str:
        return "edgerazor"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # Pure-Python path works on any GPU / CPU
        return 50

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EdgeRazorConfig":
        """Create config from a model's ``quantization_config`` dict."""
        weight_bits = config.get("weight_bits", 4)
        return cls(
            weight_bits=weight_bits,
            er_block_size=config.get("er_block_size"),
            ie_block_size=config.get("ie_block_size"),
            activation_bits=config.get("activation_bits", 16),
            kv_cache_bits=config.get("kv_cache_bits", 16),
            quant_mode=config.get("quant_mode", ""),
            modules_to_not_convert=config.get("modules_to_not_convert", []),
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

        if hf_quant_cfg and hf_quant_cfg.get("quant_method") == "edgerazor":
            logger.info(
                "Auto-detected EdgeRazor model (quantization_config.quant_method=edgerazor)"
            )
            return "edgerazor"

        if hf_config is not None:
            qc = getattr(hf_config, "quantization_config", None)
            if qc and qc.get("quant_method") == "edgerazor":
                logger.info("Auto-detected EdgeRazor model")
                return "edgerazor"
            if hasattr(hf_config, "edgerazor_qconfig"):
                logger.info(
                    "Auto-detected EdgeRazor model (edgerazor_qconfig=%s)",
                    hf_config.edgerazor_qconfig,
                )
                return "edgerazor"
            if hasattr(hf_config, "quant_mode") and hf_config.quant_mode:
                logger.info(
                    "Auto-detected EdgeRazor model (quant_mode=%s)",
                    hf_config.quant_mode,
                )
                return "edgerazor"

        return None

    # ── backend selection ────────────────────────────────────────

    def _select_backend(self):
        """Select linear method backend based on GPU capability.

        W4A16 / W4A8 / W1.58A16 / W1.58A8 → Marlin (W1.58 upcast to W4).
        Falls back to pure-Python on older GPUs or CPU.
        """
        global _BACKEND
        from .linear_marlin import can_use_marlin

        use_marlin = can_use_marlin(self) and self.weight_bits in (1.58, 4)

        if use_marlin:
            if _BACKEND != "marlin":
                _BACKEND = "marlin"
                logger.info(
                    "[EdgeRazor] Backend: Marlin (CUDA sm>=75 fused kernel, W%dA%d)",
                    self.weight_bits,
                    self.activation_bits,
                )
            return self._marlin_method()
        else:
            if _BACKEND != "py":
                _BACKEND = "py"
                logger.info(
                    "[EdgeRazor] Backend: pure-Python "
                    "(dequant + torch.matmul, W%dA%d)",
                    self.weight_bits,
                    self.activation_bits,
                )
            return self._py_method()

    def _marlin_method(self):
        from .linear_marlin import EdgeRazorMarlinLinearMethod

        return EdgeRazorMarlinLinearMethod(self)

    def _py_method(self):
        from .linear_py import EdgeRazorPyLinearMethod

        return EdgeRazorPyLinearMethod(self)

    # ── dispatch ─────────────────────────────────────────────────

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if self._is_layer_skipped(prefix):
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod

                return UnquantizedLinearMethod()
            return self._select_backend()
        if isinstance(layer, VocabParallelEmbedding):
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                UnquantizedEmbeddingMethod,
            )

            return UnquantizedEmbeddingMethod()
        return None

    def _is_layer_skipped(self, prefix: str) -> bool:
        return any(m in prefix for m in self.modules_to_not_convert)
