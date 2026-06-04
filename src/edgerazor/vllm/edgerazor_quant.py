"""
EdgeRazor quantization plugin for vLLM.

Supports W1.58 (ternary) and W4 (INT4) weight quantization.

Layer selection is driven by ``quant_mode`` via :mod:`.quant_mode_parse`.
When ``quant_mode`` is absent, the default is: quantize decoder Linear layers,
skip embedding and lm_head.

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

from .quant_mode_parse import QuantModeConfig as _QuantModeConfig
from .quant_ops import (
    ER_W1_58A8_BLOCK_SIZE,
    ER_W4A8_BLOCK_SIZE,
    IE_W1_58A8_BLOCK_SIZE,
    IE_W4A8_BLOCK_SIZE,
    resolve_quant_block,
)

logger = init_logger("vllm.edgerazor.quant")

_BACKEND = None  # cached backend name for logging

SUPPORTED_W_BITS = (1.58, 1.88, 2.79, 4)
SUPPORTED_A_BITS = (8, 16)

SUPPORTED_BACKENDS = ("marlin", "py")

@register_quantization_config("edgerazor")
class EdgeRazorConfig(QuantizationConfig):
    """Quantization config for EdgeRazor quantized models.

    Supports W1.58 (ternary) and W4 (INT4) weight quantization.

    Auto-detected via ``edgerazor_qconfig`` in config.json or explicit
    ``--quantization edgerazor``.
    """

    def __init__(
        self,
        weight_bits: float | None = None,
        weight_block_size: int | list[int] | None = None,
        er_block_size: int | None = None,
        ie_block_size: int | None = None,
        activation_bits: int | None = None,
        kv_cache_bits: int | None = None,
        quant_mode: str = "",
        modules_to_not_convert: list[str] | None = None,
        backend: str | None = None,
        quant_emb: bool | None = None,
        quant_lm_head: bool | None = None,
        quant_emb_bits: float | None = None,
        quant_lm_head_bits: float | None = None,
    ) -> None:
        super().__init__()

        # ── resolve quant_mode ─────────────────────────────────
        if quant_mode:
            self._quant_mode_cfg: _QuantModeConfig | None = _QuantModeConfig(
                quant_mode,
                weight_bits_override=weight_bits,
                activation_bits_override=activation_bits,
            )
            weight_bits = self._quant_mode_cfg.weight_bits
            activation_bits = self._quant_mode_cfg.activation_bits
            kv_cache_bits = self._quant_mode_cfg.kv_cache_bits
            modules_to_not_convert = modules_to_not_convert or (
                self._quant_mode_cfg.exclude_names
            )
        else:
            self._quant_mode_cfg = None

        # Apply defaults when not specified and no quant_mode
        if weight_bits is None:
            weight_bits = 4
        if activation_bits is None:
            activation_bits = 16

        # ── validate ───────────────────────────────────────────
        if weight_bits not in SUPPORTED_W_BITS:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}. "
                f"EdgeRazor supports: {SUPPORTED_W_BITS}"
            )
        if activation_bits not in SUPPORTED_A_BITS:
            raise ValueError(
                f"Unsupported activation_bits={activation_bits}. "
                f"EdgeRazor supports: {SUPPORTED_A_BITS}"
            )
        if backend is not None and backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend={backend}. "
                f"EdgeRazor supports: {SUPPORTED_BACKENDS}"
            )
        self.weight_bits = weight_bits
        self._user_backend = backend
        self.activation_bits = activation_bits
        self.kv_cache_bits = kv_cache_bits
        self.quant_mode = quant_mode
        self.modules_to_not_convert = modules_to_not_convert or []
        self._quant_emb = quant_emb
        self._quant_lm_head = quant_lm_head
        self._quant_emb_bits = quant_emb_bits
        self._quant_lm_head_bits = quant_lm_head_bits

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

        wbits_label = "1.58" if self.weight_bits == 1.58 else str(self.weight_bits)
        logger.info(
            "[EdgeRazor] Quantization config created: "
            "weight_bits=%s, activation_bits=%d, kv_cache_bits=%d, "
            "weight_block_size=%s, quant_mode=%s",
            wbits_label,
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
        return cls(
            weight_bits=config.get("weight_bits"),
            er_block_size=config.get("er_block_size"),
            ie_block_size=config.get("ie_block_size"),
            activation_bits=config.get("activation_bits"),
            kv_cache_bits=config.get("kv_cache_bits"),
            quant_mode=config.get("quant_mode", ""),
            modules_to_not_convert=config.get("modules_to_not_convert", []),
            backend=config.get("backend"),
            quant_emb=config.get("quant_emb"),
            quant_lm_head=config.get("quant_lm_head"),
            quant_emb_bits=config.get("quant_emb_bits"),
            quant_lm_head_bits=config.get("quant_lm_head_bits"),
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
        """Select linear method backend.

        Priority:
        1. User-specified ``backend`` in config.json, if the environment
           supports it (Marlin requires CUDA sm>=75 + weight_bits in (1.58, 4)).
        2. Auto-select: Marlin when supported, otherwise pure-Python.
        """
        global _BACKEND
        from .linear_marlin import can_use_marlin

        # Marlin W4A8 / W1.58A8 kernel is experimental — route A8 to
        # the Python backend which handles activation dequantization explicitly.
        marlin_ok = (
            can_use_marlin(self)
            and self.weight_bits in (1.58, 4)
            and self.activation_bits != 8
        )
        preferred = self._user_backend

        # Resolve: user choice wins if compatible, else auto
        if preferred == "marlin" and marlin_ok:
            use_marlin = True
        elif preferred == "py":
            use_marlin = False
        elif preferred is not None:
            logger.warning(
                "[EdgeRazor] Backend '%s' unavailable (marlin_ok=%s), "
                "auto-selecting.",
                preferred, marlin_ok,
            )
            use_marlin = marlin_ok
        else:
            use_marlin = marlin_ok

        if use_marlin:
            if _BACKEND != "marlin":
                _BACKEND = "marlin"
                logger.info(
                    "[EdgeRazor] Backend: Marlin (CUDA sm>=75 fused kernel, W%sA%d)",
                    "1.58" if self.weight_bits == 1.58 else str(self.weight_bits),
                    self.activation_bits,
                )
            return self._marlin_method()
        else:
            if _BACKEND != "py":
                _BACKEND = "py"
                logger.info(
                    "[EdgeRazor] Backend: pure-Python "
                    "(dequant + torch.matmul, W%sA%d)",
                    "1.58" if self.weight_bits == 1.58 else str(self.weight_bits),
                    self.activation_bits,
                )
            return self._py_method()

    def _marlin_method(self):
        from .linear_marlin import EdgeRazorMarlinLinearMethod

        return EdgeRazorMarlinLinearMethod(self)

    def _py_method(self):
        from .linear_py import EdgeRazorPyLinearMethod

        return EdgeRazorPyLinearMethod(self)

    # ── per-layer helpers ───────────────────────────────────────

    def _layer_weight_bits(self, prefix: str) -> float:
        """Resolve *weight_bits* for a specific layer.

        Priority (highest first):

        1. ``quant_emb_bits`` / ``quant_lm_head_bits`` from config.json
        2. :class:`QuantModeConfig` overrides + base function
        3. Global ``weight_bits``
        """
        if "embed_tokens" in prefix and self._quant_emb_bits is not None:
            return self._quant_emb_bits
        if prefix.endswith("lm_head") and self._quant_lm_head_bits is not None:
            return self._quant_lm_head_bits

        if self._quant_mode_cfg is not None:
            return self._quant_mode_cfg.get_weight_bits(prefix)
        return self.weight_bits

    def _is_layer_quantized(self, prefix: str) -> bool:
        """Check whether *prefix* should be quantized.

        Priority (highest first):

        1. ``quant_emb`` / ``quant_lm_head`` from config.json
        2. *quant_mode* overrides + base function
        3. Default: quantize decoder Linear layers, skip embedding / lm_head.
        """
        # Priority 1: explicit quant_emb / quant_lm_head (bool) or bits
        # Setting bits implies quantization is enabled
        if "embed_tokens" in prefix:
            if self._quant_emb is not None:
                return self._quant_emb
            if self._quant_emb_bits is not None:
                return True  # bits set → implicitly enable quantization
        if prefix.endswith("lm_head"):
            if self._quant_lm_head is not None:
                return self._quant_lm_head
            if self._quant_lm_head_bits is not None:
                return True

        # Priority 2: quant_mode config (if present)
        if self._quant_mode_cfg is not None:
            return self._quant_mode_cfg.is_layer_quantized(prefix)

        # Priority 3: default — skip embedding and lm_head
        if "embed_tokens" in prefix or prefix.endswith("lm_head"):
            return False
        return not any(m in prefix for m in self.modules_to_not_convert)

    def _clone_with_weight_bits(self, weight_bits: float) -> "EdgeRazorConfig":
        """Shallow clone with overridden *weight_bits* + recalculated block sizes."""
        import copy
        clone = copy.copy(self)
        clone.weight_bits = weight_bits

        # Re-resolve ER / IE block sizes for the new weight_bits
        clone.er_block_size = (
            ER_W1_58A8_BLOCK_SIZE if weight_bits == 1.58 else ER_W4A8_BLOCK_SIZE
        )
        clone.ie_block_size = (
            IE_W1_58A8_BLOCK_SIZE if weight_bits == 1.58 else IE_W4A8_BLOCK_SIZE
        )
        resolved = resolve_quant_block(clone.er_block_size, clone.ie_block_size)
        clone._quant_block_size = resolved[0]
        clone._scale_block_size = resolved[1]
        clone._needs_scale_split = resolved[2]
        clone.weight_block_size = [clone.ie_block_size]
        return clone

    # ── dispatch ─────────────────────────────────────────────────

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # ── decoder Linear layers ────────────────────────────────
        if isinstance(layer, LinearBase):
            return self._dispatch_linear(layer, prefix)

        # ── Embedding / LM head ───────────────────────────────────
        if isinstance(layer, VocabParallelEmbedding):
            return self._dispatch_embedding(layer, prefix)

        return None

    # ── per-type helpers ─────────────────────────────────────────

    def _dispatch_linear(
        self, layer: LinearBase, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        if not self._is_layer_quantized(prefix):
            return UnquantizedLinearMethod()

        wb = self._layer_weight_bits(prefix)
        if wb not in SUPPORTED_W_BITS:
            return UnquantizedLinearMethod()

        layer._edgerazor_layer_name = prefix

        # Mixed precision (1.88 / 2.79): Python backend only (no Marlin kernel)
        if wb in (1.88, 2.79):
            from .linear_py import EdgeRazorPyMixedPrecisionLinearMethod
            cfg = self if wb == self.weight_bits else self._clone_with_weight_bits(wb)
            return EdgeRazorPyMixedPrecisionLinearMethod(cfg)

        cfg = self if wb == self.weight_bits else self._clone_with_weight_bits(wb)
        return cfg._select_backend()

    def _dispatch_embedding(
        self, layer: VocabParallelEmbedding, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        if not self._is_layer_quantized(prefix):
            return UnquantizedEmbeddingMethod()

        wb = self._layer_weight_bits(prefix)
        if wb not in SUPPORTED_W_BITS:
            return UnquantizedEmbeddingMethod()

        layer._edgerazor_layer_name = prefix

        # vLLM: type(self) is VocabParallelEmbedding distinguishes true
        # embeddings from ParallelLMHead (subclass used as linear proj).
        if type(layer) is not VocabParallelEmbedding:
            # lm_head — route to linear backend (Marlin / Python)
            cfg = self if wb == self.weight_bits else self._clone_with_weight_bits(wb)
            return cfg._select_backend()

        # embed_tokens — sparse-lookup embedding method.
        # When tie_word_embeddings=True, vLLM sets lm_head = embed_tokens,
        # so EdgeRazorEmbeddingMethod handles both .embedding() and .apply().
        from .embedding import EdgeRazorEmbeddingMethod
        cfg = self if wb == self.weight_bits else self._clone_with_weight_bits(wb)
        return EdgeRazorEmbeddingMethod(cfg)
