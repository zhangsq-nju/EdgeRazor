"""EdgeRazor vLLM Plugin — W1.58-A8 / W1.58-A16 / W4-A16 / W4-A8 quantized inference.

Auto-registered with vLLM via ``@register_quantization_config("edgerazor")``.

.. code:: bash

    pip install edgerazor[vllm]
    vllm serve /path/to/model --quantization edgerazor

Backends (auto-selected):
  - ``linear_marlin.py``  —  CUDA sm>=75, fused Marlin GEMM kernel (W4 only)
  - ``linear_py.py``      —  pure-Python dequant + torch.matmul (fallback / W1.58)
"""

# ── Pure quant ops (no vLLM dependency) ─────────────────────────

from .quant_ops import (
    ER_W1_58A8_BLOCK_SIZE,
    ER_W4A8_BLOCK_SIZE,
    IE_W1_58A8_BLOCK_SIZE,
    IE_W4A8_BLOCK_SIZE,
    INT1_58_MAX,
    INT4_MAX,
    INT8_MAX,
    W1_58A8_BLOCK_SIZE,
    W4A8_BLOCK_SIZE,
    dequantize_weight,
    pack_int2,
    pack_int4,
    quantize_activation_per_block_int8,
    quantize_activation_per_token_int8,
    quantize_weight_per_block_int2,
    quantize_weight_per_block_int4,
    quantize_weight_ternary_to_int4,
    resolve_quant_block,
    unpack_int2,
    unpack_int4,
)

__all__ = [
    # Pure ops
    "dequantize_weight",
    "pack_int4", "unpack_int4", "pack_int2", "unpack_int2",
    "quantize_weight_per_block_int4", "quantize_weight_per_block_int2",
    "quantize_weight_ternary_to_int4",
    "quantize_activation_per_token_int8", "quantize_activation_per_block_int8",
    "resolve_quant_block",
    # Block-size constants
    "ER_W1_58A8_BLOCK_SIZE", "ER_W4A8_BLOCK_SIZE",
    "IE_W1_58A8_BLOCK_SIZE", "IE_W4A8_BLOCK_SIZE",
    "W1_58A8_BLOCK_SIZE", "W4A8_BLOCK_SIZE",
    "INT1_58_MAX", "INT4_MAX", "INT8_MAX",
    # Plugin
    "register", "EdgeRazorConfig",
]

# ── vLLM-dependent classes — lazy import via register() ─────────

EdgeRazorConfig = None  # type: ignore[assignment]


def register() -> bool:
    """Plugin entry point for vLLM auto-discovery (``vllm.general_plugins``).

    Called automatically by vLLM at startup via the ``pyproject.toml``
    entry point::

        [project.entry-points."vllm.general_plugins"]
        edgerazor = "edgerazor.vllm:register"
    """
    global EdgeRazorConfig
    try:
        from .edgerazor_quant import (  # noqa: F401
            EdgeRazorConfig as _EdgeRazorConfig,
        )
        EdgeRazorConfig = _EdgeRazorConfig
        return True
    except Exception:
        return False
