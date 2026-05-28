"""EdgeRazor vLLM Plugin — W4-A8-KV8 quantized inference.

Auto-registered with vLLM via ``@register_quantization_config("edgerazor")``.
Putting this module under ``src/edgerazor/vllm/`` lets users install it as an
*optional* extra of the main ``edgerazor`` package:

.. code:: bash

    pip install edgerazor[vllm]
    vllm serve /path/to/model --quantization edgerazor

When vLLM is not installed, the pure quant ops (``quant_ops.py``) are still
importable and usable standalone.
"""

# Pure quant ops (no vLLM dependency) — always available
from .quant_ops import (
    INT4_MAX,
    INT8_MAX,
    W4A8_BLOCK_SIZE,
    dequantize_weight,
    pack_int4,
    quantize_activation_per_block_int8,
    quantize_activation_per_token_int8,
    quantize_weight_per_block_int4,
    quantize_weight_ternary_to_int4,
    unpack_int4,
)

__all__ = [
    # Pure ops
    "dequantize_weight",
    "pack_int4",
    "unpack_int4",
    "quantize_weight_per_block_int4",
    "quantize_weight_ternary_to_int4",
    "quantize_activation_per_token_int8",
    "quantize_activation_per_block_int8",
    "W4A8_BLOCK_SIZE",
    "INT4_MAX",
    "INT8_MAX",
    "register",
    "EdgeRazorConfig",
    "EdgeRazorLinearMethod",
]

# vLLM-dependent classes — imported lazily via register() to avoid triggering
# vLLM/PyTorch import chains during plugin discovery (module-level imports
# can surface unrelated PyTorch inductor bugs as plugin load failures).
EdgeRazorConfig = None  # type: ignore[assignment,misc]
EdgeRazorLinearMethod = None  # type: ignore[assignment,misc]


def register() -> bool:
    """Plugin entry point for vLLM auto-discovery (``vllm.general_plugins``).

    This function is called automatically by vLLM at startup via the
    ``pyproject.toml`` entry point::

        [project.entry-points."vllm.general_plugins"]
        edgerazor = "edgerazor.vllm:register"

    It imports the quantization module which triggers
    ``@register_quantization_config``, making ``--quantization edgerazor``
    available without any manual imports.

    Returns ``True`` on success, ``False`` if vLLM is not installed or the
    import chain fails (e.g. due to vLLM/PyTorch version incompatibilities).
    """
    global EdgeRazorConfig, EdgeRazorLinearMethod
    try:
        from .edgerazor_quant import (  # noqa: F401
            EdgeRazorConfig as _EdgeRazorConfig,
        )
        from .edgerazor_quant import (
            EdgeRazorLinearMethod as _EdgeRazorLinearMethod,
        )
        EdgeRazorConfig = _EdgeRazorConfig
        EdgeRazorLinearMethod = _EdgeRazorLinearMethod
        return True
    except Exception:
        return False
