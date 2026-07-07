"""
Mapping of quantization functions and classes.

`str -> function/class`
"""

from collections import OrderedDict

import torch.nn as nn

# Import directly from the source module to avoid circular import
from .util.quant_function import (
    state_quant_uniform_symmetric_absmax_per_block_int2,
    state_quant_uniform_symmetric_absmax_per_block_int4,
    state_quant_uniform_symmetric_absmax_per_block_int8,
    state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic,  # deprecated
    state_quant_uniform_symmetric_absmax_per_token_int2,
    state_quant_uniform_symmetric_absmax_per_token_int4,
    state_quant_uniform_symmetric_absmax_per_token_int8,
    weight_quant_uniform_asymmetric_max_per_block_int4,
    weight_quant_uniform_asymmetric_max_per_channel_int4,
    weight_quant_uniform_asymmetric_max_per_tensor_int4,
    weight_quant_uniform_symmetric_absmax_per_block_int1_58,
    weight_quant_uniform_symmetric_absmax_per_block_int4,
    weight_quant_uniform_symmetric_absmax_per_block_int5,
    weight_quant_uniform_symmetric_absmax_per_block_int8,
    weight_quant_uniform_symmetric_absmax_per_channel_int1_58,
    weight_quant_uniform_symmetric_absmax_per_channel_int4,
    weight_quant_uniform_symmetric_absmax_per_tensor_int1_58,
    weight_quant_uniform_symmetric_absmax_per_tensor_int4,
    weight_quant_uniform_symmetric_clip_per_block_int1_58,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_column_wise,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse,
    weight_quant_uniform_symmetric_clip_per_channel_int1_58,
    weight_quant_uniform_symmetric_clip_per_tensor_int1_58,
)

# Collect all quantization functions automatically
_quant_functions = [
    # INT1_58 (Ternary) Weight Quantization - Clip Method
    weight_quant_uniform_symmetric_clip_per_tensor_int1_58,
    weight_quant_uniform_symmetric_clip_per_channel_int1_58,
    weight_quant_uniform_symmetric_clip_per_block_int1_58,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_column_wise,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse,
    # INT1_58 (Ternary) Weight Quantization - Absmax Method
    weight_quant_uniform_symmetric_absmax_per_tensor_int1_58,
    weight_quant_uniform_symmetric_absmax_per_channel_int1_58,
    weight_quant_uniform_symmetric_absmax_per_block_int1_58,
    # INT4 Weight Quantization - Symmetric Absmax Method
    weight_quant_uniform_symmetric_absmax_per_tensor_int4,
    weight_quant_uniform_symmetric_absmax_per_channel_int4,
    weight_quant_uniform_symmetric_absmax_per_block_int4,
    # INT4+ Weight Quantization - Symmetric Absmax Method
    weight_quant_uniform_symmetric_absmax_per_block_int5,
    weight_quant_uniform_symmetric_absmax_per_block_int8,
    # INT4 Weight Quantization - Asymmetric Max Method
    weight_quant_uniform_asymmetric_max_per_tensor_int4,
    weight_quant_uniform_asymmetric_max_per_channel_int4,
    weight_quant_uniform_asymmetric_max_per_block_int4,
    # INT2 State Quantization - Absmax Method
    state_quant_uniform_symmetric_absmax_per_token_int2,
    state_quant_uniform_symmetric_absmax_per_block_int2,
    # INT4 State Quantization - Absmax Method
    state_quant_uniform_symmetric_absmax_per_token_int4,
    state_quant_uniform_symmetric_absmax_per_block_int4,
    state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic,
    # INT8 State Quantization - Absmax Method
    state_quant_uniform_symmetric_absmax_per_token_int8,
    state_quant_uniform_symmetric_absmax_per_block_int8,
]

# Build the map automatically: function_name -> function
quant_function_map = {func.__name__: func for func in _quant_functions}

modules_map = {
    "linear": nn.Linear,
    "embedding": nn.Embedding,
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "conv3d": nn.Conv3d,
    "multiheadattention": nn.MultiheadAttention,
}


# Quantization function name constants
_W4_ABSMAX = "weight_quant_uniform_symmetric_absmax_per_block_int4"
_W1_58_CLIP = "weight_quant_uniform_symmetric_clip_per_block_int1_58"
_W1_58_CLIP_MP = "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse"
_A_INT8 = "state_quant_uniform_symmetric_absmax_per_block_int8"

_EMBINT4_OVERRIDES = [
    {"name": ".*embed_tokens", "weight_function": _W4_ABSMAX, "w_scale_factor": -1},
    {"name": ".*lm_head", "weight_function": _W4_ABSMAX, "w_scale_factor": -1},
]


def create_quant_config(
    *,
    w_func=None,
    mp_prop=-1.0,
    w_scale_factor=None,
    with_activation_kv=False,
    is_w_quantized=True,
    target_types: list[str] | None = None,
    w_block_size=256,
    a_block_size=256,
    kv_block_size=128,
    exclude_names=None,
    overrides=None,
):
    """Create a QAT quantization config.

    Args:
        w_func: Weight quantization function name. If None, auto-derived from mp_prop
            (0 -> pure W1.58 clip, !=0 -> mixed-precision W1.58 clip).
        mp_prop: Mixed precision proportion.
        w_scale_factor: Weight scale factor. None omits the key from the config.
        with_activation_kv: Whether to include activation and KV cache quantization.
        is_w_quantized: Whether weights are already quantized (inference mode). Note the default value is True, as we tend to release models with fake quantized weights.
        target_types: Module types to quantize. Defaults to ["linear", "embedding"].
    """
    if w_func is None:
        w_func = _W1_58_CLIP if mp_prop == 0 else _W1_58_CLIP_MP

    if target_types is None:
        target_types = ["linear", "embedding"]
    if with_activation_kv and "kv_cache" not in target_types:
        target_types.append("kv_cache")

    func_items = [
        ("epsilon", 1e-05),
        ("weight_function", w_func),
    ]
    if w_scale_factor is not None:
        func_items.append(("w_scale_factor", w_scale_factor))
    func_items.extend([
        ("w_block_size", w_block_size),
        ("w_mixed_precision_prop", mp_prop),
        ("is_w_quantized", is_w_quantized),
        ("activation_function", _A_INT8 if with_activation_kv else ""),
        ("a_block_size", a_block_size if with_activation_kv else -1),
        ("a_mixed_precision_prop", -1.0),
        ("kv_cache_function", _A_INT8 if with_activation_kv else ""),
        ("kv_block_size", kv_block_size if with_activation_kv else -1),
        ("kv_mixed_precision_prop", -1.0),
    ])

    items = [
        ("method", "QAT"),
        ("select", OrderedDict([
            ("target_types", target_types),
            ("target_names", []),
            ("exclude_types", []),
            ("exclude_names", exclude_names or []),
        ])),
        ("function", OrderedDict(func_items)),
    ]
    if overrides is not None:
        items.append(("overrides", overrides))
    items.append(("training", "all"))

    return OrderedDict(items)


# W4 Quantization Configs
w4a16kv16_qwen3 = create_quant_config(w_func=_W4_ABSMAX, w_scale_factor=2.0)

w4a8kv8_qwen3 = create_quant_config(w_func=_W4_ABSMAX, with_activation_kv=True)

w4a8kv8_qwen2_5_omni = create_quant_config(
    w_func=_W4_ABSMAX, with_activation_kv=True,
    w_block_size=32, a_block_size=32, kv_block_size=32,
    exclude_names=["thinker.audio_tower.*", "talker.*", "token2wav.*"],
)

w4a8kv8_mobilellm = create_quant_config(
    w_func=_W4_ABSMAX, with_activation_kv=True, kv_block_size=256,
)

# W1.58 Pure Quantization Configs
w1_58a16kv16_qwen3 = create_quant_config(mp_prop=0.00, w_scale_factor=2.0)

w1_58a8kv8_qwen3 = create_quant_config(
    mp_prop=0.00, w_scale_factor=2.0, with_activation_kv=True,
)

# W2.79 Mixed-Precision Quantization Configs
w2_79a16kv16_embint4_qwen3 = create_quant_config(
    mp_prop=0.50, w_scale_factor=2.0, overrides=_EMBINT4_OVERRIDES,
)
w2_79a8kv8_embint4_qwen3 = create_quant_config(
    mp_prop=0.50, w_scale_factor=2.0, with_activation_kv=True, overrides=_EMBINT4_OVERRIDES,
)
w2_79a8kv8_embint4_mobilellm = create_quant_config(
    mp_prop=0.50, w_scale_factor=2.0, with_activation_kv=True,
    w_block_size=64, a_block_size=64, kv_block_size=64, overrides=_EMBINT4_OVERRIDES,
)

# W1.88 Mixed-Precision Quantization Configs
w1_88a16kv16_embint4_qwen3 = create_quant_config(
    mp_prop=0.125, w_scale_factor=2.0, overrides=_EMBINT4_OVERRIDES,
)
w1_88a8kv8_embint4_qwen3 = create_quant_config(
    mp_prop=0.125, w_scale_factor=2.0, with_activation_kv=True, overrides=_EMBINT4_OVERRIDES,
)
w1_88a8kv8_embint4_mobilellm = create_quant_config(
    mp_prop=0.125, w_scale_factor=2.0, with_activation_kv=True,
    w_block_size=64, a_block_size=64, kv_block_size=64, overrides=_EMBINT4_OVERRIDES,
)

# W1.58 Quantization Configs with Embedding INT4
w1_58a16kv16_embint4_qwen3 = create_quant_config(
    mp_prop=0.00, w_scale_factor=2.0, overrides=_EMBINT4_OVERRIDES,
)
w1_58a8kv8_embint4_qwen3 = create_quant_config(
    mp_prop=0.00, w_scale_factor=2.0, with_activation_kv=True, overrides=_EMBINT4_OVERRIDES,
)
w1_58a8kv8_embint4_mobilellm = create_quant_config(
    mp_prop=0.00, w_scale_factor=2.0, with_activation_kv=True,
    w_block_size=64, a_block_size=64, kv_block_size=64, overrides=_EMBINT4_OVERRIDES,
)

# ── New generic configs (model-family agnostic, v1.3.4+) ──

# W4 standard (all block_size=64)
w4a8kv8 = create_quant_config(
    w_func=_W4_ABSMAX, with_activation_kv=True,
    w_block_size=64, a_block_size=64, kv_block_size=64,
)
w4a8kv8_embint4 = create_quant_config(
    w_func=_W4_ABSMAX, with_activation_kv=True,
    w_block_size=64, a_block_size=64, kv_block_size=64, overrides=_EMBINT4_OVERRIDES,
)
w4a8 = create_quant_config(w_func=_W4_ABSMAX, w_block_size=64)

# W2.79 mixed-precision (w/a=256, kv=64)
w2_79a8kv8 = create_quant_config(mp_prop=0.50, with_activation_kv=True, kv_block_size=64)
w2_79a8kv8_embint4 = create_quant_config(
    mp_prop=0.50, with_activation_kv=True, kv_block_size=64, overrides=_EMBINT4_OVERRIDES,
)

# W1.88 mixed-precision (w/a=256, kv=64)
w1_88a8kv8 = create_quant_config(mp_prop=0.125, with_activation_kv=True, kv_block_size=64)
w1_88a8kv8_embint4 = create_quant_config(
    mp_prop=0.125, with_activation_kv=True, kv_block_size=64, overrides=_EMBINT4_OVERRIDES,
)

# W1.58 pure ternary (w/a=256, kv=64)
w1_58a8kv8 = create_quant_config(mp_prop=0.00, with_activation_kv=True, kv_block_size=64)
w1_58a8kv8_embint4 = create_quant_config(
    mp_prop=0.00, with_activation_kv=True, kv_block_size=64, overrides=_EMBINT4_OVERRIDES,
)
w1_58a8 = create_quant_config(mp_prop=0.00)

# W1.58 with activation quantized, KV in FP16, decoder-only linear (no embedding/lm_head)
w1_58a8kv16 = create_quant_config(
    mp_prop=0.00,
    with_activation_kv=True,
    w_scale_factor=2.0,
    target_types=["linear"],
    exclude_names=["lm_head"],
)
# kv16 → clear KV quant settings; keep only activation
w1_58a8kv16["function"]["kv_cache_function"] = ""
w1_58a8kv16["function"]["kv_block_size"] = -1
w1_58a8kv16["select"]["target_types"].remove("kv_cache")

# Activation + KV only (w/a=256, kv=64, no weight modules)
a8kv8 = create_quant_config(
    w_func=_W1_58_CLIP, with_activation_kv=True, is_w_quantized=False,
    target_types=[], kv_block_size=64,
)


# Map quant_mode string to imported config dict
quant_config_map = {
    # ── Model-agnostic configs (v1.3.4+) ──
    "w4a8kv8": w4a8kv8,
    "w4a8kv8_embint4": w4a8kv8_embint4,
    "w4a8": w4a8,
    "w2_79a8kv8": w2_79a8kv8,
    "w2_79a8kv8_embint4": w2_79a8kv8_embint4,
    "w1_88a8kv8": w1_88a8kv8,
    "w1_88a8kv8_embint4": w1_88a8kv8_embint4,
    "w1_58a8kv8": w1_58a8kv8,
    "w1_58a8kv8_embint4": w1_58a8kv8_embint4,
    "w1_58a8": w1_58a8,
    "w1_58a8kv16": w1_58a8kv16,
    "a8kv8": a8kv8,

    # ── Legacy model-specific configs (backward compatible) ──
    # Qwen3-specific
    "w4a8kv8_qwen3": w4a8kv8_qwen3,
    "w2_79a8kv8_embint4_qwen3": w2_79a8kv8_embint4_qwen3,
    "w1_88a8kv8_embint4_qwen3": w1_88a8kv8_embint4_qwen3,
    "w1_58a8kv8_embint4_qwen3": w1_58a8kv8_embint4_qwen3,
    # Qwen2.5-Omni-specific
    "w4a8kv8_qwen2_5_omni": w4a8kv8_qwen2_5_omni,
    # MobileLLM-specific
    "w4a8kv8_mobilellm": w4a8kv8_mobilellm,
    "w2_79a8kv8_embint4_mobilellm": w2_79a8kv8_embint4_mobilellm,
    "w1_88a8kv8_embint4_mobilellm": w1_88a8kv8_embint4_mobilellm,
    "w1_58a8kv8_embint4_mobilellm": w1_58a8kv8_embint4_mobilellm,
}

# Legacy aliases: old model-specific names → new generic names
_LEGACY_ALIASES = {
    "w4a8kv8_qwen3": "w4a8kv8",
    "w4a8kv8_qwen2_5_omni": "w4a8kv8",
    "w4a8kv8_mobilellm": "w4a8kv8",
    "w2_79a8kv8_embint4_qwen3": "w2_79a8kv8_embint4",
    "w1_88a8kv8_embint4_qwen3": "w1_88a8kv8_embint4",
    "w1_58a8kv8_embint4_qwen3": "w1_58a8kv8_embint4",
    "w2_79a8kv8_embint4_mobilellm": "w2_79a8kv8_embint4",
    "w1_88a8kv8_embint4_mobilellm": "w1_88a8kv8_embint4",
    "w1_58a8kv8_embint4_mobilellm": "w1_58a8kv8_embint4",
}
