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


def create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.01,
    with_activation_kv=False,
    a_block_size=256,
    kv_block_size=128,
):
    """
    Create w1_58 quantization config.

    Args:
        mp_prop: mixed precision proportion (e.g. 0.01 or 0.05)
        with_activation_kv: whether to include activation and kv_cache quantization
    """
    target_types = ["linear", "embedding"]
    if with_activation_kv:
        target_types.append("kv_cache")
    
    config = OrderedDict(
        [
            ("method", "QAT"),
            (
                "select",
                OrderedDict(
                    [
                        ("target_types", target_types),
                        ("target_names", []),
                        ("exclude_types", []),
                        ("exclude_names", []),
                    ]
                ),
            ),
            (
                "function",
                OrderedDict(
                    [
                        ("epsilon", 1e-05),
                        (
                            "weight_function",
                            w_func,
                        ),
                        ("w_scale_factor", 2.0),
                        ("w_block_size", 256),
                        ("w_mixed_precision_prop", mp_prop),
                        ("is_w_quantized", True),
                        ("activation_function", ""),
                        ("a_block_size", -1),
                        ("a_mixed_precision_prop", -1.0),
                        ("kv_cache_function", ""),
                        ("kv_block_size", -1),
                        ("kv_mixed_precision_prop", -1.0),
                    ]
                ),
            ),
            ("training", "all"),
        ]
    )

    if with_activation_kv:
        config["function"]["activation_function"] = (
            "state_quant_uniform_symmetric_absmax_per_block_int8"
        )
        config["function"]["a_block_size"] = a_block_size
        config["function"]["kv_cache_function"] = (
            "state_quant_uniform_symmetric_absmax_per_block_int8"
        )
        config["function"]["kv_block_size"] = kv_block_size

    return config


def create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=False,
    w_block_size=256,
    a_block_size=256,
    kv_block_size=128,
):
    """
    Create w1_58 quantization config, using int4 quantization for embedding and lm_head.

    Args:
        mp_prop: mixed precision proportion (e.g. 0.125, 0.25, or 0.50)
        with_activation_kv: whether to include activation and kv_cache quantization
    """
    target_types = ["linear", "embedding"]
    if with_activation_kv:
        target_types.append("kv_cache")

    config = OrderedDict(
        [
            ("method", "QAT"),
            (
                "select",
                OrderedDict(
                    [
                        ("target_types", target_types),
                        ("target_names", []),
                        ("exclude_types", []),
                        ("exclude_names", []),
                    ]
                ),
            ),
            (
                "function",
                OrderedDict(
                    [
                        ("epsilon", 1e-05),
                        (
                            "weight_function",
                            w_func,
                        ),
                        ("w_scale_factor", 2.0),
                        ("w_block_size", w_block_size),
                        ("w_mixed_precision_prop", mp_prop),
                        ("is_w_quantized", True),
                        ("activation_function", ""),
                        ("a_block_size", -1),
                        ("a_mixed_precision_prop", -1.0),
                        ("kv_cache_function", ""),
                        ("kv_block_size", -1),
                        ("kv_mixed_precision_prop", -1.0),
                    ]
                ),
            ),
            (
                "overrides",
                [
                    {
                        "name": ".*embed_tokens",
                        "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                        "w_scale_factor": -1,
                    },
                    {
                        "name": ".*lm_head",
                        "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                        "w_scale_factor": -1,
                    },
                ],
            ),
            ("training", "all"),
        ]
    )

    if with_activation_kv:
        config["function"]["activation_function"] = (
            "state_quant_uniform_symmetric_absmax_per_block_int8"
        )
        config["function"]["a_block_size"] = a_block_size
        config["function"]["kv_cache_function"] = (
            "state_quant_uniform_symmetric_absmax_per_block_int8"
        )
        config["function"]["kv_block_size"] = kv_block_size

    return config


w4a16kv16_qwen3 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_scale_factor", 2.0),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    ("activation_function", ""),
                    ("a_block_size", -1),
                    ("a_mixed_precision_prop", -1.0),
                    ("kv_cache_function", ""),
                    ("kv_block_size", -1),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w4a8kv8_qwen3 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "kv_cache"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 256),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 128),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w4a8kv8_qwen2_5_omni = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "kv_cache"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", ["thinker.audio_tower.*", "talker.*", "token2wav.*"]),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_block_size", 32),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 32),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 32),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w4a8kv8_mobilellm = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "kv_cache"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 256),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 256),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w1_58a16kv16_qwen3 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                    ),
                    ("w_scale_factor", 2.0),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", 0.00),
                    ("is_w_quantized", True),
                    ("activation_function", ""),
                    ("a_block_size", -1),
                    ("a_mixed_precision_prop", -1.0),
                    ("kv_cache_function", ""),
                    ("kv_block_size", -1),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w1_58a8kv8_qwen3 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "kv_cache"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                    ),
                    ("w_scale_factor", 2.0),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", 0.00),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 256),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 128),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

## Standard quantization config
# W2.79 Quantization Configs
w2_79a16kv16_embint4_qwen3 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=False,
)
w2_79a8kv8_embint4_qwen3 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=True,
)
w2_79a8kv8_embint4_mobilellm = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=True,
    w_block_size=64,
    a_block_size=64,
    kv_block_size=64,
)

# W1.88 Quantization Configs
w1_88a16kv16_embint4_qwen3 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=False,
)
w1_88a8kv8_embint4_qwen3 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=True,
)
w1_88a8kv8_embint4_mobilellm = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=True,
    w_block_size=64,
    a_block_size=64,
    kv_block_size=64,
)

# W1.58 Quantization Configs
w1_58a16kv16_embint4_qwen3 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.00,
    with_activation_kv=False,
)
w1_58a8kv8_embint4_qwen3 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.00,
    with_activation_kv=True,
)
w1_58a8kv8_embint4_mobilellm = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.00,
    with_activation_kv=True,
    w_block_size=64,
    a_block_size=64,
    kv_block_size=64,
)

# Map quant_mode string to imported config dict
quant_config_map = {
    # Qwen3-specific configs with embedding int4
    "w4a8kv8_qwen3": w4a8kv8_qwen3,
    "w2_79a8kv8_embint4_qwen3": w2_79a8kv8_embint4_qwen3,
    "w1_88a8kv8_embint4_qwen3": w1_88a8kv8_embint4_qwen3,
    "w1_58a8kv8_embint4_qwen3": w1_58a8kv8_embint4_qwen3,
    # Qwen2.5-Omni-specific configs with embedding int4
    "w4a8kv8_qwen2_5_omni": w4a8kv8_qwen2_5_omni,
    # MobileLLM-specific configs with embedding int4
    "w4a8kv8_mobilellm": w4a8kv8_mobilellm,
    "w2_79a8kv8_embint4_mobilellm": w2_79a8kv8_embint4_mobilellm,
    "w1_88a8kv8_embint4_mobilellm": w1_88a8kv8_embint4_mobilellm,
    "w1_58a8kv8_embint4_mobilellm": w1_58a8kv8_embint4_mobilellm,
}
