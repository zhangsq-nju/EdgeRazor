"""Quantization-Aware Training (QAT) module
- Training: Quantization-Aware Training (QAT)
- Inference: llama.cpp, ggml, etc
- Details of quantization implementation:
    - Bitwidth: 1.58-bit (ternary), 2-bit, 3-bit, 4-bit
    - Granularity: Per-Tensor, Per-Channel, Per-Token, Per-Group (Block)
    - Algorithm: Uniform, Non-uniform
"""
# ruff: noqa: F401

from .block import (
    QMultiheadAttention,
    QuantizedKVState,
    create_quantized_kv_cache,
)
from .module import QConv1d, QConv2d, QConv3d, QEmbedding, QLinear
from .qat import QAT
from .util.quant_config import QuantConfig

__all__ = [
    # Main QAT class
    "QAT",
    "QuantConfig",
    # Quantized modules
    "QLinear",
    "QEmbedding",
    "QConv1d",
    "QConv2d",
    "QConv3d",
    # Quantized attention blocks
    "QMultiheadAttention",
    # KV Cache quantization
    "QuantizedKVState",
    "create_quantized_kv_cache",
]
