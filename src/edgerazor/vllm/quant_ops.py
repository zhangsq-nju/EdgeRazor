"""
Pure quantization operations — no vLLM dependency.

These functions implement the EdgeRazor per-block INT4 weight quantization
format, which is compatible with GGUF Q4_0:

  W4-A8:  block_size=32, symmetric absmax per-block, 2×INT4 packed per uint8

Reference quant functions (from edgerazor.qat.util.quant_function):
  - weight_quant_uniform_symmetric_absmax_per_block_int4
  - state_quant_uniform_symmetric_absmax_per_token_int8
"""

from __future__ import annotations

import torch
from torch import Tensor

# ──────────────────────────────────────────────
# Constants (maps 1:1 with quant_function_config.py)
# ──────────────────────────────────────────────

W4A8_BLOCK_SIZE = 32  # from edgerazor.qat.util.quant_function_config
INT4_MAX = 7           # 2^(4-1) - 1
INT8_MAX = 127         # 2^(8-1) - 1

# block sizes for different schemes
W2A8_BLOCK_SIZE = 256
W5A8_BLOCK_SIZE = 32
W8A8_BLOCK_SIZE = 32


# ──────────────────────────────────────────────
# Pack / Unpack INT4
# ──────────────────────────────────────────────

def pack_int4(w_int: Tensor) -> Tensor:
    """Pack signed INT4 values (..., block_size) → uint8 (..., block_size//2).

    Each pair of adjacent INT4 values ``[a, b]`` along the last dimension is
    packed as::

        byte = ((a + 8) << 4) | (b + 8)

    ``a + 8`` shifts [-7, 7] to [1, 15] for unsigned nibble storage.

    Args:
        w_int: signed INT8 tensor with values in [-7, 7], last dim is even.

    Returns:
        Packed uint8 tensor, last dim halved.
    """
    w_shifted = w_int.to(torch.uint8) + 8  # [-7,7] → [1,15]
    even = w_shifted[..., ::2]
    odd = w_shifted[..., 1::2]
    return (even << 4) | odd


def unpack_int4(qweight: Tensor) -> Tensor:
    """Unpack uint8 (..., block_size//2) → signed INT8 (..., block_size).

    Args:
        qweight: packed uint8 tensor.

    Returns:
        Unpacked signed INT8 tensor with values in [-7, 7], last dim doubled.
    """
    even = ((qweight >> 4) & 0x0F).to(torch.int8) - 8
    odd = (qweight & 0x0F).to(torch.int8) - 8
    return torch.stack([even, odd], dim=-1).flatten(-2)


# ──────────────────────────────────────────────
# Per-block quantization / dequantization
# ──────────────────────────────────────────────

def quantize_weight_per_block_int4(
    w: Tensor,
    block_size: int = W4A8_BLOCK_SIZE,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Quantize bf16/fp16 weight to per-block INT4.

    The weight is reshaped to ``(out_dim, nblocks, block_size)``, a symmetric
    absmax scale is computed per block, and values are rounded into [-7, 7].

    Args:
        w: weight tensor, shape ``(out_dim, in_dim)``.
        block_size: group size per block (default 32, matching GGUF Q4_0).
        epsilon: minimum scale to avoid division by zero.

    Returns:
        ``(qweight, qweight_scale)`` where:
          - ``qweight``: packed uint8 ``(out_dim, in_dim // 2)``
          - ``qweight_scale``: bf16 ``(out_dim, nblocks)``
    """
    out_dim, in_dim = w.shape
    w_blocks = w.view(out_dim, -1, block_size)

    # Per-block symmetric absmax scale
    w_scale = w_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=epsilon) / INT4_MAX

    # Quantize to INT4 [-7, 7] and pack
    w_int = (w_blocks / w_scale).round().clamp(-INT4_MAX, INT4_MAX).to(torch.int8)
    qweight = pack_int4(w_int.view(out_dim, -1))
    qweight_scale = w_scale.squeeze(-1).contiguous().to(torch.bfloat16)

    return qweight, qweight_scale


def dequantize_weight(
    qweight: Tensor,
    qweight_scale: Tensor,
    block_size: int = W4A8_BLOCK_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize per-block INT4 weights to the target dtype.

    Args:
        qweight: packed uint8 ``(out_dim, in_dim // 2)``.
        qweight_scale: bf16 ``(out_dim, nblocks)``.
        block_size: group size per block.
        out_dtype: target dtype for dequantized weight.

    Returns:
        Dequantized weight tensor ``(out_dim, in_dim)``.
    """
    w_int = unpack_int4(qweight)  # (out, in) int8
    scale = qweight_scale.repeat_interleave(block_size, dim=1)  # (out, in)
    return (w_int.float() * scale.float()).to(out_dtype)


# ──────────────────────────────────────────────
# Activation quantization (per-token INT8)
# ──────────────────────────────────────────────

def quantize_activation_per_token_int8(
    x: Tensor,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Dynamically quantize activation to INT8 per-token.

    Quantizes along the last dimension (per-token). Equivalent to
    ``state_quant_uniform_symmetric_absmax_per_token_int8``.

    Args:
        x: activation tensor, shape ``([batch,] seq_len, hidden_dim)``.
        epsilon: minimum scale to avoid division by zero.

    Returns:
        ``(x_quant, x_scale)`` where:
          - ``x_quant``: INT8 tensor ``(..., hidden_dim)``
          - ``x_scale``: per-token scale, same shape prefix, last dim = 1
    """
    x_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=epsilon) / INT8_MAX
    x_int = (x / x_scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return x_int, x_scale


def quantize_activation_per_block_int8(
    x: Tensor,
    block_size: int = W2A8_BLOCK_SIZE,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Dynamically quantize activation to INT8 per-block.

    Divides the last dimension into blocks and quantizes each separately.
    Equivalent to ``state_quant_uniform_symmetric_absmax_per_block_int8``.

    Args:
        x: activation tensor, shape ``([batch,] seq_len, hidden_dim)``.
        block_size: group size per block (default 256, matching W2A8).
        epsilon: minimum scale to avoid division by zero.

    Returns:
        ``(x_quant, x_scale)`` where:
          - ``x_quant``: INT8 tensor, same shape as ``x``
          - ``x_scale``: per-block scale, expanded last dim into nblocks
    """
    shape = x.shape
    x_blocks = x.view(*shape[:-1], -1, block_size)
    x_scale = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=epsilon) / INT8_MAX
    x_int = (x_blocks / x_scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return x_int.view(shape), x_scale.squeeze(-1)


# ──────────────────────────────────────────────
# Weight packing for 1.58-bit (ternary → degraded to 4-bit)
# ──────────────────────────────────────────────

def quantize_weight_ternary_to_int4(
    w: Tensor,
    block_size: int = W2A8_BLOCK_SIZE,
    w_scale_factor: float = 2.0,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Quantize ternary (1.58-bit) weights to per-block INT4.

    The 1.58-bit weight uses clip method: scale = mean(|w|) * w_scale_factor.
    Values are ternarized to {-1, 0, 1} then packed as 4-bit for inference.

    Equivalent to ``weight_quant_uniform_symmetric_clip_per_block_int1_58``
    but repacked into INT4 format.

    Args:
        w: weight tensor, shape ``(out_dim, in_dim)``.
        block_size: group size (default 256 for W2A8).
        w_scale_factor: multiplier for mean abs value (default 2.0).
        epsilon: minimum scale to avoid division by zero.

    Returns:
        ``(qweight, qweight_scale)`` packed as INT4 format (compatible with
        the same dequantize_weight function).
    """
    out_dim, in_dim = w.shape
    w_blocks = w.view(out_dim, -1, block_size)

    # Per-block clip scale (mean-abs method)
    w_scale = w_blocks.abs().mean(dim=-1, keepdim=True).mul_(w_scale_factor).clamp(min=epsilon)

    # Ternarize to {-1, 0, 1}
    w_ternary = (w_blocks / w_scale).round().clamp(-1, 1)

    # Repack as INT4 (ternary values fit in 4-bit)
    qweight = pack_int4(w_ternary.view(out_dim, -1).to(torch.int8))
    qweight_scale = w_scale.squeeze(-1).contiguous().to(torch.bfloat16)

    return qweight, qweight_scale
