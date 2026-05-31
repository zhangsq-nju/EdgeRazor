"""
Pure quantization operations — no vLLM dependency.

EdgeRazor per-block weight quantization formats:

  W4  (INT4):  block_size=256, symmetric absmax, 2×INT4 per uint8
  W1.58 (W2):   block_size=256, clip (mean-abs), 4×INT2 per uint8

Reference quant functions (from edgerazor.qat.util.quant_function):
  - weight_quant_uniform_symmetric_absmax_per_block_int4
  - weight_quant_uniform_symmetric_clip_per_block_int1_58
  - state_quant_uniform_symmetric_absmax_per_token_int8
"""

import torch
from torch import Tensor

# ──────────────────────────────────────────────
# Constants (maps 1:1 with quant_function_config.py)
# ──────────────────────────────────────────────

INT1_58_MAX = 1           # ternary {-1, 0, 1}
INT4_MAX    = 7           # 2^(4-1) - 1
INT8_MAX    = 127         # 2^(8-1) - 1

# ── Block-size constants ──────────────────────────────────────────
# ER_*  = EdgeRazor training block size   (from quant_function_config.py)
# IE_*  = Inference Engine block size     (target packing granularity)
#
# When ER > IE, the scale is computed over ER elements then replicated
# IE/ER times — the weight-int values are identical to training, and
# the dequantized result is bit-exact regardless of IE granularity.
#
# When ER <= IE, quantization runs directly at IE granularity.

# EdgeRazor weight block sizes (used for training)
ER_W1_58A8_BLOCK_SIZE = 256
ER_W4A8_BLOCK_SIZE    = 256

# Inference engine weight block sizes (used for inference / packing)
IE_W1_58A8_BLOCK_SIZE = 256
IE_W4A8_BLOCK_SIZE    = 32

# ── Compat aliases (kept for external imports) ────────────────────
W1_58A8_BLOCK_SIZE = ER_W1_58A8_BLOCK_SIZE
W4A8_BLOCK_SIZE    = ER_W4A8_BLOCK_SIZE

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
# Pack / Unpack INT2 (W1.58-A8 ternary → 2-bit)
# ──────────────────────────────────────────────

def pack_w2(w_int: Tensor) -> Tensor:
    """Pack signed INT2 values { -2, -1, 0, 1 } → uint8 (4 values per byte).

    Each group of 4 adjacent INT2 values ``[a, b, c, d]`` along the last
    dimension is packed as::

        byte = (a+2) | ((b+2) << 2) | ((c+2) << 4) | ((d+2) << 6)

    The ``+2`` offset shifts [-2, 1] to [0, 3] for unsigned 2-bit storage.
    For ternary values { -1, 0, 1 }, the stored values are {1, 2, 3}.

    Args:
        w_int: signed INT8 tensor with values in [-2, 1], last dim is a
               multiple of 4.

    Returns:
        Packed uint8 tensor, last dim quartered.
    """
    w_shifted = (w_int + 2).to(torch.uint8)  # [-2,1] → [0,3]
    v0 = w_shifted[..., 0::4]
    v1 = w_shifted[..., 1::4]
    v2 = w_shifted[..., 2::4]
    v3 = w_shifted[..., 3::4]
    return v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)


def unpack_w2(qweight: Tensor) -> Tensor:
    """Unpack uint8 (..., block_size//4) → signed INT8 (..., block_size).

    Reverses :func:`pack_w2`.  Values are returned in [-2, 1].

    Args:
        qweight: packed uint8 tensor.

    Returns:
        Unpacked signed INT8 tensor with values in [-2, 1], last dim quadrupled.
    """
    v0 = (qweight & 0x03).to(torch.int8) - 2
    v1 = ((qweight >> 2) & 0x03).to(torch.int8) - 2
    v2 = ((qweight >> 4) & 0x03).to(torch.int8) - 2
    v3 = ((qweight >> 6) & 0x03).to(torch.int8) - 2
    return torch.stack([v0, v1, v2, v3], dim=-1).flatten(-2)


# ──────────────────────────────────────────────
# ER / IE block-size resolution
# ──────────────────────────────────────────────

def resolve_quant_block(er_block: int, ie_block: int) -> tuple[int, int, bool]:
    """Resolve training vs inference block sizes.

    Returns ``(quant_block, scale_block, needs_split)``:

    - ``quant_block``: block size used to compute weight-int values.
    - ``scale_block``: block size of the stored scale tensor.
    - ``needs_split``: whether scales must be replicated (ER > IE).

    Raises ``ValueError`` if *er_block* is not an integer multiple of
    *ie_block* (required for scale replication correctness).
    """
    if er_block <= ie_block:
        return ie_block, ie_block, False

    if er_block % ie_block != 0:
        raise ValueError(
            f"EdgeRazor training block_size ({er_block}) must be an "
            f"integer multiple of inference block_size ({ie_block})."
        )
    return er_block, ie_block, True


# ──────────────────────────────────────────────
# Per-block quantization / dequantization
# ──────────────────────────────────────────────

def quantize_weight_per_block_int4(
    w: Tensor,
    er_block_size: int = ER_W4A8_BLOCK_SIZE,
    ie_block_size: int = IE_W4A8_BLOCK_SIZE,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Quantize bf16/fp16 weight to per-block INT4 with ER/IE support.

    When *er_block_size* > *ie_block_size*, the INT4 values are computed
    over the larger ER block (matching training), then the resulting scale
    is replicated *ie_block_size* times for the packed storage layout.
    This guarantees dequantized weights are identical to training.

    Args:
        w: weight tensor, shape ``(out_dim, in_dim)``.
        er_block_size: EdgeRazor training block size (default 256).
        ie_block_size: inference engine packing block size (default 32).
        epsilon: minimum scale to avoid division by zero.

    Returns:
        ``(qweight, qweight_scale)`` where:
          - ``qweight``: packed uint8 ``(out_dim, in_dim // 2)``
          - ``qweight_scale``: bf16 ``(out_dim, in_dim // ie_block_size)``
    """
    out_dim, in_dim = w.shape
    quant_block, scale_block, needs_split = resolve_quant_block(
        er_block_size, ie_block_size,
    )

    w_blocks = w.view(out_dim, -1, quant_block)
    w_scale = w_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=epsilon) / INT4_MAX

    w_int = (w_blocks / w_scale).round().clamp(-INT4_MAX, INT4_MAX).to(torch.int8)
    qweight = pack_int4(w_int.view(out_dim, -1))

    if needs_split:
        n = er_block_size // ie_block_size
        w_scale = w_scale.squeeze(-1).repeat_interleave(n, dim=1)

    qweight_scale = w_scale.squeeze(-1).contiguous().to(torch.bfloat16)

    return qweight, qweight_scale


# ──────────────────────────────────────────────
# Dequantize weight dispatcher
# ──────────────────────────────────────────────

def dequantize_weight(
    qweight: Tensor,
    qweight_scale: Tensor,
    block_size: int = IE_W4A8_BLOCK_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
    weight_bits: int = 4,
) -> Tensor:
    """Dequantize per-block quantized weights to the target dtype.

    *block_size* must match the *ie_block_size* used during packing.

    Args:
        qweight: packed uint8.
        qweight_scale: bf16 scale tensor.
        block_size: inference scale block size.
        out_dtype: target dtype for dequantized weight.
        weight_bits: weight bit-width (4 for INT4, 1 for W1.58 ternary).

    Returns:
        Dequantized weight tensor ``(out_dim, in_dim)``.
    """
    if weight_bits == 4:
        w_int = unpack_int4(qweight)
    elif weight_bits == 1.58:
        w_int = unpack_w2(qweight)
    else:
        raise ValueError(f"Unsupported weight_bits={weight_bits}")
    scale = qweight_scale.repeat_interleave(block_size, dim=1)
    return (w_int.float() * scale.float()).to(out_dtype)


# ──────────────────────────────────────────────
# Weight quantization for 1.58-bit (ternary → W2 pack)
# ──────────────────────────────────────────────

def quantize_weight_per_block_w2(
    w: Tensor,
    er_block_size: int = ER_W1_58A8_BLOCK_SIZE,
    ie_block_size: int = IE_W1_58A8_BLOCK_SIZE,
    w_scale_factor: float = 2.0,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Quantize bf16/fp16 weight to ternary {-1, 0, 1}, packed as W2.

    Uses the clip method (mean-abs scale) from EdgeRazor training,
    equivalent to ``weight_quant_uniform_symmetric_clip_per_block_int1_58``.

    Args:
        w: weight tensor, shape ``(out_dim, in_dim)``.
        er_block_size: EdgeRazor training block size (default 256).
        ie_block_size: inference engine packing block size (default 32).
        w_scale_factor: multiplier for mean abs value (default 2.0).
        epsilon: minimum scale to avoid division by zero.

    Returns:
        ``(qweight, qweight_scale)`` where:
          - ``qweight``: packed uint8 ``(out_dim, in_dim // 4)``
          - ``qweight_scale``: bf16 ``(out_dim, in_dim // ie_block_size)``
    """
    out_dim, in_dim = w.shape
    quant_block, scale_block, needs_split = resolve_quant_block(
        er_block_size, ie_block_size,
    )

    w_blocks = w.view(out_dim, -1, quant_block)

    # Per-block clip scale (mean-abs method from EdgeRazor training)
    w_scale = (
        w_blocks.abs().mean(dim=-1, keepdim=True)
        .mul_(w_scale_factor).clamp(min=epsilon)
    )

    # Ternarize to {-1, 0, 1}
    w_ternary = (w_blocks / w_scale).round().clamp(-1, 1).to(torch.int8)
    qweight = pack_w2(w_ternary.view(out_dim, -1))

    if needs_split:
        n = er_block_size // ie_block_size
        w_scale = w_scale.squeeze(-1).repeat_interleave(n, dim=1)

    qweight_scale = w_scale.squeeze(-1).contiguous().to(torch.bfloat16)
    return qweight, qweight_scale


# ── Legacy wrapper (ternary → INT4 pack, for backwards compat) ────

def quantize_weight_ternary_to_int4(
    w: Tensor,
    block_size: int = W1_58A8_BLOCK_SIZE,
    w_scale_factor: float = 2.0,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Quantize ternary (1.58-bit) weights to per-block INT4 (legacy).

    Prefer :func:`quantize_weight_per_block_w2` for new code — it packs
    4 ternary values per byte instead of 2, halving memory usage.

    Returns ``(qweight, qweight_scale)`` in INT4-packed format, compatible
    with ``dequantize_weight(qweight, qweight_scale, weight_bits=4)``.
    """
    out_dim, in_dim = w.shape
    w_blocks = w.view(out_dim, -1, block_size)

    w_scale = (
        w_blocks.abs().mean(dim=-1, keepdim=True)
        .mul_(w_scale_factor).clamp(min=epsilon)
    )

    w_ternary = (w_blocks / w_scale).round().clamp(-1, 1)
    qweight = pack_int4(w_ternary.view(out_dim, -1).to(torch.int8))
    qweight_scale = w_scale.squeeze(-1).contiguous().to(torch.bfloat16)

    return qweight, qweight_scale


# ──────────────────────────────────────────────
# Activation quantization (per-token / per-block INT8)
# ──────────────────────────────────────────────

def quantize_activation_per_token_int8(
    x: Tensor,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Dynamically quantize activation to INT8 per-token.

    Quantizes along the last dimension (per-token). Equivalent to
    ``state_quant_uniform_symmetric_absmax_per_token_int8``.
    """
    x_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=epsilon) / INT8_MAX
    x_int = (x / x_scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return x_int, x_scale


def quantize_activation_per_block_int8(
    x: Tensor,
    block_size: int = W1_58A8_BLOCK_SIZE,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Dynamically quantize activation to INT8 per-block.

    Divides the last dimension into blocks and quantizes each separately.
    Equivalent to ``state_quant_uniform_symmetric_absmax_per_block_int8``.
    """
    shape = x.shape
    x_blocks = x.view(*shape[:-1], -1, block_size)
    x_scale = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=epsilon) / INT8_MAX
    x_int = (x_blocks / x_scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return x_int.view(shape), x_scale.squeeze(-1)
