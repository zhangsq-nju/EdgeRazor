r"""
# Quantization Functions

## Function Naming Convention

### Weight Quantization
For example, `weight_quant_uniform_symmetric_clip_per_block_int1_58`:

- `weight_quant`: specifies that this function is for weight quantization.
- `uniform`: specifies that this function is uniform quantization.
- `symmetric_clip`: specifies the quantization method used (symmetric clipping).
- `per_block`: specifies that the quantization is performed per-block.
- `int1_58`: specifies the bit-width for corresponding values.

### State Quantization (Activation & KV Cache)
For example, `state_quant_uniform_symmetric_absmax_per_token_int8`:

- `state_quant`: specifies that this function is for state quantization (activation/kv_cache).
- `uniform`: specifies that this function is uniform quantization.
- `symmetric_absmax`: specifies the quantization method used (symmetric absmax).
- `per_token`: specifies that the quantization is performed per-token (last dimension).
- `int8`: specifies the bit-width for quantized values.

"""
import torch
from torch import Tensor

from .quant_function_config import (
    mixed_precision_prop,
    w2a8_block_size,
    w4a8_block_size,
    w5a8_block_size,
    w8a8_block_size,
)
from .quant_function_decorator import mixed_precision_quantize, per_block_reshape

# =============================================================
# INT1_58 (Ternary) Weight Quantization - Clip Method
# =============================================================


def weight_quant_uniform_symmetric_clip_per_tensor_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-tensor using clip method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed as mean(|w|) * w_scale_factor.
    
    Args:
        w: Weight tensor to quantize
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Calculate the scale factor: mean(|w|) * w_scale_factor
        w_scale = w.abs().mean().mul_(w_scale_factor).clamp_(min=epsilon)

        # Quantize to INT1_58: {-1, 0, 1}
        w_quantized = w.div(w_scale).round_().clamp_(-1, 1)

        return w_quantized * w_scale


def weight_quant_uniform_symmetric_clip_per_channel_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-channel using clip method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per output channel.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Compute scale factor for each output channel
        # Shape: (out_dim, 1)
        w_scale = w.abs().mean(dim=-1, keepdim=True).mul_(w_scale_factor).clamp_(min=epsilon)

        # Quantize to INT1_58: {-1, 0, 1}
        w_quantized = w.div(w_scale).round_().clamp_(-1, 1)

    return w_quantized * w_scale


@per_block_reshape
def weight_quant_uniform_symmetric_clip_per_block_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-block using clip method.

    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per block within each output channel.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block

    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Compute scale factor for each block
        # Shape: (out_dim, block_num, 1)
        w_scale = w.abs().mean(dim=-1, keepdim=True).mul_(w_scale_factor).clamp_(min=epsilon)

        # Quantize to INT1_58: {-1, 0, 1}
        w_quant = w.div(w_scale).round_().clamp_(-1, 1)

        w_quant = w_quant * w_scale

    return w_quant


@mixed_precision_quantize
def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using dynamic mixed precision.

    Dynamic Strategy (Variance-based Block Selection):
    - Divides weight into blocks and calculates variance for each block
    - Dynamically selects top mixed_precision_prop (e.g., 1%) blocks with highest variance
    - High-variance blocks (important weights) are quantized to INT4 using absmax method
    - Remaining blocks are quantized to INT1_58 using clip method

    Example:
    - With mixed_precision_prop=0.01, the top 1% most variable blocks use INT4: [-7, 7]
    - The remaining 99% less variable blocks use INT1_58: {-1, 0, 1}

    This approach adapts to weight distribution, allocating higher precision to
    more important blocks based on their variance.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT4)

    Returns:
        1D boolean mask (num_blocks,) — True for INT4 blocks, False for INT1_58
    """
    with torch.no_grad():
        # w is pre-reshaped by decorator to [..., -1, block_size]; flatten to 2D
        num_blocks = w.numel() // block_size
        w_flat = w.view(num_blocks, block_size)

        # Calculate importance score for each block (using variance as metric)
        block_variance = w_flat.var(dim=-1)

        # Determine how many blocks should use INT4 (higher precision)
        num_int4_blocks = int(num_blocks * mixed_precision_prop)

        # Create mask for INT4 blocks
        int4_mask = torch.zeros_like(block_variance, dtype=torch.bool)
        if num_int4_blocks > 0:
            # Select top-k blocks with highest variance for INT4 quantization
            _, top_indices = torch.topk(block_variance, k=num_int4_blocks, dim=-1)
            int4_mask.scatter_(dim=-1, index=top_indices, value=True)
        # else: mask stays all False → all blocks quantized to INT1_58

    return int4_mask


@mixed_precision_quantize
def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using static mixed precision.

    Static Strategy (Position-based Block Selection):
    - Divides weight into blocks in sequential (row-major) order
    - Statically assigns the first mixed_precision_prop (e.g., 1%) blocks to INT4
    - First blocks (by position) are quantized to INT4 using absmax method
    - Remaining blocks are quantized to INT1_58 using clip method

    Example:
    - With mixed_precision_prop=0.01, the first 1% blocks use INT4: [-7, 7]
    - The remaining 99% blocks use INT1_58: {-1, 0, 1}
    - With mixed_precision_prop=0.0, all blocks use INT1_58

    This approach uses a fixed pattern without analyzing weight importance,
    assuming that earlier blocks in the tensor are more critical.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT4)

    Returns:
        1D boolean mask (num_blocks,) — True for INT4 blocks, False for INT1_58
    """
    with torch.no_grad():
        # w is pre-reshaped by decorator to [..., -1, block_size]; flatten to 2D
        num_blocks = w.numel() // block_size

        # First num_int4_blocks (row-major order) get INT4 precision
        num_int4_blocks = int(num_blocks * mixed_precision_prop)

        int4_mask = torch.zeros(num_blocks, dtype=torch.bool, device=w.device)
        if num_int4_blocks > 0:
            int4_mask[:num_int4_blocks] = True
        # else: mask stays all False → all blocks quantized to INT1_58

    return int4_mask


@mixed_precision_quantize
def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_column_wise(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using static column-wise mixed precision.

    Column-wise Static Strategy:
    - Reshapes weight (out_dim, in_dim) to (out_dim, blocks_per_channel, block_size)
    - For a weight of shape (1024, 1024) with block_size=128:
      - blocks_per_channel = 1024 / 128 = 8
      - Reshaped to (1024, 8, 128), treating as (1024, 8) quantization blocks
    - Column-wise means quantizing along the column dimension (blocks_per_channel axis)
    - Each column (all 1024 rows at the same block position) shares the same precision
    - The first few columns (block positions) use INT4, remaining columns use INT1.58

    Difference from `_static` version:
    - `_static`: Assigns first N blocks globally in row-major order
    - `_static_column_wise`: First N columns (block positions) across ALL rows use INT4

    Example (out_dim=1024, in_dim=1024, block_size=128, mixed_precision_prop=0.125):
    - blocks_per_channel = 8, num_int4_blocks = floor(8 * 0.125) = 1
    - Column 0 (blocks at position 0 for all 1024 rows) uses INT4
    - Columns 1-7 use INT1.58
    - Total INT4 blocks: 1024 × 1 = 1024, INT1.58 blocks: 1024 × 7 = 7168
    - With mixed_precision_prop=0.0, all columns use INT1_58

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of columns (block positions) to quantize to INT4

    Returns:
        1D boolean mask (out_dim * blocks_per_channel,) — True for INT4, False for INT1_58
    """
    with torch.no_grad():
        # w is pre-reshaped by decorator to (out_dim, blocks_per_channel, block_size)
        out_dim = w.shape[0]
        blocks_per_channel = w.shape[1]

        # Determine how many blocks per channel should use INT4
        num_int4_blocks_per_channel = int(blocks_per_channel * mixed_precision_prop)

        # Create per-column mask, then broadcast to all rows
        block_indices = torch.arange(blocks_per_channel, device=w.device)
        col_mask = block_indices < num_int4_blocks_per_channel  # (blocks_per_channel,)
        int4_mask = col_mask.unsqueeze(0).expand(out_dim, -1).reshape(-1)

    return int4_mask


@mixed_precision_quantize
def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using static row-wise sparse mixed precision.

    Row-wise Sparse Static Strategy:
    - Reshapes weight (out_dim, in_dim) to (out_dim, blocks_per_channel, block_size)
    - For a weight of shape (1024, 1024) with block_size=128:
      - blocks_per_channel = 1024 / 128 = 8
      - Reshaped to (1024, 8, 128), treating as (1024, 8) quantization blocks
    - Row-wise means quantizing along the row dimension (out_dim axis)
    - Every n-th row uses INT4 for ALL its blocks (entire row), where n = 1/prop
    - Remaining rows use INT1.58 for ALL their blocks

    Difference from `_static_column_wise`:
    - `_static_column_wise`: First N columns use INT4 (vertical allocation)
    - `_static_row_wise_sparse`: Every n-th row uses INT4 (horizontal sparse allocation)

    Example (out_dim=1024, in_dim=1024, block_size=128, mixed_precision_prop=0.01):
    - blocks_per_channel = 8, n = 1/0.01 = 100
    - Rows 0, 100, 200, ..., 900 use INT4 for all 8 blocks (10 rows total)
    - Other rows use INT1.58 for all 8 blocks (1014 rows)
    - Total INT4 blocks: 10 × 8 = 80, INT1.58 blocks: 1014 × 8 = 8112
    - Actual INT4 proportion: 80 / 8192 ≈ 0.98%
    - With mixed_precision_prop=0.0, all blocks use INT1_58

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion determines row spacing (n = 1/prop)

    Returns:
        1D boolean mask (out_dim * blocks_per_channel,) — True for INT4, False for INT1_58
    """
    with torch.no_grad():
        # w is pre-reshaped by decorator to (out_dim, blocks_per_channel, block_size)
        out_dim = w.shape[0]
        blocks_per_channel = w.shape[1]

        if mixed_precision_prop <= 0:
            # No rows get INT4 — all blocks are INT1_58
            return torch.zeros(out_dim * blocks_per_channel, dtype=torch.bool, device=w.device)

        # Every n-th row gets INT4 for all its blocks, where n = 1/prop
        row_spacing = max(1, int(1.0 / mixed_precision_prop))
        row_indices = torch.arange(out_dim, device=w.device)
        row_mask = row_indices % row_spacing == 0  # (out_dim,)
        int4_mask = row_mask.unsqueeze(1).expand(out_dim, blocks_per_channel).reshape(-1)

    return int4_mask


@mixed_precision_quantize
def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using sparse static mixed precision.

    Sparse Static Strategy (Interspersed Block Selection):
    - Divides weight into sequential blocks.
    - Within every 100-block chunk, the first `100 * mixed_precision_prop` blocks
      are statically assigned to INT4 precision.
    - This creates a sparse, regular pattern of higher-precision blocks.

    Example:
    - With `mixed_precision_prop=0.01`, the 1st block of every 100 is INT4,
      while blocks 2-100 are INT1_58.
    - Block indices 0, 100, 200, ... get INT4.
    - Block indices 1-99, 101-199, ... get INT1_58.
    - With `mixed_precision_prop=0.0`, all blocks use INT1_58.

    This differs from the standard static method by distributing INT4 blocks
    evenly throughout the tensor rather than concentrating them at the start.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT4)

    Returns:
        1D boolean mask (num_blocks,) — True for INT4 blocks, False for INT1_58
    """
    with torch.no_grad():
        num_blocks = w.numel() // block_size

        # --- Create the sparse mask for INT4 blocks ---
        group_size = 100
        num_int4_per_group = int(group_size * mixed_precision_prop)

        # Create indices from 0 to num_blocks - 1
        indices = torch.arange(num_blocks, device=w.device)

        # Mask is True for the first `num_int4_per_group` blocks in every `group_size` chunk
        int4_mask = (indices % group_size) < num_int4_per_group

    return int4_mask


# =============================================================
# INT1_58 (Ternary) Weight Quantization - Absmax Method
# =============================================================


def weight_quant_uniform_symmetric_absmax_per_tensor_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-tensor using absmax method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed as max(|w|).
    
    Args:
        w: Weight tensor to quantize
        epsilon: Small value to prevent division by zero
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Calculate the scale factor using maximum absolute value
        w_scale = w.abs().max().clamp_(min=epsilon)
        
        # Quantize to INT1_58: {-1, 0, 1}
        w_quantized = w.div(w_scale).round_().clamp_(-1, 1)
        
        return w_quantized * w_scale


def weight_quant_uniform_symmetric_absmax_per_channel_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-channel using absmax method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per output channel using max(|w|).
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Compute scale factor for each output channel using maximum absolute value
        # Shape: (out_dim, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon)
        
        # Quantize to INT1_58: {-1, 0, 1}
        w_quantized = w.div(w_scale).round_().clamp_(-1, 1)
    
    return w_quantized * w_scale


@per_block_reshape
def weight_quant_uniform_symmetric_absmax_per_block_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-block using absmax method.

    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per block using max(|w|).

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Compute scale factor for each block using maximum absolute value
        # Shape: (out_dim, block_num, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon)

        # Quantize to INT1_58: {-1, 0, 1}
        w_quant = w.div(w_scale).round_().clamp_(-1, 1)

        w_quant = w_quant * w_scale

    return w_quant


# =============================================================
# INT4 Weight Quantization - Absmax Method
# =============================================================


def weight_quant_uniform_symmetric_absmax_per_tensor_int4(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT4 per-tensor using absmax method.
    
    Quantizes weight to INT4: [-7, 7] * w_scale.
    Scale factor is computed as max(|w|) / 7.
    
    Args:
        w: Weight tensor to quantize
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized weight tensor with values in [-7, 7] * w_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Calculate the scale factor using maximum absolute value
        w_scale = w.abs().max().clamp_(min=epsilon) / max_val

        # Quantize to INT4: [-7, 7]
        w_quantized = w.div(w_scale).round_().clamp_(-max_val, max_val)

        return w_quantized * w_scale


def weight_quant_uniform_symmetric_absmax_per_channel_int4(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT4 per-channel using absmax method.
    
    Quantizes weight to INT4: [-7, 7] * w_scale.
    Scale factor is computed per output channel.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized weight tensor with values in [-7, 7] * w_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Compute scale factor for each output channel
        # Shape: (out_dim, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

        # Quantize to INT4: [-7, 7]
        w_quantized = w.div(w_scale).round_().clamp_(-max_val, max_val)

    return w_quantized * w_scale


@per_block_reshape
def weight_quant_uniform_symmetric_absmax_per_block_int4(
    w: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT4 per-block using absmax method.

    Quantizes weight to INT4: [-7, 7] * w_scale.
    Scale factor is computed per block within each output channel.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized weight tensor with values in [-7, 7] * w_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Compute scale factor for each block
        # Shape: (out_dim, block_num, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

        # Quantize to INT4: [-7, 7]
        w_quant = w.div(w_scale).round_().clamp_(-max_val, max_val)

        w_quant = w_quant * w_scale

    return w_quant


@per_block_reshape
def weight_quant_uniform_symmetric_absmax_per_block_int5(
    w: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w5a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT5 per-block using absmax method.

    Quantizes weight to INT5: [-15, 15] * w_scale.
    Scale factor is computed per block within each output channel.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized weight tensor with values in [-15, 15] * w_scale
    """
    bits = 5
    max_val = 2**(bits - 1) - 1  # 15 for INT5

    with torch.no_grad():
        # Compute scale factor for each block
        # Shape: (out_dim, block_num, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

        # Quantize to INT5: [-15, 15]
        w_quant = w.div(w_scale).round_().clamp_(-max_val, max_val)

        w_quant = w_quant * w_scale

    return w_quant


@per_block_reshape
def weight_quant_uniform_symmetric_absmax_per_block_int8(
    w: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w8a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT8 per-block using absmax method.

    Quantizes weight to INT8: [-127, 127] * w_scale.
    Scale factor is computed per block within each output channel.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized weight tensor with values in [-127, 127] * w_scale
    """
    bits = 8
    max_val = 2**(bits - 1) - 1  # 127 for INT8

    with torch.no_grad():
        # Compute scale factor for each block
        # Shape: (out_dim, block_num, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

        # Quantize to INT8: [-127, 127]
        w_quant = w.div(w_scale).round_().clamp_(-max_val, max_val)

        w_quant = w_quant * w_scale

    return w_quant


def weight_quant_uniform_asymmetric_max_per_tensor_int4(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT4 per-tensor using asymmetric max method.

    Quantizes weight to INT4: {-8, -7, ..., 6, 7} * w_scale.
    Scale = max(w) / 7, ensuring the max value is exactly representable
    at reference level q=7.

    Representable range: [-8/7 * max(w), max(w)] — covers symmetric
    distributions fully since 8/7 > 1.

    For all-negative weights (max(w) < 0), falls back to absmax / 7.

    Args:
        w: Weight tensor to quantize
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized weight tensor with values in {-8, ..., 7} * w_scale
    """
    bits = 4
    ref_max = 2 ** (bits - 1) - 1   # 7, maps to w_max
    zero_point = 2 ** (bits - 1)     # 8, maps to 0

    with torch.no_grad():
        w_max = w.max()
        # Positive scale: w_max lands at q=7 (storage 15), giving
        # representable range [-8/7*w_max, w_max].
        w_scale = w_max / ref_max

        # All-negative weights: w_max < 0 → w_scale < 0 would invert
        # the grid. Fall back to absmax to keep range well-formed.
        if w_scale < 0:
            w_scale = w.abs().max() / ref_max

        w_scale = w_scale.clamp_(min=epsilon)

        # q = round(w / scale + zp)  →  storage ∈ [0, 15]
        # ŵ = (q - zp) * scale       →  reference ∈ {-8, ..., 7} * scale
        q_storage = (w / w_scale + zero_point).round().clamp(0, 2**bits - 1)
        w_quant = (q_storage - zero_point) * w_scale

    return w_quant


def weight_quant_uniform_asymmetric_max_per_channel_int4(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT4 per-channel using asymmetric max method.

    Quantizes weight to INT4: {-8, -7, ..., 6, 7} * w_scale.
    Scale = max(w, dim=-1) / 7 per output channel.

    Representable range: [-8/7 * max(w), max(w)] — covers symmetric
    distributions fully since 8/7 > 1.

    For channels where max(w) < 0, falls back to absmax / 7.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized weight tensor with values in {-8, ..., 7} * w_scale
    """
    bits = 4
    ref_max = 2 ** (bits - 1) - 1   # 7
    zero_point = 2 ** (bits - 1)     # 8

    with torch.no_grad():
        # Shape: (out_dim, 1)
        w_max = w.max(dim=-1, keepdim=True).values
        w_scale = w_max / ref_max

        # Fallback for channels with negative max
        neg_mask = w_scale < 0
        if neg_mask.any():
            fallback = w[neg_mask.expand_as(w)].view(neg_mask.sum(), -1).abs().max(dim=-1, keepdim=True).values / ref_max
            w_scale = torch.where(neg_mask, fallback, w_scale)

        w_scale = w_scale.clamp_(min=epsilon)

        q_storage = (w / w_scale + zero_point).round().clamp(0, 2**bits - 1)
        w_quant = (q_storage - zero_point) * w_scale

    return w_quant


@per_block_reshape
def weight_quant_uniform_asymmetric_max_per_block_int4(
    w: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w4a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT4 per-block using asymmetric max method.

    Quantizes weight to INT4: {-8, -7, ..., 6, 7} * w_scale.
    Scale = max(w, dim=-1) / 7 per block.

    Representable range: [-8/7 * max(w), max(w)] — covers symmetric
    distributions fully since 8/7 > 1.

    For blocks where max(w) < 0, falls back to absmax / 7.

    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block (default 32 for Q4_0)

    Returns:
        Quantized weight tensor with values in {-8, ..., 7} * w_scale
    """
    bits = 4
    ref_max = 2 ** (bits - 1) - 1   # 7
    zero_point = 2 ** (bits - 1)     # 8

    with torch.no_grad():
        # Shape after per_block_reshape: (out_dim, block_num, block_size)
        w_max = w.max(dim=-1, keepdim=True).values
        w_scale = w_max / ref_max

        # Fallback for blocks with negative max
        neg_mask = w_scale < 0
        if neg_mask.any():
            fallback = w[neg_mask.expand_as(w)].view(neg_mask.sum(), -1).abs().max(dim=-1, keepdim=True).values / ref_max
            w_scale = torch.where(neg_mask, fallback, w_scale)

        w_scale = w_scale.clamp_(min=epsilon)

        q_storage = (w / w_scale + zero_point).round().clamp(0, 2**bits - 1)
        w_quant = (q_storage - zero_point) * w_scale

    return w_quant


# =============================================================
# INT2 State Quantization - Absmax Method
# =============================================================


def state_quant_uniform_symmetric_absmax_per_token_int2(
    x: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT2 per-token using absmax method.
    
    Quantizes along the last dimension (per-token/per-feature).
    Assumes input was normalized before quantization (e.g., LayerNorm, RMSNorm).
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized tensor with values in [-1, 1] * x_scale
    """
    bits = 2
    max_val = 2**(bits - 1) - 1  # 1 for INT2

    with torch.no_grad():
        # Compute scale factor for each token (last dimension)
        # Shape: ([batch,] seq_len, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT2: [-1, 1]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        
        return x_quant * x_scale


@per_block_reshape
def state_quant_uniform_symmetric_absmax_per_block_int2(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT2 per-block using absmax method.

    Applies group quantization by dividing features into blocks.

    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized tensor with values in [-1, 1] * x_scale
    """
    bits = 2
    max_val = 2**(bits - 1) - 1  # 1 for INT2

    with torch.no_grad():
        # Compute scale factor for each block
        # Shape: ([batch,] seq_len, block_num, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

        # Quantize to INT2: [-1, 1]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        x_quant = x_quant * x_scale

        return x_quant


# =============================================================
# INT4 State Quantization - Absmax Method
# =============================================================


def state_quant_uniform_symmetric_absmax_per_token_int4(
    x: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT4 per-token using absmax method.
    
    Quantizes along the last dimension (per-token/per-feature).
    Assumes input was normalized before quantization (e.g., LayerNorm, RMSNorm).
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized tensor with values in [-7, 7] * x_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Compute scale factor for each token (last dimension)
        # Shape: ([batch,] seq_len, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT4: [-7, 7]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        
        return x_quant * x_scale


@per_block_reshape
def state_quant_uniform_symmetric_absmax_per_block_int4(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT4 per-block using absmax method.

    Applies group quantization by dividing features into blocks.

    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized tensor with values in [-7, 7] * x_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Compute scale factor for each block
        # Shape: ([batch,] seq_len, block_num, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

        # Quantize to INT4: [-7, 7]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        x_quant = x_quant * x_scale

        return x_quant


@per_block_reshape
def state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w4a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT4 and INT8 per-block using dynamic mixed precision.

    Dynamic Strategy (Variance-based Block Selection):
    - Divides activation into blocks and calculates variance for each block
    - Dynamically selects top mixed_precision_prop (e.g., 1%) blocks with highest variance
    - High-variance blocks (important activations) are quantized to INT8 using absmax method
    - Remaining blocks are quantized to INT4 using absmax method

    Example:
    - With mixed_precision_prop=0.01, the top 1% most variable blocks use INT8: [-127, 127]
    - The remaining 99% less variable blocks use INT4: [-7, 7]

    This approach adapts to activation distribution, allocating higher precision to
    more important blocks based on their variance.

    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT8)

    Returns:
        Quantized tensor with dynamic mixed precision quantization
    """
    bits_int4 = 4
    max_val_int4 = 2**(bits_int4 - 1) - 1  # 7 for INT4
    bits_int8 = 8
    max_val_int8 = 2**(bits_int8 - 1) - 1  # 127 for INT8

    with torch.no_grad():
        # Flatten pre-shaped tensor to (num_blocks, block_size)
        num_blocks = x.numel() // block_size
        x_flat = x.reshape(num_blocks, block_size)

        # Calculate importance score for each block (using variance as metric)
        # Shape: (num_blocks,)
        block_variance = x_flat.var(dim=-1)

        # Determine how many blocks should use INT8 (higher precision)
        num_int8_blocks = min(max(1, int(num_blocks * mixed_precision_prop)), num_blocks)

        # Select top-k blocks with highest variance for INT8 quantization
        # Shape: (num_int8_blocks,)
        _, top_indices = torch.topk(block_variance, k=num_int8_blocks, dim=-1)

        # Create mask for INT8 blocks
        # Shape: (num_blocks,)
        int8_mask = torch.zeros_like(block_variance, dtype=torch.bool)
        int8_mask.scatter_(dim=-1, index=top_indices, value=True)

        # Expand mask to match block dimension
        # Shape: (num_blocks, 1)
        int8_mask_expanded = int8_mask.unsqueeze(-1)

        # ===== INT8 Quantization (absmax method) for selected blocks =====
        # Compute scale factor for INT8 blocks using maximum absolute value
        x_scale_int8 = x_flat.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val_int8
        # Quantize to INT8: [-127, 127]
        x_quant_int8 = (x_flat / x_scale_int8).round().clamp(-max_val_int8, max_val_int8) * x_scale_int8

        # ===== INT4 Quantization (absmax method) for remaining blocks =====
        # Compute scale factor for INT4 blocks using maximum absolute value
        x_scale_int4 = x_flat.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val_int4
        # Quantize to INT4: [-7, 7]
        x_quant_int4 = (x_flat / x_scale_int4).round().clamp(-max_val_int4, max_val_int4) * x_scale_int4

        # ===== Combine INT8 and INT4 blocks =====
        # Use INT8 for high-variance blocks, INT4 for others
        x_quant_flat = torch.where(int8_mask_expanded, x_quant_int8, x_quant_int4)

    return x_quant_flat


# =============================================================
# INT8 State Quantization - Absmax Method
# =============================================================


def state_quant_uniform_symmetric_absmax_per_token_int8(
    x: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT8 per-token using absmax method.
    
    Quantizes along the last dimension (per-token/per-feature).
    Assumes input was normalized before quantization (e.g., LayerNorm, RMSNorm).
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized tensor with values in [-127, 127] * x_scale
    """
    bits = 8
    max_val = 2**(bits - 1) - 1  # 127 for INT8

    with torch.no_grad():
        # Compute scale factor for each token (last dimension)
        # Shape: ([batch,] seq_len, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT8: [-127, 127]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        
        return x_quant * x_scale


@per_block_reshape
def state_quant_uniform_symmetric_absmax_per_block_int8(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT8 per-block using absmax method.

    Applies group quantization by dividing features into blocks.

    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized tensor with values in [-127, 127] * x_scale
    """
    bits = 8
    max_val = 2**(bits - 1) - 1  # 127 for INT8

    with torch.no_grad():
        # Compute scale factor for each block
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

        # Quantize to INT8: [-127, 127]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        x_quant = x_quant * x_scale

        return x_quant
