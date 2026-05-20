"""
Decorators for quantization functions.

Provides:
- per_block_reshape: reusable reshape pre/post-processing for per-block quantization
- mixed_precision_quantize: combined reshape + efficient masked quantization for
  mixed-precision functions, reducing each function body to just mask creation
- _apply_masked_quantize: helper that only quantizes each block to its target
  precision (vs the wasteful torch.where approach that computes both for all blocks)
"""

import functools
import inspect

import torch
from torch import Tensor

# ──────────────────────────────────────────────
# Mixed-precision quantization helper
# ──────────────────────────────────────────────

def _apply_masked_quantize(
    w_flat: Tensor,
    int4_mask: Tensor,
    epsilon: float,
    w_scale_factor: float,
    bits_int4: int = 4,
) -> Tensor:
    """Apply mixed-precision quantization only to the blocks that need each precision.

    Unlike the torch.where approach which computes both INT4 and INT1_58
    for all blocks, this only quantizes INT4 on masked blocks and INT1_58
    on the rest, halving the arithmetic.

    Args:
        w_flat: Weight blocks (num_blocks, block_size)
        int4_mask: 1D boolean mask (num_blocks,) — True → INT4, False → INT1_58
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for mean absolute value in INT1_58 scale
        bits_int4: Bit-width for INT4 quantization (default 4 → range [-7, 7])
    """
    max_val_int4 = 2 ** (bits_int4 - 1) - 1

    w_quant = torch.empty_like(w_flat)

    # INT4 blocks (absmax scale)
    if int4_mask.any():
        w_int4 = w_flat[int4_mask]
        scale_int4 = w_int4.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val_int4
        w_quant[int4_mask] = (w_int4 / scale_int4).round().clamp_(-max_val_int4, max_val_int4) * scale_int4

    # INT1_58 blocks (mean-abs scale)
    int1_58_mask = ~int4_mask
    if int1_58_mask.any():
        w_int1_58 = w_flat[int1_58_mask]
        scale_int1_58 = w_int1_58.abs().mean(dim=-1, keepdim=True).mul_(w_scale_factor).clamp_(min=epsilon)
        w_quant[int1_58_mask] = (w_int1_58 / scale_int1_58).round().clamp_(-1, 1) * scale_int1_58

    return w_quant


# ──────────────────────────────────────────────
# Decorators
# ──────────────────────────────────────────────

def per_block_reshape(func):
    """Decorator that handles per-block reshape for quantization functions.

    Reshapes the first argument (tensor) to (-1, block_size) before calling
    the decorated function, and reshapes the result back to the original shape.

    Three reshape strategies (in order):
    1. Last dimension divisible by block_size → preserve leading dims
    2. Total elements divisible by block_size → flatten to 2D
    3. Otherwise → zero-pad, reshape, then truncate after the function returns

    The ``block_size`` parameter is resolved from the function's keyword
    arguments, falling back to the signature default.

    .. note::
        ``block_size`` must be passed as a **keyword argument** (e.g.
        ``block_size=64``). Positional arguments are not inspected by the
        decorator and will be ignored, causing incorrect reshape behavior.

    Usage::

        @per_block_reshape
        def weight_quant_...(w, epsilon=1e-5, block_size=128):
            # w is already reshaped to [..., -1, block_size]
            ...
            return quantized_w  # reshaped back to original shape by decorator
    """
    sig = inspect.signature(func)
    block_size_param = sig.parameters.get('block_size')
    block_size_default = block_size_param.default if block_size_param is not None else None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract the tensor: first positional arg, or keyword 'w'/'x'
        if args:
            x = args[0]
            args = args[1:]
        elif 'w' in kwargs:
            x = kwargs.pop('w')
        elif 'x' in kwargs:
            x = kwargs.pop('x')
        else:
            raise TypeError(
                f"{func.__name__}() missing required tensor argument; "
                f"pass as first positional arg or keyword 'w'/'x'"
            )

        block_size = kwargs.get('block_size', block_size_default)

        original_shape = x.shape
        original_numel = x.numel()

        if block_size is not None and original_shape[-1] % block_size == 0:
            # Case 1: last dimension naturally divisible by block_size
            intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
            x_reshaped = x.view(intermediate_shape)
            need_truncate = False
        elif original_numel % block_size == 0:
            # Case 2: total elements divisible by block_size
            x_reshaped = x.reshape(-1, block_size)
            need_truncate = False
        else:
            # Case 3: zero-pad to make divisible, then truncate after
            remainder = original_numel % block_size
            pad_elements = block_size - remainder
            x_flat = x.contiguous().view(-1)
            x_padded = torch.nn.functional.pad(x_flat, (0, pad_elements), value=0)
            x_reshaped = x_padded.reshape(-1, block_size)
            need_truncate = True

        result = func(x_reshaped, *args, **kwargs)

        if need_truncate:
            return result.view(-1)[:original_numel].view(original_shape)
        return result.reshape(original_shape)

    return wrapper


def mixed_precision_quantize(func):
    """Decorator for mixed-precision quantization functions.

    Combines per-block reshape with efficient masked quantization:
    1. Reshapes input to [..., -1, block_size]
    2. Extracts epsilon, w_scale_factor from kwargs
    3. Calls decorated function → expects 1D int4_mask
    4. Flattens to 2D and applies _apply_masked_quantize
    5. Reshapes result back to original shape

    The decorated function only needs to create and return the mask::

        @mixed_precision_quantize
        def weight_quant_..._mp_...(w, epsilon=1e-5, w_scale_factor=2.0, block_size=128, ...):
            # w is already reshaped to [..., -1, block_size]
            with torch.no_grad():
                # … create int4_mask (1D boolean, num_blocks total) …
                return int4_mask
    """
    sig = inspect.signature(func)
    block_size_param = sig.parameters.get('block_size')
    block_size_default = block_size_param.default if block_size_param is not None else None
    eps_param = sig.parameters.get('epsilon')
    eps_default = eps_param.default if eps_param is not None else 1e-5
    wsf_param = sig.parameters.get('w_scale_factor')
    wsf_default = wsf_param.default if wsf_param is not None else 2.0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract the tensor: first positional arg, or keyword 'w'/'x'
        if args:
            x = args[0]
            args = args[1:]
        elif 'w' in kwargs:
            x = kwargs.pop('w')
        elif 'x' in kwargs:
            x = kwargs.pop('x')
        else:
            raise TypeError(
                f"{func.__name__}() missing required tensor argument; "
                f"pass as first positional arg or keyword 'w'/'x'"
            )

        block_size = kwargs.get('block_size', block_size_default)
        epsilon = kwargs.get('epsilon', eps_default)
        w_scale_factor = kwargs.get('w_scale_factor', wsf_default)

        original_shape = x.shape
        if block_size is not None and original_shape[-1] % block_size == 0:
            intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
        else:
            intermediate_shape = [-1, block_size]

        x_reshaped = x.view(intermediate_shape)

        # Decorated function returns 1D mask
        int4_mask = func(x_reshaped, *args, **kwargs)

        # Flatten to 2D and quantize
        w_flat = x_reshaped.reshape(-1, block_size)
        result = _apply_masked_quantize(w_flat, int4_mask, epsilon, w_scale_factor)

        return result.reshape(original_shape)

    return wrapper
