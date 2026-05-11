"""
Decorators for quantization functions.

Provides reusable reshape preprocessing for per-block quantization,
eliminating ~100 lines of repeated reshape logic across 12 functions.
"""

import functools
import inspect

from torch import Tensor


def per_block_reshape(func):
    """Decorator that handles per-block reshape for quantization functions.

    Reshapes the first argument (tensor) to [..., -1, block_size] before
    calling the decorated function, and reshapes the result back to the
    original shape.

    The ``block_size`` parameter is resolved from the function's keyword
    arguments, falling back to the signature default.

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
    def wrapper(x: Tensor, *args, **kwargs):
        block_size = kwargs.get('block_size', block_size_default)

        original_shape = x.shape
        if block_size is not None and original_shape[-1] % block_size == 0:
            intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
        else:
            intermediate_shape = [-1, block_size]

        x_reshaped = x.view(intermediate_shape)
        result = func(x_reshaped, *args, **kwargs)
        return result.reshape(original_shape)

    return wrapper
