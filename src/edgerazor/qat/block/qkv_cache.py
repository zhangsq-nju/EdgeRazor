"""KV Cache Quantization via Cache Wrapper.

Instead of subclassing each model's Attention module to inject quantization
into forward(), we wrap the transformers Cache object. The quantization
happens inside cache.update() — the universal interception point where
key_states/value_states are stored.

This single ~50 line file replaces the 5 per-model qkv_cache_*.py files
(~700 lines) that each copied the full forward() from transformers.

Works with any model using the transformers Cache API:
Llama, Qwen3, Qwen3Moe, Olmoe, Qwen2_5Omni, Gemma, Mistral, Phi, etc.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache

from ..util.quant_config import QuantConfig


class QuantizedKVState:
    """Wrap a transformers Cache to apply KV quantization during update().

    The STE (Straight-Through Estimator) pattern:
        key_quant = quant_fn(key)
        key = key + (key_quant - key).detach()

    makes the forward pass use quantized values while gradients flow back
    through the original (unquantized) key/value computation path.

    Usage:
        cache = QuantizedKVState(DynamicCache(), quant_fn, kv_kwargs)
        output = model(input_ids, past_key_values=cache)
    """

    def __init__(
        self,
        cache: Cache,
        quant_fn: callable,
        kv_kwargs: dict[str, Any],
    ):
        if cache is None:
            raise TypeError("cache must be a transformers Cache instance, got None")
        if quant_fn is None:
            raise TypeError("quant_fn must be a callable, got None")

        self._cache = cache
        self._quant_fn = quant_fn
        self._kv_kwargs = kv_kwargs

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply STE quantization then delegate to wrapped cache."""
        key_quant = self._quant_fn(x=key_states, **self._kv_kwargs)
        value_quant = self._quant_fn(x=value_states, **self._kv_kwargs)

        key_states = key_states + (key_quant - key_states).detach()
        value_states = value_states + (value_quant - value_states).detach()

        return self._cache.update(key_states, value_states, layer_idx, cache_kwargs)

    def __getattr__(self, name: str):
        # _cache / _quant_fn / _kv_kwargs are our own attrs
        if name in ('_cache', '_quant_fn', '_kv_kwargs'):
            raise AttributeError(name)
        return getattr(self._cache, name)

    def __len__(self) -> int:
        return len(self._cache)


def create_quantized_kv_cache(
    config: QuantConfig,
) -> QuantizedKVState | None:
    """Create a QuantizedKVState from a QuantConfig if KV cache quant is configured.

    Returns None if kv_cache_function is not set.
    """
    kv_cache_function = config.function.kv_cache_function
    if not kv_cache_function or kv_cache_function == "":
        return None

    if isinstance(kv_cache_function, str):
        from ..map import quant_function_map

        if kv_cache_function not in quant_function_map:
            raise ValueError(
                f"Unknown kv_cache_function: {kv_cache_function}. "
                f"Available: {', '.join(quant_function_map.keys())}"
            )
        kv_cache_function = quant_function_map[kv_cache_function]

    kv_kwargs = {'epsilon': config.function.epsilon}
    if config.function.kv_block_size > 0:
        kv_kwargs['block_size'] = config.function.kv_block_size
    if config.function.kv_mixed_precision_prop > 0:
        kv_kwargs['mixed_precision_prop'] = config.function.kv_mixed_precision_prop

    return QuantizedKVState(
        cache=DynamicCache(),
        quant_fn=kv_cache_function,
        kv_kwargs=kv_kwargs,
    )
