"""Unit tests for QuantizedKVState — the generic KV Cache quantization wrapper.

QuantizedKVState replaces all per-model QKVCache*Attention subclasses by
wrapping a transformers Cache and applying STE quantization during update().

Key/value shape convention follows transformers: (batch, num_heads, seq_len, head_dim).
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from edgerazor.qat.block.qkv_cache import QuantizedKVState

# Standard transformers KV shape: (batch, num_heads, seq_len, head_dim)
B, H, S, D = 2, 8, 4, 64


def _kv(seq_len=S, requires_grad=False):
    """Create test key/value tensors in (B, H, S, D) format."""
    return torch.randn(B, H, seq_len, D, requires_grad=requires_grad)


def _dummy_ternary_quant(x: torch.Tensor, epsilon: float = 1e-5,
                         block_size: int = 64) -> torch.Tensor:
    """Simple ternary quantization: {-1, 0, 1} * scale."""
    scale = x.abs().max() / 2.0
    q = torch.zeros_like(x)
    q[x > scale * 0.25] = scale
    q[x < -scale * 0.25] = -scale
    return q


# ──────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────

class TestQuantizedKVStateConstruction:
    def test_wrap_dynamic_cache(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        assert wrapped._cache is cache
        assert wrapped._quant_fn is _dummy_ternary_quant

    def test_wrap_requires_cache(self):
        with pytest.raises(TypeError):
            QuantizedKVState(None, quant_fn=_dummy_ternary_quant,
                             kv_kwargs={})  # type: ignore[arg-type]

    def test_wrap_requires_quant_fn(self):
        cache = DynamicCache()
        with pytest.raises(TypeError):
            QuantizedKVState(cache, quant_fn=None, kv_kwargs={})  # type: ignore[arg-type]


# ──────────────────────────────────────────────
# Attribute delegation
# ──────────────────────────────────────────────

class TestQuantizedKVStateDelegation:
    def test_delegates_get_seq_length(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        assert wrapped.get_seq_length() == 0

    def test_delegates_get_max_cache_shape(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        # DynamicCache returns -1 when not initialized (no layers cached)
        assert wrapped.get_max_cache_shape() == -1

    def test_delegates_reset(self):
        cache = DynamicCache()
        cache.update(_kv(), _kv(), 0)
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        # reset() delegates to underlying cache (no-op on DynamicCache)
        wrapped.reset()
        assert wrapped.get_seq_length(0) == S  # DynamicCache.reset is a no-op

    def test_delegates_crop(self):
        cache = DynamicCache()
        cache.update(_kv(S), _kv(S), 0)
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        wrapped.crop(2)
        assert wrapped.get_seq_length(0) == 2

    def test_len_delegation(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        assert len(wrapped) == 0

    def test_delegates_unknown_attr(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        with pytest.raises(AttributeError):
            wrapped.nonexistent_method()


# ──────────────────────────────────────────────
# update() — correct shape propagation
# ──────────────────────────────────────────────

class TestQuantizedKVStateUpdate:
    def test_update_returns_correct_shapes(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        k, v = _kv(), _kv()
        k_out, v_out = wrapped.update(k, v, 0)
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_update_stores_quantized_values(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        k, v = _kv(), _kv()
        k_out, v_out = wrapped.update(k, v, 0)
        assert not torch.equal(k_out, k)
        assert not torch.equal(v_out, v)

    def test_update_with_layer_idx(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        wrapped.update(_kv(), _kv(), layer_idx=0)
        wrapped.update(_kv(), _kv(), layer_idx=1)

        assert wrapped.get_seq_length(0) == S
        assert wrapped.get_seq_length(1) == S

    def test_update_passes_cache_kwargs(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        k, v = _kv(), _kv()
        k_out, v_out = wrapped.update(k, v, 0, {
            'sin': torch.randn(B, S, D),
            'cos': torch.randn(B, S, D),
            'cache_position': torch.arange(S),
        })
        assert k_out.shape == k.shape


# ──────────────────────────────────────────────
# STE gradient flow
# ──────────────────────────────────────────────

class TestQuantizedKVStateSTE:
    def test_ste_forward_uses_quantized_value(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        k, v = _kv(), _kv()
        with torch.no_grad():
            k_out, v_out = wrapped.update(k, v, 0)
        assert not torch.equal(k_out, k)

    def test_ste_backward_flow_to_original(self):
        k = _kv(requires_grad=True)
        v = _kv(requires_grad=True)

        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        k_out, v_out = wrapped.update(k, v, 0)

        loss = k_out.sum() + v_out.sum()
        loss.backward()

        assert k.grad is not None
        assert v.grad is not None
        assert k.grad.abs().sum() > 0
        assert v.grad.abs().sum() > 0

    def test_ste_params_update(self):
        k = _kv(requires_grad=True)
        k_before = k.clone()

        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )

        optimizer = torch.optim.SGD([k], lr=0.1)
        for _ in range(3):
            optimizer.zero_grad()
            k_out, _ = wrapped.update(k, torch.zeros_like(k), 0)
            loss = k_out.sum()
            loss.backward()
            optimizer.step()

        assert not torch.equal(k, k_before)


# ──────────────────────────────────────────────
# Multiple layers
# ──────────────────────────────────────────────

class TestQuantizedKVStateMultiLayer:
    def test_multiple_layer_updates(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        for layer_idx in range(4):
            k = _kv(requires_grad=True)
            v = _kv(requires_grad=True)
            k_out, v_out = wrapped.update(k, v, layer_idx)
            assert k_out.shape == k.shape
        assert len(wrapped) == 4

    def test_accumulate_across_steps(self):
        """Simulate autoregressive generation: accumulate kv across steps."""
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        for step in range(5):
            k = torch.randn(B, H, 1, D)  # 1 token per step
            v = torch.randn(B, H, 1, D)
            k_out, v_out = wrapped.update(k, v, 0)
            assert k_out.shape == (B, H, step + 1, D)
            assert v_out.shape == (B, H, step + 1, D)

    def test_quantized_cache_correctly_retrieved(self):
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_dummy_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        k, v = _kv(), _kv()
        k_out, v_out = wrapped.update(k, v, 0)
        assert not torch.equal(k_out, k)
        assert not torch.equal(v_out, v)


# ──────────────────────────────────────────────
# Quantization function integration
# ──────────────────────────────────────────────

class TestQuantizedKVStateQuantFn:
    def test_per_block_quant_integration(self):
        """Test that per-block quantization function works through wrapper."""
        from edgerazor.qat.util.quant_function import (
            per_block_reshape,
            weight_quant_uniform_symmetric_clip_per_block_int1_58,
        )

        @per_block_reshape
        def per_block_quant(w, epsilon, block_size):
            return weight_quant_uniform_symmetric_clip_per_block_int1_58(
                w, epsilon=epsilon, w_scale_factor=2.0, block_size=block_size
            )

        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=per_block_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        k = torch.randn(B, H, 8, 128)  # longer seq for block testing
        v = torch.randn(B, H, 8, 128)
        k_out, v_out = wrapped.update(k, v, 0)

        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
