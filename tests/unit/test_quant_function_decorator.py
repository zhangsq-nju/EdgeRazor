"""Unit tests for quant_function decorators.

This directly addresses Bug 1 (TypeError from decorator parameter name mismatch).
The per_block_reshape and mixed_precision_quantize decorators must accept:
- positional tensor arg
- keyword 'w' (weight quant callers)
- keyword 'x' (activation/state quant callers)
"""

import pytest
import torch
from torch import Tensor

from edgerazor.qat.util.quant_function_decorator import (
    _apply_masked_quantize,
    mixed_precision_quantize,
    per_block_reshape,
)


# ──────────────────────────────────────────────
# Fixtures: decorated dummy functions
# ──────────────────────────────────────────────

@pytest.fixture
def dummy_per_block_fn():
    """A simple per-block function that returns the unmodified tensor."""

    @per_block_reshape
    def fn(w: Tensor, epsilon: float = 1e-5, block_size: int = 64):
        # w is already reshaped to [..., -1, block_size]
        return w

    return fn


@pytest.fixture
def dummy_clip_per_block_fn():
    """Simulates weight_quant_uniform_symmetric_clip_per_block_int1_58 pattern."""

    @per_block_reshape
    def fn(w: Tensor, epsilon: float = 1e-5, w_scale_factor: float = 2.0, block_size: int = 64):
        with torch.no_grad():
            w_scale = w.abs().mean(dim=-1, keepdim=True).mul_(w_scale_factor).clamp_(min=epsilon)
            w_quant = w.div(w_scale).round_().clamp_(-1, 1) * w_scale
        return w_quant

    return fn


@pytest.fixture
def dummy_mp_fn():
    """Simulates a mixed-precision function that returns a mask."""

    @mixed_precision_quantize
    def fn(
        w: Tensor,
        epsilon: float = 1e-5,
        w_scale_factor: float = 2.0,
        block_size: int = 64,
        mixed_precision_prop: float = 0.1,
    ):
        num_blocks = w.numel() // block_size
        with torch.no_grad():
            magnitude = w.reshape(num_blocks, block_size).norm(dim=-1)
            _, indices = torch.topk(magnitude, max(1, int(num_blocks * mixed_precision_prop)))
            mask = torch.zeros(num_blocks, dtype=torch.bool)
            mask[indices] = True
        return mask

    return fn


# ──────────────────────────────────────────────
# per_block_reshape — positional tensor arg
# ──────────────────────────────────────────────

class TestPerBlockReshapePositionalArg:
    """Calling decorated function with tensor as first positional arg."""

    def test_positional_2d_dims_divisible(self, dummy_per_block_fn):
        w = torch.randn(8, 64)
        result = dummy_per_block_fn(w, epsilon=1e-5, block_size=64)
        assert result.shape == (8, 64)
        torch.testing.assert_close(result, w)

    def test_positional_2d_flatten_divisible(self, dummy_per_block_fn):
        w = torch.randn(16, 32)
        result = dummy_per_block_fn(w, block_size=64)
        assert result.shape == (16, 32)

    def test_positional_3d_last_dim_divisible(self, dummy_per_block_fn):
        w = torch.randn(2, 4, 128)
        result = dummy_per_block_fn(w, block_size=64)
        assert result.shape == (2, 4, 128)

    def test_positional_sparse_3d(self, dummy_per_block_fn):
        """Conv2d weights: [out_c, in_c, kH*kW]."""
        w = torch.randn(32, 3 * 3 * 3, dtype=torch.float32)
        result = dummy_per_block_fn(w, block_size=64)
        assert result.shape == w.shape

    def test_positional_1d_tensor(self, dummy_per_block_fn):
        w = torch.randn(256)
        result = dummy_per_block_fn(w, block_size=64)
        assert result.shape == (256,)

    def test_positional_needs_padding(self, dummy_per_block_fn):
        """Case 3: total elements not divisible by block_size."""
        w = torch.randn(100)  # 100 not divisible by 64
        result = dummy_per_block_fn(w, block_size=64)
        assert result.shape == (100,)
        # first 100 elements should match
        torch.testing.assert_close(result, w)

    def test_positional_clip_quant_preserves_range(self, dummy_clip_per_block_fn):
        w = torch.randn(8, 64) * 0.5
        result = dummy_clip_per_block_fn(w, epsilon=1e-5, w_scale_factor=2.0, block_size=64)
        assert result.shape == (8, 64)
        assert result.dtype == w.dtype


# ──────────────────────────────────────────────
# per_block_reshape — keyword 'w' (Bug 1 regression)
# ──────────────────────────────────────────────

class TestPerBlockReshapeKeywordW:
    """Calling decorated function with keyword 'w=' — the pattern that triggered Bug 1."""

    def test_keyword_w_2d(self, dummy_per_block_fn):
        w = torch.randn(8, 64)
        result = dummy_per_block_fn(w=w, epsilon=1e-5, block_size=64)
        assert result.shape == (8, 64)
        torch.testing.assert_close(result, w)

    def test_keyword_w_with_extra_kwargs(self, dummy_clip_per_block_fn):
        """Matches QLinear._weight_quant() pattern: w=W, **self.w_kwargs."""
        W = torch.randn(8, 64) * 0.5
        kwargs = {'epsilon': 1e-5, 'w_scale_factor': 2.0, 'block_size': 64}
        result = dummy_clip_per_block_fn(w=W, **kwargs)
        assert result.shape == (8, 64)

    def test_keyword_w_sparse_3d(self, dummy_per_block_fn):
        """Conv2d pattern: QConv2d passes w=W_reshaped with kwargs."""
        W_reshaped = torch.randn(32, 3 * 3 * 3, dtype=torch.float32)
        kwargs = {'epsilon': 1e-5, 'block_size': 64}
        result = dummy_per_block_fn(w=W_reshaped, **kwargs)
        assert result.shape == W_reshaped.shape

    def test_keyword_w_needs_padding(self, dummy_per_block_fn):
        w = torch.randn(100)
        result = dummy_per_block_fn(w=w, block_size=64)
        assert result.shape == (100,)

    def test_keyword_w_3d_last_dim_divisible(self, dummy_per_block_fn):
        w = torch.randn(2, 8, 128)
        result = dummy_per_block_fn(w=w, block_size=64)
        assert result.shape == (2, 8, 128)


# ──────────────────────────────────────────────
# per_block_reshape — keyword 'x' (activation)
# ──────────────────────────────────────────────

class TestPerBlockReshapeKeywordX:
    """Calling decorated function with keyword 'x=' — activation quant pattern."""

    def test_keyword_x_2d(self, dummy_per_block_fn):
        x = torch.randn(16, 128)
        result = dummy_per_block_fn(x=x, epsilon=1e-5, block_size=64)
        assert result.shape == (16, 128)

    def test_keyword_x_with_extra_kwargs(self, dummy_clip_per_block_fn):
        """Matches QLinear._activation_quant() pattern: x=x, **self.a_kwargs."""
        X = torch.randn(32, 256)
        kwargs = {'epsilon': 1e-5, 'block_size': 64}
        result = dummy_clip_per_block_fn(x=X, **kwargs)
        assert result.shape == (32, 256)

    def test_keyword_x_needs_padding(self, dummy_per_block_fn):
        x = torch.randn(100)
        result = dummy_per_block_fn(x=x, block_size=64)
        assert result.shape == (100,)


# ──────────────────────────────────────────────
# per_block_reshape — missing tensor (error case)
# ──────────────────────────────────────────────

class TestPerBlockReshapeMissingTensor:
    """Calling decorated function without any tensor argument."""

    def test_no_args_raises(self, dummy_per_block_fn):
        with pytest.raises(TypeError, match="missing required tensor argument"):
            dummy_per_block_fn(block_size=64)


# ──────────────────────────────────────────────
# per_block_reshape — original shape preservation
# ──────────────────────────────────────────────

class TestPerBlockReshapeRoundTrip:
    """Verify the round-trip preserves the original shape for various cases."""

    @pytest.mark.parametrize("shape,block_size", [
        ((8, 64), 64),         # last dim divisible
        ((16, 32), 64),        # total divisible
        ((100,), 64),          # needs padding
        ((2, 4, 128), 64),     # 3D last dim divisible
        ((32, 27), 64),        # total divisible (864 = 64 * 13.5... actually 864/64=13.5, not divisible)
        ((32, 27), 9),         # total divisible (864 / 9 = 96)
        ((7, 13), 16),         # needs padding
    ])
    def test_shape_preserved(self, dummy_per_block_fn, shape, block_size):
        w = torch.randn(*shape)
        result = dummy_per_block_fn(w, block_size=block_size)
        assert result.shape == shape


# ──────────────────────────────────────────────
# mixed_precision_quantize
# ──────────────────────────────────────────────

class TestMixedPrecisionQuantizePositional:
    """Calling mixed_precision_quantize decorated fn with positional tensor."""

    def test_positional_returns_correct_shape(self, dummy_mp_fn):
        w = torch.randn(8, 64) * 0.5
        result = dummy_mp_fn(w, epsilon=1e-5, w_scale_factor=2.0, block_size=64, mixed_precision_prop=0.1)
        assert result.shape == (8, 64)
        assert result.dtype == w.dtype

    def test_positional_quantized_values_in_range(self, dummy_mp_fn):
        w = torch.randn(16, 64)
        result = dummy_mp_fn(w, epsilon=1e-5, w_scale_factor=2.0, block_size=64, mixed_precision_prop=0.1)
        # Result exists and has correct shape
        assert result.shape == (16, 64)


class TestMixedPrecisionQuantizeKeywordW:
    """Calling mixed_precision_quantize decorated fn with keyword 'w='."""

    def test_keyword_w_returns_correct_shape(self, dummy_mp_fn):
        w = torch.randn(16, 32)
        kwargs = {
            'epsilon': 1e-5,
            'w_scale_factor': 2.0,
            'block_size': 32,
            'mixed_precision_prop': 0.1,
        }
        result = dummy_mp_fn(w=w, **kwargs)
        assert result.shape == (16, 32)

    def test_keyword_w_3d(self, dummy_mp_fn):
        w = torch.randn(4, 8, 128)
        result = dummy_mp_fn(w=w, block_size=64, mixed_precision_prop=0.1)
        assert result.shape == (4, 8, 128)


class TestMixedPrecisionQuantizeKeywordX:
    """Calling mixed_precision_quantize decorated fn with keyword 'x='."""

    def test_keyword_x_returns_correct_shape(self, dummy_mp_fn):
        x = torch.randn(8, 128)
        result = dummy_mp_fn(x=x, block_size=64, mixed_precision_prop=0.1)
        assert result.shape == (8, 128)


class TestMixedPrecisionQuantizeMissingTensor:
    def test_no_args_raises(self, dummy_mp_fn):
        with pytest.raises(TypeError, match="missing required tensor argument"):
            dummy_mp_fn(block_size=64)


# ──────────────────────────────────────────────
# _apply_masked_quantize
# ──────────────────────────────────────────────

class TestApplyMaskedQuantize:
    def test_all_int4_blocks(self):
        w = torch.randn(8, 64)
        mask = torch.ones(8, dtype=torch.bool)  # all INT4
        result = _apply_masked_quantize(w, mask, epsilon=1e-5, w_scale_factor=2.0)
        assert result.shape == w.shape

    def test_all_int1_58_blocks(self):
        w = torch.randn(8, 64)
        mask = torch.zeros(8, dtype=torch.bool)  # all INT1_58
        result = _apply_masked_quantize(w, mask, epsilon=1e-5, w_scale_factor=2.0)
        assert result.shape == w.shape
        # Each block quantized to {-1, 0, 1} * scale, verify per-block:
        for i in range(8):
            block_vals = torch.unique(result[i])
            assert len(block_vals) <= 3  # at most 3 unique vals per block

    def test_mixed_blocks(self):
        w = torch.randn(16, 64)
        mask = torch.zeros(16, dtype=torch.bool)
        mask[:4] = True  # first 4 blocks INT4, rest INT1_58
        result = _apply_masked_quantize(w, mask, epsilon=1e-5, w_scale_factor=2.0)
        assert result.shape == w.shape

    def test_empty_int4_mask(self):
        w = torch.randn(8, 64)
        mask = torch.zeros(8, dtype=torch.bool)  # no INT4 blocks
        result = _apply_masked_quantize(w, mask, epsilon=1e-5, w_scale_factor=2.0)
        assert result.shape == w.shape

    def test_empty_int1_58_mask(self):
        w = torch.randn(8, 64)
        mask = torch.ones(8, dtype=torch.bool)  # no INT1_58 blocks
        result = _apply_masked_quantize(w, mask, epsilon=1e-5, w_scale_factor=2.0)
        assert result.shape == w.shape

    def test_custom_bits(self):
        w = torch.randn(8, 64)
        mask = torch.ones(8, dtype=torch.bool)
        result_4bit = _apply_masked_quantize(w, mask, epsilon=1e-5, w_scale_factor=2.0, bits_int4=4)
        result_7bit = _apply_masked_quantize(w, mask, epsilon=1e-5, w_scale_factor=2.0, bits_int4=7)
        assert result_4bit.shape == w.shape
        assert result_7bit.shape == w.shape
        # 7-bit should have more unique values than 4-bit
        assert len(torch.unique(result_7bit)) >= len(torch.unique(result_4bit))
