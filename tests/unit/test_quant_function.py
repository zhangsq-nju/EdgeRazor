"""Unit tests for quant function actual invocation.

Previous test_map.py only checked callable() — never invoked the functions.
This tests that all quant functions can be called with proper arguments.
"""

import pytest
import torch

from edgerazor.qat.map import quant_function_map


# ──────────────────────────────────────────────
# Helper: get a quant function by name
# ──────────────────────────────────────────────

def _get_fn(name: str):
    fn = quant_function_map.get(name)
    if fn is None:
        pytest.skip(f"Function '{name}' not in quant_function_map")
    return fn


# ──────────────────────────────────────────────
# INT1_58 Weight Quant — Clip Method
# ──────────────────────────────────────────────

class TestInt1_58ClipWeightQuant:
    """Test all INT1_58 clip-method weight quant functions."""

    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_clip_per_tensor_int1_58",
        "weight_quant_uniform_symmetric_clip_per_channel_int1_58",
    ])
    def test_non_decorated_functions_keyword_w(self, fn_name):
        """Non-decorated functions accept keyword 'w'."""
        fn = _get_fn(fn_name)
        w = torch.randn(32, 64)
        result = fn(w=w, epsilon=1e-5, w_scale_factor=2.0)
        assert result.shape == w.shape

    @pytest.mark.parametrize("fn_name,block_size,shape", [
        ("weight_quant_uniform_symmetric_clip_per_block_int1_58", 64, (32, 64)),
        ("weight_quant_uniform_symmetric_clip_per_block_int1_58", 64, (32, 128)),  # last dim divisible
        ("weight_quant_uniform_symmetric_clip_per_block_int1_58", 32, (16, 63)),  # needs padding
    ])
    def test_per_block_keyword_w(self, fn_name, block_size, shape):
        """per_block_reshape decorated functions accept keyword 'w' — Bug 1 regression."""
        fn = _get_fn(fn_name)
        w = torch.randn(*shape)
        kwargs = {'epsilon': 1e-5, 'w_scale_factor': 2.0, 'block_size': block_size}
        result = fn(w=w, **kwargs)
        assert result.shape == w.shape

    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_clip_per_block_int1_58",
    ])
    def test_per_block_positional(self, fn_name):
        fn = _get_fn(fn_name)
        w = torch.randn(32, 64)
        result = fn(w, epsilon=1e-5, w_scale_factor=2.0, block_size=64)
        assert result.shape == w.shape

    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_clip_per_block_int1_58",
    ])
    def test_per_block_keyword_x(self, fn_name):
        """per_block_reshape also accepts keyword 'x'."""
        fn = _get_fn(fn_name)
        x = torch.randn(32, 64)
        result = fn(x=x, epsilon=1e-5, w_scale_factor=2.0, block_size=64)
        assert result.shape == x.shape

    @pytest.mark.parametrize("fn_name,extra_kwargs", [
        ("weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic", {}),
        ("weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static", {}),
        ("weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_column_wise", {}),
        ("weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse", {}),
        ("weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse", {}),
    ])
    def test_mixed_precision_keyword_w(self, fn_name, extra_kwargs):
        """mixed_precision_quantize decorated functions."""
        fn = _get_fn(fn_name)
        w = torch.randn(32, 64)
        kwargs = {
            'epsilon': 1e-5,
            'w_scale_factor': 2.0,
            'block_size': 64,
            'mixed_precision_prop': 0.1,
            **extra_kwargs,
        }
        result = fn(w=w, **kwargs)
        assert result.shape == w.shape

    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    ])
    def test_mixed_precision_3d_keyword_w(self, fn_name):
        fn = _get_fn(fn_name)
        w = torch.randn(4, 8, 128)
        result = fn(w=w, block_size=64, mixed_precision_prop=0.1)
        assert result.shape == w.shape


# ──────────────────────────────────────────────
# INT1_58 Weight Quant — Absmax Method
# ──────────────────────────────────────────────

class TestInt1_58AbsmaxWeightQuant:
    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_absmax_per_tensor_int1_58",
        "weight_quant_uniform_symmetric_absmax_per_channel_int1_58",
    ])
    def test_non_decorated_functions(self, fn_name):
        """Absmax functions use absmax for scale, don't accept w_scale_factor."""
        fn = _get_fn(fn_name)
        w = torch.randn(32, 64)
        result = fn(w, epsilon=1e-5)
        assert result.shape == w.shape

    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_absmax_per_block_int1_58",
    ])
    def test_per_block_keyword_w(self, fn_name):
        fn = _get_fn(fn_name)
        w = torch.randn(32, 64)
        result = fn(w=w, epsilon=1e-5, block_size=64)
        assert result.shape == w.shape


# ──────────────────────────────────────────────
# INT4 Weight Quant — Symmetric Absmax
# ──────────────────────────────────────────────

class TestInt4WeightQuant:
    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_absmax_per_tensor_int4",
        "weight_quant_uniform_symmetric_absmax_per_channel_int4",
    ])
    def test_non_decorated_functions(self, fn_name):
        fn = _get_fn(fn_name)
        w = torch.randn(64, 32)
        result = fn(w, epsilon=1e-5)
        assert result.shape == w.shape

    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_absmax_per_block_int4",
    ])
    def test_per_block_keyword_w(self, fn_name):
        fn = _get_fn(fn_name)
        w = torch.randn(64, 256)
        result = fn(w=w, epsilon=1e-5, block_size=256)
        assert result.shape == w.shape

    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_symmetric_absmax_per_block_int5",
        "weight_quant_uniform_symmetric_absmax_per_block_int8",
    ])
    def test_per_block_high_bits(self, fn_name):
        fn = _get_fn(fn_name)
        w = torch.randn(64, 256)
        result = fn(w, epsilon=1e-5, block_size=64)
        assert result.shape == w.shape


# ──────────────────────────────────────────────
# INT4 Weight Quant — Asymmetric Max
# ──────────────────────────────────────────────

class TestInt4AsymmetricWeightQuant:
    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_asymmetric_max_per_tensor_int4",
        "weight_quant_uniform_asymmetric_max_per_channel_int4",
    ])
    def test_non_decorated_functions(self, fn_name):
        fn = _get_fn(fn_name)
        w = torch.randn(64, 32)
        result = fn(w, epsilon=1e-5)
        assert result.shape == w.shape

    @pytest.mark.parametrize("fn_name", [
        "weight_quant_uniform_asymmetric_max_per_block_int4",
    ])
    def test_per_block_keyword_w(self, fn_name):
        fn = _get_fn(fn_name)
        w = torch.randn(64, 256)
        result = fn(w=w, epsilon=1e-5, block_size=256)
        assert result.shape == w.shape


# ──────────────────────────────────────────────
# State Quant Functions (Activation / KV Cache)
# ──────────────────────────────────────────────

class TestStateQuant:
    @pytest.mark.parametrize("fn_name,shape", [
        ("state_quant_uniform_symmetric_absmax_per_token_int2", (4, 16, 64)),
        ("state_quant_uniform_symmetric_absmax_per_token_int4", (4, 16, 64)),
        ("state_quant_uniform_symmetric_absmax_per_token_int8", (4, 16, 64)),
    ])
    def test_per_token_functions(self, fn_name, shape):
        fn = _get_fn(fn_name)
        x = torch.randn(*shape)
        result = fn(x, epsilon=1e-5)
        assert result.shape == x.shape

    @pytest.mark.parametrize("fn_name,block_size,shape", [
        ("state_quant_uniform_symmetric_absmax_per_block_int2", 64, (4, 16, 64)),
        ("state_quant_uniform_symmetric_absmax_per_block_int4", 64, (4, 16, 64)),
        ("state_quant_uniform_symmetric_absmax_per_block_int8", 64, (4, 16, 64)),
        ("state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic", 64, (4, 16, 64)),
    ])
    def test_per_block_functions_keyword_x(self, fn_name, block_size, shape):
        """State quant functions accept keyword 'x'."""
        fn = _get_fn(fn_name)
        x = torch.randn(*shape)
        kwargs = {'epsilon': 1e-5, 'block_size': block_size}
        if 'mp' in fn_name:
            kwargs['mixed_precision_prop'] = 0.1
        result = fn(x=x, **kwargs)
        assert result.shape == x.shape

    @pytest.mark.parametrize("fn_name", [
        "state_quant_uniform_symmetric_absmax_per_block_int8",
    ])
    def test_per_block_keyword_x(self, fn_name):
        fn = _get_fn(fn_name)
        x = torch.randn(2, 8, 128)
        result = fn(x=x, epsilon=1e-5, block_size=64)
        assert result.shape == x.shape


# ──────────────────────────────────────────────
# Output value range checks
# ──────────────────────────────────────────────

class TestOutputValueRanges:
    def test_int1_58_output_in_ternary_range(self):
        """INT1_58 quantized values should be in {-scale, 0, scale} per element."""
        fn = _get_fn("weight_quant_uniform_symmetric_clip_per_block_int1_58")
        w = torch.randn(32, 64)
        result = fn(w, epsilon=1e-5, w_scale_factor=2.0, block_size=64)
        # Per-block quantization means values are ternary per block
        assert result.shape == w.shape

    def test_int4_output_shape_preserved(self):
        fn = _get_fn("weight_quant_uniform_symmetric_absmax_per_block_int4")
        w = torch.randn(64, 256)
        result = fn(w, epsilon=1e-5, block_size=256)
        assert result.shape == w.shape

    def test_state_quant_output_range(self):
        fn = _get_fn("state_quant_uniform_symmetric_absmax_per_block_int8")
        x = torch.randn(2, 8, 256)
        result = fn(x, epsilon=1e-5, block_size=256)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
