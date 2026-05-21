"""Unit tests for QLinear module: forward, backward, weight quant, activation quant.

Previous tests only checked structural replacement (isinstance(fc, QLinear))
but never called forward() — which is how Bug 1 (TypeError) was missed.
"""

import copy

import pytest
import torch
import torch.nn as nn

from edgerazor.qat.module.qlinear import QLinear
from edgerazor.qat.util.quant_config import QuantConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_config(
    weight_function: str = "weight_quant_uniform_symmetric_clip_per_block_int1_58",
    w_scale_factor: float = 2.0,
    w_block_size: int = 64,
    w_mixed_precision_prop: float = -1.0,
    activation_function: str = "",
    a_block_size: int = -1,
    is_w_quantized: bool = False,
    bias: bool = False,
) -> QuantConfig:
    return QuantConfig({
        "method": "QAT",
        "select": {"target_types": ["linear"], "target_names": [],
                    "exclude_types": [], "exclude_names": []},
        "function": {
            "epsilon": 1e-5,
            "weight_function": weight_function,
            "w_scale_factor": w_scale_factor,
            "w_block_size": w_block_size,
            "w_mixed_precision_prop": w_mixed_precision_prop,
            "is_w_quantized": is_w_quantized,
            "activation_function": activation_function,
            "a_block_size": a_block_size,
            "a_mixed_precision_prop": -1.0,
            "kv_cache_function": "",
            "kv_block_size": -1,
            "kv_mixed_precision_prop": -1.0,
        },
        "training": "all",
    })


def _make_qlinear(in_features=16, out_features=8, bias=False, **kwargs):
    cfg = _make_config(**kwargs)
    return QLinear(in_features, out_features, bias=bias, quant_config=cfg)


# ──────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────

class TestQLinearConstruction:
    def test_basic_construction(self):
        q = _make_qlinear(16, 8)
        assert q.in_features == 16
        assert q.out_features == 8
        assert q.weight.shape == (8, 16)

    def test_construction_with_activation_quant(self):
        q = _make_qlinear(16, 8, activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
                          a_block_size=256)
        assert q.a_quant_function is not None
        assert q.a_block_size == 256

    def test_construction_without_config_raises(self):
        with pytest.raises(ValueError, match="quant_config must be provided"):
            QLinear(16, 8, quant_config=None)

    def test_construction_with_int4_weight(self):
        q = _make_qlinear(32, 16,
                          weight_function="weight_quant_uniform_symmetric_absmax_per_block_int4",
                          w_block_size=256)
        assert q.w_quant_function is not None
        assert q.w_block_size == 256

    def test_construction_preserves_original_weight_values(self):
        lin = nn.Linear(16, 8)
        q = _make_qlinear(16, 8)
        q.weight.data = lin.weight.data.clone()
        assert torch.equal(q.weight, lin.weight)


# ──────────────────────────────────────────────
# _weight_quant
# ──────────────────────────────────────────────

class TestWeightQuant:
    def test_returns_tensor_with_same_shape(self):
        q = _make_qlinear(16, 8, w_block_size=32)
        w_quant = q._weight_quant(replace_self=False)
        assert w_quant.shape == (8, 16)
        assert w_quant.dtype == q.weight.dtype

    def test_does_not_modify_weight_when_replace_self_false(self):
        q = _make_qlinear(16, 8)
        original = q.weight.data.clone()
        q._weight_quant(replace_self=False)
        assert torch.equal(q.weight.data, original)

    def test_replace_self_replaces_weight(self):
        q = _make_qlinear(16, 8, is_w_quantized=False)
        original = q.weight.data.clone()
        q._weight_quant(replace_self=True)
        assert q.is_w_quantized is True
        # Weight should have changed (quantized)
        assert not torch.equal(q.weight.data, original)

    def test_replace_self_twice_raises(self):
        q = _make_qlinear(16, 8, is_w_quantized=False)
        q._weight_quant(replace_self=True)
        with pytest.raises(RuntimeError, match="already ternarized"):
            q._weight_quant(replace_self=True)

    def test_replace_self_skips_if_already_quantized(self):
        q = _make_qlinear(16, 8, is_w_quantized=True)
        with pytest.raises(RuntimeError, match="already ternarized"):
            q._weight_quant(replace_self=True)

    def test_quant_with_keyword_w(self):
        """Directly test the keyword 'w=' pattern — the core Bug 1 regression."""
        q = _make_qlinear(64, 32, w_block_size=32)
        # Simulate what _weight_quant does internally
        W = q.weight.data.clone()
        result = q.w_quant_function(w=W, **q.w_kwargs)
        assert result.shape == (32, 64)

    def test_int4_quant_keyword_w(self):
        q = _make_qlinear(32, 16,
                          weight_function="weight_quant_uniform_symmetric_absmax_per_block_int4",
                          w_scale_factor=-1, w_block_size=32)
        W = q.weight.data.clone()
        result = q.w_quant_function(w=W, **q.w_kwargs)
        assert result.shape == (16, 32)

    def test_quant_with_padded_dims(self):
        """Test quantizing weights where last dim is not divisible by block_size."""
        q = _make_qlinear(50, 20, w_block_size=32)  # 50*20=1000 not divisible by 32
        W = q.weight.data.clone()
        result = q.w_quant_function(w=W, **q.w_kwargs)
        assert result.shape == (20, 50)


# ──────────────────────────────────────────────
# _activation_quant
# ──────────────────────────────────────────────

class TestActivationQuant:
    def test_returns_tensor_with_same_shape(self):
        q = _make_qlinear(16, 8,
                          activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
                          a_block_size=64)
        x = torch.randn(4, 16)
        x_quant = q._activation_quant(x)
        assert x_quant.shape == (4, 16)

    def test_quant_with_keyword_x(self):
        """Directly test the keyword 'x=' pattern."""
        q = _make_qlinear(16, 8,
                          activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
                          a_block_size=64)
        X = torch.randn(4, 16)
        result = q.a_quant_function(x=X, **q.a_kwargs)
        assert result.shape == (4, 16)

    def test_none_activation_skipped_in_forward(self):
        q = _make_qlinear(16, 8, activation_function="")
        assert q.a_quant_function is None


# ──────────────────────────────────────────────
# forward — training mode (STE)
# ──────────────────────────────────────────────

class TestForwardTraining:
    def test_forward_training_returns_correct_shape(self):
        q = _make_qlinear(16, 8).train()
        x = torch.randn(4, 16)
        out = q(x)
        assert out.shape == (4, 8)

    def test_forward_training_requires_grad(self):
        q = _make_qlinear(16, 8).train()
        x = torch.randn(4, 16, requires_grad=True)
        out = q(x)
        assert out.requires_grad

    def test_forward_training_with_bias(self):
        q = _make_qlinear(16, 8, bias=True).train()
        q.bias.data = torch.randn(8)
        x = torch.randn(4, 16)
        out = q(x)
        assert out.shape == (4, 8)

    def test_backward_through_ste(self):
        """Verify STE: gradients flow through quantized weights to the weight parameter."""
        q = _make_qlinear(16, 8).train()
        q.weight.data = torch.randn(8, 16) * 0.5
        # Make a copy of weights before forward
        w_before = q.weight.data.clone()
        x = torch.randn(4, 16)
        out = q(x)
        loss = out.sum()
        loss.backward()
        # Weight gradients should exist
        assert q.weight.grad is not None
        # Weight should not have changed during forward (STE: W + (w_quant - W).detach())
        assert torch.equal(q.weight.data, w_before)

    def test_backward_through_ste_multiple_steps(self):
        """Verify gradients flow over multiple forward/backward steps."""
        q = _make_qlinear(16, 8).train()
        q.weight.data = torch.randn(8, 16) * 0.5
        optimizer = torch.optim.SGD(q.parameters(), lr=0.01)
        w_before = q.weight.data.clone()
        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(4, 16)
            out = q(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
        # Weight should have changed after optimization
        assert not torch.equal(q.weight.data, w_before)

    def test_backward_with_activation_quant(self):
        """STE backward through both weight and activation quantization."""
        q = _make_qlinear(16, 8,
                          activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
                          a_block_size=64).train()
        q.weight.data = torch.randn(8, 16) * 0.5
        x = torch.randn(4, 16)
        out = q(x)
        loss = out.sum()
        loss.backward()
        assert q.weight.grad is not None

    def test_gradient_is_not_zero(self):
        """Quantized weights should still produce non-zero gradients."""
        q = _make_qlinear(16, 8).train()
        q.weight.data = torch.randn(8, 16) * 0.5
        x = torch.randn(4, 16)
        out = q(x)
        loss = out.sum()
        loss.backward()
        assert q.weight.grad.abs().sum() > 0


# ──────────────────────────────────────────────
# forward — eval mode (inference)
# ──────────────────────────────────────────────

class TestForwardEval:
    def test_forward_eval_returns_correct_shape(self):
        q = _make_qlinear(16, 8).eval()
        x = torch.randn(4, 16)
        out = q(x)
        assert out.shape == (4, 8)

    def test_forward_eval_no_grad(self):
        q = _make_qlinear(16, 8).eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            out = q(x)
        assert not out.requires_grad

    def test_forward_eval_deterministic(self):
        """Multiple forward passes should produce identical results in eval mode."""
        q = _make_qlinear(16, 8).eval()
        x = torch.randn(4, 16)
        out1 = q(x)
        out2 = q(x)
        torch.testing.assert_close(out1, out2)

    def test_forward_eval_with_pre_quantized_weights(self):
        """In eval mode with is_w_quantized=True, use weight directly without re-quantizing."""
        q = _make_qlinear(16, 8, is_w_quantized=True).eval()
        q.weight.data = torch.randn(8, 16) * 0.5
        w_before = q.weight.data.clone()
        x = torch.randn(4, 16)
        with torch.no_grad():
            out = q(x)
        assert out.shape == (4, 8)
        # Weight should NOT be modified during eval forward
        assert torch.equal(q.weight.data, w_before)

    def test_forward_eval_not_pre_quantized(self):
        """In eval mode without is_w_quantized, quantize on the fly."""
        q = _make_qlinear(16, 8, is_w_quantized=False).eval()
        q.weight.data = torch.randn(8, 16) * 0.5
        x = torch.randn(4, 16)
        with torch.no_grad():
            out = q(x)
        assert out.shape == (4, 8)

    def test_forward_eval_with_bias(self):
        q = _make_qlinear(16, 8, bias=True, is_w_quantized=True).eval()
        q.bias.data = torch.randn(8)
        q.weight.data = torch.randn(8, 16) * 0.5
        x = torch.randn(4, 16)
        with torch.no_grad():
            out = q(x)
        assert out.shape == (4, 8)

    def test_forward_eval_with_activation_quant(self):
        q = _make_qlinear(16, 8,
                          activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
                          a_block_size=64).eval()
        q.weight.data = torch.randn(8, 16) * 0.5
        x = torch.randn(4, 16)
        with torch.no_grad():
            out = q(x)
        assert out.shape == (4, 8)


# ──────────────────────────────────────────────
# Mixed-precision quant
# ──────────────────────────────────────────────

class TestMixedPrecisionQLinear:
    def test_mixed_precision_weight_quant(self):
        q = _make_qlinear(32, 16,
                          weight_function="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                          w_scale_factor=2.0, w_block_size=32, w_mixed_precision_prop=0.1)
        q.weight.data = torch.randn(16, 32) * 0.5
        w_quant = q._weight_quant(replace_self=False)
        assert w_quant.shape == (16, 32)

    def test_mixed_precision_forward_training(self):
        q = _make_qlinear(32, 16,
                          weight_function="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                          w_scale_factor=2.0, w_block_size=32, w_mixed_precision_prop=0.1).train()
        q.weight.data = torch.randn(16, 32) * 0.5
        x = torch.randn(4, 32)
        out = q(x)
        assert out.shape == (4, 16)
        loss = out.sum()
        loss.backward()
        assert q.weight.grad is not None

    def test_mixed_precision_replace_self(self):
        q = _make_qlinear(32, 16,
                          weight_function="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                          w_scale_factor=2.0, w_block_size=32, w_mixed_precision_prop=0.1,
                          is_w_quantized=False)
        q.weight.data = torch.randn(16, 32) * 0.5
        q._weight_quant(replace_self=True)
        assert q.is_w_quantized is True


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────

class TestQLinearEdgeCases:
    def test_single_input(self):
        """Single input sample (batch_size=1)."""
        q = _make_qlinear(16, 8).train()
        x = torch.randn(1, 16)
        out = q(x)
        assert out.shape == (1, 8)

    def test_large_block_size(self):
        """block_size larger than input features."""
        q = _make_qlinear(16, 8, w_block_size=256).train()
        x = torch.randn(4, 16)
        out = q(x)
        assert out.shape == (4, 8)

    def test_block_size_equal_to_elements(self):
        """block_size exactly equals weight elements."""
        q = _make_qlinear(16, 8, w_block_size=8 * 16).train()
        x = torch.randn(4, 16)
        out = q(x)
        assert out.shape == (4, 8)

    def test_gradient_does_not_flow_through_quantized_inference(self):
        """In eval mode, forward should not require grad."""
        q = _make_qlinear(16, 8).eval()
        x = torch.randn(4, 16)
        out = q(x)
        assert not out.requires_grad

    def test_weight_not_modified_during_training_forward(self):
        """In training, the original weight should remain unchanged during forward pass."""
        q = _make_qlinear(16, 8).train()
        q.weight.data = torch.randn(8, 16)
        w_before = q.weight.data.clone()
        x = torch.randn(4, 16)
        q(x)
        assert torch.equal(q.weight.data, w_before)

    def test_weight_quant_fn_receives_correct_params(self):
        """Verify w_kwargs contains the right keys and values."""
        q = _make_qlinear(16, 8, w_scale_factor=2.0, w_block_size=64)
        assert 'epsilon' in q.w_kwargs
        assert 'w_scale_factor' in q.w_kwargs
        assert 'block_size' in q.w_kwargs
        assert q.w_kwargs['block_size'] == 64
        assert q.w_kwargs['w_scale_factor'] == 2.0
