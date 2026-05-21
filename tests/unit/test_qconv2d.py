"""Unit tests for QConv2d module: forward, backward, weight quant with reshape."""

import pytest
import torch

from edgerazor.qat.module.qconv2d import QConv2d
from edgerazor.qat.util.quant_config import QuantConfig


def _make_config(
    weight_function: str = "weight_quant_uniform_symmetric_clip_per_block_int1_58",
    w_scale_factor: float = 2.0,
    w_block_size: int = 64,
    activation_function: str = "",
    a_block_size: int = -1,
    is_w_quantized: bool = False,
) -> QuantConfig:
    return QuantConfig({
        "method": "QAT",
        "select": {"target_types": ["conv2d"], "target_names": [],
                    "exclude_types": [], "exclude_names": []},
        "function": {
            "epsilon": 1e-5,
            "weight_function": weight_function,
            "w_scale_factor": w_scale_factor,
            "w_block_size": w_block_size,
            "w_mixed_precision_prop": -1.0,
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


def _make_qconv2d(in_channels=3, out_channels=16, kernel_size=3, **kwargs):
    cfg = _make_config(**kwargs)
    return QConv2d(in_channels, out_channels, kernel_size, quant_config=cfg)


class TestQConv2dConstruction:
    def test_basic_construction(self):
        conv = _make_qconv2d(3, 16, 3)
        assert conv.in_channels == 3
        assert conv.out_channels == 16
        assert conv.weight.shape == (16, 3, 3, 3)

    def test_construction_without_config_raises(self):
        with pytest.raises(ValueError, match="quant_config must be provided"):
            QConv2d(3, 16, 3, quant_config=None)


class TestWeightQuant:
    def test_returns_correct_shape(self):
        """Weight quant reshapes [out, in, kH, kW] -> [out, in*kH*kW] and back."""
        conv = _make_qconv2d(3, 16, 3, w_block_size=32)
        w_quant = conv._weight_quant(replace_self=False)
        assert w_quant.shape == (16, 3, 3, 3)

    def test_does_not_modify_weight_when_replace_self_false(self):
        conv = _make_qconv2d(3, 16, 3)
        original = conv.weight.data.clone()
        conv._weight_quant(replace_self=False)
        assert torch.equal(conv.weight.data, original)

    def test_replace_self_replaces_weight(self):
        conv = _make_qconv2d(3, 16, 3, is_w_quantized=False)
        original = conv.weight.data.clone()
        conv._weight_quant(replace_self=True)
        assert conv.is_w_quantized is True
        assert not torch.equal(conv.weight.data, original)

    def test_quant_with_keyword_w(self):
        """Directly test keyword w= pattern used by QConv2d._weight_quant."""
        conv = _make_qconv2d(3, 16, 3, w_block_size=32)
        W = conv.weight.data.clone()
        W_reshaped = W.flatten(1)  # (16, 27)
        result = conv.w_quant_function(w=W_reshaped, **conv.w_kwargs)
        result = result.view(W.shape)
        assert result.shape == (16, 3, 3, 3)


class TestForwardTraining:
    def test_forward_training_returns_correct_shape(self):
        conv = _make_qconv2d(3, 16, 3).train()
        x = torch.randn(4, 3, 32, 32)
        out = conv(x)
        assert out.shape == (4, 16, 30, 30)  # No padding, kernel=3

    def test_backward_through_ste(self):
        conv = _make_qconv2d(3, 16, 3).train()
        conv.weight.data = torch.randn(16, 3, 3, 3) * 0.5
        w_before = conv.weight.data.clone()
        x = torch.randn(4, 3, 32, 32)
        out = conv(x)
        loss = out.sum()
        loss.backward()
        assert conv.weight.grad is not None
        assert torch.equal(conv.weight.data, w_before)

    def test_gradient_nonzero(self):
        conv = _make_qconv2d(3, 16, 3).train()
        conv.weight.data = torch.randn(16, 3, 3, 3) * 0.5
        x = torch.randn(4, 3, 32, 32)
        out = conv(x)
        loss = out.sum()
        loss.backward()
        assert conv.weight.grad.abs().sum() > 0

    def test_forward_training_with_activation_quant(self):
        conv = _make_qconv2d(3, 16, 3,
                             activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
                             a_block_size=64).train()
        conv.weight.data = torch.randn(16, 3, 3, 3) * 0.5
        x = torch.randn(4, 3, 32, 32)
        out = conv(x)
        assert out.shape == (4, 16, 30, 30)
        loss = out.sum()
        loss.backward()
        assert conv.weight.grad is not None


class TestForwardEval:
    def test_forward_eval_returns_correct_shape(self):
        conv = _make_qconv2d(3, 16, 3).eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = conv(x)
        assert out.shape == (4, 16, 30, 30)

    def test_forward_eval_deterministic(self):
        conv = _make_qconv2d(3, 16, 3).eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out1 = conv(x)
            out2 = conv(x)
        torch.testing.assert_close(out1, out2)

    def test_forward_eval_with_pre_quantized(self):
        conv = _make_qconv2d(3, 16, 3, is_w_quantized=True).eval()
        conv.weight.data = torch.randn(16, 3, 3, 3) * 0.5
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = conv(x)
        assert out.shape == (4, 16, 30, 30)


class TestQConv2dActivation:
    def test_activation_quant_shape(self):
        conv = _make_qconv2d(3, 16, 3,
                             activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
                             a_block_size=64)
        x = torch.randn(4, 3, 32, 32)
        x_quant = conv._activation_quant(x)
        assert x_quant.shape == x.shape
