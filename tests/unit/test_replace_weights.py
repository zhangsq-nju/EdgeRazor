"""Unit tests for replace_applied_quantized_weights.

Tests the ability to permanently replace weights with their quantized versions
after training — which is critical for saving quantized model weights.
"""

import copy

import pytest
import torch
import torch.nn as nn

from edgerazor.qat import QAT
from edgerazor.qat.module import QConv2d, QEmbedding, QLinear
from edgerazor.qat.util.quant_config import QuantConfig


def _make_config(
    weight_function: str = "weight_quant_uniform_symmetric_clip_per_block_int1_58",
    w_scale_factor: float = 2.0,
    w_block_size: int = 64,
    is_w_quantized: bool = False,
    activation_function: str = "",
) -> QuantConfig:
    return QuantConfig({
        "method": "QAT",
        "select": {
            "target_types": ["linear", "embedding", "conv2d"],
            "target_names": [],
            "exclude_types": [],
            "exclude_names": [],
        },
        "function": {
            "epsilon": 1e-5,
            "weight_function": weight_function,
            "w_scale_factor": w_scale_factor,
            "w_block_size": w_block_size,
            "w_mixed_precision_prop": -1.0,
            "is_w_quantized": is_w_quantized,
            "activation_function": activation_function,
            "a_block_size": -1,
            "a_mixed_precision_prop": -1.0,
            "kv_cache_function": "",
            "kv_block_size": -1,
            "kv_mixed_precision_prop": -1.0,
        },
        "training": "all",
    })


class TestReplaceQLinearWeights:
    def test_replace_changes_weight(self):
        config = _make_config(is_w_quantized=False)
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(16, 8))
        quantized = qat.quantize(model)
        assert isinstance(quantized[0], QLinear)
        original_weight = quantized[0].weight.data.clone()
        qat.replace_quantized_weights(quantized)
        assert quantized[0].is_w_quantized
        assert not torch.equal(quantized[0].weight.data, original_weight)

    def test_replace_already_quantized_raises(self):
        config = _make_config(is_w_quantized=True)
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(16, 8))
        quantized = qat.quantize(model)
        assert quantized[0].is_w_quantized
        # Already quantized — replace_quantized_weights uses _weight_quant(replace_self=True)
        # which should raise since is_w_quantized is True
        with pytest.raises(RuntimeError, match="already ternarized"):
            qat.replace_quantized_weights(quantized)

    def test_replace_only_quantized_modules(self):
        config = _make_config(is_w_quantized=False)
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(16, 8))
        quantized = qat.quantize(model)
        original_weight = quantized[0].weight.data.clone()
        qat.replace_quantized_weights(quantized)
        assert quantized[0].is_w_quantized
        # Values should be in quantized form
        assert not torch.equal(quantized[0].weight.data, original_weight)


class TestReplaceQEmbeddingWeights:
    def test_replace_changes_embedding_weight(self):
        config = _make_config(is_w_quantized=False, w_block_size=32)
        qat = QAT(config)
        model = nn.Sequential(nn.Embedding(100, 32))
        quantized = qat.quantize(model)
        assert isinstance(quantized[0], QEmbedding)
        original_weight = quantized[0].weight.data.clone()
        qat.replace_quantized_weights(quantized)
        assert quantized[0].is_w_quantized
        assert not torch.equal(quantized[0].weight.data, original_weight)


class TestReplaceQConv2dWeights:
    def test_replace_changes_conv_weight(self):
        config = _make_config(is_w_quantized=False, w_block_size=32)
        qat = QAT(config)
        model = nn.Sequential(nn.Conv2d(3, 16, 3))
        quantized = qat.quantize(model)
        assert isinstance(quantized[0], QConv2d)
        original_weight = quantized[0].weight.data.clone()
        qat.replace_quantized_weights(quantized)
        assert quantized[0].is_w_quantized
        assert not torch.equal(quantized[0].weight.data, original_weight)


class TestReplaceMultiModuleModel:
    def test_replace_all_in_multi_module_model(self):
        config = _make_config(is_w_quantized=False, w_block_size=32)
        qat = QAT(config)
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 10),
        )
        quantized = qat.quantize(model)
        assert isinstance(quantized[0], QEmbedding)
        assert isinstance(quantized[1], QLinear)
        assert isinstance(quantized[2], QLinear)
        qat.replace_quantized_weights(quantized)
        assert quantized[0].is_w_quantized
        assert quantized[1].is_w_quantized
        assert quantized[2].is_w_quantized

    def test_replace_preserves_inference_functionality(self):
        """After replacing weights, the model should still produce valid output."""
        config = _make_config(is_w_quantized=False, w_block_size=32)
        qat = QAT(config)
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        quantized = qat.quantize(model)
        qat.replace_quantized_weights(quantized)
        quantized.eval()
        x = torch.randint(0, 100, (4, 8))
        with torch.no_grad():
            out = quantized(x)
        assert out.shape == (4, 8, 10)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestSaveLoadAfterReplace:
    def test_save_load_state_dict_after_replace(self, temp_dir):
        """After replacing weights, state_dict should save/load correctly."""
        config = _make_config(is_w_quantized=False, w_block_size=32)
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(16, 8))
        quantized = qat.quantize(model)
        qat.replace_quantized_weights(quantized)
        save_path = temp_dir / "quantized_model.pt"
        torch.save(quantized.state_dict(), save_path)
        loaded = nn.Sequential(nn.Linear(16, 8))
        loaded = qat.quantize(loaded)
        loaded.load_state_dict(torch.load(save_path))
        # Note: is_w_quantized is not part of state_dict (it's a regular attribute,
        # not a buffer), so after load_state_dict it stays False. That's expected.
        torch.testing.assert_close(loaded[0].weight, quantized[0].weight)
