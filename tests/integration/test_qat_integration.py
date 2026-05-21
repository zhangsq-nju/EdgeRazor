"""Integration tests for QAT: quantize + forward + backward + save/load pipeline.

Goes beyond unit tests by testing the full QAT pipeline end-to-end:
select modules, quantize them, forward, backward, replace weights, save/load.
"""

import copy

import pytest
import torch
import torch.nn as nn

from edgerazor.qat import QAT
from edgerazor.qat.module import QConv2d, QEmbedding, QLinear
from edgerazor.qat.util.quant_config import QuantConfig


def _make_config(target_types=("linear",), **kwargs) -> QuantConfig:
    return QuantConfig({
        "method": "QAT",
        "select": {
            "target_types": list(target_types),
            "target_names": [],
            "exclude_types": [],
            "exclude_names": [],
        },
        "function": {
            "epsilon": float(kwargs.get("epsilon", 1e-5)),
            "weight_function": kwargs.get("weight_function",
                "weight_quant_uniform_symmetric_clip_per_block_int1_58"),
            "w_scale_factor": float(kwargs.get("w_scale_factor", -1.0)),
            "w_block_size": int(kwargs.get("w_block_size", 64)),
            "w_mixed_precision_prop": float(kwargs.get("w_mixed_precision_prop", -1.0)),
            "is_w_quantized": kwargs.get("is_w_quantized", False),
            "activation_function": kwargs.get("activation_function", ""),
            "a_block_size": int(kwargs.get("a_block_size", -1)),
            "a_mixed_precision_prop": float(kwargs.get("a_mixed_precision_prop", -1.0)),
            "kv_cache_function": kwargs.get("kv_cache_function", ""),
            "kv_block_size": int(kwargs.get("kv_block_size", -1)),
            "kv_mixed_precision_prop": float(kwargs.get("kv_mixed_precision_prop", -1.0)),
        },
        "training": kwargs.get("training", "all"),
    })


# ──────────────────────────────────────────────
# Full pipeline: quantize → train → eval → replace
# ──────────────────────────────────────────────

class TestQATFullPipeline:
    """End-to-end QAT pipeline from quantize through train to save."""

    def test_full_int1_58_pipeline(self):
        """Full QAT pipeline with INT1_58 ternary quantization."""
        config = _make_config(
            target_types=("linear", "embedding"),
            weight_function="weight_quant_uniform_symmetric_clip_per_block_int1_58",
            w_block_size=64, w_scale_factor=2.0, is_w_quantized=False,
        )
        qat = QAT(config)

        # Build model
        model = nn.Sequential(
            nn.Embedding(100, 64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

        # Quantize
        quantized = qat.quantize(model)
        assert isinstance(quantized[0], QEmbedding)
        assert isinstance(quantized[1], QLinear)
        assert isinstance(quantized[3], QLinear)

        # Training loop
        quantized.train()
        optimizer = torch.optim.SGD(quantized.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            x = torch.randint(0, 100, (4, 16))
            out = quantized(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

        # Replace weights
        qat.replace_quantized_weights(quantized)
        assert quantized[0].is_w_quantized
        assert quantized[1].is_w_quantized
        assert quantized[3].is_w_quantized

        # Inference after replace
        quantized.eval()
        with torch.no_grad():
            out = quantized(torch.randint(0, 100, (4, 16)))
        assert out.shape == (4, 16, 10)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_full_int4_pipeline(self):
        """Full QAT pipeline with INT4 quantization."""
        config = _make_config(
            target_types=("linear", "embedding"),
            weight_function="weight_quant_uniform_symmetric_absmax_per_block_int4",
            w_block_size=256, is_w_quantized=False,
        )
        qat = QAT(config)

        model = nn.Sequential(
            nn.Embedding(100, 64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

        quantized = qat.quantize(model)
        assert isinstance(quantized[0], QEmbedding)
        assert isinstance(quantized[1], QLinear)
        assert isinstance(quantized[3], QLinear)

        quantized.train()
        optimizer = torch.optim.SGD(quantized.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randint(0, 100, (2, 16))
            out = quantized(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

        qat.replace_quantized_weights(quantized)
        for m in quantized:
            if hasattr(m, 'is_w_quantized'):
                assert m.is_w_quantized


class TestQATBackwardGradientFlow:
    """Verify gradients flow correctly through quantized model."""

    def test_gradients_flow_through_all_layers(self):
        config = _make_config(
            target_types=("linear", "embedding"),
            weight_function="weight_quant_uniform_symmetric_clip_per_block_int1_58",
            w_block_size=64, is_w_quantized=False,
        )
        qat = QAT(config)
        model = nn.Sequential(
            nn.Embedding(50, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 5),
        )
        quantized = qat.quantize(model)
        quantized.train()
        x = torch.randint(0, 50, (4, 8))
        out = quantized(x)
        loss = out.sum()
        loss.backward()
        for name, p in quantized.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradient"

    def test_gradients_computed_for_all_batches(self):
        config = _make_config(
            target_types=("linear",),
            weight_function="weight_quant_uniform_symmetric_absmax_per_block_int4",
            w_block_size=256, is_w_quantized=False,
        )
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
        quantized = qat.quantize(model)
        quantized.train()
        optimizer = torch.optim.Adam(quantized.parameters(), lr=0.001)
        params_before = {n: p.clone() for n, p in quantized.named_parameters() if p.requires_grad}
        for _ in range(5):
            optimizer.zero_grad()
            x = torch.randn(8, 32)
            out = quantized(x)
            loss = out.mean()
            loss.backward()
            optimizer.step()
        for n, p in quantized.named_parameters():
            if p.requires_grad:
                assert not torch.equal(p, params_before[n]), f"{n} did not update"


class TestQATSaveLoad:
    """Save/load quantized model state dict."""

    def test_save_load_state_dict(self, temp_dir):
        config = _make_config(
            target_types=("linear", "embedding"),
            weight_function="weight_quant_uniform_symmetric_clip_per_block_int1_58",
            w_block_size=64, is_w_quantized=False,
        )
        qat = QAT(config)
        model = nn.Sequential(nn.Embedding(50, 32), nn.Linear(32, 16))
        quantized = qat.quantize(model)
        quantized.train()
        x = torch.randint(0, 50, (4, 8))
        quantized(x).sum().backward()
        qat.replace_quantized_weights(quantized)
        save_path = temp_dir / "model.pt"
        torch.save(quantized.state_dict(), save_path)
        new_model = nn.Sequential(nn.Embedding(50, 32), nn.Linear(32, 16))
        new_quantized = qat.quantize(new_model)
        new_quantized.load_state_dict(torch.load(save_path))
        torch.testing.assert_close(new_quantized[0].weight, quantized[0].weight)
        torch.testing.assert_close(new_quantized[1].weight, quantized[1].weight)


class TestQATWithActivationQuant:
    """QAT with activation quantization enabled."""

    def test_activation_quant_forward_backward(self):
        config = _make_config(
            target_types=("linear",),
            weight_function="weight_quant_uniform_symmetric_absmax_per_block_int4",
            w_block_size=256,
            activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
            a_block_size=64,
            is_w_quantized=False,
        )
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        quantized = qat.quantize(model)
        quantized.train()
        x = torch.randn(4, 128)
        out = quantized(x)
        loss = out.sum()
        loss.backward()
        for name, p in quantized.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"

    def test_activation_quant_eval(self):
        config = _make_config(
            target_types=("linear",),
            weight_function="weight_quant_uniform_symmetric_absmax_per_block_int4",
            w_block_size=256,
            activation_function="state_quant_uniform_symmetric_absmax_per_block_int8",
            a_block_size=64,
            is_w_quantized=False,
        )
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(128, 64))
        quantized = qat.quantize(model)
        quantized.eval()
        with torch.no_grad():
            out = quantized(torch.randn(4, 128))
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()


class TestQATMixedPrecisionPipeline:
    """QAT pipeline with mixed-precision quantization."""

    def test_mixed_precision_forward_backward(self):
        config = _make_config(
            target_types=("linear",),
            weight_function="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
            w_block_size=64, w_mixed_precision_prop=0.1, is_w_quantized=False,
        )
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        quantized = qat.quantize(model)
        quantized.train()
        x = torch.randn(4, 128)
        out = quantized(x)
        loss = out.sum()
        loss.backward()
        for name, p in quantized.named_parameters():
            assert p.grad is not None

    def test_mixed_precision_replace_weights(self):
        config = _make_config(
            target_types=("linear",),
            weight_function="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
            w_block_size=64, w_mixed_precision_prop=0.1, is_w_quantized=False,
        )
        qat = QAT(config)
        model = nn.Sequential(nn.Linear(128, 64))
        quantized = qat.quantize(model)
        qat.replace_quantized_weights(quantized)
        assert quantized[0].is_w_quantized


class TestQATExcludePatterns:
    """Verify exclude patterns work in integration."""

    def test_exclude_names(self):
        config = _make_config(
            target_types=("linear",),
            weight_function="weight_quant_uniform_symmetric_absmax_per_block_int4",
            w_block_size=256, is_w_quantized=False,
        )
        config.select.exclude_names.add("1")  # exclude second Linear
        qat = QAT(config)
        model = nn.Sequential(
            nn.Linear(16, 32),   # [0] — included
            nn.Linear(32, 8),    # [1] — excluded by name "1"
        )
        quantized = qat.quantize(model)
        from edgerazor.qat.module import QLinear
        assert isinstance(quantized[0], QLinear)
        assert isinstance(quantized[1], nn.Linear)  # not quantized
