"""Unit tests for QEmbedding module: forward, backward, weight quant, STE."""

import pytest
import torch
import torch.nn as nn

from edgerazor.qat.module.qembedding import QEmbedding
from edgerazor.qat.util.quant_config import QuantConfig


def _make_config(
    weight_function: str = "weight_quant_uniform_symmetric_clip_per_block_int1_58",
    w_scale_factor: float = 2.0,
    w_block_size: int = 64,
    is_w_quantized: bool = False,
) -> QuantConfig:
    return QuantConfig({
        "method": "QAT",
        "select": {"target_types": ["embedding"], "target_names": [],
                    "exclude_types": [], "exclude_names": []},
        "function": {
            "epsilon": 1e-5,
            "weight_function": weight_function,
            "w_scale_factor": w_scale_factor,
            "w_block_size": w_block_size,
            "w_mixed_precision_prop": -1.0,
            "is_w_quantized": is_w_quantized,
            "activation_function": "",
            "a_block_size": -1,
            "a_mixed_precision_prop": -1.0,
            "kv_cache_function": "",
            "kv_block_size": -1,
            "kv_mixed_precision_prop": -1.0,
        },
        "training": "all",
    })


def _make_qembedding(num_embeddings=100, embedding_dim=32, **kwargs):
    cfg = _make_config(**kwargs)
    return QEmbedding(num_embeddings, embedding_dim, quant_config=cfg)


class TestQEmbeddingConstruction:
    def test_basic_construction(self):
        emb = _make_qembedding(100, 32)
        assert emb.num_embeddings == 100
        assert emb.embedding_dim == 32
        assert emb.weight.shape == (100, 32)

    def test_construction_without_config_raises(self):
        with pytest.raises(ValueError, match="quant_config must be provided"):
            QEmbedding(100, 32, quant_config=None)


class TestWeightQuant:
    def test_returns_tensor_with_same_shape(self):
        emb = _make_qembedding(100, 32, w_block_size=32)
        w_quant = emb._weight_quant(replace_self=False)
        assert w_quant.shape == (100, 32)

    def test_does_not_modify_weight_when_replace_self_false(self):
        emb = _make_qembedding(100, 32)
        original = emb.weight.data.clone()
        emb._weight_quant(replace_self=False)
        assert torch.equal(emb.weight.data, original)

    def test_replace_self_replaces_weight(self):
        emb = _make_qembedding(100, 32, is_w_quantized=False)
        original = emb.weight.data.clone()
        emb._weight_quant(replace_self=True)
        assert emb.is_w_quantized is True
        assert not torch.equal(emb.weight.data, original)

    def test_quant_with_keyword_w(self):
        emb = _make_qembedding(100, 32, w_block_size=32)
        W = emb.weight.data.clone()
        result = emb.w_quant_function(w=W, **emb.w_kwargs)
        assert result.shape == (100, 32)


class TestForwardTraining:
    def test_forward_training_returns_correct_shape(self):
        emb = _make_qembedding(100, 32).train()
        x = torch.randint(0, 100, (4, 8))  # (batch=4, seq=8)
        out = emb(x)
        assert out.shape == (4, 8, 32)

    def test_backward_through_ste(self):
        emb = _make_qembedding(100, 32).train()
        emb.weight.data = torch.randn(100, 32) * 0.5
        w_before = emb.weight.data.clone()
        x = torch.randint(0, 100, (4, 8))
        out = emb(x)
        loss = out.sum()
        loss.backward()
        assert emb.weight.grad is not None
        assert torch.equal(emb.weight.data, w_before)

    def test_gradient_nonzero(self):
        emb = _make_qembedding(100, 32).train()
        emb.weight.data = torch.randn(100, 32) * 0.5
        x = torch.randint(0, 100, (4, 8))
        out = emb(x)
        loss = out.sum()
        loss.backward()
        assert emb.weight.grad.abs().sum() > 0


class TestForwardEval:
    def test_forward_eval_returns_correct_shape(self):
        emb = _make_qembedding(100, 32).eval()
        x = torch.randint(0, 100, (4, 8))
        with torch.no_grad():
            out = emb(x)
        assert out.shape == (4, 8, 32)

    def test_forward_eval_deterministic(self):
        emb = _make_qembedding(100, 32).eval()
        x = torch.randint(0, 100, (4, 8))
        with torch.no_grad():
            out1 = emb(x)
            out2 = emb(x)
        torch.testing.assert_close(out1, out2)

    def test_forward_eval_with_pre_quantized(self):
        emb = _make_qembedding(100, 32, is_w_quantized=True).eval()
        emb.weight.data = torch.randn(100, 32) * 0.5
        x = torch.randint(0, 100, (4, 8))
        with torch.no_grad():
            out = emb(x)
        assert out.shape == (4, 8, 32)
