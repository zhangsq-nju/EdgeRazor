"""TDD tests for quant_emb / quant_lm_head config.json fields.

These fields are the highest-priority toggle for whether embedding and
lm_head layers get quantized, overriding all quant_mode / default settings.
"""

import pytest

from edgerazor.vllm.edgerazor_quant import EdgeRazorConfig


# ── helpers ─────────────────────────────────────────────────


def _cfg(**kw):
    """Create an EdgeRazorConfig with minimal defaults for testing."""
    defaults = dict(
        weight_bits=4,
        activation_bits=16,
        quant_mode="",
    )
    defaults.update(kw)
    return EdgeRazorConfig(**defaults)


# ── quant_emb ────────────────────────────────────────────────


class TestQuantEmb:
    """quant_emb controls whether embed_tokens is quantized."""

    def test_quant_emb_false_disables_embedding_quant(self):
        """quant_emb=False → embed_tokens NOT quantized, overriding quant_mode."""
        cfg = _cfg(
            quant_mode="w4a8kv8_qwen3",  # normally quantizes embedding
            quant_emb=False,
        )
        assert cfg._is_layer_quantized("model.embed_tokens") is False

    def test_quant_emb_true_enables_embedding_quant(self):
        """quant_emb=True → embed_tokens IS quantized, overriding default skip."""
        cfg = _cfg(
            quant_mode="",  # default: skip embedding
            quant_emb=True,
        )
        assert cfg._is_layer_quantized("model.embed_tokens") is True

    def test_quant_emb_none_defers_to_quant_mode(self):
        """quant_emb not set → falls through to quant_mode / default logic."""
        cfg = _cfg(quant_mode="w4a8kv8_qwen3")  # quantizes embedding
        assert cfg._is_layer_quantized("model.embed_tokens") is True

    def test_quant_emb_none_defers_to_default_skip(self):
        """quant_emb not set, no quant_mode → default: skip embedding."""
        cfg = _cfg(quant_mode="")
        assert cfg._is_layer_quantized("model.embed_tokens") is False

    def test_quant_emb_false_still_allows_decoder_quant(self):
        """quant_emb=False only affects embedding, not decoder layers."""
        cfg = _cfg(
            quant_mode="w4a8kv8_qwen3",
            quant_emb=False,
        )
        assert cfg._is_layer_quantized("model.layers.0.self_attn.q_proj") is True

    def test_quant_emb_does_not_affect_lm_head(self):
        """quant_emb=False only affects embed_tokens, not lm_head."""
        cfg = _cfg(
            quant_mode="w4a8kv8_qwen3",
            quant_emb=False,
        )
        # lm_head is still quantized (controlled by quant_lm_head, not quant_emb)
        assert cfg._is_layer_quantized("lm_head") is True


# ── quant_lm_head ────────────────────────────────────────────


class TestQuantLMHead:
    """quant_lm_head controls whether lm_head is quantized."""

    def test_quant_lm_head_false_disables_lm_head_quant(self):
        """quant_lm_head=False → lm_head NOT quantized."""
        cfg = _cfg(
            quant_mode="w4a8kv8_qwen3",  # normally quantizes lm_head
            quant_lm_head=False,
        )
        assert cfg._is_layer_quantized("lm_head") is False

    def test_quant_lm_head_true_enables_lm_head_quant(self):
        """quant_lm_head=True → lm_head IS quantized, overriding default skip."""
        cfg = _cfg(
            quant_mode="",  # default: skip lm_head
            quant_lm_head=True,
        )
        assert cfg._is_layer_quantized("lm_head") is True

    def test_quant_lm_head_none_defers_to_quant_mode(self):
        """quant_lm_head not set → falls through to quant_mode logic."""
        cfg = _cfg(quant_mode="w4a8kv8_qwen3")  # quantizes lm_head
        assert cfg._is_layer_quantized("lm_head") is True

    def test_quant_lm_head_none_defers_to_default_skip(self):
        """quant_lm_head not set, no quant_mode → default: skip lm_head."""
        cfg = _cfg(quant_mode="")
        assert cfg._is_layer_quantized("lm_head") is False

    def test_quant_lm_head_false_still_allows_decoder_quant(self):
        """quant_lm_head=False only affects lm_head, not decoder layers."""
        cfg = _cfg(
            quant_mode="w4a8kv8_qwen3",
            quant_lm_head=False,
        )
        assert cfg._is_layer_quantized("model.layers.0.mlp.gate_proj") is True


# ── combination tests ────────────────────────────────────────


class TestCombined:
    """quant_emb and quant_lm_head can be set independently."""

    def test_both_false(self):
        cfg = _cfg(
            quant_mode="w4a8kv8_qwen3",
            quant_emb=False,
            quant_lm_head=False,
        )
        assert cfg._is_layer_quantized("model.embed_tokens") is False
        assert cfg._is_layer_quantized("lm_head") is False
        # decoder still quantized
        assert cfg._is_layer_quantized("model.layers.0.self_attn.q_proj") is True

    def test_emb_true_lm_head_false(self):
        cfg = _cfg(
            quant_mode="",
            quant_emb=True,
            quant_lm_head=False,
        )
        assert cfg._is_layer_quantized("model.embed_tokens") is True
        assert cfg._is_layer_quantized("lm_head") is False


# ── from_config integration ──────────────────────────────────


class TestFromConfig:
    """from_config reads quant_emb / quant_lm_head from config dict."""

    def test_from_config_reads_quant_emb_false(self):
        cfg = EdgeRazorConfig.from_config({
            "quant_method": "edgerazor",
            "quant_emb": False,
        })
        assert cfg._is_layer_quantized("model.embed_tokens") is False

    def test_from_config_reads_quant_lm_head_true(self):
        cfg = EdgeRazorConfig.from_config({
            "quant_method": "edgerazor",
            "quant_lm_head": True,
        })
        assert cfg._is_layer_quantized("lm_head") is True

    def test_from_config_with_quant_mode(self):
        """quant_emb/lm_head take priority over quant_mode."""
        cfg = EdgeRazorConfig.from_config({
            "quant_method": "edgerazor",
            "quant_mode": "w4a8kv8_qwen3",  # normally quantizes embedding
            "quant_emb": False,
            "quant_lm_head": False,
        })
        assert cfg._is_layer_quantized("model.embed_tokens") is False
        assert cfg._is_layer_quantized("lm_head") is False

    def test_from_config_fields_absent(self):
        """Missing quant_emb/lm_head → None → fall through to normal logic."""
        cfg = EdgeRazorConfig.from_config({
            "quant_method": "edgerazor",
            "quant_mode": "",
        })
        assert cfg._is_layer_quantized("model.embed_tokens") is False
        assert cfg._is_layer_quantized("lm_head") is False
