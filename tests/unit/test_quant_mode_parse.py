"""Unit tests for quant_mode_parse — per-layer bit-width resolution."""

import pytest

from edgerazor.vllm.quant_mode_parse import QuantModeConfig


# ── Tracer bullet: basic weight_bits extraction ──────────────


class TestWeightBitsExtraction:
    """Extract bit-width from quant_mode config."""

    def test_w4a8kv8(self):
        qm = QuantModeConfig("w4a8kv8")
        assert qm.weight_bits == 4
        assert qm.activation_bits == 8

    def test_w1_58a8kv8(self):
        qm = QuantModeConfig("w1_58a8kv8")
        assert qm.weight_bits == 1.58
        assert qm.activation_bits == 8

    def test_w1_58a8_weight_only(self):
        """w1_58a8 has no activation quant — weight-only config."""
        qm = QuantModeConfig("w1_58a8")
        assert qm.weight_bits == 1.58
        assert qm.activation_bits == 16  # no activation quant

    def test_w4a8_weight_only(self):
        """w4a8 has no activation quant — weight-only config."""
        qm = QuantModeConfig("w4a8")
        assert qm.weight_bits == 4
        assert qm.activation_bits == 16  # no activation quant


# ── Per-layer overrides ──────────────────────────────────────


class TestOverrides:
    """quant_mode overrides provide per-layer bit-width."""

    def test_decoder_uses_base(self):
        """Decoder layers use the base quantization (not overrides)."""
        qm = QuantModeConfig("w1_58a8kv8_embint4")
        assert qm.get_weight_bits("model.layers.0.self_attn.q_proj") == 1.58
        assert qm.get_weight_bits("model.layers.0.mlp.gate_proj") == 1.58
        assert qm.get_weight_bits("model.layers.35.mlp.down_proj") == 1.58

    def test_embed_tokens_overridden_to_w4(self):
        qm = QuantModeConfig("w1_58a8kv8_embint4")
        assert qm.get_weight_bits("model.embed_tokens") == 4
        assert qm.get_weight_bits("embed_tokens") == 4

    def test_lm_head_overridden_to_w4(self):
        qm = QuantModeConfig("w1_58a8kv8_embint4")
        assert qm.get_weight_bits("lm_head") == 4
        assert qm.get_weight_bits("model.lm_head") == 4

    def test_no_overrides_for_pure_configs(self):
        """w4a8kv8 has no overrides — all layers use base weight_bits."""
        qm = QuantModeConfig("w4a8kv8")
        assert qm.get_weight_bits("model.layers.0.self_attn.q_proj") == 4
        assert qm.get_weight_bits("model.embed_tokens") == 4
        assert qm.get_weight_bits("lm_head") == 4


# ── User override from config.json ───────────────────────────


class TestUserOverride:
    """config.json weight_bits / activation_bits override the base."""

    def test_weight_bits_override(self):
        qm = QuantModeConfig("w1_58a8kv8", weight_bits_override=4)
        assert qm.weight_bits == 4
        assert qm.get_weight_bits("model.layers.0.self_attn.q_proj") == 4

    def test_weight_bits_override_does_not_affect_overrides(self):
        """User override changes base but NOT per-layer overrides."""
        qm = QuantModeConfig("w1_58a8kv8_embint4", weight_bits_override=2)
        # Overrides still win for specific layers
        assert qm.get_weight_bits("model.embed_tokens") == 4  # override wins
        assert qm.get_weight_bits("model.layers.0.self_attn.q_proj") == 2  # base overridden

    def test_activation_bits_override(self):
        qm = QuantModeConfig("w4a8kv8", activation_bits_override=16)
        assert qm.activation_bits == 16


# ── Exclusion ────────────────────────────────────────────────


class TestExclusion:
    """exclude_names from quant_mode select config."""

    def test_omni_excludes_audio_tower(self):
        qm = QuantModeConfig("w4a8kv8_qwen2_5_omni")
        assert not qm.is_layer_quantized("thinker.audio_tower.0.fc")
        assert not qm.is_layer_quantized("talker.codec.0")
        assert qm.is_layer_quantized("model.layers.0.self_attn.q_proj")

    def test_no_exclude_for_standard_configs(self):
        qm = QuantModeConfig("w4a8kv8")
        assert qm.is_layer_quantized("model.layers.0.self_attn.q_proj")
        assert qm.is_layer_quantized("lm_head")


# ── Error cases ──────────────────────────────────────────────


class TestErrors:
    """Invalid quant_mode raises clear error."""

    def test_unknown_quant_mode(self):
        with pytest.raises(ValueError, match="Unknown quant_mode"):
            QuantModeConfig("nonexistent_mode")


# ── Target types ─────────────────────────────────────────────


class TestTargetTypes:
    def test_default_target_types(self):
        qm = QuantModeConfig("w4a8kv8")
        assert "linear" in qm.target_types

    def test_a8kv8_kv_cache_target_only(self):
        """a8kv8 has no weight modules, only kv_cache target type."""
        qm = QuantModeConfig("a8kv8")
        assert qm.target_types == ["kv_cache"]


# ── Clone block-size correctness ──────────────────────────────


class TestCloneBlockSize:
    """Clone for per-layer override must use the correct block size."""

    def test_embed_override_clone_uses_w4_ie(self):
        """embed_tokens override W4 → clone uses IE=32, not IE=256."""
        from edgerazor.vllm.edgerazor_quant import EdgeRazorConfig
        cfg = EdgeRazorConfig(quant_mode="w1_58a8kv8_embint4")
        # base is W1.58 → IE=256
        assert cfg.ie_block_size == 256

        # embed_tokens override → W4 → clone should use W4 IE=32
        wb = cfg._layer_weight_bits("model.embed_tokens")
        assert wb == 4
        clone = cfg._clone_with_weight_bits(wb)
        assert clone.weight_bits == 4
        assert clone.ie_block_size == 32
        assert clone.er_block_size == 256

    def test_lm_head_override_clone_uses_w4_ie(self):
        """lm_head override W4 → clone uses IE=32."""
        from edgerazor.vllm.edgerazor_quant import EdgeRazorConfig
        cfg = EdgeRazorConfig(quant_mode="w1_58a8kv8_embint4")
        wb = cfg._layer_weight_bits("lm_head")
        assert wb == 4
        clone = cfg._clone_with_weight_bits(wb)
        assert clone.ie_block_size == 32
        assert clone._scale_block_size == 32

    def test_decoder_no_override_uses_base_ie(self):
        """Decoder layers use base IE (256 for W1.58)."""
        from edgerazor.vllm.edgerazor_quant import EdgeRazorConfig
        cfg = EdgeRazorConfig(quant_mode="w1_58a8kv8_embint4")
        wb = cfg._layer_weight_bits("model.layers.0.self_attn.q_proj")
        assert wb == 1.58
        # wb == base weight_bits → no clone needed
        assert wb == cfg.weight_bits
