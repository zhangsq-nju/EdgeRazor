"""Integration tests for v1.3.4 unified model loading and export.

Tests:
- EdgeRazorConfig.from_quant_mode() factory method
- New quant_config_map entries and legacy aliases
- Export tool generates correct HF repo structure
- replace_quantized_weights auto-sets is_w_quantized
- modeling_edgerazor.py template is valid Python
"""
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from edgerazor import EdgeRazor, EdgeRazorConfig, quant_config_map
from edgerazor.qat.map import _LEGACY_ALIASES


# ──────────────────────────────────────────────
# from_quant_mode tests
# ──────────────────────────────────────────────


class TestFromQuantMode:
    """Tests for EdgeRazorConfig.from_quant_mode()."""

    @pytest.mark.parametrize("mode", [
        "w4a8kv8",
        "w4a8kv8_embint4",
        "w4a8",
        "w2_79a8kv8",
        "w2_79a8kv8_embint4",
        "w1_88a8kv8",
        "w1_88a8kv8_embint4",
        "w1_58a8kv8",
        "w1_58a8kv8_embint4",
        "w1_58a8",
        "a8kv8",
    ])
    def test_from_quant_mode_creates_config(self, mode):
        """All new preset names should create a valid EdgeRazorConfig."""
        cfg = EdgeRazorConfig.from_quant_mode(mode, is_w_quantized=True)
        assert cfg.has_qat
        assert cfg.qat_config is not None
        assert cfg.qat_config.function.is_w_quantized is True

    @pytest.mark.parametrize("legacy_name, expected_generic", [
        ("w4a8kv8_qwen3", "w4a8kv8"),
        ("w4a8kv8_mobilellm", "w4a8kv8"),
        ("w2_79a8kv8_embint4_qwen3", "w2_79a8kv8_embint4"),
        ("w1_88a8kv8_embint4_qwen3", "w1_88a8kv8_embint4"),
        ("w1_58a8kv8_embint4_qwen3", "w1_58a8kv8_embint4"),
        ("w2_79a8kv8_embint4_mobilellm", "w2_79a8kv8_embint4"),
        ("w1_88a8kv8_embint4_mobilellm", "w1_88a8kv8_embint4"),
        ("w1_58a8kv8_embint4_mobilellm", "w1_58a8kv8_embint4"),
    ])
    def test_legacy_aliases_resolve(self, legacy_name, expected_generic):
        """Legacy model-specific names should resolve to new generic names."""
        assert legacy_name in _LEGACY_ALIASES
        assert _LEGACY_ALIASES[legacy_name] == expected_generic

    def test_legacy_quant_mode_still_in_map(self):
        """Old keys must remain directly in quant_config_map."""
        legacy_keys = [
            "w4a8kv8_qwen3",
            "w2_79a8kv8_embint4_qwen3",
            "w1_88a8kv8_embint4_qwen3",
            "w1_58a8kv8_embint4_qwen3",
            "w4a8kv8_qwen2_5_omni",
            "w4a8kv8_mobilellm",
        ]
        for key in legacy_keys:
            assert key in quant_config_map, f"Legacy key {key!r} missing from quant_config_map"

    def test_from_quant_mode_unknown_raises(self):
        """Unknown quant_mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown quant_mode"):
            EdgeRazorConfig.from_quant_mode("nonexistent_mode_xyz")

    def test_from_quant_mode_with_legacy_name(self):
        """from_quant_mode should resolve legacy names transparently."""
        cfg = EdgeRazorConfig.from_quant_mode("w1_58a8kv8_embint4_qwen3")
        assert cfg.has_qat
        assert cfg.qat_config is not None

    def test_from_quant_mode_is_w_quantized_false(self):
        """is_w_quantized=False should propagate to function config."""
        cfg = EdgeRazorConfig.from_quant_mode("w4a8kv8", is_w_quantized=False)
        assert cfg.qat_config.function.is_w_quantized is False


# ──────────────────────────────────────────────
# New config entries tests
# ──────────────────────────────────────────────


class TestNewQuantConfigMapEntries:
    """Tests for new v1.3.4 quant_config_map entries."""

    def test_all_new_entries_are_dicts(self):
        """All new entries should be valid config dicts."""
        new_keys = [
            "w4a8kv8", "w4a8kv8_embint4", "w4a8",
            "w2_79a8kv8", "w2_79a8kv8_embint4",
            "w1_88a8kv8", "w1_88a8kv8_embint4",
            "w1_58a8kv8", "w1_58a8kv8_embint4", "w1_58a8",
            "a8kv8",
        ]
        for key in new_keys:
            assert key in quant_config_map, f"Missing key: {key!r}"
            cfg = quant_config_map[key]
            assert isinstance(cfg, dict)
            assert "method" in cfg
            assert cfg["method"] == "QAT"

    def test_a8kv8_has_kv_cache_target(self):
        """a8kv8 should include kv_cache target (KV-only mode)."""
        cfg = quant_config_map["a8kv8"]
        targets = cfg["select"]["target_types"]
        # linear/embedding still included for activation quantization;
        # weights are not quantized (is_w_quantized=False).
        assert "kv_cache" in targets

    def test_a8kv8_is_w_quantized_false(self):
        """a8kv8 should have is_w_quantized=False (KV/activation only)."""
        cfg = quant_config_map["a8kv8"]
        assert cfg["function"]["is_w_quantized"] is False

    def test_w4a8_has_no_kv_cache(self):
        """w4a8 should not include kv_cache in target_types."""
        cfg = quant_config_map["w4a8"]
        targets = cfg["select"]["target_types"]
        assert "kv_cache" not in targets
        assert "linear" in targets
        assert "embedding" in targets

    def test_w1_58a8_has_no_kv_cache(self):
        """w1_58a8 should not include kv_cache in target_types."""
        cfg = quant_config_map["w1_58a8"]
        targets = cfg["select"]["target_types"]
        assert "kv_cache" not in targets

    def test_embint4_variants_have_overrides(self):
        """_embint4 variants should have embedding overrides."""
        for key in ["w4a8kv8_embint4", "w1_58a8kv8_embint4"]:
            cfg = quant_config_map[key]
            assert "overrides" in cfg
            assert len(cfg["overrides"]) > 0


# ──────────────────────────────────────────────
# replace_quantized_weights auto-set test
# ──────────────────────────────────────────────


class TestReplaceQuantizedWeightsAutoSet:
    """Tests that replace_quantized_weights auto-sets is_w_quantized."""

    def test_sets_is_w_quantized_on_config(self):
        """After replace_quantized_weights, model.config.is_w_quantized should be True."""
        class ModelStub(nn.Module):
            pass

        model = ModelStub()
        model.config = type("Cfg", (), {"is_w_quantized": False})()

        edgerazor = EdgeRazor(config=quant_config_map["w4a8"])
        edgerazor.quantize(model)
        result = edgerazor.replace_quantized_weights(model)

        assert result.config.is_w_quantized is True


# ──────────────────────────────────────────────
# Export tool tests
# ──────────────────────────────────────────────


class TestExportTool:
    """Tests for the export CLI tool."""

    def test_export_generates_config_json(self, temp_dir):
        """Export should create a patched config.json with edgerazor_qconfig."""
        src = temp_dir / "src"
        dst = temp_dir / "dst"
        src.mkdir()

        # Create a minimal source config.json
        base_config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 1024,
            "vocab_size": 151936,
        }
        with open(src / "config.json", "w") as f:
            json.dump(base_config, f)

        from edgerazor.export import export

        export(
            src_dir=src,
            dst_dir=dst,
            quant_mode="w1_58a8kv8_embint4",
            copy_weights=False,
        )

        # Verify config.json was patched
        assert (dst / "config.json").exists()
        with open(dst / "config.json") as f:
            out_cfg = json.load(f)

        assert out_cfg["edgerazor_qconfig"] == "w1_58a8kv8_embint4"
        assert out_cfg["is_w_quantized"] is True
        assert "auto_map" in out_cfg
        assert out_cfg["auto_map"]["AutoModelForCausalLM"] == \
            "modeling_edgerazor.EdgeRazorForCausalLM"

    def test_export_copies_modeling_edgerazor(self, temp_dir):
        """Export should copy the modeling_edgerazor.py template."""
        src = temp_dir / "src"
        dst = temp_dir / "dst"
        src.mkdir()

        base_config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
        }
        with open(src / "config.json", "w") as f:
            json.dump(base_config, f)

        from edgerazor.export import export

        export(src_dir=src, dst_dir=dst, quant_mode="w4a8kv8", copy_weights=False)

        assert (dst / "modeling_edgerazor.py").exists()
        content = (dst / "modeling_edgerazor.py").read_text()
        assert "class EdgeRazorForCausalLM" in content

    def test_export_no_w_quantized(self, temp_dir):
        """Export with --no_w_quantized should set is_w_quantized=False."""
        src = temp_dir / "src"
        dst = temp_dir / "dst"
        src.mkdir()

        base_config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
        with open(src / "config.json", "w") as f:
            json.dump(base_config, f)

        from edgerazor.export import export

        export(
            src_dir=src, dst_dir=dst, quant_mode="w4a8kv8",
            is_w_quantized=False, copy_weights=False,
        )

        with open(dst / "config.json") as f:
            out_cfg = json.load(f)
        assert out_cfg["is_w_quantized"] is False


# ──────────────────────────────────────────────
# modeling_edgerazor.py template validation
# ──────────────────────────────────────────────


class TestModelingEdgeRazorTemplate:
    """Validates the modeling_edgerazor.py template."""

    def test_template_is_valid_python(self):
        """The template should be syntactically valid Python."""
        template = (
            Path(__file__).parents[2]
            / "src" / "edgerazor" / "templates" / "modeling_edgerazor.py"
        )
        assert template.exists(), f"Template not found at {template}"
        source = template.read_text()
        compile(source, str(template), "exec")

    def test_template_defines_edge_razor_class(self):
        """Template must define EdgeRazorForCausalLM class."""
        template = (
            Path(__file__).parents[2]
            / "src" / "edgerazor" / "templates" / "modeling_edgerazor.py"
        )
        source = template.read_text()
        assert "class EdgeRazorForCausalLM" in source
        assert "def from_pretrained" in source
        assert "def _resolve_edgerazor_config" in source
        assert "def _inject_kv_cache" in source
