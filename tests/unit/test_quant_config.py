"""Unit tests for QuantConfig and related dataclasses."""

import json
import tempfile
from pathlib import Path

import pytest
import torch.nn as nn
import yaml

from edgerazor.qat.util.quant_config import (
    FunctionConfig,
    OverrideConfig,
    QuantConfig,
    SelectConfig,
)
from edgerazor.qat.util.quant_selector import QuantSelector


class TestFunctionConfig:
    def test_default_creation(self):
        cfg = FunctionConfig(
            epsilon=1e-5,
            weight_function="w_func",
            w_scale_factor=2.0,
            w_block_size=256,
            w_mixed_precision_prop=-1.0,
            is_w_quantized=True,
            activation_function=None,
            a_block_size=-1,
            a_mixed_precision_prop=-1.0,
            kv_cache_function=None,
            kv_block_size=-1,
            kv_mixed_precision_prop=-1.0,
        )
        assert cfg.epsilon == 1e-5
        assert cfg.weight_function == "w_func"
        assert cfg.is_w_quantized is True

    def test_copy_is_independent(self):
        cfg = FunctionConfig(
            epsilon=1e-5,
            weight_function="w_func",
            w_scale_factor=2.0,
            w_block_size=256,
            w_mixed_precision_prop=-1.0,
            is_w_quantized=True,
            activation_function=None,
            a_block_size=-1,
            a_mixed_precision_prop=-1.0,
            kv_cache_function=None,
            kv_block_size=-1,
            kv_mixed_precision_prop=-1.0,
        )
        copied = cfg.copy()
        assert copied.weight_function == cfg.weight_function
        # Mutate copy and verify original is unchanged
        copied = copied.merge({"w_block_size": 128})
        assert cfg.w_block_size == 256
        assert copied.w_block_size == 128

    def test_merge_overrides_existing_fields(self):
        cfg = FunctionConfig(
            epsilon=1e-5,
            weight_function="w_func",
            w_scale_factor=2.0,
            w_block_size=256,
            w_mixed_precision_prop=-1.0,
            is_w_quantized=True,
            activation_function=None,
            a_block_size=-1,
            a_mixed_precision_prop=-1.0,
            kv_cache_function=None,
            kv_block_size=-1,
            kv_mixed_precision_prop=-1.0,
        )
        merged = cfg.merge({
            "weight_function": "new_func",
            "w_block_size": 128,
            "epsilon": 1e-6,
        })
        assert merged.weight_function == "new_func"
        assert merged.w_block_size == 128
        assert merged.epsilon == 1e-6
        # Unchanged fields
        assert merged.w_scale_factor == 2.0

    def test_merge_activation_function(self):
        cfg = FunctionConfig(
            epsilon=1e-5,
            weight_function="w_func",
            w_scale_factor=2.0,
            w_block_size=256,
            w_mixed_precision_prop=-1.0,
            is_w_quantized=True,
            activation_function=None,
            a_block_size=-1,
            a_mixed_precision_prop=-1.0,
            kv_cache_function=None,
            kv_block_size=-1,
            kv_mixed_precision_prop=-1.0,
        )
        merged = cfg.merge({"activation_function": "state_quant_func"})
        assert merged.activation_function == "state_quant_func"


class TestOverrideConfig:
    def test_type_only_match(self):
        override = OverrideConfig(module_type="linear")
        assert override.matches("fc1", nn.Linear) is True
        assert override.matches("fc1", nn.Conv2d) is False

    def test_name_only_match(self):
        override = OverrideConfig(module_name=".*fc.*")
        assert override.matches("fc1", nn.Linear) is True
        assert override.matches("embed", nn.Linear) is False

    def test_type_and_name_match(self):
        override = OverrideConfig(module_type="linear", module_name=".*fc.*")
        assert override.matches("fc1", nn.Linear) is True
        assert override.matches("fc1", nn.Conv2d) is False
        assert override.matches("embed", nn.Linear) is False

    def test_neither_specified_returns_false(self):
        override = OverrideConfig()
        assert override.matches("fc1", nn.Linear) is False


class TestQuantConfigCreation:
    def test_from_empty_dict(self):
        cfg = QuantConfig({})
        assert cfg.method == "QAT"
        assert cfg.function.is_w_quantized is False

    def test_from_dict_with_none(self):
        cfg = QuantConfig(None)
        assert cfg.method == "QAT"

    def test_from_minimal_dict(self, basic_qat_config_dict):
        cfg = QuantConfig(basic_qat_config_dict)
        assert cfg.method == "QAT"
        assert cfg.function.w_block_size == 256
        assert nn.Linear in cfg.select.target_types

    def test_target_types_mapping(self):
        cfg = QuantConfig({
            "select": {"target_types": ["linear", "conv2d"]},
            "function": {},
        })
        assert nn.Linear in cfg.select.target_types
        assert nn.Conv2d in cfg.select.target_types

    def test_unknown_target_type_raises(self):
        with pytest.raises(ValueError, match="Unknown module type"):
            QuantConfig({"select": {"target_types": ["nonexistent_module"]}, "function": {}})

    def test_exclude_types(self):
        cfg = QuantConfig({
            "select": {
                "target_types": ["linear", "embedding"],
                "exclude_types": ["embedding"],
            },
            "function": {},
        })
        assert nn.Embedding in cfg.select.exclude_types

    def test_qat_configuration_wrapper(self):
        cfg = QuantConfig({"qat_configuration": {"method": "QAT", "function": {}, "select": {}}})
        assert cfg.method == "QAT"

    def test_overrides_parsing(self, basic_qat_config_dict):
        config_dict = basic_qat_config_dict.copy()
        config_dict["overrides"] = [
            {"name": ".*fc.*", "weight_function": "other_func", "w_block_size": 128}
        ]
        cfg = QuantConfig(config_dict)
        assert len(cfg.overrides) == 1
        assert cfg.overrides[0].module_name == ".*fc.*"

    def test_get_function_config_applies_overrides(self, basic_qat_config_dict):
        config_dict = basic_qat_config_dict.copy()
        config_dict["overrides"] = [
            {"name": ".*special.*", "w_block_size": 128}
        ]
        cfg = QuantConfig(config_dict)
        # Module that doesn't match override gets global config
        default_func = cfg.get_function_config("fc1", nn.Linear)
        assert default_func.w_block_size == 256
        # Module that matches override gets overridden config
        overridden_func = cfg.get_function_config("special_layer", nn.Linear)
        assert overridden_func.w_block_size == 128


class TestQuantConfigSerialization:
    def test_to_dict_basic(self, basic_qat_config_dict):
        cfg = QuantConfig(basic_qat_config_dict)
        result = cfg.to_dict()
        assert result["method"] == "QAT"
        assert "select" in result
        assert "function" in result

    def test_to_dict_and_back_roundtrip(self, basic_qat_config_dict):
        cfg = QuantConfig(basic_qat_config_dict)
        d = cfg.to_dict()
        # Note: function names are mapped to objects during init; round-trip works
        # but we need to reconstruct from the dict's string forms
        restored = QuantConfig(d)
        assert restored.method == cfg.method
        assert restored.function.w_block_size == cfg.function.w_block_size

    def test_to_yaml_and_from_yaml(self, basic_qat_config_dict, temp_dir):
        cfg = QuantConfig(basic_qat_config_dict)
        yaml_path = temp_dir / "test_config.yaml"
        cfg.to_yaml(yaml_path)
        assert yaml_path.exists()
        loaded = QuantConfig.from_yaml(yaml_path)
        assert loaded.method == cfg.method

    def test_to_json_and_from_json(self, basic_qat_config_dict, temp_dir):
        cfg = QuantConfig(basic_qat_config_dict)
        json_path = temp_dir / "test_config.json"
        cfg.to_json(json_path)
        assert json_path.exists()
        loaded = QuantConfig.from_json(json_path)
        assert loaded.method == cfg.method

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            QuantConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_string(self):
        yaml_str = """
method: QAT
select:
  target_types: [linear]
function:
  weight_function: weight_quant_uniform_symmetric_absmax_per_block_int4
  w_block_size: 128
"""
        cfg = QuantConfig.from_yaml_string(yaml_str)
        assert cfg.function.w_block_size == 128

    def test_from_json_string(self):
        json_str = '{"method": "QAT", "select": {"target_types": ["linear"]}, "function": {}}'
        cfg = QuantConfig.from_json_string(json_str)
        assert cfg.method == "QAT"
        assert nn.Linear in cfg.select.target_types


class TestQuantConfigStrRepr:
    def test_str_contains_method(self, basic_qat_config_dict):
        cfg = QuantConfig(basic_qat_config_dict)
        s = str(cfg)
        assert "QuantConfig" in s
        assert "QAT" in s

    def test_repr_equals_str(self, basic_qat_config_dict):
        cfg = QuantConfig(basic_qat_config_dict)
        assert repr(cfg) == str(cfg)


class TestKVCacheMetaTarget:
    """Tests for the 'kv_cache' pseudo-module target type."""

    def test_kv_cache_in_target_types_sets_flag(self):
        cfg = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear", "kv_cache"]},
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            },
        })
        assert cfg.select.kv_cache is True
        assert nn.Linear in cfg.select.target_types
        # kv_cache should NOT be in the nn.Module target_types set
        assert len(cfg.select.target_types) == 1

    def test_kv_cache_not_in_target_types(self):
        cfg = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear"]},
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            },
        })
        assert cfg.select.kv_cache is False

    def test_kv_cache_in_exclude_types_disables(self):
        """If kv_cache is in target_types but also in exclude_types, it's disabled."""
        cfg = QuantConfig({
            "method": "QAT",
            "select": {
                "target_types": ["linear", "kv_cache"],
                "exclude_types": ["kv_cache"],
            },
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            },
        })
        assert cfg.select.kv_cache is False

    def test_kv_cache_default_false(self):
        cfg = QuantConfig({})
        assert cfg.select.kv_cache is False

    def test_to_dict_includes_kv_cache(self):
        cfg = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear", "kv_cache"]},
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            },
        })
        d = cfg.to_dict()
        assert "kv_cache" in d["select"]["target_types"]
        assert "linear" in d["select"]["target_types"]

    def test_to_dict_roundtrip_preserves_kv_cache(self):
        cfg = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear", "kv_cache"]},
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            },
        })
        d = cfg.to_dict()
        restored = QuantConfig(d)
        assert restored.select.kv_cache is True
        assert nn.Linear in restored.select.target_types

    def test_to_dict_without_kv_cache(self):
        cfg = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear"]},
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            },
        })
        d = cfg.to_dict()
        assert "kv_cache" not in d["select"]["target_types"]


class TestFunctionNameResolution:
    """Test that function names get resolved to actual callables."""

    def test_weight_function_resolved(self):
        cfg = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear"]},
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            },
        })
        assert callable(cfg.function.weight_function)

    def test_unknown_weight_function_raises(self):
        with pytest.raises(ValueError, match="Unknown weight function"):
            QuantConfig({
                "method": "QAT",
                "select": {"target_types": ["linear"]},
                "function": {"weight_function": "nonexistent_func"},
            })

    def test_empty_activation_function_is_none(self):
        cfg = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear"]},
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                "activation_function": "",
            },
        })
        assert cfg.function.activation_function is None

    def test_empty_kv_cache_function_is_none(self):
        cfg = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear"]},
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                "kv_cache_function": "",
            },
        })
        assert cfg.function.kv_cache_function is None
