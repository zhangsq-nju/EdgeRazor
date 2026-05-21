"""Unit tests for quant_config_map, quant_function_map, and modules_map."""

import torch.nn as nn

from edgerazor.qat.map import (
    create_w1_58_config,
    create_w1_58_config_embint4,
    modules_map,
    quant_config_map,
    quant_function_map,
)


class TestQuantFunctionMap:
    def test_contains_all_expected_categories(self):
        names = list(quant_function_map.keys())
        # Weight quant functions
        assert any("weight_quant" in n for n in names)
        # State quant functions
        assert any("state_quant" in n for n in names)

    def test_functions_are_callable(self):
        for name, func in quant_function_map.items():
            assert callable(func), f"{name} should be callable"

    def test_function_names_match_keys(self):
        for name, func in quant_function_map.items():
            assert func.__name__ == name, f"{func.__name__} != {name}"


class TestModulesMap:
    def test_basic_modules_present(self):
        assert "linear" in modules_map
        assert "embedding" in modules_map
        assert "conv1d" in modules_map
        assert "conv2d" in modules_map
        assert "conv3d" in modules_map
        assert "multiheadattention" in modules_map

    def test_modules_are_correct_types(self):
        assert modules_map["linear"] == nn.Linear
        assert modules_map["embedding"] == nn.Embedding
        assert modules_map["conv1d"] == nn.Conv1d
        assert modules_map["conv2d"] == nn.Conv2d
        assert modules_map["conv3d"] == nn.Conv3d

    def test_no_legacy_attention_types(self):
        assert "llamaattention" not in modules_map
        assert "qwen3attention" not in modules_map
        assert "qwen2_5omniattention" not in modules_map
        assert "olmoeattention" not in modules_map


class TestQuantConfigMap:
    def test_contains_prebuilt_configs(self):
        assert "w4a8kv8_qwen3" in quant_config_map
        assert "w1_58a8kv8_embint4_qwen3" in quant_config_map
        assert "w4a8kv8_qwen2_5_omni" in quant_config_map
        assert "w4a8kv8_mobilellm" in quant_config_map

    def test_all_configs_are_dicts(self):
        for name, config in quant_config_map.items():
            assert isinstance(config, dict), f"{name} should be a dict"

    def test_all_configs_have_required_keys(self):
        for name, config in quant_config_map.items():
            assert "method" in config, f"{name} missing 'method'"
            assert config["method"] == "QAT", f"{name} method should be QAT"
            assert "select" in config, f"{name} missing 'select'"
            assert "function" in config, f"{name} missing 'function'"


class TestConfigBuilders:
    def test_create_w1_58_config_basic(self):
        config = create_w1_58_config()
        assert config["method"] == "QAT"
        assert "linear" in config["select"]["target_types"]
        assert "embedding" in config["select"]["target_types"]

    def test_create_w1_58_config_with_activation_kv(self):
        config = create_w1_58_config(with_activation_kv=True)
        targets = config["select"]["target_types"]
        assert "kv_cache" in targets
        func = config["function"]
        assert func["activation_function"] != ""
        assert func["kv_cache_function"] != ""

    def test_create_w1_58_config_embint4_basic(self):
        config = create_w1_58_config_embint4()
        assert config["method"] == "QAT"
        assert "overrides" in config
        overrides = config["overrides"]
        assert len(overrides) == 2
        name_patterns = [o["name"] for o in overrides]
        assert any("embed_tokens" in n for n in name_patterns)
        assert any("lm_head" in n for n in name_patterns)

    def test_create_w1_58_config_embint4_with_custom_mp_prop(self):
        config = create_w1_58_config_embint4(mp_prop=0.25)
        assert config["function"]["w_mixed_precision_prop"] == 0.25

    def test_create_w1_58_config_embint4_with_kv(self):
        config = create_w1_58_config_embint4(with_activation_kv=True)
        func = config["function"]
        assert func["activation_function"] != ""
        assert func["kv_cache_function"] != ""
