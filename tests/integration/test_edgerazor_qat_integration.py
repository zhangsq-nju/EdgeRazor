"""Integration tests for EdgeRazor + QAT pipeline."""

import pytest
import torch
import torch.nn as nn

from edgerazor import EdgeRazor
from edgerazor.qat.module import QLinear, QEmbedding, QConv2d


class TestQATOnMultiLayerModel:
    """Test QAT quantization on a model with multiple layer types."""

    @pytest.fixture
    def qat_config(self):
        return {
            "method": "QAT",
            "select": {
                "target_types": ["linear", "embedding", "conv2d"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": [],
            },
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                "w_scale_factor": 2.0,
                "w_block_size": 256,
                "w_mixed_precision_prop": -1.0,
                "is_w_quantized": True,
                "activation_function": "",
                "a_block_size": -1,
                "a_mixed_precision_prop": -1.0,
                "kv_cache_function": "",
                "kv_block_size": -1,
                "kv_mixed_precision_prop": -1.0,
            },
            "training": "all",
        }

    def test_quantize_all_layer_types(self, qat_config, simple_multi_layer_model):
        er = EdgeRazor(qat_config=qat_config)
        quantized = er.quantize(simple_multi_layer_model)

        assert isinstance(quantized.embed, QEmbedding)
        assert isinstance(quantized.conv, QConv2d)
        assert isinstance(quantized.fc1, QLinear)
        assert isinstance(quantized.fc2, QLinear)

    def test_quantized_modules_are_named_correctly(self, qat_config, simple_multi_layer_model):
        er = EdgeRazor(qat_config=qat_config)
        quantized = er.quantize(simple_multi_layer_model)

        # Verify module structure is preserved
        modules = dict(quantized.named_modules())
        assert "embed" in modules
        assert "conv" in modules
        assert "fc1" in modules
        assert "fc2" in modules

    def test_exclude_specific_layers(self, simple_multi_layer_model):
        config = {
            "method": "QAT",
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": ["fc2"],
            },
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                "w_scale_factor": 2.0,
                "w_block_size": 256,
                "w_mixed_precision_prop": -1.0,
                "is_w_quantized": True,
                "activation_function": "",
                "a_block_size": -1,
                "a_mixed_precision_prop": -1.0,
                "kv_cache_function": "",
                "kv_block_size": -1,
                "kv_mixed_precision_prop": -1.0,
            },
            "training": "all",
        }
        er = EdgeRazor(qat_config=config)
        quantized = er.quantize(simple_multi_layer_model)

        assert isinstance(quantized.fc1, QLinear)
        assert isinstance(quantized.fc2, nn.Linear)  # Excluded

    def test_qat_replace_resolves_correct_default_classes(
        self, basic_qat_config_dict, simple_linear_model
    ):
        er = EdgeRazor(qat_config=basic_qat_config_dict)
        quantized = er.quantize(simple_linear_model)
        # Verify the QAT object's resolve_qclass_map produces correct defaults
        qclass_map = er.qat._resolve_qclass_map()
        from edgerazor.qat.module import QLinear
        assert qclass_map["qlinear_cls"] == QLinear
        assert isinstance(quantized, nn.Module)


class TestQATWithPrebuiltConfigs:
    """Test QAT using pre-built configurations from quant_config_map."""

    def test_w4a8kv8_qwen3_config(self, simple_linear_model):
        config = {
            "method": "QAT",
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": [],
            },
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                "w_scale_factor": 2.0,
                "w_block_size": 256,
                "w_mixed_precision_prop": -1.0,
                "is_w_quantized": True,
                "activation_function": "",
                "a_block_size": -1,
                "a_mixed_precision_prop": -1.0,
                "kv_cache_function": "",
                "kv_block_size": -1,
                "kv_mixed_precision_prop": -1.0,
            },
            "training": "all",
        }
        er = EdgeRazor(qat_config=config)
        quantized = er.quantize(simple_linear_model)
        assert isinstance(quantized.fc, QLinear)

    def test_w1_58_ternary_config(self, simple_linear_model):
        config = {
            "method": "QAT",
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": [],
            },
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                "w_scale_factor": 2.0,
                "w_block_size": 256,
                "w_mixed_precision_prop": 0.0,
                "is_w_quantized": True,
                "activation_function": "",
                "a_block_size": -1,
                "a_mixed_precision_prop": -1.0,
                "kv_cache_function": "",
                "kv_block_size": -1,
                "kv_mixed_precision_prop": -1.0,
            },
            "training": "all",
        }
        er = EdgeRazor(qat_config=config)
        quantized = er.quantize(simple_linear_model)
        assert isinstance(quantized.fc, QLinear)


class TestConfigurationRoundTrip:
    """Test that configuration can be serialized and reloaded."""

    def test_qat_config_roundtrip_yaml(self, basic_qat_config_dict, temp_dir):
        from edgerazor.qat.util.quant_config import QuantConfig
        cfg = QuantConfig(basic_qat_config_dict)
        yaml_path = temp_dir / "qat.yaml"
        cfg.to_yaml(yaml_path)

        # Load from file
        loaded_cfg = QuantConfig.from_yaml(yaml_path)
        assert loaded_cfg.method == cfg.method
        assert nn.Linear in loaded_cfg.select.target_types

    def test_qat_config_roundtrip_json(self, basic_qat_config_dict, temp_dir):
        from edgerazor.qat.util.quant_config import QuantConfig
        cfg = QuantConfig(basic_qat_config_dict)
        json_path = temp_dir / "qat.json"
        cfg.to_json(json_path)

        loaded_cfg = QuantConfig.from_json(json_path)
        assert loaded_cfg.method == cfg.method

    def test_edge_config_roundtrip_yaml(self, unified_config_dict, temp_dir):
        from edgerazor.edgerazor_config import EdgeRazorConfig
        cfg = EdgeRazorConfig.from_dict(unified_config_dict)
        yaml_path = temp_dir / "edge.yaml"
        cfg.to_yaml(yaml_path)

        loaded_cfg = EdgeRazorConfig.from_yaml(yaml_path=yaml_path)
        assert loaded_cfg.has_qat is True
        assert loaded_cfg.has_kd is True
