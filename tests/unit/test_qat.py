"""Unit tests for QAT class."""

from unittest.mock import patch

import pytest
import torch.nn as nn

from edgerazor.qat import QAT
from edgerazor.qat.module import QLinear, QEmbedding, QConv2d
from edgerazor.qat.util.quant_config import QuantConfig


class TestQATInit:
    def test_init_with_dict(self, basic_qat_config_dict):
        qat = QAT(basic_qat_config_dict)
        assert qat.config is not None
        assert qat.config.method == "QAT"

    def test_init_with_quant_config(self, basic_qat_config_dict):
        cfg = QuantConfig(basic_qat_config_dict)
        qat = QAT(cfg)
        assert qat.config is cfg

    def test_init_with_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Invalid configuration type"):
            QAT(42)

    def test_init_with_unsupported_file_format_raises(self, temp_dir):
        txt_file = temp_dir / "config.txt"
        txt_file.write_text("not a config")
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            QAT(str(txt_file))


class TestQATQuantize:
    def test_quantize_linear_model(self, basic_qat_config_dict, simple_linear_model):
        qat = QAT(basic_qat_config_dict)
        quantized = qat.quantize(simple_linear_model)
        assert isinstance(quantized.fc, QLinear)

    def test_quantize_embedding_layer(self, simple_multi_layer_model):
        config = {
            "method": "QAT",
            "select": {
                "target_types": ["embedding"],
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
        qat = QAT(config)
        quantized = qat.quantize(simple_multi_layer_model)
        assert isinstance(quantized.embed, QEmbedding)

    def test_quantize_conv2d_model(self, simple_cnn_model):
        config = {
            "method": "QAT",
            "select": {
                "target_types": ["conv2d"],
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
        qat = QAT(config)
        quantized = qat.quantize(simple_cnn_model)
        assert isinstance(quantized.conv2d, QConv2d)

    def test_quantize_no_matching_modules(self, simple_linear_model):
        config = {
            "method": "QAT",
            "select": {
                "target_types": ["conv2d"],
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
        qat = QAT(config)
        result = qat.quantize(simple_linear_model)
        assert result is simple_linear_model

    def test_quantize_preserves_parameter_count(self, basic_qat_config_dict, simple_linear_model):
        qat = QAT(basic_qat_config_dict)
        original_params = sum(p.numel() for p in simple_linear_model.parameters())
        quantized = qat.quantize(simple_linear_model)
        new_params = sum(p.numel() for p in quantized.parameters())
        assert new_params == original_params


class TestQATResolveQclassMap:
    def test_default_classes_used(self, basic_qat_config_dict):
        qat = QAT(basic_qat_config_dict)
        qclass_map = qat._resolve_qclass_map()
        assert qclass_map["qlinear_cls"] == QLinear
        assert qclass_map["qembedding_cls"] == QEmbedding

    def test_custom_class_override(self, basic_qat_config_dict):
        class CustomLinear(nn.Module):
            pass

        qat = QAT(basic_qat_config_dict)
        qclass_map = qat._resolve_qclass_map(qlinear_cls=CustomLinear)
        assert qclass_map["qlinear_cls"] == CustomLinear


class TestQATReplaceQuantizedWeights:
    def test_resolve_qclass_map_for_replace(
        self, basic_qat_config_dict
    ):
        qat = QAT(basic_qat_config_dict)
        qclass_map = qat._resolve_qclass_map(
            qlinear_cls=None,
            qembedding_cls=None,
            qconv1d_cls=None,
            qconv2d_cls=None,
            qconv3d_cls=None,
            qmultiheadattention_cls=None,
        )
        # Verify weight-related keys are resolved to default classes
        from edgerazor.qat.module import QLinear, QEmbedding, QConv2d
        assert qclass_map["qlinear_cls"] == QLinear
        assert qclass_map["qembedding_cls"] == QEmbedding
        assert qclass_map["qconv2d_cls"] == QConv2d
