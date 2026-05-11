"""Unit tests for DistillConfig and LossConfig."""

import pytest

from edgerazor.kd.util.distill_config import DistillConfig, LossConfig


class TestLossConfig:
    def test_default_creation(self):
        cfg = LossConfig()
        assert cfg.loss_type == "logits"
        assert cfg.alpha == 0.5
        assert cfg.temperature == 2.0

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="loss_type must be one of"):
            LossConfig(loss_type="invalid_type")

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction must be one of"):
            LossConfig(reduction="invalid_reduction")

    def test_valid_loss_types(self):
        for loss_type in ["logits", "hidden_states", "attentions", "past_key_values"]:
            cfg = LossConfig(loss_type=loss_type)
            assert cfg.loss_type == loss_type

    def test_valid_reductions(self):
        for reduction in ["sum", "mean", "batch_mean", "none"]:
            cfg = LossConfig(reduction=reduction)
            assert cfg.reduction == reduction

    def test_layer_index_string_validation(self):
        with pytest.raises(ValueError, match="layer_index string must be"):
            LossConfig(layer_index="invalid_layer_name")

    def test_layer_index_valid_strings(self):
        for name in ["low", "mid", "high", "adaptive"]:
            cfg = LossConfig(layer_index=name)
            assert cfg.layer_index == name

    def test_layer_index_list_validation(self):
        with pytest.raises(ValueError, match="layer_index string must be"):
            LossConfig(layer_index=["low", "invalid_name"])

    def test_layer_index_adaptive_metric_validation(self):
        with pytest.raises(ValueError, match="layer_index_adaptive_metric"):
            LossConfig(layer_index_adaptive_metric="invalid_metric")

    def test_custom_values(self):
        cfg = LossConfig(
            loss_type="hidden_states",
            loss_function="compute_mse",
            alpha=0.8,
            temperature=4.0,
            layer_index=[0, 3, 6],
            padding_id=-1,
        )
        assert cfg.loss_type == "hidden_states"
        assert cfg.alpha == 0.8
        assert cfg.temperature == 4.0
        assert cfg.layer_index == [0, 3, 6]
        assert cfg.padding_id == -1


class TestDistillConfig:
    def test_from_dict_basic(self, basic_kd_config_dict):
        cfg = DistillConfig.from_dict(basic_kd_config_dict)
        assert cfg.method == "KD"
        assert "loss_1" in cfg.losses
        assert cfg.losses["loss_1"].loss_type == "logits"

    def test_from_dict_multi_loss(self):
        config_dict = {
            "method": "KD",
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
            },
            "loss_2": {
                "loss_type": "hidden_states",
                "loss_function": "compute_mse",
                "alpha": 0.3,
                "layer_index": [0, -1],
            },
        }
        cfg = DistillConfig.from_dict(config_dict)
        assert len(cfg.losses) == 2
        assert cfg.losses["loss_1"].alpha == 0.5
        assert cfg.losses["loss_2"].alpha == 0.3

    def test_from_dict_kd_configuration_wrapper(self):
        config_dict = {
            "kd_configuration": {
                "method": "KD",
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "compute_kld_forward",
                    "alpha": 1.0,
                },
            }
        }
        cfg = DistillConfig.from_dict(config_dict)
        assert cfg.method == "KD"
        assert cfg.losses["loss_1"].loss_function == "compute_kld_forward"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be 'KD'"):
            DistillConfig(method="INVALID")

    def test_no_losses_raises(self):
        with pytest.raises(ValueError, match="At least one loss"):
            DistillConfig(method="KD", losses={})

    def test_loss_task_alpha_default(self):
        cfg = DistillConfig.from_dict({
            "method": "KD",
            "loss_1": {"loss_type": "logits", "loss_function": "compute_kld_reverse", "alpha": 0.5},
        })
        assert cfg.loss_task_alpha == 1.0

    def test_custom_loss_task_alpha(self):
        cfg = DistillConfig.from_dict({
            "method": "KD",
            "loss_task_alpha": 0.7,
            "loss_1": {"loss_type": "logits", "loss_function": "compute_kld_reverse", "alpha": 0.5},
        })
        assert cfg.loss_task_alpha == 0.7

    def test_auto_convert_dict_loss_to_loss_config(self):
        cfg = DistillConfig.from_dict({
            "method": "KD",
            "loss_1": {"loss_type": "logits", "loss_function": "compute_kld_reverse", "alpha": 0.5},
        })
        assert isinstance(cfg.losses["loss_1"], LossConfig)

    def test_to_dict_roundtrip(self, basic_kd_config_dict):
        cfg = DistillConfig.from_dict(basic_kd_config_dict)
        d = cfg.to_dict()
        assert d["method"] == "KD"
        assert "loss_1" in d

    def test_to_yaml_and_from_yaml(self, basic_kd_config_dict, temp_dir):
        cfg = DistillConfig.from_dict(basic_kd_config_dict)
        yaml_path = temp_dir / "test_kd.yaml"
        cfg.to_yaml(yaml_path)
        assert yaml_path.exists()
        loaded = DistillConfig.from_yaml(yaml_path)
        assert loaded.method == cfg.method

    def test_to_json_and_from_json(self, basic_kd_config_dict, temp_dir):
        cfg = DistillConfig.from_dict(basic_kd_config_dict)
        json_path = temp_dir / "test_kd.json"
        cfg.to_json(json_path)
        assert json_path.exists()
        loaded = DistillConfig.from_json(json_path)
        assert loaded.method == cfg.method

    def test_repr(self, basic_kd_config_dict):
        cfg = DistillConfig.from_dict(basic_kd_config_dict)
        r = repr(cfg)
        assert "DistillConfig" in r
        assert "KD" in r
