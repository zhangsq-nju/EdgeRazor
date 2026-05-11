"""Unit tests for EdgeRazorConfig."""

import logging
import pickle

import pytest

from edgerazor.edgerazor_config import EdgeRazorConfig


class TestEdgeRazorConfigInit:
    def test_init_with_both_raises_validation(self):
        with pytest.raises(ValueError, match="At least one"):
            EdgeRazorConfig(qat_config=None, kd_config=None)

    def test_init_with_qat_only(self, basic_qat_config_dict):
        from edgerazor.qat.util.quant_config import QuantConfig
        qat_cfg = QuantConfig(basic_qat_config_dict)
        edge_cfg = EdgeRazorConfig(qat_config=qat_cfg)
        assert edge_cfg.has_qat is True
        assert edge_cfg.has_kd is False

    def test_init_with_kd_only(self, basic_kd_config_dict):
        from edgerazor.kd.util.distill_config import DistillConfig
        kd_cfg = DistillConfig.from_dict(basic_kd_config_dict)
        edge_cfg = EdgeRazorConfig(kd_config=kd_cfg)
        assert edge_cfg.has_qat is False
        assert edge_cfg.has_kd is True

    def test_init_with_both(self, basic_qat_config_dict, basic_kd_config_dict):
        from edgerazor.qat.util.quant_config import QuantConfig
        from edgerazor.kd.util.distill_config import DistillConfig
        qat_cfg = QuantConfig(basic_qat_config_dict)
        kd_cfg = DistillConfig.from_dict(basic_kd_config_dict)
        edge_cfg = EdgeRazorConfig(qat_config=qat_cfg, kd_config=kd_cfg)
        assert edge_cfg.has_qat is True
        assert edge_cfg.has_kd is True

    def test_default_log_level(self, basic_qat_config_dict):
        from edgerazor.qat.util.quant_config import QuantConfig
        qat_cfg = QuantConfig(basic_qat_config_dict)
        edge_cfg = EdgeRazorConfig(qat_config=qat_cfg)
        assert edge_cfg.log_level == logging.INFO


class TestEdgeRazorConfigFromDict:
    def test_unified_format(self, unified_config_dict):
        cfg = EdgeRazorConfig.from_dict(unified_config_dict)
        assert cfg.has_qat is True
        assert cfg.has_kd is True

    def test_unified_format_log_level(self, unified_config_dict):
        d = unified_config_dict.copy()
        d["log_level"] = "DEBUG"
        cfg = EdgeRazorConfig.from_dict(d)
        assert cfg.log_level == "DEBUG"

    def test_qat_only_single_format(self, basic_qat_config_dict):
        cfg = EdgeRazorConfig.from_dict(basic_qat_config_dict)
        assert cfg.has_qat is True
        assert cfg.has_kd is False

    def test_kd_only_single_format(self, basic_kd_config_dict):
        cfg = EdgeRazorConfig.from_dict(basic_kd_config_dict)
        assert cfg.has_qat is False
        assert cfg.has_kd is True

    def test_invalid_method_in_wrapper_raises(self):
        with pytest.raises(ValueError, match="Invalid method in qat_configuration"):
            EdgeRazorConfig.from_dict({
                "qat_configuration": {"method": "KD", "function": {}, "select": {}}
            })

    def test_invalid_kd_method_in_wrapper_raises(self):
        with pytest.raises(ValueError, match="Invalid method in kd_configuration"):
            EdgeRazorConfig.from_dict({
                "kd_configuration": {"method": "QAT", "loss_1": {}}
            })


class TestEdgeRazorConfigLoad:
    def test_load_already_edge_config(self, unified_config_dict):
        cfg1 = EdgeRazorConfig.from_dict(unified_config_dict)
        cfg2 = EdgeRazorConfig.load(cfg1)
        assert cfg2 is cfg1

    def test_load_from_dict_unified(self, unified_config_dict):
        cfg = EdgeRazorConfig.load(unified_config_dict)
        assert cfg.has_qat is True
        assert cfg.has_kd is True

    def test_load_no_config_raises(self):
        with pytest.raises(ValueError, match="No configuration"):
            EdgeRazorConfig.load()

    def test_load_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported config type"):
            EdgeRazorConfig.load(42)


class TestEdgeRazorConfigSerialization:
    def test_to_dict_unified(self, unified_config_dict):
        cfg = EdgeRazorConfig.from_dict(unified_config_dict)
        d = cfg.to_dict()
        assert "qat_configuration" in d
        assert "kd_configuration" in d

    def test_to_dict_qat_only(self, basic_qat_config_dict):
        cfg = EdgeRazorConfig.from_dict(basic_qat_config_dict)
        d = cfg.to_dict()
        assert "qat_configuration" in d
        assert "kd_configuration" not in d

    def test_to_yaml(self, unified_config_dict, temp_dir):
        cfg = EdgeRazorConfig.from_dict(unified_config_dict)
        yaml_path = temp_dir / "unified.yaml"
        cfg.to_yaml(yaml_path)
        assert yaml_path.exists()

    def test_to_json(self, unified_config_dict, temp_dir):
        cfg = EdgeRazorConfig.from_dict(unified_config_dict)
        json_path = temp_dir / "unified.json"
        cfg.to_json(json_path)
        assert json_path.exists()

    def test_pickle_roundtrip(self, unified_config_dict):
        cfg = EdgeRazorConfig.from_dict(unified_config_dict)
        data = pickle.dumps(cfg)
        restored = pickle.loads(data)
        assert restored.has_qat == cfg.has_qat
        assert restored.has_kd == cfg.has_kd


class TestEdgeRazorConfigRepr:
    def test_repr_with_qat_only(self, basic_qat_config_dict):
        cfg = EdgeRazorConfig.from_dict(basic_qat_config_dict)
        r = repr(cfg)
        assert "EdgeRazorConfig" in r
        assert "QAT=enabled" in r
        assert "KD=disabled" in r

    def test_repr_with_both(self, unified_config_dict):
        cfg = EdgeRazorConfig.from_dict(unified_config_dict)
        r = repr(cfg)
        assert "QAT=enabled" in r
        assert "KD=enabled" in r
