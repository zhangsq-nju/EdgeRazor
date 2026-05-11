"""Unit tests for EdgeRazor main class."""

from unittest.mock import Mock, patch

import pytest
import torch

from edgerazor import EdgeRazor


class TestEdgeRazorInit:
    def test_init_with_qat_only(self, basic_qat_config_dict):
        er = EdgeRazor(qat_config=basic_qat_config_dict)
        assert er.is_qat_enabled is True
        assert er.is_kd_enabled is False

    def test_init_with_kd_only(self, basic_kd_config_dict):
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        assert er.is_qat_enabled is False
        assert er.is_kd_enabled is True

    def test_init_with_both(self, basic_qat_config_dict, basic_kd_config_dict):
        er = EdgeRazor(
            qat_config=basic_qat_config_dict,
            kd_config=basic_kd_config_dict,
        )
        assert er.is_qat_enabled is True
        assert er.is_kd_enabled is True

    def test_init_with_unified_dict(self, unified_config_dict):
        er = EdgeRazor(config=unified_config_dict)
        assert er.is_qat_enabled is True
        assert er.is_kd_enabled is True

    def test_init_with_unified_dict_log_level(self, unified_config_dict):
        d = unified_config_dict.copy()
        d["log_level"] = "DEBUG"
        er = EdgeRazor(config=d)
        assert er.is_qat_enabled is True
        assert er.is_kd_enabled is True

    def test_init_with_no_config_raises(self):
        with pytest.raises(ValueError):
            EdgeRazor()

    def test_init_with_unified_log_level_overrides_components(
        self, unified_config_dict
    ):
        d = unified_config_dict.copy()
        d["log_level"] = "WARNING"
        er = EdgeRazor(config=d)
        assert er.is_qat_enabled is True


class TestEdgeRazorComputeLoss:
    def test_compute_loss_without_kd_uses_task_loss(
        self, basic_qat_config_dict, dummy_student_outputs, dummy_labels
    ):
        er = EdgeRazor(qat_config=basic_qat_config_dict)
        assert er.is_kd_enabled is False

        total_loss, loss_dict = er.compute_loss(
            dummy_student_outputs, {}, dummy_labels
        )
        assert isinstance(total_loss, torch.Tensor)
        assert "task_loss" in loss_dict
        assert "distill_loss" in loss_dict
        assert loss_dict["distill_loss"] == 0.0

    def test_compute_loss_without_kd_missing_loss_raises(self, basic_qat_config_dict):
        er = EdgeRazor(qat_config=basic_qat_config_dict)
        with pytest.raises(ValueError, match="must contain 'loss'"):
            er.compute_loss({"logits": torch.randn(2, 3)}, {}, torch.tensor([1, 2]))

    def test_compute_loss_with_kd(
        self, basic_kd_config_dict, dummy_student_outputs, dummy_teacher_outputs, dummy_labels
    ):
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        assert er.is_kd_enabled is True

        total_loss, loss_dict = er.compute_loss(
            dummy_student_outputs, dummy_teacher_outputs, dummy_labels
        )
        assert isinstance(total_loss, torch.Tensor)
        assert "distill_loss" in loss_dict
        assert "distill_loss_details" in loss_dict
        assert "total_loss" in loss_dict

    def test_compute_loss_with_kd_teacher_as_tensor(
        self, basic_kd_config_dict, dummy_student_outputs, dummy_labels
    ):
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        teacher_logits = torch.randn(2, 4, 10)

        total_loss, loss_dict = er.compute_loss(
            dummy_student_outputs, teacher_logits, dummy_labels
        )
        assert isinstance(total_loss, torch.Tensor)


class TestEdgeRazorQuantize:
    def test_quantize_without_qat_returns_model_unchanged(
        self, basic_kd_config_dict, simple_linear_model
    ):
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        result = er.quantize(simple_linear_model)
        assert result is simple_linear_model

    def test_quantize_with_qat_calls_qat_quantize(
        self, basic_qat_config_dict, simple_linear_model
    ):
        er = EdgeRazor(qat_config=basic_qat_config_dict)
        # The model after quantize should have QLinear modules
        result = er.quantize(simple_linear_model)
        # The fc layer should now be a QLinear
        from edgerazor.qat.module import QLinear
        assert isinstance(result.fc, QLinear)

    def test_replace_quantized_weights_without_qat(
        self, basic_kd_config_dict, simple_linear_model
    ):
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        result = er.replace_quantized_weights(simple_linear_model)
        assert result is simple_linear_model


class TestEdgeRazorProperties:
    def test_is_qat_enabled_true(self, basic_qat_config_dict):
        er = EdgeRazor(qat_config=basic_qat_config_dict)
        assert er.is_qat_enabled is True

    def test_is_kd_enabled_false(self, basic_qat_config_dict):
        er = EdgeRazor(qat_config=basic_qat_config_dict)
        assert er.is_kd_enabled is False

    def test_is_qat_enabled_false(self, basic_kd_config_dict):
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        assert er.is_qat_enabled is False

    def test_is_kd_enabled_true(self, basic_kd_config_dict):
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        assert er.is_kd_enabled is True


class TestEdgeRazorRepr:
    def test_repr_with_qat_only(self, basic_qat_config_dict):
        er = EdgeRazor(qat_config=basic_qat_config_dict)
        r = repr(er)
        assert "EdgeRazor" in r
        assert "QAT=enabled" in r
        assert "KD=disabled" in r

    def test_repr_with_both(self, basic_qat_config_dict, basic_kd_config_dict):
        er = EdgeRazor(
            qat_config=basic_qat_config_dict,
            kd_config=basic_kd_config_dict,
        )
        r = repr(er)
        assert "QAT=enabled" in r
        assert "KD=enabled" in r
