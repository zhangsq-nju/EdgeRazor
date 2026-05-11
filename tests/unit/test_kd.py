"""Unit tests for KD class."""

import pytest
import torch

from edgerazor.kd import KD
from edgerazor.kd.util.distill_config import DistillConfig


class TestKDInit:
    def test_init_with_dict(self, basic_kd_config_dict):
        kd = KD(basic_kd_config_dict)
        assert kd.config is not None
        assert kd.config.method == "KD"

    def test_init_with_distill_config(self, basic_kd_config_dict):
        cfg = DistillConfig.from_dict(basic_kd_config_dict)
        kd = KD(cfg)
        assert kd.config is cfg

    def test_init_with_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Invalid config type"):
            KD(42)

    def test_init_with_unsupported_file_format_raises(self, temp_dir):
        txt_file = temp_dir / "config.txt"
        txt_file.write_text("not a config")
        with pytest.raises(ValueError, match="Unsupported file format"):
            KD(str(txt_file))

    def test_loss_functions_registered(self, basic_kd_config_dict):
        kd = KD(basic_kd_config_dict)
        assert "loss_1" in kd.loss_functions
        assert callable(kd.loss_functions["loss_1"])


class TestKDComputeLoss:
    def test_logits_distillation_basic(
        self, basic_kd_config_dict, dummy_student_outputs, dummy_teacher_outputs, dummy_labels
    ):
        kd = KD(basic_kd_config_dict)
        total_loss, loss_dict = kd.compute_loss(
            dummy_student_outputs, dummy_teacher_outputs, dummy_labels
        )
        assert isinstance(total_loss, torch.Tensor)
        assert "task_loss" in loss_dict
        assert "distill_loss" in loss_dict
        assert "distill_loss_details" in loss_dict
        assert len(loss_dict["distill_loss_details"]) > 0

    def test_teacher_outputs_as_tensor(
        self, basic_kd_config_dict, dummy_student_outputs, dummy_labels
    ):
        kd = KD(basic_kd_config_dict)
        teacher_logits = torch.randn(2, 4, 10)
        total_loss, loss_dict = kd.compute_loss(
            dummy_student_outputs, teacher_logits, dummy_labels
        )
        assert isinstance(total_loss, torch.Tensor)

    def test_missing_task_loss_raises(
        self, basic_kd_config_dict, dummy_teacher_outputs, dummy_labels
    ):
        kd = KD(basic_kd_config_dict)
        student_outputs = {"logits": torch.randn(2, 4, 10)}
        with pytest.raises(ValueError, match="task_loss not found"):
            kd.compute_loss(student_outputs, dummy_teacher_outputs, dummy_labels)

    def test_model_output_object(
        self, basic_kd_config_dict, dummy_labels
    ):
        kd = KD(basic_kd_config_dict)

        # Simulate a ModelOutput with attributes
        class FakeModelOutput:
            def __init__(self):
                self.loss = torch.nn.functional.cross_entropy(
                    torch.randn(8, 10), torch.randint(0, 10, (8,))
                )
                self.logits = torch.randn(2, 4, 10)

        student_out = FakeModelOutput()
        teacher_out = FakeModelOutput()

        total_loss, loss_dict = kd.compute_loss(
            student_out, teacher_out, dummy_labels
        )
        assert isinstance(total_loss, torch.Tensor)

    def test_missing_logits_warning(
        self, basic_kd_config_dict, dummy_labels
    ):
        kd = KD(basic_kd_config_dict)
        student_out = {
            "loss": torch.tensor(2.5, requires_grad=True),
            "logits": None,
        }
        teacher_out = {"logits": torch.randn(2, 4, 10)}

        total_loss, loss_dict = kd.compute_loss(
            student_out, teacher_out, dummy_labels
        )
        # distill_loss should be 0 since logits were missing
        assert loss_dict["distill_loss"] == 0.0

    def test_loss_task_alpha_scaling(
        self, basic_kd_config_dict, dummy_student_outputs, dummy_teacher_outputs, dummy_labels
    ):
        kd = KD(basic_kd_config_dict)
        _, loss_dict = kd.compute_loss(
            dummy_student_outputs, dummy_teacher_outputs, dummy_labels
        )

        # With alpha=1.0, total_loss = task_loss + distill_loss
        expected = loss_dict["task_loss"] + loss_dict["distill_loss"]
        assert abs(loss_dict["total_loss"] - expected) < 1e-4

    def test_multi_loss_kd(self, dummy_student_outputs, dummy_teacher_outputs, dummy_labels):
        config = {
            "method": "KD",
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
                "temperature": 2.0,
            },
            "loss_2": {
                "loss_type": "logits",
                "loss_function": "compute_kld_forward",
                "alpha": 0.3,
                "temperature": 4.0,
            },
        }
        kd = KD(config)
        total_loss, loss_dict = kd.compute_loss(
            dummy_student_outputs, dummy_teacher_outputs, dummy_labels
        )
        assert len(loss_dict["distill_loss_details"]) == 2


class TestKDRepr:
    def test_repr(self, basic_kd_config_dict):
        kd = KD(basic_kd_config_dict)
        r = repr(kd)
        assert "KD" in r
        assert "logits" in r
