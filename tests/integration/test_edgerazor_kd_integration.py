"""Integration tests for EdgeRazor + KD pipeline."""

import pytest
import torch

from edgerazor import EdgeRazor


class TestKDLogitsDistillation:
    """Test KD with logits-based distillation in various configurations."""

    def test_kld_reverse_distillation(self, basic_kd_config_dict):
        """Test KLD reverse (standard logits distillation)."""
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = {"loss": torch.tensor(3.0, requires_grad=True), "logits": torch.randn(2, 4, 10)}
        teacher = {"logits": torch.randn(2, 4, 10)}
        labels = torch.randint(0, 10, (2, 4))

        total_loss, loss_dict = er.compute_loss(student, teacher, labels)
        assert total_loss.item() != 3.0  # distill_loss should be added
        assert loss_dict["distill_loss"] > 0

    def test_kld_forward_distillation(self):
        config = {
            "method": "KD",
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_forward",
                "alpha": 0.5,
                "temperature": 3.0,
            },
        }
        er = EdgeRazor(kd_config=config)
        student = {"loss": torch.tensor(2.0, requires_grad=True), "logits": torch.randn(2, 4, 10)}
        teacher = {"logits": torch.randn(2, 4, 10)}
        labels = torch.randint(0, 10, (2, 4))

        total_loss, loss_dict = er.compute_loss(student, teacher, labels)
        assert "distill_loss_details" in loss_dict

    def test_kld_confidence_distillation(self):
        config = {
            "method": "KD",
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_confidence",
                "alpha": 0.7,
                "temperature": 2.0,
                "confidence_k": 3,
            },
        }
        er = EdgeRazor(kd_config=config)
        student = {"loss": torch.tensor(2.5, requires_grad=True), "logits": torch.randn(2, 4, 10)}
        teacher = {"logits": torch.randn(2, 4, 10)}
        labels = torch.randint(0, 10, (2, 4))

        total_loss, loss_dict = er.compute_loss(student, teacher, labels)
        assert isinstance(total_loss, torch.Tensor)

    def test_gradient_flows_through_distill_loss(self, basic_kd_config_dict):
        """Verify that gradients can flow back through the KD loss."""
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student_logits = torch.randn(2, 4, 10, requires_grad=True)
        task_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, 10), torch.randint(0, 10, (8,))
        )
        student = {"loss": task_loss, "logits": student_logits}
        teacher = {"logits": torch.randn(2, 4, 10)}
        labels = torch.randint(0, 10, (2, 4))

        total_loss, _ = er.compute_loss(student, teacher, labels)
        total_loss.backward()

        assert student_logits.grad is not None
        assert not torch.all(student_logits.grad == 0)


class TestKDMultiLossCombinations:
    """Test KD with multiple loss types combined."""

    def test_two_logits_losses(self):
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
        er = EdgeRazor(kd_config=config)
        student = {"loss": torch.tensor(3.0, requires_grad=True), "logits": torch.randn(2, 4, 10)}
        teacher = {"logits": torch.randn(2, 4, 10)}
        labels = torch.randint(0, 10, (2, 4))

        _, loss_dict = er.compute_loss(student, teacher, labels)
        assert len(loss_dict["distill_loss_details"]) == 2
        # distill_loss = alpha_1 * loss_1 + alpha_2 * loss_2
        details = loss_dict["distill_loss_details"]
        expected_distill = 0.5 * details["loss_1"] + 0.3 * details["loss_2"]
        assert abs(loss_dict["distill_loss"] - expected_distill) < 1e-4

    def test_task_loss_alpha_scaling(self):
        config = {
            "method": "KD",
            "loss_task_alpha": 0.5,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
                "temperature": 2.0,
            },
        }
        er = EdgeRazor(kd_config=config)
        task_loss_val = torch.tensor(4.0)
        student = {"loss": task_loss_val, "logits": torch.randn(2, 4, 10)}
        teacher = {"logits": torch.randn(2, 4, 10)}
        labels = torch.randint(0, 10, (2, 4))

        total_loss, loss_dict = er.compute_loss(student, teacher, labels)
        # total = 0.5 * task_loss + distill_loss
        expected = 0.5 * 4.0 + loss_dict["distill_loss"]
        assert abs(total_loss.item() - expected) < 1e-4


class TestKDWithModelOutputObjects:
    """Test KD handling transformers-style ModelOutput objects."""

    def test_model_output_with_distill(self, basic_kd_config_dict):
        er = EdgeRazor(kd_config=basic_kd_config_dict)

        class CausalLMOutput:
            def __init__(self):
                self.loss = torch.tensor(2.5, requires_grad=True)
                self.logits = torch.randn(2, 4, 10)
                self.hidden_states = None
                self.attentions = None
                self.past_key_values = None

        student = CausalLMOutput()
        teacher = CausalLMOutput()
        labels = torch.randint(0, 10, (2, 4))

        total_loss, loss_dict = er.compute_loss(student, teacher, labels)
        assert isinstance(total_loss, torch.Tensor)
        assert loss_dict["task_loss"] == 2.5


class TestQATPlusKDCombined:
    """Test combined QAT + KD pipeline."""

    def test_quantize_then_compute_kd_loss(
        self, basic_qat_config_dict, basic_kd_config_dict, simple_linear_model
    ):
        """Full pipeline: quantize model, then compute KD loss."""
        er = EdgeRazor(
            qat_config=basic_qat_config_dict,
            kd_config=basic_kd_config_dict,
        )
        assert er.is_qat_enabled and er.is_kd_enabled

        # Quantize the model
        quantized = er.quantize(simple_linear_model)
        assert quantized is not None

        # Compute KD loss with dummy outputs
        student = {"loss": torch.tensor(3.0, requires_grad=True), "logits": torch.randn(2, 4, 10)}
        teacher = {"logits": torch.randn(2, 4, 10)}
        labels = torch.randint(0, 10, (2, 4))

        total_loss, loss_dict = er.compute_loss(student, teacher, labels)
        assert loss_dict["distill_loss"] > 0
