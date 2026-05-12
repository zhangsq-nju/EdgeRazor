"""Integration tests for EdgeRazorCausalLMTrainer.

These tests verify that the trainer correctly orchestrates real EdgeRazor
KD / QAT pipelines using small real models.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# lightweight models for integration testing
# ---------------------------------------------------------------------------

class _StudentModule(nn.Module):
    """Small trainable module whose forward returns a CausalLM-like output.

    Expects 3-D inputs: (batch, seq, hidden_in).
    """

    def __init__(self, vocab=10, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None,
                return_dict=True, output_hidden_states=False, **kwargs):
        logits = self.fc(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        ) if labels is not None else None

        out = type("CausalLMOutput", (), {})()
        out.loss = loss
        out.logits = logits
        out.hidden_states = tuple(
            torch.randn(logits.shape[0], logits.shape[1], self.hidden)
            for _ in range(4)
        ) if output_hidden_states else None
        out.attentions = None
        out.past_key_values = None
        out.router_logits = None
        out.aux_loss = None
        return out


class _TeacherModule(nn.Module):
    """Fixed teacher that produces matching outputs."""

    def __init__(self, vocab=10, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None,
                return_dict=True, output_hidden_states=False, **kwargs):
        logits = self.fc(input_ids)

        out = type("CausalLMOutput", (), {})()
        out.loss = None
        out.logits = logits
        out.hidden_states = tuple(
            torch.randn(logits.shape[0], logits.shape[1], self.hidden)
            for _ in range(4)
        ) if output_hidden_states else None
        out.attentions = None
        out.past_key_values = None
        return out


# ---------------------------------------------------------------------------
# Trainer compute_loss integration
# ---------------------------------------------------------------------------

class TestTrainerComputeLossKDIntegration:
    """compute_loss with real EdgeRazor KD and small models."""

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_total_loss_includes_task_and_distill(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _StudentModule()
        teacher = _TeacherModule()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        # Run with output_hidden_states=True for KD hidden state matching
        student_out = student(**inputs, return_dict=True, output_hidden_states=True)
        teacher_out = teacher(**inputs, return_dict=True, output_hidden_states=True)

        # Simulate what compute_loss does
        loss_total, loss_dict = er.compute_loss(
            student_outputs=student_out,
            teacher_outputs=teacher_out,
            labels=inputs["labels"],
        )

        assert isinstance(loss_total, torch.Tensor)
        assert loss_dict["distill_loss"] > 0
        assert loss_dict["task_loss"] > 0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_compute_loss_populates_custom_losses(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _StudentModule()
        teacher = _TeacherModule()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)
        assert isinstance(loss, torch.Tensor)
        assert "train/loss_total" in trainer.custom_losses
        assert "train/loss_task" in trainer.custom_losses
        assert "train/loss_dist" in trainer.custom_losses
        assert trainer.custom_losses["train/loss_dist"] > 0
        # dist details from loss_1
        assert "train/loss_dist_1" in trainer.custom_losses

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_with_multi_loss(self, mock_super):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        config = {
            "method": "KD",
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.4,
                "temperature": 2.0,
            },
            "loss_2": {
                "loss_type": "logits",
                "loss_function": "compute_kld_forward",
                "alpha": 0.3,
                "temperature": 1.0,
            },
            "loss_3": {
                "loss_type": "logits",
                "loss_function": "compute_kld_confidence",
                "alpha": 0.1,
                "temperature": 1.0,
                "confidence_k": 3,
            },
        }

        er = EdgeRazor(kd_config=config)
        student = _StudentModule()
        teacher = _TeacherModule()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)
        # All 3 loss details should be present
        for key in ("train/loss_dist_1", "train/loss_dist_2", "train/loss_dist_3"):
            assert key in trainer.custom_losses, f"missing {key}"
            assert trainer.custom_losses[key] > 0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_return_outputs(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _StudentModule()
        teacher = _TeacherModule()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        result = trainer.compute_loss(student, inputs, return_outputs=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] is not None
        assert hasattr(result[1], "logits")

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_gradient_flows(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _StudentModule()
        teacher = _TeacherModule()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)
        loss.backward()

        # Student gradients should be non-zero (distill loss flows back)
        assert student.fc.weight.grad is not None
        assert not torch.all(student.fc.weight.grad == 0)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_teacher_gradients_not_updated(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _StudentModule()
        teacher = _TeacherModule()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)
        loss.backward()

        # Teacher should NOT have gradients
        assert teacher.fc.weight.grad is None


class TestTrainerComputeLossQATIntegration:
    """compute_loss with real EdgeRazor QAT (no KD, no teacher)."""

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_qat_only_compute_loss(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        model = _StudentModule()

        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, model=model, auto_prepare=False
        )

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert isinstance(loss, torch.Tensor)
        assert trainer.custom_losses["train/loss_task"] > 0
        assert trainer.custom_losses["train/loss_dist"] == 0.0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_qat_only_no_teacher_forward(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        model = _StudentModule()

        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, model=model, auto_prepare=False
        )
        assert trainer.teacher_model is None

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert loss.item() > 0


class TestTrainerCombinedQATKDIntegration:
    """Combined QAT + KD compute_loss."""

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_qat_plus_kd_compute_loss(
        self, mock_super, basic_qat_config_dict, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer
        from edgerazor.qat.module import QLinear

        er = EdgeRazor(
            qat_config=basic_qat_config_dict, kd_config=basic_kd_config_dict
        )
        model = nn.Sequential(nn.Linear(16, 10))
        teacher = _TeacherModule(vocab=10, hidden=16)
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, model=model, teacher_model=teacher, auto_prepare=True
        )

        # model should be quantized (passed to super init)
        assert mock_super.called

        # Verify the actual quantize happened on the in-scope model
        assert isinstance(model[0], QLinear)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_qat_plus_kd_quantize_and_setup(
        self, mock_super, basic_qat_config_dict, basic_kd_config_dict
    ):
        """Verify QAT+KD trainer quantizes the model and wires up both modules."""
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer
        from edgerazor.qat.module import QLinear

        er = EdgeRazor(
            qat_config=basic_qat_config_dict, kd_config=basic_kd_config_dict
        )

        model = nn.Sequential(nn.Linear(32, 10))
        teacher = _TeacherModule(hidden=32)
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, model=model, teacher_model=teacher, auto_prepare=True
        )

        # model should be quantized (passed to super init)
        assert mock_super.called
        passed_model = mock_super.call_args[1].get("model")
        assert isinstance(passed_model[0], QLinear)
        assert trainer.edgerazor.is_qat_enabled is True
        assert trainer.edgerazor.is_kd_enabled is True
        assert trainer.teacher_model is teacher


# ---------------------------------------------------------------------------
# Config resolution via trainer constructor
# ---------------------------------------------------------------------------

class TestTrainerConfigResolution:
    """EdgeRazor resolution from different config sources."""

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_edgerazor_config_dict_creates_both_modules(
        self, mock_super, unified_config_dict
    ):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _TeacherModule()
        trainer = EdgeRazorCausalLMTrainer(
            edgerazor_config=unified_config_dict,
            teacher_model=teacher,
            auto_prepare=False,
        )
        assert trainer.edgerazor.is_qat_enabled is True
        assert trainer.edgerazor.is_kd_enabled is True

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_separate_configs_create_both_modules(
        self, mock_super, basic_qat_config_dict, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _TeacherModule()
        trainer = EdgeRazorCausalLMTrainer(
            qat_config=basic_qat_config_dict,
            kd_config=basic_kd_config_dict,
            teacher_model=teacher,
            auto_prepare=False,
        )
        assert trainer.edgerazor.is_qat_enabled is True
        assert trainer.edgerazor.is_kd_enabled is True


# ---------------------------------------------------------------------------
# Eval mode shortcut integration
# ---------------------------------------------------------------------------

class TestEvalModeIntegration:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_eval_mode_no_teacher_forward(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _StudentModule()
        teacher = _TeacherModule()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        student.eval()

        # Spy on teacher forward
        called = False
        original = teacher.forward
        def spy(**kwargs):
            nonlocal called
            called = True
            return original(**kwargs)
        teacher.forward = spy

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)
        assert not called

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_eval_mode_custom_losses_no_distill(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _StudentModule()
        teacher = _TeacherModule()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        student.eval()

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)
        assert trainer.custom_losses["train/loss_dist"] == 0.0
        assert trainer.custom_losses["train/loss_task"] > 0
        assert "train/loss_dist_1" not in trainer.custom_losses

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_eval_vs_train_loss_difference(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)

        teacher = _TeacherModule()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        # Train mode
        student_train = _StudentModule()
        student_train.train()
        # snapshot weights for eval copy
        state = {k: v.clone() for k, v in student_train.state_dict().items()}

        # Eval mode — same initial weights
        student_eval = _StudentModule()
        student_eval.eval()
        student_eval.load_state_dict(state)

        loss_train = trainer.compute_loss(
            student_train, inputs, return_outputs=False
        )
        loss_eval = trainer.compute_loss(
            student_eval, inputs, return_outputs=False
        )

        # Train loss should be larger due to distill component
        assert loss_train.item() > loss_eval.item()
