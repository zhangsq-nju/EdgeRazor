"""End-to-end tests for EdgeRazorCausalLMTrainer.

Simulates complete training workflows using the trainer API:
config -> EdgeRazor -> trainer -> compute_loss -> backward -> optimizer step.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Test models (3-D input: batch, seq, hidden)
# ---------------------------------------------------------------------------

class _CausalLMStudent(nn.Module):
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
        out.hidden_states = (
            tuple(torch.randn(logits.shape[0], logits.shape[1], self.hidden)
                  for _ in range(4))
            if output_hidden_states else None
        )
        out.attentions = None
        out.past_key_values = None
        out.router_logits = None
        out.aux_loss = None
        return out


class _CausalLMTeacher(nn.Module):
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
        out.hidden_states = (
            tuple(torch.randn(logits.shape[0], logits.shape[1], self.hidden)
                  for _ in range(4))
            if output_hidden_states else None
        )
        out.attentions = None
        out.past_key_values = None
        return out


# ---------------------------------------------------------------------------
# Helper: run a manual training step through the trainer
# ---------------------------------------------------------------------------

def _training_step(trainer, model, inputs, optimizer):
    """Simulate a single training step using the trainer's compute_loss."""
    loss = trainer.compute_loss(model, inputs, return_outputs=False)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# ---------------------------------------------------------------------------
# KD full pipeline
# ---------------------------------------------------------------------------

class TestKDFullPipeline:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_training_step(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _CausalLMStudent()
        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = _training_step(trainer, student, inputs, optimizer)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert "train/loss_dist" in trainer.custom_losses
        assert trainer.custom_losses["train/loss_dist"] > 0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_multi_step_training(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _CausalLMStudent()
        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        losses = []
        for _ in range(3):
            inputs = {
                "input_ids": torch.randn(2, 4, 32),
                "attention_mask": torch.ones(2, 4),
                "labels": torch.randint(0, 10, (2, 4)),
            }
            loss = _training_step(trainer, student, inputs, optimizer)
            losses.append(loss.item())

        # Losses should change (not all identical — model is training)
        assert len(set(losses)) == 3

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_with_multi_loss_e2e(self, mock_super):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        config = {
            "method": "KD",
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_confidence",
                "alpha": 0.5,
                "temperature": 2.0,
                "confidence_k": 5,
            },
        }
        er = EdgeRazor(kd_config=config)
        student = _CausalLMStudent()
        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = _training_step(trainer, student, inputs, optimizer)
        assert loss.item() > 0
        assert student.fc.weight.grad is not None
        # Gradient should have been applied by optimizer.step()
        weight_after = student.fc.weight.clone()

        # Another step should move weights
        inputs2 = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }
        _training_step(trainer, student, inputs2, optimizer)
        assert not torch.allclose(weight_after, student.fc.weight)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_teacher_not_trained(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _CausalLMStudent()
        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        teacher_weight_before = teacher.fc.weight.clone()

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        _training_step(trainer, student, inputs, optimizer)

        # Teacher weights must not change
        assert torch.allclose(teacher.fc.weight, teacher_weight_before)


# ---------------------------------------------------------------------------
# QAT + KD combined pipeline
# ---------------------------------------------------------------------------

class TestQATKDFullPipeline:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_qat_kd_setup(self, mock_super, basic_qat_config_dict, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer
        from edgerazor.qat.module import QLinear

        er = EdgeRazor(
            qat_config=basic_qat_config_dict, kd_config=basic_kd_config_dict
        )

        model = nn.Sequential(nn.Linear(32, 10))
        teacher = _CausalLMTeacher(vocab=10, hidden=32)
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, model=model, teacher_model=teacher, auto_prepare=True
        )

        assert mock_super.called
        passed_model = mock_super.call_args[1].get("model")
        assert isinstance(passed_model[0], QLinear)
        assert trainer.edgerazor.is_qat_enabled is True
        assert trainer.edgerazor.is_kd_enabled is True
        assert trainer.teacher_model is teacher

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_qat_kd_config_resolution(
        self, mock_super, basic_qat_config_dict, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(
            qat_config=basic_qat_config_dict,
            kd_config=basic_kd_config_dict,
            teacher_model=teacher,
            auto_prepare=False,
        )

        assert trainer.edgerazor.is_qat_enabled is True
        assert trainer.edgerazor.is_kd_enabled is True


# ---------------------------------------------------------------------------
# MoE pipeline
# ---------------------------------------------------------------------------

class _MoECausalLMStudent(_CausalLMStudent):
    """Student model that emits MoE router_logits and aux_loss."""

    def __init__(self, vocab=10, hidden=32, num_experts=8):
        super().__init__(vocab=vocab, hidden=hidden)
        self.num_experts = num_experts

    def forward(self, input_ids, attention_mask=None, labels=None,
                return_dict=True, output_hidden_states=False, **kwargs):
        logits = self.fc(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        ) if labels is not None else None

        out = type("MoeCausalLMOutput", (), {})()
        out.loss = loss
        out.logits = logits
        out.hidden_states = (
            tuple(torch.randn(logits.shape[0], logits.shape[1], self.hidden)
                  for _ in range(4))
            if output_hidden_states else None
        )
        out.attentions = None
        out.past_key_values = None
        # MoE outputs: one router_logits per layer, with aux_loss
        batch_seq = logits.shape[0] * logits.shape[1]
        out.router_logits = (
            torch.randn(batch_seq, self.num_experts),
            torch.randn(batch_seq, self.num_experts),
        )
        out.aux_loss = torch.tensor(0.15)
        return out


class TestMoEPipeline:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_moe_losses_in_training_step(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _MoECausalLMStudent()
        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, teacher_model=teacher,
            router_aux_loss_coef=0.01, router_z_loss_coef=0.001,
        )

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)

        # MoE losses should be present in custom_losses
        assert "train/aux_loss" in trainer.custom_losses
        assert "train/router_z_loss" in trainer.custom_losses
        assert trainer.custom_losses["train/aux_loss"] > 0
        assert trainer.custom_losses["train/router_z_loss"] > 0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_moe_gradient_flow(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _MoECausalLMStudent()
        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, teacher_model=teacher,
            router_aux_loss_coef=0.01, router_z_loss_coef=0.001,
        )

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)
        loss.backward()

        assert student.fc.weight.grad is not None
        assert not torch.all(student.fc.weight.grad == 0)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_moe_coefficients_scale_loss(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        student1 = _MoECausalLMStudent()
        student2 = _MoECausalLMStudent()
        # Copy weights so initial loss is identical
        student2.load_state_dict(student1.state_dict())

        teacher = _CausalLMTeacher()
        teacher.eval()

        er = EdgeRazor(kd_config=basic_kd_config_dict)

        trainer1 = EdgeRazorCausalLMTrainer(
            edgerazor=er, teacher_model=teacher,
            router_aux_loss_coef=0.0, router_z_loss_coef=0.0,
        )
        trainer2 = EdgeRazorCausalLMTrainer(
            edgerazor=er, teacher_model=teacher,
            router_aux_loss_coef=0.5, router_z_loss_coef=0.1,
        )

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss1 = trainer1.compute_loss(student1, inputs, return_outputs=False)
        loss2 = trainer2.compute_loss(student2, inputs, return_outputs=False)

        # With MoE coefficients > 0, loss2 should be larger
        assert loss2.item() > loss1.item()


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------

class TestCheckpointCompatibility:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_state_dict_save_load(self, mock_super, basic_kd_config_dict, temp_dir):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _CausalLMStudent()
        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        # Run one step to get non-initial weights
        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        _training_step(trainer, student, inputs, optimizer)

        # Save state
        save_path = temp_dir / "checkpoint.pt"
        torch.save({
            "student_state_dict": student.state_dict(),
            "custom_losses": trainer.custom_losses,
        }, save_path)

        # Load into a fresh student
        student2 = _CausalLMStudent()
        checkpoint = torch.load(save_path)
        student2.load_state_dict(checkpoint["student_state_dict"])

        # Verify weights match
        for p1, p2 in zip(student.parameters(), student2.parameters()):
            assert torch.allclose(p1, p2)

        # Verify custom_losses were saved correctly
        assert checkpoint["custom_losses"]["train/loss_total"] > 0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_trainer_config_survives_roundtrip(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        teacher = _CausalLMTeacher()
        teacher.eval()

        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, teacher_model=teacher,
            router_aux_loss_coef=0.02, router_z_loss_coef=0.002,
        )

        # Verify all config persisted
        assert trainer.edgerazor is er
        assert trainer.teacher_model is teacher
        assert trainer.router_aux_loss_coef == 0.02
        assert trainer.router_z_loss_coef == 0.002
        assert trainer.edgerazor.is_kd_enabled is True


# ---------------------------------------------------------------------------
# Eval mode shortcut pipeline
# ---------------------------------------------------------------------------

class TestEvalModePipeline:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_eval_mode_skips_teacher_forward(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _CausalLMStudent()
        teacher = _CausalLMTeacher()

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
        assert trainer.custom_losses["train/loss_dist"] == 0.0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_eval_mode_no_grad_on_teacher(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _CausalLMStudent()
        teacher = _CausalLMTeacher()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        student.eval()
        teacher_weight_before = teacher.fc.weight.clone()

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(student, inputs, return_outputs=False)
        loss.backward()

        # Teacher weights must not change in eval mode
        assert torch.allclose(teacher.fc.weight, teacher_weight_before)
        assert teacher.fc.weight.grad is None

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_train_mode_runs_full_kd_gradient_flow(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        student = _CausalLMStudent()
        teacher = _CausalLMTeacher()

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        # Train mode
        student.train()

        inputs = {
            "input_ids": torch.randn(2, 4, 32),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = _training_step(trainer, student, inputs, optimizer)
        assert trainer.custom_losses["train/loss_dist"] > 0
        assert student.fc.weight.grad is not None
        assert not torch.all(student.fc.weight.grad == 0)
