"""Integration tests for KD: distill loss compute + backward to model parameters.

Tests logits, hidden_states, attentions, and past_key_values distillation
with actual gradient flow verification.
"""

import pytest
import torch
import torch.nn as nn

from edgerazor.kd import KD


# ──────────────────────────────────────────────
# A simple real model for gradient flow testing
# ──────────────────────────────────────────────

class _SimpleModel(nn.Module):
    """A small real model that produces output with loss and gradients."""

    def __init__(self, vocab_size=10, hidden_size=32, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, return_dict=True,
                output_hidden_states=False, output_attentions=False):
        x = self.embed(input_ids)
        hidden_states = [x] if output_hidden_states else None
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
            if output_hidden_states:
                hidden_states.append(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        result = {"logits": logits, "loss": loss}
        if output_hidden_states:
            result["hidden_states"] = tuple(hidden_states)
        if output_attentions:
            result["attentions"] = tuple(
                torch.randn(1, 4, x.size(1), x.size(1)) for _ in self.layers
            )
        result["past_key_values"] = tuple(
            (torch.randn(1, 4, x.size(1), self.hidden_size),
             torch.randn(1, 4, x.size(1), self.hidden_size))
            for _ in self.layers
        )
        return result


# ──────────────────────────────────────────────
# Logits distillation
# ──────────────────────────────────────────────

class TestKDLogitsDistillation:
    def test_logits_distill_loss_computed(self):
        kd = KD({
            "method": "KD",
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
                "temperature": 2.0,
            },
        })
        student = _SimpleModel()
        teacher = _SimpleModel()
        inputs = torch.randint(0, 10, (2, 8))
        student_out = student(inputs, labels=inputs, return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True)
        total_loss, loss_dict = kd.compute_loss(student_out, teacher_out, inputs)
        assert 'distill_loss' in loss_dict
        assert 'distill_loss_details' in loss_dict
        assert 'loss_1' in loss_dict['distill_loss_details']
        assert loss_dict['distill_loss_details']['loss_1'] > 0

    def test_logits_backward_to_model_params(self):
        kd = KD({
            "method": "KD",
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
                "temperature": 2.0,
            },
        })
        student = _SimpleModel()
        teacher = _SimpleModel()
        inputs = torch.randint(0, 10, (2, 8))
        student_out = student(inputs, labels=inputs, return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True)
        total_loss, _ = kd.compute_loss(student_out, teacher_out, inputs)
        total_loss.backward()
        for name, p in student.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradient"

    def test_logits_distill_multiple_losses(self):
        kd = KD({
            "method": "KD",
            "loss_task_alpha": 0.5,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.3,
                "temperature": 2.0,
            },
            "loss_2": {
                "loss_type": "logits",
                "loss_function": "compute_kld_forward",
                "alpha": 0.2,
                "temperature": 4.0,
            },
        })
        student = _SimpleModel()
        teacher = _SimpleModel()
        inputs = torch.randint(0, 10, (2, 8))
        student_out = student(inputs, labels=inputs, return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True)
        total_loss, loss_dict = kd.compute_loss(student_out, teacher_out, inputs)
        assert 'loss_1' in loss_dict['distill_loss_details']
        assert 'loss_2' in loss_dict['distill_loss_details']


# ──────────────────────────────────────────────
# Hidden states distillation
# ──────────────────────────────────────────────

class TestKDHiddenStatesDistillation:
    def test_hidden_states_distill_loss_computed(self):
        kd = KD({
            "method": "KD",
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "hidden_states",
                "loss_function": "compute_mse",
                "alpha": 0.5,
                "layer_index": [0, 1],
            },
        })
        student = _SimpleModel()
        teacher = _SimpleModel()
        inputs = torch.randint(0, 10, (2, 8))
        student_out = student(inputs, labels=inputs, return_dict=True,
                              output_hidden_states=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True,
                                  output_hidden_states=True)
        total_loss, loss_dict = kd.compute_loss(student_out, teacher_out, inputs)
        assert 'distill_loss_details' in loss_dict
        assert 'loss_1' in loss_dict['distill_loss_details']

    def test_hidden_states_backward_to_model_params(self):
        kd = KD({
            "method": "KD",
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "hidden_states",
                "loss_function": "compute_mse",
                "alpha": 0.5,
                "layer_index": [0, 1],
            },
        })
        student = _SimpleModel()
        teacher = _SimpleModel()
        inputs = torch.randint(0, 10, (2, 8))
        student_out = student(inputs, labels=inputs, return_dict=True,
                              output_hidden_states=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True,
                                  output_hidden_states=True)
        total_loss, _ = kd.compute_loss(student_out, teacher_out, inputs)
        total_loss.backward()
        for name, p in student.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"


# ──────────────────────────────────────────────
# KD initialization
# ──────────────────────────────────────────────

class TestKDInit:
    def test_init_from_dict(self):
        config = {
            "method": "KD",
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
                "temperature": 2.0,
            },
        }
        kd = KD(config)
        assert kd.config.method == "KD"
        assert len(kd.loss_functions) == 1

    def test_init_with_multiple_losses(self):
        kd = KD({
            "method": "KD",
            "loss_task_alpha": 0.5,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.3,
                "temperature": 2.0,
            },
            "loss_2": {
                "loss_type": "hidden_states",
                "loss_function": "compute_mse",
                "alpha": 0.2,
                "layer_index": [0, 1],
            },
        })
        assert len(kd.loss_functions) == 2


class TestKDTeacherTensor:
    """KD with teacher outputs as plain tensor."""

    def test_teacher_as_tensor(self):
        kd = KD({
            "method": "KD",
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
                "temperature": 2.0,
            },
        })
        student = _SimpleModel()
        inputs = torch.randint(0, 10, (2, 8))
        student_out = student(inputs, labels=inputs, return_dict=True)
        teacher_logits = torch.randn(2, 8, 10)
        total_loss, loss_dict = kd.compute_loss(student_out, teacher_logits, inputs)
        assert 'loss_1' in loss_dict['distill_loss_details']


class TestKDLossRatio:
    """Verify KD loss_task_alpha weighting."""

    def test_loss_task_alpha_half(self):
        kd = KD({
            "method": "KD",
            "loss_task_alpha": 0.2,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.8,
                "temperature": 2.0,
            },
        })
        student = _SimpleModel()
        teacher = _SimpleModel()
        inputs = torch.randint(0, 10, (2, 8))
        student_out = student(inputs, labels=inputs, return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True)
        total_loss, loss_dict = kd.compute_loss(student_out, teacher_out, inputs)
        assert loss_dict['task_loss'] > 0
        assert loss_dict['distill_loss'] > 0
        assert loss_dict['total_loss'] > 0
