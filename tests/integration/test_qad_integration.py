"""Integration tests for QAD (QAT + KD combined).

Tests that distill loss can correctly backward through quantized weights.
This is the critical path: KD loss → STE → weight gradients in quantized model.
"""

import pytest
import torch
import torch.nn as nn

from edgerazor import EdgeRazor
from edgerazor.qat.module import QEmbedding, QLinear


class _SimpleModel(nn.Module):
    """A small real model for QAD testing."""

    def __init__(self, vocab_size=10, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, return_dict=True,
                output_hidden_states=False):
        x = self.embed(input_ids)
        h = torch.relu(self.fc1(x))
        logits = self.fc2(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        result = {"logits": logits, "loss": loss}
        if output_hidden_states:
            result["hidden_states"] = (x, h)
        return result


# ──────────────────────────────────────────────
# QAD: QAT + KD combined
# ──────────────────────────────────────────────

class TestQADCombined:
    """KD distill loss backward through quantized weights."""

    def test_kd_loss_backward_through_quantized_weights(self):
        """Critical test: KD loss must flow gradients through quantized layers."""
        edgerazor = EdgeRazor(
            qat_config={
                "method": "QAT",
                "select": {
                    "target_types": ["linear", "embedding"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": [],
                },
                "function": {
                    "epsilon": 1e-5,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                    "w_scale_factor": 2.0,
                    "w_block_size": 64,
                    "w_mixed_precision_prop": -1.0,
                    "is_w_quantized": False,
                    "activation_function": "",
                    "a_block_size": -1,
                    "a_mixed_precision_prop": -1.0,
                    "kv_cache_function": "",
                    "kv_block_size": -1,
                    "kv_mixed_precision_prop": -1.0,
                },
                "training": "all",
            },
            kd_config={
                "method": "KD",
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "compute_kld_reverse",
                    "alpha": 0.5,
                    "temperature": 2.0,
                },
            },
        )

        student = _SimpleModel()
        teacher = _SimpleModel()

        # Quantize student model
        q_student = edgerazor.quantize(student)
        q_student.train()
        assert isinstance(q_student.embed, QEmbedding)
        assert isinstance(q_student.fc1, QLinear)
        assert isinstance(q_student.fc2, QLinear)

        # Capture weights before backward
        params_before = {n: p.clone() for n, p in q_student.named_parameters()}

        inputs = torch.randint(0, 10, (2, 8))
        student_out = q_student(inputs, labels=inputs, return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True)

        total_loss, loss_dict = edgerazor.compute_loss(student_out, teacher_out, inputs)
        total_loss.backward()

        # Verify gradients exist in quantized layers
        for name, p in q_student.named_parameters():
            assert p.grad is not None, f"{name} has no gradient in QAD mode"

    def test_optimization_updates_quantized_weights_with_kd(self):
        """Full training step with QAD: forward → KD loss → backward → update."""
        edgerazor = EdgeRazor(
            qat_config={
                "method": "QAT",
                "select": {
                    "target_types": ["linear", "embedding"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": [],
                },
                "function": {
                    "epsilon": 1e-5,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                    "w_scale_factor": 2.0,
                    "w_block_size": 64,
                    "w_mixed_precision_prop": -1.0,
                    "is_w_quantized": False,
                    "activation_function": "",
                    "a_block_size": -1,
                    "a_mixed_precision_prop": -1.0,
                    "kv_cache_function": "",
                    "kv_block_size": -1,
                    "kv_mixed_precision_prop": -1.0,
                },
                "training": "all",
            },
            kd_config={
                "method": "KD",
                "loss_task_alpha": 0.5,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "compute_kld_reverse",
                    "alpha": 0.5,
                    "temperature": 2.0,
                },
            },
        )

        student = _SimpleModel()
        teacher = _SimpleModel()

        q_student = edgerazor.quantize(student)
        q_student.train()

        optimizer = torch.optim.SGD(q_student.parameters(), lr=0.01)
        params_before = {n: p.clone() for n, p in q_student.named_parameters() if p.requires_grad}

        for _ in range(3):
            optimizer.zero_grad()
            inputs = torch.randint(0, 10, (2, 8))
            student_out = q_student(inputs, labels=inputs, return_dict=True)
            with torch.no_grad():
                teacher_out = teacher(inputs, return_dict=True)
            total_loss, _ = edgerazor.compute_loss(student_out, teacher_out, inputs)
            total_loss.backward()
            optimizer.step()

        # Verify parameters changed
        for n, p in q_student.named_parameters():
            if p.requires_grad:
                assert not torch.equal(p, params_before[n]), f"{n} did not change"

    def test_replace_and_save_after_qad_training(self, temp_dir):
        """After QAD training, replace weights and save."""
        edgerazor = EdgeRazor(
            qat_config={
                "method": "QAT",
                "select": {
                    "target_types": ["linear", "embedding"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": [],
                },
                "function": {
                    "epsilon": 1e-5,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                    "w_scale_factor": 2.0,
                    "w_block_size": 64,
                    "w_mixed_precision_prop": -1.0,
                    "is_w_quantized": False,
                    "activation_function": "",
                    "a_block_size": -1,
                    "a_mixed_precision_prop": -1.0,
                    "kv_cache_function": "",
                    "kv_block_size": -1,
                    "kv_mixed_precision_prop": -1.0,
                },
                "training": "all",
            },
            kd_config={
                "method": "KD",
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "compute_kld_reverse",
                    "alpha": 0.5,
                    "temperature": 2.0,
                },
            },
        )

        student = _SimpleModel()
        q_student = edgerazor.quantize(student)
        q_student.train()

        # One training step
        inputs = torch.randint(0, 10, (2, 8))
        student_out = q_student(inputs, labels=inputs, return_dict=True)
        with torch.no_grad():
            teacher_out = _SimpleModel()(inputs, return_dict=True)
        total_loss, _ = edgerazor.compute_loss(student_out, teacher_out, inputs)
        total_loss.backward()

        # Replace weights and save
        edgerazor.replace_quantized_weights(q_student)
        assert q_student.embed.is_w_quantized
        assert q_student.fc1.is_w_quantized
        assert q_student.fc2.is_w_quantized

        save_path = temp_dir / "qad_model.pt"
        torch.save(q_student.state_dict(), save_path)

        # Load back and verify
        new_student = _SimpleModel()
        new_q = edgerazor.quantize(new_student)
        new_q.load_state_dict(torch.load(save_path))
        torch.testing.assert_close(new_q.embed.weight, q_student.embed.weight)
        torch.testing.assert_close(new_q.fc1.weight, q_student.fc1.weight)
        torch.testing.assert_close(new_q.fc2.weight, q_student.fc2.weight)


class TestQADHiddenStatesDistillation:
    """QAD with hidden_states distillation through quantized weights."""

    def test_hidden_states_kd_through_quantized(self):
        edgerazor = EdgeRazor(
            qat_config={
                "method": "QAT",
                "select": {
                    "target_types": ["linear", "embedding"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": [],
                },
                "function": {
                    "epsilon": 1e-5,
                    "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    "w_block_size": 256,
                    "w_mixed_precision_prop": -1.0,
                    "is_w_quantized": False,
                    "activation_function": "",
                    "a_block_size": -1,
                    "a_mixed_precision_prop": -1.0,
                    "kv_cache_function": "",
                    "kv_block_size": -1,
                    "kv_mixed_precision_prop": -1.0,
                },
                "training": "all",
            },
            kd_config={
                "method": "KD",
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "hidden_states",
                    "loss_function": "compute_mse",
                    "alpha": 0.3,
                    "layer_index": [0, 1],
                },
            },
        )

        student = _SimpleModel()
        teacher = _SimpleModel()
        q_student = edgerazor.quantize(student)
        q_student.train()

        inputs = torch.randint(0, 10, (2, 8))
        student_out = q_student(inputs, labels=inputs, return_dict=True,
                                output_hidden_states=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True,
                                  output_hidden_states=True)
        total_loss, loss_dict = edgerazor.compute_loss(student_out, teacher_out, inputs)
        total_loss.backward()

        for name, p in q_student.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"


class TestQADMixedPrecision:
    """QAD with mixed-precision quantization + KD."""

    def test_mixed_precision_with_kd(self):
        edgerazor = EdgeRazor(
            qat_config={
                "method": "QAT",
                "select": {
                    "target_types": ["linear"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": [],
                },
                "function": {
                    "epsilon": 1e-5,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                    "w_scale_factor": 2.0,
                    "w_block_size": 64,
                    "w_mixed_precision_prop": 0.1,
                    "is_w_quantized": False,
                    "activation_function": "",
                    "a_block_size": -1,
                    "a_mixed_precision_prop": -1.0,
                    "kv_cache_function": "",
                    "kv_block_size": -1,
                    "kv_mixed_precision_prop": -1.0,
                },
                "training": "all",
            },
            kd_config={
                "method": "KD",
                "loss_task_alpha": 0.5,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "compute_kld_reverse",
                    "alpha": 0.5,
                    "temperature": 2.0,
                },
            },
        )

        student = _SimpleModel()
        teacher = _SimpleModel()
        q_student = edgerazor.quantize(student)
        q_student.train()

        inputs = torch.randint(0, 10, (2, 8))
        student_out = q_student(inputs, labels=inputs, return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True)
        total_loss, _ = edgerazor.compute_loss(student_out, teacher_out, inputs)
        total_loss.backward()

        for name, p in q_student.named_parameters():
            assert p.grad is not None
