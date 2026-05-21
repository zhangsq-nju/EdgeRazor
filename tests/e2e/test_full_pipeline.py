"""End-to-end tests for the full EdgeRazor pipeline.

Tests the complete lifecycle:
1. Build model + config
2. Quantize (QAT)
3. Training loop with KD (QAD)
4. Replace weights
5. Save model
6. Load model
7. Inference

These are the highest-level tests that verify the framework works
as an integrated whole, not just individual components.
"""

import json

import pytest
import torch
import torch.nn as nn

from edgerazor import EdgeRazor
from edgerazor.qat.module import QEmbedding, QLinear


# ──────────────────────────────────────────────
# Test model
# ──────────────────────────────────────────────

class _E2EModel(nn.Module):
    """A small but realistic model for E2E testing."""

    def __init__(self, vocab_size=100, hidden_size=64, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, return_dict=True,
                output_hidden_states=False):
        x = self.embed(input_ids)
        hidden_states = [x] if output_hidden_states else None
        for layer in self.layers:
            x = layer(x)
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
        return result


# ──────────────────────────────────────────────
# E2E: QAT only
# ──────────────────────────────────────────────

class TestE2EQATOnly:
    def test_qat_full_lifecycle(self, temp_dir):
        """Full QAT lifecycle: quantize → train → replace → save → load → infer."""
        config = {
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
        }

        er = EdgeRazor(qat_config=config)
        model = _E2EModel()
        original_param_count = sum(p.numel() for p in model.parameters())

        # 1. Quantize
        q_model = er.quantize(model)
        assert isinstance(q_model.embed, QEmbedding)
        for layer in q_model.layers:
            assert isinstance(layer[0], QLinear)
        assert isinstance(q_model.lm_head, QLinear)
        assert sum(p.numel() for p in q_model.parameters()) == original_param_count

        # 2. Train
        q_model.train()
        optimizer = torch.optim.Adam(q_model.parameters(), lr=0.001)
        pretrain_params = {n: p.clone() for n, p in q_model.named_parameters()}
        for _ in range(5):
            optimizer.zero_grad()
            inputs = torch.randint(0, 100, (4, 16))
            out = q_model(inputs, labels=inputs, return_dict=True)
            out['loss'].backward()
            optimizer.step()
        for n, p in q_model.named_parameters():
            assert not torch.equal(p, pretrain_params[n]), f"{n} did not train"

        # 3. Replace weights
        er.replace_quantized_weights(q_model)
        assert q_model.embed.is_w_quantized

        # 4. Save
        save_path = temp_dir / "e2e_qat.pt"
        torch.save(q_model.state_dict(), save_path)

        # 5. Load into new model
        new_model = _E2EModel()
        new_q = er.quantize(new_model)
        new_q.load_state_dict(torch.load(save_path))
        torch.testing.assert_close(new_q.embed.weight, q_model.embed.weight)
        torch.testing.assert_close(new_q.lm_head.weight, q_model.lm_head.weight)

        # 6. Inference
        new_q.eval()
        with torch.no_grad():
            test_input = torch.randint(0, 100, (2, 8))
            out = new_q(test_input, return_dict=True)
        assert out['logits'].shape == (2, 8, 100)
        assert not torch.isnan(out['logits']).any()
        assert not torch.isinf(out['logits']).any()


# ──────────────────────────────────────────────
# E2E: QAT + KD (QAD)
# ──────────────────────────────────────────────

class TestE2EQAD:
    def test_qad_full_lifecycle(self, temp_dir):
        """Full QAD lifecycle: quantize → train with KD → replace → save → load → infer."""
        er = EdgeRazor(
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

        student = _E2EModel()
        teacher = _E2EModel()

        # 1. Quantize
        q_student = er.quantize(student)
        q_student.train()
        assert isinstance(q_student.embed, QEmbedding)

        # 2. Train with KD
        optimizer = torch.optim.Adam(q_student.parameters(), lr=0.001)
        pretrain_params = {n: p.clone() for n, p in q_student.named_parameters()}

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            inputs = torch.randint(0, 100, (4, 16))
            student_out = q_student(inputs, labels=inputs, return_dict=True)
            with torch.no_grad():
                teacher_out = teacher(inputs, return_dict=True)
            total_loss, loss_dict = er.compute_loss(student_out, teacher_out, inputs)
            total_loss.backward()
            optimizer.step()
            losses.append(loss_dict['total_loss'])

        assert len(losses) == 5
        for n, p in q_student.named_parameters():
            assert not torch.equal(p, pretrain_params[n]), f"{n} did not train"

        # 3. Replace
        er.replace_quantized_weights(q_student)

        # 4. Save
        save_path = temp_dir / "e2e_qad.pt"
        torch.save(q_student.state_dict(), save_path)

        # 5. Load
        new_model = _E2EModel()
        new_q = er.quantize(new_model)
        new_q.load_state_dict(torch.load(save_path))

        # 6. Inference
        new_q.eval()
        with torch.no_grad():
            out = new_q(torch.randint(0, 100, (2, 8)), return_dict=True)
        assert out['logits'].shape == (2, 8, 100)
        assert not torch.isnan(out['logits']).any()


# ──────────────────────────────────────────────
# E2E: JSON config save/load round-trip
# ──────────────────────────────────────────────

class TestE2EConfigRoundTrip:
    def test_config_json_round_trip(self, temp_dir):
        """Save config to JSON, load back, verify same behavior."""
        original_config = {
            "method": "QAT",
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": [],
            },
            "function": {
                "epsilon": 1e-5,
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                "w_scale_factor": 2.0,
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
        }

        # Save
        config_path = temp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(original_config, f)

        # Load and apply
        er = EdgeRazor(qat_config=str(config_path))
        model = nn.Sequential(nn.Linear(16, 8))
        q_model = er.quantize(model)
        assert isinstance(q_model[0], QLinear)


# ──────────────────────────────────────────────
# E2E: Gradient through full QAD pipeline
# ──────────────────────────────────────────────

class TestE2EGradientFlow:
    def test_all_parameters_receive_gradients_qad(self):
        """Every parameter in the quantized model should get gradients."""
        er = EdgeRazor(
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
                "loss_2": {
                    "loss_type": "hidden_states",
                    "loss_function": "compute_mse",
                    "alpha": 0.3,
                    "layer_index": [0, 1],
                },
            },
        )

        student = _E2EModel()
        teacher = _E2EModel()
        q_student = er.quantize(student)
        q_student.train()

        inputs = torch.randint(0, 100, (4, 16))
        student_out = q_student(inputs, labels=inputs, return_dict=True,
                                output_hidden_states=True)
        with torch.no_grad():
            teacher_out = teacher(inputs, return_dict=True,
                                  output_hidden_states=True)
        total_loss, loss_dict = er.compute_loss(student_out, teacher_out, inputs)
        total_loss.backward()

        grad_params = 0
        zero_grad_params = 0
        for name, p in q_student.named_parameters():
            assert p.grad is not None, f"{name}: None gradient"
            grad_params += 1
            if p.grad.abs().sum() == 0:
                zero_grad_params += 1

        assert grad_params > 0
        # In a well-behaved model, most params should have non-zero gradients
        assert zero_grad_params < grad_params, \
            f"All {zero_grad_params}/{grad_params} params have zero gradients"


# ──────────────────────────────────────────────
# E2E: Model output determinism
# ──────────────────────────────────────────────

class TestE2EDeterminism:
    def test_inference_deterministic(self):
        """Quantized model should produce same output for same input."""
        er = EdgeRazor(qat_config={
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
        })

        model = _E2EModel()
        q_model = er.quantize(model)
        q_model.eval()

        torch.manual_seed(42)
        inputs = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            out1 = q_model(inputs, return_dict=True)['logits']
            out2 = q_model(inputs, return_dict=True)['logits']
        torch.testing.assert_close(out1, out2)
