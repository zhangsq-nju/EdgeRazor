"""Unit tests for EdgeRazorCausalLMTrainer."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# lightweight test models
# ---------------------------------------------------------------------------

class _DummyStudentModel(nn.Module):
    """Returns configurable outputs, bypassing real forward."""

    def __init__(self, output):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self._output = output

    def forward(self, **kwargs):
        return self._output


class _DummyTeacherModel(nn.Module):
    """Returns configurable outputs."""

    def __init__(self, output):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self._output = output

    def forward(self, **kwargs):
        return self._output


def _make_student_output(**overrides):
    """Return a plain object with CausalLM-like attributes."""
    out = type("CausalLMOutput", (), {})()
    out.loss = torch.tensor(2.5)
    out.logits = torch.randn(2, 4, 10)
    out.hidden_states = tuple(torch.randn(2, 4, 32) for _ in range(4))
    out.attentions = None
    out.past_key_values = None
    out.router_logits = None
    out.aux_loss = None
    for k, v in overrides.items():
        setattr(out, k, v)
    return out


def _make_moe_output():
    """MoE outputs use 2-D router_logits [batch*seq, num_experts] per layer."""
    out = _make_student_output()
    out.router_logits = (
        torch.randn(8, 8),   # [batch*seq=8, num_experts=8]
        torch.randn(8, 8),
    )
    out.aux_loss = torch.tensor(0.15)
    return out


# ---------------------------------------------------------------------------
# Init / validation
# ---------------------------------------------------------------------------

class TestEdgeRazorCausalLMTrainerInit:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_init_with_edgerazor_instance_kd(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)
        assert trainer.edgerazor is er
        assert trainer.teacher_model is teacher

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_init_with_edgerazor_instance_qat(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)
        assert trainer.edgerazor is er

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_init_with_kd_config(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(
            kd_config=basic_kd_config_dict, teacher_model=teacher
        )
        assert trainer.edgerazor.is_kd_enabled is True

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_init_with_qat_config(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        model = nn.Linear(16, 8)
        trainer = EdgeRazorCausalLMTrainer(
            qat_config=basic_qat_config_dict, model=model, auto_prepare=False
        )
        assert trainer.edgerazor.is_qat_enabled is True

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_init_with_qat_and_kd_together(
        self, mock_super, basic_qat_config_dict, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(
            qat_config=basic_qat_config_dict,
            kd_config=basic_kd_config_dict,
            teacher_model=teacher,
            auto_prepare=False,
        )
        assert trainer.edgerazor.is_qat_enabled is True
        assert trainer.edgerazor.is_kd_enabled is True

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_init_with_edgerazor_config_dict(self, mock_super, unified_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(
            edgerazor_config=unified_config_dict,
            teacher_model=teacher,
            auto_prepare=False,
        )
        assert trainer.edgerazor.is_qat_enabled is True
        assert trainer.edgerazor.is_kd_enabled is True

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_init_with_edgerazor_config_path(
        self, mock_super, unified_config_dict, temp_dir
    ):
        import yaml
        from edgerazor import EdgeRazorCausalLMTrainer

        yaml_path = temp_dir / "unified.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(unified_config_dict, f)

        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(
            edgerazor_config=str(yaml_path),
            teacher_model=teacher,
            auto_prepare=False,
        )
        assert trainer.edgerazor.is_qat_enabled is True
        assert trainer.edgerazor.is_kd_enabled is True

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_no_config_raises(self, mock_super):
        from edgerazor import EdgeRazorCausalLMTrainer

        with pytest.raises(ValueError, match="must be provided"):
            EdgeRazorCausalLMTrainer()

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_edgerazor_and_qat_config_mutually_exclusive(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        with pytest.raises(ValueError, match="cannot be combined"):
            EdgeRazorCausalLMTrainer(edgerazor=er, qat_config={}, teacher_model=teacher)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_edgerazor_and_edgerazor_config_mutually_exclusive(
        self, mock_super, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        er = EdgeRazor(kd_config=basic_kd_config_dict)
        with pytest.raises(ValueError, match="cannot be combined"):
            EdgeRazorCausalLMTrainer(
                edgerazor=er, edgerazor_config={}, teacher_model=teacher
            )

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_edgerazor_config_and_qat_config_mutually_exclusive(self, mock_super):
        from edgerazor import EdgeRazorCausalLMTrainer

        with pytest.raises(ValueError, match="cannot be combined"):
            EdgeRazorCausalLMTrainer(edgerazor_config={}, qat_config={})

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_edgerazor_config_and_kd_config_mutually_exclusive(self, mock_super):
        from edgerazor import EdgeRazorCausalLMTrainer

        with pytest.raises(ValueError, match="cannot be combined"):
            EdgeRazorCausalLMTrainer(edgerazor_config={}, kd_config={})

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_kd_without_teacher_raises(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        with pytest.raises(ValueError, match="teacher_model is required"):
            EdgeRazorCausalLMTrainer(kd_config=basic_kd_config_dict)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_auto_prepare_true_quantizes_model(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer
        from edgerazor.qat.module import QLinear

        model = nn.Sequential(nn.Linear(16, 8))
        trainer = EdgeRazorCausalLMTrainer(
            qat_config=basic_qat_config_dict, model=model, auto_prepare=True
        )
        passed_model = mock_super.call_args[1].get("model")
        assert isinstance(passed_model[0], QLinear)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_auto_prepare_false_skips_quantize(
        self, mock_super, basic_qat_config_dict
    ):
        from edgerazor import EdgeRazorCausalLMTrainer

        model = nn.Linear(16, 8)
        trainer = EdgeRazorCausalLMTrainer(
            qat_config=basic_qat_config_dict, model=model, auto_prepare=False
        )
        passed_model = mock_super.call_args[1].get("model")
        assert passed_model is model

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_default_moe_coefficients(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(
            kd_config=basic_kd_config_dict, teacher_model=teacher
        )
        assert trainer.router_aux_loss_coef == 0.01
        assert trainer.router_z_loss_coef == 0.001

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_custom_moe_coefficients(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(
            kd_config=basic_kd_config_dict,
            teacher_model=teacher,
            router_aux_loss_coef=0.05,
            router_z_loss_coef=0.005,
        )
        assert trainer.router_aux_loss_coef == 0.05
        assert trainer.router_z_loss_coef == 0.005

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_all_hf_params_forwarded(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        args = Mock()
        data_collator = Mock()
        train_dataset = Mock()
        eval_dataset = Mock()
        processing_class = Mock()
        model_init = Mock()
        compute_loss_func = Mock()
        compute_metrics = Mock()
        callbacks = Mock()
        optimizers = (Mock(), Mock())
        optimizer_cls_and_kwargs = Mock()
        preprocess_logits_fn = Mock()

        trainer = EdgeRazorCausalLMTrainer(
            model=nn.Linear(16, 8),
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_fn,
            kd_config=basic_kd_config_dict,
            teacher_model=teacher,
        )

        kwargs = mock_super.call_args[1]
        assert kwargs["args"] is args
        assert kwargs["data_collator"] is data_collator
        assert kwargs["train_dataset"] is train_dataset
        assert kwargs["eval_dataset"] is eval_dataset
        assert kwargs["processing_class"] is processing_class
        assert kwargs["model_init"] is model_init
        assert kwargs["compute_loss_func"] is compute_loss_func
        assert kwargs["compute_metrics"] is compute_metrics
        assert kwargs["callbacks"] is callbacks
        assert kwargs["optimizers"] is optimizers
        assert kwargs["optimizer_cls_and_kwargs"] is optimizer_cls_and_kwargs
        assert kwargs["preprocess_logits_for_metrics"] is preprocess_logits_fn


# ---------------------------------------------------------------------------
# _kd_needs_* flags detection
# ---------------------------------------------------------------------------

class TestKDNeedsFlags:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_logits_only_sets_neither_flag(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(
            kd_config=basic_kd_config_dict, teacher_model=teacher
        )
        assert trainer._kd_needs_hidden_states is False
        assert trainer._kd_needs_attentions is False

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_hidden_states_loss_sets_flag(self, mock_super):
        from edgerazor import EdgeRazorCausalLMTrainer

        config = {
            "method": "KD",
            "loss_1": {
                "loss_type": "hidden_states",
                "loss_function": "compute_mse",
                "alpha": 0.5,
            },
        }
        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(kd_config=config, teacher_model=teacher)
        assert trainer._kd_needs_hidden_states is True
        assert trainer._kd_needs_attentions is False

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_attentions_loss_sets_flag(self, mock_super):
        from edgerazor import EdgeRazorCausalLMTrainer

        config = {
            "method": "KD",
            "loss_1": {
                "loss_type": "attentions",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
            },
        }
        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(kd_config=config, teacher_model=teacher)
        assert trainer._kd_needs_hidden_states is False
        assert trainer._kd_needs_attentions is True

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_mixed_loss_types_set_both_flags(self, mock_super):
        from edgerazor import EdgeRazorCausalLMTrainer

        config = {
            "method": "KD",
            "loss_1": {
                "loss_type": "hidden_states",
                "loss_function": "compute_mse",
                "alpha": 0.3,
            },
            "loss_2": {
                "loss_type": "attentions",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.3,
            },
        }
        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(kd_config=config, teacher_model=teacher)
        assert trainer._kd_needs_hidden_states is True
        assert trainer._kd_needs_attentions is True

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_past_key_values_loss_sets_neither(self, mock_super):
        """past_key_values loss doesn't need output_* flags (always returned)."""
        from edgerazor import EdgeRazorCausalLMTrainer

        config = {
            "method": "KD",
            "loss_1": {
                "loss_type": "past_key_values",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
            },
        }
        teacher = _DummyTeacherModel(_make_student_output())
        trainer = EdgeRazorCausalLMTrainer(kd_config=config, teacher_model=teacher)
        assert trainer._kd_needs_hidden_states is False
        assert trainer._kd_needs_attentions is False

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_qat_only_sets_neither_flag(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        trainer = EdgeRazorCausalLMTrainer(
            qat_config=basic_qat_config_dict, auto_prepare=False
        )
        assert trainer._kd_needs_hidden_states is False
        assert trainer._kd_needs_attentions is False


# ---------------------------------------------------------------------------
# Teacher eval mode
# ---------------------------------------------------------------------------

class TestTeacherEvalMode:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_teacher_set_to_eval_mode(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        teacher = _DummyTeacherModel(_make_student_output())
        teacher.train()
        assert teacher.training is True  # sanity

        EdgeRazorCausalLMTrainer(
            kd_config=basic_kd_config_dict, teacher_model=teacher
        )
        assert teacher.training is False

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_no_teacher_does_not_crash(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazorCausalLMTrainer

        trainer = EdgeRazorCausalLMTrainer(
            qat_config=basic_qat_config_dict, auto_prepare=False
        )
        assert trainer.teacher_model is None


# ---------------------------------------------------------------------------
# MoE aux_loss double-counting warning
# ---------------------------------------------------------------------------

class TestMoEAuxLossWarning:
    @staticmethod
    def _fake_trainer_init(self_, **kwargs):
        setattr(self_, 'model', kwargs.get('model'))

    def test_warns_when_config_has_aux_loss_coef(
        self, basic_qat_config_dict
    ):
        import warnings
        from edgerazor import EdgeRazorCausalLMTrainer

        model = nn.Linear(16, 8)
        model.config = Mock()
        model.config.router_aux_loss_coef = 0.02

        with patch("warnings.warn") as mock_warn, \
             patch("transformers.Trainer.__init__", self._fake_trainer_init):
            EdgeRazorCausalLMTrainer(
                qat_config=basic_qat_config_dict, model=model, auto_prepare=False
            )
        mock_warn.assert_called_once()
        assert "router_aux_loss_coef" in mock_warn.call_args[0][0]

    def test_no_warning_when_coef_is_zero(
        self, basic_qat_config_dict
    ):
        import warnings
        from edgerazor import EdgeRazorCausalLMTrainer

        model = nn.Linear(16, 8)
        model.config = Mock()
        model.config.router_aux_loss_coef = 0.0

        with patch("warnings.warn") as mock_warn, \
             patch("transformers.Trainer.__init__", self._fake_trainer_init):
            EdgeRazorCausalLMTrainer(
                qat_config=basic_qat_config_dict, model=model, auto_prepare=False
            )
        mock_warn.assert_not_called()

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_no_warning_when_model_has_no_config(
        self, mock_super_init, basic_qat_config_dict
    ):
        import warnings
        from edgerazor import EdgeRazorCausalLMTrainer

        model = nn.Linear(16, 8)

        with patch("warnings.warn") as mock_warn:
            EdgeRazorCausalLMTrainer(
                qat_config=basic_qat_config_dict, model=model, auto_prepare=False
            )
        mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# Eval mode shortcut
# ---------------------------------------------------------------------------

class TestEvalModeShortcut:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_eval_mode_skips_teacher_and_kd(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        teacher_out = _make_student_output()
        teacher = _DummyTeacherModel(teacher_out)

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        student_out = _make_student_output(loss=torch.tensor(3.5))
        model = _DummyStudentModel(student_out)
        model.eval()

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        # Spy on teacher forward before compute_loss
        teacher_forward_called = False
        original_forward = teacher.forward

        def spy_forward(**kwargs):
            nonlocal teacher_forward_called
            teacher_forward_called = True
            return original_forward(**kwargs)

        teacher.forward = spy_forward

        loss = trainer.compute_loss(model, inputs, return_outputs=False)

        assert not teacher_forward_called
        assert loss.item() == pytest.approx(3.5)
        assert trainer.custom_losses["train/loss_task"] == 3.5
        assert trainer.custom_losses["train/loss_dist"] == 0.0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_eval_mode_return_outputs(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        student_out = _make_student_output(loss=torch.tensor(3.0))
        model = _DummyStudentModel(student_out)
        model.eval()

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        result = trainer.compute_loss(model, inputs, return_outputs=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].item() == pytest.approx(3.0)
        assert result[1] is student_out

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_training_mode_still_runs_kd(self, mock_super, basic_kd_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        teacher_out = _make_student_output()
        teacher = _DummyTeacherModel(teacher_out)

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        student_out = _make_student_output(loss=torch.tensor(2.5))
        model = _DummyStudentModel(student_out)
        model.train()

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert trainer.custom_losses["train/loss_dist"] > 0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_eval_mode_qat_only(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        student_out = _make_student_output(loss=torch.tensor(4.0))
        model = _DummyStudentModel(student_out)
        model.eval()

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert loss.item() == pytest.approx(4.0)
        assert trainer.custom_losses["train/loss_total"] == 4.0
        assert trainer.custom_losses["train/loss_task"] == 4.0
        assert trainer.custom_losses["train/loss_dist"] == 0.0


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------

class TestEdgeRazorCausalLMTrainerComputeLoss:
    @patch("transformers.Trainer.__init__", return_value=None)
    @patch("transformers.Trainer.compute_loss")
    def test_kd_enabled_delegates_to_edgerazor(
        self, mock_super_compute, mock_super_init, basic_kd_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(kd_config=basic_kd_config_dict)
        teacher_out = _make_student_output()
        teacher = _DummyTeacherModel(teacher_out)

        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, teacher_model=teacher)

        student_out = _make_student_output()
        model = _DummyStudentModel(student_out)

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert isinstance(loss, torch.Tensor)
        assert "train/loss_total" in trainer.custom_losses
        assert "train/loss_task" in trainer.custom_losses
        assert "train/loss_dist" in trainer.custom_losses

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_qat_only_uses_model_loss(self, mock_super_init, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        student_out = _make_student_output(loss=torch.tensor(3.0))
        model = _DummyStudentModel(student_out)

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert loss.item() == pytest.approx(3.0)
        assert trainer.custom_losses["train/loss_task"] == 3.0
        assert trainer.custom_losses["train/loss_dist"] == 0.0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_return_outputs_true(self, mock_super_init, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        student_out = _make_student_output(loss=torch.tensor(3.0))
        model = _DummyStudentModel(student_out)

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        result = trainer.compute_loss(model, inputs, return_outputs=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert result[1] is student_out

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_teacher_not_called_when_absent(
        self, mock_super_init, basic_qat_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)
        assert trainer.teacher_model is None

        student_out = _make_student_output(loss=torch.tensor(1.0))
        model = _DummyStudentModel(student_out)

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert trainer.custom_losses["train/loss_task"] == 1.0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_moe_losses_added_to_total(
        self, mock_super_init, basic_qat_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, auto_prepare=False,
            router_aux_loss_coef=1.0, router_z_loss_coef=1.0,
        )

        student_out = _make_moe_output()
        student_out.loss = torch.tensor(3.0)
        model = _DummyStudentModel(student_out)

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert loss.item() > 3.0
        assert "train/aux_loss" in trainer.custom_losses
        assert "train/router_z_loss" in trainer.custom_losses

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_no_moe_when_model_has_no_router(
        self, mock_super_init, basic_qat_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(
            edgerazor=er, auto_prepare=False,
            router_aux_loss_coef=1.0, router_z_loss_coef=1.0,
        )

        student_out = _make_student_output(loss=torch.tensor(3.0))
        model = _DummyStudentModel(student_out)

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 4)),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.randint(0, 10, (2, 4)),
        }

        loss = trainer.compute_loss(model, inputs, return_outputs=False)
        assert loss.item() == 3.0
        assert "train/aux_loss" not in trainer.custom_losses
        assert "train/router_z_loss" not in trainer.custom_losses


# ---------------------------------------------------------------------------
# _compute_moe_losses
# ---------------------------------------------------------------------------

class TestEdgeRazorCausalLMTrainerMoeLosses:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_both_router_logits_and_aux_loss(
        self, mock_super, basic_qat_config_dict
    ):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        student_out = _make_moe_output()
        losses = trainer._compute_moe_losses(student_out)

        assert "router_z_loss" in losses
        assert "aux_loss" in losses
        assert isinstance(losses["router_z_loss"], torch.Tensor)
        assert isinstance(losses["aux_loss"], torch.Tensor)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_router_logits_only(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        out = _make_student_output(
            router_logits=(torch.randn(8, 8), torch.randn(8, 8))
        )
        losses = trainer._compute_moe_losses(out)
        assert "router_z_loss" in losses
        assert "aux_loss" not in losses

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_aux_loss_only(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        out = _make_student_output(aux_loss=torch.tensor(0.5))
        losses = trainer._compute_moe_losses(out)
        assert "aux_loss" in losses
        assert "router_z_loss" not in losses

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_no_moe_outputs(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        out = _make_student_output()
        assert trainer._compute_moe_losses(out) == {}

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_empty_router_logits_skipped(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        out = _make_student_output(router_logits=())
        assert "router_z_loss" not in trainer._compute_moe_losses(out)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_aux_loss_none_skipped(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        out = _make_student_output(aux_loss=None)
        assert "aux_loss" not in trainer._compute_moe_losses(out)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_z_loss_coefficient_applied(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)

        t1 = EdgeRazorCausalLMTrainer(
            edgerazor=er, auto_prepare=False, router_z_loss_coef=0.5
        )
        t2 = EdgeRazorCausalLMTrainer(
            edgerazor=er, auto_prepare=False, router_z_loss_coef=2.0
        )

        out = _make_moe_output()
        l1 = t1._compute_moe_losses(out)
        l2 = t2._compute_moe_losses(out)

        ratio = l2["router_z_loss"] / l1["router_z_loss"]
        assert ratio.item() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# _to_item
# ---------------------------------------------------------------------------

class TestToItem:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_tensor_to_float(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        assert trainer._to_item(torch.tensor(3.5)) == 3.5
        assert trainer._to_item(torch.tensor(0.0)) == 0.0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_float_passthrough(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        assert trainer._to_item(3.5) == 3.5
        assert trainer._to_item(0) == 0.0
        assert trainer._to_item(0.0) == 0.0


# ---------------------------------------------------------------------------
# _track_losses
# ---------------------------------------------------------------------------

class TestTrackLosses:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_basic_losses_tracked(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        trainer._track_losses(
            loss_total=torch.tensor(5.0),
            loss_task=torch.tensor(3.0),
            loss_dist=torch.tensor(2.0),
            loss_dict={},
            moe_losses={},
        )

        assert trainer.custom_losses["train/loss_total"] == 5.0
        assert trainer.custom_losses["train/loss_task"] == 3.0
        assert trainer.custom_losses["train/loss_dist"] == 2.0

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_distill_details_propagated(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        loss_dict = {
            "distill_loss_details": {
                "loss_1": torch.tensor(0.7),
                "loss_2": torch.tensor(0.3),
                "loss_gate_kld": torch.tensor(0.1),
            }
        }

        trainer._track_losses(
            loss_total=torch.tensor(5.0),
            loss_task=torch.tensor(3.0),
            loss_dist=torch.tensor(1.1),
            loss_dict=loss_dict,
            moe_losses={},
        )

        assert trainer.custom_losses["train/loss_dist_1"] == pytest.approx(0.7)
        assert trainer.custom_losses["train/loss_dist_2"] == pytest.approx(0.3)
        assert trainer.custom_losses["train/loss_dist_gate_kld"] == pytest.approx(0.1)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_moe_losses_added(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        moe_losses = {
            "router_z_loss": torch.tensor(0.05),
            "aux_loss": torch.tensor(0.15),
        }

        trainer._track_losses(
            loss_total=torch.tensor(5.0),
            loss_task=torch.tensor(3.0),
            loss_dist=torch.tensor(2.0),
            loss_dict={},
            moe_losses=moe_losses,
        )

        assert trainer.custom_losses["train/router_z_loss"] == pytest.approx(0.05)
        assert trainer.custom_losses["train/aux_loss"] == pytest.approx(0.15)

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_empty_loss_dict_handled(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)

        trainer._track_losses(
            loss_total=torch.tensor(1.0),
            loss_task=torch.tensor(1.0),
            loss_dist=0.0,
            loss_dict={},
            moe_losses={},
        )
        assert trainer.custom_losses["train/loss_total"] == 1.0
        assert len(trainer.custom_losses) == 3  # total, task, dist


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------

class TestLog:
    @patch("transformers.Trainer.__init__", return_value=None)
    def test_custom_losses_injected(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)
        trainer.custom_losses = {"train/loss_total": 4.2, "train/loss_task": 3.1}

        logs = {"loss": 4.2}
        with patch("transformers.Trainer.log") as mock_log:
            trainer.log(logs, start_time=None)
        assert "train/loss_total" in logs
        assert logs["train/loss_total"] == 4.2
        assert logs["train/loss_task"] == 3.1
        mock_log.assert_called_once()

    @patch("transformers.Trainer.__init__", return_value=None)
    def test_log_calls_super_with_start_time(self, mock_super, basic_qat_config_dict):
        from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer

        er = EdgeRazor(qat_config=basic_qat_config_dict)
        trainer = EdgeRazorCausalLMTrainer(edgerazor=er, auto_prepare=False)
        trainer.custom_losses = {}

        logs = {"loss": 3.0}
        with patch("transformers.Trainer.log") as mock_log:
            trainer.log(logs, start_time=123.0)
        mock_log.assert_called_once_with(logs, 123.0)
