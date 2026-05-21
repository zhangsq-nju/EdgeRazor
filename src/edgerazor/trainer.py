"""
EdgeRazor Trainer for low-bit LLMs.

Provides a drop-in HuggingFace Trainer subclass that integrates EdgeRazor's
quantization-aware training (QAT) and knowledge distillation (KD) capabilities
into the standard HuggingFace training loop.

Examples:
    >>> from edgerazor import EdgeRazor, EdgeRazorCausalLMTrainer
    >>> from transformers import AutoModelForCausalLM, TrainingArguments
    >>>
    >>> student = AutoModelForCausalLM.from_pretrained("student")
    >>> teacher = AutoModelForCausalLM.from_pretrained("teacher")
    >>>
    >>> # Option 1: Pass EdgeRazor instance
    >>> edgerazor = EdgeRazor(kd_config="kd_logits.yaml")
    >>> trainer = EdgeRazorCausalLMTrainer(
    ...     model=student,
    ...     teacher_model=teacher,
    ...     edgerazor=edgerazor,
    ...     args=TrainingArguments(...),
    ...     train_dataset=train_dataset,
    ... )
    >>>
    >>> # Option 2: Pass configs directly
    >>> trainer = EdgeRazorCausalLMTrainer(
    ...     model=student,
    ...     teacher_model=teacher,
    ...     kd_config="kd_logits.yaml",
    ...     args=TrainingArguments(...),
    ...     train_dataset=train_dataset,
    ... )
    >>>
    >>> trainer.train()
"""

# ruff: noqa: UP045

from pathlib import Path
from typing import Optional

import torch
from transformers import Trainer

from .edgerazor import EdgeRazor
from .edgerazor_config import EdgeRazorConfig
from .kd.util.moe_loss import router_z_loss_func


class EdgeRazorCausalLMTrainer(Trainer):
    """HuggingFace Trainer for low-bit Causal LM training with EdgeRazor.

    Handles the full training loop for:
    - Knowledge distillation (KD): teacher-student logit/hidden matching
    - Quantization-aware training (QAT): on-the-fly low-bit quantization
    - Combined QAT + KD: simultaneous compression and distillation
    - MoE models: automatic router auxiliary loss and z-loss

    Parameters
    ----------
    model : PreTrainedModel
        Student model (e.g. AutoModelForCausalLM).
    teacher_model : PreTrainedModel, optional
        Teacher model for knowledge distillation. Required when KD is enabled.
    edgerazor : EdgeRazor, optional
        Pre-configured EdgeRazor instance. Mutually exclusive with
        ``qat_config`` / ``kd_config``.
    qat_config : dict or str, optional
        QAT configuration dict or YAML path. Creates an EdgeRazor instance
        internally when ``edgerazor`` is not provided.
    kd_config : dict or str, optional
        KD configuration dict or YAML path. Requires ``teacher_model``.
    router_aux_loss_coef : float
        Coefficient for MoE router auxiliary loss. Default 0.01.
    router_z_loss_coef : float
        Coefficient for MoE router z-loss. Default 0.001.
    auto_prepare : bool
        When True (default) and QAT is enabled, call ``edgerazor.quantize()``
        on the model before training. Set to False if you already prepared.
    **kwargs
        All other keyword arguments are forwarded to ``transformers.Trainer``
        (args, data_collator, train_dataset, eval_dataset, processing_class, ...).
    """

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_loss_func=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        optimizer_cls_and_kwargs=None,
        preprocess_logits_for_metrics=None,
        # EdgeRazor params
        teacher_model=None,
        edgerazor: EdgeRazor | None = None,
        edgerazor_config: dict | str | Path | EdgeRazorConfig | None = None,
        qat_config: dict | str | Path | None = None,
        kd_config: dict | str | Path | None = None,
        # MoE params
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        # QAT
        auto_prepare: bool = True,
    ):
        # --- resolve EdgeRazor instance ---
        # edgerazor / edgerazor_config are mutually exclusive with qat_config / kd_config
        if edgerazor is not None and (
            edgerazor_config is not None or qat_config is not None or kd_config is not None
        ):
            raise ValueError(
                "'edgerazor' cannot be combined with 'edgerazor_config', "
                "'qat_config', or 'kd_config'."
            )
        if edgerazor_config is not None and (qat_config is not None or kd_config is not None):
            raise ValueError(
                "'edgerazor_config' cannot be combined with 'qat_config' or 'kd_config'."
            )

        if edgerazor is not None:
            self.edgerazor = edgerazor
        elif edgerazor_config is not None:
            self.edgerazor = EdgeRazor(config=edgerazor_config)
        elif qat_config is not None or kd_config is not None:
            self.edgerazor = EdgeRazor(qat_config=qat_config, kd_config=kd_config)
        else:
            raise ValueError(
                "One of 'edgerazor', 'edgerazor_config', "
                "'qat_config', or 'kd_config' must be provided."
            )

        if self.edgerazor.is_kd_enabled and teacher_model is None:
            raise ValueError("teacher_model is required when KD is enabled.")

        # --- pre-compute which model outputs KD requires ---
        self._kd_needs_hidden_states = False
        self._kd_needs_attentions = False
        if self.edgerazor.is_kd_enabled:
            for loss_cfg in self.edgerazor.kd.config.losses.values():
                if loss_cfg.loss_type == 'hidden_states':
                    self._kd_needs_hidden_states = True
                elif loss_cfg.loss_type == 'attentions':
                    self._kd_needs_attentions = True

        # --- auto-prepare model with QAT ---
        if auto_prepare and self.edgerazor.is_qat_enabled:
            model = self.edgerazor.quantize(model)

        # --- store model config for KV cache creation ---
        self._model_config = getattr(model, 'config', None)
        self._kv_cache_enabled = (
            self.edgerazor.is_qat_enabled
            and self.edgerazor.qat.selector.has_kv_cache
        )

        super().__init__(
            model=model,
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
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Warn if model config already adds aux_loss internally (double-counting risk).
        # EdgeRazorCausalLMTrainer manages aux_loss externally via router_aux_loss_coef.
        _m = getattr(self, 'model', None)
        if _m is not None and hasattr(_m, 'config'):
            _cfg_aux = getattr(_m.config, 'router_aux_loss_coef', 0.0)
            if _cfg_aux > 0:
                import warnings
                warnings.warn(
                    f"model.config.router_aux_loss_coef={_cfg_aux} > 0. "
                    "EdgeRazorCausalLMTrainer adds router aux_loss externally. "
                    "Set model.config.router_aux_loss_coef=0 to avoid double-counting.",
                    stacklevel=2,
                )

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.custom_losses: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        student_device = next(model.parameters()).device

        # Move inputs to student device once
        inputs = {k: v.to(student_device) for k, v in inputs.items()}
        labels = inputs["labels"]

        _is_training = model.training

        # --- student forward ---
        forward_kwargs: dict = {
            "return_dict": True,
            "output_hidden_states": _is_training and self._kd_needs_hidden_states,
            "output_attentions": _is_training and self._kd_needs_attentions,
            "output_router_logits": True,
        }
        if self._kv_cache_enabled:
            forward_kwargs['past_key_values'] = self.edgerazor.create_kv_cache(
                model_config=self._model_config,
            )

        student_outputs = model(**inputs, **forward_kwargs)

        # --- eval: skip teacher forward and KD, return task loss only ---
        if not _is_training:
            loss_total = student_outputs.loss
            self.custom_losses = {
                "train/loss_total": self._to_item(loss_total),
                "train/loss_task": self._to_item(loss_total),
                "train/loss_dist": 0.0,
            }
            return (loss_total, student_outputs) if return_outputs else loss_total

        # --- teacher forward (no grad) ---
        if self.teacher_model is not None:
            self.teacher_model.to(student_device)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=self._kd_needs_hidden_states,
                    output_attentions=self._kd_needs_attentions,
                )
        else:
            teacher_outputs = None

        # --- compute distill / task loss ---
        if self.edgerazor.is_kd_enabled:
            loss_total, loss_dict = self.edgerazor.compute_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                labels=labels,
            )
            loss_task = loss_dict.get("task_loss", 0.0)
            loss_dist = loss_dict.get("distill_loss", 0.0)
        else:
            loss_total = student_outputs.loss
            loss_task = loss_total
            loss_dist = 0.0
            loss_dict = {}

        # --- add MoE router losses (auto-detected) ---
        moe_losses = self._compute_moe_losses(student_outputs)
        for v in moe_losses.values():
            loss_total = loss_total + v

        # --- populate metrics ---
        self._track_losses(loss_total, loss_task, loss_dist, loss_dict, moe_losses)

        return (loss_total, student_outputs) if return_outputs else loss_total

    def _compute_moe_losses(self, student_outputs) -> dict[str, torch.Tensor]:
        """Auto-detect MoE router losses from model outputs.

        Works with any MoE model that exposes ``router_logits`` and/or
        ``aux_loss`` on its output (Olmoe, DeepSeek, Qwen-MoE, ...).
        """
        losses: dict[str, torch.Tensor] = {}

        if (
            hasattr(student_outputs, "router_logits")
            and student_outputs.router_logits
        ):
            router_logits = torch.stack(
                list(student_outputs.router_logits), dim=0
            )
            losses["router_z_loss"] = self.router_z_loss_coef * router_z_loss_func(
                router_logits
            )

        if (
            hasattr(student_outputs, "aux_loss")
            and student_outputs.aux_loss is not None
        ):
            losses["aux_loss"] = self.router_aux_loss_coef * student_outputs.aux_loss

        return losses

    # ------------------------------------------------------------------
    # Metric tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _to_item(v) -> float:
        """Safely convert a tensor or number to a Python float."""
        return v.item() if isinstance(v, torch.Tensor) else float(v)

    def _track_losses(
        self,
        loss_total,
        loss_task,
        loss_dist,
        loss_dict: dict,
        moe_losses: dict[str, torch.Tensor],
    ) -> None:
        """Populate ``self.custom_losses`` for logging."""
        self.custom_losses = {
            "train/loss_total": self._to_item(loss_total),
            "train/loss_task": self._to_item(loss_task),
            "train/loss_dist": self._to_item(loss_dist),
        }

        # per-loss KD details (loss_1, loss_2, loss_gate_kld, ...)
        for key, value in loss_dict.get("distill_loss_details", {}).items():
            ind = key.removeprefix("loss_")
            self.custom_losses[f"train/loss_dist_{ind}"] = self._to_item(value)

        # MoE-specific losses
        for name, value in moe_losses.items():
            self.custom_losses[f"train/{name}"] = self._to_item(value)

    # ------------------------------------------------------------------
    # Logging hook
    # ------------------------------------------------------------------

    def log(
        self, logs: dict[str, float], start_time: Optional[float] = None
    ) -> None:
        logs.update(self.custom_losses)
        super().log(logs, start_time)
