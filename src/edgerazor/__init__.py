"""EdgeRazor: Lightweight Model Training Framework

EdgeRazor provides unified tools for model compression and optimization:
- QAT (Quantization-Aware Training): Low-bit quantization (1.58-bit to 8-bit)
- KD (Knowledge Distillation): Teacher-student learning for model compression
- Combined QAT + KD: Simultaneous quantization and distillation

Quick Start:
    >>> from edgerazor import EdgeRazor
    >>>
    >>> # QAT + KD combined training
    >>> edgerazor = EdgeRazor(
    ...     qat_config="configs/q_resnet_w4_a8.yaml",
    ...     kd_config="configs/kd_logits.yaml"
    ... )
    >>>
    >>> # Prepare student model with quantization
    >>> student = edgerazor.prepare(student_model)
    >>>
    >>> # Training loop
    >>> for inputs, labels in train_loader:
    ...     student_out = student(**inputs, return_dict=True, output_hidden_states=True, output_attentions=True)
    ...     teacher_out = teacher(**inputs, return_dict=True, output_hidden_states=True, output_attentions=True)
    ...
    ...     # Compute combined loss (task + distillation)
    ...     total_loss, loss_dict = edgerazor.compute_loss(
    ...         student_out, teacher_out, labels
    ...     )
    ...
    ...     total_loss.backward()
    ...     optimizer.step()
"""
# ruff: noqa: F401

from .edgerazor import EdgeRazor
from .edgerazor_config import EdgeRazorConfig
from .kd import KD, DistillConfig
from .qat import QAT, QuantConfig
from .qat.map import quant_config_map
from .trainer import EdgeRazorCausalLMTrainer

__version__ = "1.3.4"

__all__ = [
    "EdgeRazor",                 # Unified API
    "EdgeRazorConfig",           # Unified configuration class
    "QAT",                       # QAT API
    "QuantConfig",               # QAT configuration class
    "KD",                        # KD API
    "DistillConfig",             # KD configuration class
    "quant_config_map",          # Quantization configuration map
    "EdgeRazorCausalLMTrainer",  # Trainer API
]
