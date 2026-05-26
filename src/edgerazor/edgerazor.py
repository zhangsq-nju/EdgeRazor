from pathlib import Path

import torch
import torch.nn as nn

from .edgerazor_config import EdgeRazorConfig
from .kd import KD
from .log import get_logger, print_logo, set_component_level, setup_logging
from .qat import QAT


class EdgeRazor:
    """
    Unified API for EdgeRazor framework combining QAT and KD.
    
    Features:
    - QAT: Quantization-aware training (model structure modification)
    - KD: Knowledge distillation (training loss modification)
    - QAT + KD: Combined compression for maximum efficiency
    
    Examples:
        >>> # QAT only
        >>> edgerazor = EdgeRazor(qat_config="configs/q_vit_w1.58_a8.yaml")
        >>> model = edgerazor.prepare(model)
        
        >>> # KD only
        >>> edgerazor = EdgeRazor(kd_config="configs/kd_fd.yaml")
        >>> loss, loss_dict = edgerazor.compute_loss(
        ...     student_outputs, teacher_outputs, labels
        ... )
        
        >>> # QAT + KD combined
        >>> edgerazor = EdgeRazor(
        ...     qat_config="q_vit_w1.58_a8.yaml",
        ...     kd_config="kd_fd.yaml"
        ... )
        >>> # or unified config:
        >>> edgerazor = EdgeRazor(config="unified_config.yaml")
        >>>
        >>> student = edgerazor.prepare(student_model)
        >>> # Training loop
        >>> student_outputs = student(inputs)
        >>> with torch.no_grad():
        ...     teacher_outputs = teacher_model(inputs)
        >>> loss, loss_dict = edgerazor.compute_loss(
        ...     student_outputs, teacher_outputs, labels
        ... )
    """
    
    def __init__(
        self,
        config: str | Path | dict | EdgeRazorConfig | None = None,
        qat_config: dict | str | Path | None = None,
        kd_config: dict | str | Path | None = None,
    ):
        """
        Initialize EdgeRazor with configuration.
        
        Args:
            config: Unified configuration (file path, dict, or EdgeRazorConfig)
            qat_config: QAT-only configuration (file path or dict)
            kd_config: KD-only configuration (file path or dict)
        
        Raises:
            ValueError: If no configuration provided
        """
        # Load configuration using EdgeRazorConfig.load()
        edge_config = EdgeRazorConfig.load(config, qat_config, kd_config)

        # Print logo on first EdgeRazor startup
        print_logo()

        # Initialize logging with resolved log level
        setup_logging(level=edge_config.log_level)

        # Set per-component log levels from individual configs.
        # When a top-level log_level is set in the unified config it overrides
        # all component-specific levels, so we skip per-component setup.
        if not getattr(edge_config, '_has_top_level_log', False):
            if edge_config.has_qat and edge_config.qat_config is not None:
                set_component_level('QAT', edge_config.qat_config.log_level)
            if edge_config.has_kd and edge_config.kd_config is not None:
                set_component_level('KD', edge_config.kd_config.log_level)

        self.logger = get_logger('EdgeRazor')

        # Initialize QAT and KD modules
        self.model = None
        self.qat = QAT(edge_config.qat_config.to_dict()) if edge_config.has_qat else None
        self.kd = KD(edge_config.kd_config.to_dict()) if edge_config.has_kd else None

        # Log initialization status
        status = []
        if self.qat:
            status.append("QAT")
        if self.kd:
            status.append("KD")

        self.logger.info(
            f"EdgeRazor initialized ({' + '.join(status)} enabled | Log Level: {edge_config.log_level})"
            if status else "EdgeRazor initialized (no modules)"
        )
    
    def quantize(self, model: nn.Module) -> nn.Module:
        """
        Apply QAT quantization to model (if enabled).
        
        Args:
            model: PyTorch model
        
        Returns:
            Quantized model (if QAT enabled), otherwise unchanged
        """
        self.model = self.qat.quantize(model) if self.qat else model
        return self.model

    def replace_quantized_weights(self, model: nn.Module) -> nn.Module:
        """
        Replace model weights with quantized versions (if QAT enabled).

        After replacement, ``model.config.is_w_quantized`` is set to True
        so downstream tools (export, inference loader) can detect pre-quantized
        weights without re-quantizing.

        Args:
            model: PyTorch model

        Returns:
            Model with quantized weights (if QAT enabled), otherwise unchanged
        """
        if self.qat:
            self.model = self.qat.replace_quantized_weights(model)
            # Mark the model config so the inference loader knows weights are
            # already quantized and only need dequant/replacement, not STE training.
            cfg = getattr(self.model, 'config', None)
            if cfg is not None and hasattr(cfg, 'is_w_quantized'):
                cfg.is_w_quantized = True
                self.logger.debug("Set model.config.is_w_quantized = True")
        else:
            self.model = model
        return self.model

    def create_kv_cache(self, model_config=None):
        """Create a QuantizedKVState for KV cache quantization if configured.

        Returns None when QAT is disabled or kv_cache is not selected.
        The returned cache wraps a fresh DynamicCache and must be passed as
        ``past_key_values`` to the model's forward call.

        Args:
            model_config: Optional PretrainedConfig for DynamicCache so
                sliding-window and hybrid layers are handled correctly.
        """
        if self.qat is None:
            return None
        return self.qat.create_kv_cache(model_config=model_config)

    def compute_loss(
        self,
        student_outputs: dict | torch.Tensor,
        teacher_outputs: dict | torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute training loss with KD (if enabled).
        
        Formula: total_loss = loss_task_alpha * task_loss + distill_loss
        
        Args:
            student_outputs: Student outputs (dict/ModelOutput with 'loss' field)
            teacher_outputs: Teacher outputs (dict/ModelOutput/Tensor)
            labels: Ground truth labels
        
        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: Combined loss tensor
                - loss_dict: dict with keys:
                    - 'task_loss': float, task-specific loss value
                    - 'distill_loss': float, total distillation loss (0.0 if KD disabled)
                    - 'distill_loss_details': dict, individual loss values (empty if KD disabled)
                    - 'total_loss': float, final total loss
        """
        if self.kd:
            return self.kd.compute_loss(student_outputs, teacher_outputs, labels)
        
        # KD disabled: return task loss only with consistent format
        task_loss = student_outputs.get('loss') if isinstance(student_outputs, dict) else getattr(student_outputs, 'loss', None)
        if task_loss is None:
            raise ValueError("student_outputs must contain 'loss' field")
        
        loss_value = task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss
        loss_dict = {
            'task_loss': loss_value,
            'distill_loss': 0.0,
            'distill_loss_details': {},
            'total_loss': loss_value
        }
        return task_loss, loss_dict
    
    @property
    def is_qat_enabled(self) -> bool:
        """Check if QAT is enabled."""
        return self.qat is not None
    
    @property
    def is_kd_enabled(self) -> bool:
        """Check if KD is enabled."""
        return self.kd is not None
    
    def __repr__(self) -> str:
        """String representation."""
        status = []
        status.append(f"QAT={'enabled' if self.qat else 'disabled'}")
        status.append(f"KD={'enabled' if self.kd else 'disabled'}")
        return f"EdgeRazor({', '.join(status)})"
