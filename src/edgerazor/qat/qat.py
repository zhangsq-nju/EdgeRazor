from pathlib import Path

import torch.nn as nn
from transformers import PreTrainedModel

from ..log import get_logger
from .block import QMultiheadAttention
from .module import QConv1d, QConv2d, QConv3d, QEmbedding, QLinear
from .quantize import apply_quantization, replace_applied_quantized_weights
from .util import QuantConfig, QuantSelector


class QAT:
    """
    Quantization Aware Training (QAT) implementation for EdgeRazor framework.
    
    This class provides quantization-aware training functionality for PyTorch/Transformers neural networks,
    enabling deployment-ready quantized models for edge devices.
    
    Quantized model weights:
    - `1-bit`: {-1, 1} * scaling_factor
    - `1.58-bit` (ternary): {-1, 0, 1} * scaling_factor
    - `2-bit`: {-2, -1, 0, 1} * scaling_factor
    - `4-bit`: {-8, -6, ..., 0, ..., 6, 7} * scaling_factor
    - `8-bit`: {-128, -127, ..., 0, ..., 127} * scaling_factor
    """
    # Spec: (key, display_name, default_class)
    # Order matters: subclasses must precede parent classes for correct isinstance matching.
    _QCLASS_SPEC = [
        ('qlinear_cls',                        'Linear',                QLinear),
        ('qembedding_cls',                     'Embedding',             QEmbedding),
        ('qconv1d_cls',                        'Conv1d',                QConv1d),
        ('qconv2d_cls',                        'Conv2d',                QConv2d),
        ('qconv3d_cls',                        'Conv3d',                QConv3d),
        ('qmultiheadattention_cls',            'MultiheadAttention',    QMultiheadAttention),
    ]

    def __init__(self, config: dict | str | Path | QuantConfig):
        """
        Initialize QAT with configuration.
        
        Args:
            config: Configuration for quantization. Can be:
                - dict: Python dictionary containing configuration
                - str/Path: Path to YAML (.yaml/.yml) or JSON (.json) configuration file
                - QuantConfig: Pre-constructed QuantConfig object
                
        Examples:
            >>> # From YAML file
            >>> qat = QAT("configs/q_resnet_w1.58_a16.yaml")
            
            >>> # From JSON file
            >>> qat = QAT("configs/q_resnet_w4_a8.json")
            
            >>> # From Python dict
            >>> qat = QAT({"method": "QAT", "select": {...}, "function": {...}})
            
            >>> # From QuantConfig object
            >>> config = QuantConfig.from_yaml("config.yaml")
            >>> qat = QAT(config)
        """
        # Get component logger
        self.logger = get_logger('QAT')

        # Log initialization
        self.logger.info("Initializing Quantization Aware Training (QAT)")

        # Load configuration
        self.config = self._load_configuration(config)

        # Log configuration details
        self._log_configuration()

        # Initialize quantization selector
        self.selector = QuantSelector(self.config.select)
        self.logger.info("Quantization selector initialized")

        self.logger.info("QAT initialization completed")

    def _load_configuration(self, config: dict | str | Path | QuantConfig) -> QuantConfig:
        """
        Load and parse configuration from various input types.
        Do not support JSON or YAML strings using QAT.
        
        Args:
            config: Configuration input in various formats:
                - QuantConfig: Pre-constructed configuration object
                - dict: Python dictionary with configuration parameters
                - str/Path: File path to YAML (.yaml/.yml) or JSON (.json) configuration file
        
        Returns:
            QuantConfig: Parsed configuration object
            
        Raises:
            TypeError: If config type is not supported
            ValueError: If file format is not supported
            FileNotFoundError: If configuration file does not exist
        """
        try:
            if isinstance(config, QuantConfig):
                self.logger.info("Using provided QuantConfig object")
                return config

            elif isinstance(config, dict):
                self.logger.info("Loading configuration from Python dictionary")
                return QuantConfig(config)

            elif isinstance(config, (str, Path)):
                config_path = Path(config) if isinstance(config, str) else config
                self.logger.info(f"Loading configuration from: {config_path}")

                suffix = config_path.suffix.lower()
                if suffix in ['.yaml', '.yml']:
                    loaded_config = QuantConfig.from_yaml(config_path)
                    self.logger.debug("Configuration loaded from YAML file")
                    return loaded_config
                elif suffix == '.json':
                    loaded_config = QuantConfig.from_json(config_path)
                    self.logger.debug("Configuration loaded from JSON file")
                    return loaded_config
                else:
                    raise ValueError(
                        f"Unsupported configuration file format: {suffix}. "
                        f"Supported formats: .yaml, .yml, .json"
                    )
            else:
                raise TypeError(
                    f"Invalid configuration type: {type(config).__name__}. "
                    f"Expected: dict, str, Path, or QuantConfig"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def _log_configuration(self):
        """Log detailed configuration information at DEBUG level."""
        self.logger.debug("=== QAT Configuration Details ===")

        # Log method
        self.logger.debug(f"Method: {self.config.method}")

        # Log function configuration
        self.logger.debug("Function Configuration (Global Defaults):")
        self.logger.debug(f"  Weight: {self.config.function.weight_function}")
        self.logger.debug(f"    Scale Factor: {self.config.function.w_scale_factor}")
        self.logger.debug(f"    Block Size: {self.config.function.w_block_size}")
        self.logger.debug(f"    Mixed Precision Prop: {self.config.function.w_mixed_precision_prop}")
        self.logger.debug(f"    Is Quantized: {self.config.function.is_w_quantized}")
        self.logger.debug(f"  State (Activation): {self.config.function.activation_function}")
        self.logger.debug(f"    Block Size: {self.config.function.a_block_size}")
        self.logger.debug(f"    Mixed Precision Prop: {self.config.function.a_mixed_precision_prop}")
        self.logger.debug(f"  State (KV Cache): {self.config.function.kv_cache_function}")
        self.logger.debug(f"    Block Size: {self.config.function.kv_block_size}")
        self.logger.debug(f"    Mixed Precision Prop: {self.config.function.kv_mixed_precision_prop}")
        self.logger.debug(f"  Epsilon: {self.config.function.epsilon}")

        # Log overrides if any
        if hasattr(self.config, 'overrides') and self.config.overrides:
            self.logger.debug(f"Per-Layer Overrides: {len(self.config.overrides)} rule(s) defined")
            for idx, override in enumerate(self.config.overrides, 1):
                override_desc = []
                if override.module_type:
                    override_desc.append(f"type={override.module_type}")
                if override.module_name:
                    override_desc.append(f"name={override.module_name}")
                self.logger.debug(f"  Override {idx}: [{', '.join(override_desc)}]")
                for key, value in override.overrides.items():
                    self.logger.debug(f"    {key}: {value}")
        else:
            self.logger.debug("Per-Layer Overrides: None")

        # Log selection configuration
        self.logger.debug("Selection Configuration:")
        self.logger.debug(f"  Target Types: {self.config.select.target_types}")
        self.logger.debug(f"  Target Names: {self.config.select.target_names}")
        self.logger.debug(f"  Exclude Types: {self.config.select.exclude_types}")
        self.logger.debug(f"  Exclude Names: {self.config.select.exclude_names}")

        # Log training configuration
        self.logger.debug(f"Training: {self.config.training}")

        self.logger.debug("=== End Configuration Details ===")

    def _resolve_qclass_map(self, **overrides):
        """
        Resolve quantized class map: use provided custom class or fall back to default.

        Args:
            **overrides: Keyword arguments mapping spec keys to custom classes (or None).

        Returns:
            dict[str, type]: Mapping of spec keys to resolved classes.
        """
        result = {}
        for key, display, default_cls in self._QCLASS_SPEC:
            provided = overrides.get(key)
            if provided is not None:
                result[key] = provided
                self.logger.debug(f"Using custom {provided.__name__} class for {display} layers")
            else:
                result[key] = default_cls
                self.logger.debug(f"Using default {default_cls.__name__} class for {display} layers")
        return result

    def quantize(
        self,
        model: nn.Module,
        qlinear_cls: nn.Module = None,
        qembedding_cls: nn.Module = None,
        qconv1d_cls: nn.Module = None,
        qconv2d_cls: nn.Module = None,
        qconv3d_cls: nn.Module = None,
        qmultiheadattention_cls: nn.Module = None,
    ) -> nn.Module:
        """
        Apply quantization to the model.

        Args:
            model: PyTorch model to quantize
            qlinear_cls: Custom quantized Linear class (default: QLinear)
            qembedding_cls: Custom quantized Embedding class (default: QEmbedding)
            qconv1d_cls: Custom quantized Conv1d class (default: QConv1d)
            qconv2d_cls: Custom quantized Conv2d class (default: QConv2d)
            qconv3d_cls: Custom quantized Conv3d class (default: QConv3d)
            qmultiheadattention_cls: Custom quantized MultiheadAttention class (default: QMultiheadAttention)

        Returns:
            Quantized model
        """
        self.logger.info("Starting model quantization…")
        self.logger.debug("=" * 80)
        self.logger.debug("Starting model quantization")
        self.logger.debug("=" * 80)

        # Log model information
        self.logger.debug(f"Model type: {type(model).__name__}")

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.debug(f"Total parameters: {total_params:,}")

        # Analyze model structure and generate quantization plan
        self.logger.debug("Analyzing model structure...")
        quant_map = self.selector.analyze_model(model)

        # Log analysis results
        total_modules = len(quant_map)
        modules_to_quantize = len(self.selector.get_modules_to_quantize())
        self.logger.debug(f"Total modules analyzed: {total_modules}")
        self.logger.debug(f"Modules to quantize: {modules_to_quantize}")
        self.logger.debug(f"Modules to skip: {total_modules - modules_to_quantize}")

        # Print detailed quantization plan
        self.logger.debug("--- Detailed quantization plan ---")
        self.logger.debug("-" * 80)
        for name, info in sorted(quant_map.items()):
            status = "✓ QUANT" if info.should_quant else "✗ SKIP"
            module_type = info.module_type.__name__
            self.logger.debug(f"  {status:<10} {name:<30} [{module_type}]")
        self.logger.debug("-" * 80)

        if modules_to_quantize == 0:
            self.logger.warning("No modules selected for quantization!")
            self.logger.warning("Please check your selection configuration")
            return model

        # Resolve quantized class map (custom or default)
        qclass_map = self._resolve_qclass_map(
            qlinear_cls=qlinear_cls,
            qembedding_cls=qembedding_cls,
            qconv1d_cls=qconv1d_cls,
            qconv2d_cls=qconv2d_cls,
            qconv3d_cls=qconv3d_cls,
            qmultiheadattention_cls=qmultiheadattention_cls,
        )

        self.logger.info("Applying quantization to selected modules...")

        try:
            quantized_model = apply_quantization(
                model=model,
                quant_config=self.config,
                selector=self.selector,
                **qclass_map,
            )

            self.logger.info("Quantization applied successfully!")

        except Exception as e:
            self.logger.error(f"Failed to apply quantization: {e}")
            raise

        # Log quantization results
        self.logger.debug("-" * 80)
        self.logger.debug("Quantization results:")
        self.logger.debug("-" * 80)

        # Count quantized modules by type
        from collections import Counter
        qclass_counts = Counter()
        for _, module in quantized_model.named_modules():
            for key, _display, qclass in self._QCLASS_SPEC:
                if isinstance(module, qclass):
                    qclass_counts[key] += 1
                    break

        for key, display, _qclass in self._QCLASS_SPEC:
            count = qclass_counts.get(key, 0)
            if count:
                self.logger.debug(f"  Quantized {display} modules: {count}")
        total_quantized = sum(qclass_counts.values())
        self.logger.debug(f"  Total quantized modules: {total_quantized}")
        
        # Fix `tie_word_embeddings=True` issue
        model_class_name = quantized_model.__class__.__name__
        if isinstance(quantized_model, (PreTrainedModel, )) and ('CausalLM' in model_class_name or 'GPT' in model_class_name):
            if quantized_model.config.tie_word_embeddings:
                quantized_model.tie_weights()
            self.logger.debug(f"  Quantized model tie_word_embeddings: {quantized_model.config.tie_word_embeddings}")
        else:
            self.logger.debug("  Quantized model is not a CausalLM/GPT model or does not use tie_word_embeddings")

        # Calculate quantized parameters using weight object ID deduplication
        # This prevents double-counting shared weights (e.g., tied embeddings)
        _weight_qclass_keys = (
            'qlinear_cls', 'qembedding_cls', 'qconv1d_cls', 'qconv2d_cls',
            'qconv3d_cls', 'qmultiheadattention_cls',
        )
        _weight_qclasses = tuple(qclass_map[k] for k in _weight_qclass_keys)
        counted_weight_ids = set()
        quantized_params = 0
        for _, module in quantized_model.named_modules():
            if isinstance(module, _weight_qclasses):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_id = id(module.weight)
                    if weight_id not in counted_weight_ids:
                        quantized_params += module.weight.numel()
                        counted_weight_ids.add(weight_id)
        
        self.logger.debug(f"  Quantized parameters: {quantized_params:,} ({quantized_params/total_params*100:.10f}%)")
        self.logger.debug(f"  Unique weight objects: {len(counted_weight_ids)}")

        self.logger.debug("-" * 80)
        self.logger.info("Model quantization completed successfully!")

        return quantized_model

    def replace_quantized_weights(
        self,
        model: nn.Module,
        qlinear_cls: nn.Module = None,
        qembedding_cls: nn.Module = None,
        qconv1d_cls: nn.Module = None,
        qconv2d_cls: nn.Module = None,
        qconv3d_cls: nn.Module = None,
        qmultiheadattention_cls: nn.Module = None,
    ) -> nn.Module:
        """
        Replace model weights with their quantized versions.
        
        Args:
            model: PyTorch model with quantized modules
            
        Returns:
            Model with quantized weights
        """
        self.logger.info("Starting replacement of quantized weights…")
        self.logger.debug("=" * 80)
        self.logger.debug("Starting replacement of quantized weights")
        self.logger.debug("=" * 80)
        
        qclass_map = self._resolve_qclass_map(
            qlinear_cls=qlinear_cls,
            qembedding_cls=qembedding_cls,
            qconv1d_cls=qconv1d_cls,
            qconv2d_cls=qconv2d_cls,
            qconv3d_cls=qconv3d_cls,
            qmultiheadattention_cls=qmultiheadattention_cls,
        )

        try:
            updated_model = replace_applied_quantized_weights(
                model=model,
                selector=self.selector,
                replace_weights=True,
                **qclass_map,
            )

            self.logger.info("Quantized weights replaced successfully!")

        except Exception as e:
            self.logger.error(f"Failed to replace quantized weights: {e}")
            raise

        self.logger.debug("=" * 80)
        self.logger.debug("Replacement of quantized weights completed successfully!")
        self.logger.debug("=" * 80)

        return updated_model
