"""
EdgeRazor unified configuration module.

This module provides a unified configuration class that can load both QAT and KD
configurations from a single file or separate sources.
"""
# ruff: noqa: UP035

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .kd.util import DistillConfig
from .qat.util import QuantConfig


class EdgeRazorConfig:
    """
    Unified configuration class for EdgeRazor framework.
    
    This class can load both QAT (Quantization-Aware Training) and KD (Knowledge Distillation)
    configurations from a single YAML/JSON file or from separate sources.
    
    Configuration file format (unified):
        ```yaml
        # QAT Configuration
        method: QAT
        qat_configuration:
          select: ...
          function: ...
          training: all
        
        # KD Configuration
        method: KD
        kd_configuration:
          loss_task_alpha: 1.0
          loss_1: ...
          loss_2: ...
        ```
    
    Examples:
        >>> # Load from unified config file
        >>> config = EdgeRazorConfig.from_yaml("config.yaml")
        >>> print(config.qat_config)  # QuantConfig object
        >>> print(config.kd_config)   # DistillConfig object
        
        >>> # Load from separate files
        >>> config = EdgeRazorConfig.from_yaml(
        ...     qat_yaml="qat_config.yaml",
        ...     kd_yaml="kd_config.yaml"
        ... )
        
        >>> # Create from dict
        >>> config = EdgeRazorConfig.from_dict({
        ...     'qat_configuration': {...},
        ...     'kd_configuration': {...}
        ... })
    """
    
    def __init__(
        self,
        qat_config: QuantConfig | None = None,
        kd_config: DistillConfig | None = None,
        log_level: str | int = logging.INFO,
    ):
        """
        Initialize EdgeRazorConfig with QAT and/or KD configurations.
        
        Args:
            qat_config: QuantConfig object for QAT (optional)
            kd_config: DistillConfig object for KD (optional)
        
        Raises:
            ValueError: If both qat_config and kd_config are None
        """
        if qat_config is None and kd_config is None:
            raise ValueError(
                "At least one of qat_config or kd_config must be provided. "
                "Cannot initialize EdgeRazorConfig with both disabled."
            )
        
        self.qat_config = qat_config
        self.kd_config = kd_config
        self.log_level = log_level
    
    @staticmethod
    def _resolve_log_level(config_dict=None, qat_config=None, kd_config=None, default=logging.ERROR):
        """Resolve log_level with priority: EdgeRazor > QAT > KD > default."""
        if config_dict and config_dict.get('log_level') is not None:
            return config_dict.get('log_level')
        if qat_config is not None and getattr(qat_config, 'log_level', None) is not None:
            return qat_config.log_level
        if kd_config is not None and getattr(kd_config, 'log_level', None) is not None:
            return kd_config.log_level
        return default

    @staticmethod
    def _load_component_config(config, method, config_constructor, wrapper_key):
        """Load a single component config (QAT or KD) from dict or file path.

        Args:
            config: dict or file path (str/Path)
            method: 'QAT' or 'KD'
            config_constructor: callable(dict) -> config object
            wrapper_key: e.g. 'qat_configuration' or 'kd_configuration'
        """
        if isinstance(config, dict):
            data = config.copy()
        elif isinstance(config, (str, Path)):
            path = Path(config)
            with open(path, encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if wrapper_key in data:
                data = data[wrapper_key]
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        if 'method' not in data:
            data['method'] = method
        elif data['method'].upper() != method:
            raise ValueError(
                f"Invalid method in config: '{data['method']}'. Expected '{method}'."
            )

        return config_constructor(data)

    @classmethod
    def load(
        cls,
        config: "EdgeRazorConfig | str | Path | dict | None" = None,
        qat_config: "str | Path | dict | None" = None,
        kd_config: "str | Path | dict | None" = None,
    ) -> "EdgeRazorConfig":
        """
        Universal loader for EdgeRazorConfig from any source.
        
        This is the recommended entry point for loading configurations.
        Automatically detects file format and config type.
        
        Args:
            config: Unified config (file path, dict, or EdgeRazorConfig)
            qat_config: QAT-only config (file path or dict)
            kd_config: KD-only config (file path or dict)
        
        Returns:
            EdgeRazorConfig instance
        
        Examples:
            >>> # From unified YAML/JSON file
            >>> config = EdgeRazorConfig.load("unified_config.yaml")
            
            >>> # From separate files
            >>> config = EdgeRazorConfig.load(
            ...     qat_config="qat.yaml",
            ...     kd_config="kd.yaml"
            ... )
            
            >>> # From dict
            >>> config = EdgeRazorConfig.load({
            ...     'qat_configuration': {...},
            ...     'kd_configuration': {...}
            ... })
            
            >>> # Mixed: QAT from file, KD from dict
            >>> config = EdgeRazorConfig.load(
            ...     qat_config="qat.yaml",
            ...     kd_config={'method': 'KD', 'loss_1': {...}}
            ... )
        """
        # Already an EdgeRazorConfig
        if isinstance(config, EdgeRazorConfig):
            return config
        
        # Unified configuration
        if config is not None:
            if isinstance(config, dict):
                return cls.from_dict(config)
            elif isinstance(config, (str, Path)):
                path = Path(config)
                if path.suffix in ['.yaml', '.yml']:
                    return cls.from_yaml(yaml_path=path)
                elif path.suffix == '.json':
                    return cls.from_json(json_path=path)
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
            else:
                raise TypeError(f"Unsupported config type: {type(config)}")
        
        # Separate configurations
        if qat_config is not None or kd_config is not None:
            qat_cfg = cls._load_component_config(
                qat_config, 'QAT', QuantConfig, 'qat_configuration'
            ) if qat_config is not None else None

            kd_cfg = cls._load_component_config(
                kd_config, 'KD', DistillConfig.from_dict, 'kd_configuration'
            ) if kd_config is not None else None

            log_level = cls._resolve_log_level(
                qat_config=qat_cfg, kd_config=kd_cfg, default=logging.ERROR
            )
            return cls(qat_config=qat_cfg, kd_config=kd_cfg, log_level=log_level)
        
        raise ValueError("No configuration provided")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EdgeRazorConfig":
        """
        Create EdgeRazorConfig from a dictionary.
        
        Supports three formats:
        1. Unified: {'qat_configuration': {...}, 'kd_configuration': {...}}
        2. QAT-only: {'method': 'QAT', 'select': {...}, 'function': {...}}
        3. KD-only: {'method': 'KD', 'loss_1': {...}, 'loss_2': {...}}
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            EdgeRazorConfig instance
        """
        qat_config = None
        kd_config = None
        
        # Check for unified format (has wrapper keys)
        has_qat_wrapper = 'qat_configuration' in config_dict
        has_kd_wrapper = 'kd_configuration' in config_dict
        
        if has_qat_wrapper or has_kd_wrapper:
            # Unified configuration format
            if has_qat_wrapper:
                qat_dict = config_dict['qat_configuration'].copy()
                if 'method' not in qat_dict:
                    qat_dict['method'] = 'QAT'
                elif qat_dict['method'].upper() != 'QAT':
                    raise ValueError(
                        f"Invalid method in qat_configuration: '{qat_dict['method']}'. "
                        f"Expected 'QAT' but got '{qat_dict['method']}'"
                    )
                qat_config = QuantConfig(qat_dict)
            
            if has_kd_wrapper:
                kd_dict = config_dict['kd_configuration'].copy()
                if 'method' not in kd_dict:
                    kd_dict['method'] = 'KD'
                elif kd_dict['method'].upper() != 'KD':
                    raise ValueError(
                        f"Invalid method in kd_configuration: '{kd_dict['method']}'. "
                        f"Expected 'KD' but got '{kd_dict['method']}'"
                    )
                kd_config = DistillConfig.from_dict(kd_dict)
        else:
            # Single configuration - auto-detect type
            method = config_dict.get('method', '').upper()
            has_qat_keys = any(k in config_dict for k in ['select', 'function', 'training'])
            has_kd_keys = any(k.startswith('loss_') for k in config_dict.keys())
            
            if method == 'QAT' or has_qat_keys:
                # QAT configuration
                qat_dict = config_dict.copy()
                if 'method' not in qat_dict:
                    qat_dict['method'] = 'QAT'
                qat_config = QuantConfig(qat_dict)
            elif method == 'KD' or has_kd_keys:
                # KD configuration
                kd_dict = config_dict.copy()
                if 'method' not in kd_dict:
                    kd_dict['method'] = 'KD'
                kd_config = DistillConfig.from_dict(kd_dict)
        
        log_level = cls._resolve_log_level(
            config_dict=config_dict, qat_config=qat_config,
            kd_config=kd_config, default=logging.ERROR
        )
        return cls(qat_config=qat_config, kd_config=kd_config, log_level=log_level)

    @classmethod
    def _from_file(
        cls,
        file_path: Union[str, Path] | None,
        sub_qat_path: Union[str, Path] | None,
        sub_kd_path: Union[str, Path] | None,
        reader_fn,
        config_loader_name: str,
        file_type: str,
    ) -> "EdgeRazorConfig":
        """Shared logic for from_yaml and from_json."""
        if file_path is not None:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = reader_fn(f)
            return cls.from_dict(config_dict)

        if sub_qat_path is not None or sub_kd_path is not None:
            qat_config = None
            kd_config = None

            if sub_qat_path is not None:
                sub_qat_path = Path(sub_qat_path)
                if not sub_qat_path.exists():
                    raise FileNotFoundError(f"QAT config file not found: {sub_qat_path}")
                qat_config = getattr(QuantConfig, config_loader_name)(sub_qat_path)

            if sub_kd_path is not None:
                sub_kd_path = Path(sub_kd_path)
                if not sub_kd_path.exists():
                    raise FileNotFoundError(f"KD config file not found: {sub_kd_path}")
                kd_config = getattr(DistillConfig, config_loader_name)(sub_kd_path)

            log_level = cls._resolve_log_level(
                qat_config=qat_config, kd_config=kd_config, default=logging.ERROR
            )
            return cls(qat_config=qat_config, kd_config=kd_config, log_level=log_level)

        raise ValueError(
            f"Must provide either '{file_type}_path' (unified config) or "
            f"'qat_{file_type}'/'kd_{file_type}' (separate configs)"
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path] | None = None,
        qat_yaml: Union[str, Path] | None = None,
        kd_yaml: Union[str, Path] | None = None,
    ) -> "EdgeRazorConfig":
        """Load EdgeRazorConfig from YAML file(s)."""
        return cls._from_file(yaml_path, qat_yaml, kd_yaml, yaml.safe_load, 'from_yaml', 'yaml')

    @classmethod
    def from_json(
        cls,
        json_path: Union[str, Path] | None = None,
        qat_json: Union[str, Path] | None = None,
        kd_json: Union[str, Path] | None = None,
    ) -> "EdgeRazorConfig":
        """Load EdgeRazorConfig from JSON file(s)."""
        return cls._from_file(json_path, qat_json, kd_json, json.load, 'from_json', 'json')
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert EdgeRazorConfig to dictionary.
        
        Returns:
            Dictionary with 'qat_configuration' and/or 'kd_configuration' keys
        """
        result = {}
        
        if self.qat_config is not None:
            result['qat_configuration'] = self.qat_config.to_dict()
        
        if self.kd_config is not None:
            result['kd_configuration'] = self.kd_config.to_dict()
        
        return result
    
    def to_yaml(self, yaml_path: Union[str, Path]):
        """
        Save EdgeRazorConfig to a unified YAML file.
        
        Args:
            yaml_path: Path where to save the YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
    
    def to_json(self, json_path: Union[str, Path]):
        """
        Save EdgeRazorConfig to a unified JSON file.
        
        Args:
            json_path: Path where to save the JSON configuration
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    
    @property
    def has_qat(self) -> bool:
        """Check if QAT configuration is present"""
        return self.qat_config is not None
    
    @property
    def has_kd(self) -> bool:
        """Check if KD configuration is present"""
        return self.kd_config is not None
    
    def __repr__(self) -> str:
        """String representation of EdgeRazorConfig"""
        parts = []
        if self.has_qat:
            parts.append("QAT=enabled")
        else:
            parts.append("QAT=disabled")
        
        if self.has_kd:
            parts.append("KD=enabled")
        else:
            parts.append("KD=disabled")
        
        return f"EdgeRazorConfig({', '.join(parts)})"
    
    def __reduce__(self):
        """Support for pickle serialization."""
        return (self.__class__.from_dict, (self.to_dict(),))
