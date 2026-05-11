import json
import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml

from ..map import modules_map, quant_function_map

# Pre-built reverse mappings for O(1) type/function → name lookup in to_dict()
_modules_map_reverse = {v: k for k, v in modules_map.items()}
_quant_function_map_reverse = {v: k for k, v in quant_function_map.items()}

def _safe_json_default(obj):
    """Fallback JSON serializer for objects not natively JSON-serializable.

    Handles torch.Tensor (via float/list), and falls back to str/repr
    for other non-serializable types, preventing silent TypeError at dump time.
    """
    try:
        return float(obj)
    except (TypeError, ValueError):
        pass
    try:
        return str(obj)
    except Exception:
        return repr(obj)


# Custom YAML representer for OrderedDict to maintain order
def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())


yaml.add_representer(OrderedDict, represent_ordereddict)


@dataclass
class FunctionConfig:
    """Configuration for quantization functions"""
    epsilon: float

    # Weight quantization function + configuration
    weight_function: str
    w_scale_factor: float
    w_block_size: int
    w_mixed_precision_prop: float
    is_w_quantized: bool

    # State quantization function (Activation) + configuration (can be None to skip quantization)
    activation_function: str | None
    a_block_size: int
    a_mixed_precision_prop: float

    # State quantization function (KV Cache) + configuration (can be None to skip quantization)
    kv_cache_function: str | None
    kv_block_size: int
    kv_mixed_precision_prop: float

    def copy(self) -> 'FunctionConfig':
        """Create a copy of this FunctionConfig"""
        return FunctionConfig(
            epsilon=self.epsilon,
            weight_function=self.weight_function,
            w_scale_factor=self.w_scale_factor,
            w_block_size=self.w_block_size,
            w_mixed_precision_prop=self.w_mixed_precision_prop,
            is_w_quantized=self.is_w_quantized,
            activation_function=self.activation_function,
            a_block_size=self.a_block_size,
            a_mixed_precision_prop=self.a_mixed_precision_prop,
            kv_cache_function=self.kv_cache_function,
            kv_block_size=self.kv_block_size,
            kv_mixed_precision_prop=self.kv_mixed_precision_prop,
        )

    def merge(self, overrides: dict[str, Any]) -> 'FunctionConfig':
        """
        Create a new FunctionConfig by merging overrides into this config
        
        Args:
            overrides: Dictionary of parameters to override
            
        Returns:
            New FunctionConfig with overrides applied
        """
        merged = self.copy()
        
        # Update with overrides (only if key exists in overrides)
        if 'epsilon' in overrides:
            merged.epsilon = float(overrides['epsilon'])
        if 'weight_function' in overrides:
            merged.weight_function = overrides['weight_function']
        if 'w_scale_factor' in overrides:
            merged.w_scale_factor = float(overrides['w_scale_factor'])
        if 'w_block_size' in overrides:
            merged.w_block_size = int(overrides['w_block_size'])
        if 'w_mixed_precision_prop' in overrides:
            merged.w_mixed_precision_prop = float(overrides['w_mixed_precision_prop'])
        if 'is_w_quantized' in overrides:
            merged.is_w_quantized = overrides['is_w_quantized']
        if 'activation_function' in overrides:
            merged.activation_function = overrides['activation_function']
        if 'a_block_size' in overrides:
            merged.a_block_size = int(overrides['a_block_size'])
        if 'a_mixed_precision_prop' in overrides:
            merged.a_mixed_precision_prop = float(overrides['a_mixed_precision_prop'])
        if 'kv_cache_function' in overrides:
            merged.kv_cache_function = overrides['kv_cache_function']
        if 'kv_block_size' in overrides:
            merged.kv_block_size = int(overrides['kv_block_size'])
        if 'kv_mixed_precision_prop' in overrides:
            merged.kv_mixed_precision_prop = float(overrides['kv_mixed_precision_prop'])
            
        return merged


@dataclass
class SelectConfig:
    """Configuration for quantization selection"""
    # Module types to quantize
    target_types: set[type[nn.Module]] = field(default_factory=set)

    # Module name patterns to quantize (supports regular expressions)
    target_names: set[str] = field(default_factory=set)

    # Module types to exclude
    exclude_types: set[type[nn.Module]] = field(default_factory=set)

    # Module names to exclude
    exclude_names: set[str] = field(default_factory=set)


@dataclass
class OverrideConfig:
    """Configuration for per-module function overrides"""
    # Module type to override (e.g., "linear", "conv2d")
    module_type: str | None = None
    
    # Module name pattern to override (supports regex)
    module_name: str | None = None
    
    # Override parameters (only specified parameters will override global config)
    overrides: dict[str, Any] = field(default_factory=dict)
    
    def matches(self, module_name: str, module_type: type[nn.Module]) -> bool:
        """
        Check if this override applies to the given module
        
        Args:
            module_name: Name of the module
            module_type: Type of the module
            
        Returns:
            True if this override should be applied
        """
        import re
        
        # Check type match
        type_match = False
        if self.module_type is not None:
            if self.module_type in modules_map:
                type_match = modules_map[self.module_type] == module_type
            else:
                type_match = False
        
        # Check name match (regex pattern)
        name_match = False
        if self.module_name is not None:
            try:
                name_match = re.match(self.module_name, module_name) is not None
            except re.error:
                # If regex is invalid, try exact match
                name_match = self.module_name == module_name
        
        # If both are specified, both must match
        # If only one is specified, that one must match
        if self.module_type is not None and self.module_name is not None:
            return type_match and name_match
        elif self.module_type is not None:
            return type_match
        elif self.module_name is not None:
            return name_match
        else:
            return False


class QuantConfig:
    """Main quantization configuration class"""

    def __init__(self, config_dict: dict[str, Any] | None = None):
        """
        Initialize QuantConfig from a dictionary or with default values

        Args:
            config_dict: Dictionary containing configuration parameters
        """
        if config_dict is None:
            config_dict = {}

        # Handle qat_configuration wrapper (for unified config format)
        if 'qat_configuration' in config_dict:
            config_dict = config_dict['qat_configuration'].copy()

        self.method = config_dict.get("method", "QAT")

        # Parse function configuration
        function_dict = config_dict.get("function", {})
        self.function = FunctionConfig(
            epsilon=float(function_dict.get("epsilon", 1e-5)),
            weight_function=function_dict.get(
                "weight_function",
                "weight_quant_uniform_symmetric_clip_per_block_int1_58"
            ),
            w_scale_factor=float(function_dict.get("w_scale_factor", -1)),
            w_block_size=int(function_dict.get("w_block_size", -1)),
            w_mixed_precision_prop=float(function_dict.get("w_mixed_precision_prop", -1.0)),
            is_w_quantized=function_dict.get("is_w_quantized", False),
            activation_function=function_dict.get("activation_function", None),
            a_block_size=int(function_dict.get("a_block_size", -1)),
            a_mixed_precision_prop=float(function_dict.get("a_mixed_precision_prop", -1.0)),
            kv_cache_function=function_dict.get("kv_cache_function", None),
            kv_block_size=int(function_dict.get("kv_block_size", -1)),
            kv_mixed_precision_prop=float(function_dict.get("kv_mixed_precision_prop", -1.0))
        )

        # Parse overrides configuration (new feature, optional for backward compatibility)
        overrides_list = config_dict.get("overrides", [])
        self.overrides: list[OverrideConfig] = []
        for override_dict in overrides_list:
            override = OverrideConfig(
                module_type=override_dict.get("type", None),
                module_name=override_dict.get("name", None),
                overrides={k: v for k, v in override_dict.items() if k not in ["type", "name"]}
            )
            self.overrides.append(override)

        # Parse module configuration
        select_dict = config_dict.get("select", {})

        # Parse target_types (supports regex patterns like "conv.*", ".*", etc.)
        target_types_list = select_dict.get("target_types", [])
        target_types_set = set()
        for module_name in target_types_list:
            if module_name in modules_map:
                # Exact match
                target_types_set.add(modules_map[module_name])
            else:
                # Try regex pattern matching
                try:
                    pattern = re.compile(module_name)
                    matched = False
                    for key in modules_map.keys():
                        if pattern.fullmatch(key):
                            target_types_set.add(modules_map[key])
                            matched = True
                    if not matched:
                        raise ValueError(
                            f"Unknown module type: '{module_name}'. "
                            f"Available modules: {', '.join(modules_map.keys())}"
                        )
                except re.error as e:
                    raise ValueError(
                        f"Invalid regex pattern: '{module_name}'. Error: {e}"
                    ) from e

        # Parse exclude_types (supports regex patterns like "conv.*", ".*", etc.)
        exclude_types_list = select_dict.get("exclude_types", [])
        exclude_types_set = set()
        for module_name in exclude_types_list:
            if module_name in modules_map:
                # Exact match
                exclude_types_set.add(modules_map[module_name])
            else:
                # Try regex pattern matching
                try:
                    pattern = re.compile(module_name)
                    matched = False
                    for key in modules_map.keys():
                        if pattern.fullmatch(key):
                            exclude_types_set.add(modules_map[key])
                            matched = True
                    if not matched:
                        raise ValueError(
                            f"Unknown module type: '{module_name}'. "
                            f"Available modules: {', '.join(modules_map.keys())}"
                        )
                except re.error as e:
                    raise ValueError(
                        f"Invalid regex pattern: '{module_name}'. Error: {e}"
                    ) from e

        self.select = SelectConfig(
            target_types=target_types_set,
            target_names=set(select_dict.get("target_names", [])),
            exclude_types=exclude_types_set,
            exclude_names=set(select_dict.get("exclude_names", []))
        )

        self.training = config_dict.get("training", "all")
        self.log_level = config_dict.get("log_level", "ERROR")

        # Parse and map string identifiers to actual functions and modules
        self._init_map_operate()

    def get_function_config(self, module_name: str, module_type: type[nn.Module]) -> FunctionConfig:
        """
        Get the effective function configuration for a specific module.
        Applies overrides if any match the module.
        
        Args:
            module_name: Name of the module
            module_type: Type of the module
            
        Returns:
            FunctionConfig with overrides applied (if any)
        """
        # Start with global config
        effective_config = self.function
        
        # Apply matching overrides in order (later overrides take precedence)
        for override in self.overrides:
            if override.matches(module_name, module_type):
                # Merge this override into the effective config
                effective_config = effective_config.merge(override.overrides)
        
        # Map function strings to actual functions for the effective config
        effective_config = self._map_functions_in_config(effective_config)
        
        return effective_config
    
    def _map_functions_in_config(self, config: FunctionConfig) -> FunctionConfig:
        """
        Map string function names to actual function objects in a FunctionConfig
        
        Args:
            config: FunctionConfig with potential string function names
            
        Returns:
            FunctionConfig with actual function objects
        """
        mapped_config = config.copy()
        
        # Map weight function
        if isinstance(mapped_config.weight_function, str):
            if mapped_config.weight_function in quant_function_map:
                mapped_config.weight_function = quant_function_map[mapped_config.weight_function]
            else:
                raise ValueError(
                    f"Unknown weight function: '{mapped_config.weight_function}'. "
                    f"Available functions: {', '.join(quant_function_map.keys())}"
                )
        
        # Map activation function
        if mapped_config.activation_function == "":
            mapped_config.activation_function = None
        elif isinstance(mapped_config.activation_function, str) and mapped_config.activation_function is not None:
            if mapped_config.activation_function in quant_function_map:
                mapped_config.activation_function = quant_function_map[mapped_config.activation_function]
            else:
                raise ValueError(
                    f"Unknown activation function: '{mapped_config.activation_function}'. "
                    f"Available functions: {', '.join(quant_function_map.keys())}"
                )
        
        # Map kv_cache function
        if mapped_config.kv_cache_function == "":
            mapped_config.kv_cache_function = None
        elif isinstance(mapped_config.kv_cache_function, str) and mapped_config.kv_cache_function is not None:
            if mapped_config.kv_cache_function in quant_function_map:
                mapped_config.kv_cache_function = quant_function_map[mapped_config.kv_cache_function]
            else:
                raise ValueError(
                    f"Unknown kv_cache function: '{mapped_config.kv_cache_function}'. "
                    f"Available functions: {', '.join(quant_function_map.keys())}"
                )
        
        return mapped_config

    def _init_map_operate(self):
        """
        Map string identifiers to actual functions:
        - self.function.weight_function
        - self.function.activation_function
        - self.function.kv_cache_function
        """
        # Map weight function string to actual function
        if self.function.weight_function is not None:
            if self.function.weight_function in quant_function_map:
                self.function.weight_function = quant_function_map[self.function.weight_function]
            else:
                raise ValueError(
                    f"Unknown weight function: '{self.function.weight_function}'. "
                    f"Available functions: {', '.join(quant_function_map.keys())}"
                )

        # Map activation function string to actual function
        # Treat empty string as None (no quantization)
        if self.function.activation_function == "":
            self.function.activation_function = None

        if self.function.activation_function is not None:
            if self.function.activation_function in quant_function_map:
                self.function.activation_function = quant_function_map[self.function.activation_function]
            else:
                raise ValueError(
                    f"Unknown activation function: '{self.function.activation_function}'. "
                    f"Available functions: {', '.join(quant_function_map.keys())}"
                )

        # Map kv_cache function string to actual function
        # Treat empty string as None (no quantization)
        if self.function.kv_cache_function == "":
            self.function.kv_cache_function = None

        if self.function.kv_cache_function is not None:
            if self.function.kv_cache_function in quant_function_map:
                self.function.kv_cache_function = quant_function_map[self.function.kv_cache_function]
            else:
                raise ValueError(
                    f"Unknown kv_cache function: '{self.function.kv_cache_function}'. "
                    f"Available functions: {', '.join(quant_function_map.keys())}"
                )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'QuantConfig':
        """
        Create QuantConfig from a YAML file

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            QuantConfig instance

        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the file is not a valid YAML file
        """
        # Ensure yaml_path is a Path object
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        # Check if file exists
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")

        # Check if it's a file (not a directory)
        if not yaml_path.is_file():
            raise ValueError(f"Path is not a file: {yaml_path}")

        try:
            with open(yaml_path, encoding='utf-8') as file:
                config_dict = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {yaml_path}. Error: {e}") from e

        if config_dict is None:
            warnings.warn(
                f"YAML file '{yaml_path}' is empty or contains only null values. "
                f"Using default configuration values.",
                UserWarning,
                stacklevel=2
            )
            config_dict = {}

        return cls(config_dict)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> 'QuantConfig':
        """
        Create QuantConfig from a YAML string
        
        Args:
            yaml_string: YAML configuration as string
            
        Returns:
            QuantConfig instance
        """
        try:
            config_dict = yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML string. Error: {e}") from e

        if config_dict is None:
            warnings.warn(
                "YAML string is empty or contains only null values. "
                "Using default configuration values.",
                UserWarning,
                stacklevel=2
            )
            config_dict = {}

        return cls(config_dict)

    @classmethod
    def from_json(cls, json_path: str | Path) -> 'QuantConfig':
        """
        Create QuantConfig from a JSON file

        Args:
            json_path: Path to the JSON configuration file

        Returns:
            QuantConfig instance

        Raises:
            FileNotFoundError: If the JSON file does not exist
            ValueError: If the file is not a valid JSON file
        """
        # Ensure json_path is a Path object
        if isinstance(json_path, str):
            json_path = Path(json_path)

        # Check if file exists
        if not json_path.exists():
            raise FileNotFoundError(f"JSON configuration file not found: {json_path}")

        # Check if it's a file (not a directory)
        if not json_path.is_file():
            raise ValueError(f"Path is not a file: {json_path}")

        try:
            with open(json_path, encoding='utf-8') as file:
                config_dict = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {json_path}. Error: {e}") from e

        if config_dict is None:
            warnings.warn(
                f"JSON file '{json_path}' is empty or contains only null values. "
                f"Using default configuration values.",
                UserWarning,
                stacklevel=2
            )
            config_dict = {}

        return cls(config_dict)

    @classmethod
    def from_json_string(cls, json_string: str) -> 'QuantConfig':
        """
        Create QuantConfig from a JSON string
        
        Args:
            json_string: JSON configuration as string
            
        Returns:
            QuantConfig instance
            
        Raises:
            ValueError: If the string is not valid JSON
        """
        try:
            config_dict = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string. Error: {e}") from e

        if config_dict is None:
            warnings.warn(
                "JSON string is empty or contains only null values. "
                "Using default configuration values.",
                UserWarning,
                stacklevel=2
            )
            config_dict = {}

        return cls(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert QuantConfig to dictionary with specified field order
        
        Returns:
            Dictionary representation of the configuration with ordered fields:
            method, select, function, overrides, training
        """
        # Convert module types back to string names for serialization (O(1) via reverse map)
        target_types_list = []
        for module_type in self.select.target_types:
            name = _modules_map_reverse.get(module_type)
            if name is not None:
                target_types_list.append(name)

        exclude_types_list = []
        for module_type in self.select.exclude_types:
            name = _modules_map_reverse.get(module_type)
            if name is not None:
                exclude_types_list.append(name)

        # Convert function objects back to string names for serialization (O(1) via reverse map)
        weight_func_str = self.function.weight_function
        if callable(weight_func_str):
            weight_func_str = _quant_function_map_reverse.get(weight_func_str, weight_func_str)

        activation_func_str = self.function.activation_function
        if callable(activation_func_str):
            activation_func_str = _quant_function_map_reverse.get(activation_func_str, activation_func_str)
        elif activation_func_str is None:
            activation_func_str = ""

        kv_cache_func_str = self.function.kv_cache_function
        if callable(kv_cache_func_str):
            kv_cache_func_str = _quant_function_map_reverse.get(kv_cache_func_str, kv_cache_func_str)
        elif kv_cache_func_str is None:
            kv_cache_func_str = ""

        # Convert overrides to list of dicts
        overrides_list = []
        for override in self.overrides:
            override_dict = {}
            if override.module_type is not None:
                override_dict["type"] = override.module_type
            if override.module_name is not None:
                override_dict["name"] = override.module_name
            # Merge the override parameters
            override_dict.update(override.overrides)
            overrides_list.append(override_dict)

        result = OrderedDict([
            ("method", self.method),
            ("select", OrderedDict([
                ("target_types", target_types_list),
                ("target_names", list(self.select.target_names)),
                ("exclude_types", exclude_types_list),
                ("exclude_names", list(self.select.exclude_names))
            ])),
            ("function", OrderedDict([
                ("epsilon", self.function.epsilon),
                ("weight_function", weight_func_str),
                ("w_scale_factor", self.function.w_scale_factor),
                ("w_block_size", self.function.w_block_size),
                ("w_mixed_precision_prop", self.function.w_mixed_precision_prop),
                ("is_w_quantized", self.function.is_w_quantized),
                ("activation_function", activation_func_str),
                ("a_block_size", self.function.a_block_size),
                ("a_mixed_precision_prop", self.function.a_mixed_precision_prop),
                ("kv_cache_function", kv_cache_func_str),
                ("kv_block_size", self.function.kv_block_size),
                ("kv_mixed_precision_prop", self.function.kv_mixed_precision_prop)
            ]))
        ])
        
        # Add overrides if present
        if self.overrides:
            result["overrides"] = overrides_list
        
        # Add training at the end
        result["training"] = self.training
        
        return result
    
    def to_yaml(self, yaml_path: str | Path) -> None:
        """
        Save QuantConfig to a YAML file
        
        Args:
            yaml_path: Path where to save the YAML configuration
        """
        # Ensure yaml_path is a Path object
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        # Create parent directories if they don't exist
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False, indent=2, sort_keys=False)

    def to_json(self, json_path: str | Path) -> None:
        """
        Save QuantConfig to a JSON file
        
        Args:
            json_path: Path where to save the JSON configuration
        """
        # Ensure json_path is a Path object
        if isinstance(json_path, str):
            json_path = Path(json_path)

        # Create parent directories if they don't exist
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False, default=_safe_json_default)

    def __str__(self) -> str:
        """String representation of the configuration"""
        # Format overrides if present
        overrides_str = ""
        if self.overrides:
            overrides_str = ",\n    overrides=[\n"
            for idx, override in enumerate(self.overrides):
                overrides_str += "        {{"
                if override.module_type:
                    overrides_str += f"type='{override.module_type}'"
                if override.module_name:
                    if override.module_type:
                        overrides_str += f", name='{override.module_name}'"
                    else:
                        overrides_str += f"name='{override.module_name}'"
                if override.overrides:
                    overrides_str += f", overrides={override.overrides}"
                overrides_str += f"}}"
                if idx < len(self.overrides) - 1:
                    overrides_str += ",\n"
                else:
                    overrides_str += "\n"
            overrides_str += "    ]"
        
        return (
            f"QuantConfig(\n"
            f"    method='{self.method}',\n"
            f"    select=SelectConfig(\n"
            f"        target_types={self.select.target_types},\n"
            f"        target_names={self.select.target_names},\n"
            f"        exclude_types={self.select.exclude_types},\n"
            f"        exclude_names={self.select.exclude_names}\n"
            f"    ),\n"
            f"    function=FunctionConfig(\n"
            f"        epsilon={self.function.epsilon},\n"
            f"        weight_function='{self.function.weight_function}',\n"
            f"        w_scale_factor={self.function.w_scale_factor},\n"
            f"        w_block_size={self.function.w_block_size},\n"
            f"        w_mixed_precision_prop={self.function.w_mixed_precision_prop},\n"
            f"        is_w_quantized={self.function.is_w_quantized},\n"
            f"        activation_function='{self.function.activation_function}',\n"
            f"        a_block_size={self.function.a_block_size},\n"
            f"        a_mixed_precision_prop={self.function.a_mixed_precision_prop},\n"
            f"        kv_cache_function='{self.function.kv_cache_function}',\n"
            f"        kv_block_size={self.function.kv_block_size},\n"
            f"        kv_mixed_precision_prop={self.function.kv_mixed_precision_prop}\n"
            f"    ){overrides_str},\n"
            f"    training='{self.training}'\n"
            f")"
        )

    def __repr__(self) -> str:
        """Representation of the configuration"""
        return self.__str__()
