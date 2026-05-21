
import torch.nn as nn

from .quant_config import SelectConfig


class ModuleQuantInfo:
    """Module quantization information"""
    def __init__(
        self,
        module_name: str,
        module_type: type[nn.Module],
        should_quant: bool = False,
        children: list['ModuleQuantInfo'] | None = None
    ):
        self.module_name = module_name
        self.module_type = module_type
        self.should_quant = should_quant
        self.children = children or []

    def __repr__(self):
        return f"ModuleQuantInfo(name={self.module_name}, type={self.module_type.__name__}, quant={self.should_quant})"


class QuantSelector:
    """Quantization selector"""

    def __init__(self, quant_select_config: SelectConfig):
        self.quant_select_config = quant_select_config
        self.quant_map: dict[str, ModuleQuantInfo] = {}

    def _should_quantize_by_type(self, module: nn.Module) -> bool:
        """Determine whether quantization should be applied based on type (excluding exclusion rules)"""
        module_type = type(module)

        # Check if in target types
        if module_type in self.quant_select_config.target_types:
            return True

        # Check if it's a subclass of target types
        for target_type in self.quant_select_config.target_types:
            if isinstance(module, target_type):
                return True

        return False

    def _is_excluded_by_type(self, module: nn.Module) -> bool:
        """Determine whether exclusion should be applied based on type"""
        module_type = type(module)
        return module_type in self.quant_select_config.exclude_types

    def _should_quantize_by_name(self, module_name: str) -> bool:
        """Determine whether quantization should be applied based on name (excluding exclusion rules)"""
        import re

        # Check if in target names
        if module_name in self.quant_select_config.target_names:
            return True

        # Check target name patterns
        for target_pattern in self.quant_select_config.target_names:
            if re.match(target_pattern, module_name):
                return True

        return False

    def _is_excluded_by_name(self, module_name: str) -> bool:
        """Determine whether exclusion should be applied based on name"""
        import re

        # Check if in excluded names
        if module_name in self.quant_select_config.exclude_names:
            return True

        # Check excluded name patterns
        for exclude_pattern in self.quant_select_config.exclude_names:
            if re.match(exclude_pattern, module_name):
                return True

        return False

    def _is_composite_module(self, module: nn.Module) -> bool:
        """Determine if it is a composite module (contains submodules)"""
        return len(list(module.children())) > 0

    def analyze_model(self, model: nn.Module) -> dict[str, ModuleQuantInfo]:
        """Analyze model and generate quantization mapping"""
        self.quant_map.clear()
        self._analyze_recursive(model, "")
        return self.quant_map

    def _analyze_recursive(
        self,
        module: nn.Module,
        prefix: str = ""
    ):
        """Recursively analyze modules"""

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Determine whether current module should be quantized
            # Priority check exclusion rules: if in any exclusion list, do not quantize directly

            # Check if excluded (type or name)
            is_excluded_by_type = self._is_excluded_by_type(child)
            is_excluded_by_name = self._is_excluded_by_name(full_name)

            if is_excluded_by_type or is_excluded_by_name:
                # If any exclusion rule is hit, do not quantize
                should_quant = False
            else:
                # If not excluded, then check if should be quantized
                # As long as one quantization rule is hit, quantize
                should_quant_by_type = self._should_quantize_by_type(child)
                should_quant_by_name = self._should_quantize_by_name(full_name)
                should_quant = should_quant_by_type or should_quant_by_name

            # Create quantization information
            quant_info = ModuleQuantInfo(
                module_name=full_name,
                module_type=type(child),
                should_quant=should_quant
            )

            self.quant_map[full_name] = quant_info

            # If it's a composite module, recursively process submodules
            if self._is_composite_module(child):
                self._analyze_recursive(child, full_name)

    @property
    def has_kv_cache(self) -> bool:
        """Whether KV cache quantization is enabled via the 'kv_cache' meta-target."""
        return self.quant_select_config.kv_cache

    def get_modules_to_quantize(self) -> list[str]:
        """Get list of module names to be quantized"""
        return [
            f"name={name}, type={info.module_type.__name__}" for name, info in self.quant_map.items()
            if info.should_quant
        ]

    def should_quantize(self, module_name: str) -> bool:
        """Determine whether the specified module should be quantized"""
        if module_name in self.quant_map:
            info = self.quant_map[module_name]
            return info.should_quant
        return False

    def print_quant_plan(self):
        """Print quantization plan"""
        print("=" * 80)
        print("Quantization Plan")
        print("=" * 80)

        for name, info in sorted(self.quant_map.items()):
            status = "✓ QUANT" if info.should_quant else "✗ SKIP"
            module_type = info.module_type.__name__
            print(f"{status:<10} {name:<30} [{module_type}]")

        print("=" * 80)
        print(f"Total modules to quantize: {len(self.get_modules_to_quantize())}")
        print("=" * 80)
