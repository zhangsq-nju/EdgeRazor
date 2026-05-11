"""Unit tests for QuantSelector."""

import torch.nn as nn

from edgerazor.qat.util.quant_config import SelectConfig
from edgerazor.qat.util.quant_selector import ModuleQuantInfo, QuantSelector


class TestModuleQuantInfo:
    def test_creation(self):
        info = ModuleQuantInfo("fc1", nn.Linear, should_quant=True)
        assert info.module_name == "fc1"
        assert info.module_type == nn.Linear
        assert info.should_quant is True

    def test_repr(self):
        info = ModuleQuantInfo("fc1", nn.Linear, should_quant=True)
        r = repr(info)
        assert "fc1" in r
        assert "Linear" in r
        assert "True" in r


class TestQuantSelectorBasic:
    @staticmethod
    def _make_selector(target_types=None, target_names=None,
                       exclude_types=None, exclude_names=None):
        config = SelectConfig(
            target_types=target_types or set(),
            target_names=target_names or set(),
            exclude_types=exclude_types or set(),
            exclude_names=exclude_names or set(),
        )
        return QuantSelector(config)

    def test_empty_config_nothing_to_quantize(self, simple_linear_model):
        selector = self._make_selector()
        quant_map = selector.analyze_model(simple_linear_model)
        assert len(quant_map) > 0
        assert all(not info.should_quant for info in quant_map.values())

    def test_target_by_type(self, simple_linear_model):
        selector = self._make_selector(target_types={nn.Linear})
        quant_map = selector.analyze_model(simple_linear_model)
        fc_info = quant_map["fc"]
        assert fc_info.should_quant is True

    def test_target_by_name(self, simple_multi_layer_model):
        selector = self._make_selector(target_names={"fc1"})
        quant_map = selector.analyze_model(simple_multi_layer_model)
        assert quant_map["fc1"].should_quant is True
        assert quant_map["fc2"].should_quant is False

    def test_target_by_name_pattern(self, simple_multi_layer_model):
        selector = self._make_selector(target_names={".*fc.*"})
        quant_map = selector.analyze_model(simple_multi_layer_model)
        assert quant_map["fc1"].should_quant is True
        assert quant_map["fc2"].should_quant is True

    def test_exclude_by_type(self, simple_multi_layer_model):
        selector = self._make_selector(
            target_types={nn.Linear, nn.Embedding},
            exclude_types={nn.Embedding},
        )
        quant_map = selector.analyze_model(simple_multi_layer_model)
        assert quant_map["embed"].should_quant is False
        assert quant_map["fc1"].should_quant is True

    def test_exclude_by_name(self, simple_multi_layer_model):
        selector = self._make_selector(
            target_types={nn.Linear},
            exclude_names={"fc2"},
        )
        quant_map = selector.analyze_model(simple_multi_layer_model)
        assert quant_map["fc1"].should_quant is True
        assert quant_map["fc2"].should_quant is False

    def test_exclude_takes_priority(self, simple_linear_model):
        selector = self._make_selector(
            target_types={nn.Linear},
            exclude_types={nn.Linear},
        )
        quant_map = selector.analyze_model(simple_linear_model)
        assert quant_map["fc"].should_quant is False

    def test_should_quantize_unknown_module(self, simple_linear_model):
        selector = self._make_selector(target_types={nn.Linear})
        selector.analyze_model(simple_linear_model)
        assert selector.should_quantize("nonexistent") is False

    def test_get_modules_to_quantize(self, simple_multi_layer_model):
        selector = self._make_selector(target_types={nn.Linear})
        selector.analyze_model(simple_multi_layer_model)
        modules = selector.get_modules_to_quantize()
        assert len(modules) == 2
        assert all("Linear" in m for m in modules)


class TestQuantSelectorComposite:
    def test_nested_module_analysis(self):
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 5),
                )

        model = NestedModel()
        selector = QuantSelector(SelectConfig(
            target_types={nn.Linear},
            exclude_names={"block.2"},
        ))
        quant_map = selector.analyze_model(model)
        assert quant_map["block.0"].should_quant is True
        assert quant_map["block.2"].should_quant is False

    def test_deeply_nested_modules(self):
        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Sequential(
                    nn.Linear(16, 8),
                    nn.Sequential(
                        nn.Linear(8, 4),
                        nn.Conv2d(4, 8, 3),
                    ),
                )
                self.b = nn.Linear(8, 2)

        model = DeepModel()
        selector = QuantSelector(SelectConfig(
            target_types={nn.Linear, nn.Conv2d},
        ))
        quant_map = selector.analyze_model(model)
        # Check that nested modules are found
        assert "a.0" in quant_map
        assert "a.1.0" in quant_map
        assert "a.1.1" in quant_map
        assert "b" in quant_map

    def test_subclass_type_matching(self):
        class CustomLinear(nn.Linear):
            pass

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom = CustomLinear(16, 8)

        model = CustomModel()
        selector = QuantSelector(SelectConfig(target_types={nn.Linear}))
        quant_map = selector.analyze_model(model)
        assert quant_map["custom"].should_quant is True
