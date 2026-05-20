import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeFlashAttention2,
    OlmoeSdpaAttention,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from .block import (
    copy_llamaattention_to_qkvcache_llamaattention,
    copy_multiheadattention_to_qmultiheadattention,
    copy_olmoeattention_qkvcache_olmoeattention,
    copy_qwen2_5omniattention_to_qkvcache_qwen2_5omniattention,
    copy_qwen3attention_to_qkvcache_qwen3attention,
    copy_qwen3moeattention_to_qkvcache_qwen3moeattention,
)
from .module import (
    copy_conv1d_to_qconv1d,
    copy_conv2d_to_qconv2d,
    copy_conv3d_to_qconv3d,
    copy_embedding_to_qembedding,
    copy_linear_to_qlinear,
)
from .util import QuantConfig, QuantSelector

# (source_type, qclass_key, copy_function)
_BLOCK_REPLACEMENT_SPEC = [
    (nn.MultiheadAttention,  'qmultiheadattention_cls',            copy_multiheadattention_to_qmultiheadattention),
    # Olmoe subclasses before parent
    (OlmoeFlashAttention2,   'qkvcacheolmoeflashattention2_cls',   copy_olmoeattention_qkvcache_olmoeattention),
    (OlmoeSdpaAttention,     'qkvcacheolmoesdpaattention_cls',     copy_olmoeattention_qkvcache_olmoeattention),
    (OlmoeAttention,         'qkvcacheolmoeattention_cls',         copy_olmoeattention_qkvcache_olmoeattention),
    (Qwen2_5OmniAttention,   'qkvcacheqwen2_5omniattention_cls',   copy_qwen2_5omniattention_to_qkvcache_qwen2_5omniattention),
    (Qwen3MoeAttention,      'qkvcacheqwen3moeattention_cls',      copy_qwen3moeattention_to_qkvcache_qwen3moeattention),
    (Qwen3Attention,         'qkvcacheqwen3attention_cls',         copy_qwen3attention_to_qkvcache_qwen3attention),
    (LlamaAttention,         'qkvcachellamaattention_cls',         copy_llamaattention_to_qkvcache_llamaattention),
]

_MODULE_REPLACEMENT_SPEC = [
    (nn.Linear,              'qlinear_cls',                        copy_linear_to_qlinear),
    (nn.Embedding,           'qembedding_cls',                     copy_embedding_to_qembedding),
    (nn.Conv1d,              'qconv1d_cls',                        copy_conv1d_to_qconv1d),
    (nn.Conv2d,              'qconv2d_cls',                        copy_conv2d_to_qconv2d),
    (nn.Conv3d,              'qconv3d_cls',                        copy_conv3d_to_qconv3d),
]

# (qclass_key, has_tie_embeddings_special_case)
_WEIGHT_REPLACE_SPEC = [
    ('qlinear_cls',              False),
    ('qembedding_cls',           True),
    ('qconv1d_cls',              False),
    ('qconv2d_cls',              False),
    ('qconv3d_cls',              False),
    ('qmultiheadattention_cls',  False),
]


class ModuleSpecificQuantConfig:
    """
    Wrapper that provides module-specific QuantConfig with overrides applied.
    This allows each module to have its own effective configuration while
    maintaining the same interface as QuantConfig.
    """
    def __init__(self, base_config: QuantConfig, module_name: str, module_type: type[nn.Module]):
        self.base_config = base_config
        self.module_name = module_name
        self.module_type = module_type
        
        # Get the effective function config for this specific module
        self.function = base_config.get_function_config(module_name, module_type)
        
        # Pass through other attributes from base config
        self.method = base_config.method
        self.select = base_config.select
        self.training = base_config.training
        self.overrides = base_config.overrides


def apply_quantization(
    model: nn.Module,
    quant_config: QuantConfig,
    selector: QuantSelector,
    qlinear_cls: nn.Module = None,
    qembedding_cls: nn.Module = None,
    qconv1d_cls: nn.Module = None,
    qconv2d_cls: nn.Module = None,
    qconv3d_cls: nn.Module = None,
    qmultiheadattention_cls: nn.Module = None,
    qkvcacheolmoeattention_cls: nn.Module = None,
    qkvcacheolmoeflashattention2_cls: nn.Module = None,
    qkvcacheolmoesdpaattention_cls: nn.Module = None,
    qkvcacheqwen2_5omniattention_cls: nn.Module = None,
    qkvcacheqwen3attention_cls: nn.Module = None,
    qkvcacheqwen3moeattention_cls: nn.Module = None,
    qkvcachellamaattention_cls: nn.Module = None,
) -> nn.Module:
    """Apply quantization to the model."""

    qclass_map = {
        'qlinear_cls': qlinear_cls,
        'qembedding_cls': qembedding_cls,
        'qconv1d_cls': qconv1d_cls,
        'qconv2d_cls': qconv2d_cls,
        'qconv3d_cls': qconv3d_cls,
        'qmultiheadattention_cls': qmultiheadattention_cls,
        'qkvcacheolmoeattention_cls': qkvcacheolmoeattention_cls,
        'qkvcacheolmoeflashattention2_cls': qkvcacheolmoeflashattention2_cls,
        'qkvcacheolmoesdpaattention_cls': qkvcacheolmoesdpaattention_cls,
        'qkvcacheqwen2_5omniattention_cls': qkvcacheqwen2_5omniattention_cls,
        'qkvcacheqwen3attention_cls': qkvcacheqwen3attention_cls,
        'qkvcacheqwen3moeattention_cls': qkvcacheqwen3moeattention_cls,
        'qkvcachellamaattention_cls': qkvcachellamaattention_cls,
    }

    def _replace_structure(parent_module, child_name, new_module):
        setattr(parent_module, child_name, new_module)

    # Collect block and module replacements in a single pass
    block_replacements = []
    module_replacements = []

    for name, module in model.named_modules():
        if not selector.should_quantize(name):
            continue

        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
        else:
            parent_module = model
            child_name = name

        module_config = ModuleSpecificQuantConfig(quant_config, name, type(module))

        # Try block replacements (attention blocks)
        matched = False
        for src_type, qclass_key, copy_fn in _BLOCK_REPLACEMENT_SPEC:
            if isinstance(module, src_type):
                qclass = qclass_map.get(qclass_key)
                if qclass is not None:
                    new_module = copy_fn(module, qclass, module_config)
                    block_replacements.append((parent_module, child_name, new_module))
                    matched = True
                break

        if matched:
            continue

        # Try module replacements (Linear/Embedding/Conv)
        for src_type, qclass_key, copy_fn in _MODULE_REPLACEMENT_SPEC:
            if isinstance(module, src_type):
                qclass = qclass_map.get(qclass_key)
                if qclass is not None:
                    new_module = copy_fn(module, qclass, module_config)
                    module_replacements.append((parent_module, child_name, new_module))
                break

    # Execute block replacements first (attention blocks contain sub-modules)
    for parent, child_name, new_module in block_replacements:
        _replace_structure(parent, child_name, new_module)

    # Execute module replacements second
    for parent, child_name, new_module in module_replacements:
        _replace_structure(parent, child_name, new_module)

    return model

def replace_applied_quantized_weights(
    model: nn.Module,
    selector: QuantSelector,
    qlinear_cls: nn.Module = None,
    qembedding_cls: nn.Module = None,
    qconv1d_cls: nn.Module = None,
    qconv2d_cls: nn.Module = None,
    qconv3d_cls: nn.Module = None,
    qmultiheadattention_cls: nn.Module = None,
    replace_weights=True,
    **kwargs,
) -> nn.Module:
    """Replace weights in quantized modules with their quantized versions."""
    if not replace_weights:
        return model

    qclass_map = {
        'qlinear_cls': qlinear_cls,
        'qembedding_cls': qembedding_cls,
        'qconv1d_cls': qconv1d_cls,
        'qconv2d_cls': qconv2d_cls,
        'qconv3d_cls': qconv3d_cls,
        'qmultiheadattention_cls': qmultiheadattention_cls,
    }

    for name, module in model.named_modules():
        if not selector.should_quantize(name):
            continue

        for qclass_key, has_tie_embed in _WEIGHT_REPLACE_SPEC:
            qclass = qclass_map.get(qclass_key)
            if qclass is not None and isinstance(module, qclass):
                if hasattr(module, '_weight_quant'):
                    if has_tie_embed and isinstance(model, (PreTrainedModel,)):
                        cls_name = model.__class__.__name__
                        if 'CausalLM' in cls_name or 'GPT' in cls_name:
                            if model.config.tie_word_embeddings:
                                print(f"⚠️ Skipped replacing weights for QEmbedding '{name}' due to tied embeddings.")
                                continue
                    module._weight_quant(replace_self=True)
                break

    return model
