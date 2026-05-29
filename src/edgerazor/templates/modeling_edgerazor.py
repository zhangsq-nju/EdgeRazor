"""Universal EdgeRazor loader — one file for all model families.

Place this file in any EdgeRazor-quantized model repo alongside this
``config.json`` addition:

    {
      "auto_map": {
        "AutoModelForCausalLM": "modeling_edgerazor.EdgeRazorForCausalLM"
      },
      "quantization_config": {
        "quant_method": "edgerazor",
        "quant_mode": "w1_58a8kv8_embint4",
        "is_w_quantized": true
      }
    }

Then load as:

    model = AutoModelForCausalLM.from_pretrained(
        "XXX-EdgeRazor", trust_remote_code=True
    )
"""
from transformers import AutoConfig, AutoModelForCausalLM


class EdgeRazorForCausalLM:
    """Load any CausalLM with EdgeRazor W-A-KV quantization.

    Used via ``auto_map`` in ``config.json``. The loader:
    1. Reads the base model config (trust_remote_code=False, no recursion)
    2. Loads the standard HF model with weights
    3. Applies EdgeRazor quantization on top (module replacement + weight quant)
    4. Injects KV cache quantization into the forward pass
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Pop trust_remote_code — the user passed it to enable auto_map, but
        # all internal calls use trust_remote_code=False to avoid recursion.
        kwargs.pop('trust_remote_code', None)

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False
        )
        edgerazor_cfg = _resolve_edgerazor_config(config)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            trust_remote_code=False,
            **kwargs,
        )

        # quantize() replaces nn.Linear → QLinear (structure swap).
        # replace_quantized_weights is NOT called here — it is a training /
        # export step. QLinear handles both cases at inference time:
        #   is_w_quantized=True  → w_quant = W       (identity)
        #   is_w_quantized=False → w_quant = quant(W) (on-the-fly)
        if edgerazor_cfg:
            from edgerazor import EdgeRazor
            er = EdgeRazor(config=edgerazor_cfg)
            er.quantize(model)
            _inject_kv_cache(model, er)

        return model


def _resolve_edgerazor_config(config):
    """Resolve EdgeRazor config from HF config.

    Priority:
    1. ``edgerazor_config`` — inline full config dict (highest)
    2. ``quantization_config.quant_mode`` — standard path (v1.3.4+)
    3. ``edgerazor_qconfig`` — top-level preset name (backward compat)
    4. ``quant_mode`` — legacy key (backward compat)
    """
    from edgerazor import EdgeRazorConfig

    # 1. Inline full config dict
    cfg = getattr(config, 'edgerazor_config', None)
    if cfg is not None:
        return cfg

    # 2. quantization_config dict (standard path)
    qc = getattr(config, 'quantization_config', None)
    if qc is not None and qc.get('quant_method') == 'edgerazor':
        quant_mode = qc.get('quant_mode')
        if quant_mode:
            return EdgeRazorConfig.from_quant_mode(
                quant_mode,
                is_w_quantized=qc.get('is_w_quantized', True),
            )

    # 3. Top-level edgerazor_qconfig (backward compat)
    qconfig = getattr(config, 'edgerazor_qconfig', None)
    if qconfig is not None:
        return EdgeRazorConfig.from_quant_mode(
            qconfig,
            is_w_quantized=getattr(config, 'is_w_quantized', True),
        )

    # 4. Legacy quant_mode (backward compat)
    quant_mode = getattr(config, 'quant_mode', None)
    if quant_mode:
        from edgerazor.qat.map import _LEGACY_ALIASES, quant_config_map
        resolved = _LEGACY_ALIASES.get(quant_mode, quant_mode)
        if resolved not in quant_config_map:
            raise ValueError(
                f"Unknown quant_mode: '{quant_mode}'. "
                f"Available: {list(quant_config_map.keys())}"
            )
        return EdgeRazorConfig.from_quant_mode(
            resolved,
            is_w_quantized=getattr(config, 'is_w_quantized', True),
        )

    return None


def _inject_kv_cache(model, edgerazor):
    """Monkey-patch model.forward to inject QuantizedKVState for generation."""
    if not edgerazor.is_qat_enabled:
        return
    if not edgerazor.qat.selector.has_kv_cache:
        return

    _original_forward = model.forward

    def _forward(*args, **kwargs):
        if kwargs.get('past_key_values') is None and kwargs.get('use_cache', True):
            kv_cache = edgerazor.create_kv_cache(model_config=model.config)
            if kv_cache is not None:
                kwargs['past_key_values'] = kv_cache
        return _original_forward(*args, **kwargs)

    model.forward = _forward
