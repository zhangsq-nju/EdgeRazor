"""
Parse ``quant_mode`` to determine per-layer weight / activation bit widths.

Reuses the QAT module's ``quant_config_map`` and weight-function naming
conventions to derive bit-width information without duplicating config data.

The resolution order (highest priority first):

1. *quant_mode overrides* — name-pattern → specific weight_bits
   (e.g. ``.*embed_tokens`` → W4 when the rest of the model is W1.58)
2. *config.json* ``weight_bits`` / ``activation_bits`` — optional user override
   of the global default
3. *quant_mode base function* — the default weight / activation function in
   the quant_mode config

Usage::

    from edgerazor.vllm.quant_mode_parse import QuantModeConfig

    qm = QuantModeConfig("w1_58a8kv8_embint4")
    qm.weight_bits          # 1.58  (base)
    qm.activation_bits      # 8
    qm.get_weight_bits("model.layers.0.self_attn.q_proj")  # 1.58
    qm.get_weight_bits("model.embed_tokens")               # 4  (override)
    qm.get_weight_bits("lm_head")                          # 4  (override)
"""

import re
from typing import Any

from edgerazor.qat.map import quant_config_map

# ──────────────────────────────────────────────
# weight-function name → bit-width mapping
# ──────────────────────────────────────────────


def _extract_weight_bits(func_name: str) -> float:
    """Derive the quant bit-width from an EdgeRazor weight function name.

    >>> _extract_weight_bits("weight_quant_uniform_symmetric_clip_per_block_int1_58")
    1.58
    >>> _extract_weight_bits("weight_quant_uniform_symmetric_absmax_per_block_int4")
    4
    """
    if not func_name:
        return 16  # no quantization
    if "int1_58" in func_name:
        return 1.58
    if "int8" in func_name:
        return 8
    if "int5" in func_name:
        return 5
    if "int4" in func_name:
        return 4
    if "int2" in func_name:
        return 2
    raise ValueError(f"Cannot determine bit-width from '{func_name}'")


def _extract_activation_bits(func_name: str) -> int:
    if not func_name:
        return 16  # no activation quantization
    if "int8" in func_name:
        return 8
    if "int4" in func_name:
        return 4
    return 16


# ──────────────────────────────────────────────
# QuantModeConfig
# ──────────────────────────────────────────────


class QuantModeConfig:
    """Parsed per-layer quantization config derived from *quant_mode*.

    Parameters
    ----------
    quant_mode : str
        A key in :data:`edgerazor.qat.map.quant_config_map`, e.g.
        ``"w1_58a8kv8_embint4"``.
    weight_bits_override : float or None
        Optional user override from ``config.json`` ``weight_bits`` field.
    activation_bits_override : int or None
        Optional user override from ``config.json`` ``activation_bits`` field.
    """

    def __init__(
        self,
        quant_mode: str,
        weight_bits_override: float | None = None,
        activation_bits_override: int | None = None,
    ) -> None:
        if quant_mode not in quant_config_map:
            raise ValueError(
                f"Unknown quant_mode={quant_mode!r}. "
                f"Known modes: {list(quant_config_map)}"
            )

        raw = quant_config_map[quant_mode]
        # ── resolve QAT config ──────────────────────────────────
        qat = raw  # already the full QAT dict from map.py
        select_cfg = qat["select"]
        func_cfg = qat["function"]

        # base bit-width from the main weight function
        w_func = func_cfg.get("weight_function", "")
        self._base_weight_bits = _extract_weight_bits(w_func)
        self._base_activation_bits = _extract_activation_bits(
            func_cfg.get("activation_function", "")
        )
        self._kv_cache_bits = _extract_activation_bits(
            func_cfg.get("kv_cache_function", "")
        )

        # ── apply user overrides from config.json ───────────────
        self.weight_bits = (
            weight_bits_override
            if weight_bits_override is not None
            else self._base_weight_bits
        )
        self.activation_bits = (
            activation_bits_override
            if activation_bits_override is not None
            else self._base_activation_bits
        )
        self.kv_cache_bits = self._kv_cache_bits
        self.quant_mode = quant_mode

        # ── target types ────────────────────────────────────────
        self.target_types = select_cfg.get("target_types", ["linear"])
        self.exclude_names: list[str] = select_cfg.get("exclude_names", [])

        # ── per-layer overrides ─────────────────────────────────
        self._overrides: list[tuple[re.Pattern, str]] = []
        for ov in qat.get("overrides", []):
            pattern = re.compile(ov["name"])
            w_func_ov = ov.get("weight_function", w_func)
            self._overrides.append((pattern, w_func_ov))

    # ── per-layer lookup ────────────────────────────────────────

    def get_weight_bits(self, layer_name: str) -> float:
        """Resolve *weight_bits* for a specific layer.

        Checks overrides first (highest priority), then falls back to
        the global ``weight_bits``.
        """
        for pattern, w_func in self._overrides:
            if pattern.search(layer_name):
                return _extract_weight_bits(w_func)
        return self.weight_bits

    def is_layer_quantized(self, layer_name: str) -> bool:
        """Check whether *layer_name* should be quantized at all."""
        for pattern in self.exclude_names:
            if re.search(pattern, layer_name):
                return False
        return True

    # ── QAT config accessors ─────────────────────────────────────

    def get_qat_config(self) -> dict[str, Any]:
        """Return the raw QAT config dict for downstream consumers."""
        return quant_config_map[self.quant_mode]
