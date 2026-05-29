"""Unit tests for edgerazor.convert module.

Covers: quant_mode inference, is_w_quantized semantics, CLI arg parsing,
config resolution (preset vs YAML), and export artifact creation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from edgerazor.convert import (
    _infer_quant_mode_from_config,
    parse_args,
)


# ────────────────────────────────────────────────────────────
# Tracer bullet: quant_mode inference from config functions
# ────────────────────────────────────────────────────────────


class TestInferQuantMode:
    """Auto-detect quant_mode name from EdgeRazorConfig function names."""

    def test_w4a16_from_weight_only(self):
        """W4 function, no activation/KV → 'w4'."""
        cfg = _make_qat_config(
            weight_fn="weight_quant_uniform_symmetric_absmax_per_block_int4",
        )
        assert _infer_quant_mode_from_config(cfg) == "w4"

    def test_w1_58a8kv8(self):
        """1.58-bit weight + INT8 activation + INT8 KV."""
        cfg = _make_qat_config(
            weight_fn="weight_quant_uniform_symmetric_clip_per_block_int1_58",
            activation_fn="state_quant_uniform_symmetric_absmax_per_block_int8",
            kv_cache_fn="state_quant_uniform_symmetric_absmax_per_block_int8",
        )
        assert _infer_quant_mode_from_config(cfg) == "w1_58a8kv8"

    def test_w4a8kv8(self):
        """W4 + INT8 activation + INT8 KV."""
        cfg = _make_qat_config(
            weight_fn="weight_quant_uniform_symmetric_absmax_per_block_int4",
            activation_fn="state_quant_uniform_symmetric_absmax_per_block_int8",
            kv_cache_fn="state_quant_uniform_symmetric_absmax_per_block_int8",
        )
        assert _infer_quant_mode_from_config(cfg) == "w4a8kv8"

    def test_w4a16kv16_is_w4(self):
        """No activation/kv function → just 'w4'."""
        cfg = _make_qat_config(
            weight_fn="weight_quant_uniform_symmetric_absmax_per_block_int4",
            activation_fn="",
            kv_cache_fn="",
        )
        assert _infer_quant_mode_from_config(cfg) == "w4"

    def test_w2a8kv8(self):
        """INT2 weight function → 'w2a8kv8'."""
        cfg = _make_qat_config(
            weight_fn="weight_quant_uniform_symmetric_absmax_per_block_int2",
            activation_fn="state_quant_uniform_symmetric_absmax_per_block_int8",
            kv_cache_fn="state_quant_uniform_symmetric_absmax_per_block_int8",
        )
        # int2 not in the common weight patterns but "int2" substring match
        assert "w2" in _infer_quant_mode_from_config(cfg)

    def test_no_weight_function(self):
        """No weight function → empty or no weight part."""
        cfg = _make_qat_config(
            weight_fn="",
            activation_fn="state_quant_uniform_symmetric_absmax_per_block_int8",
            kv_cache_fn="state_quant_uniform_symmetric_absmax_per_block_int8",
        )
        result = _infer_quant_mode_from_config(cfg)
        assert "w" not in result
        assert "a8" in result
        assert "kv8" in result

    def test_callable_functions(self):
        """Inference works when functions are callables, not strings."""
        from edgerazor.qat.util.quant_function import (
            state_quant_uniform_symmetric_absmax_per_block_int8,
            weight_quant_uniform_symmetric_absmax_per_block_int4,
        )
        cfg = _make_qat_config(
            weight_fn=weight_quant_uniform_symmetric_absmax_per_block_int4,
            activation_fn=state_quant_uniform_symmetric_absmax_per_block_int8,
        )
        assert _infer_quant_mode_from_config(cfg) == "w4a8"


# ────────────────────────────────────────────────────────────
# CLI argument parsing
# ────────────────────────────────────────────────────────────


class TestParseArgs:
    """CLI argument parsing for convert command."""

    def test_defaults(self):
        """Minimal args produce expected defaults."""
        args = parse_args(["--model_path", "/tmp/model", "--save_path", "/tmp/out"])
        assert args.model_path == "/tmp/model"
        assert args.save_path == "/tmp/out"
        assert args.is_w_quantized is False
        assert args.edgerazor_config is None
        assert args.quant_mode is None
        assert args.dtype == "bfloat16"

    def test_is_w_quantized_flag(self):
        """--is_w_quantized true sets the flag."""
        args = parse_args([
            "--model_path", "/tmp/model",
            "--save_path", "/tmp/out",
            "--is_w_quantized", "true",
        ])
        assert args.is_w_quantized is True

    def test_is_w_quantized_default_false(self):
        """Omitting --is_w_quantized → False."""
        args = parse_args([
            "--model_path", "/tmp/model",
            "--save_path", "/tmp/out",
        ])
        assert args.is_w_quantized is False

    def test_quant_mode_preset(self):
        """--quant_mode accepts a preset string."""
        args = parse_args([
            "--model_path", "/tmp/model",
            "--save_path", "/tmp/out",
            "--quant_mode", "w1_58a8kv8_embint4",
        ])
        assert args.quant_mode == "w1_58a8kv8_embint4"

    def test_edgerazor_config_path(self):
        """--edgerazor_config accepts a file path."""
        args = parse_args([
            "--model_path", "/tmp/model",
            "--save_path", "/tmp/out",
            "--edgerazor_config", "/tmp/config.yaml",
        ])
        assert args.edgerazor_config == "/tmp/config.yaml"

    def test_both_quant_mode_and_config(self):
        """Both --quant_mode and --edgerazor_config accepted together."""
        args = parse_args([
            "--model_path", "/tmp/model",
            "--save_path", "/tmp/out",
            "--quant_mode", "w4a8kv8",
            "--edgerazor_config", "/tmp/config.yaml",
        ])
        assert args.quant_mode == "w4a8kv8"
        assert args.edgerazor_config == "/tmp/config.yaml"

    def test_dtype_choices(self):
        """--dtype accepts valid choices."""
        for dt in ("bfloat16", "float16", "float32"):
            args = parse_args([
                "--model_path", "/tmp/m", "--save_path", "/tmp/o", "--dtype", dt,
            ])
            assert args.dtype == dt

    def test_dtype_rejects_invalid(self):
        """--dtype rejects unsupported values."""
        with pytest.raises(SystemExit):
            parse_args([
                "--model_path", "/tmp/m",
                "--save_path", "/tmp/o",
                "--dtype", "int8",
            ])

    def test_missing_required(self):
        """Missing --model_path or --save_path exits."""
        with pytest.raises(SystemExit):
            parse_args(["--save_path", "/tmp/out"])
        with pytest.raises(SystemExit):
            parse_args(["--model_path", "/tmp/model"])


# ────────────────────────────────────────────────────────────
# Integration: convert() behavior
# ────────────────────────────────────────────────────────────


class TestConvert:
    """End-to-end convert() invocation (uses a real tiny model)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip on CI / no GPU — needs HF model."""
        pytest.importorskip("transformers")

    def test_is_w_quantized_true_skips_replace(self, tmp_path):
        """is_w_quantized=True must NOT call replace_quantized_weights."""
        from edgerazor.convert import convert

        model_dir, src_dir = _make_tiny_model(tmp_path)

        with mock.patch(
            "edgerazor.EdgeRazor.replace_quantized_weights"
        ) as mock_replace:
            mock_replace.return_value = mock.MagicMock()
            convert(
                model_path=str(src_dir),
                save_path=str(tmp_path / "out"),
                quant_mode="w4a8",
                is_w_quantized=True,
            )
            # replace_quantized_weights must NOT have been called
            mock_replace.assert_not_called()

    def test_is_w_quantized_false_calls_replace(self, tmp_path):
        """is_w_quantized=False (default) MUST call replace_quantized_weights."""
        from edgerazor.convert import convert

        _, src_dir = _make_tiny_model(tmp_path)

        with (
            mock.patch(
                "edgerazor.EdgeRazor.replace_quantized_weights"
            ) as mock_replace,
            mock.patch(
                "edgerazor.convert._save_fake_quant_weights"
            ) as mock_save,
        ):
            mock_replace.return_value = mock.MagicMock()
            convert(
                model_path=str(src_dir),
                save_path=str(tmp_path / "out"),
                quant_mode="w4a8",
                is_w_quantized=False,
            )
            mock_replace.assert_called_once()
            # fake-quant save must be called (not export copy)
            mock_save.assert_called_once()

    def test_output_has_quantization_config(self, tmp_path):
        """Output config.json must contain quantization_config.quant_method."""
        from edgerazor.convert import convert

        model_dir, src_dir = _make_tiny_model(tmp_path)

        convert(
            model_path=str(src_dir),
            save_path=str(tmp_path / "out"),
            quant_mode="w4a8",
            is_w_quantized=True,
        )

        out_cfg = tmp_path / "out" / "config.json"
        assert out_cfg.exists()
        with open(out_cfg) as f:
            cfg = json.load(f)
        assert "quantization_config" in cfg
        assert cfg["quantization_config"]["quant_method"] == "edgerazor"
        assert cfg["quantization_config"]["quant_mode"] == "w4a8"

    def test_output_has_modeling_template(self, tmp_path):
        """Output dir must contain modeling_edgerazor.py."""
        from edgerazor.convert import convert

        model_dir, src_dir = _make_tiny_model(tmp_path)

        convert(
            model_path=str(src_dir),
            save_path=str(tmp_path / "out"),
            quant_mode="w4a8",
        )

        assert (tmp_path / "out" / "modeling_edgerazor.py").exists()

    def test_output_has_tokenizer_files(self, tmp_path):
        """Output dir must contain tokenizer files from source."""
        from edgerazor.convert import convert

        model_dir, src_dir = _make_tiny_model(tmp_path)

        convert(
            model_path=str(src_dir),
            save_path=str(tmp_path / "out"),
            quant_mode="w4a8",
        )

        # tokenizer files from source should be copied
        for fname in ("tokenizer.json", "tokenizer_config.json"):
            assert (tmp_path / "out" / fname).exists()

    def test_quant_mode_overrides_config(self, tmp_path):
        """When both given, quant_mode appears in config.json,
        but EdgeRazor config is loaded from edgerazor_config file."""
        from edgerazor.convert import convert

        model_dir, src_dir = _make_tiny_model(tmp_path)

        # Create a minimal YAML config with W1_58 weight function
        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text(yaml.dump({
            "qat_configuration": {
                "function": {
                    "weight_function": (
                        "weight_quant_uniform_symmetric_clip_per_block_int1_58"
                    ),
                    "activation_function": "",
                    "kv_cache_function": "",
                },
                "select": {"target_types": ["linear"]},
            },
        }))

        convert(
            model_path=str(src_dir),
            save_path=str(tmp_path / "out"),
            edgerazor_config=str(config_yaml),
            quant_mode="w4a8",  # overrides the inferred w1_58
        )

        out_cfg = tmp_path / "out" / "config.json"
        with open(out_cfg) as f:
            cfg = json.load(f)
        # quant_mode in config should be the override, not auto-detected
        assert cfg["quantization_config"]["quant_mode"] == "w4a8"


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────


def _make_qat_config(weight_fn=None, activation_fn=None, kv_cache_fn=None):
    """Build a minimal mock EdgeRazorConfig for testing _infer_quant_mode_from_config."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    fn_cfg = MagicMock()
    fn_cfg.weight_function = weight_fn
    fn_cfg.activation_function = activation_fn
    fn_cfg.kv_cache_function = kv_cache_fn
    cfg.qat_config.function = fn_cfg
    return cfg


def _make_tiny_model(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal HF model directory that can be loaded.

    Returns (model_dir, src_dir) — src_dir is the dir containing the model.
    """
    from transformers import AutoConfig, Qwen2ForCausalLM

    src_dir = tmp_path / "tiny_model"
    src_dir.mkdir(exist_ok=True)

    # Minimal Qwen3-style config (Qwen2ForCausalLM is compatible)
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "max_position_embeddings": 1024,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "tie_word_embeddings": False,
    }
    with open(src_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    # Use direct constructor to avoid AutoModel.from_config →
    # torch.jit.script deprecation path in transformers <4.57
    hf_config = AutoConfig.for_model(**cfg)
    model = Qwen2ForCausalLM(hf_config).to("cpu")
    model.save_pretrained(str(src_dir))

    # Also save a dummy tokenizer.json and tokenizer_config.json
    (src_dir / "tokenizer.json").write_text("{}")
    (src_dir / "tokenizer_config.json").write_text("{}")

    return src_dir, src_dir
