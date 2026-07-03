"""Convert an unquantized HF model to an EdgeRazor-quantized checkpoint.

Two modes via ``--is_w_quantized``:

* ``false`` (default) — load, quantize, **replace** weights with fake-quant
  values, then save.
* ``true`` — load, quantize (STE path only), copy original unquantized
  weights, and write ``quantization_config`` so the quantized loader
  applies quantization at load time (e.g. ``modeling_edgerazor.py`` /
  vLLM).

Config can be specified as a YAML file (``--edgerazor_config``) or a
preset name (``--quant_mode``).  When both are given, ``quant_mode``
wins for the ``quantization_config.quant_mode`` field but the YAML
config is used for the actual quantization.

Usage::

    # CLI — quant_mode preset (fake-quant)
    python -m edgerazor.convert \\
        --model_path ./Qwen3-0.6B \\
        --save_path ./Qwen3-0.6B-W4A16 \\
        --quant_mode w4a8 \\
        --is_w_quantized true

    # CLI — YAML config (real quant weights)
    python -m edgerazor.convert \\
        --model_path ./Qwen3-0.6B \\
        --save_path ./Qwen3-0.6B-W4A16 \\
        --edgerazor_config configs/w4a16kv16_qwen3.yaml

    # Python API
    from edgerazor.convert import convert
    convert(
        model_path="./Qwen3-0.6B",
        save_path="./Qwen3-0.6B-W4A16",
        quant_mode="w4a8",
        is_w_quantized=True,
    )
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.utils import logging as hf_logging

from edgerazor import EdgeRazor
from edgerazor.edgerazor_config import EdgeRazorConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Validation helpers
# ────────────────────────────────────────────────────────────


def _validate_model_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {path}")
    if not (path / "model.safetensors").exists() and \
       not (path / "model.safetensors.index.json").exists():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json found "
            f"in {path}. Is this a valid HF model directory?"
        )


def _validate_config_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"EdgeRazor config file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(
            f"EdgeRazor config path is a directory, not a file: {path}"
        )
    if path.suffix not in (".yaml", ".yml", ".json"):
        raise ValueError(
            f"EdgeRazor config must be .yaml/.yml/.json, got: {path.suffix}"
        )


def _check_save_path(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(
            f"Save path exists but is not a directory: {path}"
        )
    contents = list(path.iterdir())
    if contents:
        print(
            f"Warning: save_path {path} already exists and contains "
            f"{len(contents)} item(s). Existing files may be overwritten.",
            file=sys.stderr,
        )


# ────────────────────────────────────────────────────────────
# quant_mode inference from EdgeRazor config
# ────────────────────────────────────────────────────────────


def _infer_quant_mode_from_config(er_config: EdgeRazorConfig) -> str:
    """Auto-detect a ``quant_mode`` name from function names in a config.

    Inspects ``weight_function``, ``activation_function``, and
    ``kv_cache_function`` (strings or callables) for bit-width
    substrings and synthesises a name like ``"w4a8kv8"``.

    No model-family suffix is appended.
    """
    fn_cfg = er_config.qat_config.function
    parts: list[str] = []

    # Weight bits
    w = _fn_name(fn_cfg.weight_function)
    if w:
        _add_weight_bits(parts, w)

    # Activation bits
    a = _fn_name(fn_cfg.activation_function)
    if a:
        _add_state_bits(parts, a, prefix="a")

    # KV-cache bits
    k = _fn_name(fn_cfg.kv_cache_function)
    if k:
        _add_state_bits(parts, k, prefix="kv")

    return "".join(parts) if parts else "unknown"


def _fn_name(fn) -> str:
    """Return the string name of *fn* (callable or str), or '' if falsy."""
    if not fn:
        return ""
    if callable(fn):
        return fn.__name__
    return str(fn)


def _add_weight_bits(parts: list[str], name: str) -> None:
    if "int1_58" in name or "int1.58" in name:
        parts.append("w1_58")
    elif "int8" in name:
        parts.append("w8")
    elif "int5" in name:
        parts.append("w5")
    elif "int4" in name:
        parts.append("w4")
    elif "int2" in name:
        parts.append("w2")


def _add_state_bits(parts: list[str], name: str, prefix: str) -> None:
    if "int8" in name:
        parts.append(f"{prefix}8")
    elif "int4" in name:
        parts.append(f"{prefix}4")
    elif "int2" in name:
        parts.append(f"{prefix}2")


# ────────────────────────────────────────────────────────────
# Asset helpers
# ────────────────────────────────────────────────────────────


def _get_template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def _copy_template(target_dir: Path, filename: str, overwrite: bool = False) -> Path:
    src = _get_template_dir() / filename
    dst = target_dir / filename
    if dst.exists() and not overwrite:
        print(f"  Skipping {filename} (already exists)")
        return dst
    shutil.copy2(src, dst)
    print(f"  Copied {filename}")
    return dst


def _copy_safetensors(src: Path, dst: Path) -> None:
    for pattern in ['*.safetensors', '*.bin']:
        for f in src.glob(pattern):
            target = dst / f.name
            if not target.exists():
                shutil.copy2(f, target)
                print(f"  Copied {f.name}")
            else:
                print(f"  Skipping {f.name} (already exists)")


def _copy_orig_files(src: Path, dst: Path) -> None:
    orig_files = [
        'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'vocab.json', 'merges.txt',
        'added_tokens.json', 'chat_template.jinja',
        'generation_config.json',
    ]
    for name in orig_files:
        f = src / name
        if f.exists() and not (dst / name).exists():
            shutil.copy2(f, dst / name)
            print(f"  Copied {name}")


def _copy_model_assets(src: Path, dst: Path) -> None:
    """Copy all files/dirs from *src* to *dst*, skipping weight files."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in ("model.safetensors", "model.safetensors.index.json"):
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _save_fake_quant_weights(
    qmodel: AutoModelForCausalLM, save_path: Path,
) -> None:
    qmodel.to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print("Saving fake-quant weights to temporary directory ...")
        qmodel.save_pretrained(str(tmp))
        src_safetensors = tmp / "model.safetensors"
        # sharded checkpoints write model-00001-of-XXXXX.safetensors
        if not src_safetensors.exists():
            candidates = sorted(tmp.glob("model*.safetensors"))
            if not candidates:
                raise RuntimeError(
                    "save_pretrained did not produce any safetensors file"
                )
            # Move all safetensors + index
            for f in candidates:
                shutil.move(str(f), str(save_path / f.name))
            src_index = tmp / "model.safetensors.index.json"
            if src_index.exists():
                shutil.move(
                    str(src_index),
                    str(save_path / "model.safetensors.index.json"),
                )
            print(f"Moved safetensors to {save_path}")
            return
        shutil.move(str(src_safetensors), str(save_path / "model.safetensors"))
        src_index = tmp / "model.safetensors.index.json"
        if src_index.exists():
            shutil.move(
                str(src_index),
                str(save_path / "model.safetensors.index.json"),
            )
        print(f"Moved model.safetensors to {save_path}")


def _resolve_quant_bits(quant_mode: str) -> tuple[int, int, int]:
    mode = quant_mode.lower()
    w_bits, a_bits, kv_bits = 4, 16, 16

    if "w1_58" in mode or "w1.58" in mode:
        w_bits = 4  # 1.58-bit degraded to 4-bit packing
    elif "w4" in mode:
        w_bits = 4
    elif "w2" in mode:
        w_bits = 2

    if "a8" in mode:
        a_bits = 8
    if "kv8" in mode:
        kv_bits = 8

    return w_bits, a_bits, kv_bits


def _patch_config_json(
    target_dir: Path,
    quant_mode: str,
    is_w_quantized: bool = True,
    backend: str = "marlin",
    weight_bits: int | None = None,
    activation_bits: int | None = None,
    kv_cache_bits: int | None = None,
    auto_map_key: str = "AutoModelForCausalLM",
    auto_map_value: str = "modeling_edgerazor.EdgeRazorForCausalLM",
) -> None:
    config_path = target_dir / "config.json"
    if not config_path.exists():
        print(f"  Warning: config.json not found at {config_path}")
        return

    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)

    if 'auto_map' not in cfg:
        cfg['auto_map'] = {}
    cfg['auto_map'][auto_map_key] = auto_map_value

    w_bits, a_bits, kv_bits = _resolve_quant_bits(quant_mode)
    cfg['quantization_config'] = {
        "quant_method": "edgerazor",
        "quant_mode": quant_mode,
        "backend": backend,
        "weight_bits": weight_bits if weight_bits is not None else w_bits,
        "activation_bits": activation_bits if activation_bits is not None else a_bits,
        "kv_cache_bits": kv_cache_bits if kv_cache_bits is not None else kv_bits,
        "is_w_quantized": is_w_quantized,
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(
        f"  Patched config.json (quantization_config.edgerazor, "
        f"quant_mode={quant_mode})"
    )


def _generate_readme(dst: Path, quant_mode: str) -> None:
    readme_path = dst / "README.md"
    if readme_path.exists():
        return
    content = (
        f"---\n"
        f"license: apache-2.0\n"
        f"language: en\n"
        f"tags:\n"
        f"- edgerazor\n"
        f"- quantized\n"
        f"- edge\n"
        f"pipeline_tag: text-generation\n"
        f"---\n\n"
        f"# EdgeRazor Quantized Model ({quant_mode})\n\n"
        f"This model was quantized with "
        f"[EdgeRazor](https://github.com/zhangsq-nju/EdgeRazor).\n"
    )
    readme_path.write_text(content, encoding='utf-8')
    print(f"  Generated README.md")


# ────────────────────────────────────────────────────────────
# Main convert function
# ────────────────────────────────────────────────────────────


def convert(
    *,
    model_path: str | Path,
    save_path: str | Path,
    edgerazor_config: str | Path | None = None,
    quant_mode: str | None = None,
    is_w_quantized: bool = False,
    dtype: str = "bfloat16",
    generate_readme: bool = False,
    backend: str = "marlin",
    weight_bits: int | None = None,
    activation_bits: int | None = None,
    kv_cache_bits: int | None = None,
) -> Path:
    """Convert an unquantized HF model to an EdgeRazor checkpoint.

    Args:
        model_path: Directory containing the unquantized HF model.
        save_path: Directory where the quantized model will be saved.
        edgerazor_config: Path to an EdgeRazor QAT config (.yaml, .yml, .json).
        quant_mode: Preset name, e.g. ``"w4a8kv8"``.  Takes precedence over
            *edgerazor_config* for the ``quantization_config.quant_mode`` field
            in the output ``config.json``.
        is_w_quantized: If ``True``, skip ``replace_quantized_weights`` and
            copy the original weights (export / fake-quant mode).  If ``False``
            (default), apply and save the quantized weights.
        dtype: Torch dtype for loading (default: ``"bfloat16"``).
        generate_readme: Whether to generate a minimal ``README.md``.
        backend: Backend name written to ``quantization_config.backend``
            (default: ``"marlin"``).
        weight_bits: Override weight bit-width in ``quantization_config``.
            Auto-detected from *quant_mode* when not given.
        activation_bits: Override activation bit-width.
        kv_cache_bits: Override KV-cache bit-width.

    Returns:
        Path to the output directory.
    """
    model_path = Path(model_path)
    save_path = Path(save_path)
    torch_dtype = getattr(torch, dtype)

    _validate_model_path(model_path)
    _check_save_path(save_path)

    # --- Resolve EdgeRazor config ---
    er_config: EdgeRazorConfig | None = None
    effective_quant_mode: str | None = quant_mode

    if edgerazor_config is not None:
        config_path = Path(edgerazor_config)
        _validate_config_path(config_path)
        print(f"Loading EdgeRazor config from {config_path} ...")
        er_config = EdgeRazorConfig.load(str(config_path))

        # Auto-detect quant_mode when not given
        if effective_quant_mode is None:
            effective_quant_mode = _infer_quant_mode_from_config(er_config)
            print(f"  Auto-detected quant_mode: {effective_quant_mode}")

    if quant_mode is not None:
        print(f"Using quant_mode preset: {quant_mode}")

    if effective_quant_mode is None:
        raise ValueError(
            "Either --quant_mode or --edgerazor_config must be provided."
        )

    hf_logging.set_verbosity_error()

    # --- Load and quantize ---
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()

    print("Applying EdgeRazor quantization ...")
    # Build EdgeRazor from config or preset
    if er_config is not None:
        edgerazor = EdgeRazor(config=er_config)
    else:
        edgerazor = EdgeRazor(config={
            "qat_configuration": {"function": {}, "select": {}},
        })

    qmodel = edgerazor.quantize(model)

    if is_w_quantized:
        print("is_w_quantized=True — skipping replace_quantized_weights")
    else:
        print("Replacing weights with quantized values ...")
        qmodel = edgerazor.replace_quantized_weights(qmodel)

    # --- Assemble output ---
    save_path.mkdir(parents=True, exist_ok=True)

    if is_w_quantized:
        # Copy original weights and other assets
        print(f"Copying model assets to {save_path} ...")
        _copy_safetensors(model_path, save_path)
        _copy_model_assets(model_path, save_path)
    else:
        print(f"Copying model assets to {save_path} ...")
        _copy_model_assets(model_path, save_path)
        print("Saving fake-quant weights ...")
        _save_fake_quant_weights(qmodel, save_path)

    # Copy tokenizer + generation files if not already there
    _copy_orig_files(model_path, save_path)

    # Copy config.json from source if not present
    if (model_path / 'config.json').exists() and \
       not (save_path / 'config.json').exists():
        shutil.copy2(model_path / 'config.json', save_path / 'config.json')
        print("  Copied config.json")

    # Patch config.json
    _patch_config_json(
        save_path,
        effective_quant_mode,
        is_w_quantized=is_w_quantized,
        backend=backend,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        kv_cache_bits=kv_cache_bits,
    )

    # Copy modeling_edgerazor.py template
    _copy_template(save_path, 'modeling_edgerazor.py', overwrite=True)

    # Optional README
    if generate_readme:
        _generate_readme(save_path, effective_quant_mode)

    print(f"Done - model saved to {save_path}")
    print(f"Upload with: huggingface-cli upload USER/REPO {save_path}")
    return save_path


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m edgerazor.convert",
        description=(
            "Convert an unquantized HF CausalLM to an EdgeRazor-quantized "
            "checkpoint."
        ),
    )
    # Source
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model_path", type=str,
        help="Directory containing the unquantized HF model.",
    )
    src.add_argument(
        "--from_training_output", type=str, dest="training_output",
        help="Path to training output directory (contains model + tokenizer).",
    )

    # Destination
    p.add_argument(
        "--save_path", type=str, required=True,
        help="Directory where the quantized model will be saved.",
    )

    # Config (one or both)
    p.add_argument(
        "--edgerazor_config", type=str, default=None,
        help="Path to EdgeRazor QAT config (.yaml, .yml, .json).",
    )
    p.add_argument(
        "--quant_mode", type=str, default=None,
        help="EdgeRazor quant preset (e.g. w1_58a8kv8_embint4).",
    )

    # Behaviour
    p.add_argument(
        "--is_w_quantized", type=lambda s: s.lower() == "true", default=False,
        help="If true, skip replace_quantized_weights (default: false).",
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for loading (default: bfloat16).",
    )
    p.add_argument(
        "--readme", action="store_true",
        help="Generate a minimal README.md.",
    )

    # quantization_config overrides
    p.add_argument(
        "--backend", type=str, default="marlin",
        help="Backend written to quantization_config.backend (default: marlin).",
    )
    p.add_argument(
        "--weight_bits", type=int, default=None,
        help="Override weight bit-width in quantization_config. "
             "Auto-detected from quant_mode when omitted.",
    )
    p.add_argument(
        "--activation_bits", type=int, default=None,
        help="Override activation bit-width in quantization_config.",
    )
    p.add_argument(
        "--kv_cache_bits", type=int, default=None,
        help="Override KV-cache bit-width in quantization_config.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model = args.model_path or args.training_output

    convert(
        model_path=model,
        save_path=args.save_path,
        edgerazor_config=args.edgerazor_config,
        quant_mode=args.quant_mode,
        is_w_quantized=args.is_w_quantized,
        dtype=args.dtype,
        generate_readme=args.readme,
        backend=args.backend,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        kv_cache_bits=args.kv_cache_bits,
    )


if __name__ == "__main__":
    main()
