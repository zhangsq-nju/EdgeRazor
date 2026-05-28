"""Convert an unquantized HF model to a fake-quantized W-A16-KV16 checkpoint.

This tool applies EdgeRazor's weight quantization to a standard model and
saves the resulting fake-quant weights as a new safetensors file, while
preserving all other model assets (config, tokenizer, etc.).

Typical usage::

    python -m toolkit.convert_unquant_to_quant \\
        --model_path ./Qwen3-0.6B \\
        --save_path ./Qwen3-0.6B-W4A16 \\
        --edgerazor_config configs/w4a16kv16_qwen3.yaml

The output directory contains everything needed to load the model with
``AutoModelForCausalLM.from_pretrained(save_path, trust_remote_code=True)``
when combined with the universal ``modeling_edgerazor.py`` loader.

Notes:
    - Only weight quantization (WX) is applied.  Activation and KV-cache
      quantization require the v1.3.4+ ``modeling_edgerazor.py`` +
      ``quantization_config`` inference pipeline.
    - The input model is loaded at ``torch.bfloat16``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.utils import logging as hf_logging

from edgerazor import EdgeRazor


def _validate_args(args: argparse.Namespace) -> None:
    """Validate parsed CLI arguments and emit clear errors for bad inputs."""
    model_path = Path(args.model_path)
    save_path = Path(args.save_path)
    config_path = Path(args.edgerazor_config)

    # --- model_path ---
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")
    safetensors = model_path / "model.safetensors"
    index_json = model_path / "model.safetensors.index.json"
    if not safetensors.exists() and not index_json.exists():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json found "
            f"in {model_path}. Is this a valid HF model directory?"
        )

    # --- edgerazor_config ---
    if not config_path.exists():
        raise FileNotFoundError(
            f"EdgeRazor config file not found: {config_path}"
        )
    if not config_path.is_file():
        raise IsADirectoryError(
            f"EdgeRazor config path is a directory, not a file: {config_path}"
        )
    if config_path.suffix not in (".yaml", ".yml", ".json"):
        raise ValueError(
            f"EdgeRazor config must be .yaml/.yml/.json, got: {config_path.suffix}"
        )

    # --- save_path (warn if already exists and is non-empty) ---
    if save_path.exists():
        if not save_path.is_dir():
            raise NotADirectoryError(
                f"Save path exists but is not a directory: {save_path}"
            )
        contents = list(save_path.iterdir())
        if contents:
            print(
                f"Warning: save_path {save_path} already exists and contains "
                f"{len(contents)} item(s). Existing files may be overwritten.",
                file=sys.stderr,
            )


def _copy_model_assets(src: Path, dst: Path) -> None:
    """Copy all files and directories from *src* to *dst*, skipping weights."""
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
    qmodel: AutoModelForCausalLM, save_path: Path
) -> None:
    """Save only the fake-quant safetensors into *save_path* via a temp dir."""
    qmodel.to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print("Saving fake-quant weights to temporary directory ...")
        qmodel.save_pretrained(str(tmp))
        src_safetensors = tmp / "model.safetensors"
        if not src_safetensors.exists():
            raise RuntimeError(
                "save_pretrained did not produce model.safetensors"
            )
        shutil.move(str(src_safetensors), str(save_path / "model.safetensors"))
        # Also move index if present (sharded models)
        src_index = tmp / "model.safetensors.index.json"
        if src_index.exists():
            shutil.move(str(src_index), str(save_path / "model.safetensors.index.json"))
        print(f"Moved model.safetensors to {save_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m toolkit.convert_unquant_to_quant",
        description=(
            "Convert an unquantized HF CausalLM to a WX quantized checkpoint."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        metavar="PATH",
        help="Directory containing the unquantized HF model.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        metavar="PATH",
        help="Directory where the quantized model will be saved.",
    )
    parser.add_argument(
        "--edgerazor_config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the EdgeRazor QAT config (.yaml, .yml, or .json).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        metavar="DTYPE",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for loading and saving (default: bfloat16).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- validate inputs ---
    _validate_args(args)

    model_path = Path(args.model_path)
    save_path = Path(args.save_path)
    config_path = Path(args.edgerazor_config)
    dtype = getattr(torch, args.dtype)

    # Suppress verbose HF warnings
    hf_logging.set_verbosity_error()

    # --- 1. Load unquantized model ---
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval()

    # --- 2. Apply EdgeRazor fake-quant ---
    print(f"Applying EdgeRazor config from {config_path} ...")
    edgerazor = EdgeRazor(config=str(config_path))
    qmodel = edgerazor.quantize(model)
    qmodel = edgerazor.replace_quantized_weights(qmodel)

    # --- 3. Assemble output directory ---
    print(f"Copying model assets to {save_path} ...")
    _copy_model_assets(model_path, save_path)

    print("Saving fake-quant weights ...")
    _save_fake_quant_weights(qmodel, save_path)

    print(f"Done — quantized model saved to {save_path}")


if __name__ == "__main__":
    main()
