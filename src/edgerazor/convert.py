"""Convert an unquantized HF model to a fake-quantized W-A16-KV16 checkpoint.

This tool applies EdgeRazor's weight quantization to a standard model and
saves the resulting fake-quant weights as a new safetensors file, while
preserving all other model assets (config, tokenizer, etc.).

Usage::

    # CLI
    python -m edgerazor.convert \\
        --model_path ./Qwen3-0.6B \\
        --save_path ./Qwen3-0.6B-W4A16 \\
        --edgerazor_config configs/w4a16kv16_qwen3.yaml

    # Python API
    from edgerazor.convert import convert
    convert(
        model_path="./Qwen3-0.6B",
        save_path="./Qwen3-0.6B-W4A16",
        edgerazor_config="configs/w4a16kv16_qwen3.yaml",
    )

The output directory contains everything needed to load the model with
``AutoModelForCausalLM.from_pretrained(save_path, trust_remote_code=True)``
when combined with the universal ``modeling_edgerazor.py`` loader.

Notes:
    - Only weight quantization (WX) is applied.  Activation and KV-cache
      quantization require the v1.3.4+ ``modeling_edgerazor.py`` +
      ``quantization_config`` inference pipeline.
    - The input model is loaded at the specified ``dtype`` (default bfloat16).
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.utils import logging as hf_logging

from edgerazor import EdgeRazor


def _validate_model_path(path: Path) -> None:
    """Check that *path* is a directory containing HF model weights."""
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
    """Check that *path* is a valid EdgeRazor config file."""
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
    """Warn if *path* exists and is non-empty."""
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
    """Save fake-quant safetensors into *save_path* via a temp directory."""
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
        src_index = tmp / "model.safetensors.index.json"
        if src_index.exists():
            shutil.move(
                str(src_index),
                str(save_path / "model.safetensors.index.json"),
            )
        print(f"Moved model.safetensors to {save_path}")


def convert(
    *,
    model_path: str | Path,
    save_path: str | Path,
    edgerazor_config: str | Path,
    dtype: str = "bfloat16",
) -> Path:
    """Convert an unquantized HF model to a fake-quantized checkpoint.

    Args:
        model_path: Directory containing the unquantized HF model.
        save_path: Directory where the quantized model will be saved.
        edgerazor_config: Path to the EdgeRazor QAT config (.yaml, .yml, or .json).
        dtype: Torch dtype for loading (default: ``"bfloat16"``).

    Returns:
        Path to the output directory.
    """
    model_path = Path(model_path)
    save_path = Path(save_path)
    config_path = Path(edgerazor_config)
    torch_dtype = getattr(torch, dtype)

    _validate_model_path(model_path)
    _validate_config_path(config_path)
    _check_save_path(save_path)

    hf_logging.set_verbosity_error()

    # 1. Load unquantized model
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()

    # 2. Apply EdgeRazor fake-quant
    print(f"Applying EdgeRazor config from {config_path} ...")
    edgerazor = EdgeRazor(config=str(config_path))
    qmodel = edgerazor.quantize(model)
    qmodel = edgerazor.replace_quantized_weights(qmodel)

    # 3. Assemble output directory
    print(f"Copying model assets to {save_path} ...")
    _copy_model_assets(model_path, save_path)

    print("Saving fake-quant weights ...")
    _save_fake_quant_weights(qmodel, save_path)

    print(f"Done - quantized model saved to {save_path}")
    return save_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m edgerazor.convert",
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
        help="Torch dtype for loading (default: bfloat16).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point - parse args and delegate to :func:`convert`."""
    args = parse_args(argv)
    convert(
        model_path=args.model_path,
        save_path=args.save_path,
        edgerazor_config=args.edgerazor_config,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
