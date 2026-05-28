"""Export tool for building EdgeRazor-quantized HF model repos.

Additions to ``config.json``:
```
  "auto_map": {
    "AutoModelForCausalLM": "modeling_edgerazor.EdgeRazorForCausalLM"
  },
  "quantization_config": {
    "quant_method": "edgerazor",
    "quant_mode": "w4a8kv8",
    "weight_bits": 4,
    "activation_bits": 8,
    "kv_cache_bits": 8,
    "is_w_quantized": true
  }
```

Usage::

    # CLI — from a training output directory
    python -m edgerazor.export \\
        --from_training_output ./output/final-model \\
        --quant_mode w1_58a8kv8_embint4 \\
        --output ./Qwen3-0.6B-EdgeRazor-1.58bit

    # CLI — from a source model directory
    python -m edgerazor.export \\
        --model ./checkpoints/final \\
        --quant_mode w1_58a8kv8_embint4 \\
        --output ./Qwen3-0.6B-EdgeRazor-1.58bit

    # Python API
    from edgerazor.export import export
    export(
        src_dir="./checkpoints/final",
        dst_dir="./Qwen3-0.6B-EdgeRazor-1.58bit",
        quant_mode="w1_58a8kv8_embint4",
    )

The tool generates:
- ``config.json`` — standard HF config + ``auto_map`` + ``quantization_config``
- ``modeling_edgerazor.py`` — universal loader (copied from templates)
- ``model.safetensors`` — quantized weights (copied / symlinked)
- ``tokenizer.json``, ``tokenizer_config.json``, ``special_tokens_map.json``, etc. (copied from source)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def _get_template_dir() -> Path:
    """Resolve the templates directory shipped with the package."""
    return Path(__file__).resolve().parent / "templates"


def _validate_src_dir(path: Path) -> None:
    """Check that *path* is a directory with model files."""
    if not path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {path}")


def _copy_template(target_dir: Path, filename: str, overwrite: bool = False) -> Path:
    """Copy a template file to the target directory."""
    src = _get_template_dir() / filename
    dst = target_dir / filename
    if dst.exists() and not overwrite:
        print(f"  Skipping {filename} (already exists)")
        return dst
    shutil.copy2(src, dst)
    print(f"  Copied {filename}")
    return dst


def _resolve_quant_bits(quant_mode: str) -> tuple[int, int, int]:
    """Resolve weight / activation / KV cache bits from quant mode name.

    Returns (weight_bits, activation_bits, kv_cache_bits).
    """
    mode = quant_mode.lower()
    # Defaults
    w_bits, a_bits, kv_bits = 4, 0, 0

    if "w1_58" in mode or "w1.58" in mode:
        w_bits = 4  # 1.58-bit degraded to 4-bit packing
    elif "w4" in mode:
        w_bits = 4
    elif "w2" in mode:
        w_bits = 4  # degrade to 4-bit

    if "a8" in mode:
        a_bits = 8
    if "kv8" in mode:
        kv_bits = 8

    return w_bits, a_bits, kv_bits


def _patch_config_json(
    target_dir: Path,
    quant_mode: str,
    is_w_quantized: bool = True,
    auto_map_key: str = "AutoModelForCausalLM",
    auto_map_value: str = "modeling_edgerazor.EdgeRazorForCausalLM",
) -> None:
    """Add ``auto_map`` and ``quantization_config`` to ``config.json``.

    ``quantization_config`` serves as the single source of truth for all
    EdgeRazor-specific settings: quant_method (for vLLM auto-detection),
    quant_mode, bit widths, and is_w_quantized.
    """
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
        "weight_bits": w_bits,
        "activation_bits": a_bits,
        "kv_cache_bits": kv_bits,
        "is_w_quantized": is_w_quantized,
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  Patched config.json (quantization_config.edgerazor, "
          f"quant_mode={quant_mode})")


def _copy_safetensors(src: Path, dst: Path) -> None:
    """Copy or symlink safetensors files."""
    for pattern in ['*.safetensors', '*.bin']:
        for f in src.glob(pattern):
            target = dst / f.name
            if not target.exists():
                shutil.copy2(f, target)
                print(f"  Copied {f.name}")
            else:
                print(f"  Skipping {f.name} (already exists)")


def _copy_orig_files(src: Path, dst: Path) -> None:
    """Copy tokenizer-related files."""
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


def _generate_readme(dst: Path, quant_mode: str) -> None:
    """Generate a minimal README.md."""
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
        f"This model was quantized with [EdgeRazor](https://github.com/zhangsq-nju/EdgeRazor).\n\n"
        f"## Usage\n\n"
        f"```python\n"
        f"from transformers import AutoModelForCausalLM\n"
        f"import torch\n\n"
        f"model = AutoModelForCausalLM.from_pretrained(\n"
        f"    'REPO_NAME',\n"
        f"    trust_remote_code=True,\n"
        f"    torch_dtype=torch.bfloat16,\n"
        f"    device_map='auto',\n"
        f")\n"
        f"```\n\n"
        f"## Quantization Config\n\n"
        f"- Quant mode: `{quant_mode}`\n"
    )
    readme_path.write_text(content, encoding='utf-8')
    print(f"  Generated README.md")


def export(
    *,
    src_dir: str | Path,
    dst_dir: str | Path,
    quant_mode: str,
    is_w_quantized: bool = True,
    copy_weights: bool = True,
    generate_readme: bool = False,
) -> Path:
    """Export an EdgeRazor model repo.

    Args:
        src_dir: Source directory with model files (safetensors, tokenizer, config).
        dst_dir: Destination directory for the HF repo.
        quant_mode: Preset name, e.g. ``"w1_58a8kv8_embint4"``.
        is_w_quantized: Whether weights are already quantized. Written into
            ``quantization_config.is_w_quantized``.
        copy_weights: Whether to copy safetensors/bin weights.
        generate_readme: Whether to generate a minimal README.md.

    Returns:
        Path to the output directory.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)

    _validate_src_dir(src)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Exporting EdgeRazor model:")
    print(f"  Source: {src}")
    print(f"  Destination: {dst}")
    print(f"  Quant mode: {quant_mode}")

    # 1. Copy weights
    if copy_weights:
        _copy_safetensors(src, dst)

    # 2. Copy config.json from source (if not already present)
    if (src / 'config.json').exists() and not (dst / 'config.json').exists():
        shutil.copy2(src / 'config.json', dst / 'config.json')
        print(f"  Copied config.json")

    # 3. Patch config.json with EdgeRazor fields
    _patch_config_json(dst, quant_mode, is_w_quantized=is_w_quantized)

    # 4. Copy modeling_edgerazor.py template
    _copy_template(dst, 'modeling_edgerazor.py', overwrite=True)

    # 5. Copy tokenizer files
    _copy_orig_files(src, dst)

    # 6. Optional README
    if generate_readme:
        _generate_readme(dst, quant_mode)

    print(f"\nDone! Model exported to: {dst}")
    print(f"Upload with: huggingface-cli upload USER/REPO {dst}")
    return dst


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — parse args and delegate to :func:`export`."""
    ap = argparse.ArgumentParser(
        prog='python -m edgerazor.export',
        description='Export an EdgeRazor-quantized HF model repo.',
    )
    # Source
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        '--model', type=str,
        help='Path to source model checkpoint.',
    )
    src_group.add_argument(
        '--from_training_output', type=str, dest='training_output',
        help='Path to training output directory (contains model + tokenizer).',
    )

    # Config
    ap.add_argument(
        '--quant_mode', type=str, required=True,
        help='EdgeRazor quant preset (e.g. w1_58a8kv8_embint4).',
    )
    ap.add_argument(
        '--no_w_quantized', action='store_true',
        help='Set is_w_quantized=False in quantization_config.',
    )
    ap.add_argument(
        '--output', type=str, required=True,
        help='Output directory for the HF model repo.',
    )
    ap.add_argument(
        '--no_copy_weights', action='store_true',
        help='Skip copying weight files.',
    )
    ap.add_argument(
        '--readme', action='store_true',
        help='Generate a minimal README.md.',
    )

    args = ap.parse_args(argv)
    src = args.model or args.training_output

    export(
        src_dir=src,
        dst_dir=args.output,
        quant_mode=args.quant_mode,
        is_w_quantized=not args.no_w_quantized,
        copy_weights=not args.no_copy_weights,
        generate_readme=args.readme,
    )


if __name__ == '__main__':
    main()
