"""
Argument parser for ViT training on MNIST with QAT support.
"""
import argparse
import time
from pathlib import Path


def parse_args():
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train ViT on MNIST with optional Quantization Aware Training (QAT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Quantization configuration
    parser.add_argument(
        "--quant_config",
        type=str,
        default=None,
        help="Path to quantization configuration YAML file. If not provided, train without quantization.",
    )
    
    parser.add_argument(
        "--kd_config",
        type=str,
        default=None,
        help="Path to knowledge distillation configuration YAML file. If not provided, train without KD.",
    )
    
    parser.add_argument(
        "--edgerazor_config",
        type=str,
        default=None,
        help="Path to unified EdgeRazor configuration YAML file (for QAT + KD). If provided, overrides --quant_config and --kd_config.",
    )
    
    parser.add_argument(
        "--teacher_pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained teacher model checkpoint for knowledge distillation.",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and evaluation")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")

    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay coefficient for AdamW optimizer")

    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")

    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing")

    # Data configuration
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for MNIST dataset")

    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="../../../model", help="Directory for saving model checkpoints and logs")
    
    parser.add_argument("--detailed_run_name", action="store_true", help="If set, use detailed run name with timestamp and hyperparameters for run identification")

    parser.add_argument("--save_freq", type=int, default=1, help="Save model checkpoint every N epochs")

    parser.add_argument("--save_total_limit", type=int, default=1, help="Maximum number of checkpoint files to keep (excluding best model)")

    parser.add_argument("--save_optimizer_state", action="store_true", help="Save optimizer and scheduler states in checkpoints (for resuming training)")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs with no improvement after which training will be stopped (0 to disable)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training")

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Data type for model weights and computations")

    # Logging
    parser.add_argument("--log_interval", type=int, default=50, help="Log training metrics every N batches")

    parser.add_argument("--no_tensorboard", action="store_true", help="Disable TensorBoard logging")

    args = parser.parse_args()

    # Convert paths to Path objects
    args.data_root = Path(args.data_root)
    args.output_dir = Path(args.output_dir)

    if args.quant_config is not None:
        args.quant_config = Path(args.quant_config)
    
    if args.kd_config is not None:
        args.kd_config = Path(args.kd_config)
    
    if args.edgerazor_config is not None:
        args.edgerazor_config = Path(args.edgerazor_config)
    
    if args.teacher_pretrained_path is not None:
        args.teacher_pretrained_path = Path(args.teacher_pretrained_path)

    if args.quant_config is not None and not args.quant_config.exists():
        raise FileNotFoundError(f"Quantization config file not found: {args.quant_config}")
    
    if args.kd_config is not None and not args.kd_config.exists():
        raise FileNotFoundError(f"KD config file not found: {args.kd_config}")
    
    if args.edgerazor_config is not None and not args.edgerazor_config.exists():
        raise FileNotFoundError(f"EdgeRazor config file not found: {args.edgerazor_config}")
    
    if args.teacher_pretrained_path is not None and not args.teacher_pretrained_path.exists():
        raise FileNotFoundError(f"Teacher pretrained checkpoint not found: {args.teacher_pretrained_path}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args


def get_run_name(args):
    """
    Generate a descriptive name for the training run.

    Args:
        args: Parsed command-line arguments

    Returns:
        str: Run name for logging and checkpointing
    """
    # If detailed_run_name is not set, use only the config stem
    if not args.detailed_run_name:
        if args.edgerazor_config is not None:
            return args.edgerazor_config.stem
        elif args.quant_config is not None:
            return args.quant_config.stem
        elif args.kd_config is not None:
            return args.kd_config.stem
        else:
            return "fp_vit_w16_a16"
    
    # Otherwise, use detailed run name with timestamp and hyperparameters
    # Base name with timestamp
    run_name = time.strftime("%Y%m%d_%H%M%S")

    # Add key hyperparameters
    run_name += f"_bs{args.batch_size}"
    run_name += f"_lr{args.lr}"
    run_name += f"_wd{args.weight_decay}"

    # Add quantization info if applicable
    if args.edgerazor_config is not None:
        # EdgeRazor unified config has highest priority
        config_stem = args.edgerazor_config.stem
        run_name += f"_{config_stem}"
    elif args.quant_config is not None:
        # QAT-only config
        config_stem = args.quant_config.stem
        run_name += f"_{config_stem}"
    elif args.kd_config is not None:
        # KD-only config
        config_stem = args.kd_config.stem
        run_name += f"_{config_stem}"
    else:
        run_name += "_fp_vit_w16_a16"  # Full precision

    return run_name
