from arg import parse_args, get_run_name
import os
from pathlib import Path
import torch
import numpy as np
import json
from tqdm import tqdm
from transformers import ViTConfig, ViTModel
import timm
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score
from medmnist import RetinaMNIST

from edgerazor import EdgeRazor

class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
        )
        self.current_step = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.base_lrs[i] * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            self.cosine_scheduler.step()

    def get_lr(self):
        """Get current learning rate."""
        return [group["lr"] for group in self.optimizer.param_groups]

def init_weights_kaiming(module):
    """
    Initialize model weights using Kaiming initialization.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        # Kaiming initialization for convolutional layers
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        # Standard initialization for normalization layers
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def cleanup_checkpoints(checkpoint_dir, save_total_limit):
    """
    Remove old checkpoint files, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        save_total_limit: Maximum number of checkpoint files to keep (excluding best_model.pth)
    """
    if save_total_limit is None or save_total_limit <= 0:
        return
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Get all epoch checkpoint files (exclude best_model.pth)
    epoch_checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pth"),
        key=lambda x: x.stat().st_mtime,  # Sort by modification time
        reverse=True  # Newest first
    )
    
    # Remove old checkpoints if exceeding limit
    if len(epoch_checkpoints) > save_total_limit:
        for checkpoint_to_remove in epoch_checkpoints[save_total_limit:]:
            checkpoint_to_remove.unlink()

def prepare_dataloaders(data_root, batch_size, num_workers=4):
    """Prepare MNIST dataloaders with preprocessing for ViT."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to ViT input size
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            GrayscaleToRGB(),  # Convert grayscale to RGB
        ]
    )

    train_dataset = RetinaMNIST(split="train", download=True, transform=transform)
    test_dataset = RetinaMNIST(split="val", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, global_step, args, edgerazor=None, teacher_model=None):
    """Train for one epoch."""
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    
    running_loss = 0.0
    running_task_loss = 0.0
    running_distill_loss = 0.0
    correct = 0
    total = 0
    
    # Check if KD is enabled
    use_kd = edgerazor is not None and edgerazor.is_kd_enabled and teacher_model is not None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        images = images.to(dtype=next(model.parameters()).dtype)  # Convert images to model's weight dtype for consistency
        labels = labels.squeeze()  # Ensure labels are of shape (batch_size,)


        optimizer.zero_grad()

        if use_kd:
            # Knowledge distillation mode: get full outputs
            student_outputs = model(
                images,
                labels=labels,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True
            )
            
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    images,
                    labels=labels,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True
                )
            
            # Compute loss using EdgeRazor (includes task loss + distillation loss)
            # Note: ViT doesn't use attention_mask (no padding in image patches) => labels=None
            loss, loss_dict = edgerazor.compute_loss(
                student_outputs,
                teacher_outputs,
                labels=None,
            )
            
            task_loss_value = loss_dict.get('task_loss', 0.0)
            distill_loss_value = loss_dict.get('distill_loss', 0.0)
            running_task_loss += task_loss_value
            running_distill_loss += distill_loss_value
            
            # Get logits for accuracy calculation
            outputs = student_outputs['logits']
        else:
            # Standard training mode
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        batch_correct = predicted.eq(labels).sum().item()
        batch_total = labels.size(0)
        batch_acc = batch_correct / batch_total

        running_loss += loss.item()
        correct += batch_correct
        total += batch_total

        # Update progress bar
        pbar_info = {
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.4f}",
            "lr": f"{scheduler.get_lr()[0]:.6f}",
        }
        if use_kd:
            pbar_info["task"] = f"{task_loss_value:.4f}"
            pbar_info["dist"] = f"{distill_loss_value:.4f}"
        
        pbar.set_postfix(pbar_info)

        global_step += 1

    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    if use_kd:
        epoch_task_loss = running_task_loss / len(train_loader)
        epoch_distill_loss = running_distill_loss / len(train_loader)
        print(
            f"Epoch {epoch+1:02d} Training   - Loss: {epoch_loss:.4f} "
            f"(Task: {epoch_task_loss:.4f}, Distill: {epoch_distill_loss:.4f}), "
            f"Accuracy: {epoch_acc:.4f} ({correct}/{total})"
        )
    else:
        print(f"Epoch {epoch+1:02d} Training   - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f} ({correct}/{total})")

    return epoch_loss, epoch_acc, global_step

def evaluate(model, dataloader, criterion, device, epoch, args, split="Test"):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [{split}]")
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images = images.to(dtype=next(model.parameters()).dtype)  # Convert images to model's weight dtype for consistency
            labels = labels.squeeze()  # Ensure labels are of shape (batch_size,)

            # Get outputs
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

    # Calculate metrics
    loss = running_loss / len(dataloader)
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"Epoch {epoch+1:02d} {split:<8} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'':>16}Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return loss, accuracy, precision, recall, f1

def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=5)
    model_path = "../../../model/model.pth"
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    
    # Initialize Edgerazor (QAT and/or KD)
    edgerazor = None
    if args.edgerazor_config is not None:
        print("Initializing EdgeRazor (Unified QAT + KD)")
        edgerazor = EdgeRazor(config=args.edgerazor_config)

        # Apply QAT if enabled
        if edgerazor.is_qat_enabled:
            print("Applying Quantization Aware Training (QAT)...")
            model = edgerazor.quantize(model)
            print("✓ QAT applied")
        else:
            print("QAT: disabled")
        
        # Log KD status
        if edgerazor.is_kd_enabled:
            print("Knowledge Distillation (KD): enabled")
        else:
            print("Knowledge Distillation (KD): disabled")
    elif args.quant_config is not None:
        print("Initializing EdgeRazor (QAT only)")

        edgerazor = EdgeRazor(qat_config=args.quant_config)
        model = edgerazor.quantize(model)
    elif args.kd_config is not None:
        print("Initializing EdgeRazor (KD only)")

        edgerazor = EdgeRazor(kd_config=args.kd_config)
    else:
        print("Training with full precision (no quantization or distillation)")
    
    # Convert model to specified dtype and move to device
    model = model.to(device).to(dtype)

    # Create Teacher model if KD is enabled
    teacher_model = None
    if edgerazor is not None and edgerazor.is_kd_enabled:
        print("Creating Teacher Model for Knowledge Distillation")

        
        # Create teacher model with same architecture
        teacher_model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=5)
        teacher_model.apply(init_weights_kaiming)
        
        # Load pretrained weights if provided
        if args.teacher_pretrained_path is not None:
            print(f"Loading pretrained teacher weights from: {args.teacher_pretrained_path}")
            
            try:
                # Load checkpoint with weights_only=False to support metadata (e.g., args with Path objects)
                # This is safe when loading checkpoints from trusted sources
                checkpoint = torch.load(args.teacher_pretrained_path, map_location="cpu", weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        print("  Loaded from checkpoint format (key: 'model_state_dict')")
                        
                        # Log checkpoint metadata if available
                        if "epoch" in checkpoint:
                            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
                        if "val_acc" in checkpoint:
                            print(f"  Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
                    else:
                        state_dict = checkpoint
                        print("  Loaded from state dict format")
                else:
                    state_dict = checkpoint
                    print("  Loaded from state dict format")
                
                # Load weights into teacher model
                missing_keys, unexpected_keys = teacher_model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"  Missing keys in teacher checkpoint: {len(missing_keys)}")
                    if len(missing_keys) <= 10:
                        for key in missing_keys:
                            print(f"    - {key}")
                    else:
                        for key in missing_keys[:5]:
                            print(f"    - {key}")
                        print(f"    ... and {len(missing_keys) - 5} more")
                
                if unexpected_keys:
                    print(f"  Unexpected keys in teacher checkpoint: {len(unexpected_keys)}")
                    if len(unexpected_keys) <= 10:
                        for key in unexpected_keys:
                            print(f"    - {key}")
                    else:
                        for key in unexpected_keys[:5]:
                            print(f"    - {key}")
                        print(f"    ... and {len(unexpected_keys) - 5} more")
                
                if not missing_keys and not unexpected_keys:
                    print("  ✓ All keys matched successfully")
                
                print("✓ Pretrained teacher weights loaded successfully")
                
            except FileNotFoundError:
                print(f"✗ Teacher checkpoint not found: {args.teacher_pretrained_path}")
                print("  Training will continue with randomly initialized teacher model")
            except Exception as e:
                print(f"✗ Error loading teacher checkpoint: {e}")
                print("  Training will continue with randomly initialized teacher model")
        else:
            print("No pretrained teacher weights provided (--teacher_pretrained_path not set)")
            print("Teacher model initialized with random weights")
        
        teacher_model = teacher_model.to(device=device, dtype=dtype)
        teacher_model.eval()
        
        # Count teacher parameters
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print("Teacher model summary:")
        print(f"  Total parameters: {teacher_params:,}")

    # Display actual weight dtype
    sample_param = next(model.parameters())
    print("Model dtype information:")
    print(f"  Weight dtype: {sample_param.dtype}")
    print(f"  Weight device: {sample_param.device}")
    print("")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print("")

    # Prepare data
    print("Preparing MNIST dataloaders...")
    train_loader, test_loader = prepare_dataloaders(args.data_root, args.batch_size, args.num_workers)
    print(f"  Training samples: {len(train_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print("")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps, min_lr=args.min_lr
    )

    print("Training setup:")
    print("  Optimizer: AdamW")
    print("  Loss function: CrossEntropyLoss")
    print("  Scheduler: Warmup + Cosine Annealing")
    print(f"  Total steps: {total_steps:,}")

    # Generate run name once for consistent directory naming
    run_name = get_run_name(args)

    # Training loop
    print("Starting training...")

    global_step = 0
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0  # Early stopping counter

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, global_step, args,
            edgerazor=edgerazor, teacher_model=teacher_model
        )

        # Evaluate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, test_loader, criterion, device, epoch, args, split="Test"
        )

        # Check for improvement and update early stopping counter
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_dir = args.output_dir / "checkpoints" / run_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Build checkpoint dictionary
            checkpoint = model.state_dict()

            if is_best:
                checkpoint_path = checkpoint_dir / "best_model.pth"
                print(f"✓ New best accuracy: {best_acc:.4f}, saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
            
            checkpoint_path = checkpoint_dir / f"epoch_{epoch+1:02d}.pth"
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            
            cleanup_checkpoints(checkpoint_dir, args.save_total_limit)
            
        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
            print(f"Best validation accuracy: {best_acc:.4f} (Epoch {best_epoch})")
            break

    # Final summary
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.4f} (Epoch {best_epoch})")

if __name__ == "__main__":
    main()

