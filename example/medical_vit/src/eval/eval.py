import argparse
import os
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import subprocess
import re


def parser_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ViT on validation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pth_path",
        type=str,
        required=True,
        help="Path to the .pth model file"
    )

    parser.add_argument(
        "--gguf_path",
        type=str,
        default=None,
        help="Path to the .gguf quantized model file. If not provided, evaluation will be performed using the .pth model.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/data",
        help="Path to the dataset root directory containing 'val' subdirectory.",
    )

    return parser.parse_args()

def evaluate(gguf_path, test_dir):
    y_true = []
    y_pred = []

    image_tasks = []

    model_size_pattern = r"model size\s*=\s*([\d\.]+)\s*MB"
    load_time_pattern = r"model load time\s*=\s*([\d\.]+)\s*ms"
    process_time_pattern = r"processing time\s*=\s*([\d\.]+)\s*ms"

    model_size_mbs = []
    load_time_mss = []
    inference_time_mss = []

    for label_dir in sorted(os.listdir(test_dir)):
        full_label_path = os.path.join(test_dir, label_dir)
        if os.path.isdir(full_label_path):
            true_label = int(label_dir)
            for img_name in os.listdir(full_label_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(full_label_path, img_name)
                    image_tasks.append((img_path, true_label))

    for img_path, true_label in tqdm(image_tasks, desc="Evaluating"):
        cmd = ["/opt/code-dependency/vit.cpp/build/bin/vit", "-t", "4", "-m", gguf_path, "-i", img_path]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        pred_label = None
        if result.returncode != 0:
            print(f"Error processing {img_path}: {result.stderr}")
        match = re.search(r">\s*LABEL_(\d+)\s*:", result.stdout)
        if match:
            pred_label = int(match.group(1))
        
        if pred_label is not None:
            y_true.append(true_label)
            y_pred.append(pred_label)
        else:
            print(f"Could not extract prediction for {img_path}")

        output = result.stderr
        size_match = re.search(model_size_pattern, output)
        load_match = re.search(load_time_pattern, output)
        process_match = re.search(process_time_pattern, output)
        if size_match and load_match and process_match:
            model_size_mbs.append(float(size_match.group(1)))
            load_time_mss.append(float(load_match.group(1)))
            inference_time_mss.append(float(process_match.group(1)))
        
    model_size_mb = sum(model_size_mbs) / len(model_size_mbs) if model_size_mbs else 1e9
    load_time_ms = sum(load_time_mss) / len(load_time_mss) if load_time_mss else 1e9
    inference_time_ms = sum(inference_time_mss) / len(inference_time_mss) if inference_time_mss else 1e9

    
    if not y_true:
        print("No valid predictions were made. Check the output of the vit inference.")
        return 0, 0, 0, 0, 0
    
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return accuracy, precision, recall, f1, model_size_mb, load_time_ms, inference_time_ms


def convert_model(pth_path, output_path):
    """
    Convert .pth model to vit.cpp compatible format .gguf
    """
    print(f"Convert Model: {pth_path} ...")
    convert_script_path = "../quantize/convert-pth-to-gguf.py"
    cmd = ["python", convert_script_path, "--model_name", "vit_small_patch16_224", "--ckpt_path", pth_path, "--output_path", output_path]
    
    with open(output_path, "wb") as f:
        subprocess.run(cmd, stdout=f, check=True)
    
    print(f"Convert successfully, output gguf: {output_path}")
    return output_path


def main():
    args = parser_args()

    # Load model
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=5)

    if not os.path.isfile(args.pth_path):
        print(f"Error: .pth file not found at {args.pth_path}")
        return

    checkpoint = torch.load(args.pth_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    print("load model successfully")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # acc eval
    data_root = args.data_path

    # Convert to .gguf if path not provided
    if not args.gguf_path:
        print("No .gguf path provided, skipping quantized model evaluation.")

        # Convert model to .gguf
        gguf_path = convert_model(args.pth_path, "./q_vit_w4_a16.gguf")
    else:
        gguf_path = args.gguf_path

    test_dir = os.path.join(data_root, "val")
    accuracy, precision, recall, f1, model_size_mb, load_time_ms, inference_time_ms = evaluate(gguf_path, test_dir)
    
    # get model performance
    print("Model Effectiveness Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("Model Efficiency Metrics:")
    print(f"  Model Size (MB): {model_size_mb}")
    print(f"  Load Time (ms): {load_time_ms}")
    print(f"  Inference Time (ms): {inference_time_ms}")
    print(f"  Total Latency (ms): {load_time_ms + inference_time_ms}")


if __name__ == "__main__":
    main()
