## Instructions

Table of Contents

- [Instructions](#instructions)
- [Installation](#installation)
  - [Install EdgeRazor](#install-edgerazor)
  - [Install vit.cpp and compile](#install-vitcpp-and-compile)
- [Train and Save the model](#train-and-save-the-model)
  - [Train baseline (W16-A16)](#train-baseline-w16-a16)
  - [Train quantized model (W8-A16, QAT)](#train-quantized-model-w8-a16-qat)
  - [Train quantized model (W4-A16, QAT)](#train-quantized-model-w4-a16-qat)
- [Quantize .pth to .gguf](#quantize-pth-to-gguf)
- [Inference / Demo with .gguf](#inference--demo-with-gguf)
- [Evaluation](#evaluation)

## Installation

### Install EdgeRazor

Install EdgeRazor with the following steps:

```
cd EdgeRazor
conda create -n cacc_eval python=3.10.19 -y
conda activate cacc_eval
pip install -e .[cu128]
cd example/medical_vit
pip install -r requirements.txt
```

### Install vit.cpp and compile

Install and set up vit.cpp with the following steps:

```
cd vit.cpp
python convert-pth-to-ggml.py --list
```

Then compile ggml:

```
# build ggml and vit 
mkdir build && cd build
cmake .. && make -j4
```

Run inference to test the installation:

```
# run inference
./bin/vit -t 4 -m ../ggml-model-f16.gguf -i ../assets/tench.jpg
```

Run quantization and its inference:

```
./bin/quantize ../ggml-model-f16.gguf ../ggml-model-q4_0.gguf 2
./bin/vit -t 4 -m ../ggml-model-q4_0.gguf -i ../assets/tench.jpg
```

## Train and Save the model

The `src/train/` directory contains training code and scripts, supporting both full-precision training and quantization-aware training (QAT).

Default data and output paths in scripts can be modified in `src/train/train.sh`.

> The core goal is to choose appropriate bit-widths (**W**eight[X]-**A**ctivation[X]), implement quantization algorithms, train to obtain .pth weights, then convert and quantize them into GGUF format for edge inference.

### Train baseline (W16-A16)

```
cd src/train
bash train.sh
```

After training, files are saved under `model/checkpoints/<run_name>/` in the project root:
- `best_model.pth`
- `epoch_XX.pth`

`<run_name>` defaults to `fp_vit_w16_a16`, and can also be auto-generated via config or `--detailed_run_name`.

### Train quantized model (W8-A16, QAT)

> ⚠️ Read `Quantization.md` first and complete the quantization function implementation.

You can directly run the quantized training section in `train.sh`, or execute manually:

```
cd src/train
python -m train \
  --batch_size 128 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 0.2 \
  --warmup_steps 320 \
  --output_dir /path/to/model \
  --early_stopping_patience 5 \
  --quant_config ./q_vit_w8_a16.yaml \
  --data_root /path/to/data_root
```

### Train quantized model (W4-A16, QAT)

> ⚠️ Read `Quantization.md` to understand how quantization functions are implemented.

You can modify the config in `train.sh` and run quantized training, or execute manually:

```
cd src/train
python -m train \
  --batch_size 128 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 0.2 \
  --warmup_steps 320 \
  --output_dir /path/to/model \
  --early_stopping_patience 5 \
  --quant_config /path/to/q_vit_w4_a16.yaml \
  --data_root /path/to/data_root
```

Weight quantization config for QAT is in `src/train/q_vit_w4_a16.yaml`.

## Quantize .pth to .gguf

The weight conversion scripts are in `src/quantize/`:

```
cd src/quantize
bash convert-pth-to-gguf.sh
```

By default, it converts `/path/to/model/run/checkpoints/fp_vit_w16_a16/best_model.pth` to `fp_model.gguf`.
To convert another model, modify `CKPT_PATH` and `OUTPUT_PATH` in the script.

For further quantization (gguf -> int4), run:

```
cd src/quantize
bash quantize.sh
```

This outputs `q_vit_w4_a16-quant.gguf`.

## Inference / Demo with .gguf

Run inference using `vit.cpp` (compile first):

```
cd vit.cpp/build/bin
./vit \
  -t 4 \
  -m GGUF_PATH \
  -i /path/to/img.png
```

You can also use `/opt/code-dependency/vit.cpp` for inference (already compiled):

```bash
cd /opt/code-dependency/vit.cpp/build/bin
./vit \
  -t 4 \
  -m GGUF_PATH \
  -i /path/to/img.png
```

## Evaluation

Evaluation scripts are in `src/eval/`. They report metrics such as accuracy and F1, and call vit.cpp to measure model size and inference time.

```
cd src/eval
bash eval.sh
```

Or specify arguments manually:

```
cd src/eval
python eval.py \
  --pth_path PTH_MODEL_PATH \
  --gguf_path GGUF_MODEL_PATH
```
