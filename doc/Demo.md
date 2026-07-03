## 1. Deploy on Edge Devices using llama.cpp

The checkpoints trained by EdgeRazor are compatible with GGUF formats to facilitate edge deployment.

### Model List

- https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-GGUF
- https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF

### Model Formats

- Q4_0: support CPU and GPU
- TQ1_0, TQ2_0: support CPU

### PC and Android

<div align="center">
    <img src="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Demo.gif" alt="EdgeRazor Demo" width="100%">
</div>

## 2. Deploy on Edge GPU Devices using vLLM

The checkpoints trained by EdgeRazor are compatible with vllm to facilitate edge deployment on GPU.

### Model List

- https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-4bit
- https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-2.79bit
- https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-1.88bit
- https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-1.58bit
- https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-4bit
- https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-2.79bit
- https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-1.88bit
- https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-1.58bit

### Model Formats

- 4-bit: marlin / python
- 2.79-bit: python
- 1.88-bit: python
- 1.58-bit: python

### NVIDIA RTX 5090

```bash
vllm serve zhangsq-nju/Qwen3-0.6B-EdgeRazor-4bit --quantization edgerazor
```
