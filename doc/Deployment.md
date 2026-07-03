## Procedure of Compatible Formats

Preparation:

```bash
pip install edgerazor[vllm]
```

Instructions:

1. Conduct quantization training pipeline
2. Save the unquantized model weights: `/path/to/Qwen3-0.6B-4bit-unquant`, `/path/to/Qwen3-0.6B-1.58bit-unquant`
3. Convert the unquantized model weights into fake quantized weights: `/path/to/Qwen3-0.6B-EdgeRazor-4bit`, `/path/to/Qwen3-0.6B-EdgeRazor-1.58bit`
4. Convert into GGUF format
5. Deploy on edge devices with GGUF/HF formats using llama.cpp/vLLM

```bash
# 3. Convert the unquantized model weights into fake quantized weights: `/path/to/Qwen3-0.6B-W4A16`
python -m edgerazor.convert \
    --model_path /path/to/Qwen3-0.6B-1.58bit-unquant \
    --save_path /path/to/Qwen3-0.6B-EdgeRazor-1.58bit \
    --quant_mode w4a8kv8_qwen3 \
    --backend marlin \
    --weight_bits 4 \
    --activation_bits 16 \
    --kv_cache_bits 16 \
    --is_w_quantized true

# 4. Convert into GGUF format
convert_hf_to_gguf.py --outfile /path/to/Qwen3-0.6B-BF16.gguf --outtype bf16 /path/to/Qwen3-0.6B-EdgeRazor-1.58bit
llama-quantize --output-tensor-type q4_0 --token-embedding-type q4_0 /path/to/Qwen3-0.6B-F16.gguf /path/to/Qwen3-0.6B-EdgeRazor-1.58bit-TQ2_0.gguf TQ2_0

# 5. Deploy on edge devices with GGUF/HF formats using llama.cpp/vLLM
# 5.1 Using llama.cpp
llama-server --n-gpu-layers 0 --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 -m /path/to/Qwen3-0.6B-F16.gguf /path/to/Qwen3-0.6B-EdgeRazor-1.58bit-TQ2_0.gguf
# 5.2 Using vLLM
vllm serve /path/to/Qwen3-0.6B-EdgeRazor-1.58bit --quantization edgerazor
```
