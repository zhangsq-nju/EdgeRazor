"""
Configuration for quantization:
- `block_size`: compatible with GGML quantization models.
"""

# Block sizes for different quantization schemes
w2a8_block_size = 256
w4a8_block_size = 256
w5a8_block_size = 256
w8a8_block_size = 256

# Default mixed-precision proportion
mixed_precision_prop = 0.01
