## Analysis and Implementation of the Quantization Algorithm

This document describes how to implement the Q8_0 quantization algorithm for the vit.cpp inference framework by adding an INT8 weight quantization function in EdgeRazor.

1. **vit.cpp supports Q8_0 quantized inference**

```bash
# vit.cpp supports the q8_0 quantization type
➤ ./vit --help
usage: ./vit [options]

options:
  -h, --help              show this help message and exit
  -m FNAME, --model       model path (default: ../ggml-model-f16.gguf)
  -i FNAME, --inp         input file (default: ../assets/tench.jpg)
  -t N, --threads         number of threads to use during computation (default: 4)
  -k N, --topk            top k classes to print (default: 5)
  -s SEED, --seed         RNG seed (default: -1)
  -e FLOAT, --epsilon     epsilon constant in Layer Norm layers (default: 0.000001)

➤ ./quantize --help
usage: ./quantize model-f32.bin model-quant.bin type
  type = 2 - q4_0
  type = 3 - q4_1
  type = 6 - q5_0
  type = 7 - q5_1
  type = 8 - q8_0 # <= this indicates vit.cpp supports Q8_0; the corresponding implementation can be found in GGML source code
```

In `vit.cpp/ggml/src/ggml.c` line 519:

```python
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float               = quantize_row_q8_0,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q8_0_reference,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
```

2. **Analyze matrix multiplication for Q8_0 quantized inference**

   In `vit.cpp/ggml/src/ggml-quants.c`, you can find the matrix multiplication implementation for Q8_0 inference: `ggml_vec_dot_q8_0_q8_0`.

   ```python
    void ggml_vec_dot_q8_0_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
        const int qk = QK8_0;
        const int nb = n / qk;

        assert(n % qk == 0);

        const block_q8_0 * restrict x = vx;
        const block_q8_0 * restrict y = vy;

        ...
    }
   ```

   From this code, we can identify key properties of dot-product computation between two Q8_0 vectors:

   - Per-block: every `qk` elements form one block with an independent scale factor.
   - Symmetric: symmetric quantization with no zero-point offset.
   - Absmax: the scale factor is determined by the maximum absolute value in each block. Quantized values are in the INT8 range, i.e. $[-128,127]$, which can be constrained to $[-127,127]$ for improved symmetry.

3. Add the corresponding implementation in EdgeRazor: `weight_quant_uniform_symmetric_absmax_per_block_int8`
   1. `/opt/code-dependency/EdgeRazor/src/edgerazor/qat/map.py`:

      ![image-20260412181720675](./Quantization.assets/image-20260412181720675.png)
      In `src/edgerazor/qat/map.py`:

    ```python
    from .util.quant_function import (
        weight_quant_uniform_symmetric_absmax_per_block_int5,
        weight_quant_uniform_symmetric_absmax_per_block_int8,
    )
    ```

    ```python
    _quant_functions = [
        # INT4+ Weight Quantization - Symmetric Absmax Method
        weight_quant_uniform_symmetric_absmax_per_block_int5,
        weight_quant_uniform_symmetric_absmax_per_block_int8,
    ]
    ```

   2. `/opt/code-dependency/EdgeRazor/src/edgerazor/qat/util/quant_function.py`:

      ```python
    def weight_quant_uniform_symmetric_absmax_per_block_int8(
        w: Tensor,
        epsilon: float = 1e-5,
        block_size: int = w8a8_block_size,
    ) -> Tensor:
        """
        Quantize weight to INT8 per-block using absmax method.

        Quantizes weight to INT8: [-127, 127] * w_scale.
        Scale factor is computed per block within each output channel.

        Args:
            w: Weight tensor to quantize, shape (out_dim, in_dim)
            epsilon: Small value to prevent division by zero
            block_size: Size of each quantization block

        Returns:
            Quantized weight tensor with values in [-127, 127] * w_scale
        """
        bits = 8
        max_val = 2**(bits - 1) - 1  # 127 for INT8

        with torch.no_grad():
            # Reshape to [..., -1, block_size]
            original_shape = w.shape
            if original_shape[-1] % block_size == 0:
                intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
            else:
                intermediate_shape = [-1, block_size]
            w = w.view(intermediate_shape)

            # Compute scale factor for each block
            # Shape: (out_dim, block_num, 1)
            w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

            # Quantize to INT8: [-127, 127]
            w_quant = w.div(w_scale).round_().clamp_(-max_val, max_val)

            w_quant = w_quant * w_scale
            w_quant = w_quant.view(original_shape)

        return w_quant
      ```

      ```python
      def weight_quant_uniform_symmetric_absmax_per_block_int8(
          w: Tensor,
          epsilon: float = 1e-5,
          block_size: int = w8a8_block_size,
      ) -> Tensor:
          bits = 8
          max_val = 2**(bits - 1) - 1  # 127 for INT8

          with torch.no_grad():
              # Reshape to [..., -1, block_size]
              original_shape = w.shape
              if original_shape[-1] % block_size == 0:
                  intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
              else:
                  intermediate_shape = [-1, block_size]
              w = w.view(intermediate_shape)

              # Compute scale factor for each block
              # Shape: (out_dim, block_num, 1)
              w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

              # Quantize to INT8: [-127, 127]
              w_quant = w.div(w_scale).round_().clamp_(-max_val, max_val)

              w_quant = w_quant * w_scale
              w_quant = w_quant.view(original_shape)

          return w_quant
      ```

4. Train to obtain .pth weights + convert to f16 gguf + quantize to Q8_0 gguf weights
   - Configure the corresponding weight quantization function in the training config.

    ```yaml
    qat_configuration:
    method: QAT
    select:
        target_types:
        - linear
        target_names: []
        exclude_types: []
        exclude_names: []
    function:
        epsilon: 1.0e-06
        weight_function: weight_quant_uniform_symmetric_absmax_per_block_int8  # Weight quantization function
        w_block_size: 128  # Weight quantization configuration
        is_w_quantized: false
        activation_function: ''  # Activation quantization function
        a_block_size: -1  # Activation quantization configuration
        kv_cache_function: ''
        kv_block_size: -1
    training: all
    ```

5. Successfully run CPU inference with quantized weights (GGUF model file)

> ⚠️ EdgeRazor already contains a Q4_0 quantization function implementation. This sample training script focuses on analysis and implementation of Q8_0 quantization. Participants should design quantization algorithms for low-bit model training and adapt them for vit.cpp quantized inference, then upload the corresponding GGUF quantized model weights.
