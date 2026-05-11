FP_GGUF_PATH="../../../model/run/checkpoints/fp_vit_w16_a16/fp_model.gguf"
QUANTIZED_GGUF_PATH="../../../model/run/checkpoints/fp_vit_w16_a16/quant_model.gguf"
TYPE=8

# quantize
../../vit.cpp/build/bin/quantize $FP_GGUF_PATH $QUANTIZED_GGUF_PATH $TYPE

# usage: ./bin/quantize /path/to/ggml-model-f32.gguf /path/to/ggml-model-quantized.gguf type                              
#   type = 2 - q4_0                                                                                                       
#   type = 3 - q4_1                                                                                                       
#   type = 6 - q5_0                                                                                                       
#   type = 7 - q5_1                                                                                                       
#   type = 8 - q8_0