MODEL_NAME="vit_small_patch16_224"
CKPT_PATH="../../../model/run/checkpoints/fp_vit_w16_a16/best_model.pth"
OUTPUT_PATH="../../../model/run/checkpoints/fp_vit_w16_a16/fp_model.gguf"

python convert-pth-to-gguf.py \
    --ckpt_path $CKPT_PATH \
    --output_path $OUTPUT_PATH \
    --model_name $MODEL_NAME