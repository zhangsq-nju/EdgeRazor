# ============================================================================
# Training Configuration Variables
# ============================================================================
BATCH_SIZE=128
EPOCHS=50
LR=1e-4
WEIGHT_DECAY=0.2
WARMUP_STEPS=320
OUTPUT_DIR="../../../model"
EARLY_STOPPING_PATIENCE=5
DATA_ROOT="../../../data"

mkdir -p $OUTPUT_DIR

## Example training configurations are provided below (not exhaustive).
## You can modify parameters as needed or add new training setups.
# Baseline model training: W16-A16
python -m train \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --data_root $DATA_ROOT

# Quantized model training: W4-A16 [EdgeRazor already includes a Q4_0 quantization function, i.e., 4-bit weight quantization]
python -m train \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w4_a16.yaml \
    --data_root $DATA_ROOT

# Quantized model training: W8-A16 [Read Quantization.md and implement the Q8_0 quantization function in EdgeRazor before running, i.e., 8-bit weight quantization]
python -m train \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w8_a16.yaml \
    --data_root $DATA_ROOT