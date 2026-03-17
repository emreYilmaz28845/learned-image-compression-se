#!/bin/bash
# Run the full experiment pipeline:
# 1. Download datasets
# 2. Train variants A, B, C at all lambda values
# 3. Evaluate on Kodak
# 4. Plot RD curves
#
# Usage: bash run_all.sh
# Assumes conda env "learned-image-compression" is activated.

set -e

DATA_DIR="data"
CLIC_DIR="$DATA_DIR/clic"
KODAK_DIR="$DATA_DIR/kodak"
CKPT_DIR="checkpoints"
RESULTS_DIR="results"
PLOTS_DIR="plots"

LAMBDAS=(0.0018 0.0035 0.0067 0.013)
QUALITY=3
EPOCHS=100
BATCH_SIZE=8

echo "=== Step 1: Download datasets ==="
python dataset.py --data-dir "$DATA_DIR" --dataset kodak
python dataset.py --data-dir "$DATA_DIR" --dataset clic

echo ""
echo "=== Step 2: Train all variants ==="

# Variant A: save pretrained baseline (no training needed, one run covers all lambdas)
for LMBDA in "${LAMBDAS[@]}"; do
    echo "--- Variant A, lambda=$LMBDA ---"
    python train.py --variant A --lmbda "$LMBDA" --quality $QUALITY \
        --data-dir "$CLIC_DIR" --save-dir "$CKPT_DIR"
done

# Variant B: fine-tune baseline (no SE block)
for LMBDA in "${LAMBDAS[@]}"; do
    echo "--- Variant B, lambda=$LMBDA ---"
    python train.py --variant B --lmbda "$LMBDA" --quality $QUALITY \
        --data-dir "$CLIC_DIR" --epochs $EPOCHS --batch-size $BATCH_SIZE \
        --save-dir "$CKPT_DIR"
done

# Variant C: fine-tune with SE block
for LMBDA in "${LAMBDAS[@]}"; do
    echo "--- Variant C, lambda=$LMBDA ---"
    python train.py --variant C --lmbda "$LMBDA" --quality $QUALITY \
        --data-dir "$CLIC_DIR" --epochs $EPOCHS --batch-size $BATCH_SIZE \
        --save-dir "$CKPT_DIR"
done

echo ""
echo "=== Step 3: Evaluate on Kodak ==="

for LMBDA in "${LAMBDAS[@]}"; do
    echo "--- Evaluating Variant A, lambda=$LMBDA ---"
    python evaluate.py --variant A --quality $QUALITY --lmbda "$LMBDA" \
        --data-dir "$KODAK_DIR" --output "$RESULTS_DIR"

    echo "--- Evaluating Variant B, lambda=$LMBDA ---"
    python evaluate.py --variant B --quality $QUALITY --lmbda "$LMBDA" \
        --checkpoint "$CKPT_DIR/variant_B_lmbda_${LMBDA}_mse/checkpoint_best.pth.tar" \
        --data-dir "$KODAK_DIR" --output "$RESULTS_DIR"

    echo "--- Evaluating Variant C, lambda=$LMBDA ---"
    python evaluate.py --variant C --quality $QUALITY --lmbda "$LMBDA" \
        --checkpoint "$CKPT_DIR/variant_C_lmbda_${LMBDA}_mse/checkpoint_best.pth.tar" \
        --data-dir "$KODAK_DIR" --output "$RESULTS_DIR"
done

echo ""
echo "=== Step 4: Plot RD curves ==="
python plot.py --results-dir "$RESULTS_DIR" --output "$PLOTS_DIR"

echo ""
echo "=== Done! ==="
echo "Checkpoints: $CKPT_DIR"
echo "Results:     $RESULTS_DIR"
echo "Plots:       $PLOTS_DIR"
