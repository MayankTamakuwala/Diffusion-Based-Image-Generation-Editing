#!/bin/bash
# scripts/run_training.sh
# Run LoRA fine-tuning on Stable Diffusion 1.5
#
# Prerequisites:
#   1. conda activate diffusion-gen (or pip install -r requirements.txt)
#   2. bash scripts/setup_dataset.sh (creates sample training data)
#
# USAGE:
#   Full training:       bash scripts/run_training.sh
#   With W&B logging:    bash scripts/run_training.sh --report_to wandb
#   Smoke test:         bash scripts/run_training.sh --smoke_test

set -e

# ── Parse arguments ───────────────────────────────────────────
SMOKE_TEST=false
REPORT_TO="tensorboard"

for arg in "$@"; do
    case $arg in
        --smoke_test) SMOKE_TEST=true ;;
        --report_to) REPORT_TO="$2"; shift ;;
        --report_to=*) REPORT_TO="${arg#*=}" ;;
    esac
done

echo "========================================"
echo "LoRA Fine-tuning — Stable Diffusion 1.5"
echo "========================================"
echo "Config:     config/train_config.yaml"
echo "Output:     lora_weights/"
echo "Smoke test: $SMOKE_TEST"
echo "Logging:    $REPORT_TO"
echo ""

# Check dataset exists
if [ ! -d "dataset/train/images" ]; then
    echo "ERROR: dataset/train/images/ not found."
    echo "Run: bash scripts/setup_dataset.sh"
    exit 1
fi

N_IMAGES=$(ls dataset/train/images/*.png 2>/dev/null | wc -l)
echo "Training images found: $N_IMAGES"
echo ""

if [ "$SMOKE_TEST" = true ]; then
    echo "Running SMOKE TEST (5 steps, verifies code works)"
    python src/training/train_lora.py \
        --config config/train_config.yaml \
        --smoke_test \
        --report_to none
else
    # For multi-GPU training, use: accelerate launch --multi_gpu
    # For single GPU, plain python works fine too
    python src/training/train_lora.py \
        --config config/train_config.yaml \
        --report_to "$REPORT_TO"
fi

echo ""
echo "Training complete! LoRA weights saved to: lora_weights/"
echo ""
echo "Next steps:"
echo "  Test inference: bash scripts/run_inference.sh"
echo "  Run evaluation: bash scripts/run_evaluation.sh"
