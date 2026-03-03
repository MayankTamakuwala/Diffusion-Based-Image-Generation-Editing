#!/bin/bash
# scripts/setup_dataset.sh
# Create the synthetic sample dataset for testing the pipeline.
# Run this FIRST before any training or evaluation.

set -e  # Exit on any error

echo "========================================"
echo "Setting up sample dataset for testing"
echo "========================================"

# Full dataset (20 train, 5 val, 512x512)
python src/data/create_sample_dataset.py \
    --output_dir dataset \
    --num_train 20 \
    --num_val 5 \
    --size 512 \
    --seed 42

echo ""
echo "Dataset created! Directory structure:"
find dataset -type f | head -20

echo ""
echo "To use a REAL dataset:"
echo "  1. Place your ~5000 images in: dataset/train/images/"
echo "  2. Create matching captions in: dataset/train/captions/"
echo "     (filename must match: img001.png → img001.txt)"
echo "  3. Place ~500 val images in: dataset/val/images/"
echo "  4. Re-run training: bash scripts/run_training.sh"
