#!/bin/bash
# scripts/run_inference.sh
# Run all three inference modes: txt2img, img2img, controlnet
#
# Usage:
#   Full:       bash scripts/run_inference.sh
#   Smoke test: bash scripts/run_inference.sh --smoke_test

set -e

SMOKE_TEST=false
for arg in "$@"; do
    case $arg in
        --smoke_test) SMOKE_TEST=true ;;
    esac
done

echo "========================================"
echo "Running Inference Scripts"
echo "========================================"

LORA_PATH=""
if [ -d "lora_weights/final_lora_adapter" ]; then
    LORA_PATH="--lora_path lora_weights/final_lora_adapter"
    echo "LoRA: lora_weights/final_lora_adapter"
else
    echo "LoRA: Not found, using base model"
fi

mkdir -p experiments/samples

# ── 1. Text-to-Image ──────────────────────────────────────────
echo ""
echo "[1/3] Text-to-Image..."
if [ "$SMOKE_TEST" = true ]; then
    python src/inference/txt2img.py --smoke_test $LORA_PATH
else
    python src/inference/txt2img.py \
        --prompt "a majestic snow-capped mountain at golden hour, photorealistic, 8k detail" \
        --negative_prompt "blurry, low quality, watermark, deformed" \
        --steps 30 \
        --guidance_scale 7.5 \
        --seed 42 \
        --output experiments/samples/txt2img_result.png \
        $LORA_PATH
fi

# ── 2. Image-to-Image ─────────────────────────────────────────
echo ""
echo "[2/3] Image-to-Image..."
if [ "$SMOKE_TEST" = true ]; then
    python src/inference/img2img.py --smoke_test $LORA_PATH
else
    # Use the txt2img output as input (if it exists)
    INPUT_IMG="experiments/samples/txt2img_result.png"
    if [ ! -f "$INPUT_IMG" ]; then
        INPUT_IMG="dataset/val/images/$(ls dataset/val/images/ | head -1)"
    fi

    python src/inference/img2img.py \
        --input "$INPUT_IMG" \
        --prompt "a painting of this scene in vibrant impressionist style, Van Gogh inspired" \
        --strength 0.65 \
        --steps 30 \
        --seed 42 \
        --output experiments/samples/img2img_result.png \
        --save_comparison \
        $LORA_PATH
fi

# ── 3. ControlNet ─────────────────────────────────────────────
echo ""
echo "[3/3] ControlNet (Canny)..."
if [ "$SMOKE_TEST" = true ]; then
    python src/inference/controlnet_infer.py --smoke_test $LORA_PATH
else
    INPUT_IMG="dataset/val/images/$(ls dataset/val/images/ | head -1)"

    python src/inference/controlnet_infer.py \
        --input "$INPUT_IMG" \
        --prompt "a detailed technical architectural drawing, blueprint style, precise lines" \
        --controlnet_scale 1.0 \
        --steps 30 \
        --seed 42 \
        --output experiments/samples/controlnet_result.png \
        --save_edges \
        --save_triptych \
        $LORA_PATH
fi

echo ""
echo "========================================"
echo "All inference complete!"
echo "Results saved to: experiments/samples/"
ls -la experiments/samples/
