#!/bin/bash
# scripts/run_app.sh
# Launch the Gradio web app
#
# Usage:
#   Normal GPU run:     bash scripts/run_app.sh
#   With LoRA:          bash scripts/run_app.sh --lora_path lora_weights/
#   Public URL (ngrok): bash scripts/run_app.sh --share
#   CPU smoke test:     bash scripts/run_app.sh --smoke_test
#   Custom port:        bash scripts/run_app.sh --port 8080

set -e

LORA_ARG=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --lora_path=*) LORA_ARG="--lora_path ${arg#*=}" ;;
        --share) EXTRA_ARGS="$EXTRA_ARGS --share" ;;
        --smoke_test) EXTRA_ARGS="$EXTRA_ARGS --smoke_test" ;;
        --port=*) EXTRA_ARGS="$EXTRA_ARGS --port ${arg#*=}" ;;
    esac
done

echo "========================================"
echo "Launching Gradio App"
echo "========================================"
echo "Access at: http://127.0.0.1:7860"
echo ""

python app/gradio_app.py \
    --model_id "runwayml/stable-diffusion-v1-5" \
    --controlnet_id "lllyasviel/sd-controlnet-canny" \
    $LORA_ARG \
    $EXTRA_ARGS
