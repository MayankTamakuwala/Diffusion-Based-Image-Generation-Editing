#!/bin/bash
# scripts/run_evaluation.sh
# Run full FID + CLIP evaluation pipeline
#
# NOTE: Meaningful FID requires 1000+ images.
# This script generates images first, then evaluates.
# With the sample dataset (5 val images), FID will be noisy but functional.
#
# Usage:
#   Full:       bash scripts/run_evaluation.sh
#   Smoke test: bash scripts/run_evaluation.sh --smoke_test
#   With LoRA:  bash scripts/run_evaluation.sh --lora_path lora_weights/

set -e

SMOKE_TEST=false
LORA_ARG=""

for arg in "$@"; do
    case $arg in
        --smoke_test) SMOKE_TEST=true ;;
        --lora_path=*) LORA_ARG="--lora_path ${arg#*=}" ;;
    esac
done

echo "========================================"
echo "Evaluation: FID + CLIP Scores"
echo "========================================"

if [ "$SMOKE_TEST" = true ]; then
    python src/evaluation/run_eval.py \
        --config config/eval_config.yaml \
        --smoke_test \
        $LORA_ARG
else
    python src/evaluation/run_eval.py \
        --config config/eval_config.yaml \
        $LORA_ARG
fi

echo ""
echo "Metrics saved to: experiments/metrics/"
ls -lt experiments/metrics/ | head -5
