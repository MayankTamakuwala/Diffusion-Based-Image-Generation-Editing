#!/bin/bash
# scripts/run_benchmark.sh
# Run latency benchmarks for different configurations
#
# Usage:
#   bash scripts/run_benchmark.sh
#   bash scripts/run_benchmark.sh --smoke_test

set -e

SMOKE_TEST=false
for arg in "$@"; do
    case $arg in
        --smoke_test) SMOKE_TEST=true ;;
    esac
done

echo "========================================"
echo "Latency Benchmarking"
echo "========================================"

mkdir -p experiments/latency

if [ "$SMOKE_TEST" = true ]; then
    echo "SMOKE TEST: quick 3-run benchmark"
    python benchmark/latency_benchmark.py --smoke_test
    exit 0
fi

# ── Benchmark 1: Default (30 steps, 512x512) ─────────────────
echo ""
echo "[1/3] Benchmark: 30 steps, 512x512"
python benchmark/latency_benchmark.py \
    --steps 30 \
    --size 512 \
    --n_warmup 3 \
    --n_runs 20

# ── Benchmark 2: Fast mode (10 steps, 512x512) ───────────────
echo ""
echo "[2/3] Benchmark: 10 steps, 512x512 (fast mode)"
python benchmark/latency_benchmark.py \
    --steps 10 \
    --size 512 \
    --n_warmup 3 \
    --n_runs 20

# ── Benchmark 3: With LoRA (if exists) ───────────────────────
LORA_DIR="lora_weights/final_lora_adapter"
if [ -d "$LORA_DIR" ]; then
    echo ""
    echo "[3/3] Benchmark: 30 steps with LoRA"
    python benchmark/latency_benchmark.py \
        --steps 30 \
        --size 512 \
        --lora_path "$LORA_DIR" \
        --n_warmup 3 \
        --n_runs 20
else
    echo ""
    echo "[3/3] Skipping LoRA benchmark (no lora_weights/ found)"
fi

echo ""
echo "Benchmark results saved to: experiments/latency/"
ls -lt experiments/latency/ | head -10
