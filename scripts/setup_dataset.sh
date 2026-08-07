#!/bin/bash
# scripts/setup_dataset.sh
# Set up a dataset for training.
#
# MODES:
#   default         -- create 20 synthetic images (smoke-test only)
#   --wikiart       -- enable WikiArt in config + print download instructions
#   --wikiart-test  -- verify WikiArt can be loaded (downloads ~7GB first run)
#
# Usage:
#   bash scripts/setup_dataset.sh               # synthetic (no download)
#   bash scripts/setup_dataset.sh --wikiart     # switch config to WikiArt
#   bash scripts/setup_dataset.sh --wikiart-test

set -e

MODE="synthetic"
for arg in "$@"; do
    case $arg in
        --wikiart) MODE="wikiart" ;;
        --wikiart-test) MODE="wikiart-test" ;;
    esac
done

echo "========================================"
echo "Dataset Setup  (mode: $MODE)"
echo "========================================"

if [ "$MODE" = "synthetic" ]; then
    # ── Synthetic fallback (always works, no internet needed) ──
    python src/data/create_sample_dataset.py \
        --output_dir dataset \
        --num_train 20 \
        --num_val 5 \
        --size 512 \
        --seed 42

    echo ""
    echo "Synthetic dataset created."
    find dataset -type f | head -20

elif [ "$MODE" = "wikiart" ]; then
    # ── Switch config to use WikiArt ──────────────────────────
    echo ""
    echo "Enabling WikiArt in config/train_config.yaml ..."
    # Use Python + OmegaConf to safely flip the flag
    python - << 'PYEOF'
from omegaconf import OmegaConf
cfg = OmegaConf.load("config/train_config.yaml")
cfg.hf_dataset.enabled = True
OmegaConf.save(cfg, "config/train_config.yaml")
print("  hf_dataset.enabled set to: true")
print("  style_filter:", cfg.hf_dataset.style_filter)
PYEOF

    echo ""
    echo "WikiArt enabled. On first training run the dataset (~7 GB) will be"
    echo "downloaded automatically to ~/.cache/huggingface/datasets/"
    echo ""
    echo "To pre-download now (optional, avoids timeout during training):"
    echo "  python -c \"from datasets import load_dataset; load_dataset('huggan/wikiart', split='train')\""
    echo ""
    echo "To change style (e.g. Realism), edit config/train_config.yaml:"
    echo "  hf_dataset:"
    echo "    style_filter: \"Realism\""
    echo ""
    echo "Available styles:"
    python - << 'PYEOF'
try:
    from datasets import load_dataset
    ds = load_dataset("huggan/wikiart", split="train[:1]")
    print("  " + "\n  ".join(ds.features["style"].names))
except Exception:
    print("  (install datasets first: pip install datasets)")
    print("  Known styles: Impressionism, Realism, Romanticism, Expressionism,")
    print("    Post_Impressionism, Surrealism, Art_Nouveau_Modern, Symbolism, Cubism, Baroque")
PYEOF

elif [ "$MODE" = "wikiart-test" ]; then
    # ── Verify WikiArt loads correctly ────────────────────────
    echo "Loading 5 WikiArt samples to verify connectivity and format..."
    python - << 'PYEOF'
from datasets import load_dataset
from src.data.dataset import WikiArtHFDataset

# Quick load: 5 samples only
ds = WikiArtHFDataset(style_filter="Impressionism", split="train", max_samples=5)
print(f"WikiArtHFDataset loaded: {len(ds)} samples")
sample = ds[0]
print(f"  pixel_values shape: {sample['pixel_values'].shape}")
print(f"  caption:            {sample['caption']}")
print("WikiArt test PASSED")
PYEOF
fi
