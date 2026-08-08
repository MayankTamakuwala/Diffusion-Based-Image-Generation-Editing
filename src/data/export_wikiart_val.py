"""
src/data/export_wikiart_val.py
------------------------------
Export the held-out WikiArt validation split to disk as PNGs + caption files.

WHY THIS EXISTS:
  FID and CLIP both need real images on disk:
    - FID compares the *distribution* of generated images against a
      reference set of real ones.
    - CLIP score needs (image, caption) pairs.

  Training reads WikiArt straight from the HuggingFace cache as tensors, but
  the evaluation code works with directories of files. This script bridges
  the two, writing the val split in the layout the eval pipeline expects.

WHY NOT JUST USE dataset/val/?
  That directory holds synthetic smoke-test images ("a forest path covered
  in autumn leaves"). Computing FID for an Impressionism model against those
  measures nothing -- FID would be enormous and tell you only that paintings
  are not photos of forests.

WHY THE VAL SPLIT SPECIFICALLY:
  WikiArtHFDataset splits train/val with a fixed seed (42). Training used
  split="train", so split="val" is genuinely held out -- the model has never
  seen these images. Evaluating against training images would flatter the
  score and mean nothing.

HOW MANY IMAGES?
  FID is biased upward on small samples, badly so below ~500. The default of
  1000 is the usual floor for a number worth quoting. Whatever you pick, use
  the SAME count for every model you compare -- FID values computed with
  different sample sizes are not comparable to each other.

PREPROCESSING:
  Images are resized and center-cropped to the generation resolution (512).
  This matters: if real images are 1024x768 and generated ones are 512x512,
  FID partly measures the resolution difference rather than image quality.

Usage:
  python src/data/export_wikiart_val.py
  python src/data/export_wikiart_val.py --num_images 500 --style Realism
  python src/data/export_wikiart_val.py --output_dir dataset/wikiart_val
"""

import argparse
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

# Running this file directly puts its own directory on sys.path, not the repo
# root, so "from src...." would fail. Add the repo root before any src import.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.data.dataset import WikiArtHFDataset
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def export_wikiart_val(
    output_dir: str | Path = "dataset/wikiart_val",
    style_filter: str | None = "Impressionism",
    num_images: int = 1000,
    resolution: int = 512,
    val_fraction: float = 0.05,
) -> dict:
    """
    Write the held-out WikiArt val split to disk as images + captions.

    Produces the standard layout the rest of the pipeline expects:
        <output_dir>/images/wikiart_val_00000.png
        <output_dir>/captions/wikiart_val_00000.txt

    Args:
        output_dir: Destination root.
        style_filter: WikiArt style, or None for all styles.
        num_images: How many val images to export.
        resolution: Square size; must match your generation resolution.
        val_fraction: Must match the value used at training time so the
            split boundary lands in the same place.

    Returns:
        Summary dict with counts and paths.
    """
    out_root = Path(output_dir)
    images_dir = out_root / "images"
    captions_dir = out_root / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # random_flip=False: this is reference data, not augmented training data.
    # Every run must produce byte-identical output or FID drifts between runs.
    logger.info(
        f"Loading WikiArt val split (style={style_filter}, "
        f"val_fraction={val_fraction})..."
    )
    ds = WikiArtHFDataset(
        style_filter=style_filter,
        split="val",
        val_fraction=val_fraction,
        resolution=resolution,
        center_crop=True,
        random_flip=False,
        tokenizer=None,
        max_samples=num_images,
    )

    logger.info(f"Val split has {len(ds)} images; exporting to {out_root}")

    # Resize + center crop only -- no ToTensor/Normalize. We want ordinary
    # 8-bit PNGs, not the [-1, 1] tensors the training path produces.
    to_square = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
    ])

    n_written = 0
    n_failed = 0

    for idx in tqdm(range(len(ds)), desc="Exporting val split"):
        sample = ds.data[idx]
        stem = f"wikiart_val_{idx:05d}"

        try:
            image = sample["image"].convert("RGB")
        except Exception as e:
            logger.warning(f"Skipping index {idx}, could not decode image: {e}")
            n_failed += 1
            continue

        to_square(image).save(images_dir / f"{stem}.png")

        # Same caption format the model was trained on, so CLIP score
        # measures prompt adherence rather than a change of phrasing.
        caption = ds._build_caption(sample)
        (captions_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")

        n_written += 1

    summary = {
        "output_dir": str(out_root),
        "images_dir": str(images_dir),
        "captions_dir": str(captions_dir),
        "num_exported": n_written,
        "num_failed": n_failed,
        "resolution": resolution,
        "style_filter": style_filter,
    }

    logger.info(f"Exported {n_written} images ({n_failed} failed)")
    if n_written < 500:
        logger.warning(
            f"Only {n_written} reference images. FID is biased upward below "
            f"~500 samples -- treat the absolute value as indicative only. "
            f"Comparisons between models at the SAME count remain valid."
        )

    return summary


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Export held-out WikiArt val images + captions for FID/CLIP"
    )
    parser.add_argument("--output_dir", type=str, default="dataset/wikiart_val")
    parser.add_argument("--style", type=str, default="Impressionism",
                        help="WikiArt style filter; pass 'all' for every style")
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Must match the training config for a consistent split")
    args = parser.parse_args()

    style = None if args.style.lower() == "all" else args.style

    summary = export_wikiart_val(
        output_dir=args.output_dir,
        style_filter=style,
        num_images=args.num_images,
        resolution=args.resolution,
        val_fraction=args.val_fraction,
    )

    print("\n" + "=" * 60)
    print("WIKIART VAL EXPORT")
    print("=" * 60)
    for key, val in summary.items():
        print(f"  {key:16s} {val}")
    print("=" * 60)
    print(f"\nPoint eval_config.yaml at:\n  {summary['output_dir']}")


if __name__ == "__main__":
    main()
