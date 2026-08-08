"""
src/evaluation/fid_eval.py
--------------------------
Fréchet Inception Distance (FID) computation using cleanfid.

WHAT IS FID?
FID measures the similarity between two image distributions:
  - The REAL distribution (your validation/test images)
  - The GENERATED distribution (images from your model)

HOW FID IS COMPUTED:
  1. Feed all real images through Inception-v3 (pretrained on ImageNet)
     → extract 2048-dim feature vectors for each image
  2. Fit a multivariate Gaussian to those features: μ_real, Σ_real
  3. Do the same for generated images: μ_gen, Σ_gen
  4. FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real·Σ_gen))

INTERPRETATION:
  FID = 0: Generated images are statistically identical to real images
  FID < 10: Excellent (professional quality)
  FID < 50: Good (similar to many published models)
  FID < 100: Acceptable
  FID > 200: Poor quality or severe distribution mismatch

WHY cleanfid OVER pytorch-fid?
  pytorch-fid had a bug where images are resized differently than the
  original TensorFlow implementation, leading to inconsistent FID values
  across papers and implementations.

  cleanfid fixes this by:
  - Using the correct antialiased resizing (matches TF's bilinear resize)
  - Supporting "clean" FID mode which is now standard in papers
  - Also supporting legacy mode for comparison with older benchmarks

WHY YOU NEED 1000+ IMAGES:
  FID estimates a distribution. With < 500 images, the Gaussian fit is
  unreliable and FID scores are noisy (high variance). 1000 is minimum;
  5000+ gives stable estimates matching published benchmarks.

Usage:
  python src/evaluation/fid_eval.py --real dataset/val/images --generated experiments/eval_generated
  python src/evaluation/fid_eval.py --smoke_test
"""

import argparse
from pathlib import Path
from typing import Optional

# Running this file directly puts its own directory on sys.path, not the repo
# root, so "from src...." would fail. Add the repo root before any src import.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def compute_fid(
    real_images_dir: str | Path,
    generated_images_dir: str | Path,
    mode: str = "clean",
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
) -> float:
    """
    Compute FID between a real and generated image directory.

    Both directories should contain image files (PNG/JPEG).
    Images do NOT need to be paired — FID is a distributional metric.

    Args:
        real_images_dir: Directory of ground truth images.
        generated_images_dir: Directory of generated images.
        mode: "clean" (standard, matches paper benchmarks) | "legacy_pytorch"
        batch_size: Batch size for Inception-v3 feature extraction.
        num_workers: DataLoader workers.
        device: "auto" | "cuda" | "cpu"

    Returns:
        FID score (float). Lower is better.
    """
    try:
        from cleanfid import fid as cleanfid_module
    except ImportError:
        raise ImportError(
            "cleanfid is not installed. Run: pip install clean-fid"
        )

    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    real_dir = Path(real_images_dir)
    gen_dir = Path(generated_images_dir)

    if not real_dir.exists():
        raise FileNotFoundError(f"Real images directory not found: {real_dir}")
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generated images directory not found: {gen_dir}")

    # Count images
    real_count = len(list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")))
    gen_count = len(list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg")) + list(gen_dir.glob("*.jpeg")))

    logger.info(f"Computing FID: {real_count} real images, {gen_count} generated images")

    if real_count < 10 or gen_count < 10:
        logger.warning(
            f"Very few images detected (real={real_count}, gen={gen_count}). "
            f"FID scores will be unreliable. Use 1000+ images for meaningful FID."
        )

    if real_count < 1000 or gen_count < 1000:
        logger.warning(
            "For reliable FID, you need 1000+ images in each directory. "
            "Current scores may have high variance."
        )

    # Compute FID using cleanfid
    fid_score = cleanfid_module.compute_fid(
        fdir1=str(real_dir),
        fdir2=str(gen_dir),
        mode=mode,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info(f"FID score: {fid_score:.4f}")
    return fid_score


def generate_images_for_fid(
    prompts: list[str],
    output_dir: str | Path,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    batch_size: int = 4,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    device: str = "auto",
) -> Path:
    """
    Generate images for FID evaluation from a list of prompts.

    For FID evaluation, we want the generated images to match the
    distribution of the real images as closely as possible.
    Ideally, use the same captions as your val set for generation.

    Args:
        prompts: List of text prompts to generate from.
        output_dir: Where to save generated images.
        model_id: SD model.
        lora_path: Optional LoRA adapter.
        lora_scale: Adapter strength, 0.0-1.0+. Must be threaded through
            explicitly -- a silently-dropped scale produces a run that
            looks correct but measures the wrong model.
        num_steps: Inference steps.
        guidance_scale: CFG scale.
        batch_size: Images per generation batch.
        width, height: Output resolution.
        seed: Starting seed (each image gets seed + index for variety).
        device: Compute device.

    Returns:
        Path to the output directory.
    """
    import torch
    from tqdm.auto import tqdm
    from src.models.pipeline_utils import load_txt2img_pipeline
    from src.utils.seed_utils import seed_generator

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {len(prompts)} images for FID evaluation → {output_path}")

    pipe = load_txt2img_pipeline(
        model_id=model_id,
        lora_weights_path=lora_path,
        lora_scale=lora_scale,
        device=device,
    )

    # Generate in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating eval images"):
        batch_prompts = prompts[i:i + batch_size]

        # Use different seeds per image for diversity (not the same seed for all!)
        # This ensures the generated distribution spans a wide range of images.
        batch_generators = [
            seed_generator(seed + i + j)
            for j in range(len(batch_prompts))
        ]

        with torch.no_grad():
            outputs = pipe(
                prompt=batch_prompts,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=batch_generators,
            )

        for j, img in enumerate(outputs.images):
            img_idx = i + j
            img.save(output_path / f"generated_{img_idx:05d}.png")

    logger.info(f"Generated {len(prompts)} images → {output_path}")
    return output_path


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Compute FID score")
    parser.add_argument("--real", type=str, default="dataset/val/images",
                        help="Directory of real images")
    parser.add_argument("--generated", type=str, default="experiments/eval_generated",
                        help="Directory of generated images")
    parser.add_argument("--mode", type=str, default="clean", choices=["clean", "legacy_pytorch"],
                        help="FID computation mode")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        logger.info("SMOKE TEST: creating tiny synthetic sets to verify FID runs")
        # Create tiny synthetic sets just to verify the code runs
        import numpy as np
        from PIL import Image
        import tempfile, os

        tmp_real = Path(tempfile.mkdtemp())
        tmp_gen = Path(tempfile.mkdtemp())

        for i in range(10):
            np.random.seed(i)
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_real / f"real_{i}.png")
            np.random.seed(i + 1000)
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_gen / f"gen_{i}.png")

        args.real = str(tmp_real)
        args.generated = str(tmp_gen)
        logger.info(f"Created synthetic dirs: {tmp_real}, {tmp_gen}")

    fid_score = compute_fid(
        real_images_dir=args.real,
        generated_images_dir=args.generated,
        mode=args.mode,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\nFID Score: {fid_score:.4f}")
    print("(Lower is better; <50 is good; 0 = perfect match)")


if __name__ == "__main__":
    main()
