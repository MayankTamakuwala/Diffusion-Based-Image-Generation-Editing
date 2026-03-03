"""
src/evaluation/run_eval.py
--------------------------
Orchestrate full evaluation pipeline: generate images → FID → CLIP → save JSON.

This script ties together all evaluation steps:
  1. Generate N images from validation prompts (using your model)
  2. Compute FID between generated and real validation images
  3. Compute CLIP similarity between generated images and their prompts
  4. Save everything to experiments/metrics/metrics_<timestamp>.json

The JSON output looks like this (resume-ready for interviews):
{
  "timestamp": "2024-03-15T14:30:22",
  "config": { ... training/eval config ... },
  "metrics": {
    "fid": 42.3,
    "clip_mean": 0.312,
    "clip_std": 0.018
  },
  "runtime": {
    "generation_s": 245.2,
    "fid_s": 38.1,
    "clip_s": 12.4
  },
  "model_info": {
    "base_model": "runwayml/stable-diffusion-v1-5",
    "lora_path": "lora_weights/final_lora_adapter",
    "num_inference_steps": 30
  }
}

Usage:
  python src/evaluation/run_eval.py --config config/eval_config.yaml
  python src/evaluation/run_eval.py --config config/eval_config.yaml --lora_path lora_weights/
  python src/evaluation/run_eval.py --smoke_test
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from src.utils.logging_utils import get_logger, setup_logging, get_timestamped_filename

logger = get_logger(__name__)


def load_val_prompts(
    val_data_dir: str | Path,
    max_count: Optional[int] = None,
    fallback: str = "a high quality photo",
) -> list[str]:
    """
    Load validation prompts from the dataset captions directory.

    Falls back to `fallback` if no caption files are found.

    Args:
        val_data_dir: Path to val/ directory (contains captions/ subdir).
        max_count: Maximum number of prompts to return (for FID generation).
        fallback: Caption to use when no .txt files exist.

    Returns:
        List of prompt strings.
    """
    val_path = Path(val_data_dir)
    captions_dir = val_path / "captions"
    images_dir = val_path / "images"

    if captions_dir.exists():
        caption_files = sorted(captions_dir.glob("*.txt"))
        prompts = [f.read_text(encoding="utf-8").strip() for f in caption_files if f.stat().st_size > 0]
    else:
        # Fall back: use one prompt per image
        IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
        image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        prompts = [fallback] * len(image_files)

    if not prompts:
        logger.warning(f"No prompts found in {val_data_dir}. Using fallback.")
        prompts = [fallback] * 100

    if max_count and len(prompts) > max_count:
        prompts = prompts[:max_count]

    # If we need more generated images than we have prompts, repeat prompts
    return prompts


def run_evaluation(
    config: OmegaConf,
    smoke_test: bool = False,
) -> dict:
    """
    Run full evaluation pipeline.

    Args:
        config: OmegaConf config from eval_config.yaml.
        smoke_test: If True, use minimal images for quick testing.

    Returns:
        Dictionary containing all metrics and metadata.
    """
    if smoke_test:
        config.generation.num_images = config.smoke_test.num_images
        config.generation.batch_size = config.smoke_test.batch_size

    results = {
        "timestamp": datetime.now().isoformat(),
        "smoke_test": smoke_test,
        "model_info": {
            "base_model": config.model.base_model,
            "lora_path": config.model.lora_weights_path,
            "num_inference_steps": config.generation.num_inference_steps,
            "guidance_scale": config.generation.guidance_scale,
            "scheduler": config.generation.scheduler,
        },
        "config": OmegaConf.to_container(config, resolve=True),
        "metrics": {},
        "runtime": {},
    }

    # ── Step 1: Load Validation Prompts ───────────────────────────────────
    logger.info("Loading validation prompts...")
    prompts = load_val_prompts(
        val_data_dir="dataset/val",
        max_count=config.generation.num_images,
        fallback="a high quality photograph",
    )
    # If we have fewer prompts than requested, repeat them
    while len(prompts) < config.generation.num_images:
        prompts.extend(prompts)
    prompts = prompts[:config.generation.num_images]

    logger.info(f"Using {len(prompts)} prompts for generation")

    # ── Step 2: Generate Images ────────────────────────────────────────────
    from src.evaluation.fid_eval import generate_images_for_fid
    gen_dir = Path(config.generation.output_dir)

    logger.info(f"Generating {len(prompts)} images → {gen_dir}")
    t_gen_start = time.perf_counter()

    generate_images_for_fid(
        prompts=prompts,
        output_dir=gen_dir,
        model_id=config.model.base_model,
        lora_path=config.model.lora_weights_path,
        num_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        batch_size=config.generation.batch_size,
        width=config.generation.width,
        height=config.generation.height,
        seed=config.generation.seed,
    )

    results["runtime"]["generation_s"] = round(time.perf_counter() - t_gen_start, 2)
    logger.info(f"Generation done in {results['runtime']['generation_s']}s")

    # ── Step 3: Compute FID ────────────────────────────────────────────────
    from src.evaluation.fid_eval import compute_fid
    real_dir = config.fid.real_images_dir

    logger.info(f"Computing FID: real={real_dir}, gen={gen_dir}")
    t_fid_start = time.perf_counter()

    try:
        fid_score = compute_fid(
            real_images_dir=real_dir,
            generated_images_dir=gen_dir,
            mode=config.fid.mode,
        )
        results["metrics"]["fid"] = round(fid_score, 4)
    except Exception as e:
        logger.error(f"FID computation failed: {e}")
        results["metrics"]["fid"] = None
        results["metrics"]["fid_error"] = str(e)

    results["runtime"]["fid_s"] = round(time.perf_counter() - t_fid_start, 2)

    # ── Step 4: Compute CLIP Score ─────────────────────────────────────────
    from src.evaluation.clip_eval import compute_clip_score
    from PIL import Image as PILImage

    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    gen_image_paths = sorted([p for p in gen_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    # Pair images with their generation prompts
    paired_prompts = prompts[:len(gen_image_paths)]

    logger.info(f"Computing CLIP score for {len(gen_image_paths)} image-text pairs")
    t_clip_start = time.perf_counter()

    try:
        clip_result = compute_clip_score(
            images=[str(p) for p in gen_image_paths],
            texts=paired_prompts,
            model_name=config.clip.model_name.replace("/", "-"),
            pretrained=config.clip.pretrained,
            batch_size=config.clip.batch_size,
        )
        results["metrics"]["clip_mean"] = round(clip_result["mean_score"], 4)
        results["metrics"]["clip_std"] = round(clip_result["std"], 4)
        results["metrics"]["clip_min"] = round(clip_result["min_score"], 4)
        results["metrics"]["clip_max"] = round(clip_result["max_score"], 4)
        results["metrics"]["clip_n_pairs"] = clip_result["num_pairs"]
    except Exception as e:
        logger.error(f"CLIP score computation failed: {e}")
        results["metrics"]["clip_mean"] = None
        results["metrics"]["clip_error"] = str(e)

    results["runtime"]["clip_s"] = round(time.perf_counter() - t_clip_start, 2)

    # ── Step 5: Save Results ───────────────────────────────────────────────
    metrics_dir = Path(config.output.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / get_timestamped_filename("metrics", ".json")
    with open(metrics_file, "w") as f:
        # Exclude per-pair scores from JSON (too large)
        save_results = {k: v for k, v in results.items() if k != "all_clip_scores"}
        json.dump(save_results, f, indent=2, default=str)

    logger.info(f"Results saved to: {metrics_file}")

    # ── Print Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:          {results['model_info']['base_model']}")
    print(f"LoRA:           {results['model_info']['lora_path'] or 'None (base model)'}")
    print(f"Num images:     {config.generation.num_images}")
    print()
    fid = results["metrics"].get("fid")
    clip = results["metrics"].get("clip_mean")
    clip_std = results["metrics"].get("clip_std")
    print(f"FID Score:      {fid:.4f}" if fid is not None else "FID Score:      ERROR")
    print(f"CLIP Score:     {clip:.4f} ± {clip_std:.4f}" if clip is not None else "CLIP Score:     ERROR")
    print()
    print(f"Runtime:        Generation={results['runtime'].get('generation_s', '?')}s, "
          f"FID={results['runtime'].get('fid_s', '?')}s, "
          f"CLIP={results['runtime'].get('clip_s', '?')}s")
    print(f"Results file:   {metrics_file}")
    print("=" * 60)

    return results


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run full FID + CLIP evaluation")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Override config's lora_weights_path")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Override number of images to generate")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.lora_path:
        config.model.lora_weights_path = args.lora_path
    if args.num_images:
        config.generation.num_images = args.num_images

    run_evaluation(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
