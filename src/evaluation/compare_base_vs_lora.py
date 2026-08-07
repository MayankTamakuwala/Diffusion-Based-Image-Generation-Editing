"""
src/evaluation/compare_base_vs_lora.py
--------------------------------------
Compare base Stable Diffusion against the LoRA fine-tune, on identical
prompts and identical seeds.

WHY THIS SCRIPT EXISTS:
  Validation images generated during training look Impressionist -- but the
  validation prompt literally says "an Impressionism painting, by Claude
  Monet", and base SD 1.5 already knows Monet. So those images cannot tell
  you what the LoRA contributed. Neither can an FID number on its own: FID
  is only meaningful relative to a baseline computed the same way.

  This produces both halves of the comparison:
    1. Visual A/B  -- same prompt, same seed, base vs LoRA, side by side.
    2. Metric A/B  -- FID and CLIP for each arm against the same held-out set.

WHY THE SAME SEED MATTERS:
  Diffusion output depends heavily on the initial noise. Comparing base at
  seed 0 against LoRA at seed 1 shows you two different images and tells you
  nothing. Fixing the seed means the ONLY difference between the two images
  is the adapter, so any change is attributable to fine-tuning.

WHY SEQUENTIAL PIPELINE LOADING:
  We load base, generate everything, free it, then load LoRA. Holding two SD
  pipelines at once works on a big GPU but needlessly doubles VRAM, and this
  script should run anywhere.

READING THE RESULTS:
  FID  - lower is better. Measures distance between the generated
         distribution and real held-out WikiArt. A large drop from base to
         LoRA means the fine-tune moved output toward the target domain.
  CLIP - higher is better, but only slightly informative here. Both arms use
         the same prompts, and a style adapter mostly changes appearance
         rather than semantic prompt adherence. A big CLIP *drop* is the
         signal worth worrying about: it means the model overfit style at
         the cost of following the prompt.

Usage:
  # visual grid only (fast, ~1 min)
  python src/evaluation/compare_base_vs_lora.py --lora_path lora_weights/final_lora_adapter

  # visual grid + full FID/CLIP for both arms (slow, needs GPU)
  python src/evaluation/compare_base_vs_lora.py \
      --lora_path lora_weights/final_lora_adapter --metrics --num_images 1000

  python src/evaluation/compare_base_vs_lora.py --lora_path ... --smoke_test
"""

import argparse
import gc
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# Running this file directly puts its own directory on sys.path, not the repo
# root, so "from src...." would fail. Add the repo root before any src import.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.utils.logging_utils import get_logger, setup_logging, get_timestamped_filename
from src.utils.seed_utils import seed_generator

logger = get_logger(__name__)

LABEL_STRIP_PX = 32


def load_eval_prompts(val_dir: str | Path, count: int) -> list[str]:
    """
    Read the first `count` captions from the exported val split.

    Using the real val captions (rather than hand-written prompts) keeps the
    comparison honest: these are the same caption format the model trained
    on, drawn from images it has never seen.
    """
    captions_dir = Path(val_dir) / "captions"
    if not captions_dir.exists():
        raise FileNotFoundError(
            f"No captions at {captions_dir}.\n"
            f"Run: python src/data/export_wikiart_val.py"
        )

    caption_files = sorted(captions_dir.glob("*.txt"))[:count]
    prompts = [f.read_text(encoding="utf-8").strip() for f in caption_files]
    prompts = [p for p in prompts if p]

    if not prompts:
        raise ValueError(f"All caption files in {captions_dir} were empty")

    return prompts


def _generate_with_pipeline(
    pipe,
    prompts: list[str],
    seeds: list[int],
    num_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    desc: str,
) -> list[Image.Image]:
    """Generate one image per (prompt, seed) pair with an already-loaded pipeline."""
    images = []
    for prompt, seed in tqdm(list(zip(prompts, seeds)), desc=desc):
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=seed_generator(seed),
            )
        images.append(out.images[0])
    return images


def _free(pipe) -> None:
    """Release a pipeline's VRAM before loading the next one."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_comparison_grid(
    base_images: list[Image.Image],
    lora_images: list[Image.Image],
    prompts: list[str],
    output_path: str | Path,
) -> Path:
    """
    Stack base and LoRA output into a labelled two-column contact sheet.

    Column 1 = base SD 1.5, column 2 = base + LoRA, one row per prompt, with
    the shared seed noted so a reviewer can regenerate any cell.
    """
    if not base_images:
        raise ValueError("No images to compose")

    cell_w, cell_h = base_images[0].size
    n_rows = len(base_images)

    grid = Image.new(
        "RGB",
        (cell_w * 2, cell_h * n_rows + LABEL_STRIP_PX),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(grid)
    draw.text((cell_w // 2 - 30, 10), "BASE SD 1.5", fill=(0, 0, 0))
    draw.text((cell_w + cell_w // 2 - 40, 10), "BASE + LoRA", fill=(0, 0, 0))

    for row, (base_img, lora_img) in enumerate(zip(base_images, lora_images)):
        y = LABEL_STRIP_PX + row * cell_h
        grid.paste(base_img, (0, y))
        grid.paste(lora_img, (cell_w, y))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    logger.info(f"Comparison grid saved: {out_path}")

    # Also write the prompt/seed manifest -- the grid is unreadable without it.
    manifest = out_path.with_suffix(".txt")
    lines = [f"row {i}: {p}" for i, p in enumerate(prompts)]
    manifest.write_text("\n".join(lines), encoding="utf-8")

    return out_path


def run_visual_comparison(
    config: OmegaConf,
    lora_path: str,
    num_samples: int = 6,
    output_dir: str | Path = "experiments/comparisons",
) -> dict:
    """
    Generate the side-by-side base vs LoRA grid.

    Returns a summary dict with the grid path and the prompts/seeds used.
    """
    from src.models.pipeline_utils import load_txt2img_pipeline

    val_dir = config.get("eval_data", {}).get("val_dir", "dataset/wikiart_val")
    prompts = load_eval_prompts(val_dir, num_samples)
    # Deterministic, distinct seeds -- shared across both arms.
    seeds = [config.generation.seed + i for i in range(len(prompts))]

    gen_kwargs = dict(
        num_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        width=config.generation.width,
        height=config.generation.height,
    )

    logger.info(f"Comparing {len(prompts)} prompts at seeds {seeds}")

    # ── Arm 1: base model ──────────────────────────────────────
    logger.info("Loading BASE pipeline (no LoRA)...")
    base_pipe = load_txt2img_pipeline(
        model_id=config.model.base_model,
        lora_weights_path=None,
        scheduler_name=config.generation.scheduler,
    )
    base_images = _generate_with_pipeline(
        base_pipe, prompts, seeds, desc="base", **gen_kwargs
    )
    _free(base_pipe)

    # ── Arm 2: LoRA ────────────────────────────────────────────
    logger.info(f"Loading LoRA pipeline from {lora_path}...")
    lora_pipe = load_txt2img_pipeline(
        model_id=config.model.base_model,
        lora_weights_path=lora_path,
        scheduler_name=config.generation.scheduler,
    )
    lora_images = _generate_with_pipeline(
        lora_pipe, prompts, seeds, desc="lora", **gen_kwargs
    )
    _free(lora_pipe)

    out_root = Path(output_dir)
    grid_path = out_root / get_timestamped_filename("base_vs_lora", ".png")
    build_comparison_grid(base_images, lora_images, prompts, grid_path)

    # Save the individual frames too, for picking single examples later.
    for i, (b_img, l_img) in enumerate(zip(base_images, lora_images)):
        b_img.save(out_root / f"pair_{i:02d}_seed{seeds[i]}_base.png")
        l_img.save(out_root / f"pair_{i:02d}_seed{seeds[i]}_lora.png")

    return {
        "grid": str(grid_path),
        "prompts": prompts,
        "seeds": seeds,
        "num_pairs": len(prompts),
    }


def run_metric_comparison(config: OmegaConf, lora_path: str, smoke_test: bool) -> dict:
    """
    Run the full FID + CLIP evaluation for both arms against the same
    reference set, and return both sets of metrics plus the deltas.
    """
    from src.evaluation.run_eval import run_evaluation

    arms = {}
    for arm_name, arm_lora in (("base", None), ("lora", lora_path)):
        logger.info(f"===== Evaluating arm: {arm_name} =====")
        arm_config = deepcopy(config)
        arm_config.model.lora_weights_path = arm_lora
        # Separate output dirs, or the second arm's FID would be computed
        # over a directory still holding the first arm's images.
        arm_config.generation.output_dir = f"experiments/eval_generated_{arm_name}"
        arms[arm_name] = run_evaluation(arm_config, smoke_test=smoke_test)["metrics"]

    base_m, lora_m = arms["base"], arms["lora"]

    def _delta(key):
        b, l = base_m.get(key), lora_m.get(key)
        return round(l - b, 4) if (b is not None and l is not None) else None

    return {
        "base": base_m,
        "lora": lora_m,
        "delta": {
            "fid": _delta("fid"),          # negative = LoRA closer to real
            "clip_mean": _delta("clip_mean"),  # positive = better adherence
        },
    }


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Compare base SD against the LoRA fine-tune at fixed seeds"
    )
    parser.add_argument("--config", type=str, default="config/eval_config.yaml")
    parser.add_argument("--lora_path", type=str, default="lora_weights/final_lora_adapter")
    parser.add_argument("--num_samples", type=int, default=6,
                        help="How many prompts in the visual grid")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Override images per arm for FID (default: config value)")
    parser.add_argument("--metrics", action="store_true",
                        help="Also run FID + CLIP for both arms (slow)")
    parser.add_argument("--no_visual", action="store_true",
                        help="Skip the visual grid, metrics only")
    parser.add_argument("--output_dir", type=str, default="experiments/comparisons")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if not Path(args.lora_path).exists():
        raise FileNotFoundError(
            f"No adapter at {args.lora_path}. Train first, or pass --lora_path."
        )

    config = OmegaConf.load(args.config)
    if args.num_images:
        config.generation.num_images = args.num_images
    if args.smoke_test:
        args.num_samples = min(args.num_samples, 2)

    results = {
        "timestamp": datetime.now().isoformat(),
        "base_model": config.model.base_model,
        "lora_path": args.lora_path,
    }

    if not args.no_visual:
        results["visual"] = run_visual_comparison(
            config, args.lora_path, args.num_samples, args.output_dir
        )

    if args.metrics:
        results["metrics"] = run_metric_comparison(config, args.lora_path, args.smoke_test)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / get_timestamped_filename("comparison", ".json")
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ── Report ─────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("BASE vs LoRA COMPARISON")
    print("=" * 62)
    print(f"Base model:  {config.model.base_model}")
    print(f"LoRA:        {args.lora_path}")

    if "visual" in results:
        print(f"\nGrid:        {results['visual']['grid']}")
        print(f"Pairs:       {results['visual']['num_pairs']} (matched seeds)")

    if "metrics" in results:
        m = results["metrics"]
        print(f"\n{'Metric':<14}{'Base':>12}{'LoRA':>12}{'Delta':>12}")
        print("-" * 50)
        for key, better in (("fid", "lower"), ("clip_mean", "higher")):
            b = m["base"].get(key)
            l = m["lora"].get(key)
            d = m["delta"].get(key)
            fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "ERROR"
            print(f"{key:<14}{fmt(b):>12}{fmt(l):>12}{fmt(d):>12}   ({better} is better)")

        fid_delta = m["delta"].get("fid")
        if fid_delta is not None:
            verdict = "moved output toward real WikiArt" if fid_delta < 0 else "moved output AWAY from real WikiArt"
            print(f"\nFID delta {fid_delta:+.4f}: the adapter {verdict}.")

    print(f"\nSaved:       {json_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
