"""
src/inference/txt2img.py
------------------------
Text-to-Image generation script.

HOW TEXT-TO-IMAGE WORKS:
  1. Start with pure Gaussian noise z_T ~ N(0, I) in latent space (4×64×64)
  2. Encode your text prompt → text embeddings (CLIP text encoder)
  3. Run denoising loop for T steps:
     - UNet predicts the noise at each step
     - Scheduler removes predicted noise → slightly cleaner latent
  4. Decode final latent z_0 through VAE → pixel image (3×512×512)

The key insight: we NEVER work directly in pixel space during generation.
Stable Diffusion operates in a compressed latent space (8x smaller),
which makes it 64x faster than pixel-space diffusion.

CLASSIFIER-FREE GUIDANCE (CFG):
  During inference, we run the UNet twice per step:
    - Once conditioned on text prompt → text-guided prediction
    - Once conditioned on empty string → unconditional prediction
  Final prediction = uncond + guidance_scale * (cond - uncond)

  guidance_scale=1: purely random (ignores text)
  guidance_scale=7.5: balanced (default, good quality)
  guidance_scale=15+: very strict text adherence but artifacts

Usage:
  python src/inference/txt2img.py --prompt "a cat in space"
  python src/inference/txt2img.py --prompt "a cat" --seed 42 --steps 20 --output out.png
  python src/inference/txt2img.py --smoke_test
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

# Running this file directly puts its own directory on sys.path, not the repo
# root, so "from src...." would fail. Add the repo root before any src import.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.models.pipeline_utils import load_txt2img_pipeline, SCHEDULER_MAP
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed_utils import seed_generator, seed_everything

logger = get_logger(__name__)


def run_txt2img(
    prompt: str,
    negative_prompt: str = "blurry, low quality, watermark, deformed",
    width: int = 512,
    height: int = 512,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = 42,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: str | None = None,
    scheduler: str = "DPMSolverMultistepScheduler",
    output_path: str | None = None,
    device: str = "auto",
    show_timing: bool = True,
) -> tuple[Image.Image, dict]:
    """
    Generate an image from a text prompt.

    Args:
        prompt: Text description of the desired image.
        negative_prompt: Text describing what to AVOID in the image.
        width, height: Output image dimensions (must be multiples of 8).
        num_steps: Denoising steps (more = better quality, slower).
        guidance_scale: How strictly to follow the prompt (CFG scale).
        seed: Random seed for reproducibility. None = random.
        model_id: HuggingFace model ID or local path.
        lora_path: Path to LoRA adapter directory (optional).
        scheduler: Name of denoising scheduler.
        output_path: If provided, save image to this path.
        device: "auto" | "cuda" | "cpu" | "mps"
        show_timing: Print timing breakdown.

    Returns:
        (PIL Image, timing_dict)
    """
    # ── Load Pipeline ──────────────────────────────────────────
    t0 = time.perf_counter()
    pipe = load_txt2img_pipeline(
        model_id=model_id,
        lora_weights_path=lora_path,
        scheduler_name=scheduler,
        device=device,
    )
    t_load = time.perf_counter() - t0

    # ── Generate ───────────────────────────────────────────────
    t1 = time.perf_counter()
    generator = seed_generator(seed)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=1,
    )
    image = output.images[0]
    t_inference = time.perf_counter() - t1

    timing = {
        "load_s": round(t_load, 3),
        "inference_s": round(t_inference, 3),
        "ms_per_step": round(t_inference * 1000 / num_steps, 1),
    }

    if show_timing:
        logger.info(
            f"Generated in {t_inference:.2f}s "
            f"({timing['ms_per_step']}ms/step, {num_steps} steps)"
        )

    # ── Save ───────────────────────────────────────────────────
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        logger.info(f"Saved to: {out}")

    return image, timing


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Text-to-Image generation with Stable Diffusion")
    parser.add_argument("--prompt", type=str, default="a beautiful mountain lake at sunset, photorealistic")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, watermark, deformed, ugly")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter dir")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", type=str, default="DPMSolverMultistepScheduler",
                        choices=list(SCHEDULER_MAP.keys()))
    parser.add_argument("--output", type=str, default="experiments/samples/txt2img_output.png")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick test with 2 steps at 256x256")
    args = parser.parse_args()

    if args.smoke_test:
        logger.info("SMOKE TEST: using 2 steps at 256x256")
        args.steps = 2
        args.width = 256
        args.height = 256
        args.prompt = "a red circle on white background"
        args.output = "experiments/samples/smoke_test_txt2img.png"

    image, timing = run_txt2img(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        model_id=args.model_id,
        lora_path=args.lora_path,
        scheduler=args.scheduler,
        output_path=args.output,
        device=args.device,
    )

    print(f"\nResult: {args.width}x{args.height} image")
    print(f"Timing: {timing}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
