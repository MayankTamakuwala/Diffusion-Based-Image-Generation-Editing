"""
src/inference/img2img.py
------------------------
Image-to-Image generation script.

HOW IMG2IMG DIFFERS FROM TXT2IMG:
  Instead of starting from pure noise, we:
  1. Encode the INPUT image to latents via VAE
  2. ADD PARTIAL NOISE to those latents based on `strength`:
     - strength=0.0: don't change anything (pure copy)
     - strength=0.5: start 50% through denoising (moderate change)
     - strength=1.0: add full noise → effectively txt2img (completely replace)
  3. Run the denoising loop for (strength × num_steps) steps
  4. Decode back to pixel space

WHY PARTIAL NOISE?
  The partial noise preserves the overall structure and composition
  of the input image. The denoising process then fills in details
  guided by your prompt. This is why img2img is great for:
  - Style transfer (same composition, different style)
  - Variation generation (similar image, different details)
  - Sketch-to-photo (rough sketch → realistic photo)
  - Photo upscaling/enhancement

STRENGTH GUIDE:
  0.3 — subtle changes; keep most of original
  0.5 — moderate changes; rough structure preserved
  0.75 — significant changes; only broad composition preserved
  0.9 — drastic changes; barely resembles original

Usage:
  python src/inference/img2img.py --input photo.jpg --prompt "in anime style"
  python src/inference/img2img.py --input photo.jpg --prompt "oil painting" --strength 0.6
  python src/inference/img2img.py --smoke_test
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.pipeline_utils import load_img2img_pipeline, SCHEDULER_MAP
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed_utils import seed_generator

logger = get_logger(__name__)


def load_and_preprocess_image(
    image_path: str | Path,
    target_width: int = 512,
    target_height: int = 512,
) -> Image.Image:
    """
    Load an image and resize it to target dimensions.

    Stable Diffusion requires dimensions divisible by 8 (due to VAE downsampling).
    We use LANCZOS resampling which gives higher quality than bilinear.

    Args:
        image_path: Path to input image.
        target_width, target_height: Resize target (should be multiples of 8).

    Returns:
        RGB PIL Image resized to (target_width, target_height).
    """
    img = Image.open(image_path).convert("RGB")
    # Snap to nearest multiple of 8 (just in case)
    target_width = (target_width // 8) * 8
    target_height = (target_height // 8) * 8
    img = img.resize((target_width, target_height), Image.LANCZOS)
    return img


def run_img2img(
    input_image: Image.Image | str | Path,
    prompt: str,
    negative_prompt: str = "blurry, low quality, watermark, deformed",
    strength: float = 0.75,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = 42,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: str | None = None,
    scheduler: str = "DPMSolverMultistepScheduler",
    output_path: str | None = None,
    device: str = "auto",
) -> tuple[Image.Image, dict]:
    """
    Transform an input image using a text prompt.

    Args:
        input_image: PIL Image or path to input image.
        prompt: Text description of the desired output.
        negative_prompt: What to avoid.
        strength: How much to change the image (0.0-1.0).
        num_steps: Total denoising steps (effective steps = strength × num_steps).
        guidance_scale: CFG scale.
        seed: Random seed.
        model_id: SD model.
        lora_path: Optional LoRA adapter.
        scheduler: Denoising scheduler name.
        output_path: If set, save output image here.
        device: Compute device.

    Returns:
        (output PIL Image, timing_dict)
    """
    # ── Load and preprocess input ──────────────────────────────
    t0 = time.perf_counter()

    if isinstance(input_image, (str, Path)):
        input_image = load_and_preprocess_image(input_image)
    else:
        # Ensure it's RGB
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

    t_preprocess = time.perf_counter() - t0

    # ── Load pipeline ──────────────────────────────────────────
    t1 = time.perf_counter()
    pipe = load_img2img_pipeline(
        model_id=model_id,
        lora_weights_path=lora_path,
        scheduler_name=scheduler,
        device=device,
    )
    t_load = time.perf_counter() - t1

    # ── Generate ───────────────────────────────────────────────
    t2 = time.perf_counter()
    generator = seed_generator(seed)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=strength,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    image = output.images[0]
    t_inference = time.perf_counter() - t2

    timing = {
        "preprocess_ms": round(t_preprocess * 1000, 1),
        "load_s": round(t_load, 3),
        "inference_s": round(t_inference, 3),
        "effective_steps": round(strength * num_steps),
    }

    logger.info(
        f"img2img complete in {t_inference:.2f}s "
        f"(strength={strength}, ~{timing['effective_steps']} effective steps)"
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        logger.info(f"Saved to: {out}")

    return image, timing


def create_side_by_side(original: Image.Image, generated: Image.Image) -> Image.Image:
    """Create a side-by-side comparison image for easy visual inspection."""
    w, h = original.size
    canvas = Image.new("RGB", (w * 2, h), color=(200, 200, 200))
    canvas.paste(original, (0, 0))
    canvas.paste(generated, (w, 0))
    return canvas


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Image-to-Image generation with Stable Diffusion")
    parser.add_argument("--input", type=str, default=None, help="Input image path")
    parser.add_argument("--prompt", type=str, default="a beautiful oil painting of this scene, impressionist style")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, watermark")
    parser.add_argument("--strength", type=float, default=0.75, help="Change strength 0.0-1.0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default="DPMSolverMultistepScheduler",
                        choices=list(SCHEDULER_MAP.keys()))
    parser.add_argument("--output", type=str, default="experiments/samples/img2img_output.png")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save original + generated side-by-side")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        logger.info("SMOKE TEST: creating synthetic input image")
        # Create a simple test image
        import numpy as np
        test_img = Image.fromarray(
            np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        )
        args.steps = 2
        args.prompt = "a colorful abstract painting"
        args.output = "experiments/samples/smoke_test_img2img.png"
        image, timing = run_img2img(
            input_image=test_img,
            prompt=args.prompt,
            strength=args.strength,
            num_steps=args.steps,
            seed=args.seed,
            model_id=args.model_id,
            output_path=args.output,
            device=args.device,
        )
    else:
        if args.input is None:
            parser.error("--input is required (or use --smoke_test)")

        image, timing = run_img2img(
            input_image=args.input,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            model_id=args.model_id,
            lora_path=args.lora_path,
            scheduler=args.scheduler,
            output_path=args.output,
            device=args.device,
        )

        if args.save_comparison and args.input:
            original = Image.open(args.input).convert("RGB")
            comparison = create_side_by_side(original, image)
            comp_path = Path(args.output).with_suffix("_comparison.png")
            comparison.save(comp_path)
            logger.info(f"Comparison saved: {comp_path}")

    print(f"\nResult saved to: {args.output}")
    print(f"Timing: {timing}")


if __name__ == "__main__":
    main()
