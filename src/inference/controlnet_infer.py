"""
src/inference/controlnet_infer.py
----------------------------------
ControlNet (Canny Edge) conditioned generation.

WHAT IS CONTROLNET?
ControlNet adds structural control to Stable Diffusion generation.
Without ControlNet: your prompt might put objects anywhere.
With ControlNet (canny): the generated image MUST follow the edge structure
of your input image, while still matching your text prompt.

HOW CONTROLNET WORKS (technically):
  1. Input image → Canny edge detection → edge map (binary: edges=white, bg=black)
  2. ControlNet processes this edge map through a copy of SD's UNet ENCODER
  3. The ControlNet outputs are added to the base UNet decoder via "zero convolutions"
     (initialized to zero, so initially the ControlNet adds nothing — stable start)
  4. These additions steer generation to match the edge structure

ARCHITECTURE:
  Normal SD UNet:  Encoder → Middle → Decoder
  SD + ControlNet: Encoder → Middle → Decoder
                   ControlNet processes: edge_map → ControlNet features
                   ControlNet features ADDED TO each decoder block

WHY CANNY?
Canny edges are the most widely supported ControlNet conditioning type.
The canny ControlNet from lllyasviel (the original ControlNet author) was
trained on large datasets and works reliably for:
  - Architecture/structural preservation
  - Pose-guided generation (with OpenPose ControlNet)
  - Depth-guided generation (with depth ControlNet)

CONTROLNET_CONDITIONING_SCALE:
  Controls how strongly to follow the edge map:
  0.5 = loosely follow edges (more creative freedom)
  1.0 = strictly follow edges (accurate structure)
  1.5+ = very strict (may cause artifacts)

Usage:
  python src/inference/controlnet_infer.py --input photo.jpg --prompt "a pencil sketch"
  python src/inference/controlnet_infer.py --smoke_test
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.pipeline_utils import (
    load_controlnet_pipeline,
    get_canny_edge_map,
    SCHEDULER_MAP,
)
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed_utils import seed_generator

logger = get_logger(__name__)


def run_controlnet(
    input_image: Image.Image | str | Path,
    prompt: str,
    negative_prompt: str = "blurry, low quality, watermark, deformed",
    width: int = 512,
    height: int = 512,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    canny_low: int = 100,
    canny_high: int = 200,
    seed: int | None = 42,
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny",
    lora_path: str | None = None,
    scheduler: str = "DPMSolverMultistepScheduler",
    output_path: str | None = None,
    save_edge_map: bool = False,
    device: str = "auto",
) -> tuple[Image.Image, Image.Image, dict]:
    """
    Generate an image guided by Canny edges from an input image.

    Args:
        input_image: Source image for edge extraction.
        prompt: Text description of desired output.
        negative_prompt: What to avoid.
        width, height: Output size (should match input for best results).
        num_steps: Denoising steps.
        guidance_scale: CFG scale.
        controlnet_conditioning_scale: ControlNet influence strength.
        canny_low, canny_high: Canny edge detection thresholds.
        seed: Reproducibility seed.
        base_model_id: SD base model.
        controlnet_model_id: ControlNet checkpoint.
        lora_path: Optional LoRA adapter.
        scheduler: Denoising scheduler.
        output_path: If set, save generated image here.
        save_edge_map: If True, also save the edge map image.
        device: Compute device.

    Returns:
        (generated_image, edge_map, timing_dict)
    """
    # ── Load and preprocess input ──────────────────────────────
    t0 = time.perf_counter()

    if isinstance(input_image, (str, Path)):
        input_image = Image.open(input_image).convert("RGB")
    else:
        input_image = input_image.convert("RGB")

    # Snap to multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8
    input_image = input_image.resize((width, height), Image.LANCZOS)

    # Compute Canny edge map — this is the ControlNet conditioning signal
    edge_map = get_canny_edge_map(input_image, low_threshold=canny_low, high_threshold=canny_high)
    t_preprocess = time.perf_counter() - t0

    # ── Load pipeline ──────────────────────────────────────────
    t1 = time.perf_counter()
    pipe = load_controlnet_pipeline(
        base_model_id=base_model_id,
        controlnet_model_id=controlnet_model_id,
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
        image=edge_map,              # ControlNet conditioning input
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    )
    generated_image = output.images[0]
    t_inference = time.perf_counter() - t2

    timing = {
        "preprocess_ms": round(t_preprocess * 1000, 1),
        "load_s": round(t_load, 3),
        "inference_s": round(t_inference, 3),
        "ms_per_step": round(t_inference * 1000 / num_steps, 1),
    }

    logger.info(
        f"ControlNet generation complete: {t_inference:.2f}s, "
        f"conditioning_scale={controlnet_conditioning_scale}"
    )

    # ── Save outputs ───────────────────────────────────────────
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        generated_image.save(out)
        logger.info(f"Generated image saved: {out}")

        if save_edge_map:
            edge_path = out.with_name(out.stem + "_edges" + out.suffix)
            edge_map.save(edge_path)
            logger.info(f"Edge map saved: {edge_path}")

    return generated_image, edge_map, timing


def create_triptych(
    original: Image.Image,
    edge_map: Image.Image,
    generated: Image.Image,
) -> Image.Image:
    """
    Create a 3-panel comparison: original | edges | generated.

    Great for demos and documentation — shows the full pipeline visually.
    """
    w, h = original.size
    canvas = Image.new("RGB", (w * 3 + 20, h), color=(40, 40, 40))
    canvas.paste(original, (0, 0))
    canvas.paste(edge_map, (w + 10, 0))
    canvas.paste(generated, (w * 2 + 20, 0))
    return canvas


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="ControlNet (Canny) conditioned generation")
    parser.add_argument("--input", type=str, default=None, help="Input image path")
    parser.add_argument("--prompt", type=str,
                        default="a detailed architectural drawing, technical illustration, high quality")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, watermark, deformed")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0,
                        help="ControlNet conditioning strength (0.5-1.5)")
    parser.add_argument("--canny_low", type=int, default=100)
    parser.add_argument("--canny_high", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default="DPMSolverMultistepScheduler",
                        choices=list(SCHEDULER_MAP.keys()))
    parser.add_argument("--output", type=str, default="experiments/samples/controlnet_output.png")
    parser.add_argument("--save_edges", action="store_true", help="Also save edge map")
    parser.add_argument("--save_triptych", action="store_true", help="Save 3-panel comparison")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    # ── Smoke test: use a synthetic image ─────────────────────
    if args.smoke_test:
        logger.info("SMOKE TEST: using synthetic checkerboard input")
        # Create a synthetic checkerboard image (has lots of edges for canny)
        checker = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    checker[i:i+32, j:j+32] = 200
        input_image = Image.fromarray(checker)
        args.steps = 2
        args.width = 256
        args.height = 256
        args.prompt = "colorful abstract art"
        args.output = "experiments/samples/smoke_test_controlnet.png"

        generated, edge_map, timing = run_controlnet(
            input_image=input_image,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            seed=args.seed,
            base_model_id=args.base_model,
            controlnet_model_id=args.controlnet_model,
            output_path=args.output,
            device=args.device,
        )
    else:
        if args.input is None:
            parser.error("--input is required (or use --smoke_test)")

        generated, edge_map, timing = run_controlnet(
            input_image=args.input,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            seed=args.seed,
            base_model_id=args.base_model,
            controlnet_model_id=args.controlnet_model,
            lora_path=args.lora_path,
            scheduler=args.scheduler,
            output_path=args.output,
            save_edge_map=args.save_edges,
            device=args.device,
        )

        if args.save_triptych and args.input:
            original = Image.open(args.input).convert("RGB")
            triptych = create_triptych(original, edge_map, generated)
            trip_path = Path(args.output).with_name(
                Path(args.output).stem + "_triptych.png"
            )
            triptych.save(trip_path)
            logger.info(f"Triptych saved: {trip_path}")

    print(f"\nOutput: {args.output}")
    print(f"Timing: {timing}")


if __name__ == "__main__":
    main()
