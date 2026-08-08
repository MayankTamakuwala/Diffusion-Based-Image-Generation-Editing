"""
app/gradio_app.py
-----------------
Gradio Web UI for Diffusion-Based Image Generation & Editing.

TABS:
  1. Text-to-Image  — generate from text prompt
  2. Image-to-Image — transform an existing image
  3. ControlNet     — edge-guided generation

ARCHITECTURE:
  - Pipelines load ONCE at startup (expensive — models are ~3GB each)
  - All three pipelines share the same base model (different classes)
  - LoRA weights can be loaded/unloaded via sidebar
  - Each inference call:
      preprocess → run pipeline → postprocess → display image + timing

LATENCY TARGETS:
  - GPU (RTX 3080): ~80-120ms/step, ~2-4s total (30 steps)
  - Fast mode (10 steps): ~0.8-1.5s
  - CPU: ~5-10s/step (30 steps = 2.5-5 minutes) — use smoke_test mode

DESIGN CHOICES:
  - Gradio Blocks API (more flexible than Interface API)
  - gr.State for persistent pipeline cache (avoid reloading between requests)
  - Timing breakdown panel shows: preprocess, inference, postprocess
  - All user-facing sliders have min/max guardrails

HOW GRADIO WORKS (for beginners):
  - gr.Blocks() creates a "canvas" for your UI
  - gr.Textbox(), gr.Slider(), gr.Image() are UI widgets
  - btn.click(fn, inputs, outputs) connects a button to a Python function
  - When user clicks "Generate", Gradio calls your Python function and
    displays the returned PIL Image in the output gr.Image widget
  - Gradio runs a local web server; you access it at http://127.0.0.1:7860

Usage:
  python app/gradio_app.py
  python app/gradio_app.py --smoke_test   (fast CPU mode)
  python app/gradio_app.py --share        (create public URL via ngrok)
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from PIL import Image

# ── Import our pipeline utilities ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Running this file directly puts its own directory on sys.path, not the repo
# root, so "from src...." would fail. Add the repo root before any src import.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.models.pipeline_utils import (
    load_txt2img_pipeline,
    load_img2img_pipeline,
    load_controlnet_pipeline,
    get_canny_edge_map,
    get_device_and_dtype,
    SCHEDULER_MAP,
)
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed_utils import seed_generator, get_random_seed

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PIPELINE CACHE
# We load pipelines once at startup and keep them alive for the app's lifetime.
# Loading takes ~10-30 seconds (download + GPU placement).
# Subsequent inference is fast because everything is already in GPU memory.
# ─────────────────────────────────────────────────────────────────────────────
_pipelines = {
    "txt2img": None,
    "img2img": None,
    "controlnet": None,
}
_current_lora_path = None  # track which LoRA is loaded


def get_device_info() -> str:
    """Return a human-readable device description for the UI info panel."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"GPU: {gpu_name} ({vram:.1f} GB VRAM)"
    elif torch.backends.mps.is_available():
        return "Device: Apple Silicon MPS"
    else:
        return "Device: CPU (slow — use smoke_test mode)"


def warmup_pipeline(pipe, device: torch.device, smoke_test: bool = False) -> None:
    """
    Run a single forward pass to warm up CUDA kernels.

    WHY WARMUP?
    The first inference call is always slow because:
    1. CUDA kernels must be JIT-compiled (even without torch.compile)
    2. Memory is allocated and garbage-collected once
    3. cuDNN benchmarks to find fastest convolution algorithm

    After warmup, subsequent calls are consistently fast.
    Warmup takes ~1-5 seconds.
    """
    if device.type == "cpu":
        return  # No warmup on CPU (too slow)

    logger.info("Running warmup pass...")
    try:
        with torch.no_grad():
            pipe(
                prompt="warmup",
                num_inference_steps=1,
                width=256 if smoke_test else 512,
                height=256 if smoke_test else 512,
                output_type="latent",  # skip VAE decode (faster warmup)
            )
        logger.info("Warmup complete")
    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {e}")


def load_pipelines(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    controlnet_id: str = "lllyasviel/sd-controlnet-canny",
    lora_path: Optional[str] = None,
    lora_scale: float = 0.6,
    enable_xformers: bool = True,
    enable_torch_compile: bool = False,
    smoke_test: bool = False,
) -> str:
    """
    Load all three pipelines and cache them globally.

    Called once at startup. Returns status message for UI.

    Args:
        model_id: Base SD model.
        controlnet_id: ControlNet checkpoint.
        lora_path: Optional LoRA adapter path.
        lora_scale: Adapter strength. Defaults to 0.6, the value chosen by
            matched-seed sweep -- full strength dissolves composition in
            complex scenes. See config/eval_config.yaml.
        enable_xformers: Enable xformers attention.
        enable_torch_compile: Compile UNet.
        smoke_test: If True, skip warmup on GPU.

    Returns:
        Status string for display in UI.
    """
    global _pipelines, _current_lora_path

    device, dtype = get_device_and_dtype("auto", "auto")
    status_parts = []

    logger.info(f"Loading pipelines: {model_id} on {device}")

    # Load txt2img
    logger.info("Loading txt2img pipeline...")
    t0 = time.perf_counter()
    _pipelines["txt2img"] = load_txt2img_pipeline(
        model_id=model_id,
        lora_weights_path=lora_path,
        lora_scale=lora_scale,
        enable_xformers=enable_xformers,
        enable_torch_compile=enable_torch_compile,
    )
    if not smoke_test:
        warmup_pipeline(_pipelines["txt2img"], device)
    t_txt2img = time.perf_counter() - t0
    status_parts.append(f"txt2img: {t_txt2img:.1f}s")

    # Load img2img
    # Share weights with txt2img where possible
    # In diffusers, we can create img2img from txt2img components to save memory
    logger.info("Loading img2img pipeline...")
    t1 = time.perf_counter()
    _pipelines["img2img"] = load_img2img_pipeline(
        model_id=model_id,
        lora_weights_path=lora_path,
        lora_scale=lora_scale,
        enable_xformers=enable_xformers,
    )
    t_img2img = time.perf_counter() - t1
    status_parts.append(f"img2img: {t_img2img:.1f}s")

    # Load controlnet
    logger.info("Loading ControlNet pipeline...")
    t2 = time.perf_counter()
    _pipelines["controlnet"] = load_controlnet_pipeline(
        base_model_id=model_id,
        controlnet_model_id=controlnet_id,
        lora_weights_path=lora_path,
        lora_scale=lora_scale,
        enable_xformers=enable_xformers,
    )
    t_controlnet = time.perf_counter() - t2
    status_parts.append(f"controlnet: {t_controlnet:.1f}s")

    _current_lora_path = lora_path

    total_t = t_txt2img + t_img2img + t_controlnet
    status = f"✓ All pipelines loaded in {total_t:.1f}s | " + " | ".join(status_parts)
    status += f"\n{get_device_info()}"

    logger.info(status)
    return status


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Text-to-Image
# ─────────────────────────────────────────────────────────────────────────────

def generate_txt2img(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    scheduler: str,
    fast_mode: bool,
    randomize_seed: bool,
) -> tuple[Image.Image | None, str, int]:
    """
    Text-to-image generation handler.

    This function is called by Gradio when the user clicks "Generate".
    Gradio automatically passes the widget values as arguments.

    Returns:
        (image, timing_info_html, seed_used)
    """
    if _pipelines["txt2img"] is None:
        return None, "ERROR: Pipeline not loaded. Check startup logs.", seed

    # Fast mode overrides
    if fast_mode:
        width, height = 512, 512
        steps = 10
        guidance_scale = 5.0

    # Enforce limits to prevent OOM
    width = min(max(width, 256), 1024)
    height = min(max(height, 256), 1024)
    steps = min(max(steps, 1), 100)

    if randomize_seed:
        seed = get_random_seed()

    # Switch scheduler if needed
    from src.models.pipeline_utils import set_scheduler
    try:
        set_scheduler(_pipelines["txt2img"], scheduler)
    except Exception as e:
        logger.warning(f"Scheduler switch failed: {e}")

    # Time each phase separately
    t_start = time.perf_counter()

    generator = seed_generator(seed)

    t_pre = time.perf_counter() - t_start
    t_infer_start = time.perf_counter()

    try:
        output = _pipelines["txt2img"](
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = output.images[0]
    except Exception as e:
        logger.error(f"txt2img generation failed: {e}")
        return None, f"Generation failed: {str(e)}", seed

    t_infer = time.perf_counter() - t_infer_start

    t_post_start = time.perf_counter()
    # Post-processing is minimal (already PIL Image from diffusers)
    t_post = time.perf_counter() - t_post_start

    total_ms = (t_infer + t_pre + t_post) * 1000
    timing_info = (
        f"**Timing:**  "
        f"Preprocess: {t_pre*1000:.1f}ms | "
        f"Inference: {t_infer*1000:.0f}ms ({t_infer*1000/steps:.1f}ms/step) | "
        f"Postprocess: {t_post*1000:.1f}ms | "
        f"**Total: {total_ms:.0f}ms**\n"
        f"**Device:** {get_device_info()} | "
        f"**Seed:** {seed} | "
        f"**Size:** {width}×{height} | "
        f"**Steps:** {steps} | "
        f"**CFG:** {guidance_scale}"
    )

    # Save to experiments
    Path("experiments/samples").mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    image.save(f"experiments/samples/txt2img_{ts}.png")

    return image, timing_info, seed


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Image-to-Image
# ─────────────────────────────────────────────────────────────────────────────

def generate_img2img(
    input_image: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: int,
    scheduler: str,
    fast_mode: bool,
    randomize_seed: bool,
) -> tuple[Image.Image | None, str, int]:
    """Image-to-image transformation handler."""
    if _pipelines["img2img"] is None:
        return None, "ERROR: Pipeline not loaded.", seed
    if input_image is None:
        return None, "ERROR: Please upload an input image.", seed

    if fast_mode:
        steps = 10
        guidance_scale = 5.0

    steps = min(max(steps, 1), 100)

    if randomize_seed:
        seed = get_random_seed()

    from src.models.pipeline_utils import set_scheduler
    try:
        set_scheduler(_pipelines["img2img"], scheduler)
    except Exception:
        pass

    t_pre_start = time.perf_counter()

    # Ensure image is RGB and properly sized
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")
    # Snap to multiple of 8
    w, h = input_image.size
    w = (w // 8) * 8
    h = (h // 8) * 8
    input_image = input_image.resize((w, h), Image.LANCZOS)

    generator = seed_generator(seed)
    t_pre = time.perf_counter() - t_pre_start

    t_infer_start = time.perf_counter()
    try:
        output = _pipelines["img2img"](
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=input_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = output.images[0]
    except Exception as e:
        logger.error(f"img2img failed: {e}")
        return None, f"Generation failed: {str(e)}", seed

    t_infer = time.perf_counter() - t_infer_start
    t_post = 0.001  # negligible for img2img

    total_ms = (t_infer + t_pre + t_post) * 1000
    effective_steps = round(strength * steps)

    timing_info = (
        f"**Timing:**  "
        f"Preprocess: {t_pre*1000:.1f}ms | "
        f"Inference: {t_infer*1000:.0f}ms (~{effective_steps} effective steps) | "
        f"**Total: {total_ms:.0f}ms**\n"
        f"**Device:** {get_device_info()} | "
        f"**Seed:** {seed} | "
        f"**Strength:** {strength} | "
        f"**CFG:** {guidance_scale}"
    )

    Path("experiments/samples").mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    image.save(f"experiments/samples/img2img_{ts}.png")

    return image, timing_info, seed


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ControlNet (Canny)
# ─────────────────────────────────────────────────────────────────────────────

def generate_controlnet(
    input_image: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    canny_low: int,
    canny_high: int,
    seed: int,
    scheduler: str,
    fast_mode: bool,
    randomize_seed: bool,
) -> tuple[Image.Image | None, Image.Image | None, str, int]:
    """ControlNet (Canny) conditioned generation handler."""
    if _pipelines["controlnet"] is None:
        return None, None, "ERROR: Pipeline not loaded.", seed
    if input_image is None:
        return None, None, "ERROR: Please upload an input image.", seed

    if fast_mode:
        width, height = 512, 512
        steps = 10
        guidance_scale = 5.0

    width = min(max((width // 8) * 8, 256), 1024)
    height = min(max((height // 8) * 8, 256), 1024)
    steps = min(max(steps, 1), 100)

    if randomize_seed:
        seed = get_random_seed()

    from src.models.pipeline_utils import set_scheduler
    try:
        set_scheduler(_pipelines["controlnet"], scheduler)
    except Exception:
        pass

    t_pre_start = time.perf_counter()

    # Preprocess: resize + Canny edges
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")
    input_image = input_image.resize((width, height), Image.LANCZOS)
    edge_map = get_canny_edge_map(input_image, low_threshold=canny_low, high_threshold=canny_high)

    generator = seed_generator(seed)
    t_pre = time.perf_counter() - t_pre_start

    t_infer_start = time.perf_counter()
    try:
        output = _pipelines["controlnet"](
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=edge_map,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
        )
        generated = output.images[0]
    except Exception as e:
        logger.error(f"ControlNet generation failed: {e}")
        return None, edge_map, f"Generation failed: {str(e)}", seed

    t_infer = time.perf_counter() - t_infer_start

    t_post_start = time.perf_counter()
    t_post = time.perf_counter() - t_post_start

    total_ms = (t_infer + t_pre + t_post) * 1000
    timing_info = (
        f"**Timing:**  "
        f"Preprocess (incl. Canny): {t_pre*1000:.1f}ms | "
        f"Inference: {t_infer*1000:.0f}ms | "
        f"**Total: {total_ms:.0f}ms**\n"
        f"**Device:** {get_device_info()} | "
        f"**Seed:** {seed} | "
        f"**ControlNet scale:** {controlnet_scale} | "
        f"**Canny:** ({canny_low}, {canny_high})"
    )

    Path("experiments/samples").mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    generated.save(f"experiments/samples/controlnet_{ts}.png")
    edge_map.save(f"experiments/samples/controlnet_{ts}_edges.png")

    return generated, edge_map, timing_info, seed


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Management
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(lora_path: str) -> str:
    """Load and apply LoRA weights to all pipelines."""
    global _current_lora_path

    lora_path = lora_path.strip()
    if not lora_path:
        return "No LoRA path provided."

    lp = Path(lora_path)
    if not lp.exists():
        return f"ERROR: LoRA path not found: {lora_path}"

    status_parts = []
    for name, pipe in _pipelines.items():
        if pipe is None:
            status_parts.append(f"{name}: not loaded")
            continue
        try:
            pipe.load_lora_weights(str(lp))
            status_parts.append(f"{name}: ✓")
        except Exception as e:
            status_parts.append(f"{name}: ERROR ({e})")

    _current_lora_path = lora_path
    return f"LoRA applied: {' | '.join(status_parts)}"


def unapply_lora() -> str:
    """Remove LoRA weights from all pipelines."""
    global _current_lora_path

    status_parts = []
    for name, pipe in _pipelines.items():
        if pipe is None:
            continue
        try:
            pipe.unload_lora_weights()
            status_parts.append(f"{name}: ✓")
        except Exception as e:
            status_parts.append(f"{name}: ERROR ({e})")

    _current_lora_path = None
    return f"LoRA unloaded: {' | '.join(status_parts)}"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

SCHEDULER_CHOICES = list(SCHEDULER_MAP.keys())


def build_ui(startup_status: str) -> gr.Blocks:
    """
    Build the Gradio Blocks UI.

    Gradio Blocks lets you arrange UI components in rows and columns,
    define multiple tabs, and wire up Python functions to UI events.

    Returns a gr.Blocks object that can be .launch()ed.
    """

    # Shared negative prompt (shown in all tabs)
    DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, deformed, ugly, bad anatomy"

    with gr.Blocks(
        title="Diffusion Image Generation",
        theme=gr.themes.Soft(),
    ) as demo:

        # ── Header ────────────────────────────────────────────
        gr.Markdown(
            "# Diffusion-Based Image Generation & Editing\n"
            "Stable Diffusion 1.5 + LoRA + ControlNet | "
            "Built with PyTorch + HuggingFace Diffusers"
        )

        # ── Status Bar ────────────────────────────────────────
        with gr.Row():
            status_text = gr.Markdown(
                f"**Status:** {startup_status}"
            )

        # ── LoRA Sidebar ──────────────────────────────────────
        with gr.Accordion("LoRA Management (optional)", open=False):
            with gr.Row():
                lora_path_input = gr.Textbox(
                    label="LoRA Adapter Path",
                    placeholder="lora_weights/final_lora_adapter",
                    info="Path to a trained LoRA adapter directory",
                )
                lora_apply_btn = gr.Button("Apply LoRA", variant="secondary")
                lora_unapply_btn = gr.Button("Unapply LoRA", variant="secondary")
            lora_status = gr.Textbox(label="LoRA Status", interactive=False)

            lora_apply_btn.click(
                fn=apply_lora,
                inputs=[lora_path_input],
                outputs=[lora_status],
            )
            lora_unapply_btn.click(
                fn=unapply_lora,
                inputs=[],
                outputs=[lora_status],
            )

        # ── Tabs ──────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Text-to-Image ──────────────────────────
            with gr.Tab("Text-to-Image"):
                gr.Markdown(
                    "Generate images from text prompts. "
                    "More steps = higher quality but slower. "
                    "Enable **Fast mode** for quick previews."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t2i_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="a majestic mountain at sunset, photorealistic, 4k",
                            lines=3,
                        )
                        t2i_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value=DEFAULT_NEGATIVE,
                            lines=2,
                        )
                        with gr.Row():
                            t2i_width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                            t2i_height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                        with gr.Row():
                            t2i_steps = gr.Slider(1, 100, value=30, step=1, label="Steps",
                                                   info="20-30 for quality; 10 for speed")
                            t2i_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5,
                                                 label="Guidance Scale (CFG)",
                                                 info="7.5 is standard; higher = stricter prompt")
                        with gr.Row():
                            t2i_seed = gr.Number(value=42, label="Seed",
                                                  info="Same seed = same image")
                            t2i_rand_seed = gr.Checkbox(label="Randomize Seed", value=False)
                        t2i_scheduler = gr.Dropdown(
                            SCHEDULER_CHOICES,
                            value="DPMSolverMultistepScheduler",
                            label="Scheduler",
                            info="DPM++ 2M = fastest good quality; DDIM = deterministic",
                        )
                        t2i_fast = gr.Checkbox(
                            label="Fast Mode (512×512, 10 steps, CFG=5)",
                            value=False,
                        )
                        t2i_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t2i_output = gr.Image(label="Generated Image", type="pil")
                        t2i_timing = gr.Markdown("*Click Generate to see timing breakdown*")
                        t2i_seed_out = gr.Number(label="Seed Used", interactive=False)

                t2i_btn.click(
                    fn=generate_txt2img,
                    inputs=[
                        t2i_prompt, t2i_neg_prompt,
                        t2i_width, t2i_height,
                        t2i_steps, t2i_cfg,
                        t2i_seed, t2i_scheduler,
                        t2i_fast, t2i_rand_seed,
                    ],
                    outputs=[t2i_output, t2i_timing, t2i_seed_out],
                )

                gr.Examples(
                    examples=[
                        ["a majestic snow-capped mountain at golden hour, photorealistic, 8k", DEFAULT_NEGATIVE, 512, 512, 30, 7.5, 42, "DPMSolverMultistepScheduler", False, False],
                        ["a cyberpunk cityscape at night, neon lights, rain reflections, ultra detailed", DEFAULT_NEGATIVE, 512, 512, 25, 8.0, 123, "EulerDiscreteScheduler", False, False],
                        ["a watercolor portrait of a cat, soft pastel colors, impressionist", DEFAULT_NEGATIVE, 512, 512, 20, 6.0, 777, "DPMSolverMultistepScheduler", False, False],
                    ],
                    inputs=[t2i_prompt, t2i_neg_prompt, t2i_width, t2i_height,
                            t2i_steps, t2i_cfg, t2i_seed, t2i_scheduler, t2i_fast, t2i_rand_seed],
                    label="Example Prompts",
                )

            # ── Tab 2: Image-to-Image ─────────────────────────
            with gr.Tab("Image-to-Image"):
                gr.Markdown(
                    "Transform an existing image guided by a text prompt. "
                    "**Strength** controls how much the original is changed: "
                    "0.3 = subtle, 0.75 = significant, 1.0 = completely replace."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        i2i_input = gr.Image(
                            label="Input Image",
                            type="pil",
                            image_mode="RGB",
                        )
                        i2i_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="a painting of this scene in Van Gogh style",
                            lines=3,
                        )
                        i2i_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value=DEFAULT_NEGATIVE,
                            lines=2,
                        )
                        i2i_strength = gr.Slider(
                            0.0, 1.0, value=0.75, step=0.05,
                            label="Strength",
                            info="0.0 = keep original, 1.0 = completely replace",
                        )
                        with gr.Row():
                            i2i_steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
                            i2i_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                        with gr.Row():
                            i2i_seed = gr.Number(value=42, label="Seed")
                            i2i_rand_seed = gr.Checkbox(label="Randomize Seed", value=False)
                        i2i_scheduler = gr.Dropdown(
                            SCHEDULER_CHOICES,
                            value="DPMSolverMultistepScheduler",
                            label="Scheduler",
                        )
                        i2i_fast = gr.Checkbox(label="Fast Mode (10 steps, CFG=5)", value=False)
                        i2i_btn = gr.Button("Transform", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        i2i_output = gr.Image(label="Transformed Image", type="pil")
                        i2i_timing = gr.Markdown("*Upload an image and click Transform*")
                        i2i_seed_out = gr.Number(label="Seed Used", interactive=False)

                i2i_btn.click(
                    fn=generate_img2img,
                    inputs=[
                        i2i_input, i2i_prompt, i2i_neg_prompt,
                        i2i_strength, i2i_steps, i2i_cfg,
                        i2i_seed, i2i_scheduler,
                        i2i_fast, i2i_rand_seed,
                    ],
                    outputs=[i2i_output, i2i_timing, i2i_seed_out],
                )

            # ── Tab 3: ControlNet (Canny) ─────────────────────
            with gr.Tab("ControlNet (Canny)"):
                gr.Markdown(
                    "Generate images that follow the **edge structure** of your input image. "
                    "The Canny edge detector extracts outlines, and ControlNet ensures the "
                    "generated image follows those outlines while matching your text prompt."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        cn_input = gr.Image(
                            label="Input Image (source for edges)",
                            type="pil",
                            image_mode="RGB",
                        )
                        cn_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="a detailed architectural drawing, technical illustration",
                            lines=3,
                        )
                        cn_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value=DEFAULT_NEGATIVE,
                            lines=2,
                        )
                        with gr.Row():
                            cn_width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                            cn_height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                        with gr.Row():
                            cn_steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
                            cn_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                        cn_controlnet_scale = gr.Slider(
                            0.1, 2.0, value=1.0, step=0.1,
                            label="ControlNet Conditioning Scale",
                            info="How strongly to follow the edge map (0.5=loose, 1.0=strict)",
                        )
                        with gr.Row():
                            cn_canny_low = gr.Slider(0, 255, value=100, step=5,
                                                      label="Canny Low Threshold")
                            cn_canny_high = gr.Slider(0, 255, value=200, step=5,
                                                       label="Canny High Threshold")
                        with gr.Row():
                            cn_seed = gr.Number(value=42, label="Seed")
                            cn_rand_seed = gr.Checkbox(label="Randomize Seed", value=False)
                        cn_scheduler = gr.Dropdown(
                            SCHEDULER_CHOICES,
                            value="DPMSolverMultistepScheduler",
                            label="Scheduler",
                        )
                        cn_fast = gr.Checkbox(label="Fast Mode (512×512, 10 steps)", value=False)
                        cn_btn = gr.Button("Generate with ControlNet", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        cn_output = gr.Image(label="Generated Image", type="pil")
                        cn_edges = gr.Image(label="Canny Edge Map (ControlNet input)", type="pil")
                        cn_timing = gr.Markdown("*Upload an image and click Generate*")
                        cn_seed_out = gr.Number(label="Seed Used", interactive=False)

                cn_btn.click(
                    fn=generate_controlnet,
                    inputs=[
                        cn_input, cn_prompt, cn_neg_prompt,
                        cn_width, cn_height,
                        cn_steps, cn_cfg,
                        cn_controlnet_scale,
                        cn_canny_low, cn_canny_high,
                        cn_seed, cn_scheduler,
                        cn_fast, cn_rand_seed,
                    ],
                    outputs=[cn_output, cn_edges, cn_timing, cn_seed_out],
                )

        # ── Footer ────────────────────────────────────────────
        gr.Markdown(
            "---\n"
            "All generated images are saved to `experiments/samples/` | "
            "Powered by [HuggingFace Diffusers](https://github.com/huggingface/diffusers)"
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run the Gradio diffusion app")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_id", type=str, default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter path to pre-load")
    parser.add_argument("--no_xformers", action="store_true", help="Disable xformers")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile on UNet")
    parser.add_argument("--share", action="store_true", help="Create public Gradio URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Skip warmup; use defaults for CPU-friendly testing")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Diffusion Image Generation App")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"ControlNet: {args.controlnet_id}")
    logger.info(f"LoRA: {args.lora_path or 'None'}")
    logger.info(f"Device: {get_device_info()}")

    # Load all pipelines at startup
    startup_status = load_pipelines(
        model_id=args.model_id,
        controlnet_id=args.controlnet_id,
        lora_path=args.lora_path,
        enable_xformers=not args.no_xformers,
        enable_torch_compile=args.torch_compile,
        smoke_test=args.smoke_test,
    )

    # Build and launch the UI
    demo = build_ui(startup_status)

    logger.info(f"Starting Gradio app on port {args.port}")
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_api=False,    # hide API docs in UI (cleaner)
        inbrowser=True,    # auto-open browser tab
    )


if __name__ == "__main__":
    main()
