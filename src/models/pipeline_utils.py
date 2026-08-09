"""
src/models/pipeline_utils.py
-----------------------------
Utilities for loading and configuring Stable Diffusion pipelines.

WHAT IS A DIFFUSERS "PIPELINE"?
A pipeline bundles together all the components needed for generation:
  - VAE (Variational Autoencoder): encodes images to latent space and decodes back
  - UNet: the denoising network (where LoRA gets applied)
  - Text Encoder: converts your prompt text to embeddings
  - Tokenizer: converts text to token IDs for the encoder
  - Scheduler: controls the denoising process (step size, noise schedule)

PIPELINES USED:
  - StableDiffusionPipeline:          text-to-image
  - StableDiffusionImg2ImgPipeline:   image-to-image
  - StableDiffusionControlNetPipeline: ControlNet conditioned generation

PERFORMANCE STACK (applied when available):
  1. fp16 precision   — halves VRAM and speeds up computation ~1.5-2x
  2. xformers         — memory-efficient attention (often 20-40% faster)
  3. torch.compile    — fuses CUDA kernels (1.1-1.5x faster, needs warmup)
  4. attention slicing — fallback for very low VRAM (< 6GB)
  5. VAE slicing      — reduces VRAM for large image decoding

SCHEDULER MAPPING:
We support several schedulers. DPM++ 2M Karras is the current best
quality/speed tradeoff for SD 1.5 at 20-30 steps.
"""

import torch
from pathlib import Path
from typing import Optional

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    # Schedulers
    DDIMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Maps human-readable scheduler names to diffusers classes
SCHEDULER_MAP = {
    "DDIMScheduler": DDIMScheduler,
    "PNDMScheduler": PNDMScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "HeunDiscreteScheduler": HeunDiscreteScheduler,
}


def get_device_and_dtype(
    device: str = "auto",
    dtype: str = "auto",
) -> tuple[torch.device, torch.dtype]:
    """
    Determine the optimal device and dtype for inference.

    Args:
        device: "auto" (detect) | "cuda" | "cpu" | "mps" (Apple Silicon)
        dtype: "auto" (fp16 on GPU, fp32 on CPU) | "fp16" | "fp32" | "bf16"

    Returns:
        (device, dtype) tuple ready to pass to pipeline.to()
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    resolved_device = torch.device(device)

    if dtype == "auto":
        # fp16 on CUDA/MPS (faster + less VRAM), fp32 on CPU (fp16 unsupported)
        if resolved_device.type == "cuda":
            resolved_dtype = torch.float16
        elif resolved_device.type == "mps":
            resolved_dtype = torch.float16
        else:
            resolved_dtype = torch.float32
    else:
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(dtype.lower(), torch.float32)

    logger.info(f"Device: {resolved_device} | Dtype: {resolved_dtype}")
    return resolved_device, resolved_dtype


def apply_optimizations(
    pipe,
    device: torch.device,
    enable_xformers: bool = True,
    enable_torch_compile: bool = False,
    attention_slicing: bool = False,
    vae_slicing: bool = True,
    compile_mode: str = "reduce-overhead",
) -> None:
    """
    Apply in-place performance optimizations to a pipeline.

    This modifies `pipe` directly (no return value needed).

    Args:
        pipe: Any diffusers pipeline.
        device: The device the pipeline is on.
        enable_xformers: Use xformers memory-efficient attention.
        enable_torch_compile: Compile the UNet with torch.compile.
        attention_slicing: Slice attention heads to save VRAM (slower).
        vae_slicing: Slice VAE decode to save VRAM.
        compile_mode: torch.compile mode string.
    """
    # xformers: replaces standard attention with a CUDA-fused implementation
    # Typically saves 20-40% VRAM and speeds up by 10-30%.
    # Requires: pip install xformers (version must match your torch version)
    if enable_xformers and device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
        except Exception as e:
            logger.warning(f"xformers not available: {e}. Using standard attention.")

    # attention slicing: processes attention in chunks to fit low-VRAM GPUs
    # SLOWER than xformers — use only if xformers isn't available and VRAM < 6GB
    if attention_slicing:
        pipe.enable_attention_slicing()
        logger.info("Attention slicing enabled (low-VRAM mode)")

    # VAE slicing: decodes image rows one at a time instead of all at once
    # Useful when generating large images (768x768+) on < 8GB VRAM
    if vae_slicing:
        pipe.enable_vae_slicing()
        logger.info("VAE slicing enabled")

    # torch.compile: JIT-compiles the UNet into optimized CUDA kernels
    # - First inference is slow (compile time: 30-120 seconds)
    # - Subsequent inferences are 10-40% faster
    # - Only effective on Linux + CUDA; skip on macOS
    if enable_torch_compile and device.type == "cuda":
        import platform
        if platform.system() == "Linux":
            try:
                pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=False)
                logger.info(f"UNet compiled with torch.compile (mode={compile_mode})")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        else:
            logger.warning("torch.compile skipped (only supported on Linux with CUDA)")


def set_scheduler(pipe, scheduler_name: str) -> None:
    """
    Replace the pipeline's scheduler with the named one.

    Schedulers define HOW denoising happens at each step.
    They share config (from the base model) but differ in algorithm.

    Args:
        pipe: Diffusers pipeline.
        scheduler_name: Key from SCHEDULER_MAP (e.g. "DPMSolverMultistepScheduler").
    """
    if scheduler_name not in SCHEDULER_MAP:
        logger.warning(
            f"Unknown scheduler '{scheduler_name}'. "
            f"Available: {list(SCHEDULER_MAP.keys())}. Using current scheduler."
        )
        return

    scheduler_cls = SCHEDULER_MAP[scheduler_name]
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    logger.debug(f"Scheduler set to: {scheduler_name}")


def load_lora_into_pipeline(
    pipe,
    lora_weights_path: str | Path,
    lora_scale: float = 1.0,
) -> None:
    """
    Attach LoRA weights to a pipeline, handling both on-disk formats.

    WHY THIS ISN'T JUST pipe.load_lora_weights():
      Our trainer wraps the UNet with PEFT's get_peft_model() and saves via
      save_pretrained(), which writes:
          adapter_config.json
          adapter_model.safetensors
      Diffusers' load_lora_weights() instead looks for
      "pytorch_lora_weights.safetensors" by default and expects diffusers-
      style key names, so pointing it at a PEFT adapter directory fails.

    So: detect the format and take the matching path.
      - PEFT adapter  -> PeftModel.from_pretrained() then merge_and_unload()
      - diffusers/kohya -> pipe.load_lora_weights()

    WHY merge_and_unload() FOR THE PEFT PATH:
      It folds ΔW = B·A back into the base weights and returns a plain
      UNet2DConditionModel. That keeps the pipeline free of a PeftModel
      wrapper (which confuses code reaching for pipe.unet attributes) and
      removes the extra per-layer matmuls, so inference is marginally faster.
      The trade-off is you can no longer toggle the adapter off -- fine here,
      since comparing base vs LoRA means loading two separate pipelines.

    WHAT lora_scale DOES:
      A LoRA contributes ΔW = (alpha/rank) · B·A to each adapted weight.
      lora_scale multiplies that contribution before it is merged, so:
          1.0 = full strength, exactly as trained (the default)
          0.7 = 70% of the learned delta
          0.0 = base model, adapter has no effect
      This is NOT a substitute for training properly, but a style adapter
      trained to convergence is often too strong at 1.0: it imposes the
      dataset's palette and brushwork so hard that structural detail
      dissolves. Dialling it to 0.6-0.8 usually keeps the style while
      restoring composition. Cheap to sweep, so sweep it before retraining.

    Args:
        pipe: Any diffusers pipeline with a .unet attribute.
        lora_weights_path: Directory holding the adapter.
        lora_scale: Multiplier on the adapter's contribution, 0.0-1.0+.
    """
    lora_path = Path(lora_weights_path)
    is_peft_adapter = (lora_path / "adapter_config.json").exists()

    if is_peft_adapter:
        from peft import PeftModel

        logger.info(f"Loading PEFT adapter from: {lora_path} (scale={lora_scale})")
        peft_unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))

        if lora_scale != 1.0:
            # PEFT stores the alpha/rank factor per adapter on each injected
            # layer. Scaling it here means merge_and_unload() folds in the
            # already-attenuated delta, so there is no runtime cost.
            n_scaled = 0
            for module in peft_unet.modules():
                if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                    for adapter_name in module.scaling:
                        module.scaling[adapter_name] *= lora_scale
                        n_scaled += 1
            logger.info(f"Scaled {n_scaled} LoRA layers by {lora_scale}")

        pipe.unet = peft_unet.merge_and_unload()
        logger.info("PEFT adapter merged into UNet")
    else:
        logger.info(f"Loading diffusers-format LoRA from: {lora_path}")
        pipe.load_lora_weights(str(lora_path))
        if lora_scale != 1.0:
            # Diffusers-format adapters stay unmerged, so strength is applied
            # per-call via cross_attention_kwargs rather than baked in.
            pipe.set_adapters(["default_0"], adapter_weights=[lora_scale])
            logger.info(f"Set diffusers adapter weight to {lora_scale}")


def load_txt2img_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_weights_path: Optional[str] = None,
    lora_scale: float = 1.0,
    scheduler_name: str = "DPMSolverMultistepScheduler",
    device: str = "auto",
    dtype: str = "auto",
    enable_xformers: bool = True,
    enable_torch_compile: bool = False,
    vae_slicing: bool = True,
) -> StableDiffusionPipeline:
    """
    Load and optimize a text-to-image pipeline.

    Args:
        model_id: HuggingFace model ID or local path.
        lora_weights_path: Path to LoRA adapter directory (optional).
        scheduler_name: Which scheduler to use.
        device: Target device.
        dtype: Tensor precision.
        enable_xformers: Enable xformers attention.
        enable_torch_compile: Compile UNet with torch.compile.
        vae_slicing: Enable VAE slice-decoding.

    Returns:
        Configured StableDiffusionPipeline ready for inference.
    """
    resolved_device, resolved_dtype = get_device_and_dtype(device, dtype)

    logger.info(f"Loading txt2img pipeline: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=resolved_dtype,
        safety_checker=None,       # disable NSFW filter (speeds up inference)
        requires_safety_checker=False,
    )
    pipe = pipe.to(resolved_device)

    # Load LoRA weights if provided
    if lora_weights_path and Path(lora_weights_path).exists():
        load_lora_into_pipeline(pipe, lora_weights_path, lora_scale=lora_scale)

    set_scheduler(pipe, scheduler_name)
    apply_optimizations(
        pipe, resolved_device,
        enable_xformers=enable_xformers,
        enable_torch_compile=enable_torch_compile,
        vae_slicing=vae_slicing,
    )

    logger.info("txt2img pipeline ready")
    return pipe


def load_img2img_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_weights_path: Optional[str] = None,
    lora_scale: float = 1.0,
    scheduler_name: str = "DPMSolverMultistepScheduler",
    device: str = "auto",
    dtype: str = "auto",
    enable_xformers: bool = True,
    vae_slicing: bool = True,
) -> StableDiffusionImg2ImgPipeline:
    """
    Load and optimize an image-to-image pipeline.

    img2img takes an existing image + prompt and "re-imagines" it.
    The `strength` parameter (0-1) controls how much to deviate from the original.
    """
    resolved_device, resolved_dtype = get_device_and_dtype(device, dtype)

    logger.info(f"Loading img2img pipeline: {model_id}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=resolved_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(resolved_device)

    if lora_weights_path and Path(lora_weights_path).exists():
        load_lora_into_pipeline(pipe, lora_weights_path, lora_scale=lora_scale)

    set_scheduler(pipe, scheduler_name)
    apply_optimizations(pipe, resolved_device, enable_xformers=enable_xformers, vae_slicing=vae_slicing)

    logger.info("img2img pipeline ready")
    return pipe


def load_controlnet_pipeline(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny",
    lora_weights_path: Optional[str] = None,
    lora_scale: float = 1.0,
    scheduler_name: str = "DPMSolverMultistepScheduler",
    device: str = "auto",
    dtype: str = "auto",
    enable_xformers: bool = True,
    vae_slicing: bool = True,
) -> StableDiffusionControlNetPipeline:
    """
    Load and optimize a ControlNet-conditioned pipeline.

    ControlNet adds an additional conditioning signal to generation.
    For canny ControlNet: you provide an edge map (from Canny edge detection)
    and the generated image will respect those edges.

    The ControlNet model is a copy of the UNet encoder with trainable
    "zero-convolution" connections to the base UNet decoder. It runs in
    parallel with the base UNet and feeds structural guidance.

    Args:
        base_model_id: SD base model.
        controlnet_model_id: ControlNet checkpoint (e.g., canny).
        lora_weights_path: Optional LoRA adapter.
        scheduler_name: Denoising scheduler.
        device: Compute device.
        dtype: Tensor precision.
        enable_xformers: Enable xformers.
        vae_slicing: Enable VAE slicing.

    Returns:
        Configured StableDiffusionControlNetPipeline.
    """
    resolved_device, resolved_dtype = get_device_and_dtype(device, dtype)

    logger.info(f"Loading ControlNet model: {controlnet_model_id}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=resolved_dtype,
    )

    logger.info(f"Loading ControlNet pipeline: {base_model_id}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=resolved_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(resolved_device)

    if lora_weights_path and Path(lora_weights_path).exists():
        load_lora_into_pipeline(pipe, lora_weights_path, lora_scale=lora_scale)

    set_scheduler(pipe, scheduler_name)
    apply_optimizations(pipe, resolved_device, enable_xformers=enable_xformers, vae_slicing=vae_slicing)

    logger.info("ControlNet pipeline ready")
    return pipe


def get_canny_edge_map(
    image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> "Image.Image":
    """
    Compute Canny edge detection map from an image.

    HOW CANNY WORKS:
      1. Gaussian blur to reduce noise
      2. Gradient magnitude and direction (Sobel filters)
      3. Non-maximum suppression (thin edges to 1 pixel wide)
      4. Hysteresis thresholding:
         - strong edges: gradient > high_threshold → definitely edge
         - weak edges: low < gradient < high → edge if connected to strong edge
         - non-edges: gradient < low_threshold → not edge

    WHY CANNY FOR CONTROLNET?
    Canny edges are clean, consistent, and scale-invariant.
    The ControlNet model was trained on Canny edge maps, so it
    specifically understands this representation to guide generation.

    Args:
        image: PIL Image or numpy array (RGB).
        low_threshold: Lower hysteresis threshold (0-255).
        high_threshold: Upper hysteresis threshold (0-255).

    Returns:
        PIL Image with edge map (3-channel RGB, edges are white on black).
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    if hasattr(image, "numpy") or isinstance(image, PILImage.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # Convert to grayscale for edge detection
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Convert back to RGB (ControlNet expects 3-channel input)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)

    return PILImage.fromarray(edges_rgb)
