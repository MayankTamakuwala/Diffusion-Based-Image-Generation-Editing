"""
src/training/train_lora.py
--------------------------
LoRA Fine-tuning for Stable Diffusion 1.5

HOW STABLE DIFFUSION TRAINING WORKS:
  At each training step:
  1. Load a real image x₀
  2. Encode it to latent space: z₀ = VAE.encode(x₀)
  3. Sample random timestep t ∈ [1, T] (T=1000 for SD)
  4. Add noise to get noisy latent: zₜ = √ᾱₜ·z₀ + √(1-ᾱₜ)·ε,  ε ~ N(0,I)
  5. Feed zₜ, t, and text embeddings to UNet → predicted noise ε̂
  6. Loss = MSE(ε̂, ε)  [we're teaching UNet to predict the noise we added]
  7. Backprop only through LoRA parameters (all other weights frozen)

WHY THIS LOSS?
  This is the "simplified DDPM objective" from Ho et al. 2020.
  We're not training the model to directly predict images — we're training
  it to remove noise. This is more stable and converges faster.

HOW LoRA CHANGES THIS:
  Without LoRA: update all 860M UNet parameters
  With LoRA: freeze all 860M; add small A×B matrices with rank r
    - Each LoRA layer adds: ΔW = B×A (B∈R^{m×r}, A∈R^{r×n})
    - r=16: ~2M extra parameters vs 860M base = only 0.23% of model size
    - Training only those 2M parameters is ~10x cheaper

ACCELERATE:
  We use HuggingFace Accelerate which handles:
  - Multi-GPU training (gradient sync)
  - Mixed precision (fp16) loss scaling
  - Gradient accumulation
  You can run without Accelerate too (single GPU), but it handles edge cases.

Usage:
  # Full run
  python src/training/train_lora.py --config config/train_config.yaml

  # Smoke test (tiny, fast)
  python src/training/train_lora.py --config config/train_config.yaml --smoke_test

  # With W&B logging
  python src/training/train_lora.py --config config/train_config.yaml --report_to wandb
"""

import argparse
import itertools
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Running this file directly puts its own directory on sys.path, not the repo
# root, so "from src...." would fail. Add the repo root before any src import.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.data.dataset import get_dataloader, get_wikiart_dataloader
from src.utils.logging_utils import get_logger, setup_logging, get_timestamped_filename
from src.utils.seed_utils import seed_everything, seed_generator

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Stable Diffusion")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a quick smoke test with minimal steps (verifies code works)",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "none"],
        help="Override config's report_to setting",
    )
    return parser.parse_args()


def make_validation_images(
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    device: torch.device,
    dtype: torch.dtype,
    prompt: str,
    num_images: int = 4,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> list[Image.Image]:
    """
    Generate validation images using the current UNet state (with LoRA).

    We do a mini inference loop here (instead of loading a full pipeline)
    to avoid the overhead of re-loading models during training.

    This function is called every N training steps to visually monitor progress.
    """
    # Tokenize and encode the validation prompt
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # Encode prompt
        text_embeds = text_encoder(tokens.input_ids)[0]

        # Encode unconditional (empty prompt) for classifier-free guidance
        uncond_tokens = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        uncond_embeds = text_encoder(uncond_tokens.input_ids)[0]

        # Concatenate for batched guidance
        text_embeds_batch = torch.cat([uncond_embeds] * num_images + [text_embeds] * num_images)

        # Start from pure noise latent (512x512 image → 64x64 latent, 4 channels)
        latent_h, latent_w = 64, 64  # 512 / 8 = 64
        generator = seed_generator(seed)
        latents = torch.randn(
            (num_images, 4, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # Set up scheduler for inference
        noise_scheduler.set_timesteps(num_steps)
        latents = latents * noise_scheduler.init_noise_sigma

        # Denoising loop
        for t in noise_scheduler.timesteps:
            # Double the batch for classifier-free guidance
            latent_input = torch.cat([latents] * 2)
            latent_input = noise_scheduler.scale_model_input(latent_input, t)

            # Predict noise
            noise_pred = unet(
                latent_input,
                t,
                encoder_hidden_states=text_embeds_batch,
            ).sample

            # Apply classifier-free guidance (CFG):
            # output = uncond + scale * (cond - uncond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Step the scheduler
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to pixel space
        latents = latents / vae.config.scaling_factor
        images_tensor = vae.decode(latents).sample

    # Convert from [-1, 1] tensor to PIL Images
    images_tensor = (images_tensor / 2 + 0.5).clamp(0, 1)
    images_pil = []
    for img_t in images_tensor:
        img_np = img_t.cpu().permute(1, 2, 0).float().numpy()
        img_np = (img_np * 255).round().astype("uint8")
        images_pil.append(Image.fromarray(img_np))

    return images_pil


def flatten_config_for_tracker(cfg: dict, prefix: str = "") -> dict:
    """
    Flatten a nested config into scalars that experiment trackers accept.

    WHY THIS IS NEEDED:
    TensorBoard's add_hparams() only accepts int, float, str, bool, or Tensor.
    Our config is nested and contains None (e.g. max_train_steps: null) and
    lists (e.g. lora.target_modules), both of which raise:
        ValueError: value should be one of int, float, str, bool, or torch.Tensor

    So we flatten "training.learning_rate" -> one key, stringify lists and
    None, and drop anything else that isn't representable.
    """
    flat = {}
    for key, val in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(val, dict):
            flat.update(flatten_config_for_tracker(val, prefix=f"{name}."))
        elif isinstance(val, (list, tuple)):
            flat[name] = ", ".join(str(v) for v in val)
        elif val is None:
            flat[name] = "null"
        elif isinstance(val, (int, float, str, bool)):
            flat[name] = val
        else:
            flat[name] = str(val)
    return flat


def train(config: OmegaConf, smoke_test: bool = False) -> None:
    """
    Main training function.

    Args:
        config: OmegaConf config object from train_config.yaml
        smoke_test: If True, use minimal steps for quick verification.
    """
    # Apply smoke test overrides if requested
    if smoke_test:
        logger.info("SMOKE TEST MODE: using minimal training steps")
        for key, val in config.smoke_test.items():
            # Navigate nested config path
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                current = current[part]
        # Apply all smoke test overrides to relevant config sections
        config.dataset.resolution = config.smoke_test.resolution
        config.training.num_train_epochs = config.smoke_test.num_train_epochs
        config.training.max_train_steps = config.smoke_test.max_train_steps
        config.training.train_batch_size = config.smoke_test.train_batch_size
        config.training.gradient_accumulation_steps = config.smoke_test.gradient_accumulation_steps
        config.checkpointing.save_steps = config.smoke_test.save_steps
        config.validation.validation_steps = config.smoke_test.validation_steps

    # ── Setup Accelerator ──────────────────────────────────────────────────
    # Accelerate handles: mixed precision, gradient accumulation, device placement,
    # distributed training. We configure it here once.
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(config.logging.logging_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    report_to = config.logging.report_to
    accelerator_project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(logs_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=report_to if report_to != "none" else None,
        project_config=accelerator_project_config,
    )

    setup_logging(
        log_level="DEBUG" if smoke_test else "INFO",
        log_file=logs_dir / "train.log",
    )

    # Seeding
    seed_everything(config.training.seed)
    logger.info(f"Seed: {config.training.seed}")
    logger.info(f"Device: {accelerator.device} | Num processes: {accelerator.num_processes}")

    # ── Load Pretrained Models ─────────────────────────────────────────────
    model_id = config.model.pretrained_model_name_or_path
    logger.info(f"Loading models from: {model_id}")

    # Load each component separately for maximum control
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Noise scheduler for TRAINING (DDPMScheduler = forward diffusion process)
    # This is different from the inference scheduler (which controls denoising speed)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # ── Freeze Base Models ─────────────────────────────────────────────────
    # We ONLY train the LoRA parameters, so freeze everything else.
    # grad_checkpointing can re-compute activations during backward pass
    # to save VRAM (costs ~20% speed but saves ~30% VRAM).
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if config.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ── Inject LoRA into UNet ──────────────────────────────────────────────
    # PEFT's LoraConfig tells it WHERE to inject LoRA (which modules)
    # and HOW MUCH capacity to add (rank, alpha, dropout).
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.lora_alpha,
        target_modules=list(config.lora.target_modules),
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
    )

    # get_peft_model wraps the UNet and ONLY makes LoRA parameters trainable
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()  # Shows: e.g., "Trainable params: 2,097,152 (0.24%)"

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Only pass LoRA parameters to the optimizer (frozen params waste memory)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    logger.info(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    # ── DataLoader ─────────────────────────────────────────────────────────
    # Branch on whether we're using the local folder layout or HuggingFace
    # datasets (WikiArt). Both paths produce identical DataLoader outputs
    # so the training loop below doesn't need to change at all.
    use_hf = getattr(config, "hf_dataset", None) and config.hf_dataset.get("enabled", False)

    if use_hf:
        logger.info(
            f"Using HuggingFace dataset: {config.hf_dataset.name} "
            f"(style={config.hf_dataset.get('style_filter', 'all')})"
        )
        max_samples = (
            config.smoke_test.get("hf_max_samples", 20) if smoke_test else None
        )
        train_dataloader = get_wikiart_dataloader(
            style_filter=config.hf_dataset.get("style_filter", "Impressionism"),
            split="train",
            val_fraction=config.hf_dataset.get("val_fraction", 0.05),
            batch_size=config.training.train_batch_size,
            resolution=config.dataset.resolution,
            center_crop=config.dataset.center_crop,
            random_flip=config.dataset.random_flip,
            tokenizer=tokenizer,
            num_workers=min(config.dataset.get("num_workers", 2), os.cpu_count() or 1),
            max_samples=max_samples,
        )
    else:
        train_dataloader = get_dataloader(
            data_dir=config.dataset.train_data_dir,
            batch_size=config.training.train_batch_size,
            resolution=config.dataset.resolution,
            center_crop=config.dataset.center_crop,
            random_flip=config.dataset.random_flip,
            fallback_caption=config.dataset.fallback_caption,
            tokenizer=tokenizer,
            num_workers=min(config.dataset.get("num_workers", 2), os.cpu_count() or 1),
            shuffle=True,
        )

    # ── Learning Rate Scheduler ────────────────────────────────────────────
    # WHY LR SCHEDULING?
    # A cosine schedule starts at learning_rate, warms up for warmup_steps
    # (to avoid large initial gradient updates), then gradually decays to ~0.
    # This allows coarse learning early and fine-grained learning later.
    num_epochs = config.training.num_train_epochs
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.training.gradient_accumulation_steps
    )
    if config.training.max_train_steps:
        max_train_steps = config.training.max_train_steps
    else:
        max_train_steps = num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.lr_warmup_steps * config.training.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.training.gradient_accumulation_steps,
    )

    # ── Prepare with Accelerator ───────────────────────────────────────────
    # Accelerator wraps everything for distributed training / mixed precision
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Move frozen models to device manually (Accelerator only handles trainable)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae = vae.to(accelerator.device, dtype=weight_dtype)

    # ── Init Trackers ──────────────────────────────────────────────────────
    if accelerator.is_main_process:
        accelerator.init_trackers(
            config.logging.get("wandb_project", "diffusion-lora"),
            config=flatten_config_for_tracker(
                OmegaConf.to_container(config, resolve=True)
            ),
        )

    # ── Training Loop ──────────────────────────────────────────────────────
    global_step = 0
    first_epoch = 0
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Training steps",
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    logger.info(f"Starting training: {max_train_steps} total steps")
    logger.info(f"  Dataset: {len(train_dataloader.dataset)} images")
    logger.info(f"  Effective batch size: {config.training.train_batch_size * config.training.gradient_accumulation_steps * accelerator.num_processes}")
    logger.info(f"  LoRA rank: {config.lora.rank} | Alpha: {config.lora.lora_alpha}")

    for epoch in range(first_epoch, num_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # ── Forward Pass ──────────────────────────────────
            with accelerator.accumulate(unet):
                # 1. Encode images to latent space using VAE
                #    VAE encoder compresses 3×512×512 → 4×64×64 latents
                #    We don't train the VAE, so use no_grad + detach
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                # Scale factor: SD's VAE latents are scaled by a constant
                # (this was empirically determined during VAE training)
                latents = latents * vae.config.scaling_factor

                # 2. Sample random noise ε ~ N(0, I)
                noise = torch.randn_like(latents)

                # 3. Sample random timesteps t for each image in batch
                #    Each image gets a different timestep (curriculum)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device, dtype=torch.long
                )

                # 4. Forward diffusion: add noise to latents (q(zₜ|z₀))
                #    The amount of noise depends on timestep t
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 5. Encode text prompts to embeddings
                #    Shape: (batch, seq_len=77, embed_dim=768)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # 6. Determine prediction target
                #    SD 1.5 predicts noise ("epsilon" objective)
                #    Some models predict "v" — check noise_scheduler.config
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                # 7. UNet forward pass: predict noise at each timestep
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # 8. Compute MSE loss between predicted and actual noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # 9. Backpropagation (only LoRA params get gradients)
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps

                accelerator.backward(loss)

                # 10. Gradient clipping (prevents exploding gradients)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config.training.max_grad_norm)

                # 11. Optimizer step (updates LoRA parameters)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── After Each Sync Step ───────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log metrics
                if global_step % config.logging.log_interval == 0:
                    logs = {
                        "train_loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(**{k: f"{v:.4f}" for k, v in logs.items()})
                    accelerator.log(logs, step=global_step)
                    train_loss = 0.0

                # Checkpoint
                if global_step % config.checkpointing.save_steps == 0:
                    if accelerator.is_main_process:
                        ckpt_dir = output_dir / f"checkpoint-{global_step}"
                        _save_checkpoint(accelerator, unet, ckpt_dir, config)
                        logger.info(f"Checkpoint saved: {ckpt_dir}")

                        # Cleanup old checkpoints
                        _cleanup_checkpoints(output_dir, config.checkpointing.checkpoints_total_limit)

                # Validation images
                if global_step % config.validation.validation_steps == 0:
                    if accelerator.is_main_process and config.validation.validation_prompt:
                        logger.info("Generating validation images...")
                        _run_validation(
                            accelerator, unet, vae, text_encoder, tokenizer,
                            noise_scheduler, config, global_step, output_dir
                        )

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    # ── Final Save ────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Unwrap the PEFT model to save just the LoRA adapter
        unwrapped_unet = accelerator.unwrap_model(unet)
        final_save_path = output_dir / "final_lora_adapter"
        unwrapped_unet.save_pretrained(final_save_path)
        logger.info(f"Final LoRA adapter saved to: {final_save_path}")

    accelerator.end_training()
    logger.info("Training complete!")


def _save_checkpoint(accelerator, unet, checkpoint_dir: Path, config) -> None:
    """Save LoRA adapter weights and training state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(unet)
    unwrapped.save_pretrained(checkpoint_dir)


def _cleanup_checkpoints(output_dir: Path, keep_n: int) -> None:
    """Keep only the N most recent checkpoints to save disk space."""
    checkpoint_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if len(checkpoint_dirs) > keep_n:
        for old_ckpt in checkpoint_dirs[:-keep_n]:
            shutil.rmtree(old_ckpt)
            logger.info(f"Removed old checkpoint: {old_ckpt}")


def _run_validation(
    accelerator, unet, vae, text_encoder, tokenizer,
    noise_scheduler, config, step: int, output_dir: Path
) -> None:
    """Generate and save validation images at the current training step."""
    from diffusers import EulerDiscreteScheduler
    val_scheduler = EulerDiscreteScheduler.from_config(noise_scheduler.config)

    unet.eval()
    images = make_validation_images(
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=val_scheduler,
        device=accelerator.device,
        dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32,
        prompt=config.validation.validation_prompt,
        num_images=config.validation.num_validation_images,
        num_steps=20,
        seed=config.training.seed,
    )
    unet.train()

    val_dir = output_dir / "validation_images"
    val_dir.mkdir(exist_ok=True)
    for i, img in enumerate(images):
        img.save(val_dir / f"step_{step:06d}_img_{i}.png")
    logger.info(f"Saved {len(images)} validation images to {val_dir}")


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    if args.report_to:
        config.logging.report_to = args.report_to

    logger.info(f"Config loaded from: {args.config}")
    logger.info(f"Training with LoRA rank={config.lora.rank} on {config.model.pretrained_model_name_or_path}")

    train(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
