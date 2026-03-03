"""
src/data/dataset.py
-------------------
PyTorch Dataset for loading image-caption pairs for LoRA fine-tuning.

ARCHITECTURE OVERVIEW:
  dataset/
    train/
      images/  <- PNG or JPEG files
      captions/  <- .txt files with same stem as image
    val/
      images/
      captions/

For each image "dog_01.png", we look for "dog_01.txt" in captions/.
If no caption file exists, we fall back to a default caption.

WHY NOT LOAD EVERYTHING INTO RAM?
With 5k+ images at 512x512, that's ~5000 * 512 * 512 * 3 ≈ 3.9 GB uncompressed.
PyTorch Dataset lazy-loads: only one batch (~16 images) lives in memory at once.
The DataLoader handles background prefetching with multiple worker processes.

NORMALIZATION: [-1, 1] not [0, 1]
Stable Diffusion's VAE was trained with pixel values in [-1, 1].
We must match this convention, otherwise training diverges.
Formula: x_normalized = (x_raw / 127.5) - 1.0
"""

import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageCaptionDataset(Dataset):
    """
    A PyTorch Dataset that reads (image, caption) pairs from a directory.

    Each __getitem__ call:
      1. Loads one image from disk (lazy — only when the DataLoader asks)
      2. Applies transforms (resize, crop, flip, normalize)
      3. Loads the corresponding caption text
      4. Returns (pixel_values_tensor, caption_string)

    Args:
        data_dir: Root directory containing 'images/' and optionally 'captions/' subdirs.
        resolution: Target image size (e.g., 512 for SD 1.5).
        center_crop: If True, center-crop to resolution. If False, random crop.
        random_flip: If True, randomly flip horizontally (data augmentation).
        fallback_caption: Caption to use when no .txt file exists.
        tokenizer: Optional HF tokenizer. If provided, returns tokenized IDs too.
    """

    def __init__(
        self,
        data_dir: str | Path,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        fallback_caption: str = "a high quality photo",
        tokenizer=None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.fallback_caption = fallback_caption
        self.tokenizer = tokenizer

        # Discover all image files
        images_dir = self.data_dir / "images"
        captions_dir = self.data_dir / "captions"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Expected images directory at: {images_dir}\n"
                f"Please structure your dataset as:\n"
                f"  {data_dir}/images/*.png\n"
                f"  {data_dir}/captions/*.txt  (optional)"
            )

        self.image_paths = sorted([
            p for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {images_dir}. "
                f"Supported formats: {IMAGE_EXTENSIONS}"
            )

        self.captions_dir = captions_dir if captions_dir.exists() else None

        logger.info(
            f"Dataset loaded: {len(self.image_paths)} images from {images_dir}"
        )
        if self.captions_dir is None:
            logger.warning(
                f"No captions/ directory found. "
                f"Using fallback caption: '{self.fallback_caption}'"
            )

        # Build image transforms
        # These run on the CPU in the DataLoader worker processes
        transform_list = []

        # Step 1: Resize to slightly larger than target (for random crop)
        if center_crop:
            transform_list.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR))
            transform_list.append(transforms.CenterCrop(resolution))
        else:
            # Resize long edge, then random crop — more variety in training
            transform_list.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR))
            transform_list.append(transforms.RandomCrop(resolution))

        # Step 2: Optional horizontal flip (data augmentation)
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        # Step 3: Convert PIL Image to PyTorch tensor (0.0 to 1.0 range)
        transform_list.append(transforms.ToTensor())

        # Step 4: Normalize from [0, 1] to [-1, 1]
        # SD VAE expects this range. Mean=0.5, Std=0.5 per channel achieves it.
        transform_list.append(transforms.Normalize([0.5], [0.5]))

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        """Return number of samples. Required by DataLoader."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return one training sample.

        Returns a dict with:
          - "pixel_values": torch.Tensor of shape (3, H, W), range [-1, 1]
          - "input_ids": tokenized caption (only if tokenizer was provided)
          - "caption": raw caption string (always included)
        """
        image_path = self.image_paths[idx]

        # ── Load Image ────────────────────────────────────────
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}. Using blank image.")
            image = Image.new("RGB", (self.resolution, self.resolution), color=128)

        pixel_values = self.transform(image)

        # ── Load Caption ──────────────────────────────────────
        caption = self._load_caption(image_path)

        result = {
            "pixel_values": pixel_values,
            "caption": caption,
        }

        # ── Tokenize (optional) ───────────────────────────────
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens.input_ids.squeeze(0)

        return result

    def _load_caption(self, image_path: Path) -> str:
        """
        Load caption for an image. Falls back gracefully if not found.

        Looks for: captions/<stem>.txt
        Example: images/dog_01.png → captions/dog_01.txt
        """
        if self.captions_dir is None:
            return self.fallback_caption

        caption_path = self.captions_dir / (image_path.stem + ".txt")
        if caption_path.exists():
            try:
                text = caption_path.read_text(encoding="utf-8").strip()
                return text if text else self.fallback_caption
            except Exception as e:
                logger.debug(f"Could not read caption {caption_path}: {e}")

        return self.fallback_caption


def get_dataloader(
    data_dir: str | Path,
    batch_size: int = 4,
    resolution: int = 512,
    center_crop: bool = True,
    random_flip: bool = True,
    fallback_caption: str = "a high quality photo",
    tokenizer=None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Convenience function: create dataset + dataloader in one call.

    WHY num_workers=4?
    Image loading (disk I/O + JPEG decode + transforms) is slow.
    Workers run in parallel on separate CPU cores, pre-fetching batches
    while the GPU trains on the previous batch. This hides I/O latency.
    Rule of thumb: use 2-4 workers per GPU; don't exceed CPU core count.

    Args:
        data_dir: Path to dataset split directory (e.g., "dataset/train").
        batch_size: Number of images per batch.
        resolution: Target image resolution.
        center_crop: Center vs. random crop.
        random_flip: Horizontal flip augmentation.
        fallback_caption: Default caption if no .txt file found.
        tokenizer: HF tokenizer for automatic tokenization.
        num_workers: Parallel data loading workers.
        shuffle: Shuffle order each epoch (True for train, False for val).

    Returns:
        Configured DataLoader ready for training.
    """
    dataset = ImageCaptionDataset(
        data_dir=data_dir,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
        fallback_caption=fallback_caption,
        tokenizer=tokenizer,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,   # faster CPU→GPU transfer (pre-pins host memory)
        drop_last=True,    # drop incomplete last batch (avoids batch norm issues)
        persistent_workers=num_workers > 0,  # keep workers alive between epochs
    )

    logger.info(
        f"DataLoader: {len(dataset)} samples, {len(dataloader)} batches "
        f"(batch_size={batch_size}, workers={num_workers})"
    )
    return dataloader
