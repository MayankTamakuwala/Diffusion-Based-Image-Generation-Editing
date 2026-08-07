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


# ─────────────────────────────────────────────────────────────────────────────
# WikiArt HuggingFace Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WikiArtHFDataset(Dataset):
    """
    PyTorch Dataset that loads WikiArt from HuggingFace hub (huggan/wikiart).

    WHY A SEPARATE CLASS INSTEAD OF REUSING ImageCaptionDataset?
    The HF dataset returns PIL images and integer ClassLabel IDs in memory,
    whereas ImageCaptionDataset reads from a local file-system tree.
    The two sources need different __getitem__ logic, but share the same
    transform pipeline -- so we duplicate only what differs.

    CAPTION CONSTRUCTION:
    WikiArt has no free-text captions. We synthesise them from metadata:
        "an Impressionism painting, landscape, by Claude Monet"
    This gives the text-encoder real semantic signal and lets you use
    natural-language prompts at inference time, e.g.:
        "an impressionism painting of a river at dusk, visible brushstrokes"

    HOW THE HF DATASET IS LOADED:
    1. load_dataset("huggan/wikiart") downloads and caches ~7GB the first time.
    2. Subsequent runs are instant -- HF caches the Arrow files.
    3. We filter by style AFTER loading (filter() is cached too).
    4. We split train/val with a fixed seed for reproducibility.

    Args:
        style_filter: e.g. "Impressionism". None = use all ~80k images.
        split: "train" or "val".
        val_fraction: Fraction of data reserved for validation (default 5%).
        resolution: Target square crop size.
        center_crop: True = center crop; False = random crop.
        random_flip: Horizontal flip augmentation.
        tokenizer: If provided, also returns tokenised input_ids.
        max_samples: Cap dataset size (useful for smoke tests).
    """

    def __init__(
        self,
        style_filter: str | None = "Impressionism",
        split: str = "train",
        val_fraction: float = 0.05,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        tokenizer=None,
        max_samples: int | None = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. Run: pip install datasets"
            )

        self.resolution = resolution
        self.tokenizer = tokenizer

        logger.info(
            "Loading huggan/wikiart from HuggingFace hub "
            "(downloads ~7GB to HF cache on first run, instant thereafter)..."
        )
        raw = load_dataset("huggan/wikiart", split="train")

        # HF ClassLabel stores integer IDs; .names gives the string list
        # so self._style_names[sample["style"]] -> "Impressionism"
        self._style_names = raw.features["style"].names
        self._genre_names = raw.features["genre"].names
        self._artist_names = raw.features["artist"].names  # e.g. "claude-monet"

        # ── Filter by style ────────────────────────────────────
        if style_filter is not None:
            style_id = None
            for i, name in enumerate(self._style_names):
                if name.lower().replace("_", " ") == style_filter.lower().replace("_", " "):
                    style_id = i
                    break
            if style_id is None:
                available = ", ".join(self._style_names)
                raise ValueError(
                    f"Style '{style_filter}' not found in WikiArt.\n"
                    f"Available: {available}"
                )
            logger.info(f"Filtering to style: '{style_filter}' (id={style_id})")
            # num_proc=1: each extra process forks the parent and dirties
            # copy-on-write pages, which spikes host RAM on memory-capped
            # nodes. Filtering 80k rows is disk-bound anyway, so parallelism
            # buys little here.
            raw = raw.filter(lambda x: x["style"] == style_id, num_proc=1)
            logger.info(f"After filter: {len(raw)} images")

        # ── Train / Val split ──────────────────────────────────
        # WikiArt only ships a single "train" split, so we carve out our
        # own validation set with a fixed seed for reproducibility.
        splits = raw.train_test_split(test_size=val_fraction, seed=42)
        hf_split = splits["train"] if split == "train" else splits["test"]

        if max_samples is not None:
            hf_split = hf_split.select(range(min(max_samples, len(hf_split))))

        self.data = hf_split
        logger.info(f"WikiArtHFDataset [{split}]: {len(self.data)} samples")

        # ── Same transform pipeline as ImageCaptionDataset ────
        transform_list = []
        if center_crop:
            transform_list += [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
            ]
        else:
            transform_list += [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(resolution),
            ]
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # HF datasets returns a PIL Image directly for Image-typed columns
        try:
            image = sample["image"].convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load WikiArt image at idx {idx}: {e}. Using blank.")
            image = Image.new("RGB", (self.resolution, self.resolution), color=128)

        pixel_values = self.transform(image)
        caption = self._build_caption(sample)

        result = {"pixel_values": pixel_values, "caption": caption}

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

    def _build_caption(self, sample: dict) -> str:
        """
        Synthesise a natural-language caption from WikiArt metadata.

        Example output: "an Impressionism painting, landscape, by Claude Monet"

        WHY THIS FORMAT?
        Style token first so the text encoder anchors on it immediately.
        Genre gives scene context for better compositional conditioning.
        Artist name adds diversity across the 13k Impressionism images.
        At inference you can use just "an impressionism painting of a sunset"
        and the LoRA transfers the style even to novel compositions.
        """
        style = self._style_names[sample["style"]].replace("_", " ")
        genre = self._genre_names[sample["genre"]].replace("_", " ")
        # WikiArt stores artists as "claude-monet" -> title-case with spaces
        artist = self._artist_names[sample["artist"]].replace("-", " ").title()
        article = "an" if style[0].lower() in "aeiou" else "a"
        return f"{article} {style} painting, {genre}, by {artist}"


def get_wikiart_dataloader(
    style_filter: str | None = "Impressionism",
    split: str = "train",
    val_fraction: float = 0.05,
    batch_size: int = 4,
    resolution: int = 512,
    center_crop: bool = True,
    random_flip: bool = True,
    tokenizer=None,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> torch.utils.data.DataLoader:
    """
    Convenience wrapper: WikiArtHFDataset -> DataLoader.

    HF datasets' Arrow backend is fork-safe, so multiple workers work
    correctly without extra configuration.
    """
    dataset = WikiArtHFDataset(
        style_filter=style_filter,
        split=split,
        val_fraction=val_fraction,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    logger.info(
        f"WikiArt DataLoader [{split}]: {len(dataset)} samples, "
        f"{len(dataloader)} batches (batch_size={batch_size})"
    )
    return dataloader


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
