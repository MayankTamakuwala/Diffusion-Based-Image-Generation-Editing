"""
src/data/create_sample_dataset.py
----------------------------------
Generate a tiny synthetic dataset for testing the pipeline.

WHAT THIS CREATES:
  dataset/
    train/
      images/  <- 20 synthetic images (solid colors + noise)
      captions/  <- 20 corresponding caption .txt files
    val/
      images/  <- 5 synthetic images
      captions/  <- 5 caption files

WHY SYNTHETIC DATA?
  - We can't ship 5k real images in this repo
  - Synthetic data lets you verify the entire pipeline runs without errors
  - The training will overfit immediately (expected!) — the goal is just to
    confirm data loading, training loop, and saving all work correctly

HOW TO USE YOUR REAL DATASET:
  1. Collect your 5k+ images (PNG/JPEG)
  2. Create one .txt caption file per image with the same stem:
       my_photo_001.png → my_photo_001.txt (containing "a photo of a dog")
  3. Place them in dataset/train/images/ and dataset/train/captions/
  4. Repeat for dataset/val/ (typically 10-20% of images)
  5. You're done — the training script handles the rest

AUTOMATION TIP: If you have images but no captions:
  - Use BLIP-2 or LLaVA to auto-caption: pip install transformers
    from transformers import pipeline
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = captioner(image)[0]["generated_text"]

Usage:
  python src/data/create_sample_dataset.py
  python src/data/create_sample_dataset.py --num_train 50 --num_val 10 --size 512
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Human-readable caption templates for synthetic images
# Using these makes CLIP evaluation slightly more meaningful (vs. random strings)
CAPTION_TEMPLATES = [
    "a beautiful sunset over the mountains",
    "a dog playing in the park",
    "a cat sitting on a windowsill",
    "a cityscape at night with lights",
    "a forest path covered in autumn leaves",
    "a beach with waves and blue sky",
    "a portrait of a person smiling",
    "a close-up of colorful flowers",
    "a vintage car on an empty road",
    "a cozy coffee shop interior",
    "an abstract painting with vibrant colors",
    "a snowy landscape with pine trees",
    "a river flowing through a valley",
    "a modern kitchen with marble counters",
    "a bookshelf filled with old books",
    "a waterfall in a tropical rainforest",
    "a night sky full of stars",
    "a child flying a kite in a meadow",
    "a wooden cabin in the mountains",
    "a market stall with fresh vegetables",
]


def generate_synthetic_image(size: int = 512, seed: int = 0) -> Image.Image:
    """
    Create a synthetic image: gradient background + random shapes.

    This is purely for pipeline testing. The content is not meaningful,
    but the image format (RGB, uint8, correct size) is valid.

    Args:
        size: Square image dimension in pixels.
        seed: Random seed for reproducibility.

    Returns:
        PIL Image in RGB mode.
    """
    rng = np.random.RandomState(seed)

    # Generate a gradient background
    # gradient goes from one random color to another
    color_a = rng.randint(0, 255, 3)  # RGB tuple
    color_b = rng.randint(0, 255, 3)

    # Create gradient: each row interpolates between color_a and color_b
    gradient = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        t = i / (size - 1)
        gradient[i, :, :] = (1 - t) * color_a + t * color_b

    # Add some random circles for texture
    img = Image.fromarray(gradient, mode="RGB")
    draw = ImageDraw.Draw(img)

    num_shapes = rng.randint(3, 10)
    for _ in range(num_shapes):
        x0 = int(rng.uniform(0, size))
        y0 = int(rng.uniform(0, size))
        radius = int(rng.uniform(20, size // 4))
        fill_color = tuple(rng.randint(0, 255, 3).tolist())
        draw.ellipse(
            [x0 - radius, y0 - radius, x0 + radius, y0 + radius],
            fill=fill_color,
            outline=None,
        )

    return img


def create_sample_dataset(
    output_dir: str = "dataset",
    num_train: int = 20,
    num_val: int = 5,
    size: int = 512,
    seed: int = 42,
) -> None:
    """
    Create synthetic training and validation datasets.

    Args:
        output_dir: Root directory for the dataset.
        num_train: Number of training images to generate.
        num_val: Number of validation images to generate.
        size: Image resolution (pixels, square).
        seed: Base seed for reproducibility.
    """
    random.seed(seed)
    output_path = Path(output_dir)

    splits = [
        ("train", num_train),
        ("val", num_val),
    ]

    for split_name, count in splits:
        images_dir = output_path / split_name / "images"
        captions_dir = output_path / split_name / "captions"
        images_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCreating {split_name} split: {count} images at {size}x{size}px")

        for i in range(count):
            # Generate synthetic image
            img_seed = seed + i + (10000 if split_name == "val" else 0)
            img = generate_synthetic_image(size=size, seed=img_seed)

            # Save image
            img_name = f"sample_{i:04d}.png"
            img_path = images_dir / img_name
            img.save(img_path, format="PNG", optimize=False)

            # Create caption (cycle through templates if fewer templates than images)
            caption = CAPTION_TEMPLATES[i % len(CAPTION_TEMPLATES)]
            caption_path = captions_dir / f"sample_{i:04d}.txt"
            caption_path.write_text(caption, encoding="utf-8")

            if (i + 1) % 5 == 0 or i == count - 1:
                print(f"  [{i+1}/{count}] {img_name} — '{caption[:50]}'")

        print(f"  Saved to: {images_dir}")

    print(f"\nDataset created successfully at: {output_path.resolve()}")
    print("\nTo use your REAL dataset instead:")
    print(f"  1. Replace {output_path}/train/images/ with your ~5000 images")
    print(f"  2. Add matching .txt captions to {output_path}/train/captions/")
    print(f"  3. Add ~500 val images+captions to {output_path}/val/")
    print(f"  4. Run training: python src/training/train_lora.py")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic sample dataset for pipeline testing."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Root directory for dataset output (default: dataset/)"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=20,
        help="Number of training images to generate (default: 20)"
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=5,
        help="Number of validation images to generate (default: 5)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image resolution in pixels (default: 512)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    create_sample_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        size=args.size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
