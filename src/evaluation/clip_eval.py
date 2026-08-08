"""
src/evaluation/clip_eval.py
---------------------------
CLIP (Contrastive Language-Image Pre-Training) similarity score computation.

WHAT IS CLIP SCORE?
CLIP was trained to produce similar embeddings for matching image-text pairs.
The CLIP score is the cosine similarity between:
  - A text embedding of your prompt
  - An image embedding of your generated image

INTERPRETATION:
  Score = 0: completely unrelated (orthogonal in embedding space)
  Score = 1: perfectly aligned (identical direction in embedding space)
  Typical good scores: 0.25 - 0.35 for SD 1.5 generated images
  CLIP score >= 0.30: good text-image alignment
  CLIP score >= 0.35: excellent (very faithfully follows prompt)

WHY CLIP SCORE MATTERS:
  FID tells you if images look realistic — but not if they match prompts.
  CLIP score tells you: "Does this image look like what I asked for?"

  Example:
    Prompt: "a red apple on a blue plate"
    - Image of a generic fruit: FID might be fine, CLIP score is low
    - Image of exact red apple on blue plate: CLIP score is high

WHY OPEN_CLIP OVER ORIGINAL CLIP?
  OpenCLIP (from LAION) provides:
  - More model variants (ViT-L/14, ViT-G/14, etc.)
  - Open weights (original CLIP needs OpenAI API key sometimes)
  - Same or better quality embeddings
  - pip installable without special setup

COMPUTING CLIP SCORE STEP-BY-STEP:
  1. Load CLIP model (ViT-B/32 by default)
  2. For each (prompt, image) pair:
     a. Tokenize text → text embedding (normalized)
     b. Preprocess image → image embedding (normalized)
     c. cosine_sim = dot(text_embed, image_embed)
  3. Return mean cosine similarity over all pairs

Usage:
  python src/evaluation/clip_eval.py --images_dir experiments/eval_generated --captions_file prompts.txt
  python src/evaluation/clip_eval.py --smoke_test
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# Running this file directly puts its own directory on sys.path, not the repo
# root, so "from src...." would fail. Add the repo root before any src import.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "auto",
):
    """
    Load OpenCLIP model and preprocessing transforms.

    Args:
        model_name: CLIP model architecture (e.g., "ViT-B-32", "ViT-L-14").
        pretrained: Weight source ("openai", "laion400m_e32", etc.)
        device: Compute device.

    Returns:
        (model, tokenizer, preprocess_fn, device)
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip not installed. Run: pip install open-clip-torch")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # open_clip uses underscore, not slash: ViT-B/32 → ViT-B-32
    # but also accepts the slash format, so we normalize just in case
    model_name_clean = model_name.replace("/", "-")

    logger.info(f"Loading CLIP model: {model_name_clean} ({pretrained}) on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name_clean,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name_clean)

    model = model.to(device).eval()
    return model, tokenizer, preprocess, device


def compute_clip_score(
    images: list[Image.Image | str | Path],
    texts: list[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 32,
    device: str = "auto",
) -> dict:
    """
    Compute CLIP similarity scores between images and texts.

    IMPORTANT: images[i] should correspond to texts[i].
    The list must be the same length.

    Args:
        images: List of PIL Images or paths to image files.
        texts: List of corresponding text prompts.
        model_name: CLIP model variant.
        pretrained: Weight source.
        batch_size: Processing batch size.
        device: Compute device.

    Returns:
        dict with keys:
          - "mean_score": float, average cosine similarity
          - "scores": list of per-pair scores
          - "std": standard deviation
          - "min_score": minimum score
          - "max_score": maximum score
    """
    if len(images) != len(texts):
        raise ValueError(
            f"images ({len(images)}) and texts ({len(texts)}) must have same length"
        )

    model, tokenizer, preprocess, device = load_clip_model(model_name, pretrained, device)

    all_scores = []

    # Process in batches to handle large evaluation sets
    for i in tqdm(range(0, len(images), batch_size), desc="Computing CLIP scores"):
        batch_images = images[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]

        # Preprocess images
        img_tensors = []
        for img in batch_images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            img_tensors.append(preprocess(img))

        img_batch = torch.stack(img_tensors).to(device)

        # Tokenize text
        text_tokens = tokenizer(batch_texts).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            # Get normalized embeddings
            # CLIP models output L2-normalized vectors, so cosine_sim = dot product
            image_features = model.encode_image(img_batch)
            text_features = model.encode_text(text_tokens)

            # Normalize (just to be safe — models should already normalize)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Per-pair cosine similarity
            # (batch, embed_dim) elementwise multiply → (batch, embed_dim) → sum → (batch,)
            scores = (image_features * text_features).sum(dim=-1)
            all_scores.extend(scores.cpu().float().tolist())

    scores_arr = np.array(all_scores)

    result = {
        "mean_score": float(np.mean(scores_arr)),
        "std": float(np.std(scores_arr)),
        "min_score": float(np.min(scores_arr)),
        "max_score": float(np.max(scores_arr)),
        "num_pairs": len(all_scores),
        "scores": all_scores,
    }

    logger.info(
        f"CLIP score: {result['mean_score']:.4f} ± {result['std']:.4f} "
        f"(min={result['min_score']:.4f}, max={result['max_score']:.4f})"
    )
    return result


def load_prompts_from_file(prompts_file: str | Path) -> list[str]:
    """Load prompts from a text file (one prompt per line)."""
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    lines = prompts_path.read_text(encoding="utf-8").strip().splitlines()
    return [l.strip() for l in lines if l.strip()]


def load_prompts_from_captions_dir(
    captions_dir: str | Path,
    image_paths: list[Path],
) -> list[str]:
    """
    Load captions matching the given image paths.

    For each image path, looks for a .txt file with the same stem
    in the captions_dir.
    """
    captions_dir = Path(captions_dir)
    prompts = []
    for img_path in image_paths:
        caption_path = captions_dir / (img_path.stem + ".txt")
        if caption_path.exists():
            prompts.append(caption_path.read_text(encoding="utf-8").strip())
        else:
            prompts.append("a photo")  # fallback
    return prompts


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Compute CLIP similarity score")
    parser.add_argument("--images_dir", type=str, default="experiments/eval_generated",
                        help="Directory of generated images")
    parser.add_argument("--captions_dir", type=str, default=None,
                        help="Directory of matching .txt caption files")
    parser.add_argument("--captions_file", type=str, default=None,
                        help="Text file with one prompt per line (same order as images)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to use for all images")
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        logger.info("SMOKE TEST: creating tiny synthetic images and prompts")
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        test_prompts = [
            "a red circle on white background",
            "a blue square",
            "a green triangle",
            "a yellow star",
        ]
        test_images = []
        for i, _ in enumerate(test_prompts):
            np.random.seed(i)
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(arr.astype(np.uint8))
            test_images.append(img)

        result = compute_clip_score(
            images=test_images,
            texts=test_prompts,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
        )
        print(f"\nSmoke test CLIP score: {result['mean_score']:.4f}")
        return

    # Load images
    images_dir = Path(args.images_dir)
    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    if len(image_paths) == 0:
        print(f"No images found in {images_dir}")
        return

    logger.info(f"Found {len(image_paths)} images in {images_dir}")

    # Load prompts
    if args.prompt:
        texts = [args.prompt] * len(image_paths)
    elif args.captions_file:
        texts = load_prompts_from_file(args.captions_file)
        if len(texts) != len(image_paths):
            logger.warning(
                f"Prompt count ({len(texts)}) != image count ({len(image_paths)}). "
                f"Truncating to min."
            )
            n = min(len(texts), len(image_paths))
            texts = texts[:n]
            image_paths = image_paths[:n]
    elif args.captions_dir:
        texts = load_prompts_from_captions_dir(args.captions_dir, image_paths)
    else:
        logger.warning("No prompts provided. Using 'a photo' for all images.")
        texts = ["a photo"] * len(image_paths)

    result = compute_clip_score(
        images=[str(p) for p in image_paths],
        texts=texts,
        model_name=args.model_name,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\nCLIP Score Results:")
    print(f"  Mean:  {result['mean_score']:.4f}")
    print(f"  Std:   {result['std']:.4f}")
    print(f"  Min:   {result['min_score']:.4f}")
    print(f"  Max:   {result['max_score']:.4f}")
    print(f"  N:     {result['num_pairs']}")
    print("\nInterpretation: >0.30 = good, >0.35 = excellent")


if __name__ == "__main__":
    main()
