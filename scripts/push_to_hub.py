"""
scripts/push_to_hub.py
----------------------
Publish the trained LoRA adapter to the HuggingFace Hub, with a model card.

WHY PUBLISH THE ADAPTER:
  The adapter is ~13MB (3.19M trainable params out of 862M), so unlike the
  4GB base model it is small enough to host and download casually. Pushing
  it means the README can say "pip install, point at this repo id, generate"
  rather than "clone this, train for 13 minutes on a GPU you may not have".

WHAT GETS UPLOADED:
  adapter_config.json + adapter_model.safetensors (the PEFT adapter), plus
  a generated README.md model card carrying the evaluation numbers and a
  runnable usage snippet.

AUTHENTICATION:
  Needs a token with write access, from https://huggingface.co/settings/tokens
  Either:
      huggingface-cli login
  or:
      export HF_TOKEN=hf_...

WHY THE MODEL CARD MATTERS:
  An adapter with no card is unusable by anyone else -- they cannot know the
  base model, the trigger phrasing, or the strength it was tuned for. Since
  this adapter is specifically NOT meant to run at full strength, that
  belongs on the card rather than in a commit message nobody will read.

Usage:
  python scripts/push_to_hub.py --repo_id your-username/sd15-wikiart-impressionism-lora
  python scripts/push_to_hub.py --repo_id ... --private
  python scripts/push_to_hub.py --repo_id ... --dry_run
"""

import argparse
import json
from pathlib import Path

# Running this file directly puts its own directory on sys.path, not the repo
# root, so "from src...." would fail. Add the repo root before any src import.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.utils.adapter_utils import sanitize_adapter_config
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


MODEL_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
tags:
  - stable-diffusion
  - lora
  - text-to-image
  - diffusers
  - art
  - impressionism
license: creativeml-openrail-m
pipeline_tag: text-to-image
---

# {title}

A LoRA adapter for Stable Diffusion 1.5, fine-tuned on {n_images} Impressionism
paintings from [WikiArt](https://huggingface.co/datasets/huggan/wikiart).

## Usage

```python
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

pipe = StableDiffusionPipeline.from_pretrained(
    "{base_model}",
    torch_dtype=torch.float16,
    safety_checker=None,
)

# Merge the adapter at {lora_scale} strength (see "Adapter strength" below)
peft_unet = PeftModel.from_pretrained(pipe.unet, "{repo_id}")
for module in peft_unet.modules():
    if hasattr(module, "scaling") and isinstance(module.scaling, dict):
        for name in module.scaling:
            module.scaling[name] *= {lora_scale}
pipe.unet = peft_unet.merge_and_unload()
pipe = pipe.to("cuda")

image = pipe(
    "an Impressionism painting, landscape, by Claude Monet",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("output.png")
```

## Adapter strength

**Use ~{lora_scale}, not 1.0.** At full strength this adapter imposes palette
and brushwork hard enough to dissolve composition in complex multi-figure
scenes. A matched-seed sweep over 0.4 / 0.6 / 0.8 / 1.0 put the breakdown
between 0.6 and 0.8, while the learned behaviours survive down to 0.4.

## Prompting

Trained on captions synthesized from WikiArt metadata in the form:

```
an Impressionism painting, {{genre}}, by {{artist}}
```

Prompts in that shape work best, e.g. `an Impressionism painting, landscape,
by Claude Monet`. Free-form prompts also work; the style transfers to novel
compositions.

## What the adapter learned

Two effects are clearly attributable to fine-tuning, since neither is
prompted and neither appears in base SD 1.5 output at matched seeds:

- **No picture frames.** Base SD renders "a painting" as a photograph of a
  *framed* painting on a wall. WikiArt images are cropped to the canvas, so
  the adapter produces edge-to-edge artwork.
- **Photographed-painting colour.** Output shifts toward the muted, slightly
  aged palette of real scanned paintings, away from SD's idealised saturation.

## Evaluation

{n_eval} generated images against {n_real} held-out WikiArt Impressionism
images the model never saw (seeded train/val split):

| Model | FID ↓ | CLIP ↑ |
|---|---|---|
| Base SD 1.5 | {fid_base} | {clip_base} |
| + this LoRA @ 1.0 | {fid_lora_full} | {clip_lora_full} |
| **+ this LoRA @ {lora_scale}** | **{fid_lora}** | **{clip_lora}** |

FID is computed with [clean-fid](https://github.com/GaParmar/clean-fid); CLIP
score with OpenCLIP ViT-B-32. Note the absolute FID is inflated by the small
reference set ({n_real} images, below the ~1000 where FID stabilises) -- the
*delta* against base is the meaningful quantity, not the absolute value.

## Training

| | |
|---|---|
| Base model | `{base_model}` |
| Dataset | `huggan/wikiart`, Impressionism, {n_images} images |
| LoRA rank / alpha | {rank} / {alpha} |
| Target modules | `to_q`, `to_k`, `to_v`, `to_out.0` |
| Trainable params | {trainable} ({trainable_pct} of {total_params}) |
| Epochs / steps | {epochs} / {steps} |
| Effective batch | {batch} |
| Optimizer | AdamW, lr {lr}, cosine schedule |
| Precision | bf16 |
| Hardware | 1× NVIDIA H200, {train_time} |

## Source

Training, evaluation, and serving code: {github_url}

## Limitations

- Complex multi-figure scenes degrade at high adapter strength.
- Trained on one style; other WikiArt styles need their own adapter.
- Inherits Stable Diffusion 1.5's limitations and biases.
- Non-commercial research use, per the CreativeML OpenRAIL-M licence.
"""


def build_model_card(
    repo_id: str,
    metrics: dict,
    base_model: str,
    lora_scale: float,
    github_url: str,
) -> str:
    """Fill the model card template from measured values."""
    title = repo_id.split("/")[-1].replace("-", " ").title()

    return MODEL_CARD_TEMPLATE.format(
        title=title,
        repo_id=repo_id,
        base_model=base_model,
        lora_scale=lora_scale,
        github_url=github_url,
        n_images=metrics.get("n_train_images", "5,000"),
        n_eval=metrics.get("n_eval_images", "1,000"),
        n_real=metrics.get("n_real_images", "653"),
        fid_base=metrics.get("fid_base", "106.50"),
        clip_base=metrics.get("clip_base", "0.3141"),
        fid_lora_full=metrics.get("fid_lora_full", "102.55"),
        clip_lora_full=metrics.get("clip_lora_full", "0.3037"),
        fid_lora=metrics.get("fid_lora", "100.38"),
        clip_lora=metrics.get("clip_lora", "0.3092"),
        rank=metrics.get("rank", 16),
        alpha=metrics.get("alpha", 16),
        trainable=metrics.get("trainable", "3,188,736"),
        trainable_pct=metrics.get("trainable_pct", "0.37%"),
        total_params=metrics.get("total_params", "862,709,700"),
        epochs=metrics.get("epochs", 6),
        steps=metrics.get("steps", "3,750"),
        batch=metrics.get("batch", 8),
        lr=metrics.get("lr", "1e-4"),
        train_time=metrics.get("train_time", "12m41s"),
    )


def collect_metrics(adapter_dir: Path, lora_scale: float = 0.6) -> dict:
    """
    Pull real numbers out of the adapter config and the comparison JSONs, so
    the card reports what was measured rather than what was typed here.

    The card quotes two LoRA rows -- full strength and the recommended
    strength -- so runs must be matched by their lora_scale rather than by
    recency. Taking "the newest run" would silently fill the recommended-
    strength row with whatever was measured last, which is the same class of
    bug as the dropped --lora_scale it is reporting on.
    """
    metrics = {}

    adapter_config = adapter_dir / "adapter_config.json"
    if adapter_config.exists():
        cfg = json.loads(adapter_config.read_text())
        metrics["rank"] = cfg.get("r", 16)
        metrics["alpha"] = cfg.get("lora_alpha", 16)

    comparisons = sorted(Path("experiments/comparisons").glob("comparison_*.json"))
    by_scale: dict[float, dict] = {}

    for path in comparisons:
        data = json.loads(path.read_text())
        m = data.get("metrics")
        if not m or m.get("base", {}).get("fid") is None:
            continue
        # Runs predating the lora_scale field were all full strength.
        scale = data.get("lora_scale")
        scale = 1.0 if scale is None else float(scale)
        by_scale[scale] = {"data": m, "file": path.name}  # later file wins per scale

    if not by_scale:
        logger.warning(
            "No completed metric comparisons found in experiments/comparisons/. "
            "The card will fall back to template defaults -- verify them before "
            "publishing."
        )
        return metrics

    # Base numbers are identical across runs (verified deterministic), so any
    # run can supply them.
    any_run = next(iter(by_scale.values()))["data"]
    metrics["fid_base"] = f"{any_run['base']['fid']:.2f}"
    metrics["clip_base"] = f"{any_run['base']['clip_mean']:.4f}"

    if 1.0 in by_scale:
        full = by_scale[1.0]["data"]["lora"]
        metrics["fid_lora_full"] = f"{full['fid']:.2f}"
        metrics["clip_lora_full"] = f"{full['clip_mean']:.4f}"
        logger.info(f"Full-strength metrics from {by_scale[1.0]['file']}")

    if lora_scale in by_scale:
        rec = by_scale[lora_scale]["data"]["lora"]
        metrics["fid_lora"] = f"{rec['fid']:.2f}"
        metrics["clip_lora"] = f"{rec['clip_mean']:.4f}"
        logger.info(f"Scale-{lora_scale} metrics from {by_scale[lora_scale]['file']}")
    else:
        logger.warning(
            f"No comparison run found at lora_scale={lora_scale} "
            f"(have: {sorted(by_scale)}). The card's recommended-strength row "
            f"will use template defaults -- rsync the run from the cluster, or "
            f"re-run compare_base_vs_lora.py --lora_scale {lora_scale} --metrics."
        )

    return metrics


def push_adapter(
    adapter_dir: str | Path,
    repo_id: str,
    lora_scale: float = 0.6,
    base_model: str = "runwayml/stable-diffusion-v1-5",
    github_url: str = "https://github.com/MayankTamakuwala/Diffusion-Based-Image-Generation-Editing",
    private: bool = False,
    dry_run: bool = False,
) -> str:
    """Create the repo if needed, write the model card, upload the adapter."""
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"No adapter at {adapter_path}. Train first, or pass --adapter_dir."
        )

    weights = list(adapter_path.glob("adapter_model.*"))
    if not weights:
        raise FileNotFoundError(
            f"{adapter_path} has no adapter_model.* -- is this a PEFT adapter "
            f"directory? Contents: {[p.name for p in adapter_path.iterdir()]}"
        )

    size_mb = sum(p.stat().st_size for p in adapter_path.iterdir() if p.is_file()) / 1e6
    logger.info(f"Adapter: {adapter_path} ({size_mb:.1f} MB)")

    # The Hub validates adapter_config.json and shows a warning banner for
    # each null field, so clean it up before it becomes the first thing a
    # visitor sees.
    sanitize_adapter_config(adapter_path, base_model=base_model)

    metrics = collect_metrics(adapter_path, lora_scale=lora_scale)
    card = build_model_card(repo_id, metrics, base_model, lora_scale, github_url)

    card_path = adapter_path / "README.md"
    card_path.write_text(card, encoding="utf-8")
    logger.info(f"Model card written: {card_path}")

    if dry_run:
        print("\n" + "=" * 62)
        print("DRY RUN -- nothing uploaded. Model card preview:")
        print("=" * 62)
        print(card)
        return card_path.as_posix()

    from huggingface_hub import HfApi, create_repo

    logger.info(f"Creating repo (if needed): {repo_id}")
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    api = HfApi()
    logger.info(f"Uploading {adapter_path} -> {repo_id}")
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add SD 1.5 WikiArt Impressionism LoRA adapter",
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Uploaded: {url}")
    return url


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Push the LoRA adapter to the HF Hub")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Target repo, e.g. your-username/sd15-wikiart-impressionism-lora")
    parser.add_argument("--adapter_dir", type=str,
                        default="lora_weights/final_lora_adapter")
    parser.add_argument("--lora_scale", type=float, default=0.6,
                        help="Recommended strength to document on the card")
    parser.add_argument("--base_model", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--github_url", type=str,
                        default="https://github.com/MayankTamakuwala/Diffusion-Based-Image-Generation-Editing")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Render the model card without uploading")
    args = parser.parse_args()

    result = push_adapter(
        adapter_dir=args.adapter_dir,
        repo_id=args.repo_id,
        lora_scale=args.lora_scale,
        base_model=args.base_model,
        github_url=args.github_url,
        private=args.private,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print("\n" + "=" * 62)
        print("PUSHED TO HUB")
        print("=" * 62)
        print(f"  {result}")
        print(f"\nLoad it with:\n  PeftModel.from_pretrained(pipe.unet, \"{args.repo_id}\")")
        print("=" * 62)


if __name__ == "__main__":
    main()
