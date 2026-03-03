# Evaluation Methodology

## Overview

We evaluate generative model quality using two complementary metrics:

| Metric | What It Measures | Range | Better = |
|--------|-----------------|-------|----------|
| FID | Realism + diversity vs. real images | [0, ∞) | Lower |
| CLIP Score | Text-image alignment | [-1, 1] | Higher |

Neither metric alone is sufficient:
- A model could produce blurry images → bad FID, could still have ok CLIP score
- A model could produce realistic but off-topic images → good FID, bad CLIP score
- Both together give a complete picture of model quality

---

## FID: Fréchet Inception Distance

### What It Measures

FID measures the **distributional distance** between two image sets:
- Real image distribution P_real (your validation set)
- Generated image distribution P_gen (your model's output)

Lower FID = generated images are more similar to real images in terms of:
1. **Fidelity** (are individual images realistic?)
2. **Diversity** (do generated images cover the same range as real images?)

A model that generates the same perfect image repeatedly gets a high FID (low diversity). A model that generates diverse but blurry images also gets a high FID (low fidelity).

### Computation

```
1. Feed all images through Inception-v3 (pretrained on ImageNet)
   → extract 2048-dim "pool3" feature vectors

2. Fit Gaussian to real features:    N(μ_r, Σ_r)
3. Fit Gaussian to generated features: N(μ_g, Σ_g)

4. FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r·Σ_g))
         ↑ mean distance     ↑ covariance distance (Wasserstein-2)
```

The Fréchet distance (a.k.a. Wasserstein-2 between Gaussians) is a proper distance metric in the feature space — not just pixel distance.

### Why cleanfid?

The original pytorch-fid implementation had a subtle bug: it resized images with antialiasing disabled, while the original TF implementation used antialiased bilinear resizing. This caused FID values to differ by 5-30% across implementations, making it impossible to compare results across papers.

cleanfid fixes this and is now the standard. It also caches the Inception statistics for your real image set, so you only pay the compute cost once.

### Minimum Image Count

```python
# Rule of thumb:
n_images < 500:   FID has high variance (don't report this)
n_images < 1000:  FID is noisy (caveat when reporting)
n_images >= 1000: FID is reliable
n_images >= 5000: FID matches published benchmarks
```

In our config, we default to 1000 images for FID. For comparing against SD 1.5 published numbers (FID ≈ 8-15 on COCO), you'd need 30,000+ images.

### Reference FID Values (SD 1.5, from literature)

| Dataset | Steps | FID |
|---------|-------|-----|
| MS-COCO (512px, 30 steps) | 30 | ~8-15 |
| After LoRA fine-tune (well-matched domain) | 30 | Should decrease |
| After LoRA fine-tune (mismatch) | 30 | May increase |

**Interpretation for our project:**
- Base SD 1.5 on our synthetic val set: high FID expected (synthetic ≠ real distribution)
- After LoRA on synthetic data: slightly lower FID (model learned our specific distribution)
- On a real dataset (5k photos): meaningful FID improvement from LoRA

---

## CLIP Score

### What It Measures

CLIP Score measures **semantic alignment** between a generated image and the text prompt used to generate it.

It asks: "Does this image look like what was described?"

### Computation

```python
# Using OpenCLIP (ViT-B/32, openai weights)

# 1. Encode image
image_tensor = preprocess(pil_image)  # normalize + resize for CLIP
image_features = clip.encode_image(image_tensor)
image_features = F.normalize(image_features, dim=-1)  # unit vector

# 2. Encode text
text_tokens = tokenizer(prompt)
text_features = clip.encode_text(text_tokens)
text_features = F.normalize(text_features, dim=-1)  # unit vector

# 3. Cosine similarity = dot product of unit vectors
clip_score = (image_features * text_features).sum()
# Range: [-1, 1], where 1 = identical direction in embedding space
```

### Reference CLIP Scores (SD 1.5)

| Setting | CLIP Score (ViT-B/32) |
|---------|----------------------|
| Random image, any prompt | ~0.10-0.15 |
| SD 1.5 baseline | ~0.27-0.32 |
| SD 1.5 + well-tuned prompt | ~0.30-0.36 |
| Good LoRA fine-tune | Should improve by 0.02-0.05 |

**Note:** CLIP score depends heavily on:
1. **Prompt quality**: longer, more descriptive prompts score higher
2. **Model variant**: ViT-L/14 scores differ from ViT-B/32 (use consistently)
3. **Image diversity**: using the same seed for all images inflates the score

### Limitations

CLIP score has known failure modes:
- **Spatial reasoning**: CLIP struggles with "the cat is to the LEFT of the dog"
- **Counting**: CLIP poorly evaluates "three dogs"
- **Negation**: "a photo WITHOUT a cat" — CLIP may score high with cats present
- **Fine-grained attributes**: specific colors/textures may be ignored

For these reasons, human evaluation remains important alongside CLIP scoring.

---

## Our Evaluation Configuration

```yaml
# config/eval_config.yaml
generation:
  num_images: 1000         # FID minimum threshold
  num_inference_steps: 30  # matches typical use case
  guidance_scale: 7.5

fid:
  mode: "clean"            # Standard mode (matches published results)
  real_images_dir: "dataset/val/images"

clip:
  model_name: "ViT-B/32"   # OpenCLIP ViT-B/32 (fast, widely used)
  pretrained: "openai"     # Original CLIP weights (best benchmark baseline)
```

## Output Format

Results are saved to `experiments/metrics/metrics_<timestamp>.json`:

```json
{
  "timestamp": "2024-03-15T14:30:22",
  "model_info": {
    "base_model": "runwayml/stable-diffusion-v1-5",
    "lora_path": "lora_weights/final_lora_adapter",
    "num_inference_steps": 30,
    "guidance_scale": 7.5
  },
  "metrics": {
    "fid": 42.31,
    "clip_mean": 0.3124,
    "clip_std": 0.0182,
    "clip_min": 0.2441,
    "clip_max": 0.3891,
    "clip_n_pairs": 1000
  },
  "runtime": {
    "generation_s": 245.2,
    "fid_s": 38.1,
    "clip_s": 12.4
  }
}
```

## Reporting Results (Interview Context)

When discussing evaluation in an interview, say:

> "We evaluated using FID and CLIP score. FID measures distributional similarity between generated and real images — lower is better, with SD 1.5 baseline around 10-15 on COCO. CLIP score measures text-image semantic alignment on a [-1,1] cosine scale, where 0.30+ indicates good alignment. We used cleanfid to avoid the antialiasing bug common in older FID implementations, and OpenCLIP ViT-B/32 with OpenAI weights for CLIP to match published baselines. Results are saved as timestamped JSON files in experiments/metrics/ to track improvements across training runs."
