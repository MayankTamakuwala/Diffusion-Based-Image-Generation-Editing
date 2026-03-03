# LoRA: Low-Rank Adaptation Explained

## The Problem LoRA Solves

Full fine-tuning of Stable Diffusion's UNet requires updating all **860 million parameters**. This demands:
- ~7GB GPU VRAM just for gradients + optimizer states (in fp32)
- Weeks of compute on large datasets
- Storage of multiple complete model checkpoints

**LoRA** (Hu et al., 2021 — "LoRA: Low-Rank Adaptation of Large Language Models") solves this by observing: **the weight updates needed for fine-tuning have low intrinsic rank**.

## The Math

For a pre-trained weight matrix W₀ ∈ ℝ^{m×n}, instead of learning ΔW directly (which has m×n parameters), we constrain it to be a low-rank product:

```
ΔW = B × A    where B ∈ ℝ^{m×r},  A ∈ ℝ^{r×n},  r << min(m, n)
```

The forward pass becomes:
```
h = W₀x + ΔWx = W₀x + BAx
```

Where:
- W₀ is frozen (no gradient flows through it)
- B and A are the only trained parameters
- r is the "rank" (our choice: 16)

## Why Rank 16?

| Rank | Parameters (per 768-dim layer) | Notes |
|------|-------------------------------|-------|
| 4    | 6,144                         | Very lightweight; good for style with 100-500 images |
| 8    | 12,288                        | Good for simple concepts |
| **16** | **24,576**                | **Best tradeoff for 5k+ images; our choice** |
| 32   | 49,152                        | Diminishing returns; use for complex domains |
| 64   | 98,304                        | Overkill unless 50k+ images |

**For 5k images fine-tuning rank=16:**
- Captures enough capacity for diverse content variation
- Empirically outperforms rank=4 on CLIP score by ~3-5%
- Fits in ~2GB extra VRAM (vs ~14GB for full fine-tuning)
- Full adapter is ~30MB (vs 3.4GB for full UNet weights)

## LoRA Alpha

```python
lora_alpha = 16  # we set this equal to rank
```

The actual scaling of ΔW is `(lora_alpha / rank)`. When `alpha == rank`, the scale is 1.0 — the LoRA update is added at full strength. This is the standard choice; it simplifies hyperparameter tuning.

## Which Layers Get LoRA?

We apply LoRA to the **attention projection matrices** of every attention block in the UNet:

```
to_q   — Query projection (spatial self-attention + cross-attention)
to_k   — Key projection
to_v   — Value projection
to_out.0 — Output projection
```

**Why attention layers, not all layers?**
Research shows attention layers are responsible for:
- **Cross-attention**: how text embeddings condition the image
- **Self-attention**: spatial coherence and style

Fine-tuning only attention layers gives 90%+ of the benefit of full fine-tuning at 0.2% of the cost.

## Initialization

```python
# Matrix A: initialized with Gaussian noise (standard)
# Matrix B: initialized to ZERO
# → ΔW = B×A = 0 at start → model starts identical to pretrained
```

This is crucial: initializing B to zero ensures the fine-tuned model starts exactly as the pretrained model. Training gradually introduces the learned adaptation.

## Parameter Count Example

UNet has ~32 attention layers, each with 4 projection matrices:

```
Without LoRA: 32 layers × 4 matrices × 768² = 75,497,472 params to train
With LoRA r=16: 32 × 4 × (768×16 + 16×768) = 3,145,728 params
Savings: ~96% fewer trainable parameters
```

## Merging LoRA at Inference

For deployment, you can **merge** the LoRA weights into the base model:
```
W_merged = W₀ + (alpha/rank) × B × A
```

This produces a single weight matrix with zero inference overhead. Diffusers supports this:
```python
pipe.fuse_lora()  # merge LoRA into base weights
pipe.unfuse_lora()  # reverse the merge
```

## PEFT Configuration in This Project

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=16,           # scaling = 1.0
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,        # regularization
    bias="none",             # don't fine-tune biases
)

unet_with_lora = get_peft_model(unet, lora_config)
# Only LoRA params have requires_grad=True now
```

## References

- LoRA paper: https://arxiv.org/abs/2106.09685
- DreamBooth + LoRA: https://arxiv.org/abs/2208.12242
- Diffusers LoRA guide: https://huggingface.co/docs/diffusers/training/lora
