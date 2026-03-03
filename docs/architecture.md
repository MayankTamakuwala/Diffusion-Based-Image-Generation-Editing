# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Diffusion-Based Generation System                      │
│                                                                          │
│  ┌──────────┐   ┌──────────────────────────────────────────────────┐    │
│  │ Training │   │              Inference Pipelines                  │    │
│  │          │   │                                                   │    │
│  │ dataset/ │   │  Tab 1: Text-to-Image                            │    │
│  │   ↓      │   │   Noise → UNet(prompt) → Decoded Image           │    │
│  │ LoRA     │   │                                                   │    │
│  │ Adapter  │   │  Tab 2: Image-to-Image                           │    │
│  │ .safetensors   │   Input → VAE Encode → Add Noise → UNet → Decode │    │
│  └──────────┘   │                                                   │    │
│       │         │  Tab 3: ControlNet (Canny)                       │    │
│       │ load    │   Input → Canny → ControlNet → UNet → Decode     │    │
│       ↓         │                                                   │    │
│  ┌──────────┐   └──────────────────────────────────────────────────┘    │
│  │  Gradio  │              ↑ load_pipelines() at startup                │
│  │  Web UI  │                                                            │
│  └──────────┘                                                            │
│       │                                                                  │
│       ↓                                                                  │
│  ┌──────────────────────────────────┐                                    │
│  │         Evaluation               │                                    │
│  │  generate_images → FID + CLIP    │                                    │
│  │  → experiments/metrics/*.json    │                                    │
│  └──────────────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Stable Diffusion 1.5 Components

```
Text Prompt
    │
    ▼
┌─────────────────┐
│  CLIP Tokenizer │  Converts text → token IDs (max 77 tokens)
│  +              │
│  Text Encoder   │  Token IDs → 768-dim embeddings (77 × 768)
└────────┬────────┘
         │ text embeddings (cross-attention)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                    UNet (860M params)                   │
│                                                         │
│  Encoder:    ResBlock → Attention → Downsample (×4)    │
│  Middle:     ResBlock → Attention → ResBlock            │
│  Decoder:    Upsample → Attention → ResBlock (×4)      │
│                                                         │
│  At each Attention block:                               │
│    Self-attention (spatial)                             │
│  + Cross-attention (text conditioning via embeddings)   │
│                                                         │
│  LoRA modifies: to_q, to_k, to_v, to_out.0             │
│  (the Q/K/V/Out projections in each attention block)    │
└────────┬───────────────────────────────────────────────┘
         │ predicted noise ε̂
         │
         ▼
┌─────────────────┐
│   Scheduler     │  Removes predicted noise from noisy latent
│  (DPM++ 2M)     │  z_{t-1} = f(z_t, ε̂, t)
└────────┬────────┘
         │ clean latent z_0  (4 × 64 × 64)
         │
         ▼
┌─────────────────┐
│  VAE Decoder    │  Latent → Pixel image (3 × 512 × 512)
│  (84M params)   │  z_0 → x_0  (upsamples 8× per spatial dim)
└────────┬────────┘
         │
         ▼
     PIL Image
```

## LoRA Injection Points

```
Original UNet attention weight matrix W (frozen):
  Input(d) → W (d × d) → Output(d)

With LoRA (rank r=16):
  Input(d) → W + ΔW → Output(d)
  where ΔW = B × A  (B: d×r, A: r×d)

  Only B and A are trained (2 × d × r parameters)
  For d=768, r=16: 2 × 768 × 16 = 24,576 params per layer
  vs 768 × 768 = 589,824 params for full fine-tuning

  Savings: 96% fewer parameters per layer!

Applied to 4 projections × ~16 attention layers = ~64 LoRA pairs
Total LoRA params: ~1.6M vs 860M (UNet) = 0.19% of model size
```

## ControlNet Architecture

```
Input Image
    │
    ▼
┌─────────────┐
│ Canny Edge  │  Detects edges using Gaussian + gradient + hysteresis
│ Detection   │  Output: binary edge map (white=edge, black=background)
└──────┬──────┘
       │ edge_map (3 × H × W)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│              ControlNet Module                        │
│  (copy of SD UNet encoder + "zero convolutions")     │
│                                                       │
│  Processes edge_map through:                          │
│    conv_in → encoder_blocks (×4) → middle_block       │
│                                                       │
│  Each layer output goes through a zero_convolution    │
│  (initialized to all-zeros → adds nothing at init)    │
│  These gradually learn to inject structural guidance  │
└──────┬───────────────────────────────────────────────┘
       │ control signals (list of feature maps)
       │                         ↓ added to
       ▼
┌──────────────────────────────────────────────────────┐
│              Base SD UNet                             │
│  Encoder → Middle → Decoder                          │
│                     ↑                                 │
│              ControlNet signals injected here         │
└──────────────────────────────────────────────────────┘
       │
       ▼
  Generated Image (follows edge structure + text prompt)
```

## Data Flow: Training Step

```
1. Load batch of N images + captions from dataset/
   ↓
2. Tokenize captions → input_ids
   ↓
3. VAE.encode(images) → latents z₀  [N × 4 × 64 × 64]
   ↓
4. Sample noise ε ~ N(0, I)  [same shape as latents]
   ↓
5. Sample random timesteps t ~ Uniform[1, 1000]  [N]
   ↓
6. Add noise: z_t = sqrt(ᾱ_t) × z₀ + sqrt(1-ᾱ_t) × ε
   (forward diffusion process)
   ↓
7. TextEncoder(input_ids) → text_embeds  [N × 77 × 768]
   ↓
8. UNet_lora(z_t, t, text_embeds) → predicted_noise  ε̂
   ↓
9. Loss = MSE(ε̂, ε)  [scalar]
   ↓
10. Backprop ONLY through LoRA parameters (A, B matrices)
    All base UNet weights remain frozen
   ↓
11. AdamW.step() → update LoRA weights
```
