# ControlNet Explained

## What is ControlNet?

ControlNet (Zhang et al., 2023 — "Adding Conditional Control to Text-to-Image Diffusion Models") adds **structural conditioning** to Stable Diffusion. Without ControlNet, you can describe *what* to generate but not precisely *where* or *how*. ControlNet solves this by accepting an additional image-based conditioning signal alongside the text prompt.

## Supported Conditioning Types

| Type | Input | Use Case |
|------|-------|----------|
| **Canny** | Edge map | Architecture, precise structure |
| OpenPose | Body pose skeleton | Human pose preservation |
| Depth | Depth map | 3D scene structure |
| HED | Soft edges | Soft structure guidance |
| Segmentation | Semantic mask | Scene composition control |
| Normal Map | Normal vectors | Surface detail control |

**This project uses: Canny (model: `lllyasviel/sd-controlnet-canny`)**

## How ControlNet Works

```
Traditional SD:
  [Random Noise] → UNet(text_embed) → [Generated Image]

With ControlNet:
  [Random Noise] → UNet(text_embed + control_features) → [Structured Image]
  where control_features come from ControlNet(edge_map)
```

### Architecture Detail

ControlNet is a **copy of the UNet encoder** with additional "zero convolution" layers:

```
                    Text Prompt
                        │
                        ▼
          ┌─────────────────────────┐
          │   Stable Diffusion UNet │
          │                         │
Input ────►   Encoder (4 blocks)    │
Noise     │        │                │
          │   Middle Block          │
          │        │                │
          │   Decoder (4 blocks)◄───┼──── ControlNet outputs
          │        │                │     added here
          └────────┼────────────────┘
                   │
                Output Image

  ControlNet:
  Edge Map ──► conv_in ──► Encoder_copy (4 blocks)
                                │
                          zero_conv layers
                                │
                          Feature maps added
                          to UNet decoder
```

### Zero Convolutions

The key innovation is zero convolutions: `1×1` convolutions initialized to **weight=0, bias=0**.

At the start of training:
- Zero convolutions output zero
- ControlNet adds nothing to SD
- Model behaves identically to base SD
- No destructive noise in early training

As training progresses:
- Zero convolutions learn non-zero weights
- ControlNet gradually injects structural guidance
- Very stable training process

## Canny Edge Detection

Canny is a classic computer vision algorithm (John Canny, 1986) that finds edges in images.

### Steps:
1. **Gaussian blur** — removes high-frequency noise
2. **Sobel gradients** — finds image gradient magnitude and direction
3. **Non-maximum suppression** — thins edges to 1-pixel width
4. **Hysteresis thresholding**:
   - Strong edges: gradient > `high_threshold` → definitely edge
   - Weak edges: `low_threshold` < gradient < `high_threshold` → edge if connected to strong edge
   - Non-edges: gradient < `low_threshold` → background

### Our Thresholds (default: low=100, high=200)
```
low=50,  high=100  → many edges, more detail
low=100, high=200  → balanced (our default)
low=150, high=300  → only strong edges, cleaner map
```

## Conditioning Scale

`controlnet_conditioning_scale` controls ControlNet influence:
```
0.3 → loosely follow edges (more creative freedom)
0.7 → moderate adherence
1.0 → strict edge following (default)
1.5 → very strict (may cause artifacts at extremes)
```

## Use Cases

### Architecture Preservation
```
Input: photo of a building
Prompt: "same building in 1920s photographic style, black and white"
Result: same structure, different visual style
```

### Style Transfer with Structure
```
Input: sketch/drawing
Prompt: "photorealistic rendering of this design"
Result: realistic image that follows sketch outlines
```

### Scene Composition
```
Input: rough layout drawing
Prompt: "cozy living room, warm lighting, magazine photo"
Result: professional interior photo matching your layout
```

## Why ControlNet vs img2img?

| Feature | img2img | ControlNet |
|---------|---------|------------|
| Preserves structure | Approximately | Precisely (via edges) |
| Input dependency | Pixel-level | Edge-level only |
| Style freedom | Limited by strength | High (only edges constrained) |
| Best for | Style transfer | Structural precision |

## References

- ControlNet paper: https://arxiv.org/abs/2302.05543
- ControlNet models: https://huggingface.co/lllyasviel
- Diffusers ControlNet guide: https://huggingface.co/docs/diffusers/using-diffusers/controlnet
