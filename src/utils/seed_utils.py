"""
src/utils/seed_utils.py
-----------------------
Reproducibility helpers.

WHY SEED EVERYTHING?
In ML, randomness creeps in from:
  1. Weight initialization (we skip this — we use a pretrained model)
  2. Dropout and other stochastic layers
  3. Data shuffling
  4. Python's random module (used internally by many libraries)
  5. NumPy's random module
  6. PyTorch's CPU and GPU random number generators
  7. CUDA's non-deterministic algorithms (cuDNN)

If you don't fix all of these, two "identical" runs produce different results,
making it impossible to compare experiments fairly.
"""

import random
import os
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value. 42 is conventional but any int works.
    """
    # Python's built-in random module
    random.seed(seed)

    # OS-level seed (affects some C extensions)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy (used by Pillow, OpenCV, and many HF preprocessing steps)
    np.random.seed(seed)

    # PyTorch CPU operations
    torch.manual_seed(seed)

    # PyTorch GPU operations (all GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Make cuDNN deterministic.
    # WARNING: This may slightly reduce GPU performance because cuDNN
    # normally picks the fastest algorithm that may be non-deterministic.
    # For inference benchmarking, you may want to set these to False.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_generator(seed: int | None) -> torch.Generator | None:
    """
    Create a PyTorch Generator for use with diffusion pipelines.

    Diffusion pipelines accept a generator argument that controls the
    initial noise tensor. This lets you reproduce exact images.

    Args:
        seed: Integer seed, or None for random generation.

    Returns:
        torch.Generator on CPU (diffusers moves it to the right device),
        or None if seed is None.

    Example:
        >>> generator = seed_generator(42)
        >>> image = pipe(prompt, generator=generator).images[0]
    """
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def get_random_seed() -> int:
    """Return a random seed in [0, 2^32 - 1] for one-off generations."""
    return random.randint(0, 2**32 - 1)
