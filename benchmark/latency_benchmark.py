"""
benchmark/latency_benchmark.py
-------------------------------
Latency benchmarking for Stable Diffusion inference.

WHAT THIS MEASURES:
  - p50 (median): typical inference time — what most users experience
  - p95: worst-case ~1 in 20 requests — important for production SLAs
  - Mean and std: overall distribution shape
  - Throughput: images/second

WHY p50/p95 INSTEAD OF MEAN?
  If 19 out of 20 runs take 2 seconds but 1 takes 20 seconds (CUDA hiccup),
  the mean becomes 2.9 seconds — misleadingly high.
  p95 = 20s honestly tells you "5% of your requests will be slow".
  In practice, latency distributions are right-skewed (few slow outliers).

MEASUREMENT METHODOLOGY:
  1. Load pipeline and run N_WARMUP warmup passes (exclude from stats)
     WHY? First few calls allocate memory, compile kernels, etc.
  2. Run N_BENCHMARK passes, recording wall-clock time for each
  3. Compute percentiles from the N_BENCHMARK timings

FACTORS AFFECTING LATENCY:
  - GPU model (RTX 3080 ≈ 100ms/step at fp16, 512x512)
  - Precision (fp16: ~1.5-2x faster than fp32)
  - xformers (can save 20-40%)
  - torch.compile (10-40% faster after 60-120s warmup)
  - Scheduler (all schedulers have similar per-step cost; fewer steps wins)
  - Image size (512x512 vs 768x768: ~2.5x more compute)
  - Batch size (batching is usually not efficient for single-user)

REALISTIC NUMBERS (fp16, xformers, RTX 3080):
  512x512, 30 steps:  p50≈3.2s, p95≈3.8s
  512x512, 20 steps:  p50≈2.2s, p95≈2.6s
  512x512, 10 steps:  p50≈1.1s, p95≈1.3s
  256x256, 10 steps:  p50≈0.6s, p95≈0.7s  ← "fast mode" target

Usage:
  python benchmark/latency_benchmark.py
  python benchmark/latency_benchmark.py --steps 10 --n_runs 50 --size 512
  python benchmark/latency_benchmark.py --smoke_test
"""

import argparse
import json
import platform
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm

from src.utils.logging_utils import get_logger, setup_logging, get_timestamped_filename
from src.utils.seed_utils import seed_generator

logger = get_logger(__name__)


def get_system_info() -> dict:
    """
    Collect system/hardware info for the benchmark report.

    This info lets you compare results across machines.
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
        # cuDNN version
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["driver_version"] = "see nvidia-smi"  # torch doesn't expose this directly
    else:
        info["cpu_count"] = (
            torch.get_num_threads()
        )

    return info


def run_benchmark(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: str | None = None,
    steps: int = 30,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 7.5,
    n_warmup: int = 3,
    n_runs: int = 20,
    scheduler: str = "DPMSolverMultistepScheduler",
    enable_xformers: bool = True,
    enable_torch_compile: bool = False,
    device: str = "auto",
    smoke_test: bool = False,
) -> dict:
    """
    Run latency benchmark for txt2img.

    Args:
        model_id: SD model to benchmark.
        lora_path: Optional LoRA adapter.
        steps: Inference steps.
        width, height: Output resolution.
        guidance_scale: CFG scale.
        n_warmup: Number of warmup runs (excluded from stats).
        n_runs: Number of benchmark runs.
        scheduler: Denoising scheduler.
        enable_xformers: Use xformers.
        enable_torch_compile: Use torch.compile.
        device: Compute device.
        smoke_test: Use minimal config for quick test.

    Returns:
        Dictionary with all benchmark results.
    """
    if smoke_test:
        steps = 2
        width = 256
        height = 256
        n_warmup = 1
        n_runs = 3
        logger.info("SMOKE TEST: 2 steps, 256x256, 3 runs")

    sys_info = get_system_info()
    logger.info(f"System: {sys_info.get('gpu_name', 'CPU')}")
    logger.info(f"Benchmark config: {steps} steps, {width}x{height}, {n_runs} runs")

    # Load pipeline
    from src.models.pipeline_utils import load_txt2img_pipeline
    logger.info("Loading pipeline for benchmark...")
    pipe = load_txt2img_pipeline(
        model_id=model_id,
        lora_weights_path=lora_path,
        scheduler_name=scheduler,
        enable_xformers=enable_xformers,
        enable_torch_compile=enable_torch_compile,
    )

    prompt = "a photorealistic landscape with mountains and a lake"
    timings_ms = []

    # CUDA synchronization is critical for accurate GPU timing.
    # torch.cuda.synchronize() forces the CPU to wait for GPU to finish.
    # Without it, timings reflect CPU scheduling time, not actual GPU time.
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ── Warmup Runs ────────────────────────────────────────────
    logger.info(f"Warmup: {n_warmup} runs...")
    for i in range(n_warmup):
        sync()
        with torch.no_grad():
            pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=seed_generator(i),
            )
        sync()
    logger.info("Warmup complete. Starting benchmark...")

    # ── Benchmark Runs ─────────────────────────────────────────
    for i in tqdm(range(n_runs), desc="Benchmark"):
        sync()
        t0 = time.perf_counter()

        with torch.no_grad():
            pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=seed_generator(n_warmup + i),  # different seed each run
            )

        sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings_ms.append(elapsed_ms)
        logger.debug(f"Run {i+1}/{n_runs}: {elapsed_ms:.1f}ms")

    # ── Compute Statistics ─────────────────────────────────────
    arr = np.array(timings_ms)
    stats = {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "ms_per_step": float(np.mean(arr)) / steps,
        "throughput_img_per_sec": 1000.0 / float(np.mean(arr)),
        "all_timings_ms": timings_ms,
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": sys_info,
        "benchmark_config": {
            "model_id": model_id,
            "lora_path": lora_path,
            "steps": steps,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler,
            "precision": "fp16" if torch.cuda.is_available() else "fp32",
            "xformers": enable_xformers,
            "torch_compile": enable_torch_compile,
            "n_warmup": n_warmup,
            "n_runs": n_runs,
        },
        "results": stats,
    }

    return results


def print_benchmark_report(results: dict) -> None:
    """Print a formatted benchmark report."""
    cfg = results["benchmark_config"]
    res = results["results"]
    sys = results["system_info"]

    print("\n" + "=" * 65)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 65)
    print(f"GPU:          {sys.get('gpu_name', 'CPU')}")
    if sys.get("vram_gb"):
        print(f"VRAM:         {sys['vram_gb']} GB")
    print(f"CUDA:         {sys.get('cuda_version', 'N/A')}")
    print(f"PyTorch:      {sys['torch_version']}")
    print()
    print(f"Model:        {cfg['model_id']}")
    print(f"LoRA:         {cfg['lora_path'] or 'None'}")
    print(f"Precision:    {cfg['precision']}")
    print(f"xformers:     {cfg['xformers']}")
    print(f"torch.compile:{cfg['torch_compile']}")
    print(f"Scheduler:    {cfg['scheduler']}")
    print(f"Resolution:   {cfg['width']}×{cfg['height']}")
    print(f"Steps:        {cfg['steps']}")
    print(f"Runs:         {cfg['n_runs']} (after {cfg['n_warmup']} warmup)")
    print()
    print(f"{'Metric':<20} {'Value':>12}")
    print("-" * 34)
    print(f"{'p50 (median)':<20} {res['p50_ms']:>10.1f}ms")
    print(f"{'p95':<20} {res['p95_ms']:>10.1f}ms")
    print(f"{'p99':<20} {res['p99_ms']:>10.1f}ms")
    print(f"{'Mean':<20} {res['mean_ms']:>10.1f}ms")
    print(f"{'Std dev':<20} {res['std_ms']:>10.1f}ms")
    print(f"{'ms / step':<20} {res['ms_per_step']:>10.1f}ms")
    print(f"{'Throughput':<20} {res['throughput_img_per_sec']:>9.2f} img/s")
    print("=" * 65)

    # Target check (from resume bullet: <120ms p95 per step)
    p95_per_step = res['p95_ms'] / cfg['steps']
    target_ms_total = 120  # ms "low latency" claim per step
    if p95_per_step < target_ms_total:
        print(f"✓ p95/step = {p95_per_step:.1f}ms < 120ms target")
    else:
        print(f"✗ p95/step = {p95_per_step:.1f}ms exceeds 120ms target")
        print("  Try: enable xformers, reduce steps, use fp16, try smaller resolution")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Latency benchmark for Stable Diffusion inference")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--size", type=int, default=512, help="Square image size")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--scheduler", type=str, default="DPMSolverMultistepScheduler")
    parser.add_argument("--no_xformers", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="experiments/latency")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    results = run_benchmark(
        model_id=args.model_id,
        lora_path=args.lora_path,
        steps=args.steps,
        width=args.size,
        height=args.size,
        guidance_scale=args.guidance_scale,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        scheduler=args.scheduler,
        enable_xformers=not args.no_xformers,
        enable_torch_compile=args.torch_compile,
        device=args.device,
        smoke_test=args.smoke_test,
    )

    print_benchmark_report(results)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_file = out_dir / get_timestamped_filename("latency", ".json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"JSON results saved: {json_file}")

    # CSV (convenient for spreadsheet analysis)
    import pandas as pd
    csv_file = out_dir / get_timestamped_filename("latency", ".csv")
    df = pd.DataFrame({
        "run": range(1, len(results["results"]["all_timings_ms"]) + 1),
        "latency_ms": results["results"]["all_timings_ms"],
    })
    df.to_csv(csv_file, index=False)
    logger.info(f"CSV results saved: {csv_file}")

    print(f"\nResults saved to:\n  {json_file}\n  {csv_file}")


if __name__ == "__main__":
    main()
