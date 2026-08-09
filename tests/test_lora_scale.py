"""
tests/test_lora_scale.py
------------------------
Regression tests: LoRA strength must survive every hop from CLI to loader.

WHY THESE EXIST:
  --lora_scale was added to the CLI and threaded into the visual comparison
  path, but not into the metrics path. Running with --lora_scale 0.6
  therefore produced a full-strength evaluation that looked completely
  normal -- correct-looking logs, plausible FID and CLIP numbers -- and was
  only caught because the values came back bit-identical to a previous
  full-strength run.

  A silently-dropped parameter that yields believable wrong numbers is far
  more dangerous than a crash, so each hop in the chain is pinned here:

      CLI --lora_scale
        -> run_metric_comparison(lora_scale=)
        -> config.model.lora_scale
        -> run_evaluation()
        -> generate_images_for_fid(lora_scale=)
        -> load_txt2img_pipeline(lora_scale=)
        -> load_lora_into_pipeline(lora_scale=)

  These run without a GPU or model weights by stubbing the pipeline loader
  and recording what it was called with.
"""

import inspect
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestLoraScaleSignatures:
    """Every function in the chain must accept a lora_scale parameter."""

    def test_pipeline_loaders_accept_scale(self):
        from src.models import pipeline_utils as pu

        for name in (
            "load_lora_into_pipeline",
            "load_txt2img_pipeline",
            "load_img2img_pipeline",
            "load_controlnet_pipeline",
        ):
            fn = getattr(pu, name)
            assert "lora_scale" in inspect.signature(fn).parameters, (
                f"{name} dropped lora_scale; callers would silently get full strength"
            )

    def test_generate_images_for_fid_accepts_scale(self):
        from src.evaluation.fid_eval import generate_images_for_fid

        assert "lora_scale" in inspect.signature(generate_images_for_fid).parameters

    def test_run_metric_comparison_accepts_scale(self):
        from src.evaluation.compare_base_vs_lora import run_metric_comparison

        assert "lora_scale" in inspect.signature(run_metric_comparison).parameters


class TestLoraScalePropagation:
    """The value must actually arrive, not merely be accepted."""

    def test_fid_generation_forwards_scale_to_loader(self, monkeypatch, tmp_path):
        """generate_images_for_fid -> load_txt2img_pipeline"""
        from src.evaluation import fid_eval
        from src.models import pipeline_utils

        recorded = {}

        class _FakeOutput:
            def __init__(self, n):
                from PIL import Image
                self.images = [Image.new("RGB", (8, 8)) for _ in range(n)]

        class _FakePipe:
            def __call__(self, prompt, **kwargs):
                return _FakeOutput(len(prompt))

        def _fake_loader(**kwargs):
            recorded.update(kwargs)
            return _FakePipe()

        monkeypatch.setattr(pipeline_utils, "load_txt2img_pipeline", _fake_loader)

        fid_eval.generate_images_for_fid(
            prompts=["a", "b"],
            output_dir=tmp_path,
            lora_path="some/adapter",
            lora_scale=0.6,
            batch_size=2,
        )

        assert recorded.get("lora_scale") == 0.6, (
            f"scale lost before the loader; got {recorded.get('lora_scale')}"
        )

    def test_metric_comparison_sets_scale_on_lora_arm_only(self, monkeypatch):
        """run_metric_comparison -> config.model.lora_scale, per arm"""
        from src.evaluation import compare_base_vs_lora as cmp_mod

        seen = []

        def _fake_run_evaluation(config, smoke_test=False):
            seen.append({
                "lora_path": config.model.lora_weights_path,
                "lora_scale": config.model.get("lora_scale", None),
                "output_dir": config.generation.output_dir,
            })
            return {"metrics": {"fid": 1.0, "clip_mean": 0.3}}

        import src.evaluation.run_eval as run_eval_mod
        monkeypatch.setattr(run_eval_mod, "run_evaluation", _fake_run_evaluation)

        config = OmegaConf.create({
            "model": {"base_model": "x", "lora_weights_path": None, "lora_scale": 1.0},
            "generation": {"output_dir": "unused"},
        })

        cmp_mod.run_metric_comparison(config, "some/adapter", False, lora_scale=0.6)

        assert len(seen) == 2, "expected exactly a base arm and a lora arm"
        base_arm, lora_arm = seen

        assert base_arm["lora_path"] is None
        assert lora_arm["lora_path"] == "some/adapter"
        assert lora_arm["lora_scale"] == 0.6, (
            f"lora arm evaluated at {lora_arm['lora_scale']}, not the requested 0.6"
        )
        # Distinct dirs, or the second arm's FID is computed over the first
        # arm's images and both numbers are wrong.
        assert base_arm["output_dir"] != lora_arm["output_dir"]


class TestLoraScaleConfig:
    """The shipped config must expose the knob."""

    def test_eval_config_has_lora_scale(self):
        config = OmegaConf.load("config/eval_config.yaml")
        assert "lora_scale" in config.model, (
            "eval_config.yaml lost model.lora_scale; runs would default to full strength"
        )
        assert 0.0 <= config.model.lora_scale <= 2.0
