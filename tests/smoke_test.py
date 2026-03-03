"""
tests/smoke_test.py
-------------------
Smoke tests that verify each pipeline mode runs without crashing.

WHAT ARE SMOKE TESTS?
Smoke tests are quick sanity checks that verify:
  1. All imports work (no missing dependencies)
  2. Each code path runs end-to-end without exceptions
  3. Output has the correct shape/type/format

They do NOT test quality — they just verify the code doesn't crash.
We use 1-2 inference steps and 256x256 resolution for speed.

SMOKE TEST vs. UNIT TEST:
  - Smoke test: "does it run at all?" (broad, fast)
  - Unit test: "does this specific function return the right value?" (focused)

WHY --smoke_test FLAGS EVERYWHERE?
We add --smoke_test to CLI scripts so you can quickly verify a new setup
works before committing to a full 30-minute training run. Common use case:
  1. Set up new machine
  2. Run: pytest tests/smoke_test.py -v
  3. If all pass: run full training

Usage:
  pytest tests/smoke_test.py -v          # all smoke tests
  pytest tests/smoke_test.py -v -k cpu   # only CPU tests
  pytest tests/smoke_test.py -v -k data  # only data tests (no model download)
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# Mark slow tests (require model download + GPU)
# Run only data tests by default: pytest tests/smoke_test.py -m "not slow"
# ─────────────────────────────────────────────────────────────
slow = pytest.mark.slow


class TestImports:
    """Verify all project modules can be imported."""

    def test_import_utils(self):
        from src.utils.logging_utils import get_logger, setup_logging
        from src.utils.seed_utils import seed_everything, seed_generator
        assert True

    def test_import_data(self):
        from src.data.dataset import ImageCaptionDataset, get_dataloader
        from src.data.create_sample_dataset import create_sample_dataset
        assert True

    def test_import_models(self):
        from src.models.pipeline_utils import (
            load_txt2img_pipeline,
            load_img2img_pipeline,
            load_controlnet_pipeline,
            get_canny_edge_map,
            SCHEDULER_MAP,
        )
        assert True

    def test_import_evaluation(self):
        from src.evaluation.fid_eval import compute_fid
        from src.evaluation.clip_eval import compute_clip_score
        assert True

    def test_scheduler_map_not_empty(self):
        from src.models.pipeline_utils import SCHEDULER_MAP
        assert len(SCHEDULER_MAP) >= 4


class TestSeedUtils:
    """Verify seed utilities produce reproducible results."""

    def test_seed_everything_runs(self):
        from src.utils.seed_utils import seed_everything
        seed_everything(42)  # should not raise

    def test_seed_generator_reproducibility(self):
        """Same seed should produce identical Gaussian samples."""
        import torch
        from src.utils.seed_utils import seed_generator

        gen1 = seed_generator(42)
        gen2 = seed_generator(42)

        t1 = torch.randn(10, generator=gen1)
        t2 = torch.randn(10, generator=gen2)

        assert torch.allclose(t1, t2), "Same seed should produce same random values"

    def test_seed_generator_none(self):
        from src.utils.seed_utils import seed_generator
        gen = seed_generator(None)
        assert gen is None

    def test_get_random_seed_range(self):
        from src.utils.seed_utils import get_random_seed
        for _ in range(10):
            seed = get_random_seed()
            assert 0 <= seed <= 2**32 - 1


class TestLoggingUtils:
    """Verify logging setup."""

    def test_setup_logging_runs(self):
        from src.utils.logging_utils import setup_logging, get_logger
        setup_logging(log_level="WARNING")
        logger = get_logger("test")
        logger.warning("test warning")

    def test_timestamped_filename(self):
        from src.utils.logging_utils import get_timestamped_filename
        fname = get_timestamped_filename("metrics", ".json")
        assert fname.startswith("metrics_")
        assert fname.endswith(".json")
        assert len(fname) > 20  # should have a timestamp


class TestDataset:
    """Verify dataset loading works with the sample data."""

    @pytest.fixture(autouse=True)
    def create_sample_data(self, tmp_path):
        """Create a tiny synthetic dataset in a temp directory."""
        from src.data.create_sample_dataset import create_sample_dataset
        create_sample_dataset(
            output_dir=str(tmp_path),
            num_train=4,
            num_val=2,
            size=64,  # tiny for speed
            seed=0,
        )
        self.tmp_path = tmp_path

    def test_dataset_loads(self):
        from src.data.dataset import ImageCaptionDataset
        ds = ImageCaptionDataset(
            data_dir=self.tmp_path / "train",
            resolution=64,
            center_crop=True,
            random_flip=False,
        )
        assert len(ds) == 4

    def test_dataset_item_shape(self):
        from src.data.dataset import ImageCaptionDataset
        ds = ImageCaptionDataset(
            data_dir=self.tmp_path / "train",
            resolution=64,
        )
        item = ds[0]
        assert "pixel_values" in item
        assert "caption" in item
        assert item["pixel_values"].shape == (3, 64, 64)
        # Normalized values should be in [-1, 1]
        assert item["pixel_values"].min() >= -1.1
        assert item["pixel_values"].max() <= 1.1

    def test_dataset_caption_loading(self):
        from src.data.dataset import ImageCaptionDataset
        ds = ImageCaptionDataset(
            data_dir=self.tmp_path / "train",
            resolution=64,
        )
        item = ds[0]
        assert isinstance(item["caption"], str)
        assert len(item["caption"]) > 0

    def test_dataset_fallback_caption(self):
        """Dataset without captions should use fallback."""
        from src.data.dataset import ImageCaptionDataset
        # Remove captions dir
        import shutil
        shutil.rmtree(self.tmp_path / "train" / "captions", ignore_errors=True)

        ds = ImageCaptionDataset(
            data_dir=self.tmp_path / "train",
            resolution=64,
            fallback_caption="test fallback",
        )
        item = ds[0]
        assert item["caption"] == "test fallback"

    def test_dataloader_returns_batches(self):
        from src.data.dataset import get_dataloader
        dl = get_dataloader(
            data_dir=self.tmp_path / "train",
            batch_size=2,
            resolution=64,
            num_workers=0,  # 0 workers for testing (no multiprocessing)
        )
        batch = next(iter(dl))
        assert batch["pixel_values"].shape == (2, 3, 64, 64)

    def test_missing_images_raises(self):
        from src.data.dataset import ImageCaptionDataset
        with pytest.raises(FileNotFoundError):
            ImageCaptionDataset(data_dir="/nonexistent/path")


class TestCannyEdgeMap:
    """Verify Canny edge detection."""

    def test_canny_returns_pil_image(self):
        from PIL import Image
        from src.models.pipeline_utils import get_canny_edge_map

        # Create a simple test image
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        edge_map = get_canny_edge_map(img, low_threshold=100, high_threshold=200)

        assert isinstance(edge_map, Image.Image)
        assert edge_map.mode == "RGB"
        assert edge_map.size == img.size

    def test_canny_output_binary(self):
        """Edge map should only contain 0 or 255 values."""
        from PIL import Image
        from src.models.pipeline_utils import get_canny_edge_map
        import numpy as np

        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        edge_map = get_canny_edge_map(img)
        arr = np.array(edge_map)
        unique_values = np.unique(arr)
        # All values should be 0 (background) or 255 (edge)
        assert all(v in [0, 255] for v in unique_values)


class TestSampleDatasetCreation:
    """Verify synthetic dataset creation."""

    def test_create_sample_dataset(self, tmp_path):
        from src.data.create_sample_dataset import create_sample_dataset
        create_sample_dataset(
            output_dir=str(tmp_path),
            num_train=3,
            num_val=1,
            size=32,
        )
        # Check structure
        assert (tmp_path / "train" / "images").exists()
        assert (tmp_path / "train" / "captions").exists()
        assert (tmp_path / "val" / "images").exists()

        # Check file counts
        train_imgs = list((tmp_path / "train" / "images").glob("*.png"))
        val_imgs = list((tmp_path / "val" / "images").glob("*.png"))
        assert len(train_imgs) == 3
        assert len(val_imgs) == 1

    def test_synthetic_images_are_valid(self, tmp_path):
        from src.data.create_sample_dataset import create_sample_dataset
        from PIL import Image

        create_sample_dataset(str(tmp_path), num_train=2, num_val=1, size=32)
        img_path = list((tmp_path / "train" / "images").glob("*.png"))[0]
        img = Image.open(img_path)
        assert img.mode == "RGB"
        assert img.size == (32, 32)


# ─────────────────────────────────────────────────────────────
# SLOW TESTS: Require model downloads (skip with -m "not slow")
# ─────────────────────────────────────────────────────────────

@slow
class TestTxt2ImgSmoke:
    """Smoke test for text-to-image inference (requires model download)."""

    def test_txt2img_runs(self):
        from src.inference.txt2img import run_txt2img
        image, timing = run_txt2img(
            prompt="a red circle",
            width=256,
            height=256,
            num_steps=2,
            seed=42,
        )
        from PIL import Image
        assert isinstance(image, Image.Image)
        assert image.size == (256, 256)
        assert "inference_s" in timing


@slow
class TestImg2ImgSmoke:
    """Smoke test for image-to-image (requires model download)."""

    def test_img2img_runs(self):
        from PIL import Image
        from src.inference.img2img import run_img2img

        input_img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )

        output, timing = run_img2img(
            input_image=input_img,
            prompt="a colorful painting",
            strength=0.5,
            num_steps=2,
            seed=42,
        )

        assert isinstance(output, Image.Image)


@slow
class TestControlNetSmoke:
    """Smoke test for ControlNet (requires model download)."""

    def test_controlnet_runs(self):
        from PIL import Image
        from src.inference.controlnet_infer import run_controlnet

        # Checkerboard has strong edges
        checker = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    checker[i:i+32, j:j+32] = 200
        input_img = Image.fromarray(checker)

        generated, edge_map, timing = run_controlnet(
            input_image=input_img,
            prompt="colorful art",
            width=256,
            height=256,
            num_steps=2,
            seed=42,
        )

        assert isinstance(generated, Image.Image)
        assert isinstance(edge_map, Image.Image)


if __name__ == "__main__":
    # Run non-slow tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])
