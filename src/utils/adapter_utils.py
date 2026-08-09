"""
src/utils/adapter_utils.py
--------------------------
Post-processing for PEFT adapter directories.

WHY THIS EXISTS:
  PEFT's LoraConfig is designed around transformers models, so applying it to
  a diffusers UNet leaves two fields null in adapter_config.json:

      "base_model_name_or_path": null
      "task_type": null

  The HuggingFace Hub's config parser validates these as strings and shows a
  warning banner on the model page for each:

      Configuration Parsing Warning: In adapter_config.json:
      "peft.base_model_name_or_path" must be a string

  Harmless to loading, but it is the first thing a visitor sees, and
  base_model_name_or_path in particular is real metadata people rely on to
  know which checkpoint an adapter belongs to.

WHAT WE DO ABOUT IT:
  - base_model_name_or_path: fill it in. We know the base model.
  - task_type: remove the key. PEFT's TaskType enum covers CAUSAL_LM,
    SEQ_CLS and friends -- none describe a diffusion UNet, so inventing a
    value would be worse than saying nothing. PeftConfig defaults it to None
    when absent, so loading is unaffected.
"""

import json
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def sanitize_adapter_config(
    adapter_dir: str | Path,
    base_model: str = "runwayml/stable-diffusion-v1-5",
) -> dict:
    """
    Fill in base_model_name_or_path and drop a null task_type, in place.

    Safe to run repeatedly; only writes when something actually changed.

    Args:
        adapter_dir: Directory holding adapter_config.json.
        base_model: Base checkpoint this adapter was trained against.

    Returns:
        The resulting config dict.
    """
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")

    config = json.loads(config_path.read_text())
    changed = []

    if not config.get("base_model_name_or_path"):
        config["base_model_name_or_path"] = base_model
        changed.append(f"base_model_name_or_path -> {base_model}")

    # Only drop it when null. If some future run sets a real task type,
    # leave it alone.
    if "task_type" in config and config["task_type"] is None:
        del config["task_type"]
        changed.append("task_type removed (was null)")

    if changed:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        for change in changed:
            logger.info(f"adapter_config.json: {change}")
    else:
        logger.debug("adapter_config.json already clean")

    return config
