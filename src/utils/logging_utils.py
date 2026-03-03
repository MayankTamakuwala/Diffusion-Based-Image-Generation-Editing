"""
src/utils/logging_utils.py
--------------------------
Centralized logging setup using Python's standard `logging` module + rich.

WHY CENTRALIZE LOGGING?
Instead of using `print()` everywhere (which goes away when you redirect output),
structured logging lets you:
  - Control verbosity with a single flag (DEBUG vs INFO vs WARNING)
  - Write to both console AND file simultaneously
  - Include timestamps, module names, and log levels automatically
  - Easily integrate with W&B, TensorBoard, or cloud logging later

USAGE in other modules:
    from src.utils.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.warning("Low VRAM detected")
    logger.debug("step loss: 0.423")  # only shown if DEBUG level
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# rich gives us pretty colored output in the terminal
try:
    from rich.logging import RichHandler
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Global flag to avoid setting up the root logger multiple times
_logging_configured = False


def setup_logging(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    use_rich: bool = True,
) -> None:
    """
    Configure root logger. Call ONCE at the start of your main script.

    Args:
        log_level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
        log_file:  Optional path to write logs to a file.
                   If None, only logs to console.
        use_rich:  Use rich for pretty colored terminal output.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # Console handler
    if use_rich and RICH_AVAILABLE:
        console_handler = RichHandler(
            level=level,
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    handlers.append(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # override any handlers set by imported libraries
    )

    # Silence overly verbose third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Hello from this module")
    """
    return logging.getLogger(name)


def get_timestamped_filename(prefix: str, suffix: str = ".json") -> str:
    """
    Generate a timestamped filename for saving experiment outputs.

    Example:
        >>> get_timestamped_filename("metrics")
        "metrics_20240315_143022.json"
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}{suffix}"
