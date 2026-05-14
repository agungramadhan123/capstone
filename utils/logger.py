"""
Structured logging with file + console output.
Usage: logger = setup_logger("train")
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: str = None, level=logging.INFO) -> logging.Logger:
    """Create a logger with console + file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console — force UTF-8 for Windows compatibility
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = logging.StreamHandler(utf8_stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File (optional)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_path / f"{name}_{timestamp}.log", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
