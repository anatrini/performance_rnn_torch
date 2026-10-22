"""Logging utilities for Performance RNN."""

import time
import logging
from pathlib import Path
from typing import Optional

from performance_rnn_torch.utils.paths import paths


def setup_logger(
    name: str,
    level: int = logging.INFO,
    file: bool = False,
    log_path: Optional[Path] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Name of the logger
        level: Logging level (default: logging.INFO)
        file: Whether to log to a file (default: False)
        log_path: Custom path for the log file. If None and file=True,
                 uses default path in logs directory.

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()

    # Handler to print on the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    if file:
        # Ensure logs directory exists
        paths.logs_dir.mkdir(parents=True, exist_ok=True)

        if log_path is None:
            # Generate timestamp-based log filename
            now = time.time()
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            log_path = paths.get_log_path(f'model_info_{timestamp}.log')

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

    return logger