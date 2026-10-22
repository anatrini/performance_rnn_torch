"""Logging utilities for Performance RNN.

Every module logs to a child of the logger 'prnn', which prints to the console;
a command that keeps a log file (train, optimize) adds it to 'prnn', so the
file receives the messages of all modules, the validation losses included.
"""

import logging
import time
from pathlib import Path

from performance_rnn_torch.utils.paths import paths

ROOT = 'prnn'


class _Console(logging.StreamHandler):
    """The console handler of 'prnn' (a class of its own, to add it only once)."""


def _root() -> logging.Logger:
    root = logging.getLogger(ROOT)
    root.setLevel(logging.INFO)
    root.propagate = False
    if not any(isinstance(h, _Console) for h in root.handlers):
        console_handler = _Console()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root.addHandler(console_handler)
    return root


def setup_logger(
    name: str, level: int = logging.INFO, file: bool = False, log_path: Path | None = None
) -> logging.Logger:
    """
    A logger that prints to the console, and with file=True also to a log file.

    Args:
        name: Name of the logger (a child of 'prnn')
        level: Logging level (default: logging.INFO)
        file: Also write the messages of every module to a log file, replacing
              the log file of an earlier call (default: False)
        log_path: Path of the log file. If None and file=True, a file named
                  after the current time in the logs directory.

    Returns:
        logging.Logger: Configured logger instance
    """
    root = _root()
    logger = logging.getLogger(f'{ROOT}.{name}')
    logger.setLevel(level)

    if file:
        for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
            root.removeHandler(handler)
            handler.close()
        if log_path is None:
            paths.logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_path = paths.get_log_path(f'model_info_{timestamp}.log')
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root.addHandler(file_handler)

    return logger
