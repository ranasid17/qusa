# qusa/qusa/utils/logger.py

import logging
import sys
from pathlib import Path


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Sets up a logger with console and optional file output.

    Parameters:
        name (str): Name of the logger
        log_file (str): Path to log file (optional)
        level (logging.LEVEL): Logging level

    Returns:
        logger (logging.Logger): Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicate logs
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
