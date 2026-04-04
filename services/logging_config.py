"""Zentrales Logging für alle Services."""

import logging
from pathlib import Path


def setup_service_logger(service_name: str, log_dir: str = "logs") -> logging.Logger:
    """Logger erstellen der in Datei und Console schreibt."""
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)

    # Datei-Handler
    log_file = Path(log_dir) / f"{service_name}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Console-Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Format
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger