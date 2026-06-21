"""
utils/logger.py
-----------------
One place to configure logging so every module gets consistent
formatting. In production, swap the StreamHandler for a structured
(JSON) handler shipping to your log aggregator (CloudWatch, ELK, etc.)
without touching any calling code.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("documind")
    if root.handlers:
        return  # already configured, avoid duplicate handlers on reload

    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
