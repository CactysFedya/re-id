from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    *,
    name: str = "pipeline",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    def _has_handler(handler_type: type, target: Optional[str] = None) -> bool:
        for h in logger.handlers:
            if isinstance(h, handler_type):
                if target is None:
                    return True
                if getattr(h, "baseFilename", None) == target:
                    return True
        return False

    if not _has_handler(logging.StreamHandler):
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        target = str(log_file.resolve())
        if not _has_handler(logging.FileHandler, target=target):
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
