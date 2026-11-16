"""Shared logging setup (REQUEST.md IMPORTANT NOTE: "lưu log và in ra terminal
tiến trình các bước xử lý đang ở đâu").

Every module does `logger = get_logger(__name__)` and logs each pipeline step
at INFO. Logs go to stdout (visible in the `uvicorn` terminal) AND to a
rotating file under `<OUTPUT_DIR>/_logs/app.log` so a run can be inspected
after the fact even if the terminal scrollback is gone.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def _configure_root(log_level: str, output_dir: Path) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("dco")
    root.setLevel(log_level.upper() if log_level else "INFO")
    root.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        log_dir = output_dir / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError: 
        # Filesystem might be read-only in some deploy environments -- console
        # logging alone is still fine, never let logging setup crash the app.
        console.warning("Could not open file log handler -- logging to console only.")

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Lazily configures the root `dco` logger on first use (avoids importing
    `app.settings` at module import time, which would create a circular
    import between `settings.py` and modules that both want to log)."""
    if not _CONFIGURED:
        from app.settings import get_settings

        s = get_settings()
        _configure_root(s.log_level, s.output_dir)
    return logging.getLogger(f"dco.{name}")
