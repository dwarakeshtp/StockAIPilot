"""Structured logging built on structlog.

Every log record is emitted as a JSON line to ``outputs/logs/app.jsonl`` and as
a human-readable line on the console. Context such as ``run_id`` or ``agent``
can be bound with :func:`bind_context` and is attached to every record.
"""

from __future__ import annotations

import logging as std_logging
import sys
from pathlib import Path
from typing import Any

import structlog

_CONFIGURED: bool = False


def configure_logging(log_dir: Path | None = None, level: str = "INFO") -> None:
    """Configure structlog and the standard library root logger once.

    Args:
        log_dir: Directory for the JSON log file. Console-only when ``None``.
        level: Minimum log level name.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = std_logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = std_logging.StreamHandler(sys.stderr)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False),
            foreign_pre_chain=shared_processors,
        )
    )
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = std_logging.FileHandler(log_dir / "app.jsonl", encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )
        root.addHandler(file_handler)

    for noisy in ("httpx", "urllib3", "openai", "matplotlib"):
        std_logging.getLogger(noisy).setLevel(std_logging.WARNING)
    std_logging.getLogger("cmdstanpy").disabled = True
    std_logging.getLogger("prophet").setLevel(std_logging.ERROR)

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to ``name``."""
    return structlog.get_logger(name)


def bind_context(**values: Any) -> None:
    """Bind key/value pairs to every subsequent log record in this context."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Remove all context variables bound with :func:`bind_context`."""
    structlog.contextvars.clear_contextvars()
