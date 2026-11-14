"""Identifier helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def new_run_id(prefix: str = "run") -> str:
    """Return a sortable, unique run identifier such as ``run-20261114T101500-1a2b3c``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)
