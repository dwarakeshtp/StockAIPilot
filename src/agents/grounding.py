"""Deterministic grounding check for LLM-written text.

Every number in an LLM explanation must correspond to a number in the facts
the LLM was given (allowing rounding and fraction-to-percent conversion).
Dates and SKU identifiers are stripped before checking.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SKU_RE = re.compile(r"\b[A-Z]+(?:_[A-Z0-9]+)+\b")
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")
ALWAYS_ALLOWED: frozenset[float] = frozenset({0.0, 1.0, 100.0})


def extract_numbers(text: str) -> list[str]:
    """Return numeric tokens in ``text`` excluding dates and SKU identifiers."""
    cleaned = _SKU_RE.sub(" ", _DATE_RE.sub(" ", text))
    return _NUMBER_RE.findall(cleaned)


def collect_fact_numbers(facts: Any) -> list[float]:
    """Recursively collect every numeric value contained in ``facts``."""
    found: list[float] = []
    if isinstance(facts, bool):
        return found
    if isinstance(facts, int | float):
        found.append(float(facts))
    elif isinstance(facts, dict):
        for value in facts.values():
            found.extend(collect_fact_numbers(value))
    elif isinstance(facts, list | tuple | set):
        for value in facts:
            found.extend(collect_fact_numbers(value))
    elif isinstance(facts, str):
        found.extend(float(n.replace(",", "")) for n in extract_numbers(facts))
    return found


def _matches(value: float, candidates: Iterable[float]) -> bool:
    if value in ALWAYS_ALLOWED:
        return True
    for c in candidates:
        for target in (c, c * 100):
            if abs(value - target) <= max(0.5, 0.02 * abs(target)):
                return True
    return False


def check_grounding(text: str, facts: Any) -> tuple[bool, list[str]]:
    """Check that every number in ``text`` is supported by ``facts``.

    Returns:
        ``(passed, ungrounded_tokens)``.
    """
    candidates = collect_fact_numbers(facts)
    ungrounded = [
        token for token in extract_numbers(text)
        if not _matches(abs(float(token.replace(",", ""))), [abs(c) for c in candidates])
    ]
    return (not ungrounded, ungrounded)
