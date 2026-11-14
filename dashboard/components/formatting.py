"""Formatting helpers (colours, badges, labels)."""

from __future__ import annotations

import streamlit as st

from src.agents.schemas import InventoryAction, RiskLevel, Verdict

RISK_COLORS: dict[str, str] = {RiskLevel.LOW: "green", RiskLevel.MEDIUM: "orange",
                               RiskLevel.HIGH: "red"}
VERDICT_COLORS: dict[str, str] = {
    Verdict.APPROVED: "green",
    Verdict.APPROVED_WITH_WARNINGS: "orange",
    Verdict.NEEDS_REVISION: "violet",
    Verdict.NEEDS_HUMAN_REVIEW: "red",
}
ACTION_COLORS: dict[str, str] = {InventoryAction.REORDER: "blue",
                                 InventoryAction.EXPEDITE: "red",
                                 InventoryAction.HOLD: "gray"}
SEVERITY_ICONS: dict[str, str] = {"CRITICAL": ":material/error:", "WARNING": ":material/warning:",
                                  "INFO": ":material/info:"}
PLOT_COLORS: dict[str, str] = {
    "actual": "#4a4a4a",
    "arima": "#0071CE",
    "prophet": "#E07B00",
    "ensemble": "#7B3FBF",
    "seasonal_naive": "#9E9E9E",
    "band": "rgba(0,113,206,0.15)",
    "anomaly": "#D62728",
    "stockout": "rgba(214,39,40,0.12)",
}


def verdict_badge(verdict: Verdict) -> None:
    """Render a coloured verdict badge."""
    st.badge(verdict.value.replace("_", " "), color=VERDICT_COLORS.get(verdict, "gray"))


def risk_badge(level: RiskLevel) -> None:
    """Render a coloured risk badge."""
    st.badge(f"{level.value} risk", color=RISK_COLORS.get(level, "gray"))


def action_badge(action: InventoryAction) -> None:
    """Render a coloured action badge."""
    st.badge(action.value, color=ACTION_COLORS.get(action, "gray"))


def pct(value: float | None, digits: int = 0) -> str:
    """Format a fraction as a percentage ('-' for None)."""
    return "-" if value is None else f"{value:.{digits}%}"
