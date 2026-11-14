"""Confidence scoring and risk classification (pure functions).

confidence = 100 * (w_acc * accuracy + w_prec * precision + w_agr * agreement
                    + w_dq * data_quality) - anomaly_penalty

* accuracy   = 1 - min(WAPE, 1)                 (backtest of the selected model)
* precision  = 1 - min(rel_interval_width / cap, 1)
* agreement  = ARIMA vs Prophet 7-day total agreement (0-1)
* data_quality = quality score of the history (0-1)
* anomaly_penalty = 5 points per recent anomaly (max 15)
"""

from __future__ import annotations

from src.agents.schemas import RiskLevel
from src.config.settings import RiskWeights

ANOMALY_PENALTY_PER_EVENT = 5.0
ANOMALY_PENALTY_MAX = 15.0


def relative_interval_width(yhat: list[float], lower: list[float], upper: list[float]) -> float:
    """Mean interval width divided by mean forecast (0 when the forecast is ~0)."""
    mean_yhat = sum(yhat) / max(len(yhat), 1)
    if mean_yhat <= 1e-9:
        return 0.0
    widths = [u - lo for u, lo in zip(upper, lower, strict=True)]
    return (sum(widths) / len(widths)) / mean_yhat


def confidence_components(
    wape: float | None,
    rel_width: float,
    agreement: float,
    data_quality: float,
    width_cap: float = 2.0,
) -> dict[str, float]:
    """Return each confidence component scaled to 0-1."""
    accuracy = 1 - min(wape, 1.0) if wape is not None else 0.5
    precision = 1 - min(rel_width / width_cap, 1.0)
    return {
        "accuracy": round(max(accuracy, 0.0), 4),
        "precision": round(max(precision, 0.0), 4),
        "agreement": round(min(max(agreement, 0.0), 1.0), 4),
        "data_quality": round(min(max(data_quality, 0.0), 1.0), 4),
    }


def confidence_score(
    components: dict[str, float], weights: RiskWeights, recent_anomalies: int = 0
) -> float:
    """Weighted confidence score in [0, 100] minus a recent-anomaly penalty."""
    raw = 100 * (
        weights.accuracy * components["accuracy"]
        + weights.precision * components["precision"]
        + weights.agreement * components["agreement"]
        + weights.data_quality * components["data_quality"]
    )
    penalty = min(ANOMALY_PENALTY_PER_EVENT * recent_anomalies, ANOMALY_PENALTY_MAX)
    return round(min(max(raw - penalty, 0.0), 100.0), 1)


_ORDER = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]


def classify_risk(
    score: float, high_threshold: float, low_threshold: float, escalate: bool = False
) -> RiskLevel:
    """Map a confidence score to a risk level, optionally escalating one level."""
    if score < high_threshold:
        level = RiskLevel.HIGH
    elif score < low_threshold:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW
    if escalate and level != RiskLevel.HIGH:
        level = _ORDER[_ORDER.index(level) + 1]
    return level
