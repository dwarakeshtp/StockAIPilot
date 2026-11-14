"""Risk & Uncertainty Agent: forecast confidence, anomalies and risk level."""

from __future__ import annotations

import pandas as pd

from src.agents.anomaly import detect_anomalies, detect_level_shift
from src.agents.base_agent import BaseAgent
from src.agents.confidence import (
    classify_risk,
    confidence_components,
    confidence_score,
    relative_interval_width,
)
from src.agents.schemas import ForecastOutput, RiskOutput
from src.data.quality import DataQualityReport


class RiskAgent(BaseAgent):
    """Assesses how much the forecast can be trusted and how risky the item is."""

    name = "risk_agent"
    role = "Risk & Uncertainty Agent - forecast confidence and anomaly analyst"

    def run(
        self,
        item_id: str,
        features: pd.DataFrame,
        forecast: ForecastOutput,
        data_quality: DataQualityReport,
    ) -> RiskOutput:
        """Score forecast confidence and classify risk for ``item_id``.

        Args:
            item_id: Item identifier.
            features: Feature frame of the item's history.
            forecast: Output of the Forecast Agent.
            data_quality: Data-quality report of the item's history.

        Returns:
            A validated :class:`RiskOutput`.
        """
        cfg = self.settings.risk
        anomalies = detect_anomalies(
            features, cfg.lookback_days, cfg.zscore_window, cfg.zscore_threshold,
            cfg.iqr_multiplier,
        )
        last_date = features["date"].max().date()
        recent = [a for a in anomalies if (last_date - a.date).days < cfg.recent_anomaly_days]
        shift = detect_level_shift(features, ratio=cfg.spike_ratio)

        last28 = features["y"].tail(28).mean()
        forecast_spike = bool(
            pd.notna(last28) and last28 > 0
            and forecast.mean_daily_demand / last28 >= cfg.spike_ratio
        )
        rel_width = relative_interval_width(
            [p.yhat for p in forecast.forecast],
            [p.lower for p in forecast.forecast],
            [p.upper for p in forecast.forecast],
        )
        components = confidence_components(
            forecast.selected_metrics.wape, rel_width, forecast.model_agreement_pct,
            data_quality.score, cfg.interval_width_cap,
        )
        score = confidence_score(components, cfg.weights, len(recent))
        escalate = shift.spike or shift.drop or forecast_spike
        level = classify_risk(score, cfg.high_threshold, cfg.low_threshold, escalate)
        multiplier = cfg.uncertainty_multipliers.get(level.value, 1.0)

        reasoning = [
            "Confidence components (0-1): "
            + ", ".join(f"{k}={v:.2f}" for k, v in components.items()),
            f"Confidence score {score:.1f}/100 (thresholds: HIGH<{cfg.high_threshold:g}, "
            f"LOW>={cfg.low_threshold:g}).",
            f"{len(anomalies)} anomalies in last {cfg.lookback_days} days, "
            f"{len(recent)} in last {cfg.recent_anomaly_days} days.",
            f"Recent 7-day demand is {shift.ratio:.2f}x the prior 28-day baseline.",
        ]
        if escalate:
            cause = ("recent demand spike" if shift.spike else
                     "recent demand drop" if shift.drop else "forecast spike vs recent demand")
            reasoning.append(f"Risk escalated one level due to {cause}.")
        reasoning.append(f"Risk level {level.value}: demand std inflated x{multiplier:.1f} "
                         "for safety-stock sizing.")

        output = RiskOutput(
            item_id=item_id,
            confidence_score=score,
            risk_level=level,
            confidence_components=components,
            anomalies=anomalies,
            recent_anomaly_count=len(recent),
            spike_detected=shift.spike,
            drop_detected=shift.drop,
            forecast_spike=forecast_spike,
            recent_to_baseline_ratio=shift.ratio,
            relative_interval_width=round(rel_width, 4),
            uncertainty_multiplier=multiplier,
            reasoning=reasoning,
        )
        output.narrative = self.narrate(
            "Explain how confident we are in the forecast and why the risk level was assigned.",
            {
                "item_id": item_id,
                "confidence_score": score,
                "risk_level": level.value,
                "components": components,
                "recent_anomalies": len(recent),
                "recent_to_baseline_ratio": shift.ratio,
                "spike_detected": shift.spike,
                "drop_detected": shift.drop,
                "uncertainty_multiplier": multiplier,
            },
        )
        self.logger.info("risk_assessed", item_id=item_id, score=score, level=level.value,
                         anomalies=len(anomalies))
        return output
