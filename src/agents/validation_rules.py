"""Critic validation rules (strategy pattern).

Each rule inspects the combined agent outputs and returns zero or more
:class:`ValidationIssue` objects. New rules can be added without modifying the
Critic Agent (open/closed principle) by appending to :func:`default_rules`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from src.agents.schemas import (
    ForecastOutput,
    InventoryAction,
    InventoryOutput,
    RevisionHint,
    RiskLevel,
    RiskOutput,
    ValidationIssue,
)
from src.config.settings import CriticSettings
from src.data.quality import DataQualityReport


@dataclass(frozen=True)
class CriticContext:
    """Everything a rule may inspect."""

    forecast: ForecastOutput
    risk: RiskOutput
    inventory: InventoryOutput
    data_quality: DataQualityReport
    settings: CriticSettings
    history_max: float


class ValidationRule(ABC):
    """A single validation check."""

    rule_id: ClassVar[str] = "rule"
    escalates: ClassVar[bool] = False  # forces NEEDS_HUMAN_REVIEW when triggered

    @abstractmethod
    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Return the issues found (empty list when the check passes)."""

    def issue(self, severity: str, agent: str, message: str,
              hint: RevisionHint | None = None) -> ValidationIssue:
        """Helper to build an issue tagged with this rule's id."""
        return ValidationIssue(rule=self.rule_id, severity=severity, agent=agent,
                               message=message, revision_hint=hint)


class ForecastSanityRule(ValidationRule):
    """Forecast values must be finite, non-negative and plausible."""

    rule_id = "forecast_sanity"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag non-finite, negative or implausibly large forecasts."""
        values = [v for p in ctx.forecast.forecast for v in (p.yhat, p.lower, p.upper)]
        cap = 10 * max(ctx.history_max, 1.0)
        if any(not math.isfinite(v) or v < 0 for v in values) or max(values) > cap * 5:
            return [self.issue("CRITICAL", "forecast_agent",
                               "Forecast contains non-finite, negative or implausible values.",
                               RevisionHint.USE_ALTERNATIVE_MODEL)]
        return []


class IntervalConsistencyRule(ValidationRule):
    """Prediction intervals must bracket the point forecast."""

    rule_id = "interval_consistency"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag days where lower > yhat or yhat > upper."""
        bad = [p.date for p in ctx.forecast.forecast if not p.lower <= p.yhat <= p.upper]
        if bad:
            return [self.issue("CRITICAL", "forecast_agent",
                               f"Prediction interval inconsistent on {len(bad)} day(s).",
                               RevisionHint.USE_ALTERNATIVE_MODEL)]
        return []


class ForecastAccuracyRule(ValidationRule):
    """Backtest WAPE must be below the configured threshold."""

    rule_id = "forecast_accuracy"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag poor backtest accuracy; suggest the ensemble if it backtests better."""
        wape = ctx.forecast.selected_metrics.wape
        if wape is None or wape <= ctx.settings.wape_threshold:
            return []
        ens = ctx.forecast.metrics_by_model.get("ensemble")
        hint = None
        if (ctx.forecast.model_used != "ensemble" and ens is not None and ens.wape is not None
                and ens.wape < wape):
            hint = RevisionHint.USE_ENSEMBLE
        threshold = ctx.settings.wape_threshold
        return [self.issue("WARNING", "forecast_agent",
                           f"Backtest WAPE {wape:.0%} exceeds the {threshold:.0%} threshold; "
                           "forecast is unreliable.", hint)]


class ModelDisagreementRule(ValidationRule):
    """ARIMA and Prophet should broadly agree on 7-day demand."""

    rule_id = "model_disagreement"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag large ARIMA/Prophet disagreement and suggest the ensemble."""
        disagreement = 1 - ctx.forecast.model_agreement_pct
        if disagreement <= ctx.settings.model_disagreement_threshold:
            return []
        hint = RevisionHint.USE_ENSEMBLE if ctx.forecast.model_used != "ensemble" else None
        return [self.issue("WARNING", "forecast_agent",
                           f"ARIMA and Prophet disagree by {disagreement:.0%} on 7-day demand.",
                           hint)]


class NoSkillRule(ValidationRule):
    """The selected model should beat the seasonal-naive baseline."""

    rule_id = "no_skill_vs_naive"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag negative skill versus seasonal-naive."""
        skill = ctx.forecast.skill_vs_naive
        if skill is not None and skill < 0:
            return [self.issue("WARNING", "forecast_agent",
                               f"Selected model is {abs(skill):.0%} worse than seasonal-naive "
                               "in backtest (MAE).")]
        return []


class ExcessiveOrderRule(ValidationRule):
    """Post-order cover must not exceed the maximum days of cover."""

    rule_id = "excessive_order"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag orders that create more cover than allowed."""
        inv = ctx.inventory
        if inv.reorder_qty == 0 or inv.mean_daily_demand <= 0:
            return []
        cover = (inv.inventory_position + inv.reorder_qty) / inv.mean_daily_demand
        if cover > ctx.settings.max_days_of_cover:
            return [self.issue("WARNING", "inventory_agent",
                               f"Order would create {cover:.0f} days of cover (max "
                               f"{ctx.settings.max_days_of_cover:g}); risk of overstock.")]
        return []


class SafetyStockSanityRule(ValidationRule):
    """Safety stock should be proportionate to lead-time demand."""

    rule_id = "safety_stock_sanity"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag safety stock far above lead-time demand."""
        inv = ctx.inventory
        limit = ctx.settings.safety_stock_max_lt_multiple * max(inv.lead_time_demand, 1e-9)
        if inv.safety_stock > limit:
            return [self.issue("WARNING", "inventory_agent",
                               f"Safety stock {inv.safety_stock:.0f} is more than "
                               f"{ctx.settings.safety_stock_max_lt_multiple:g}x lead-time demand "
                               f"({inv.lead_time_demand:.0f}); driven by forecast uncertainty.")]
        return []


class RiskActionConsistencyRule(ValidationRule):
    """A HIGH-risk item with real stockout exposure should not simply HOLD."""

    rule_id = "risk_action_mismatch"
    escalates = True

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag HOLD on HIGH risk with material stockout probability."""
        inv = ctx.inventory
        if (ctx.risk.risk_level == RiskLevel.HIGH and inv.action == InventoryAction.HOLD
                and inv.stockout_probability > 0.2):
            return [self.issue("WARNING", "inventory_agent",
                               f"HIGH risk item set to HOLD with {inv.stockout_probability:.0%} "
                               "stockout probability.")]
        return []


class StockoutBeforeOpenOrderRule(ValidationRule):
    """HOLD is unsafe if on-hand stock runs out before the open order arrives."""

    rule_id = "stockout_before_open_order"
    escalates = True

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag HOLD recommendations with a projected stockout."""
        inv = ctx.inventory
        if inv.action != InventoryAction.HOLD or inv.projected_stockout_date is None:
            return []
        detail = ""
        if inv.params.on_order > 0:
            detail = (f" before the open order of {inv.params.on_order:.0f} units arrives "
                      f"(day {inv.params.on_order_eta_days})")
        return [self.issue("WARNING", "inventory_agent",
                           f"HOLD recommended but on-hand stock is projected to run out on "
                           f"{inv.projected_stockout_date}{detail}; consider expediting.")]


class LowConfidenceLargeOrderRule(ValidationRule):
    """Large orders based on low-confidence forecasts need a human check."""

    rule_id = "low_confidence_large_order"
    escalates = True

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag big orders when forecast confidence is low."""
        inv = ctx.inventory
        large = inv.reorder_qty > ctx.settings.large_order_days * max(inv.mean_daily_demand, 1e-9)
        if ctx.risk.confidence_score < ctx.settings.low_confidence_threshold and large:
            return [self.issue("WARNING", "risk_agent",
                               f"Order of {inv.reorder_qty} units (>"
                               f"{ctx.settings.large_order_days:g} days of demand) relies on a "
                               f"low-confidence forecast ({ctx.risk.confidence_score:.0f}/100).")]
        return []


class DataQualityRule(ValidationRule):
    """History quality should meet the minimum score."""

    rule_id = "data_quality"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag low data-quality scores, listing the underlying issues."""
        dq = ctx.data_quality
        if dq.score >= ctx.settings.min_data_quality_score:
            return []
        reasons = "; ".join(i.message for i in dq.issues[:2]) or "see data-quality report"
        return [self.issue("WARNING", "data", f"Data quality score {dq.score:.2f}: {reasons}.")]


class HighStockoutProbabilityRule(ValidationRule):
    """Very high stockout probability deserves explicit attention."""

    rule_id = "high_stockout_probability"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Flag stockout probability above the configured threshold."""
        p = ctx.inventory.stockout_probability
        if p > ctx.settings.high_stockout_probability:
            return [self.issue("WARNING", "inventory_agent",
                               f"Stockout probability before replenishment is {p:.0%}; "
                               "act promptly.")]
        return []


class SimulatedInputsRule(ValidationRule):
    """Remind reviewers that inventory inputs are simulated."""

    rule_id = "simulated_inputs"

    def check(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Emit an INFO issue when inventory parameters are simulated."""
        if ctx.inventory.params.is_simulated:
            return [self.issue("INFO", "data", "On-hand stock, open orders and lead times are "
                               "simulated for this POC.")]
        return []


def default_rules() -> list[ValidationRule]:
    """The rule set used by the Critic Agent, in evaluation order."""
    return [
        ForecastSanityRule(),
        IntervalConsistencyRule(),
        ForecastAccuracyRule(),
        ModelDisagreementRule(),
        NoSkillRule(),
        ExcessiveOrderRule(),
        SafetyStockSanityRule(),
        RiskActionConsistencyRule(),
        StockoutBeforeOpenOrderRule(),
        LowConfidenceLargeOrderRule(),
        DataQualityRule(),
        HighStockoutProbabilityRule(),
        SimulatedInputsRule(),
    ]
