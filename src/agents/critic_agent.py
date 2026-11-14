"""Critic Agent: validates all agent outputs, explains and summarises the decision."""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.grounding import check_grounding
from src.agents.llm_client import LLMClient
from src.agents.prompts import CRITIC_SYSTEM, critic_prompt
from src.agents.schemas import (
    CriticOutput,
    ForecastOutput,
    InventoryAction,
    InventoryOutput,
    RecommendationSummary,
    RevisionRequest,
    RiskOutput,
    ValidationIssue,
    Verdict,
)
from src.agents.validation_rules import CriticContext, ValidationRule, default_rules
from src.config.settings import Settings
from src.data.quality import DataQualityReport

_SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}


class CriticAgent(BaseAgent):
    """Validates the pipeline, decides the verdict and writes the explanation."""

    name = "critic_agent"
    role = "Critic Agent - reviewer and explainer"

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        rules: list[ValidationRule] | None = None,
    ) -> None:
        super().__init__(settings, llm)
        self.rules = rules if rules is not None else default_rules()

    def validate(self, ctx: CriticContext) -> list[ValidationIssue]:
        """Run every rule and return issues sorted by severity."""
        issues = [issue for rule in self.rules for issue in rule.check(ctx)]
        return sorted(issues, key=lambda i: _SEVERITY_ORDER[i.severity])

    def decide(
        self, issues: list[ValidationIssue], revision_count: int, model_used: str
    ) -> tuple[Verdict, RevisionRequest | None]:
        """Derive the verdict (and an optional revision request) from the issues."""
        max_revisions = self.settings.critic.max_revisions
        hinted = [i for i in issues if i.revision_hint is not None]
        if hinted and revision_count < max_revisions:
            first = hinted[0]
            return Verdict.NEEDS_REVISION, RevisionRequest(
                hint=first.revision_hint, reason=first.message, previous_model=model_used
            )
        escalating = {r.rule_id for r in self.rules if r.escalates}
        if any(i.severity == "CRITICAL" or i.rule in escalating for i in issues):
            return Verdict.NEEDS_HUMAN_REVIEW, None
        if any(i.severity == "WARNING" for i in issues):
            return Verdict.APPROVED_WITH_WARNINGS, None
        return Verdict.APPROVED, None

    @staticmethod
    def _headline(inv: InventoryOutput, product_name: str | None = None) -> str:
        label = f"{product_name} ({inv.item_id})" if product_name else inv.item_id
        if inv.action == InventoryAction.HOLD:
            return f"HOLD - no new order needed for {label}"
        return f"{inv.action.value} {inv.reorder_qty} units of {label}"

    @staticmethod
    def _rationale(inv: InventoryOutput) -> str:
        position = (f"Inventory position {inv.inventory_position:.0f} vs reorder point "
                    f"{inv.reorder_point:.0f}")
        if inv.action == InventoryAction.HOLD:
            text = f"{position}: stock covers about {inv.days_of_cover:.0f} days of demand."
        else:
            text = (f"{position}: replenishment needed to reach the order-up-to level "
                    f"(lead time {inv.params.lead_time_days:g} days).")
        if inv.projected_stockout_date:
            text += f" Without action, stock runs out on {inv.projected_stockout_date}."
        return text

    def _facts(self, forecast: ForecastOutput, risk: RiskOutput, inv: InventoryOutput,
               verdict: Verdict, issues: list[ValidationIssue],
               product_name: str | None = None) -> dict[str, Any]:
        return {
            "item_id": inv.item_id,
            "product_name": product_name or inv.item_id,
            "recommended_action": inv.action.value,
            "order_quantity_units": inv.reorder_qty,
            "order_value": inv.order_value,
            "forecast_model": forecast.model_used,
            "forecast_7d_total_units": round(forecast.total_forecast, 1),
            "mean_daily_demand": round(inv.mean_daily_demand, 1),
            "backtest_wape": forecast.selected_metrics.wape,
            "backtest_mape": forecast.selected_metrics.mape,
            "confidence_score": risk.confidence_score,
            "risk_level": risk.risk_level.value,
            "safety_stock": round(inv.safety_stock, 1),
            "reorder_point": round(inv.reorder_point, 1),
            "inventory_position": round(inv.inventory_position, 1),
            "days_of_cover": round(inv.days_of_cover, 1),
            "stockout_probability": inv.stockout_probability,
            "projected_stockout_date": str(inv.projected_stockout_date or "none"),
            "lead_time_days": inv.params.lead_time_days,
            "service_level": inv.service_level,
            "verdict": verdict.value,
            "validation_issues": [f"{i.severity}: {i.message}" for i in issues],
            "inventory_inputs_simulated": inv.params.is_simulated,
        }

    def _template_explanation(self, forecast: ForecastOutput, risk: RiskOutput,
                              inv: InventoryOutput, verdict: Verdict,
                              issues: list[ValidationIssue],
                              product_name: str | None = None) -> str:
        wape = forecast.selected_metrics.wape
        accuracy = f"backtest WAPE {wape:.0%}" if wape is not None else "limited backtest data"
        parts = [
            f"{self._headline(inv, product_name)}.",
            self._rationale(inv),
            f"The {forecast.model_used} model forecasts {forecast.total_forecast:.0f} units over "
            f"the next {forecast.horizon} days (about {inv.mean_daily_demand:.1f} per day, "
            f"{accuracy}).",
            f"Forecast confidence is {risk.confidence_score:.0f}/100 ({risk.risk_level.value} "
            f"risk), so safety stock is {inv.safety_stock:.0f} units at a "
            f"{inv.service_level:.0%} service level.",
        ]
        warnings = [i.message for i in issues if i.severity in ("WARNING", "CRITICAL")]
        if warnings:
            parts.append(f"Review notes: {warnings[0]}"
                         + (f" (+{len(warnings) - 1} more)." if len(warnings) > 1 else ""))
        parts.append(f"Verdict: {verdict.value.replace('_', ' ').lower()}; the final decision "
                     "requires human approval.")
        return " ".join(parts)

    def run(
        self,
        item_id: str,
        forecast: ForecastOutput,
        risk: RiskOutput,
        inventory: InventoryOutput,
        data_quality: DataQualityReport,
        history_max: float,
        revision_count: int = 0,
        product_name: str | None = None,
    ) -> CriticOutput:
        """Validate all outputs and produce the final recommendation.

        Args:
            item_id: Item identifier.
            forecast: Forecast Agent output.
            risk: Risk Agent output.
            inventory: Inventory Agent output.
            data_quality: Data-quality report.
            history_max: Maximum historical daily units (for plausibility checks).
            revision_count: Number of forecast revisions already performed.
            product_name: Display name used in the headline and explanation.

        Returns:
            A validated :class:`CriticOutput`.
        """
        ctx = CriticContext(forecast=forecast, risk=risk, inventory=inventory,
                            data_quality=data_quality, settings=self.settings.critic,
                            history_max=history_max)
        issues = self.validate(ctx)
        verdict, revision = self.decide(issues, revision_count, forecast.model_used)

        facts = self._facts(forecast, risk, inventory, verdict, issues, product_name)
        explanation, source, grounded, ungrounded = None, "template", True, []
        self.last_llm_provider = None
        if verdict != Verdict.NEEDS_REVISION and self.llm.available:
            text = self.llm.complete(CRITIC_SYSTEM, critic_prompt(facts))
            if text:
                grounded, ungrounded = check_grounding(text, facts)
                if grounded:
                    explanation = text
                    self.last_llm_provider = self.llm.last_provider
                    source = f"llm:{self.last_llm_provider}"
                else:
                    self.logger.warning("critic_explanation_ungrounded", numbers=ungrounded[:10])
        if explanation is None:
            explanation = self._template_explanation(forecast, risk, inventory, verdict, issues,
                                                     product_name)

        key_risks = [i.message for i in issues if i.severity in ("CRITICAL", "WARNING")]
        ratio = risk.recent_to_baseline_ratio
        if risk.spike_detected:
            key_risks.append(f"Recent demand spike ({ratio:.2f}x baseline).")
        if risk.drop_detected:
            key_risks.append(f"Recent demand drop ({ratio:.2f}x baseline).")
        recommendation = RecommendationSummary(
            item_id=item_id,
            action=inventory.action,
            quantity=inventory.reorder_qty,
            order_value=inventory.order_value,
            risk_level=risk.risk_level,
            confidence_score=risk.confidence_score,
            headline=self._headline(inventory, product_name),
            rationale=self._rationale(inventory),
            key_risks=key_risks,
        )
        counts = {s: sum(1 for i in issues if i.severity == s) for s in _SEVERITY_ORDER}
        reasoning = [
            f"Ran {len(self.rules)} validation rules: {counts['CRITICAL']} critical, "
            f"{counts['WARNING']} warnings, {counts['INFO']} info.",
            *[f"[{i.severity}] {i.rule}: {i.message}" for i in issues],
            f"Verdict: {verdict.value}"
            + (f" -> asking Forecast Agent to {revision.hint.value}." if revision else "."),
            f"Explanation source: {source}"
            + ("" if grounded else f" (LLM text rejected: ungrounded numbers {ungrounded[:5]})"),
        ]
        output = CriticOutput(
            item_id=item_id,
            verdict=verdict,
            issues=issues,
            revision_request=revision,
            explanation=explanation,
            explanation_source=source,
            grounding_passed=grounded,
            ungrounded_numbers=ungrounded,
            recommendation=recommendation,
            reasoning=reasoning,
        )
        self.logger.info("critic_completed", item_id=item_id, verdict=verdict.value,
                         issues=len(issues), source=source)
        return output
