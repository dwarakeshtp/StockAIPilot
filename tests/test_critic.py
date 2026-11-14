"""Tests for validation rules and the Critic Agent."""

from __future__ import annotations

from factories import make_forecast, make_params, make_quality, make_risk

from src.agents.critic_agent import CriticAgent
from src.agents.inventory_agent import InventoryAgent
from src.agents.llm_client import LLMClient
from src.agents.schemas import InventoryAction, RevisionHint, RiskLevel, Verdict
from src.agents.validation_rules import (
    CriticContext,
    ExcessiveOrderRule,
    ForecastSanityRule,
    ModelDisagreementRule,
    StockoutBeforeOpenOrderRule,
    default_rules,
)


class FakeLLM(LLMClient):
    """LLM stub returning a fixed reply."""

    provider = "fake"

    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str | None:
        self.calls += 1
        return self.reply


def _run(settings, forecast=None, risk=None, params=None, quality=None, llm=None,
         revision_count=0):
    forecast = forecast or make_forecast()
    risk = risk or make_risk()
    inventory = InventoryAgent(settings).run("X", forecast, risk, params or make_params())
    critic = CriticAgent(settings, llm=llm)
    return critic.run("X", forecast, risk, inventory, quality or make_quality(),
                      history_max=40, revision_count=revision_count), inventory


def _ctx(settings, forecast=None, params=None, risk=None):
    forecast = forecast or make_forecast()
    risk = risk or make_risk()
    inv = InventoryAgent(settings).run("X", forecast, risk, params or make_params())
    return CriticContext(forecast=forecast, risk=risk, inventory=inv, data_quality=make_quality(),
                         settings=settings.critic, history_max=40)


def test_clean_case_is_approved(settings) -> None:
    out, inv = _run(settings, params=make_params(on_hand=120))
    assert inv.action == InventoryAction.HOLD
    assert out.verdict == Verdict.APPROVED
    assert [i.rule for i in out.issues] == ["simulated_inputs"]
    assert out.explanation_source == "template"
    assert "human approval" in out.explanation
    assert out.recommendation.requires_human_review


def test_model_disagreement_triggers_revision(settings) -> None:
    out, _ = _run(settings, forecast=make_forecast(agreement=0.3))
    assert out.verdict == Verdict.NEEDS_REVISION
    assert out.revision_request is not None
    assert out.revision_request.hint == RevisionHint.USE_ENSEMBLE


def test_revision_budget_exhausted_falls_back_to_warnings(settings) -> None:
    out, _ = _run(settings, forecast=make_forecast(agreement=0.3), revision_count=1,
                  params=make_params(on_hand=120))
    assert out.verdict == Verdict.APPROVED_WITH_WARNINGS
    assert out.revision_request is None


def test_forecast_sanity_rule_is_critical(settings) -> None:
    ctx = _ctx(settings, forecast=make_forecast(yhat_override=[5, 5, 5, 5, 5, 5, 99999]))
    issues = ForecastSanityRule().check(ctx)
    assert issues and issues[0].severity == "CRITICAL"
    assert issues[0].revision_hint == RevisionHint.USE_ALTERNATIVE_MODEL


def test_excessive_order_rule(settings) -> None:
    ctx = _ctx(settings, params=make_params(on_hand=0, moq=600, pack_size=6))
    assert ExcessiveOrderRule().check(ctx)


def test_disagreement_rule_no_hint_for_ensemble(settings) -> None:
    ctx = _ctx(settings, forecast=make_forecast(agreement=0.2, model="ensemble"))
    issues = ModelDisagreementRule().check(ctx)
    assert issues and issues[0].revision_hint is None


def test_stockout_before_open_order_escalates(settings) -> None:
    params = make_params(on_hand=25, on_order=200, on_order_eta_days=6, lead_time_days=2)
    ctx = _ctx(settings, params=params)
    assert ctx.inventory.action == InventoryAction.HOLD
    assert StockoutBeforeOpenOrderRule().check(ctx)
    out, _ = _run(settings, params=params)
    assert out.verdict == Verdict.NEEDS_HUMAN_REVIEW


def test_low_confidence_large_order_escalates(settings) -> None:
    risk = make_risk(RiskLevel.HIGH, score=30, multiplier=1.4)
    out, inv = _run(settings, risk=risk, params=make_params(on_hand=0, lead_time_days=10))
    assert inv.reorder_qty > 14 * inv.mean_daily_demand
    assert any(i.rule == "low_confidence_large_order" for i in out.issues)
    assert out.verdict == Verdict.NEEDS_HUMAN_REVIEW


def test_data_quality_rule(settings) -> None:
    out, _ = _run(settings, params=make_params(on_hand=120), quality=make_quality(0.5))
    assert any(i.rule == "data_quality" for i in out.issues)
    assert out.verdict == Verdict.APPROVED_WITH_WARNINGS


def test_grounded_llm_explanation_is_used(settings) -> None:
    llm = FakeLLM("Hold FOODS_X: inventory position 120 covers about 12 days; "
                  "requires human approval.")
    out, _ = _run(settings, params=make_params(on_hand=120), llm=llm)
    assert out.explanation_source == "llm:fake"
    assert out.grounding_passed
    assert llm.calls == 1


def test_ungrounded_llm_explanation_is_rejected(settings) -> None:
    llm = FakeLLM("Order 4321 units immediately, demand will rise 87%.")
    out, _ = _run(settings, params=make_params(on_hand=120), llm=llm)
    assert out.explanation_source == "template"
    assert not out.grounding_passed
    assert "4321" in out.ungrounded_numbers


def test_llm_not_called_when_revision_needed(settings) -> None:
    llm = FakeLLM("anything")
    out, _ = _run(settings, forecast=make_forecast(agreement=0.3), llm=llm)
    assert out.verdict == Verdict.NEEDS_REVISION
    assert llm.calls == 0


def test_default_rules_unique_ids() -> None:
    ids = [r.rule_id for r in default_rules()]
    assert len(ids) == len(set(ids)) == 13


def test_issues_sorted_by_severity(settings) -> None:
    out, _ = _run(settings, forecast=make_forecast(agreement=0.3), quality=make_quality(0.5),
                  revision_count=1)
    order = [i.severity for i in out.issues]
    assert order == sorted(order, key=["CRITICAL", "WARNING", "INFO"].index)
