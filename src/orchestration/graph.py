"""LangGraph orchestration of the data pipeline and the four agents.

Main graph::

    START -> ingest -> clean -> features -> forecast -> risk -> inventory -> critic
                                              ^                               |
                                              +---- NEEDS_REVISION (bounded) -+--> END

What-if graph (re-plans inventory for new assumptions)::

    START -> inventory -> critic -> END
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.critic_agent import CriticAgent
from src.agents.forecast_agent import ForecastAgent
from src.agents.inventory_agent import InventoryAgent
from src.agents.llm_client import LLMClient, build_llm_client
from src.agents.risk_agent import RiskAgent
from src.agents.schemas import Verdict
from src.config.settings import Settings
from src.data.cleaning import clean_item_sales
from src.data.features import build_features, build_future_frame, prophet_holidays
from src.data.repository import DataRepository
from src.orchestration.state import PipelineState
from src.orchestration.tracing import summarise_model, traced_node


@dataclass
class PipelineDependencies:
    """Objects injected into graph nodes (dependency injection for testability)."""

    settings: Settings
    repository: DataRepository
    llm: LLMClient
    forecast_agent: ForecastAgent
    risk_agent: RiskAgent
    inventory_agent: InventoryAgent
    critic_agent: CriticAgent

    @classmethod
    def build(
        cls,
        settings: Settings,
        repository: DataRepository | None = None,
        llm: LLMClient | None = None,
    ) -> PipelineDependencies:
        """Create default agents sharing one LLM client and repository."""
        llm = llm or build_llm_client(settings.llm)
        return cls(
            settings=settings,
            repository=repository or DataRepository(settings),
            llm=llm,
            forecast_agent=ForecastAgent(settings, llm),
            risk_agent=RiskAgent(settings, llm),
            inventory_agent=InventoryAgent(settings, llm),
            critic_agent=CriticAgent(settings, llm),
        )


def _make_nodes(deps: PipelineDependencies) -> dict[str, Any]:
    """Create the traced node functions bound to ``deps``."""
    horizon = deps.settings.forecast.horizon

    def ingest(state: dict[str, Any]) -> dict[str, Any]:
        item = state["item_id"]
        return {
            "raw_sales": deps.repository.item_sales(item),
            "params": deps.repository.item_inventory_params(item),
            "data_source": deps.repository.dataset.source,
        }

    def clean(state: dict[str, Any]) -> dict[str, Any]:
        cleaned, report = clean_item_sales(state["raw_sales"])
        return {"clean": cleaned, "data_quality": report,
                "history_max": float(cleaned["units"].max())}

    def features(state: dict[str, Any]) -> dict[str, Any]:
        feats = build_features(state["clean"])
        future = build_future_frame(feats, deps.repository.future_calendar, horizon)
        return {"features": feats, "future": future, "holidays": prophet_holidays(feats, future)}

    def forecast(state: dict[str, Any]) -> dict[str, Any]:
        out = deps.forecast_agent.run(state["item_id"], state["features"], state["future"],
                                      state["holidays"], revision=state.get("revision_request"))
        return {"forecast": out}

    def risk(state: dict[str, Any]) -> dict[str, Any]:
        out = deps.risk_agent.run(state["item_id"], state["features"], state["forecast"],
                                  state["data_quality"])
        return {"risk": out}

    def inventory(state: dict[str, Any]) -> dict[str, Any]:
        out = deps.inventory_agent.run(state["item_id"], state["forecast"], state["risk"],
                                       state["params"], state.get("overrides") or None)
        return {"inventory": out}

    def critic(state: dict[str, Any]) -> dict[str, Any]:
        revisions = state.get("revision_count", 0)
        out = deps.critic_agent.run(state["item_id"], state["forecast"], state["risk"],
                                    state["inventory"], state["data_quality"],
                                    state["history_max"], revision_count=revisions)
        update: dict[str, Any] = {"critic": out, "revision_request": out.revision_request}
        if out.verdict == Verdict.NEEDS_REVISION:
            update["revision_count"] = revisions + 1
        return update

    item = lambda s: {"item_id": s.get("item_id")}  # noqa: E731
    return {
        "ingest": traced_node(
            "ingest", "data_pipeline", ingest, item,
            lambda u: {"rows": len(u["raw_sales"]), "source": u["data_source"],
                       "params_simulated": bool(u["params"].get("is_simulated", True))}),
        "clean": traced_node(
            "clean", "data_pipeline", clean,
            lambda s: {"rows_in": len(s["raw_sales"])},
            lambda u: {"rows_out": len(u["clean"]),
                       **summarise_model(u["data_quality"], ("score", "suspected_stockout_days",
                                                             "missing_dates"))}),
        "features": traced_node(
            "features", "data_pipeline", features,
            lambda s: {"rows": len(s["clean"])},
            lambda u: {"feature_columns": len(u["features"].columns),
                       "future_days": len(u["future"]), "holidays": len(u["holidays"])}),
        "forecast": traced_node(
            "forecast", "forecast_agent", forecast,
            lambda s: {**item(s), "revision": (s.get("revision_request").hint.value
                                               if s.get("revision_request") else None)},
            lambda u: {**summarise_model(u["forecast"], ("model_used", "model_agreement_pct",
                                                         "skill_vs_naive", "selected_metrics")),
                       "forecast_total": round(u["forecast"].total_forecast, 1)},
            deps.forecast_agent),
        "risk": traced_node(
            "risk", "risk_agent", risk,
            lambda s: {"model_used": s["forecast"].model_used,
                       "data_quality": s["data_quality"].score},
            lambda u: summarise_model(u["risk"], ("confidence_score", "risk_level",
                                                  "recent_anomaly_count", "spike_detected",
                                                  "uncertainty_multiplier")),
            deps.risk_agent),
        "inventory": traced_node(
            "inventory", "inventory_agent", inventory,
            lambda s: {"mean_daily_demand": round(s["forecast"].mean_daily_demand, 2),
                       "risk_level": s["risk"].risk_level.value,
                       "overrides": s.get("overrides") or {}},
            lambda u: summarise_model(u["inventory"], ("action", "reorder_qty", "safety_stock",
                                                       "reorder_point", "inventory_position",
                                                       "stockout_probability")),
            deps.inventory_agent),
        "critic": traced_node(
            "critic", "critic_agent", critic,
            lambda s: {"revision_count": s.get("revision_count", 0)},
            lambda u: {**summarise_model(u["critic"], ("verdict", "explanation_source",
                                                       "grounding_passed")),
                       "issues": len(u["critic"].issues),
                       "revision": (u["critic"].revision_request.hint.value
                                    if u["critic"].revision_request else None)},
            deps.critic_agent),
    }


def _route_after_critic(state: dict[str, Any]) -> str:
    """Loop back to the Forecast Agent when the Critic asks for a revision."""
    critic = state["critic"]
    return "forecast" if critic.verdict == Verdict.NEEDS_REVISION else END


def build_pipeline_graph(deps: PipelineDependencies):
    """Compile the end-to-end pipeline graph."""
    nodes = _make_nodes(deps)
    graph = StateGraph(PipelineState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "clean")
    graph.add_edge("clean", "features")
    graph.add_edge("features", "forecast")
    graph.add_edge("forecast", "risk")
    graph.add_edge("risk", "inventory")
    graph.add_edge("inventory", "critic")
    graph.add_conditional_edges("critic", _route_after_critic, {"forecast": "forecast", END: END})
    return graph.compile()


def build_what_if_graph(deps: PipelineDependencies):
    """Compile the inventory -> critic graph used for what-if re-planning."""
    nodes = _make_nodes(deps)
    graph = StateGraph(PipelineState)
    graph.add_node("inventory", nodes["inventory"])
    graph.add_node("critic", nodes["critic"])
    graph.add_edge(START, "inventory")
    graph.add_edge("inventory", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def pipeline_mermaid(deps: PipelineDependencies) -> str:
    """Mermaid source of the compiled pipeline graph (for docs and the dashboard)."""
    return build_pipeline_graph(deps).get_graph().draw_mermaid()
