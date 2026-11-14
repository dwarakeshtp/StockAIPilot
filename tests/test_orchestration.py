"""End-to-end tests of the LangGraph pipeline, what-if graph, persistence and decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.schemas import PipelineResult, RevisionHint, Verdict
from src.orchestration.decisions import Decision, DecisionStore, DecisionType
from src.orchestration.graph import pipeline_mermaid
from src.orchestration.runner import PipelineRunner, load_latest_results

MAIN_NODES = ["ingest", "clean", "features", "forecast", "risk", "inventory", "critic"]


@pytest.fixture()
def runner(settings, repository) -> PipelineRunner:
    return PipelineRunner(settings, repository=repository)


@pytest.fixture()
def result(runner, repository) -> PipelineResult:
    return runner.run_item(repository.item_ids[0])


def test_pipeline_runs_all_nodes_in_order(result: PipelineResult) -> None:
    nodes = [t.node for t in result.trace]
    assert nodes[:7] == MAIN_NODES
    assert [t.step for t in result.trace] == list(range(1, len(nodes) + 1))
    assert all(t.status == "success" and t.duration_ms >= 0 for t in result.trace)
    assert result.critic.verdict != Verdict.NEEDS_REVISION
    assert result.data_source == "synthetic"


def test_trace_json_written(result: PipelineResult) -> None:
    path = Path(result.trace_path)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["run_id"] == result.run_id
    assert [s["node"] for s in payload["steps"]][:7] == MAIN_NODES
    forecast_step = next(s for s in payload["steps"] if s["node"] == "forecast")
    assert forecast_step["reasoning"]
    assert "model_used" in forecast_step["output_summary"]


def test_revision_loop_reruns_forecast(settings, repository) -> None:
    critic = settings.critic.model_copy(update={"model_disagreement_threshold": -1.0})
    forced = settings.model_copy(update={"critic": critic})
    out = PipelineRunner(forced, repository=repository).run_item(repository.item_ids[0])
    nodes = [t.node for t in out.trace]
    assert nodes == MAIN_NODES + ["forecast", "risk", "inventory", "critic"]
    assert out.revision_count == 1
    assert out.forecast.model_used == "ensemble"
    assert out.forecast.revision_applied == RevisionHint.USE_ENSEMBLE
    first_critic = next(t for t in out.trace if t.node == "critic")
    assert first_critic.output_summary["verdict"] == "NEEDS_REVISION"
    assert out.critic.verdict != Verdict.NEEDS_REVISION


def test_what_if_reruns_inventory_and_critic(runner, result) -> None:
    shocked = runner.what_if(result, {"demand_multiplier": 3.0, "lead_time_days": 14})
    assert [t.node for t in shocked.trace] == ["inventory", "critic"]
    assert shocked.forecast == result.forecast
    assert shocked.inventory.reorder_point > result.inventory.reorder_point
    assert shocked.overrides["demand_multiplier"] == 3.0
    assert shocked.run_id.startswith("whatif-")


def test_save_and_load_results(runner, result, settings) -> None:
    runner.save_results([result])
    loaded = load_latest_results(settings)
    assert len(loaded) == 1
    assert loaded[0].run_id == result.run_id
    assert loaded[0].critic.recommendation == result.critic.recommendation


def test_mermaid_export(runner) -> None:
    mermaid = pipeline_mermaid(runner.deps)
    for node in MAIN_NODES:
        assert node in mermaid


def test_decision_store(tmp_path, result) -> None:
    store = DecisionStore(tmp_path / "decisions.jsonl")
    approved = store.record(result, DecisionType.APPROVED, "looks right")
    assert approved.final_qty == result.critic.recommendation.quantity
    modified = store.record(result, DecisionType.MODIFIED, "supplier cap", final_qty=24)
    assert modified.final_qty == 24
    rejected = store.record(result, DecisionType.REJECTED, "promo cancelled")
    assert rejected.final_qty == 0
    assert len(store.load()) == 3
    assert store.latest_by_item()[result.item_id].decision == DecisionType.REJECTED


def test_decision_validation() -> None:
    with pytest.raises(ValidationError):
        Decision(run_id="r", item_id="X", recommended_action="REORDER", recommended_qty=10,
                 verdict="APPROVED", decision=DecisionType.APPROVED, final_qty=5, reason="ok!")
    with pytest.raises(ValidationError):
        Decision(run_id="r", item_id="X", recommended_action="REORDER", recommended_qty=10,
                 verdict="APPROVED", decision=DecisionType.REJECTED, final_qty=0, reason="")
