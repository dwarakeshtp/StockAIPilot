"""Tests for the run, decision and job stores."""

from __future__ import annotations

import pytest

from src.db.engine import get_engine
from src.db.stores import JobStatus, JobStore, RunStore, SQLDecisionStore
from src.orchestration.decisions import DEFAULT_APPROVAL_REASON, DecisionType


@pytest.fixture()
def engine(settings):
    return get_engine(settings.database.url)


def test_run_store_round_trip_and_latest(engine, pipeline_result_factory) -> None:
    store = RunStore(engine)
    old = pipeline_result_factory(item_id="A")
    new = pipeline_result_factory(item_id="A")
    other = pipeline_result_factory(item_id="B")
    what_if = pipeline_result_factory(item_id="A", run_id="whatif-20990101T000000-abcdef")
    assert store.save_many([old, new, other, what_if]) == 4

    loaded = store.get(new.run_id)
    assert loaded is not None and loaded.model_dump() == new.model_dump()
    assert store.get("missing") is None
    latest = store.latest_by_item()
    assert list(latest) == ["A", "B"] and latest["A"].run_id == new.run_id
    assert store.latest_by_item(kind="whatif")["A"].run_id == what_if.run_id
    assert list(store.latest_by_item(item_ids=["B"])) == ["B"]
    summaries = store.list(item_id="A")
    assert len(summaries) == 3 and {"run_id", "action", "verdict"} <= set(summaries[0])


def test_sql_decision_store_rules_and_latest(engine, pipeline_result_factory) -> None:
    result = pipeline_result_factory()
    store = SQLDecisionStore(engine)
    approved = store.record(result, DecisionType.APPROVED)
    assert approved.reason == DEFAULT_APPROVAL_REASON
    assert approved.final_qty == result.critic.recommendation.quantity
    with pytest.raises(ValueError, match="reason"):
        store.record(result, DecisionType.REJECTED, reason="")
    rejected = store.record(result, DecisionType.REJECTED, reason="Supplier closed")
    assert rejected.final_qty == 0
    assert [d.decision for d in store.load()] == [DecisionType.APPROVED, DecisionType.REJECTED]
    assert store.latest_by_item()[result.item_id].decision == DecisionType.REJECTED
    assert store.load(item_id="other") == []


def test_job_store_lifecycle(engine) -> None:
    store = JobStore(engine)
    job = store.create(["A", "B"])
    assert job.status == JobStatus.QUEUED and job.total == 2
    store.mark_running(job.job_id)
    store.record_progress(job.job_id, run_id="run-1")
    store.record_progress(job.job_id, error="B: boom")
    finished = store.finish(job.job_id)
    assert finished.status == JobStatus.PARTIAL and finished.done == 2
    assert finished.run_ids == ["run-1"] and finished.errors == ["B: boom"]
    assert finished.started_at is not None and finished.finished_at is not None

    ok = store.create(["A"])
    store.record_progress(ok.job_id, run_id="run-2")
    assert store.finish(ok.job_id).status == JobStatus.SUCCEEDED
    failed = store.create(["A"])
    store.record_progress(failed.job_id, error="A: boom")
    assert store.finish(failed.job_id).status == JobStatus.FAILED
    assert [j.job_id for j in store.list()][0] == failed.job_id
    assert store.get("missing") is None
