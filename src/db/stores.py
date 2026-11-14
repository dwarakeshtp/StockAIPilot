"""Persistence of pipeline runs, planner decisions and background jobs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import Engine, and_, func, insert, select, update

from src.agents.schemas import PipelineResult
from src.db.schema import create_schema, decisions, jobs, pipeline_runs
from src.orchestration.decisions import Decision, DecisionType, build_decision
from src.utils.ids import utc_now

RUN_KIND_PIPELINE = "pipeline"
RUN_KIND_WHAT_IF = "whatif"


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None


class RunStore:
    """Stores every :class:`PipelineResult` as JSON plus queryable summary columns."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        create_schema(engine)

    @staticmethod
    def kind_of(result: PipelineResult) -> str:
        """``whatif`` for what-if re-plans, else ``pipeline``."""
        return RUN_KIND_WHAT_IF if result.run_id.startswith("whatif") else RUN_KIND_PIPELINE

    def save(self, result: PipelineResult) -> None:
        """Insert (or replace) one run."""
        rec = result.critic.recommendation
        row = {
            "run_id": result.run_id, "kind": self.kind_of(result), "item_id": result.item_id,
            "created_at": _iso(result.created_at), "action": rec.action.value,
            "qty": rec.quantity, "verdict": result.critic.verdict.value,
            "revision_count": result.revision_count,
            "result_json": result.model_dump_json(),
        }
        with self.engine.begin() as conn:
            conn.execute(pipeline_runs.delete().where(pipeline_runs.c.run_id == result.run_id))
            conn.execute(insert(pipeline_runs), [row])

    def save_many(self, results: list[PipelineResult]) -> int:
        """Persist several runs; returns how many were written."""
        for result in results:
            self.save(result)
        return len(results)

    def get(self, run_id: str) -> PipelineResult | None:
        """One run by id."""
        with self.engine.connect() as conn:
            raw = conn.execute(select(pipeline_runs.c.result_json)
                               .where(pipeline_runs.c.run_id == run_id)).scalar()
        return PipelineResult.model_validate_json(raw) if raw else None

    def latest_by_item(self, kind: str = RUN_KIND_PIPELINE,
                       item_ids: list[str] | None = None) -> dict[str, PipelineResult]:
        """Most recent run per item, ordered by item id."""
        latest = (select(pipeline_runs.c.item_id,
                         func.max(pipeline_runs.c.created_at).label("created_at"))
                  .where(pipeline_runs.c.kind == kind).group_by(pipeline_runs.c.item_id)
                  .subquery())
        query = (select(pipeline_runs.c.result_json)
                 .join(latest, and_(pipeline_runs.c.item_id == latest.c.item_id,
                                    pipeline_runs.c.created_at == latest.c.created_at))
                 .where(pipeline_runs.c.kind == kind).order_by(pipeline_runs.c.item_id))
        if item_ids:
            query = query.where(pipeline_runs.c.item_id.in_(item_ids))
        with self.engine.connect() as conn:
            rows = conn.execute(query).scalars().all()
        results = [PipelineResult.model_validate_json(r) for r in rows]
        return {r.item_id: r for r in results}

    def list(self, item_id: str | None = None, kind: str | None = None,
             limit: int = 50) -> list[dict[str, Any]]:
        """Run summaries, newest first."""
        cols = [c for c in pipeline_runs.c if c.name != "result_json"]
        query = select(*cols).order_by(pipeline_runs.c.created_at.desc()).limit(limit)
        if item_id:
            query = query.where(pipeline_runs.c.item_id == item_id)
        if kind:
            query = query.where(pipeline_runs.c.kind == kind)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(query)]


class SQLDecisionStore:
    """Planner decisions in the ``decisions`` table (same API as the JSONL store)."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        create_schema(engine)

    def record(self, result: PipelineResult, decision: DecisionType, reason: str = "",
               final_qty: int | None = None, reviewer: str = "planner") -> Decision:
        """Validate and persist a decision.

        Raises:
            ValueError: with a plain-language message when the decision is invalid.
        """
        entry = build_decision(result, decision, reason, final_qty, reviewer)
        row = entry.model_dump(mode="json")
        with self.engine.begin() as conn:
            conn.execute(insert(decisions), [row])
        return entry

    def load(self, item_id: str | None = None) -> list[Decision]:
        """Decisions, oldest first."""
        query = select(decisions).order_by(decisions.c.decided_at)
        if item_id:
            query = query.where(decisions.c.item_id == item_id)
        with self.engine.connect() as conn:
            return [Decision.model_validate(dict(r._mapping)) for r in conn.execute(query)]

    def latest_by_item(self) -> dict[str, Decision]:
        """Most recent decision per item."""
        return {d.item_id: d for d in self.load()}


class JobStatus(StrEnum):
    """Lifecycle of a background pipeline job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class Job(BaseModel):
    """A batch of pipeline runs executed in the background."""

    job_id: str
    status: JobStatus
    total: int
    done: int = 0
    item_ids: list[str]
    run_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def finished(self) -> bool:
        """Whether the job reached a terminal state."""
        return self.status in (JobStatus.SUCCEEDED, JobStatus.PARTIAL, JobStatus.FAILED)


class JobStore:
    """Job rows updated by the worker and polled by clients."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        create_schema(engine)

    def create(self, item_ids: list[str]) -> Job:
        """Register a queued job."""
        job = Job(job_id=f"job-{uuid.uuid4().hex[:10]}", status=JobStatus.QUEUED,
                  total=len(item_ids), item_ids=item_ids, created_at=utc_now())
        with self.engine.begin() as conn:
            conn.execute(insert(jobs), [{
                "job_id": job.job_id, "status": job.status.value, "total": job.total,
                "done": 0, "item_ids": json.dumps(item_ids), "run_ids": "[]", "errors": "[]",
                "created_at": _iso(job.created_at)}])
        return job

    def get(self, job_id: str) -> Job | None:
        """One job by id."""
        with self.engine.connect() as conn:
            row = conn.execute(select(jobs).where(jobs.c.job_id == job_id)).first()
        if row is None:
            return None
        data = dict(row._mapping)
        for key in ("item_ids", "run_ids", "errors"):
            data[key] = json.loads(data[key] or "[]")
        return Job.model_validate(data)

    def list(self, limit: int = 20) -> list[Job]:
        """Most recent jobs first."""
        with self.engine.connect() as conn:
            ids = conn.execute(select(jobs.c.job_id).order_by(jobs.c.created_at.desc())
                               .limit(limit)).scalars().all()
        return [job for job in (self.get(i) for i in ids) if job is not None]

    def _update(self, job_id: str, **values: Any) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(jobs).where(jobs.c.job_id == job_id).values(**values))

    def mark_running(self, job_id: str) -> None:
        """Worker picked the job up."""
        self._update(job_id, status=JobStatus.RUNNING.value, started_at=_iso(utc_now()))

    def record_progress(self, job_id: str, run_id: str | None = None,
                        error: str | None = None) -> None:
        """One item finished (successfully with ``run_id`` or with ``error``)."""
        job = self.get(job_id)
        if job is None:
            return
        values: dict[str, Any] = {"done": job.done + 1}
        if run_id:
            values["run_ids"] = json.dumps([*job.run_ids, run_id])
        if error:
            values["errors"] = json.dumps([*job.errors, error])
        self._update(job_id, **values)

    def finish(self, job_id: str, error: str | None = None) -> Job | None:
        """Set the terminal status from the per-item outcomes."""
        job = self.get(job_id)
        if job is None:
            return None
        errors = [*job.errors, error] if error else job.errors
        if not errors:
            status = JobStatus.SUCCEEDED
        elif job.run_ids:
            status = JobStatus.PARTIAL
        else:
            status = JobStatus.FAILED
        self._update(job_id, status=status.value, errors=json.dumps(errors),
                     finished_at=_iso(utc_now()))
        return self.get(job_id)
