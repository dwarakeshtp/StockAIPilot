"""Application services shared by the FastAPI routers (business logic, no HTTP)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine

from src.agents.llm_client import LLMClient
from src.config.settings import Settings
from src.db.engine import get_engine
from src.db.repository import SQLiteRepository
from src.db.stores import JobStore, RunStore, SQLDecisionStore
from src.services.decisions import DecisionService
from src.services.evaluation import EvaluationService
from src.services.jobs import JobService
from src.services.pipeline import PipelineService


@dataclass
class Services:
    """Everything the API needs, built once per process."""

    settings: Settings
    engine: Engine
    repository: SQLiteRepository
    pipeline: PipelineService
    jobs: JobService
    decisions: DecisionService
    evaluation: EvaluationService

    @classmethod
    def build(cls, settings: Settings, engine: Engine | None = None,
              llm: LLMClient | None = None) -> Services:
        """Wire repository, stores and services on one engine."""
        engine = engine or get_engine(settings.database.url)
        repository = SQLiteRepository(settings, engine)
        runs, decisions = RunStore(engine), SQLDecisionStore(engine)
        pipeline = PipelineService(settings, repository, runs, decisions, llm)
        return cls(
            settings=settings, engine=engine, repository=repository, pipeline=pipeline,
            jobs=JobService(pipeline, JobStore(engine)),
            decisions=DecisionService(decisions, runs),
            evaluation=EvaluationService(settings, repository, runs),
        )

    def close(self) -> None:
        """Stop the job worker."""
        self.jobs.shutdown(wait=False)
