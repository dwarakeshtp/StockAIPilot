"""High-level façade used by scripts and the API services to run the agent pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

from src.agents.llm_client import LLMClient, describe_llm
from src.agents.schemas import PipelineResult
from src.config.settings import Settings
from src.data.repository import Repository
from src.db.engine import get_engine
from src.db.stores import RunStore
from src.logging.logger import bind_context, clear_context, get_logger
from src.orchestration.graph import (
    PipelineDependencies,
    build_forecast_graph,
    build_pipeline_graph,
    build_what_if_graph,
)
from src.orchestration.tracing import save_trace
from src.utils.ids import new_run_id, utc_now

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class PipelineRunner:
    """Runs the LangGraph pipeline per item; traces go to files, results to the RunStore."""

    def __init__(
        self,
        settings: Settings,
        repository: Repository | None = None,
        llm: LLMClient | None = None,
        run_store: RunStore | None = None,
    ) -> None:
        self.settings = settings
        self.deps = PipelineDependencies.build(settings, repository, llm)
        self.graph = build_pipeline_graph(self.deps)
        self.what_if_graph = build_what_if_graph(self.deps)
        self.forecast_graph = build_forecast_graph(self.deps)
        if run_store is not None:
            self.__dict__["run_store"] = run_store
        settings.ensure_output_dirs()

    @cached_property
    def run_store(self) -> RunStore:
        """Store for pipeline results (the configured database by default)."""
        return RunStore(get_engine(self.settings.database.url))

    @property
    def repository(self) -> Repository:
        """Data repository used by the pipeline."""
        return self.deps.repository

    @property
    def llm_description(self) -> str:
        """Human-readable LLM provider chain."""
        return describe_llm(self.deps.llm)

    def run_item(self, item_id: str, overrides: dict[str, float] | None = None) -> PipelineResult:
        """Run the full graph for one item, save its trace and return the result."""
        run_id = new_run_id()
        bind_context(run_id=run_id, item_id=item_id)
        try:
            state = self.graph.invoke(
                {"run_id": run_id, "item_id": item_id, "overrides": overrides or {},
                 "revision_count": 0, "revision_request": None, "trace": []},
                config={"recursion_limit": 30},
            )
        finally:
            clear_context()
        result = PipelineResult(
            run_id=run_id,
            item_id=item_id,
            product_name=state.get("product_name"),
            department=state.get("department"),
            created_at=utc_now(),
            data_source=state["data_source"],
            data_quality=state["data_quality"],
            forecast=state["forecast"],
            risk=state["risk"],
            inventory=state["inventory"],
            critic=state["critic"],
            revision_count=state.get("revision_count", 0),
            overrides=overrides or {},
            trace=state["trace"],
        )
        result.trace_path = str(save_trace(result, self.settings.outputs.traces_dir))
        return result

    def run_items(
        self, item_ids: Sequence[str] | None = None, progress: ProgressCallback | None = None
    ) -> list[PipelineResult]:
        """Run the pipeline for several items (all items by default)."""
        items = list(item_ids or self.repository.item_ids)
        results = []
        for idx, item in enumerate(items, start=1):
            if progress:
                progress(idx, len(items), item)
            results.append(self.run_item(item))
        return results

    def what_if(self, base: PipelineResult, overrides: dict[str, float]) -> PipelineResult:
        """Re-plan inventory and re-run the Critic under new assumptions.

        The forecast and risk assessment are reused; the Critic cannot request a
        forecast revision here (its revision budget is treated as spent).
        """
        run_id = new_run_id("whatif")
        state = self.what_if_graph.invoke(
            {
                "run_id": run_id,
                "item_id": base.item_id,
                "product_name": base.product_name,
                "overrides": overrides,
                "forecast": base.forecast,
                "risk": base.risk,
                "data_quality": base.data_quality,
                "params": self.repository.item_inventory_params(base.item_id),
                "history_max": float(self.repository.item_sales(base.item_id)["units"].max()),
                "revision_count": self.settings.critic.max_revisions,
                "trace": [],
            }
        )
        result = base.model_copy(
            update={
                "run_id": run_id,
                "created_at": utc_now(),
                "inventory": state["inventory"],
                "critic": state["critic"],
                "overrides": overrides,
                "trace": state["trace"],
            }
        )
        result.trace_path = str(save_trace(result, self.settings.outputs.traces_dir))
        return result

    def forecast_item(self, item_id: str) -> dict[str, Any]:
        """Run the data pipeline, Forecast and Risk agents only (no planning, not persisted)."""
        run_id = new_run_id("forecast")
        bind_context(run_id=run_id, item_id=item_id)
        try:
            state = self.forecast_graph.invoke(
                {"run_id": run_id, "item_id": item_id, "overrides": {}, "revision_count": 0,
                 "revision_request": None, "trace": []})
        finally:
            clear_context()
        return {"run_id": run_id, "item_id": item_id, "data_source": state["data_source"],
                "product_name": state.get("product_name"),
                "data_quality": state["data_quality"], "forecast": state["forecast"],
                "risk": state["risk"], "trace": state["trace"]}

    def save_results(self, results: Sequence[PipelineResult]) -> int:
        """Persist results in the run store; returns how many were saved."""
        return self.run_store.save_many(list(results))


def load_latest_results(settings: Settings) -> list[PipelineResult]:
    """Latest pipeline run per item from the configured database."""
    return list(RunStore(get_engine(settings.database.url)).latest_by_item().values())
