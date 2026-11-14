"""High-level façade used by scripts and the dashboard to run the agent pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from src.agents.llm_client import LLMClient, describe_llm
from src.agents.schemas import PipelineResult
from src.config.settings import Settings
from src.data.repository import DataRepository
from src.logging.logger import bind_context, clear_context, get_logger
from src.orchestration.graph import (
    PipelineDependencies,
    build_pipeline_graph,
    build_what_if_graph,
)
from src.orchestration.tracing import save_trace
from src.utils.ids import new_run_id, utc_now
from src.utils.io import read_json, write_json

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class PipelineRunner:
    """Runs the LangGraph pipeline per item and persists results and traces."""

    def __init__(
        self,
        settings: Settings,
        repository: DataRepository | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.deps = PipelineDependencies.build(settings, repository, llm)
        self.graph = build_pipeline_graph(self.deps)
        self.what_if_graph = build_what_if_graph(self.deps)
        settings.ensure_output_dirs()

    @property
    def repository(self) -> DataRepository:
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

    def save_results(self, results: Sequence[PipelineResult]) -> Path:
        """Write results to ``outputs/runs/latest.json`` (and a timestamped copy)."""
        payload = {
            "created_at": utc_now(),
            "data_source": results[0].data_source if results else None,
            "llm": self.llm_description,
            "results": [r.model_dump(mode="json") for r in results],
        }
        runs_dir = self.settings.outputs.runs_dir
        stamp = utc_now().strftime("%Y%m%dT%H%M%S")
        write_json(runs_dir / f"batch-{stamp}.json", payload)
        return write_json(runs_dir / "latest.json", payload)


def load_latest_results(settings: Settings) -> list[PipelineResult]:
    """Load the most recent saved batch (empty list if none)."""
    path = settings.outputs.runs_dir / "latest.json"
    if not path.exists():
        return []
    payload = read_json(path)
    return [PipelineResult.model_validate(r) for r in payload.get("results", [])]
