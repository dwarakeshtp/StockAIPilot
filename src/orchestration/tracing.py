"""Node tracing: timing, input/output summaries and reasoning for each step."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.agents.schemas import AgentTraceEntry, PipelineResult
from src.logging.logger import get_logger
from src.utils.ids import utc_now
from src.utils.io import to_jsonable, write_json

logger = get_logger(__name__)

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]
Summariser = Callable[[dict[str, Any]], dict[str, Any]]


def summarise_model(model: BaseModel | None, fields: tuple[str, ...]) -> dict[str, Any]:
    """Pick a few JSON-safe fields from a pydantic model."""
    if model is None:
        return {}
    data = model.model_dump(mode="json")
    return {f: data.get(f) for f in fields if f in data}


def traced_node(
    node: str,
    agent: str,
    fn: NodeFn,
    summarise_input: Summariser,
    summarise_output: Summariser,
    agent_obj: Any | None = None,
) -> NodeFn:
    """Wrap a node so it appends an :class:`AgentTraceEntry` to ``state['trace']``.

    Args:
        node: Graph node name.
        agent: Agent (or pipeline stage) responsible for the node.
        fn: Node function returning a partial state update.
        summarise_input: Builds a small dict describing the node input.
        summarise_output: Builds a small dict describing the node output (receives the update).
        agent_obj: Optional agent whose ``last_llm_provider`` is recorded.
    """

    def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        step = len(state.get("trace", [])) + 1
        started = utc_now()
        t0 = time.perf_counter()
        input_summary = to_jsonable(summarise_input(state))
        try:
            update = fn(state)
        except Exception as exc:
            logger.exception("node_failed", node=node, item_id=state.get("item_id"))
            raise RuntimeError(f"Node '{node}' failed: {exc}") from exc
        duration = (time.perf_counter() - t0) * 1000
        produced = next((v for v in update.values() if isinstance(v, BaseModel)), None)
        reasoning = list(getattr(produced, "reasoning", []) or [])
        entry = AgentTraceEntry(
            step=step,
            node=node,
            agent=agent,
            started_at=started,
            finished_at=utc_now(),
            duration_ms=round(duration, 1),
            status="success",
            input_summary=input_summary,
            output_summary=to_jsonable(summarise_output(update)),
            reasoning=reasoning,
            llm_provider=getattr(agent_obj, "last_llm_provider", None),
        )
        logger.info("node_completed", node=node, item_id=state.get("item_id"),
                    duration_ms=entry.duration_ms)
        return {**update, "trace": [entry]}

    wrapper.__name__ = f"traced_{node}"
    return wrapper


def save_trace(result: PipelineResult, traces_dir: Path) -> Path:
    """Persist the trace of one run as JSON and return its path."""
    payload = {
        "run_id": result.run_id,
        "item_id": result.item_id,
        "created_at": result.created_at,
        "data_source": result.data_source,
        "revision_count": result.revision_count,
        "overrides": result.overrides,
        "final_verdict": result.critic.verdict,
        "recommendation": result.critic.recommendation,
        "steps": result.trace,
    }
    return write_json(traces_dir / f"{result.run_id}.json", payload)
