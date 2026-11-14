"""Pipeline service: runs agents, what-if re-plans and builds recommendation views."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd

from src.agents.inventory_agent import SUPPORTED_OVERRIDES
from src.agents.llm_client import LLMClient
from src.agents.schemas import InventoryAction, PipelineResult
from src.config.settings import Settings
from src.data.cleaning import clean_item_sales
from src.db.repository import SQLiteRepository
from src.db.stores import RUN_KIND_PIPELINE, RUN_KIND_WHAT_IF, RunStore, SQLDecisionStore
from src.orchestration.decisions import DecisionType
from src.orchestration.runner import PipelineRunner

ORDER_ACTIONS = {InventoryAction.REORDER, InventoryAction.EXPEDITE}
T = TypeVar("T")


class UnknownItemError(KeyError):
    """The requested item does not exist in the database."""


class UnknownRunError(KeyError):
    """The requested run id does not exist."""


def validate_overrides(overrides: dict[str, float] | None) -> dict[str, float]:
    """Reject unknown what-if keys before any agent runs.

    Raises:
        ValueError: listing the unsupported keys.
    """
    overrides = overrides or {}
    unknown = sorted(set(overrides) - SUPPORTED_OVERRIDES)
    if unknown:
        raise ValueError(f"Unsupported overrides: {unknown}. "
                         f"Supported: {sorted(SUPPORTED_OVERRIDES)}")
    return overrides


class PipelineService:
    """Thread-safe façade over :class:`PipelineRunner` with persistence.

    Agent runs are serialised with a lock: model fitting is CPU-bound and the
    LangGraph graphs share agent instances, so one run at a time keeps results
    deterministic while reads (latest runs, recommendations) stay concurrent.
    """

    def __init__(self, settings: Settings, repository: SQLiteRepository, run_store: RunStore,
                 decision_store: SQLDecisionStore, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.repository = repository
        self.run_store = run_store
        self.decision_store = decision_store
        self.runner = PipelineRunner(settings, repository=repository, llm=llm,
                                     run_store=run_store)
        self._lock = threading.Lock()

    @property
    def llm_description(self) -> str:
        """Configured LLM provider chain."""
        return self.runner.llm_description

    def _locked(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` under the lock, surfacing agent ``ValueError``s unwrapped."""
        with self._lock:
            try:
                return fn()
            except RuntimeError as exc:
                if isinstance(exc.__cause__, ValueError):
                    raise exc.__cause__ from None
                raise

    def _require_item(self, item_id: str) -> None:
        if not self.repository.has_item(item_id):
            raise UnknownItemError(item_id)

    def run_item(self, item_id: str, overrides: dict[str, float] | None = None,
                 persist: bool = True) -> PipelineResult:
        """Run all four agents for one item and store the result.

        Raises:
            UnknownItemError: if the item does not exist.
            ValueError: if an override is invalid.
        """
        self._require_item(item_id)
        overrides = validate_overrides(overrides)
        result = self._locked(lambda: self.runner.run_item(item_id, overrides or None))
        if persist:
            self.run_store.save(result)
        return result

    def history(self, item_id: str) -> pd.DataFrame:
        """Cleaned daily history with imputation and stockout flags."""
        try:
            raw = self.repository.item_sales(item_id)
        except KeyError:
            raise UnknownItemError(item_id) from None
        clean, _ = clean_item_sales(raw)
        return clean

    def forecast(self, item_id: str) -> dict[str, Any]:
        """Data pipeline plus Forecast and Risk agents (not persisted)."""
        self._require_item(item_id)
        return self._locked(lambda: self.runner.forecast_item(item_id))

    def get_run(self, run_id: str) -> PipelineResult:
        """One stored run."""
        result = self.run_store.get(run_id)
        if result is None:
            raise UnknownRunError(run_id)
        return result

    def latest(self, item_ids: list[str] | None = None,
               kind: str = RUN_KIND_PIPELINE) -> dict[str, PipelineResult]:
        """Latest stored run per item."""
        return self.run_store.latest_by_item(kind=kind, item_ids=item_ids)

    def latest_for(self, item_id: str) -> PipelineResult:
        """Latest pipeline run of one item."""
        result = self.latest([item_id]).get(item_id)
        if result is None:
            raise UnknownRunError(f"No pipeline run for {item_id}")
        return result

    def what_if(self, overrides: dict[str, float], run_id: str | None = None,
                item_id: str | None = None) -> PipelineResult:
        """Re-plan inventory for new assumptions, reusing a stored forecast.

        The base run is ``run_id`` or the latest pipeline run of ``item_id``.
        """
        if run_id:
            base = self.get_run(run_id)
        elif item_id:
            base = self.latest_for(item_id)
        else:
            raise ValueError("Provide run_id or item_id.")
        overrides = validate_overrides(overrides)
        result = self._locked(lambda: self.runner.what_if(base, overrides))
        self.run_store.save(result)
        return result

    def latest_what_if(self, item_id: str) -> PipelineResult | None:
        """Most recent what-if re-plan of an item."""
        return self.latest([item_id], kind=RUN_KIND_WHAT_IF).get(item_id)

    def recommendations(self, item_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Flat recommendation rows (latest run + product + latest decision)."""
        decisions = self.decision_store.latest_by_item()
        rows = []
        for r in self.latest(item_ids).values():
            rec = r.critic.recommendation
            decision = decisions.get(r.item_id)
            if decision is not None and decision.run_id != r.run_id:
                decision = None
            product = self.repository.product(r.item_id)
            rows.append({
                "run_id": r.run_id, "item_id": r.item_id, "product_name": product.name,
                "department": product.department, "created_at": r.created_at.isoformat(),
                "action": rec.action.value, "quantity": rec.quantity,
                "order_value": rec.order_value, "risk_level": r.risk.risk_level.value,
                "confidence": r.risk.confidence_score,
                "stockout_probability": r.inventory.stockout_probability,
                "days_of_cover": r.inventory.days_of_cover, "model": r.forecast.model_used,
                "wape": r.forecast.selected_metrics.wape, "verdict": r.critic.verdict.value,
                "revision_count": r.revision_count, "headline": rec.headline,
                "decision": decision.decision.value if decision else "PENDING",
                "final_qty": decision.final_qty if decision else None,
            })
        return rows

    def po_draft(self) -> list[dict[str, Any]]:
        """Purchase-order lines from the latest non-rejected decisions with quantity > 0."""
        latest = self.latest()
        lines = []
        for d in self.decision_store.latest_by_item().values():
            if d.decision == DecisionType.REJECTED or d.final_qty <= 0:
                continue
            product = self.repository.product(d.item_id)
            run = latest.get(d.item_id)
            unit_cost = run.inventory.params.unit_cost if run else None
            lines.append({
                "item_id": d.item_id, "product": product.name, "department": product.department,
                "quantity": d.final_qty, "unit_cost": unit_cost,
                "line_value": round(unit_cost * d.final_qty, 2) if unit_cost else None,
                "decision": d.decision.value, "reviewer": d.reviewer, "reason": d.reason,
                "run_id": d.run_id, "decided_at": d.decided_at.isoformat(),
            })
        return lines
