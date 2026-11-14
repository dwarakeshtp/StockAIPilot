"""Decision service: planner approvals on stored recommendations."""

from __future__ import annotations

from src.db.stores import RunStore, SQLDecisionStore
from src.orchestration.decisions import Decision, DecisionType
from src.services.pipeline import UnknownRunError


class DecisionService:
    """Records decisions against a stored run (so the audit trail references real runs)."""

    def __init__(self, store: SQLDecisionStore, run_store: RunStore) -> None:
        self.store = store
        self.run_store = run_store

    def record(self, run_id: str, decision: DecisionType, reason: str = "",
               final_qty: int | None = None, reviewer: str = "planner") -> Decision:
        """Validate and store a decision.

        Raises:
            UnknownRunError: if ``run_id`` is not stored.
            ValueError: if the decision breaks a rule (plain-language message).
        """
        result = self.run_store.get(run_id)
        if result is None:
            raise UnknownRunError(run_id)
        return self.store.record(result, decision, reason, final_qty=final_qty,
                                 reviewer=reviewer or "planner")

    def list(self, item_id: str | None = None) -> list[Decision]:
        """Decision log, oldest first."""
        return self.store.load(item_id)

    def latest_by_item(self) -> dict[str, Decision]:
        """Most recent decision per item."""
        return self.store.latest_by_item()
