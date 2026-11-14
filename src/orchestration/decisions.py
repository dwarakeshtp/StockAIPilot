"""Human-in-the-loop decisions on Critic recommendations (append-only JSONL)."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.agents.schemas import InventoryAction, PipelineResult
from src.utils.ids import utc_now
from src.utils.io import append_jsonl, read_jsonl


class DecisionType(StrEnum):
    """Planner decision on a recommendation."""

    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


class Decision(BaseModel):
    """A planner's decision on one recommendation."""

    decision_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    run_id: str
    item_id: str
    recommended_action: InventoryAction
    recommended_qty: int
    verdict: str
    decision: DecisionType
    final_qty: int = Field(ge=0)
    reason: str = Field(min_length=3)
    reviewer: str = "planner"
    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _consistent(self) -> Decision:
        if self.decision == DecisionType.APPROVED and self.final_qty != self.recommended_qty:
            raise ValueError("APPROVED decisions must keep the recommended quantity")
        if self.decision == DecisionType.REJECTED and self.final_qty != 0:
            raise ValueError("REJECTED decisions must have final_qty = 0")
        return self


class DecisionStore:
    """Append-only store of planner decisions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(
        self,
        result: PipelineResult,
        decision: DecisionType,
        reason: str,
        final_qty: int | None = None,
        reviewer: str = "planner",
    ) -> Decision:
        """Validate and persist a decision for ``result``."""
        rec = result.critic.recommendation
        qty = {DecisionType.APPROVED: rec.quantity, DecisionType.REJECTED: 0}.get(
            decision, final_qty if final_qty is not None else rec.quantity
        )
        entry = Decision(
            run_id=result.run_id,
            item_id=result.item_id,
            recommended_action=rec.action,
            recommended_qty=rec.quantity,
            verdict=result.critic.verdict.value,
            decision=decision,
            final_qty=qty,
            reason=reason,
            reviewer=reviewer,
        )
        append_jsonl(self.path, entry)
        return entry

    def load(self) -> list[Decision]:
        """All recorded decisions (oldest first)."""
        return [Decision.model_validate(r) for r in read_jsonl(self.path)]

    def latest_by_item(self) -> dict[str, Decision]:
        """Most recent decision per item."""
        return {d.item_id: d for d in self.load()}
