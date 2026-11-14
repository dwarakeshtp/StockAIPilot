"""Human-in-the-loop decisions on Critic recommendations.

Validation lives in :func:`build_decision`; persistence is either the SQL
:class:`src.db.stores.SQLDecisionStore` (API) or the JSONL :class:`DecisionStore`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from src.agents.schemas import InventoryAction, PipelineResult
from src.utils.ids import utc_now
from src.utils.io import append_jsonl, read_jsonl

DEFAULT_APPROVAL_REASON = "Approved as recommended"
MIN_REASON_LENGTH = 3


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
    reason: str = ""
    reviewer: str = "planner"
    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _consistent(self) -> Decision:
        self.reason = self.reason.strip()
        if self.decision == DecisionType.APPROVED and not self.reason:
            self.reason = DEFAULT_APPROVAL_REASON
        if len(self.reason) < MIN_REASON_LENGTH:
            raise ValueError(f"Please give a reason (at least {MIN_REASON_LENGTH} characters) "
                             f"when the decision is {self.decision.value}.")
        if self.decision == DecisionType.APPROVED and self.final_qty != self.recommended_qty:
            raise ValueError("APPROVED decisions must keep the recommended quantity.")
        if self.decision == DecisionType.REJECTED and self.final_qty != 0:
            raise ValueError("REJECTED decisions must have a final quantity of 0.")
        if self.decision == DecisionType.MODIFIED and self.final_qty == self.recommended_qty:
            raise ValueError("The final quantity equals the recommendation - choose APPROVED, "
                             "or change the quantity to modify it.")
        return self


def build_decision(
    result: PipelineResult,
    decision: DecisionType,
    reason: str = "",
    final_qty: int | None = None,
    reviewer: str = "planner",
) -> Decision:
    """Validate a planner decision on ``result``'s recommendation.

    Raises:
        ValueError: with a plain-language message when the decision is invalid.
    """
    rec = result.critic.recommendation
    qty = {DecisionType.APPROVED: rec.quantity, DecisionType.REJECTED: 0}.get(
        decision, final_qty if final_qty is not None else rec.quantity
    )
    try:
        return Decision(
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
    except ValidationError as exc:
        messages = [e["msg"].removeprefix("Value error, ") for e in exc.errors()]
        raise ValueError(" ".join(messages)) from None


class DecisionStore:
    """Append-only JSONL store of planner decisions (file-based alternative to SQL)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(
        self,
        result: PipelineResult,
        decision: DecisionType,
        reason: str = "",
        final_qty: int | None = None,
        reviewer: str = "planner",
    ) -> Decision:
        """Validate and persist a decision for ``result``.

        Raises:
            ValueError: with a plain-language message when the decision is invalid.
        """
        entry = build_decision(result, decision, reason, final_qty, reviewer)
        append_jsonl(self.path, entry)
        return entry

    def load(self) -> list[Decision]:
        """All recorded decisions (oldest first)."""
        return [Decision.model_validate(r) for r in read_jsonl(self.path)]

    def latest_by_item(self) -> dict[str, Decision]:
        """Most recent decision per item."""
        return {d.item_id: d for d in self.load()}
