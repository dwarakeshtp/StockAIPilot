"""Prompt templates for agent narratives and the Critic explanation."""

from __future__ import annotations

import json
from typing import Any

from src.utils.io import to_jsonable

NARRATOR_SYSTEM = (
    "You are an analyst in a retail inventory-planning team. You explain results "
    "produced by deterministic models. Rules: use ONLY the numbers in the FACTS JSON "
    "(you may round them); never invent numbers, dates or causes; be concise and "
    "plain-spoken; no markdown headings; no bullet lists."
)

CRITIC_SYSTEM = (
    "You are the Critic Agent in a multi-agent demand-forecasting system. You write the "
    "final explanation for a store inventory manager. Rules: use ONLY numbers present in "
    "the FACTS JSON (rounding allowed); never invent figures or causes; refer to the item "
    "by its product_name; mention the "
    "recommended action and quantity, the main reason, the key risks and any validation "
    "warnings; state that the decision requires human approval; 4-6 sentences; plain "
    "text only, no markdown."
)


def narrative_prompt(agent_role: str, task: str, facts: dict[str, Any]) -> str:
    """Build the user prompt for an agent narrative."""
    return (
        f"ROLE: {agent_role}\nTASK: {task}\n"
        f"Write 2-3 sentences.\nFACTS JSON:\n{json.dumps(to_jsonable(facts), indent=2)}"
    )


def critic_prompt(facts: dict[str, Any]) -> str:
    """Build the user prompt for the Critic explanation."""
    return (
        "Explain the final inventory recommendation for this item.\n"
        f"FACTS JSON:\n{json.dumps(to_jsonable(facts), indent=2)}"
    )
