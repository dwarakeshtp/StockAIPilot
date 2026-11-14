"""Base class shared by all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from src.agents.grounding import check_grounding
from src.agents.llm_client import LLMClient, TemplateLLMClient
from src.agents.prompts import NARRATOR_SYSTEM, narrative_prompt
from src.config.settings import Settings
from src.logging.logger import get_logger


class BaseAgent(ABC):
    """Common agent plumbing: settings, logger and optional grounded LLM narration.

    Concrete agents expose a typed ``run(...)`` method returning a pydantic model.
    """

    name: ClassVar[str] = "base_agent"
    role: ClassVar[str] = "Agent"

    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.llm: LLMClient = llm or TemplateLLMClient()
        self.logger = get_logger(f"agents.{self.name}")
        self.last_llm_provider: str | None = None

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Execute the agent and return its typed pydantic output."""

    def narrate(self, task: str, facts: dict[str, Any]) -> str | None:
        """Ask the LLM for a short grounded narrative of ``facts``.

        Returns ``None`` when narration is disabled, the LLM is unavailable or the
        reply contains numbers not supported by ``facts``.
        """
        self.last_llm_provider = None
        if not self.settings.llm.narrate_all_agents or not self.llm.available:
            return None
        text = self.llm.complete(NARRATOR_SYSTEM, narrative_prompt(self.role, task, facts))
        if not text:
            return None
        passed, ungrounded = check_grounding(text, facts)
        if not passed:
            self.logger.warning("narrative_rejected_ungrounded", numbers=ungrounded[:10])
            return None
        self.last_llm_provider = self.llm.last_provider
        return text
