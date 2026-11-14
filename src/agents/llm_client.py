"""LLM access layer (OpenRouter / Cohere / deterministic template fallback).

Both OpenRouter and Cohere expose OpenAI-compatible chat endpoints, so a single
``langchain-openai`` client class covers both. Every failure (missing key,
network error, rate limit) degrades gracefully: ``complete`` returns ``None``
and callers fall back to template text. The LLM never produces numbers that
feed calculations - it only phrases explanations of validated results.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from src.config.settings import LLMSettings
from src.logging.logger import get_logger

logger = get_logger(__name__)

COHERE_COMPAT_BASE_URL = "https://api.cohere.ai/compatibility/v1"


class LLMClient(ABC):
    """Minimal chat-completion interface used by agents."""

    provider: str = "abstract"

    @property
    def available(self) -> bool:
        """Whether the client can actually call a model."""
        return True

    @property
    def last_provider(self) -> str | None:
        """Provider that produced the most recent successful completion."""
        return self.provider

    @abstractmethod
    def complete(self, system: str, user: str) -> str | None:
        """Return the model's reply, or ``None`` if unavailable or failed."""


class TemplateLLMClient(LLMClient):
    """No-op client: always returns ``None`` so callers use templates."""

    provider = "template"

    @property
    def available(self) -> bool:
        """Template client never calls a model."""
        return False

    @property
    def last_provider(self) -> str | None:
        """No model is ever used."""
        return None

    def complete(self, system: str, user: str) -> str | None:
        """Always ``None``."""
        return None


class OpenAICompatibleClient(LLMClient):
    """Chat client for any OpenAI-compatible endpoint (OpenRouter, Cohere)."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.2,
        timeout_seconds: int = 30,
        max_tokens: int = 400,
    ) -> None:
        self.provider = provider
        self.model = model
        self._kwargs = {
            "api_key": api_key,
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "timeout": timeout_seconds,
            "max_tokens": max_tokens,
            "max_retries": 1,
        }
        self._chat = None
        self._cache: dict[str, str] = {}

    def _client(self):
        if self._chat is None:
            from langchain_openai import ChatOpenAI

            headers = {"X-Title": "Agentic Demand Forecasting POC"}
            self._chat = ChatOpenAI(default_headers=headers, **self._kwargs)
        return self._chat

    def complete(self, system: str, user: str) -> str | None:
        """Call the model; cache identical prompts; return ``None`` on any failure."""
        key = hashlib.sha256(f"{system}\n---\n{user}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            reply = self._client().invoke([SystemMessage(content=system),
                                           HumanMessage(content=user)])
            text = str(reply.content).strip()
        except Exception as exc:  # noqa: BLE001 - any provider failure must degrade gracefully
            logger.warning("llm_call_failed", provider=self.provider, model=self.model,
                           error=str(exc)[:300])
            return None
        if not text:
            return None
        self._cache[key] = text
        logger.info("llm_call_succeeded", provider=self.provider, model=self.model,
                    chars=len(text))
        return text


class FallbackLLMClient(LLMClient):
    """Try several clients in order and remember which one answered."""

    provider = "fallback"

    def __init__(self, clients: list[LLMClient]) -> None:
        self.clients = [c for c in clients if c.available]
        self._last: str | None = None

    @property
    def available(self) -> bool:
        """True if at least one underlying client can call a model."""
        return bool(self.clients)

    @property
    def last_provider(self) -> str | None:
        """Provider of the last successful completion."""
        return self._last

    @property
    def description(self) -> str:
        """Human-readable provider chain, e.g. ``openrouter -> cohere``."""
        return " -> ".join(c.provider for c in self.clients) or "template"

    def complete(self, system: str, user: str) -> str | None:
        """Return the first successful reply in the chain."""
        self._last = None
        for client in self.clients:
            text = client.complete(system, user)
            if text:
                self._last = client.provider
                return text
        return None


def build_llm_client(cfg: LLMSettings) -> LLMClient:
    """Build the client chain described by ``cfg``.

    The configured provider is tried first; when ``enable_fallback`` is set and
    the other provider has a key, it is tried second. Without any usable key a
    :class:`TemplateLLMClient` is returned.
    """
    def openrouter() -> LLMClient | None:
        if not cfg.openrouter_api_key:
            return None
        return OpenAICompatibleClient("openrouter", cfg.openrouter_api_key, cfg.openrouter_model,
                                      cfg.openrouter_base_url, cfg.temperature,
                                      cfg.timeout_seconds, cfg.max_tokens)

    def cohere() -> LLMClient | None:
        if not cfg.cohere_api_key:
            return None
        return OpenAICompatibleClient("cohere", cfg.cohere_api_key, cfg.cohere_model,
                                      COHERE_COMPAT_BASE_URL, cfg.temperature,
                                      cfg.timeout_seconds, cfg.max_tokens)

    if cfg.provider == "none":
        return TemplateLLMClient()
    order = [openrouter, cohere] if cfg.provider == "openrouter" else [cohere, openrouter]
    if not cfg.enable_fallback:
        order = order[:1]
    clients = [c for c in (builder() for builder in order) if c is not None]
    if not clients:
        logger.warning("llm_no_api_key", provider=cfg.provider)
        return TemplateLLMClient()
    return FallbackLLMClient(clients)


def describe_llm(client: LLMClient) -> str:
    """Short description of an LLM client for the UI and traces."""
    if isinstance(client, FallbackLLMClient):
        return client.description
    return client.provider
