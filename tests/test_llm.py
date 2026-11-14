"""Tests for grounding checks and the LLM client chain (no network)."""

from __future__ import annotations

from src.agents.grounding import check_grounding, collect_fact_numbers, extract_numbers
from src.agents.llm_client import (
    FallbackLLMClient,
    LLMClient,
    TemplateLLMClient,
    build_llm_client,
    describe_llm,
)
from src.config.settings import LLMSettings


class StubClient(LLMClient):
    def __init__(self, provider: str, reply: str | None) -> None:
        self.provider = provider
        self.reply = reply

    def complete(self, system: str, user: str) -> str | None:
        return self.reply


def test_extract_numbers_skips_dates_and_skus() -> None:
    text = "FOODS_3_090 runs out on 2016-05-28; order 1,200 units (12.5 days, 95%)."
    assert extract_numbers(text) == ["1,200", "12.5", "95"]


def test_collect_fact_numbers_recursive() -> None:
    facts = {"a": 1, "b": [2.5, {"c": 3}], "d": "WAPE 18%", "flag": True}
    assert sorted(collect_fact_numbers(facts)) == [1, 2.5, 3, 18]


def test_grounding_accepts_rounding_and_percent() -> None:
    facts = {"qty": 216, "p": 0.0948, "ss": 28.81}
    ok, bad = check_grounding("Order 216 units; stockout risk 9%; safety stock 29.", facts)
    assert ok and not bad


def test_grounding_rejects_invented_numbers() -> None:
    ok, bad = check_grounding("Order 500 units now.", {"qty": 216})
    assert not ok and bad == ["500"]


def test_build_client_none_and_missing_keys() -> None:
    assert isinstance(build_llm_client(LLMSettings(provider="none")), TemplateLLMClient)
    assert isinstance(build_llm_client(LLMSettings(provider="openrouter")), TemplateLLMClient)


def test_build_client_chain_order() -> None:
    cfg = LLMSettings(provider="cohere", openrouter_api_key="k1", cohere_api_key="k2")
    client = build_llm_client(cfg)
    assert isinstance(client, FallbackLLMClient)
    assert describe_llm(client) == "cohere -> openrouter"
    single = build_llm_client(cfg.model_copy(update={"enable_fallback": False}))
    assert describe_llm(single) == "cohere"


def test_fallback_client_uses_next_provider() -> None:
    chain = FallbackLLMClient([StubClient("a", None), StubClient("b", "hello")])
    assert chain.complete("s", "u") == "hello"
    assert chain.last_provider == "b"
    empty = FallbackLLMClient([StubClient("a", None)])
    assert empty.complete("s", "u") is None and empty.last_provider is None
