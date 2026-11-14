"""Tests for the product catalog and how product names flow through the pipeline."""

from __future__ import annotations

import re
from pathlib import Path

from src.agents.critic_agent import CriticAgent
from src.agents.grounding import check_grounding
from src.agents.llm_client import TemplateLLMClient
from src.config.settings import load_settings
from src.data.catalog import fallback_product, load_catalog
from src.orchestration.runner import PipelineRunner
from tests.factories import make_forecast, make_params, make_quality, make_risk

CATALOG = Path(__file__).resolve().parents[1] / "data" / "reference" / "product_catalog.csv"


def test_catalog_covers_processed_items_without_digits() -> None:
    catalog = load_catalog(CATALOG)
    assert len(catalog) == 12
    for product in catalog.values():
        assert product.name and product.department != "Unknown"
        assert not re.search(r"\d", product.name), product.name
    assert catalog["FOODS_3_090"].label == "Bananas (FOODS_3_090)"


def test_missing_catalog_and_unknown_item_fall_back(tmp_path: Path) -> None:
    assert load_catalog(tmp_path / "missing.csv") == {}
    product = fallback_product("FOODS_SYN_001")
    assert product.label == "FOODS_SYN_001"


def test_settings_resolve_catalog_path() -> None:
    assert load_settings(load_env=False).data.catalog_path == CATALOG


def test_repository_product_lookup(repository) -> None:
    item = repository.item_ids[0]
    assert repository.product(item).name == item  # synthetic items are not in the catalog


def test_critic_uses_product_name_in_headline_and_stays_grounded(settings) -> None:
    from src.agents.inventory_agent import InventoryAgent

    forecast, risk = make_forecast(), make_risk()
    inventory = InventoryAgent(settings).run(
        forecast.item_id, forecast, risk, make_params(on_hand=120))
    out = CriticAgent(settings, TemplateLLMClient()).run(
        forecast.item_id, forecast, risk, inventory, make_quality(), history_max=100,
        product_name="Roma Tomatoes")
    assert "Roma Tomatoes" in out.recommendation.headline
    assert "Roma Tomatoes" in out.explanation
    facts = CriticAgent(settings, TemplateLLMClient())._facts(
        forecast, risk, inventory, out.verdict, out.issues, "Roma Tomatoes")
    assert facts["product_name"] == "Roma Tomatoes"
    assert check_grounding(out.explanation, facts)[0]


def test_pipeline_result_label(settings, repository) -> None:
    result = PipelineRunner(settings, repository=repository).run_item(repository.item_ids[0])
    assert result.product_name is None
    assert result.label == result.item_id
    named = result.model_copy(update={"product_name": "Bananas"})
    assert named.label == f"Bananas ({result.item_id})"
