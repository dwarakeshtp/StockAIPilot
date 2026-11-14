"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import PROJECT_ROOT, Settings, load_settings


def test_load_settings_from_yaml_has_expected_defaults() -> None:
    settings = load_settings(load_env=False)
    assert settings.forecast.horizon == 7
    assert settings.data.store_id == "CA_1"
    assert settings.data.category == "FOODS"
    assert settings.data.processed_dir.is_absolute()
    assert str(settings.data.processed_dir).startswith(str(PROJECT_ROOT))


def test_env_overrides_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "OpenRouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model:free")
    settings = load_settings(load_env=False)
    assert settings.llm.provider == "openrouter"
    assert settings.llm.openrouter_model == "some/model:free"


def test_invalid_service_level_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"inventory": {"service_level": 1.5}})


def test_missing_yaml_falls_back_to_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "nope.yaml", load_env=False)
    assert settings.forecast.horizon == 7


def test_ensure_output_dirs(settings: Settings) -> None:
    settings.ensure_output_dirs()
    assert settings.outputs.traces_dir.exists()
    assert settings.outputs.runs_dir.exists()
