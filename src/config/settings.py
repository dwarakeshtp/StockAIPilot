"""Typed configuration for the whole application.

Configuration is read from ``config/settings.yaml`` and validated with pydantic.
Secrets and LLM provider choices are taken from environment variables (or a
``.env`` file) so they never live in version control.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "settings.yaml"

LLMProvider = Literal["openrouter", "cohere", "none"]


class DataSettings(BaseModel):
    """Dataset location and scope (store, category, items, history window)."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    simulated_dir: Path = Path("data/simulated")
    store_id: str = "CA_1"
    state_id: str = "CA"
    category: str = "FOODS"
    n_items: int = Field(default=12, ge=1, le=200)
    history_days: int = Field(default=730, ge=120)
    item_ids: list[str] = Field(default_factory=list)
    allow_synthetic_fallback: bool = True
    random_seed: int = 42

    @property
    def sales_path(self) -> Path:
        """Path of the processed long-format sales parquet."""
        return self.processed_dir / "sales_subset.parquet"

    @property
    def future_calendar_path(self) -> Path:
        """Path of the calendar rows following the last sales date."""
        return self.processed_dir / "calendar_future.parquet"

    @property
    def inventory_params_path(self) -> Path:
        """Path of the simulated inventory parameters CSV."""
        return self.simulated_dir / "inventory_params.csv"


class ForecastSettings(BaseModel):
    """Forecasting horizon, backtesting and model hyper-parameters."""

    horizon: int = Field(default=7, ge=1, le=28)
    backtest_folds: int = Field(default=4, ge=1, le=12)
    interval_width: float = Field(default=0.8, gt=0.5, lt=1.0)
    seasonal_period: int = 7
    arima_fit_window: int = Field(default=365, ge=60)
    arima_order_grid: list[tuple[int, int, int]] = Field(
        default_factory=lambda: [(1, 0, 1), (2, 0, 1), (1, 1, 1)]
    )
    arima_seasonal_order: tuple[int, int, int] = (1, 0, 1)
    prophet_uncertainty_samples: int = 300
    selection_metric: Literal["mae", "rmse", "wape"] = "mae"


class RiskWeights(BaseModel):
    """Weights of the confidence-score components (should sum to 1)."""

    accuracy: float = 0.4
    precision: float = 0.3
    agreement: float = 0.2
    data_quality: float = 0.1


class RiskSettings(BaseModel):
    """Anomaly detection and confidence-scoring thresholds."""

    lookback_days: int = 90
    zscore_window: int = 28
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 3.0
    spike_ratio: float = 1.5
    recent_anomaly_days: int = 14
    interval_width_cap: float = 2.0
    weights: RiskWeights = Field(default_factory=RiskWeights)
    high_threshold: float = 45
    low_threshold: float = 70
    uncertainty_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"LOW": 1.0, "MEDIUM": 1.2, "HIGH": 1.4}
    )


class InventorySettings(BaseModel):
    """Inventory policy parameters."""

    service_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    review_period_days: int = Field(default=7, ge=1)
    ordering_cost: float = 25.0


class CriticSettings(BaseModel):
    """Validation thresholds used by the Critic Agent."""

    max_revisions: int = Field(default=1, ge=0, le=3)
    model_disagreement_threshold: float = 0.35
    wape_threshold: float = 0.6
    max_days_of_cover: float = 30
    safety_stock_max_lt_multiple: float = 1.5
    low_confidence_threshold: float = 45
    large_order_days: float = 14
    min_data_quality_score: float = 0.8
    high_stockout_probability: float = 0.5


class LLMSettings(BaseModel):
    """LLM provider configuration (secrets come from the environment)."""

    provider: LLMProvider = "none"
    temperature: float = 0.2
    timeout_seconds: int = 30
    max_tokens: int = 400
    narrate_all_agents: bool = True
    enable_fallback: bool = True
    openrouter_api_key: str | None = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    cohere_api_key: str | None = None
    cohere_model: str = "command-a-03-2025"

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class OutputSettings(BaseModel):
    """Locations of generated artefacts."""

    root: Path = Path("outputs")

    @property
    def traces_dir(self) -> Path:
        """Directory holding one JSON trace per pipeline run."""
        return self.root / "traces"

    @property
    def logs_dir(self) -> Path:
        """Directory holding structured JSON logs."""
        return self.root / "logs"

    @property
    def runs_dir(self) -> Path:
        """Directory holding serialised pipeline results."""
        return self.root / "runs"

    @property
    def evaluation_dir(self) -> Path:
        """Directory holding evaluation artefacts."""
        return self.root / "evaluation"

    @property
    def decisions_file(self) -> Path:
        """JSONL file with human approval decisions."""
        return self.root / "decisions.jsonl"


class Settings(BaseModel):
    """Root settings object aggregating every section."""

    data: DataSettings = Field(default_factory=DataSettings)
    forecast: ForecastSettings = Field(default_factory=ForecastSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    inventory: InventorySettings = Field(default_factory=InventorySettings)
    critic: CriticSettings = Field(default_factory=CriticSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    outputs: OutputSettings = Field(default_factory=OutputSettings)

    def resolve_paths(self, root: Path = PROJECT_ROOT) -> Settings:
        """Return a copy where every relative path is anchored at ``root``."""
        data = self.data.model_copy(
            update={
                "raw_dir": _anchor(self.data.raw_dir, root),
                "processed_dir": _anchor(self.data.processed_dir, root),
                "simulated_dir": _anchor(self.data.simulated_dir, root),
            }
        )
        outputs = self.outputs.model_copy(update={"root": _anchor(self.outputs.root, root)})
        return self.model_copy(update={"data": data, "outputs": outputs})

    def ensure_output_dirs(self) -> None:
        """Create every output directory if it does not exist."""
        for path in (
            self.outputs.traces_dir,
            self.outputs.logs_dir,
            self.outputs.runs_dir,
            self.outputs.evaluation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _anchor(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


_ENV_TO_LLM_FIELD: dict[str, str] = {
    "LLM_PROVIDER": "provider",
    "LLM_ENABLE_FALLBACK": "enable_fallback",
    "OPENROUTER_API_KEY": "openrouter_api_key",
    "OPENROUTER_MODEL": "openrouter_model",
    "COHERE_API_KEY": "cohere_api_key",
    "COHERE_MODEL": "cohere_model",
}


def _llm_env_overrides() -> dict[str, str]:
    overrides: dict[str, str] = {}
    for env_name, field_name in _ENV_TO_LLM_FIELD.items():
        value = os.getenv(env_name)
        if value is not None and value.strip() != "":
            overrides[field_name] = value.strip()
    return overrides


def load_settings(config_path: Path | None = None, load_env: bool = True) -> Settings:
    """Load settings from YAML, apply environment overrides and anchor paths.

    Args:
        config_path: Optional YAML path; defaults to ``config/settings.yaml``.
        load_env: Whether to read a ``.env`` file from the project root.

    Returns:
        A fully validated :class:`Settings` instance with absolute paths.
    """
    if load_env:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    path = config_path or DEFAULT_CONFIG_PATH
    raw: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    llm_section = {**raw.get("llm", {}), **_llm_env_overrides()}
    raw["llm"] = llm_section
    return Settings.model_validate(raw).resolve_paths()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return load_settings()
