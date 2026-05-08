from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration; override via environment variables with prefix `RETAIL_`."""

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Folder 05_customer_transaction (parent of app/).",
    )
    transaction_csv: Path | None = Field(
        default=None,
        description="CSV path; default project_root/dataset/transaction_data.csv",
    )
    artifacts_dir: Path | None = Field(
        default=None,
        description="Models and parquet; default project_root/app/artifacts",
    )
    insights_filename: str = "customer_insights.parquet"
    encoding: str = "ISO-8859-1"

    reference_date: str | None = Field(
        default=None,
        description="ISO date for recency; default max invoice date in data.",
    )
    random_state: int = 42

    # Cleaning thresholds (adaptive rules use these as bounds)
    min_customer_id_coverage: float = 0.5
    anomaly_stock_min_digits: int = 2
    duplicate_row_action: Literal["drop", "keep"] = "drop"

    # Feature / modeling
    cluster_algorithm: Literal["kmeans", "gmm"] = "kmeans"
    k_min: int = 2
    k_max: int = 10
    min_samples_for_silhouette: int = 40
    min_rows_per_cluster_heuristic: int = 25
    pca_min_features: int = 8
    pca_variance_ratio: float = 0.92
    log_transform_skew: float = 1.25

    # Recommender
    top_products_per_segment: int = 30
    recommendations_per_customer: int = 8

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Evaluation / quality monitoring (soft-report)
    eval_enabled: bool = True
    eval_min_silhouette: float = 0.20
    eval_max_davies_bouldin: float = 2.0
    eval_min_cluster_fraction: float = 0.03
    eval_row_count_warn_pct: float = 25.0
    eval_n_customers_warn_pct: float = 25.0
    eval_psi_warn: float = 0.20
    eval_segment_distribution_warn_pct: float = 15.0
    eval_recommendation_coverage_warn: float = 0.95
    eval_score_drift_warn_pct: float = 20.0
    eval_segment_change_rate_warn: float = 0.30
    eval_history_keep_runs: int = 60

    def resolved_transaction_csv(self) -> Path:
        if self.transaction_csv is not None:
            return Path(self.transaction_csv)
        return self.project_root / "dataset" / "transaction_data.csv"

    def resolved_artifacts_dir(self) -> Path:
        if self.artifacts_dir is not None:
            return Path(self.artifacts_dir)
        return Path(__file__).resolve().parent / "artifacts"

    def insights_path(self) -> Path:
        return self.resolved_artifacts_dir() / self.insights_filename

    def evaluation_dir(self) -> Path:
        return self.resolved_artifacts_dir() / "evaluation"

    def evaluation_report_path(self) -> Path:
        return self.evaluation_dir() / "evaluation_report.json"

    def evaluation_history_dir(self) -> Path:
        return self.evaluation_dir() / "history"


def get_settings() -> Settings:
    return Settings()
