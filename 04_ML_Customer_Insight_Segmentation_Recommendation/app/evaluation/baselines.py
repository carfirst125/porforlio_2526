from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.config import Settings


@dataclass
class Baseline:
    """Snapshot of the previous successful pipeline run, loaded for drift comparison.

    All fields are optional; the first ever run yields an empty Baseline and
    every drift / regression check degrades gracefully to a single info entry.
    """

    meta: dict[str, Any] | None = None
    insights: pd.DataFrame | None = None
    run_id: str | None = None

    @property
    def has_meta(self) -> bool:
        return isinstance(self.meta, dict) and bool(self.meta)

    @property
    def has_insights(self) -> bool:
        return isinstance(self.insights, pd.DataFrame) and not self.insights.empty


def load_previous_baseline(settings: Settings) -> Baseline:
    """Load the previous run's `pipeline_meta.json` and `customer_insights.parquet`.

    Must be called *before* the new pipeline run overwrites these files.
    """

    meta: dict[str, Any] | None = None
    insights: pd.DataFrame | None = None
    run_id: str | None = None

    artifacts_dir = settings.resolved_artifacts_dir()
    meta_path = artifacts_dir / "pipeline_meta.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                run_id = str(meta.get("generated_at_utc") or "")
        except (OSError, json.JSONDecodeError):
            meta = None

    insights_path = settings.insights_path()
    if insights_path.is_file():
        try:
            insights = pd.read_parquet(insights_path)
        except Exception:
            insights = None

    return Baseline(meta=meta, insights=insights, run_id=run_id or None)
