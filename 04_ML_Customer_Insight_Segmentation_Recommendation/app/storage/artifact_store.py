from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from app.config import Settings
from app.modeling.cluster_model import ClusteringResult


class ArtifactStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dir = settings.resolved_artifacts_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def bundle_path(self) -> Path:
        return self.dir / "clustering_bundle.joblib"

    @property
    def meta_path(self) -> Path:
        return self.dir / "pipeline_meta.json"

    def save_cluster_bundle(self, result: ClusteringResult, algorithm: str) -> None:
        payload: dict[str, Any] = {
            "model": result.model,
            "scaler": result.scaler,
            "pca": result.pca,
            "power_transformer": result.power_transformer,
            "feature_names": result.feature_names,
            "log1p_columns": result.log1p_columns,
            "algorithm": algorithm,
            "k": result.used_k,
        }
        joblib.dump(payload, self.bundle_path)

    def save_meta(self, meta: dict) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    def load_cluster_bundle(self) -> dict[str, Any]:
        return joblib.load(self.bundle_path)
