from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

from app.config import Settings
from app.preprocessing.profile import column_skew


@dataclass
class ClusteringResult:
    labels: np.ndarray
    model: Any
    scaler: StandardScaler
    pca: PCA | None
    power_transformer: PowerTransformer | None
    log1p_columns: list[str]
    feature_names: list[str]
    used_k: int
    metrics: dict[str, float]
    decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "used_k": self.used_k,
            "metrics": self.metrics,
            "decisions": self.decisions,
            "feature_names": self.feature_names,
            "has_pca": self.pca is not None,
            "log1p_columns": self.log1p_columns,
        }


class ClusteringTrainer:
    """
    Chooses transforms (log / Yeo-Johnson), optional PCA, k via silhouette (when sample size allows),
    and fits KMeans or GaussianMixture based on settings.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def fit(self, customer_df: pd.DataFrame, feature_cols: list[str]) -> ClusteringResult:
        decisions: list[str] = []
        X_raw = customer_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = X_raw.values.copy()

        # Adaptive: heavy-tailed positives -> log1p
        log_cols: set[str] = set()
        for j, col in enumerate(feature_cols):
            if col in ("RecencyDays", "TenureDays"):
                continue
            sk = column_skew(pd.Series(X[:, j]))
            if sk > self.settings.log_transform_skew and np.nanmin(X[:, j]) >= 0:
                X[:, j] = np.log1p(X[:, j])
                log_cols.add(col)
        if log_cols:
            decisions.append(f"log1p_columns={sorted(log_cols)}")

        # Optional Yeo-Johnson if still very skewed overall
        overall_skew = float(np.mean([abs(column_skew(pd.Series(X[:, j]))) for j in range(X.shape[1])]))
        power_transformer: PowerTransformer | None = None
        if overall_skew > 2.0:
            power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            X = power_transformer.fit_transform(X)
            decisions.append("yeo_johnson_power_transform")

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca: PCA | None = None
        X_final = Xs
        if Xs.shape[1] >= self.settings.pca_min_features:
            pca = PCA(n_components=self.settings.pca_variance_ratio, random_state=self.settings.random_state)
            X_final = pca.fit_transform(Xs)
            decisions.append(
                f"pca_n_components={X_final.shape[1]}_var_explained={float(np.sum(pca.explained_variance_ratio_)):.4f}"
            )

        n = X_final.shape[0]
        k_min = max(self.settings.k_min, 2)
        k_cap = max(2, n // max(self.settings.min_rows_per_cluster_heuristic, 1))
        k_max = min(self.settings.k_max, k_cap, max(2, n - 1))

        if k_max < k_min:
            k_max = k_min

        best_k = k_min
        best_score = -1.0

        if n >= self.settings.min_samples_for_silhouette:
            for k in range(k_min, k_max + 1):
                km = KMeans(
                    n_clusters=k,
                    random_state=self.settings.random_state,
                    n_init="auto",
                )
                labels = km.fit_predict(X_final)
                sil = float(silhouette_score(X_final, labels))
                if sil > best_score:
                    best_score = sil
                    best_k = k
            decisions.append(f"k_selected_by_silhouette={best_k}")
        else:
            best_k = min(k_max, max(k_min, min(4, k_max)))
            decisions.append(f"k_fallback_small_n={best_k}")

        if self.settings.cluster_algorithm == "kmeans":
            model = KMeans(
                n_clusters=best_k,
                random_state=self.settings.random_state,
                n_init="auto",
            )
            labels = model.fit_predict(X_final)
        else:
            model = GaussianMixture(
                n_components=best_k,
                random_state=self.settings.random_state,
                covariance_type="full",
            )
            labels = model.fit_predict(X_final)

        final_metrics = {
            "silhouette": float(silhouette_score(X_final, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X_final, labels)),
            "davies_bouldin": float(davies_bouldin_score(X_final, labels)),
        }

        return ClusteringResult(
            labels=labels,
            model=model,
            scaler=scaler,
            pca=pca,
            power_transformer=power_transformer,
            log1p_columns=sorted(log_cols),
            feature_names=list(feature_cols),
            used_k=best_k,
            metrics=final_metrics,
            decisions=decisions,
        )
