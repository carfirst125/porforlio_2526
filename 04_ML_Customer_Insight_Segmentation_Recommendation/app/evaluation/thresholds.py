from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class EvalThresholds:
    """Thresholds bundle used by checks; all values come from `Settings`."""

    min_silhouette: float
    max_davies_bouldin: float
    min_cluster_fraction: float
    row_count_warn_pct: float
    n_customers_warn_pct: float
    psi_warn: float
    segment_distribution_warn_pct: float
    recommendation_coverage_warn: float
    score_drift_warn_pct: float
    segment_change_rate_warn: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "EvalThresholds":
        return cls(
            min_silhouette=settings.eval_min_silhouette,
            max_davies_bouldin=settings.eval_max_davies_bouldin,
            min_cluster_fraction=settings.eval_min_cluster_fraction,
            row_count_warn_pct=settings.eval_row_count_warn_pct,
            n_customers_warn_pct=settings.eval_n_customers_warn_pct,
            psi_warn=settings.eval_psi_warn,
            segment_distribution_warn_pct=settings.eval_segment_distribution_warn_pct,
            recommendation_coverage_warn=settings.eval_recommendation_coverage_warn,
            score_drift_warn_pct=settings.eval_score_drift_warn_pct,
            segment_change_rate_warn=settings.eval_segment_change_rate_warn,
        )
