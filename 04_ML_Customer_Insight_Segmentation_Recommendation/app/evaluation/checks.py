"""Individual quality checks grouped by category.

Each check returns one (or a list of) `CheckResult`. Categories:
- ``data_quality``       : raw transaction dataset health
- ``feature_quality``    : customer-level feature integrity and drift
- ``model_quality``      : clustering metrics and balance
- ``business_metrics``   : insight distributions and coverage
- ``regression``         : intersection comparison vs the previous run
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.evaluation.baselines import Baseline
from app.evaluation.thresholds import EvalThresholds
from app.features.customer_features import FeatureBuildReport
from app.modeling.cluster_model import ClusteringResult
from app.preprocessing.cleaning import CleaningReport
from app.preprocessing.profile import DatasetProfile

OK = "ok"
WARN = "warning"
FAIL = "failed"
INFO = "info"

REQUIRED_RAW_COLUMNS = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)


@dataclass
class CheckResult:
    name: str
    category: str
    status: str
    value: Any = None
    threshold: Any | None = None
    message: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = _json_safe(d["value"])
        d["threshold"] = _json_safe(d["threshold"])
        return d


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def _round(value: float, ndigits: int = 4) -> float:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return 0.0
    return float(round(float(value), ndigits))


def _pct_change(curr: float, prev: float) -> float:
    if prev in (0, 0.0) or prev is None or np.isnan(prev):
        return 0.0
    return float((curr - prev) / abs(prev) * 100.0)


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    e = pd.to_numeric(expected, errors="coerce").dropna().astype(float)
    a = pd.to_numeric(actual, errors="coerce").dropna().astype(float)
    if e.empty or a.empty:
        return 0.0
    quantiles = np.unique(np.quantile(e, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    e_counts, _ = np.histogram(e, bins=quantiles)
    a_counts, _ = np.histogram(a, bins=quantiles)
    e_total = e_counts.sum()
    a_total = a_counts.sum()
    if e_total == 0 or a_total == 0:
        return 0.0
    e_pct = np.clip(e_counts / e_total, 1e-6, None)
    a_pct = np.clip(a_counts / a_total, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def run_data_checks(
    raw: pd.DataFrame,
    profile: DatasetProfile,
    cleaning_report: CleaningReport,
    baseline: Baseline,
    thresholds: EvalThresholds,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    results.append(
        CheckResult(
            name="raw_schema_columns",
            category="data_quality",
            status=OK if not missing else FAIL,
            value=list(raw.columns),
            threshold=list(REQUIRED_RAW_COLUMNS),
            message="all required columns present" if not missing else f"missing columns: {missing}",
        )
    )

    results.append(
        CheckResult(
            name="row_count_raw",
            category="data_quality",
            status=OK if profile.n_rows > 0 else FAIL,
            value=int(profile.n_rows),
            message=f"{profile.n_rows:,} raw rows",
        )
    )

    results.append(
        CheckResult(
            name="row_count_after_cleaning",
            category="data_quality",
            status=OK if cleaning_report.rows_out > 0 else FAIL,
            value=int(cleaning_report.rows_out),
            message=f"{cleaning_report.rows_out:,} rows survive cleaning",
        )
    )

    if baseline.has_meta:
        prev_rows = int((baseline.meta or {}).get("dataset_profile", {}).get("n_rows", 0))
        delta_pct = _pct_change(profile.n_rows, prev_rows)
        status = WARN if abs(delta_pct) > thresholds.row_count_warn_pct else OK
        results.append(
            CheckResult(
                name="row_count_change_pct",
                category="data_quality",
                status=status,
                value=_round(delta_pct, 2),
                threshold=f"|delta| <= {thresholds.row_count_warn_pct}%",
                message=f"raw rows changed {delta_pct:+.2f}% vs previous run ({prev_rows:,} -> {profile.n_rows:,})",
            )
        )
    else:
        results.append(
            CheckResult(
                name="row_count_change_pct",
                category="data_quality",
                status=INFO,
                value=None,
                message="no previous run available for comparison",
            )
        )

    ratio_checks = [
        ("customer_id_null_ratio", profile.customer_id_null_ratio, 0.4),
        ("duplicate_ratio", profile.duplicate_ratio, 0.10),
        ("cancellation_ratio", profile.cancellation_ratio, 0.10),
        ("zero_or_negative_price_ratio", profile.zero_or_negative_price_ratio, 0.05),
        ("invoice_date_parse_fail_ratio", profile.invoice_date_parse_fail_ratio, 0.01),
    ]
    for name, value, warn_at in ratio_checks:
        status = WARN if value > warn_at else OK
        results.append(
            CheckResult(
                name=name,
                category="data_quality",
                status=status,
                value=_round(value, 6),
                threshold=f"<= {warn_at}",
                message=f"{name}={value:.4f} (warn if > {warn_at})",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Feature quality
# ---------------------------------------------------------------------------

def run_feature_checks(
    feat_df: pd.DataFrame,
    feat_report: FeatureBuildReport,
    baseline: Baseline,
    thresholds: EvalThresholds,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    results.append(
        CheckResult(
            name="n_customers",
            category="feature_quality",
            status=OK if feat_report.n_customers > 0 else FAIL,
            value=int(feat_report.n_customers),
            message=f"{feat_report.n_customers:,} customers",
        )
    )

    if baseline.has_meta:
        prev_n = int((baseline.meta or {}).get("feature_report", {}).get("n_customers", 0))
        delta_pct = _pct_change(feat_report.n_customers, prev_n)
        status = WARN if abs(delta_pct) > thresholds.n_customers_warn_pct else OK
        results.append(
            CheckResult(
                name="n_customers_change_pct",
                category="feature_quality",
                status=status,
                value=_round(delta_pct, 2),
                threshold=f"|delta| <= {thresholds.n_customers_warn_pct}%",
                message=f"customers changed {delta_pct:+.2f}% vs previous ({prev_n:,} -> {feat_report.n_customers:,})",
            )
        )
    else:
        results.append(
            CheckResult(
                name="n_customers_change_pct",
                category="feature_quality",
                status=INFO,
                message="no previous run available for comparison",
            )
        )

    null_inf_total = 0
    per_feature: dict[str, dict[str, int]] = {}
    for col in feat_report.feature_columns:
        if col not in feat_df.columns:
            per_feature[col] = {"missing": True}
            null_inf_total += 1
            continue
        s = pd.to_numeric(feat_df[col], errors="coerce")
        nulls = int(s.isna().sum())
        infs = int(np.isinf(s.fillna(0)).sum())
        per_feature[col] = {"nulls": nulls, "infs": infs}
        null_inf_total += nulls + infs

    results.append(
        CheckResult(
            name="feature_null_or_inf_total",
            category="feature_quality",
            status=OK if null_inf_total == 0 else WARN,
            value=int(null_inf_total),
            threshold="== 0",
            message=f"per-feature nulls/infs: {per_feature}",
        )
    )

    bound_violations: list[str] = []
    if "RecencyDays" in feat_df.columns:
        if (feat_df["RecencyDays"].astype(float) < 0).any():
            bound_violations.append("RecencyDays<0")
    if "Frequency" in feat_df.columns:
        if (feat_df["Frequency"].astype(float) < 1).any():
            bound_violations.append("Frequency<1")
    if "CancelRate" in feat_df.columns:
        cr = feat_df["CancelRate"].astype(float)
        if (cr < 0).any() or (cr > 1).any():
            bound_violations.append("CancelRate out of [0,1]")
    if "Monetary" in feat_df.columns:
        if not pd.api.types.is_numeric_dtype(feat_df["Monetary"]):
            bound_violations.append("Monetary not numeric")
    results.append(
        CheckResult(
            name="feature_sanity_bounds",
            category="feature_quality",
            status=OK if not bound_violations else WARN,
            value=bound_violations or "all_within_bounds",
            message="feature values respect expected ranges" if not bound_violations else f"violations: {bound_violations}",
        )
    )

    if baseline.has_insights:
        psi_results: dict[str, float] = {}
        max_psi_col: str | None = None
        max_psi_val = 0.0
        prev_df = baseline.insights
        assert prev_df is not None
        for col in feat_report.feature_columns:
            if col in feat_df.columns and col in prev_df.columns:
                psi = _psi(prev_df[col], feat_df[col])
                psi_results[col] = _round(psi, 4)
                if psi > max_psi_val:
                    max_psi_val = psi
                    max_psi_col = col
        status = WARN if max_psi_val > thresholds.psi_warn else OK
        results.append(
            CheckResult(
                name="feature_distribution_psi",
                category="feature_quality",
                status=status,
                value=psi_results,
                threshold=f"max PSI <= {thresholds.psi_warn}",
                message=(
                    f"max PSI={max_psi_val:.4f} on '{max_psi_col}'"
                    if max_psi_col
                    else "no overlapping numeric features"
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                name="feature_distribution_psi",
                category="feature_quality",
                status=INFO,
                message="no previous insights parquet available",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Model / clustering quality
# ---------------------------------------------------------------------------

def run_model_checks(
    cluster_result: ClusteringResult,
    insights_df: pd.DataFrame,
    baseline: Baseline,
    thresholds: EvalThresholds,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    metrics = cluster_result.metrics or {}

    sil = float(metrics.get("silhouette", 0.0))
    results.append(
        CheckResult(
            name="silhouette_score",
            category="model_quality",
            status=OK if sil >= thresholds.min_silhouette else WARN,
            value=_round(sil, 4),
            threshold=f">= {thresholds.min_silhouette}",
            message=f"silhouette={sil:.4f}",
        )
    )

    db = float(metrics.get("davies_bouldin", float("inf")))
    results.append(
        CheckResult(
            name="davies_bouldin_score",
            category="model_quality",
            status=OK if db <= thresholds.max_davies_bouldin else WARN,
            value=_round(db, 4),
            threshold=f"<= {thresholds.max_davies_bouldin}",
            message=f"davies_bouldin={db:.4f}",
        )
    )

    ch = float(metrics.get("calinski_harabasz", 0.0))
    results.append(
        CheckResult(
            name="calinski_harabasz_score",
            category="model_quality",
            status=OK,
            value=_round(ch, 2),
            message=f"calinski_harabasz={ch:.2f} (trend only, no threshold)",
        )
    )

    used_k = int(cluster_result.used_k)
    if "segment_id" in insights_df.columns and len(insights_df) > 0:
        sizes = insights_df["segment_id"].value_counts(normalize=True).to_dict()
        sizes_named = {int(k): _round(float(v), 4) for k, v in sizes.items()}
        min_frac = min(sizes.values()) if sizes else 0.0
        status = OK if min_frac >= thresholds.min_cluster_fraction else WARN
        results.append(
            CheckResult(
                name="cluster_balance",
                category="model_quality",
                status=status,
                value=sizes_named,
                threshold=f"min fraction >= {thresholds.min_cluster_fraction}",
                message=f"min cluster fraction={min_frac:.4f} across k={used_k}",
            )
        )
    else:
        results.append(
            CheckResult(
                name="cluster_balance",
                category="model_quality",
                status=FAIL,
                message="segment_id column missing from insights",
            )
        )

    if baseline.has_meta:
        prev_k = int((baseline.meta or {}).get("clustering", {}).get("used_k", 0))
        if prev_k > 0:
            status = OK if prev_k == used_k else WARN
            results.append(
                CheckResult(
                    name="k_stability",
                    category="model_quality",
                    status=status,
                    value=used_k,
                    threshold=f"== previous K ({prev_k})",
                    message=f"K={used_k}, previous K={prev_k}",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="k_stability",
                    category="model_quality",
                    status=INFO,
                    value=used_k,
                    message="no previous K to compare",
                )
            )
    else:
        results.append(
            CheckResult(
                name="k_stability",
                category="model_quality",
                status=INFO,
                value=used_k,
                message="no previous run available",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Business / insight metrics
# ---------------------------------------------------------------------------

def run_business_checks(
    insights_df: pd.DataFrame,
    baseline: Baseline,
    thresholds: EvalThresholds,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    if "recommended_stock_codes" in insights_df.columns:
        coverage = float(insights_df["recommended_stock_codes"].apply(_non_empty_list).mean())
        status = OK if coverage >= thresholds.recommendation_coverage_warn else WARN
        results.append(
            CheckResult(
                name="recommendation_coverage",
                category="business_metrics",
                status=status,
                value=_round(coverage, 4),
                threshold=f">= {thresholds.recommendation_coverage_warn}",
                message=f"{coverage * 100:.2f}% of customers have at least one recommended SKU",
            )
        )
    else:
        results.append(
            CheckResult(
                name="recommendation_coverage",
                category="business_metrics",
                status=FAIL,
                message="recommended_stock_codes column missing",
            )
        )

    if "campaign_hints" in insights_df.columns:
        non_null = float(insights_df["campaign_hints"].fillna("").astype(str).str.len().gt(0).mean())
        results.append(
            CheckResult(
                name="campaign_hints_coverage",
                category="business_metrics",
                status=OK if non_null >= 0.95 else WARN,
                value=_round(non_null, 4),
                threshold=">= 0.95",
                message=f"{non_null * 100:.2f}% of customers have a non-empty campaign_hints tag",
            )
        )

    for col, name in (
        ("churn_risk_score", "churn_risk_score"),
        ("clv_estimate", "clv_estimate"),
        ("transactional_promoter_score", "transactional_promoter_score"),
        ("rfm_composite_score", "rfm_composite_score"),
    ):
        if col not in insights_df.columns:
            continue
        s = pd.to_numeric(insights_df[col], errors="coerce")
        stats = {
            "mean": _round(float(s.mean()), 4),
            "median": _round(float(s.median()), 4),
            "p90": _round(float(s.quantile(0.90)), 4),
        }
        if baseline.has_insights and col in baseline.insights.columns:  # type: ignore[union-attr]
            prev = pd.to_numeric(baseline.insights[col], errors="coerce")  # type: ignore[union-attr]
            prev_mean = float(prev.mean())
            prev_median = float(prev.median())
            mean_drift = _pct_change(stats["mean"], prev_mean)
            median_drift = _pct_change(stats["median"], prev_median)
            worst = max(abs(mean_drift), abs(median_drift))
            status = WARN if worst > thresholds.score_drift_warn_pct else OK
            results.append(
                CheckResult(
                    name=f"{name}_distribution",
                    category="business_metrics",
                    status=status,
                    value={
                        **stats,
                        "mean_drift_pct": _round(mean_drift, 2),
                        "median_drift_pct": _round(median_drift, 2),
                    },
                    threshold=f"|drift| <= {thresholds.score_drift_warn_pct}%",
                    message=(
                        f"{name}: mean drift {mean_drift:+.2f}%, median drift {median_drift:+.2f}%"
                    ),
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"{name}_distribution",
                    category="business_metrics",
                    status=INFO if not baseline.has_insights else OK,
                    value=stats,
                    message=f"{name} stats only (no previous baseline for drift)",
                )
            )

    if "segment_id" in insights_df.columns:
        curr_dist = insights_df["segment_id"].value_counts(normalize=True).to_dict()
        if baseline.has_insights and "segment_id" in baseline.insights.columns:  # type: ignore[union-attr]
            prev_dist = baseline.insights["segment_id"].value_counts(normalize=True).to_dict()  # type: ignore[union-attr]
            keys = sorted(set(curr_dist) | set(prev_dist), key=lambda k: int(k))
            curr = np.array([curr_dist.get(k, 0.0) for k in keys], dtype=float)
            prev = np.array([prev_dist.get(k, 0.0) for k in keys], dtype=float)
            l1 = float(np.sum(np.abs(curr - prev)) * 100.0 / 2.0)
            status = WARN if l1 > thresholds.segment_distribution_warn_pct else OK
            results.append(
                CheckResult(
                    name="segment_distribution_drift",
                    category="business_metrics",
                    status=status,
                    value={
                        "current": {int(k): _round(float(curr_dist.get(k, 0.0)), 4) for k in keys},
                        "previous": {int(k): _round(float(prev_dist.get(k, 0.0)), 4) for k in keys},
                        "l1_half_pct": _round(l1, 2),
                    },
                    threshold=f"L1/2 <= {thresholds.segment_distribution_warn_pct}%",
                    message=f"segment mix L1/2 distance vs previous: {l1:.2f}%",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="segment_distribution_drift",
                    category="business_metrics",
                    status=INFO,
                    value={int(k): _round(float(v), 4) for k, v in curr_dist.items()},
                    message="no previous distribution for comparison",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Customer-level regression vs previous run
# ---------------------------------------------------------------------------

def run_regression_checks(
    insights_df: pd.DataFrame,
    baseline: Baseline,
    thresholds: EvalThresholds,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    if not baseline.has_insights:
        results.append(
            CheckResult(
                name="regression_vs_previous",
                category="regression",
                status=INFO,
                message="no previous insights parquet, regression skipped",
            )
        )
        return results

    prev = baseline.insights
    assert prev is not None
    if "CustomerID" not in insights_df.columns or "CustomerID" not in prev.columns:
        results.append(
            CheckResult(
                name="regression_vs_previous",
                category="regression",
                status=FAIL,
                message="CustomerID missing in current or previous insights",
            )
        )
        return results

    curr_idx = insights_df.set_index(insights_df["CustomerID"].astype(int))
    prev_idx = prev.set_index(prev["CustomerID"].astype(int))
    common = curr_idx.index.intersection(prev_idx.index)

    results.append(
        CheckResult(
            name="customers_intersection",
            category="regression",
            status=OK if len(common) > 0 else WARN,
            value=int(len(common)),
            message=f"{len(common):,} customers present in both runs",
        )
    )

    if len(common) == 0:
        return results

    if "segment_id" in curr_idx.columns and "segment_id" in prev_idx.columns:
        curr_seg = curr_idx.loc[common, "segment_id"].astype(int)
        prev_seg = prev_idx.loc[common, "segment_id"].astype(int)
        change_rate = float((curr_seg.values != prev_seg.values).mean())
        status = WARN if change_rate > thresholds.segment_change_rate_warn else OK
        results.append(
            CheckResult(
                name="segment_change_rate",
                category="regression",
                status=status,
                value=_round(change_rate, 4),
                threshold=f"<= {thresholds.segment_change_rate_warn}",
                message=(
                    "Note: cluster IDs are not stable across runs by default; "
                    f"{change_rate * 100:.2f}% of common customers changed segment_id label"
                ),
            )
        )

    for col, name in (
        ("churn_risk_score", "churn_risk_score_stability"),
        ("clv_estimate", "clv_estimate_stability"),
        ("transactional_promoter_score", "transactional_promoter_score_stability"),
    ):
        if col in curr_idx.columns and col in prev_idx.columns:
            c = pd.to_numeric(curr_idx.loc[common, col], errors="coerce")
            p = pd.to_numeric(prev_idx.loc[common, col], errors="coerce")
            denom = p.abs().replace(0, np.nan)
            pct = ((c - p) / denom * 100.0).abs()
            mean_abs_pct = float(pct.dropna().mean()) if not pct.dropna().empty else 0.0
            status = WARN if mean_abs_pct > thresholds.score_drift_warn_pct else OK
            results.append(
                CheckResult(
                    name=name,
                    category="regression",
                    status=status,
                    value=_round(mean_abs_pct, 2),
                    threshold=f"<= {thresholds.score_drift_warn_pct}%",
                    message=f"per-customer mean |%change| of {col}: {mean_abs_pct:.2f}%",
                )
            )

    return results


def _non_empty_list(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    return False
