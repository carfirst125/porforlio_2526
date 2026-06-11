from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.config import Settings
from app.evaluation import checks
from app.evaluation.baselines import Baseline
from app.evaluation.checks import CheckResult, FAIL, INFO, OK, WARN
from app.evaluation.thresholds import EvalThresholds
from app.features.customer_features import FeatureBuildReport
from app.modeling.cluster_model import ClusteringResult
from app.preprocessing.cleaning import CleaningReport
from app.preprocessing.profile import DatasetProfile


@dataclass
class EvaluationReport:
    """Structured output of one evaluation run; serializable to JSON."""

    run_id: str
    previous_run_id: str | None
    generated_at_utc: str
    summary: dict[str, Any]
    data_quality: list[CheckResult] = field(default_factory=list)
    feature_quality: list[CheckResult] = field(default_factory=list)
    model_quality: list[CheckResult] = field(default_factory=list)
    business_metrics: list[CheckResult] = field(default_factory=list)
    regression_vs_previous: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "previous_run_id": self.previous_run_id,
            "generated_at_utc": self.generated_at_utc,
            "summary": self.summary,
            "data_quality": [c.to_dict() for c in self.data_quality],
            "feature_quality": [c.to_dict() for c in self.feature_quality],
            "model_quality": [c.to_dict() for c in self.model_quality],
            "business_metrics": [c.to_dict() for c in self.business_metrics],
            "regression_vs_previous": [c.to_dict() for c in self.regression_vs_previous],
        }

    def all_checks(self) -> list[CheckResult]:
        return [
            *self.data_quality,
            *self.feature_quality,
            *self.model_quality,
            *self.business_metrics,
            *self.regression_vs_previous,
        ]


class EvaluationOrchestrator:
    """Runs every check group and produces a consolidated report."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.thresholds = EvalThresholds.from_settings(settings)

    def evaluate(
        self,
        *,
        raw: pd.DataFrame,
        profile: DatasetProfile,
        cleaning_report: CleaningReport,
        feat_df: pd.DataFrame,
        feat_report: FeatureBuildReport,
        cluster_result: ClusteringResult,
        insights_df: pd.DataFrame,
        baseline: Baseline,
    ) -> EvaluationReport:
        data_q = checks.run_data_checks(raw, profile, cleaning_report, baseline, self.thresholds)
        feat_q = checks.run_feature_checks(feat_df, feat_report, baseline, self.thresholds)
        model_q = checks.run_model_checks(cluster_result, insights_df, baseline, self.thresholds)
        biz_q = checks.run_business_checks(insights_df, baseline, self.thresholds)
        regr_q = checks.run_regression_checks(insights_df, baseline, self.thresholds)

        all_checks = [*data_q, *feat_q, *model_q, *biz_q, *regr_q]
        summary = self._summarize(all_checks)

        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ")

        return EvaluationReport(
            run_id=run_id,
            previous_run_id=baseline.run_id,
            generated_at_utc=now.isoformat(),
            summary=summary,
            data_quality=data_q,
            feature_quality=feat_q,
            model_quality=model_q,
            business_metrics=biz_q,
            regression_vs_previous=regr_q,
        )

    @staticmethod
    def _summarize(all_checks: list[CheckResult]) -> dict[str, Any]:
        passed = sum(1 for c in all_checks if c.status == OK)
        warnings = sum(1 for c in all_checks if c.status == WARN)
        failed = sum(1 for c in all_checks if c.status == FAIL)
        info = sum(1 for c in all_checks if c.status == INFO)
        if failed > 0:
            overall = "failed"
        elif warnings > 0:
            overall = "warning"
        else:
            overall = "ok"
        return {
            "total": len(all_checks),
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "info": info,
            "overall_status": overall,
            "warning_names": [c.name for c in all_checks if c.status == WARN],
            "failed_names": [c.name for c in all_checks if c.status == FAIL],
        }
