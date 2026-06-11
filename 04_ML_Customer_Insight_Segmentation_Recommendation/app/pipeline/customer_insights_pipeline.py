from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Settings, get_settings
from app.data.loader import load_transactions
from app.evaluation import (
    EvaluationOrchestrator,
    EvaluationReporter,
    load_previous_baseline,
)
from app.features.customer_features import CustomerFeatureBuilder
from app.insights.extended_metrics import ExtendedInsightCalculator
from app.modeling.cluster_model import ClusteringTrainer
from app.modeling.recommender import SegmentRecommender
from app.preprocessing.cleaning import TransactionCleaner
from app.preprocessing.profile import DatasetProfile
from app.storage.artifact_store import ArtifactStore


@dataclass
class PipelineRunResult:
    insights_path: Path
    n_customers: int
    meta: dict[str, Any]
    evaluation_summary: dict[str, Any] | None = None


class CustomerInsightsPipeline:
    """End-to-end rebuild: load → profile → clean → features → cluster → recommend → insights."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(self) -> PipelineRunResult:
        s = self.settings
        # Load baseline before the new run overwrites the previous parquet/meta;
        # safe no-op on first run.
        baseline = load_previous_baseline(s) if s.eval_enabled else None

        raw = load_transactions(s)
        profile = DatasetProfile.from_dataframe(raw)

        cleaner = TransactionCleaner(s)
        cleaned, cleaning_report = cleaner.clean(raw, profile)

        builder = CustomerFeatureBuilder(s)
        feat_df, feat_report = builder.build(cleaned)

        trainer = ClusteringTrainer(s)
        cluster_result = trainer.fit(feat_df, feat_report.feature_columns)
        feat_df = feat_df.copy()
        feat_df["segment_id"] = cluster_result.labels.astype(int)

        extended = ExtendedInsightCalculator()
        insights_df = extended.enrich(feat_df)

        seg_tbl = insights_df[["CustomerID", "segment_id"]].copy()
        recommender = SegmentRecommender(s)
        rec_df = recommender.build_recommendations(cleaned, seg_tbl)
        insights_df = insights_df.merge(rec_df, on="CustomerID", how="left")
        insights_df["recommended_stock_codes"] = insights_df["recommended_stock_codes"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        artifacts = ArtifactStore(s)
        artifacts.save_cluster_bundle(cluster_result, algorithm=s.cluster_algorithm)
        insights_path = s.insights_path()
        insights_path.parent.mkdir(parents=True, exist_ok=True)
        insights_df.to_parquet(insights_path, index=False)

        meta = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_profile": profile.to_dict(),
            "cleaning_report": cleaning_report.to_dict(),
            "feature_report": feat_report.to_dict(),
            "clustering": cluster_result.to_dict(),
            "transaction_csv": str(s.resolved_transaction_csv()),
            "insights_note": (
                "transactional_promoter_score is a behavioral proxy; it does not replace survey-based NPS."
            ),
        }

        evaluation_summary: dict[str, Any] | None = None
        if s.eval_enabled and baseline is not None:
            report = EvaluationOrchestrator(s).evaluate(
                raw=raw,
                profile=profile,
                cleaning_report=cleaning_report,
                feat_df=feat_df,
                feat_report=feat_report,
                cluster_result=cluster_result,
                insights_df=insights_df,
                baseline=baseline,
            )
            EvaluationReporter(s).write(report)
            evaluation_summary = report.summary
            meta["evaluation_summary"] = evaluation_summary
            meta["evaluation_run_id"] = report.run_id

        artifacts.save_meta(meta)

        return PipelineRunResult(
            insights_path=insights_path,
            n_customers=len(insights_df),
            meta=meta,
            evaluation_summary=evaluation_summary,
        )
