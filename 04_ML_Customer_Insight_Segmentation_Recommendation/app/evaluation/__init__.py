"""Quality monitoring layer for the customer insights pipeline.

Exposes the orchestrator, baselines loader, and reporter so callers (pipeline,
dashboard) can build evaluation reports without depending on internal helpers.
"""

from app.evaluation.baselines import Baseline, load_previous_baseline
from app.evaluation.evaluator import EvaluationOrchestrator, EvaluationReport
from app.evaluation.reporter import EvaluationReporter
from app.evaluation.thresholds import EvalThresholds

__all__ = [
    "Baseline",
    "EvalThresholds",
    "EvaluationOrchestrator",
    "EvaluationReport",
    "EvaluationReporter",
    "load_previous_baseline",
]
