from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.evaluation.evaluator import EvaluationReport


class EvaluationReporter:
    """Persist evaluation reports to disk and trim historical snapshots."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def write(self, report: EvaluationReport) -> Path:
        eval_dir = self.settings.evaluation_dir()
        history_dir = self.settings.evaluation_history_dir()
        eval_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)

        payload = report.to_dict()

        latest = self.settings.evaluation_report_path()
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        history_path = history_dir / f"run_{report.run_id}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        self._trim_history(history_dir, keep=int(self.settings.eval_history_keep_runs))
        return latest

    @staticmethod
    def _trim_history(history_dir: Path, *, keep: int) -> None:
        if keep <= 0:
            return
        files = sorted(
            (p for p in history_dir.glob("run_*.json") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in files[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass

    @staticmethod
    def console_summary(report: EvaluationReport) -> str:
        s = report.summary
        head = (
            f"evaluation_status={s['overall_status']} "
            f"checks={s['total']} passed={s['passed']} "
            f"warnings={s['warnings']} failed={s['failed']} info={s['info']}"
        )
        warns = s.get("warning_names") or []
        fails = s.get("failed_names") or []
        if fails:
            head += f" | failed: {', '.join(fails)}"
        if warns:
            head += f" | warnings: {', '.join(warns)}"
        return head
