"""
Rebuild customer insights and clustering artifacts from the transaction CSV.

Run from folder ``05_customer_transaction``::

    python -m app.jobs.refresh_insights

Schedule with Windows Task Scheduler / cron / Airflow — not from the FastAPI process.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings
from app.pipeline.customer_insights_pipeline import CustomerInsightsPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh offline customer insights store.")
    _ = parser.parse_args(argv)

    result = CustomerInsightsPipeline(get_settings()).run()
    payload: dict = {
        "ok": True,
        "insights_path": str(result.insights_path),
        "n_customers": result.n_customers,
    }
    if result.evaluation_summary is not None:
        payload["evaluation_status"] = result.evaluation_summary.get("overall_status")
        payload["evaluation_warnings"] = result.evaluation_summary.get("warning_names", [])
        payload["evaluation_failed"] = result.evaluation_summary.get("failed_names", [])
        payload["evaluation_counts"] = {
            "total": result.evaluation_summary.get("total"),
            "passed": result.evaluation_summary.get("passed"),
            "warnings": result.evaluation_summary.get("warnings"),
            "failed": result.evaluation_summary.get("failed"),
            "info": result.evaluation_summary.get("info"),
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
