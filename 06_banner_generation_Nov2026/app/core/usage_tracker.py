"""Token/cost accounting for every LLM and Image GenAI call made during one
`app/core/pipeline.py::run_generate()` / `run_clone()` execution (2026-09-06,
user request; schema redesigned twice since -- see (2) and (3) below).

Usage:

    tracker = UsageTracker(request_name)
    ...
    tracker.record_llm(usage_sink, stage="prompt_refinement", version=version)
    ...
    tracker.record_image(usage_sink, stage="image_generation", model=model,
                          version=version, width=w, height=h)
    ...
    tracker.save(num_version=num_version)   # persists this run into usage/usage_log.db

`usage_sink` is the small `dict` every instrumented provider call in
`app/llm/providers/*.py` / `app/providers/*.py` fills in with whatever raw
token counts its own API response actually reported (see each provider's
`generate()`/`generate_vision()` for the exact keys it sets) -- passing an
empty/missing sink is always safe: `record_llm()`/`record_image()` degrade
to `input_tokens=output_tokens=0` rather than raising, since counting usage
is a best-effort accounting feature that must never abort image generation
(same soft-fail convention as `app/core/technique_selector.py`'s selector
failures).

2026-09-06 (2) (user follow-up, after reading the first version's output):
the ORIGINAL design wrote 1 CSV row per individual GenAI call, into a
separate per-request-per-run file (`output/<request_name>/_usage/usage-
<run_id>.csv`) -- `app/core/output_validator.py::check_required_elements()`
alone can make up to 7 such rows for a single (model, version) generation
attempt. The user asked instead for: (1) always the SAME file, appended to
across EVERY run of EVERY request (never a new file per run/request); (2)
exactly 1 ROW per run, covering every version generated in that run; (3) a
column with the total cost per version, plus a column with the grand total
across all N versions. That version stored the per-version breakdown as a
`version_details_json` JSON-blob column inside a single shared
`output/usage_log.csv` row, since a flat CSV schema can't hold a variable
number of sub-columns (`num_version` differs run to run).

2026-09-06 (3) (user follow-up, pointing out the (2) design's own
limitation): stuffing per-version data as JSON text inside a single CSV
cell is itself only a workaround, not a real fix -- it can't be filtered,
joined, or summed with ordinary tools without first parsing that JSON back
out of every row. Asked for a storage recommendation, the user picked a
proper 2-table SQLite database. `save()` (no more `write_csv()`/CSV at all)
now persists this run as:

    runs         -- 1 row per RUN (`run_id` primary key): request_name,
                    timestamp, num_version, total input/output tokens,
                    shared_cost_usd (cost not tied to any 1 version),
                    total_cost_usd (grand total across the whole run),
                    currency, all_priced, notes.
    run_versions -- 1 row per VERSION generated in that run
                    (`run_id, version` primary key, `run_id` joining back
                    to `runs`): that version's cost_usd, input/output
                    tokens, and which model(s) ran for it.

However many versions a run has, `run_versions` just gets that many rows --
there is no more "how many columns" problem, since the variable-cardinality
dimension (`num_version`) now lives in ROWS of its own table instead of
columns of a flat schema.

The database lives at one FIXED path, `usage/usage_log.db` -- its own
directory, a sibling of `app/` directly at the project root (user request:
deliberately NOT nested under `output_dir`, which is itself
user-configurable via `.env` and could point anywhere) -- see
`Settings.usage_log_db_file` / `app/core/output_manager.usage_db_file()`.
Each `UsageTracker` instance still keeps one internal `UsageRecord` per
GenAI call (unchanged since the very first version -- still needed to
compute the per-version and grand-total costs correctly); those internal
records are never written to disk individually, only aggregated by
`save()`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core import model_profiles
from app.logging_conf import get_logger

logger = get_logger("usage_tracker")

_CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    request_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    num_version INTEGER NOT NULL,
    total_input_tokens INTEGER NOT NULL,
    total_output_tokens INTEGER NOT NULL,
    shared_cost_usd REAL NOT NULL,
    total_cost_usd REAL NOT NULL,
    currency TEXT NOT NULL,
    all_priced INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
)
"""

_CREATE_RUN_VERSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS run_versions (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    version INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    models TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, version)
)
"""

_INSERT_RUN = """
INSERT OR REPLACE INTO runs (
    run_id, request_name, timestamp, num_version,
    total_input_tokens, total_output_tokens, shared_cost_usd,
    total_cost_usd, currency, all_priced, notes
) VALUES (
    :run_id, :request_name, :timestamp, :num_version,
    :total_input_tokens, :total_output_tokens, :shared_cost_usd,
    :total_cost_usd, :currency, :all_priced, :notes
)
"""

_INSERT_RUN_VERSION = """
INSERT INTO run_versions (
    run_id, version, cost_usd, input_tokens, output_tokens, models
) VALUES (
    :run_id, :version, :cost_usd, :input_tokens, :output_tokens, :models
)
"""


@dataclass
class UsageRecord:
    """1 individual GenAI call -- kept in-memory only (see module docstring);
    never written to disk as its own row, only aggregated by `save()`."""

    run_id: str
    request_name: str
    stage: str
    resource_type: str
    model: str
    version: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    width: int | None = None
    height: int | None = None
    provider: str = ""
    input_price_usd_per_1m: float | None = None
    output_price_usd_per_1m: float | None = None
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    currency: str = "USD"
    priced: bool = False
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class UsageTracker:
    """One instance per `run_generate()`/`run_clone()` call. Never raises --
    every public method is best-effort: a malformed `usage_sink` or a
    pricing lookup failure logs a warning and records zero cost rather than
    interrupting image generation."""

    def __init__(self, request_name: str, run_id: str | None = None) -> None:
        self.request_name = request_name
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.records: list[UsageRecord] = []
        # One tracker is shared by the concurrent per-version workers in
        # pipeline.run_generate(). Protect writes and aggregation snapshots.
        self._records_lock = threading.Lock()

    def record_llm(
        self,
        usage_sink: dict | None,
        *,
        stage: str,
        version: int | None = None,
        fallback_model: str = "",
        notes: str = "",
    ) -> UsageRecord | None:
        """`usage_sink` is what an `app/llm/providers/*.py` provider filled
        in from its own API response (see e.g. `OpenAIProvider.generate()`).
        Returns `None` (and logs) instead of raising if `usage_sink` is
        empty/None -- that just means the call happened but reported no
        usable usage (e.g. the provider call itself failed before this
        tracker was even reached), which must never break the caller."""
        sink = usage_sink or {}
        model = str(sink.get("model") or fallback_model or "unknown")
        try:
            return self._record(
                stage=stage,
                resource_type="llm",
                model=model,
                version=version,
                provider=str(sink.get("provider") or ""),
                input_tokens=_int_or_zero(sink.get("input_tokens")),
                output_tokens=_int_or_zero(sink.get("output_tokens")),
                notes=notes,
            )
        except Exception:  # noqa: BLE001 -- accounting must never break the caller
            logger.exception("Could not record LLM usage for stage=%s model=%s (continuing)", stage, model)
            return None

    def record_image(
        self,
        usage_sink: dict | None,
        *,
        stage: str,
        model: str,
        version: int | None = None,
        width: int | None = None,
        height: int | None = None,
        provider: str = "",
        notes: str = "",
    ) -> UsageRecord | None:
        """`usage_sink` is what an `app/providers/*.py` image provider
        filled in from its own API response (may be empty for providers/
        models whose response never reports token usage, e.g. FLUX -- cost
        is still computed correctly in that case from `width`/`height` via
        the per-megapixel pricing shape, see `model_profiles.estimate_cost()`)."""
        sink = usage_sink or {}
        try:
            return self._record(
                stage=stage,
                resource_type="image",
                model=model,
                version=version,
                provider=provider or str(sink.get("provider") or ""),
                input_tokens=_int_or_zero(sink.get("input_tokens")),
                output_tokens=_int_or_zero(sink.get("output_tokens")),
                input_text_tokens=sink.get("input_text_tokens"),
                input_image_tokens=sink.get("input_image_tokens"),
                output_image_tokens=sink.get("output_image_tokens"),
                width=width,
                height=height,
                notes=notes,
            )
        except Exception:  # noqa: BLE001 -- accounting must never break the caller
            logger.exception("Could not record image usage for stage=%s model=%s (continuing)", stage, model)
            return None

    def _record(
        self,
        *,
        stage: str,
        resource_type: str,
        model: str,
        version: int | None,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        input_text_tokens: int | None = None,
        input_image_tokens: int | None = None,
        output_image_tokens: int | None = None,
        width: int | None = None,
        height: int | None = None,
        notes: str = "",
    ) -> UsageRecord:
        cost = model_profiles.estimate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_text_tokens=input_text_tokens,
            input_image_tokens=input_image_tokens,
            output_image_tokens=output_image_tokens,
            width=width,
            height=height,
        )
        combined_notes = "; ".join(part for part in (notes, cost.notes) if part)
        record = UsageRecord(
            run_id=self.run_id,
            request_name=self.request_name,
            stage=stage,
            resource_type=resource_type,
            model=model,
            version=version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            width=width,
            height=height,
            provider=provider,
            input_price_usd_per_1m=cost.input_rate,
            output_price_usd_per_1m=cost.output_rate,
            input_cost_usd=round(cost.input_cost, 6),
            output_cost_usd=round(cost.output_cost, 6),
            total_cost_usd=round(cost.total_cost, 6),
            currency=cost.currency,
            priced=cost.priced,
            notes=combined_notes,
        )
        with self._records_lock:
            self.records.append(record)
        return record

    def total_cost_usd(self) -> float:
        with self._records_lock:
            records = list(self.records)
        return round(sum(r.total_cost_usd for r in records), 6)

    def _aggregate(self, num_version: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Collapse every recorded GenAI call into the 2 relational pieces
        `save()` persists:

        - `run_row`: exactly 1 row, for the `runs` table -- covers the
          WHOLE run.
        - `version_rows`: 1 row per version actually present in this run,
          for the `run_versions` table -- however many versions that turns
          out to be (this is exactly the value that differs run to run and
          motivated the (3) SQLite redesign, see module docstring).

        Calls tied to a specific `version` (image generation, per-version
        prompt refinement, per-version vision presence-checks, ...)
        contribute to that version's row; calls that apply to the WHOLE run
        instead (technique-selector batches, `content_writer`/`text_gen`'s
        single request-wide LLM call, a shared (non-per-version) prompt
        refinement) are NOT tied to any one version and are summed into
        `shared_cost_usd` on the run row instead."""
        with self._records_lock:
            records = list(self.records)

        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        shared_cost = 0.0
        version_details: dict[int, dict[str, Any]] = {}
        unpriced: list[str] = []

        for r in records:
            if not r.priced and f"{r.stage}/{r.model}" not in unpriced:
                unpriced.append(f"{r.stage}/{r.model}")
            if r.version is None:
                shared_cost += r.total_cost_usd
                continue
            entry = version_details.setdefault(
                r.version, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "models": []}
            )
            entry["cost_usd"] += r.total_cost_usd
            entry["input_tokens"] += r.input_tokens
            entry["output_tokens"] += r.output_tokens
            if r.resource_type == "image" and r.model not in entry["models"]:
                entry["models"].append(r.model)

        version_rows: list[dict[str, Any]] = []
        for version, entry in sorted(version_details.items()):
            version_rows.append(
                {
                    "run_id": self.run_id,
                    "version": version,
                    "cost_usd": round(entry["cost_usd"], 6),
                    "input_tokens": entry["input_tokens"],
                    "output_tokens": entry["output_tokens"],
                    "models": ",".join(entry["models"]),
                }
            )

        version_total = sum(row["cost_usd"] for row in version_rows)
        currency = records[0].currency if records else "USD"

        run_row = {
            "run_id": self.run_id,
            "request_name": self.request_name,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_version": num_version if num_version is not None else len(version_details),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "shared_cost_usd": round(shared_cost, 6),
            "total_cost_usd": round(shared_cost + version_total, 6),
            "currency": currency,
            "all_priced": int(not unpriced),
            "notes": ("unpriced (charged as $0): " + "; ".join(unpriced)) if unpriced else "",
        }
        return run_row, version_rows

    def save(self, num_version: int | None = None) -> Path | None:
        """Persist this ENTIRE run into the shared SQLite database at
        `usage/usage_log.db` (2026-09-06 (3) user follow-up -- see module
        docstring): 1 row into `runs`, plus 1 row per version into
        `run_versions`, joined by `run_id`.

        `num_version` should be the actual number of versions requested for
        this run (`run_generate()`'s `num_version`, or the count of
        `display_sizes` for `run_clone()`) -- passed explicitly rather than
        inferred, since a version that happened to make zero billable calls
        (e.g. it failed before any GenAI call was reached) would otherwise
        be silently missing from `run_versions`.

        Re-saving the same `run_id` (should never happen in normal use --
        1 `UsageTracker` is only ever saved once per run -- but kept safe
        regardless) replaces its `runs` row (`INSERT OR REPLACE`) and its
        `run_versions` rows (delete-then-insert) rather than duplicating
        them.

        Returns the path, or `None` if there was nothing to record or the
        write itself failed (logged, never raised -- an optional accounting
        artifact must never fail the actual generate/clone run)."""
        if not self.records:
            return None
        from app.core import output_manager

        path = output_manager.usage_db_file()
        run_row, version_rows = self._aggregate(num_version)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                conn.execute(_CREATE_RUNS_TABLE)
                conn.execute(_CREATE_RUN_VERSIONS_TABLE)
                conn.execute(_INSERT_RUN, run_row)
                conn.execute("DELETE FROM run_versions WHERE run_id = ?", (self.run_id,))
                if version_rows:
                    conn.executemany(_INSERT_RUN_VERSION, version_rows)
                conn.commit()
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            logger.exception("Could not save run usage summary to %s", path)
            return None

        logger.info(
            "Saved run usage summary to %s (run_id=%s, request=%s, total cost ~%.4f %s)",
            path, self.run_id, self.request_name, run_row["total_cost_usd"], run_row["currency"],
        )
        return path


# ---------------------------------------------------------------------------
# Chạy độc lập file này (in ra 1 run mẫu để kiểm tra schema):
#   python -m app.core.usage_tracker
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tracker = UsageTracker("DEMO_REQ")
    tracker.record_llm({"model": "gpt-5-mini", "provider": "openai", "input_tokens": 500, "output_tokens": 200}, stage="prompt_refinement", version=1)
    tracker.record_image({"provider": "fake"}, stage="image_generation", model="flux.2-pro", version=1, width=1024, height=1024)
    tracker.record_image({"provider": "fake"}, stage="image_generation", model="flux.2-pro", version=2, width=1024, height=1024)
    tracker.record_llm({"model": "gpt-5-mini", "provider": "openai", "input_tokens": 300, "output_tokens": 100}, stage="technique_selector", version=None)
    run_row, version_rows = tracker._aggregate(num_version=2)
    print("runs row:")
    print(json.dumps(run_row, ensure_ascii=False, indent=2))
    print("run_versions rows:")
    print(json.dumps(version_rows, ensure_ascii=False, indent=2))
