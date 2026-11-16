"""Unit tests -- `app/core/usage_tracker.py` (2026-09-06 user request: token/
cost accounting for every GenAI call made during a run; schema redesigned
twice since -- (2) 1 aggregated row per RUN, then (3) a proper 2-table
SQLite database, `usage/usage_log.db`, replacing the flat-CSV-with-a-JSON-
column design). Uses the REAL `config/model_profiles.json` pricing (via
`app.core.model_profiles`) so costs below are real, not mocked.
"""
from __future__ import annotations

import sqlite3

import pytest

from app.core.usage_tracker import UsageTracker


class _StubSettings:
    """`Settings.usage_log_db_file` is a read-only computed property (same
    pattern as `test_model_profiles.py`'s `_StubSettings` for
    `model_profiles_file`) -- can't be `monkeypatch.setattr()`'d directly on
    the real singleton, so tests instead swap out `output_manager.get_settings`
    for this tiny stub, which only needs to answer `usage_log_db_file`."""

    def __init__(self, path):
        self.usage_log_db_file = path


@pytest.fixture
def isolated_usage_db(monkeypatch, tmp_path):
    """Redirects the shared usage/cost SQLite database to a throwaway path
    under `tmp_path`, so evaluation tests never touch the real project's
    `usage/usage_log.db`."""
    from app.core import output_manager

    db_path = tmp_path / "usage" / "usage_log.db"
    monkeypatch.setattr(output_manager, "get_settings", lambda: _StubSettings(db_path))
    return db_path


def _fetch_run(db_path, run_id):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _fetch_versions(db_path, run_id):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM run_versions WHERE run_id = ? ORDER BY version", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def test_record_llm_with_usage_sink_computes_cost():
    tracker = UsageTracker("UNIT_TEST_REQ", run_id="testrun1")
    sink = {"model": "gpt-5-mini", "provider": "openai", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    record = tracker.record_llm(sink, stage="prompt_refinement", version=1)
    assert record is not None
    assert record.run_id == "testrun1"
    assert record.request_name == "UNIT_TEST_REQ"
    assert record.model == "gpt-5-mini"
    assert record.provider == "openai"
    assert record.input_tokens == 1_000_000
    assert record.output_tokens == 1_000_000
    assert record.priced is True
    assert record.total_cost_usd == 2.25  # $0.25 + $2.00, see test_model_profiles.py
    assert len(tracker.records) == 1


def test_record_llm_with_none_sink_degrades_to_zero_usage_never_raises():
    tracker = UsageTracker("UNIT_TEST_REQ")
    record = tracker.record_llm(None, stage="prompt_refinement", version=1, fallback_model="gpt-5-mini")
    assert record is not None
    assert record.input_tokens == 0
    assert record.output_tokens == 0
    assert record.total_cost_usd == 0.0
    assert record.model == "gpt-5-mini"  # from fallback_model, since sink had no "model" key


def test_record_llm_unknown_model_is_unpriced_but_still_recorded():
    tracker = UsageTracker("UNIT_TEST_REQ")
    sink = {"model": "no-such-model", "provider": "openai", "input_tokens": 10, "output_tokens": 5}
    record = tracker.record_llm(sink, stage="technique_selector")
    assert record is not None
    assert record.priced is False
    assert record.total_cost_usd == 0.0
    assert "no pricing profile" in record.notes


def test_record_image_flux_per_megapixel_uses_width_height_not_tokens():
    tracker = UsageTracker("UNIT_TEST_REQ")
    record = tracker.record_image(
        {},  # FLUX reports no usage object at all -- empty sink is normal
        stage="image_generation",
        model="flux.2-pro",
        version=2,
        width=1000,
        height=1000,
    )
    assert record is not None
    assert record.priced is True
    assert abs(record.total_cost_usd - 0.03) < 1e-9
    assert record.resource_type == "image"
    assert record.version == 2


def test_record_image_with_split_text_image_token_breakdown():
    tracker = UsageTracker("UNIT_TEST_REQ")
    sink = {
        "provider": "azure-openai",
        "input_text_tokens": 1_000_000,
        "input_image_tokens": 1_000_000,
        "output_image_tokens": 1_000_000,
    }
    record = tracker.record_image(sink, stage="image_generation", model="gpt-image-2", version=1)
    assert record is not None
    assert record.priced is True
    assert abs(record.total_cost_usd - 43.0) < 1e-9


def test_total_cost_usd_sums_every_record():
    tracker = UsageTracker("UNIT_TEST_REQ")
    tracker.record_llm({"model": "gpt-5-mini", "input_tokens": 1_000_000, "output_tokens": 0}, stage="a")
    tracker.record_llm({"model": "gpt-5-mini", "input_tokens": 0, "output_tokens": 1_000_000}, stage="b")
    assert tracker.total_cost_usd() == 2.25  # $0.25 + $2.00


def test_save_with_no_records_returns_none(isolated_usage_db):
    tracker = UsageTracker("UNIT_TEST_REQ_EMPTY")
    assert tracker.save() is None


def test_save_writes_one_run_row_plus_one_row_per_version(isolated_usage_db):
    from app.core import output_manager

    tracker = UsageTracker("UNIT_TEST_REQ_DB", run_id="abc123")
    # version 1: 1 LLM call + 1 image call
    tracker.record_llm({"model": "gpt-5-mini", "input_tokens": 100, "output_tokens": 50}, stage="prompt_refinement", version=1)
    tracker.record_image({}, stage="image_generation", model="flux.2-pro", version=1, width=500, height=500)
    # version 2: 1 image call only
    tracker.record_image({}, stage="image_generation", model="flux.2-pro", version=2, width=1000, height=1000)
    # a run-wide call not tied to any one version (e.g. technique selector)
    tracker.record_llm({"model": "gpt-5-mini", "input_tokens": 300, "output_tokens": 100}, stage="technique_selector", version=None)

    path = tracker.save(num_version=2)
    assert path is not None
    assert path == output_manager.usage_db_file()
    assert path.name == "usage_log.db"
    assert path.parent == isolated_usage_db.parent
    assert path.exists()

    run_row = _fetch_run(path, "abc123")
    assert run_row is not None
    assert run_row["request_name"] == "UNIT_TEST_REQ_DB"
    assert run_row["num_version"] == 2
    assert run_row["all_priced"] == 1

    versions = _fetch_versions(path, "abc123")
    assert [v["version"] for v in versions] == [1, 2]
    v1, v2 = versions
    assert v1["models"] == "flux.2-pro"
    assert abs(v1["cost_usd"] - (0.25e-6 * 100 + 2.00e-6 * 50 + 0.25 * 0.03)) < 1e-6  # refinement + 0.25MP flux
    assert abs(v2["cost_usd"] - 1.0 * 0.03) < 1e-9  # 1000x1000 = 1 MP flux

    shared_cost = run_row["shared_cost_usd"]
    # technique_selector call: 300 input tok @ $0.25/1M + 100 output tok @ $2.00/1M
    assert abs(shared_cost - (300 / 1_000_000 * 0.25 + 100 / 1_000_000 * 2.00)) < 1e-9

    version_total = sum(v["cost_usd"] for v in versions)
    assert abs(run_row["total_cost_usd"] - (shared_cost + version_total)) < 1e-9


def test_save_accumulates_across_multiple_runs_into_the_same_database(isolated_usage_db):
    """The whole point of the (2)/(3) redesigns: every run -- for ANY
    request_name -- persists into the SAME shared database, never a fresh
    one, and each run's rows never disturb another run's rows."""
    from app.core import output_manager

    t1 = UsageTracker("REQ_A", run_id="run1")
    t1.record_image({}, stage="image_generation", model="flux.2-pro", version=1, width=1000, height=1000)
    path1 = t1.save(num_version=1)

    t2 = UsageTracker("REQ_B", run_id="run2")
    t2.record_image({}, stage="image_generation", model="flux.2-pro", version=1, width=1000, height=1000)
    t2.record_image({}, stage="image_generation", model="flux.2-pro", version=2, width=1000, height=1000)
    path2 = t2.save(num_version=2)

    assert path1 == path2 == output_manager.usage_db_file()

    conn = sqlite3.connect(str(path2))
    try:
        conn.row_factory = sqlite3.Row
        runs = conn.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
    finally:
        conn.close()
    assert [r["run_id"] for r in runs] == ["run1", "run2"]  # 1 row per run, both persisted
    assert dict(runs[0])["request_name"] == "REQ_A"
    assert dict(runs[1])["request_name"] == "REQ_B"

    assert len(_fetch_versions(path2, "run1")) == 1
    assert len(_fetch_versions(path2, "run2")) == 2


def test_save_records_unpriced_model_in_notes_but_still_writes_row(isolated_usage_db):
    tracker = UsageTracker("UNIT_TEST_REQ_UNPRICED", run_id="run3")
    tracker.record_image({}, stage="image_generation", model="mai-image-2.5", version=1, width=1000, height=1000)
    path = tracker.save(num_version=1)
    assert path is not None

    run_row = _fetch_run(path, "run3")
    assert run_row["all_priced"] == 0
    assert "image_generation/mai-image-2.5" in run_row["notes"]
    versions = _fetch_versions(path, "run3")
    assert versions[0]["cost_usd"] == 0.0


def test_save_on_the_same_run_id_replaces_rather_than_duplicates(isolated_usage_db):
    """Not expected in normal pipeline use (1 UsageTracker is saved once
    per run) but `save()` is documented as idempotent per run_id -- verify
    a 2nd save() for the same run_id doesn't leave stale/duplicate rows."""
    tracker = UsageTracker("UNIT_TEST_REQ_REPLACE", run_id="run4")
    tracker.record_image({}, stage="image_generation", model="flux.2-pro", version=1, width=500, height=500)
    tracker.record_image({}, stage="image_generation", model="flux.2-pro", version=2, width=500, height=500)
    path = tracker.save(num_version=2)

    # Simulate a 2nd save() of the SAME tracker/run_id after 1 more call was
    # recorded for a 3rd version.
    tracker.record_image({}, stage="image_generation", model="flux.2-pro", version=3, width=500, height=500)
    tracker.save(num_version=3)

    run_row = _fetch_run(path, "run4")
    assert run_row["num_version"] == 3  # replaced, not duplicated

    versions = _fetch_versions(path, "run4")
    assert [v["version"] for v in versions] == [1, 2, 3]  # no stale/duplicate rows


def test_usage_db_file_path_is_fixed_regardless_of_request_name(isolated_usage_db):
    from app.core import output_manager

    assert output_manager.usage_db_file() == output_manager.usage_db_file()
    assert output_manager.usage_db_file().name == "usage_log.db"
    assert output_manager.usage_db_file().parent == isolated_usage_db.parent
def test_recording_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    tracker = UsageTracker("THREAD_SAFE")

    def record(index: int):
        tracker.record_llm(
            {"model": "gpt-5-mini", "input_tokens": 1, "output_tokens": 1},
            stage="parallel",
            version=(index % 3) + 1,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, range(100)))

    assert len(tracker.records) == 100
    run_row, version_rows = tracker._aggregate(num_version=3)
    assert run_row["total_input_tokens"] == 100
    assert sum(row["input_tokens"] for row in version_rows) == 100

