"""Stress tier -- concurrency/load against the FastAPI app in-process
(`TestClient`, image-model calls mocked -- see 3-project-evaluation.md for
why this measures the APP's own concurrency handling, not a real provider's
latency, and for the pass/fail thresholds asserted below).

Excluded from the default `pytest` run (see `pytest.ini`'s
`addopts = -m "not stress"`) since it takes longer and is meant to be run
deliberately: `pytest -m stress -v`.
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.stress


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return sorted_values[idx]


@pytest.fixture
def client(isolated_output, isolated_safe_area_guide_dir):
    from app.main import app

    return TestClient(app)


def test_health_endpoint_under_concurrent_load(client):
    """50 concurrent users x 4 requests each = 200 total hits to `/health`.
    Pass criteria (documented in 3-project-evaluation.md): 0 errors, p95 < 200ms."""
    n_workers, n_per_worker = 50, 4
    latencies: list[float] = []
    errors = 0

    def _one_call() -> tuple[float, bool]:
        start = time.perf_counter()
        try:
            r = client.get("/health")
            ok = r.status_code == 200 and r.json().get("ok") is True
        except Exception:  # noqa: BLE001
            ok = False
        return time.perf_counter() - start, ok

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_one_call) for _ in range(n_workers * n_per_worker)]
        for f in futures:
            elapsed, ok = f.result()
            latencies.append(elapsed)
            if not ok:
                errors += 1

    latencies.sort()
    p50, p95, p99 = _percentile(latencies, 0.50), _percentile(latencies, 0.95), _percentile(latencies, 0.99)
    print(f"\n/health under load: n={len(latencies)} errors={errors} p50={p50*1000:.1f}ms p95={p95*1000:.1f}ms p99={p99*1000:.1f}ms")

    assert errors == 0
    assert p95 < 0.2, f"p95 latency {p95*1000:.1f}ms exceeds the 200ms threshold"


def test_generate_under_concurrent_requests(client, fake_generate_image, two_requests_isolated_input):
    """10 concurrent `/generate` calls (mode=quality, mocked provider so no
    real network/API cost) for DIFFERENT requests, checking the pipeline
    doesn't corrupt state or drop results under concurrency (e.g. the
    prompt-cache file per request, or output paths colliding). Pass
    criteria: every call succeeds, and results are attributable to the
    correct request (no cross-talk).

    Uses `two_requests_isolated_input` (app/evaluation/conftest.py) instead
    of 2 real files under the project's own `input/` directory, since only
    `DB_NOTI_AUTOPAY_NOEL.json` is guaranteed to exist there (2026-08-20,
    see `4-project-evaluation-results.md`)."""
    name_a, name_b = two_requests_isolated_input
    request_names = [name_a, name_b] * 5  # 10 calls total

    def _one_generate(request_name: str) -> dict:
        start = time.perf_counter()
        r = client.post("/generate", json={"request_name": request_name, "mode": "quality", "custom_models": []})
        return {"request_name": request_name, "elapsed": time.perf_counter() - start, "body": r.json()}

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_one_generate, request_names))

    latencies = sorted(r["elapsed"] for r in results)
    print(f"\n/generate x{len(results)} concurrent: p50={_percentile(latencies, 0.5)*1000:.0f}ms p95={_percentile(latencies, 0.95)*1000:.0f}ms")

    for r in results:
        assert r["body"]["ok"] is True, r["body"]
        for res in r["body"]["results"]:
            # every result path must belong to the SAME request it was requested for
            assert f"/output/{r['request_name']}/" in res["path"].replace("\\", "/")

    # NOTE: only the outbound image-model HTTP call is mocked -- steps [1]-[5]
    # (read request, resolve assets, guideline lookup, real Pillow downsample
    # of the real multi-MB bucket/guideline JPG/PNGs, prompt assembly) still
    # run for real, and 10 threads doing that concurrently under Python's GIL
    # on a shared/constrained CPU is genuinely slower than "near-instant" --
    # this threshold is deliberately generous (measured baseline on the dev
    # sandbox: ~8.5s avg for 10 concurrent calls, see
    # 4-project-evaluation-results.md) and exists to catch a REGRESSION
    # (e.g. an accidental blocking sleep, a lock, N+1 file re-reads), not to
    # assert a specific real-world SLA.
    avg_latency = statistics.mean(latencies)
    assert avg_latency < 20.0, f"average /generate latency {avg_latency:.2f}s exceeds the 20s regression threshold"


# ---------------------------------------------------------------------------
# Chạy độc lập file này (tier stress bị loại khỏi `pytest` mặc định):
#   cd version_1 && pytest -m stress -v app/evaluation/stress/test_generate_stress.py
# ---------------------------------------------------------------------------
