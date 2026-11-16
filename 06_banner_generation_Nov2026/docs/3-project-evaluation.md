# 3. Project Evaluation — Methodology

Maps to `app/evaluation/{unit,smoke,stress}/`. Run with `pytest`
(`requirements-dev.txt`); default run is `unit` + `smoke` only
(`pytest.ini`: `addopts = -m "not stress"`), `stress` is opt-in
(`pytest -m stress -v`) since it's slower and its thresholds are
environment-dependent (see that tier's section below).

No tier makes a real network call to a real Gemini/Azure/OpenAI/Anthropic
endpoint — every image-model and LLM-provider call in the suite is mocked
(`app/evaluation/conftest.py::fake_generate_image`, or simply never invoked
because the relevant credential is left unconfigured in the test
environment). This is a deliberate boundary: the suite verifies the APP's
own logic (routing, naming, caching, error handling, concurrency safety),
not a third party's API uptime or exact response schema — those are
verified manually against the real endpoint the first time a given model is
actually used (see `1-project_description.md`'s "Giả định & ràng buộc" and
`2-project-technote.md` section 6).

## Unit tier (`app/evaluation/unit/`)

Pure logic, no network, no filesystem writes outside `tmp_path`. Covers:

| Module | Test cases |
|---|---|
| `app/models/schemas.py` | `parse_wh`/`wh_slug` (comma form, `x` form, whitespace, empty/malformed → error); bucket normalization (string → list, list-of-multiple, empty); `crop_box()`/`final_output_wh()`/`has_crop()` with an explicit `output.crop` vs. falling back to `generation.safe_area` vs. no-crop (canvas == output.size); missing `output.size`/`generation` rejected; a stray legacy `design_options.protect_area` field is ignored, not crashed |
| `app/core/model_usage.py` | every mode (`default`/`quality`/`eco`/`custom`/`all`/unknown); `all` unions+dedupes in declared order; `custom` ignores the JSON's own `custom` key; missing file / broken JSON → reported error, not a crash |
| `app/core/guidelines.py` | `general_guide` recursion against the REAL `guidelines.json` (2 real format_types); unknown `format_type` → reported missing, not a crash; cycle-safety (2 entries pointing at each other doesn't hang) |
| `app/core/output_manager.py` | filename convention (version suffix only when `num_version>1`); clone filename convention; `since_ts` filtering; non-result directories (`prompt/`, `release/`, ...) excluded from listings; substring lookup case-insensitivity; release copy (success + missing-source case); **collision-avoidance** (`unique_image_output_path` never returns a path that already exists) |
| `app/llm/factory.py` | every registered provider name (incl. the `azure_gpt5mini` backward-compat alias) resolves to the right class; unknown name → `ValueError`; a provider with no credentials configured raises a clear `LLMProviderError` at call time, not a silent wrong answer |

Bias: 1 "happy path" case per behavior + at least 1 boundary/error case
(empty input, missing file, unknown key, no credentials) — a module with
only happy-path tests is considered under-tested for this tier.

## Smoke tier (`app/evaluation/smoke/`)

Full FastAPI round-trip (`TestClient`, in-process ASGI — no real HTTP
socket, but exercises the real routing/serialization/pipeline code path),
against REAL fixture data already in this project (`input/DB_NOTI_AUTOPAY_NOEL.json`
+ its real bucket/guideline images under `lib/`), with ONLY the outbound
image-model HTTP call mocked. Covers every endpoint
`webapp/utils/api_client.py` calls: `/health`, `/models`, `/requests`,
`/requests/{name}` (found + not-found), `/generate` (success + "no models
resolved" failure path), `/outputs/*` (latest, at-timestamp), `/files`
(serves a real generated file + blocks a path outside the allow-list),
`/release`, `/release/{name}`, `/clone-from-release`,
`/release/{name}/clone-status`.

Pass criteria: every call returns the expected HTTP status and JSON shape;
no endpoint ever returns a raw 500 for a foreseeable failure (missing
request file, no models resolved) — REQUEST.md's "dump exception, never
crash" requirement applies to the API layer too.

## Stress tier (`app/evaluation/stress/`)

Concurrency against the in-process app (`TestClient` from multiple Python
threads via `ThreadPoolExecutor`), image-model calls still mocked. Measures
what this tier CAN measure without a real deployed server or real provider
API — the app's own concurrency-safety and its own overhead, not a real
provider's latency.

1. **`/health` under load** — 50 workers × 4 requests (200 total).
   Pass: 0 errors, p95 < 200ms.
2. **`/generate` under concurrent requests** — 10 concurrent calls, alternating
   between 2 different requests (5 each -- `two_requests_isolated_input`
   fixture, `app/evaluation/conftest.py`: a copy of the real
   `DB_NOTI_AUTOPAY_NOEL.json` + a synthetic 2nd request reusing the same
   real `lib/` assets, since only 1 real request file is guaranteed to exist
   under `input/`), checking for cross-talk
   (a result path must belong to the request it was generated for) and a
   generous regression-only latency ceiling (see that test's inline comment
   for why 20s, not something tighter — real Pillow downsampling of
   multi-MB images is genuine CPU work under 10 concurrent threads on a
   shared sandbox CPU, this isn't measuring a real provider's speed).

Metric/tool: Python's own `time.perf_counter()` per call +
`statistics.mean()` / a simple index-based percentile helper — no external
load-testing tool (`locust`, etc.) added as a dependency for a suite this
small; revisit if the app grows a real multi-user deployment target.

Pass criteria are written as **regression guards**, not real-world SLA
promises — e.g. "0 errors, p95 < 200ms" for `/health` protects against an
accidental blocking call being introduced, it does not claim the deployed
production API will serve 50 concurrent users in under 200ms once real
provider network latency (seconds, per REQUEST.md's `GENERATE_TIMEOUT_SECONDS`
budget of up to 30 minutes for `mode=all`) is in the loop.

## "What's good enough" thresholds (summary)

| Tier | Threshold |
|---|---|
| Unit | Every module in `app/core/`, `app/models/`, `app/llm/` has ≥1 happy-path + ≥1 boundary/error case. 100% pass required before merge. |
| Smoke | Every endpoint in `app/api/routes.py` has ≥1 test. 100% pass required before merge. |
| Stress | `/health`: 0 errors, p95 < 200ms. `/generate` (mocked provider): 0 cross-talk, avg latency < 20s on the dev sandbox. Run manually before a release, not required on every commit (excluded from default `pytest`). |

Real measured numbers from the most recent run are logged in
`4-project-evaluation-results.md` — update that file, never this one, when
re-running the suite (this file describes the METHOD, that file records
RESULTS).
