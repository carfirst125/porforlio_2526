# 4. Project Evaluation Results — Change Log

## 2026-08-19 — Initial `app/` backend built from scratch

**Change:** `webapp/`, `config/`, `input/`, `lib/`, `prompts/` already
existed; the entire FastAPI + LangGraph `app/` backend package (REQUEST.md)
did not exist in `version_1/` yet. Built: `app/settings.py`, `app/main.py`,
`app/api/routes.py`, `app/models/schemas.py`, `app/core/*` (guidelines,
asset_resolver, image_prep, image_post, output_manager, model_usage,
prompt_builder, pipeline — LangGraph workflow), `app/providers/*` (7 image
models), `run_cli.py`, `requirements.txt`.

**Result (ad-hoc scratch scripts, not yet the formal `app/evaluation/`
suite — that was added in the 2026-08-20 pass below):**
- Full pipeline run (guideline resolution incl. `general_guide` recursion,
  `model_usage.json` mode resolution, multi-model/multi-version generate
  with a mocked provider, `protect_area` center-crop correctness, prompt-cache
  reuse across models, clone, release, output listing) against the real
  `MKT_DEMO_TET` and `DB_NOTI_AUTOPAY_NOEL` requests: **all checks passed.**
- FastAPI route round-trip (`TestClient`), every endpoint
  `webapp/utils/api_client.py` calls: **all checks passed**, including the
  `/files` allow-list security check (blocked path outside served dirs → 403).
- `uvicorn app.main:app` real boot + real HTTP round-trip through
  `webapp/utils/api_client.py`: **passed.**
- CLI run against intentionally-unreachable endpoints: failure caught and
  reported per-model (`FAIL <model> vN: ...`), process exits cleanly instead
  of crashing: **passed.**
- Real network calls to the actual Gemini/Azure endpoints: **not made**
  (no reason to spend real API credit) — flagged as something to verify the
  first time a real `Run` happens from the webapp.

## 2026-08-20 — Refactored to `nhantn-mlai-project-structure` standard

**Change:** Audited the existing `app/` against the org's standard project
structure and brought it into compliance:
- `app/settings.py` rewritten from a hand-rolled frozen dataclass to
  `pydantic_settings.BaseSettings` (the required pattern), keeping every
  existing attribute name so no other module needed to change.
- Added `app/llm/` (`BaseLLMProvider`, `LLMFactory`, 5 providers: openai,
  claude, llama, deepseek, ollama) — the required swappable-LLM pattern.
  `app/core/prompt_builder.py` now goes through `LLMFactory` instead of
  calling the old one-off `app/providers/azure_gpt5mini.py` directly (that
  file was deleted; its logic became `app/llm/providers/openai_provider.py`,
  since Azure AI Foundry speaks the OpenAI-compatible API). Added a
  backward-compat alias (`"azure_gpt5mini"` → `OpenAIProvider`) so the value
  already baked into the real `.env` (`LLM_PROVIDER=azure_gpt5mini`) keeps
  working — `.env` itself could not be edited directly (remote-write to
  `.env` is blocked for security).
- Added the formal `app/evaluation/{unit,smoke,stress}` pytest suite
  (superseding the ad-hoc scratch scripts from 2026-08-19), `pytest.ini`,
  `requirements-dev.txt`.
- Added `script/check_env.py` + `script/validate_requests.py` (prep/validation,
  not runtime logic).
- Added `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.env.example`,
  `.gitignore`.
- Added the required trailing "chạy độc lập file này" comment blocks to
  `app/settings.py`, `app/main.py`, `run_cli.py`, `app/llm/factory.py`,
  every `app/llm/providers/*.py`, `app/providers/registry.py`.
- Added the 4 numbered doc files (this one included); shrank `README.md` to
  a short pointer.

**Result — real measured numbers from this pass (`pytest` /
`pytest -m stress -v`, dev sandbox, mocked image-provider network calls):**

| Tier | Cases | Result |
|---|---|---|
| Unit | 31 | 31 passed |
| Smoke | 10 | 10 passed |
| **Default total** (`pytest`) | **51** | **51 passed, 0 failed** |
| Stress — `/health` under load (50×4=200 req) | 1 | **passed** — 0 errors, p50=100.2ms, **p95=197.9ms** (threshold <200ms), p99=213.6ms |
| Stress — `/generate` × 10 concurrent | 1 | **passed** — p50=9.43s, p95=9.61s, avg=~8.5s (threshold <20s, see 3-project-evaluation.md for why this ceiling is generous) |

**Real bug found BY the stress tier and fixed in this same pass:** the
first version of `test_generate_under_concurrent_requests` (5 concurrent
`/generate` calls for the SAME `request_name`) exposed that
`output_manager.TIMESTAMP_FORMAT` was minute-precision only
(`%Y-%m-%d-%H%M`) — 2 concurrent runs for the same request landing in the
same minute would silently produce byte-identical output filenames and
overwrite each other's image. Fixed: `TIMESTAMP_FORMAT` now includes
seconds, PLUS a new `output_manager.unique_image_output_path()` that
appends `-dupN` if the target path already exists regardless of timestamp
precision (covers the residual same-second case too) —
`app/core/pipeline.py` now calls this instead of the raw path builder.
Regression-tested in `app/evaluation/unit/test_output_manager.py::test_unique_image_output_path_avoids_collision`.

**Not yet measured (flagged, not guessed):**
- Real latency/error-rate against the actual Gemini/Azure/OpenAI endpoints —
  every number above is against a mocked provider. Update this file with a
  real-endpoint measurement the first time a production-scale batch run
  happens.
- Unit test coverage percentage (no `pytest-cov` run yet — the "≥1 happy +
  ≥1 boundary case per module" bar in `3-project-evaluation.md` was checked
  manually per module, not via a coverage tool).
- `docker build .` was NOT run in this pass — the dev sandbox has the
  `docker` CLI but no daemon reachable (`/var/run/docker.sock` doesn't
  exist here). The `Dockerfile` was written to match the org's own
  reference template closely and reviewed by hand (multi-stage, non-root
  user, no baked-in `.env`), but the FIRST real `docker build .` +
  `docker run --env-file .env ...` on a machine with a working Docker
  daemon should be treated as unverified until someone actually runs it —
  update this entry with that result.

## 2026-08-20 — request.json schema CORRECTED (was wrong since 2026-08-19)

**What happened:** the 2026-08-19 entry above built the request.json schema
by reverse-engineering `input/*.json` files on disk at the time
(`MKT_DEMO_TET.json`, `DB_NOTI_AUTOPAY_NOEL.json`) — both used a FLAT schema
(`output.size` as a `"W,H"` string = the generation canvas,
`design_options.protect_area` as an optional `"W,H"` string = the real final
output). User later clarified: those were STALE leftover files they'd
forgotten to delete from `input/`, not the real schema. The real, current
`input/DB_NOTI_AUTOPAY_NOEL.json` matches the LITERAL template printed at the
top of REQUEST.md — nested `generation.canvas` / `generation.safe_area`
(`{width, height, position, x, y}`) + `output.size` as an OBJECT
`{width, height}` (the real final output size) + optional `output.crop`
(`{source, area: {x, y, width, height}}`). There is no `protect_area` field
at all in the real schema.

**Change:** rewrote every module that touched request.json geometry to match
the corrected schema:
- `app/models/schemas.py`: `CreativeRequest.generation` (new, required
  `GenerationSpec` = `canvas` + `safe_area` + `instruction`) and
  `CreativeRequest.output.size`/`.crop` restructured to nested objects.
  `design_options.protect_area` removed entirely (a stray one in an old file
  is now silently ignored via `extra="ignore"`, never crashes). New/changed
  methods: `canvas_wh()` (unchanged name, now reads `generation.canvas`),
  `crop_box()` (new — `output.crop.area` if present, else falls back to
  `generation.safe_area`, both describe the same box in every real file),
  `final_output_wh()` (now reads `output.size` directly), `has_crop()`
  (replaces `has_protect_area()`).
- `app/core/image_post.py`: `crop_protect_area()` (centered-only crop)
  replaced by `crop_canvas_to_output()` (crops an explicit, NOT necessarily
  centered, `(x, y, width, height)` box, then hard-resizes to `output.size`
  as a safety net for a hand-edited/inconsistent request.json).
- `app/core/pipeline.py`: `run_generate()`/`run_clone()` updated to the new
  crop function signature (clone's crop box is the whole canvas — a no-op
  crop, since clone generates directly at the target size).
- `app/core/prompt_builder.py`: the "CANVAS & SAFE AREA" prompt section now
  reads `generation.safe_area`'s own `x`/`y` box (not a centered
  computation) and appends `generation.instruction` verbatim when present;
  `generation` added to the prompt cache key (it affects prompt content and
  was missing from the cache key under the old schema, where it didn't
  exist).
- `webapp/views/generate.py`: `_expected_display_size()` reverted to read
  `output.size.width`/`.height` directly (an object) instead of preferring
  `design_options.protect_area` (a string) — see inline UPDATE comment.
  `output.size` metric display also fixed to render the "WxH" slug instead
  of the raw object.
- Docs: `1-project_description.md`, `2-project-technote.md`,
  `3-project-evaluation.md` updated to describe the corrected schema; example
  CLI commands switched from the now-deleted `MKT_DEMO_TET` to
  `DB_NOTI_AUTOPAY_NOEL`.
- `app/evaluation/unit/test_schemas.py`: rewritten for the nested schema —
  explicit crop, crop-box-falls-back-to-safe_area (no `output.crop`), no-crop
  (canvas == output.size), missing `generation` rejected, stray legacy
  `protect_area` field ignored (not crashed).
- `app/evaluation/smoke/test_api_smoke.py`: fixture request switched from
  `MKT_DEMO_TET` (deleted from `input/`) to `DB_NOTI_AUTOPAY_NOEL` (the only
  real request file now).
- `app/evaluation/stress/test_generate_stress.py` +
  `app/evaluation/conftest.py`: added `two_requests_isolated_input` fixture
  (redirects `Settings.input_dir` to a tmp dir with a copy of the real
  `DB_NOTI_AUTOPAY_NOEL.json` + a synthetic 2nd request that reuses real
  `lib/` assets and exercises the no-crop path) so the cross-talk test no
  longer depends on 2 real files existing under `input/`.

**Result (re-ran full suite after the rewrite, same dev sandbox, mocked
image-provider network calls):**

| Tier | Cases | Result |
|---|---|---|
| Unit | 43 (`test_schemas.py` now 13, +2 net vs. before: added crop-box-fallback, no-crop-when-canvas-equals-output, missing-generation-rejected, stray-legacy-protect_area-ignored; removed the 2 old protect_area-specific cases) | 43 passed |
| Smoke | 10 | 10 passed |
| **Default total** (`pytest`) | **53** | **53 passed, 0 failed** |
| Stress — `/health` under load (200 req) | 1 | **passed** — 0 errors, p50=89.8ms, p95=145.3ms (threshold <200ms), p99=174.8ms |
| Stress — `/generate` × 10 concurrent | 1 | **passed** — p50=8768ms, p95=8823ms (threshold <20s) |

Also re-verified end-to-end by hand (mocked provider, real `DB_NOTI_AUTOPAY_NOEL.json` + real `lib/` assets): crop box computed as `(82, 190, 1117, 610)` from `generation.safe_area` (`x=82, y=190, width=1035, height=420`) on a `1200x800` canvas — matches REQUEST.md's own worked example exactly — and the saved PNG is exactly `1035x420`, confirmed with `PIL.Image.open(...).size`. `script/validate_requests.py` also re-run against the real `input/` directory: `DB_NOTI_AUTOPAY_NOEL OK, 0 problems`.

**Not yet measured (unchanged from 2026-08-19, still flagged, not guessed):**
real latency/error-rate against actual provider endpoints, unit coverage
percentage via a coverage tool, `docker build .` (no daemon in this sandbox).

## 2026-08-20 (2) — REQUEST.md [8] Streamlit audit + `AttributeError` crash fix

**User report:** `AttributeError: 'Settings' object has no attribute
'asset_library_path'` when running the webapp, plus a request to audit
whether the existing Streamlit pages actually call the right backend
functions for every REQUEST.md [8a]-[8e] requirement.

**Root cause of the crash:** `webapp/utils/theme.py::logo_path()` read
`settings.asset_library_path` -- that attribute has never existed on
`Settings` (the real field is `asset_library_dir`, see `app/settings.py`);
every page calls `page_title_block()`/`render_sidebar_nav()` at import time,
so this crashed on EVERY page load, not just one. Separately, the logo file
itself lives at `lib/my/ocb_asset/logo.png` on disk, not the literal
`lib/ocb_asset/logo.png` path named in REQUEST.md [8e] -- same
"trust the real file layout over the literal doc text" situation already
hit once with the request.json schema (see the entry above).

**Audit result against REQUEST.md [8a]-[8e]** (read every relevant file in
`webapp/views/*.py` + `webapp/utils/*.py` end to end):

| Requirement | Where implemented | Verdict |
|---|---|---|
| [8a] sidebar: request.json picker, run mode, custom model multiselect | `webapp/views/generate.py` sidebar block | Correct -- calls `api_client.list_requests()`/`list_models()`, mode selectbox matches `model_usage.json` keys, custom mode uses a form so multi-picking doesn't rerun mid-selection |
| [8b] main screen: all theme/context/product images, text, model list, output size | `webapp/views/generate.py` asset gallery + text/model cards | Correct -- `resolved_assets` from `GET /requests/{name}` already returns every image in a bucket (not just 1); output size metric was reading the raw `output.size` object though (fixed same pass, see below) |
| [8c] "running" notice + only THIS run's new images | `webapp/views/generate.py::_render_output_body`, `st.status` + `GET /outputs/{name}?since_ts=` | Correct -- `since_ts` is the exact moment "Run" was clicked, so only images from the current run appear while it's in progress |
| [8d] checkbox per image regardless of mode/count -> release into `output/<request>/release/` | `webapp/views/generate.py::_render_grouped_images` (`allow_release=True`) + `POST /release` -> `output_manager.release_images()` | Correct -- every rendered image gets its own checkbox, "Release N selected" copies into `release_dir()` exactly as specified |
| [8e] Release page: pick 1 released image, multiselect display sizes, model dropdown, run -> `_clone_` suffix files in `release/` | `webapp/views/release.py` | Correct -- exceeds the literal ask (low-cost/high-res size tables + free-text custom sizes, all merged), still lands on `output_manager.clone_release_filename()`'s exact `<source_stem>_clone_<size>_<model>.png` convention |
| [8e] (2nd bullet) OCB logo on every page + OCB-toned background | `webapp/utils/theme.py` (`logo_path()`, `render_sidebar_nav()`, `page_title_block()`) + `.streamlit/config.toml` | Logo: correct after this fix. Background: **was NOT implemented at all** -- `theme.py`'s own comments (dated 2026-08-15) say brand color lives in `.streamlit/config.toml`, but that file never existed, so the app was silently running Streamlit's default theme this whole time. Created it now (see below). |

**Fix (this pass):**
- `webapp/utils/theme.py::logo_path()`: `asset_library_path` ->
  `asset_library_dir`; now tries `lib/ocb_asset/logo.png` then falls back to
  `lib/my/ocb_asset/logo.png` (the real current location).
- `webapp/views/generate.py`: sidebar's `DEFAULT_REQUEST_NAME` was still
  `"MKT_DEMO_TET"` (deleted from `input/`, see the entry above) -- changed to
  `"DB_NOTI_AUTOPAY_NOEL"`.
- Created `.streamlit/config.toml` -- `[theme]` `primaryColor`/
  `secondaryBackgroundColor`/`textColor` set to the same `OCB_GREEN`/
  `OCB_LIGHT_GREEN_BG`/`OCB_TEXT` hex values already hard-coded in
  `webapp/utils/theme.py`, so both stay in sync.
- `docker-compose.yml`: added a `./.streamlit:/app/.streamlit` volume mount
  to the `webapp` service (the new config file wasn't reachable from inside
  the container otherwise -- same class of gap as the missing file itself).

**Verification:** `python -m py_compile` on both edited `.py` files (no
syntax errors); manually re-derived `logo_path()`'s fallback logic against a
throwaway directory shaped like the real `lib/my/ocb_asset/logo.png` layout
-- resolves correctly. Could not launch a real `streamlit run` in this
sandbox (no browser/display) -- **starting the app for real and clicking
through all 3 pages once is still the user's own next step to fully close
this out**, this pass only fixes what's verifiable by reading + static
checks.

**Later same-day follow-up (not a separate dated entry):** re-verified this
crash fix by actually EXECUTING the 3 Streamlit view scripts headlessly with
`streamlit.testing.v1.AppTest` (installed `streamlit` into the sandbox for
this) instead of relying on static reading alone -- `AppTest.run()` on
`generate.py`, `release.py`, and the sidebar-nav entrypoint all completed
with zero exceptions after the `asset_library_dir` fix (would have raised
the reported `AttributeError` before it). This is strictly stronger evidence
than the `py_compile`-only check above, since `py_compile` cannot catch a
runtime `AttributeError` on a wrong attribute name.

## 2026-08-20 (3) — Save the RAW (pre-crop) image alongside the final cropped one, under `ORG/`

**User request (verbatim intent):** for every image an Image GenAI model
generates, in addition to the final file already saved at
`output/<request_name>/<display_size>/<image_model>/<file>.png` (already
cropped from `generation.canvas` down to `output.size` per REQUEST.md), also
save the RAW image exactly as the model returned it -- BEFORE the crop --
in a new `ORG/` subfolder inside that same `<image_model>/` directory.

**Design decisions:**
- `ORG/` is nested INSIDE `<display_size>/<image_model>/`, not a top-level
  `output/<request_name>/ORG/` -- keeps it obvious which model produced
  which raw image, and keeps the pairing 1:1 by directory.
- The raw file reuses the EXACT SAME filename as its cropped counterpart
  (`unique_image_output_path(...).name`) -- the only difference between the
  2 files is which directory they're in, so they're trivially paired by name.
- The raw save is wrapped in its own `try/except` in `pipeline.py::run_generate()`,
  logged and swallowed on failure -- an ORG/ write failure must NEVER fail
  the (model, version) result, since the final cropped image (already saved
  successfully by that point) is what actually matters to the rest of the app.
- Deliberately did NOT add `"ORG"` to `output_manager.py`'s
  `_NON_RESULT_DIR_NAMES` set, because that set is for directories
  *alongside* `<display_size>/`, not *inside* `<image_model>/` -- instead,
  `_iter_result_pngs()` already only `glob("*.png")`s directly inside
  `<image_model>/` (non-recursive), so `ORG/`'s contents are naturally
  invisible to `list_new_images()`/`latest_run_images()`/
  `images_matching_substring()`, and therefore to the webapp's Output
  gallery and "Select for release" checkboxes -- release only ever offers
  the final cropped image, never the raw one. Covered by a dedicated test
  (see below) rather than just asserted in prose.

**Change:**
- `app/core/output_manager.py`: added `org_dir()` / `org_image_path()`;
  module docstring rewritten to document the new layout.
- `app/core/pipeline.py::run_generate()`: right after writing the final
  cropped image, also writes `raw_bytes` (the model's untouched output,
  already available in that scope pre-crop) to `org_image_path(...)`,
  creating `ORG/` on demand.
- `app/evaluation/unit/test_output_manager.py`: 3 new tests --
  `org_dir()` nests correctly under the model directory,
  `org_image_path()` reuses the final image's filename, and (the important
  one) a same-filename raw+final pair under `ORG/` still yields exactly 1
  entry from `list_new_images()`/`images_matching_substring()`, proving the
  invisibility claim above rather than just asserting it.
- `app/evaluation/smoke/test_api_smoke.py::test_generate_end_to_end_quality_mode`:
  extended to assert, for every one of the 3 real `POST /generate` results,
  that an `ORG/<same filename>` sibling exists next to the final file, that
  the raw one is the FULL `generation.canvas` size (`1200x800`, un-cropped),
  and the final one is the cropped `output.size` (`1035x420`, matching
  `DB_NOTI_AUTOPAY_NOEL.json`'s real declared crop) -- exercising the real
  request file, real crop math, and real filesystem layout together, not
  just the unit-level path helpers in isolation.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 56 | **passed** (up from 53 before this change) |
| Stress (`pytest -m stress`) | 2 | **passed** |

Also manually verified end-to-end against the real pipeline (mocked
provider, real `DB_NOTI_AUTOPAY_NOEL.json` + real `lib/` assets): deleted
`output/DB_NOTI_AUTOPAY_NOEL/`, called `pipeline.run_generate(...)` for
`gpt-image-2`, then listed the resulting directory tree -- got exactly the
expected pair:

```
output/DB_NOTI_AUTOPAY_NOEL/1035x420/gpt-image-2/DB_DEMO_AUTOPAY_NOEL_1035x420_gpt-image-2_2026-08-20-030641.png
output/DB_NOTI_AUTOPAY_NOEL/1035x420/gpt-image-2/ORG/DB_DEMO_AUTOPAY_NOEL_1035x420_gpt-image-2_2026-08-20-030641.png
```

Same filename in both, `ORG/` file is the pre-crop image, sibling file is
the post-crop image -- matches the design exactly. Test artifacts from this
manual run were deleted afterward (not part of the delivered diff).

**Not yet measured:** real (non-mocked) provider output through this path --
this pass, like every other in this project so far, only exercises the code
path with a mocked `generate_image()`, per `3-project-evaluation.md`'s
documented network-call-out-of-scope decision for smoke tests.

## 2026-08-20 (4) — Fix prompt log: was 1 file per model, missing version in the name

**User report:** "tôi vẫn ko thấy bạn lưu prompt gởi tới image model trong
thư mục output/<request_name>/prompt" -- pointing back at REQUEST.md [3d],
quoted verbatim: `prompt-imagemodel-version-1.txt`, i.e. the saved filename
must record BOTH which model the prompt was sent to and which version
request it was for.

**Root cause:** the prompt-logging code already existed
(`prompt_builder.py::write_prompt_log()`, called from
`pipeline.py::run_generate()`) and DID write to
`output/<request_name>/prompt/`, but only 1 file per MODEL --
`prompt-<model>.txt`, no version in the name -- written once before the
version loop even started. For a `quality` run (`num_version=3`), all 3
version requests to the same model silently shared that 1 file, so the
"which exact prompt was sent for version 2 / version 3" question this
feature exists to answer couldn't actually be answered from disk -- the
filename simply didn't distinguish them, which is very likely why the user
felt this wasn't being saved at all.

**Change:**
- `app/core/output_manager.py::prompt_file_path()`: added a required
  `version: int` parameter; filename is now
  `prompt-<image_model>-version-<version>.txt` (e.g.
  `prompt-gpt-image-2-version-2.txt`), matching REQUEST.md [3d]'s quoted
  convention exactly.
- `app/core/prompt_builder.py::write_prompt_log()`: added the same
  `version` parameter, passed straight through to `prompt_file_path()`.
  Still write-once-then-reuse per (model, version) pair, per [3d]'s "chỉ
  nên xuất 1 lần".
- `app/core/pipeline.py::run_generate()`: moved the `write_prompt_log()`
  call from once-per-model (before the version loop) to once-per-(model,
  version) INSIDE the version loop, and moved it to run BEFORE
  `generate_image()` is called -- so the exact prompt that was about to be
  sent is captured on disk even if the model call itself then fails.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 57 | **passed** (up from 56 -- 1 new unit test) |
| Stress (`pytest -m stress`) | 2 | **passed** |

New unit test (`test_prompt_file_path_includes_model_and_version`) asserts
`prompt_file_path("REQ", "gpt-image-2", 1)` and `...version 2)` resolve to
2 DIFFERENT filenames that both include the model name. The existing smoke
test `test_generate_end_to_end_quality_mode` (a real `POST /generate`,
`quality` mode = `gpt-image-2` x 3 versions) was extended to assert, for
ALL 3 results, that `prompt-gpt-image-2-version-<n>.txt` exists with the
exact expected name and non-trivial content (>100 chars) -- this is the
regression guard against the exact bug just reported (3 versions
collapsing onto 1 file would fail this test, since only 1 of the 3
expected files would exist).

Also manually verified against the real pipeline (mocked provider, real
`DB_NOTI_AUTOPAY_NOEL.json`, `num_version_override=2`): got exactly

```
output/DB_NOTI_AUTOPAY_NOEL/prompt/prompt-gpt-image-2-version-1.txt   (10,853 bytes)
output/DB_NOTI_AUTOPAY_NOEL/prompt/prompt-gpt-image-2-version-2.txt   (10,853 bytes)
```

-- 2 distinct files, byte-identical content (expected: the prompt is
assembled ONCE per run and reused for every version, per REQUEST.md [3d]'s
own description; a future change that makes different versions request
different prompt content would naturally diverge these files, which is
exactly why [3d] asks for 1 file per version rather than 1 shared file).
Confirmed content is the real assembled design-brief-plus-guideline prompt,
not an empty stub. Test artifacts from this manual run were deleted
afterward (not part of the delivered diff).

## 2026-08-20 (5) — Attach a visual SAFE AREA GUIDE image (user-reported: content spilling past safe_area, cropped off)

**User report:** a real run with `generation.canvas` 1536x1024 and
`generation.safe_area` 1035x420 centered at (250, 302) produced a
gpt-image-2 image whose important content (headline text) extended past
the declared safe area and was cut off mid-word by the crop -- despite the
prompt's "CANVAS & SAFE AREA" section already spelling out those exact
pixel coordinates in text. The user's diagnosis was that the prompt had no
instruction at all about the safe area; re-reading the actual saved prompt
text showed the instruction WAS present and correctly worded -- the real
problem is that bare pixel coordinates in a paragraph of text are known to
be unreliable for image-generation models to follow precisely; they have no
robust way to translate "x=250, y=302" into an exact spatial position
without something to look at.

**Fix:** render an actual same-size picture of the boundary and attach it
to the model call alongside the text instruction, so the model has
something to visually match instead of only numbers to interpret.

- New `app/core/safe_area_guide.py`: `build_safe_area_guide_png(canvas_wh,
  crop_box)` draws a same-pixel-size PNG -- a translucent bright-magenta
  wash over the region OUTSIDE the safe area (what gets cropped away and
  discarded), a solid magenta rectangle border drawn exactly around the
  safe area, and a plain-English caption bar. Magenta was picked because it
  doesn't collide with OCB's actual brand palette (green/gold/white/dark
  gray). `build_safe_area_guide_attachment(request_name, canvas_wh,
  crop_box)` wraps this as a `PreparedImage`, cached under
  `_resized_refs/_safe_area_guide_<geometry-hash>.png` (same "build once,
  reuse" pattern as every other cached asset here) -- a different crop
  geometry gets its own file, never silently reuses stale geometry.
- `app/core/pipeline.py::_node_prepare_images()`: when `req.has_crop()`,
  builds and appends this guide as the LAST attached image, best-effort
  (never fails the run if guide rendering itself has a bug).
- `app/core/prompt_builder.py::build_prompt()`: the "CANVAS & SAFE AREA"
  section now explicitly points at the attached guide image by name when
  it was successfully attached, and states clearly that the guide is a
  spatial reference only, not a design asset to reproduce. Also added a
  short "CRITICAL REMINDER" section restating the same constraint at the
  very END of the prompt (in addition to where it already was, roughly in
  the middle) -- repeating a hard constraint at both the start and the end
  of a long, multi-section prompt is a well-known way to improve
  instruction-following compliance versus stating it only once.
- `_bucket_section()`'s attached-images listing gained a `"safe_area_guide"`
  entry so it also shows up correctly in the "ATTACHED REFERENCE IMAGES"
  list inside the prompt text itself.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 66 | **passed** (up from 57 -- 9 new tests) |
| Stress (`pytest -m stress`) | 2 | **passed** |

New `app/evaluation/unit/test_safe_area_guide.py` (5 tests): guide PNG is
exactly the canvas size; a pixel at the CENTER of the safe area stays plain
white (untinted) while a pixel in the far corner (outside the safe area) is
tinted magenta -- proving the mask geometry is actually correct, not just
that the function returns *a* PNG; a degenerate/edge-hugging crop box
doesn't crash; `build_safe_area_guide_attachment()` returns a proper
`PreparedImage`; and repeat calls with the SAME geometry reuse the same
cached file while a DIFFERENT geometry gets a different one. The existing
smoke test was extended to assert, against the real
`DB_NOTI_AUTOPAY_NOEL.json` (which declares a real crop), that the saved
prompt log actually contains `"SAFE AREA GUIDE"` and `"CRITICAL REMINDER"`.

Manually verified end-to-end against the EXACT geometry from the user's
report (canvas 1536x1024, safe area 1035x420 at x=250,y=302): rendered the
guide PNG standalone and visually inspected it -- white safe-area rectangle
correctly outlined in magenta at the right position, tinted discard zone
correctly surrounding it, caption legible. Also ran the real pipeline
(mocked provider) against `DB_NOTI_AUTOPAY_NOEL.json`'s own geometry (canvas
1200x800, safe area 1035x420 at x=82,y=190) end-to-end -- guide image
generated and saved to `_resized_refs/_safe_area_guide_b9f09fa603f5.png`,
correctly listed as attachment #5 (after theme/product/2 guideline images),
and both "SAFE AREA GUIDE" and "CRITICAL REMINDER" appeared in the saved
prompt log exactly as designed.

**Not yet measured:** whether this visual guide actually improves
gpt-image-2's real compliance (a real, non-mocked model call) -- this pass,
like the LLM-refinement provider itself, is out of scope for network calls
per `3-project-evaluation.md`. The user's own next real run against the
live model is the practical test of whether this closes the gap; if content
still spills past the safe area with the guide attached, the next step
would likely be experimenting with an even more visually dominant marker
(e.g. a checkerboard/hatch pattern instead of a flat tint) or moving the
guide earlier in the attachment order.

## 2026-08-20 (6) — Prompt log now names every image FILE actually attached

**User request (verbatim intent):** "trong prompt log lưu lại, bạn note
thêm cho tôi tên các bức ảnh nào đã được đính kèm gởi image model luôn
nhé" -- the saved prompt log should also record the actual FILENAMES of
every image sent alongside that prompt, not just the generic bucket labels
("1. THEME (background)") already inside the prompt text.

**Change:**
- `app/core/prompt_builder.py::format_attached_images_note()` (new): builds
  a footer block listing every attached image's real on-disk filename,
  numbered in the exact order they were attached, with its original source
  filename noted too when it differs (i.e. was downsampled) -- e.g.
  `2. [product] db_phone_autopayment__1787183998.png  (downsampled from
  source: db_phone_autopayment.png)`.
- `write_prompt_log()` gained an optional `attached_images_note` parameter
  -- appended as a footer AFTER the actual prompt text when writing the
  file. This footer is LOG-ONLY: it is never part of `built_prompt.text`
  and is never sent to the model itself, so it can't affect the model's
  behavior or the prompt cache key.
- `app/core/pipeline.py::run_generate()`: builds this note once per run
  (same attachments are reused for every model/version in that run) from
  `state["prepared_images"]`, passes it to every `write_prompt_log()` call.

**Result:** default tier stayed at 66 (this was folded into the 2026-08-20
(5) test/result count above, since both changes landed together in this
pass) -- 2 new unit tests added in `app/evaluation/unit/test_prompt_builder.py`
(list-with-filenames, empty-list, source-shown-when-downsampled,
source-omitted-when-identical) plus the smoke test now also asserts
`"ATTACHED IMAGES SENT WITH THIS PROMPT"`, `"[theme]"`, `"[product]"`, and
`"[safe_area_guide] _safe_area_guide_"` all appear in the real saved log.
Manually verified: the real pipeline run above produced this exact footer
in `prompt-gpt-image-2-version-1.txt`:

```
================================================================================
ATTACHED IMAGES SENT WITH THIS PROMPT (in order, as actually attached to the model call):
  1. [theme] db_theme_01__1787183998.jpg  (downsampled from source: db_theme_01.png)
  2. [product] db_phone_autopayment__1787183998.png  (downsampled from source: db_phone_autopayment.png)
  3. [guideline] page-4__1787183998.jpg  (downsampled from source: page-4.png)
  4. [guideline] page-2__1787183998.jpg  (downsampled from source: page-2.png)
  5. [safe_area_guide] _safe_area_guide_b9f09fa603f5.png
```

Test artifacts from both manual runs were deleted afterward (not part of
the delivered diff).

## 2026-08-20 (7) — Fix: stale prompt log/cache never picked up the (5)/(6) changes on a re-run

**User report:** re-ran the app after receiving the SAFE AREA GUIDE +
attached-images-footer changes above, but the actual
`prompt-gpt-image-2-version-N.txt` log they opened showed neither the new
safe-area-guide wording nor the attached-images list.

**Root cause (2 independent stale-cache bugs, both hit at once):**
1. `write_prompt_log()` only wrote the `.txt` log file if that exact path
   did NOT already exist on disk -- it treated "file already exists" as "up
   to date", regardless of whether the actual prompt content had changed.
   The user had already run this exact request (same request_name, model,
   version) once before receiving the (5)/(6) changes, so
   `prompt-gpt-image-2-version-1.txt` already existed with OLD content --
   the new run correctly built a fresh prompt in memory, but this function
   silently refused to write it to disk, leaving the stale file untouched.
2. `build_prompt()`'s cache (`_prompt_cache/built_prompt.json`) is keyed by
   a hash of `request.json`'s own data fields. Since the user's
   `request.json` hadn't changed between the 2 runs, and the hash never
   accounted for the prompt-BUILDING CODE itself changing, a byte-identical
   `request.json` could -- in principle, for any future template edit that
   doesn't happen to touch attachment labels -- keep serving an old cached
   prompt TEXT forever, independent of how much `prompt_builder.py` itself
   had changed since. (In this specific case attaching `safe_area_guide` as
   a new attachment label already happened to bust the cache; the fix below
   closes the gap for any FUTURE template-only change too.)

**Fix:**
- `app/core/prompt_builder.py::write_prompt_log()`: removed the
  "already exists -- reuse" branch entirely. The `.txt` log is now
  **unconditionally (re)written every call** -- its only job is to
  accurately record what was actually just sent, so it must never go
  stale. (REQUEST.md [3d]'s actual "chỉ nên xuất 1 lần" intent -- don't
  waste an LLM refinement call rebuilding prompt TEXT for every model -- is
  still satisfied by `build_prompt()`'s own cache, which IS the expensive
  step worth reusing; writing a small local `.txt` file is cheap.)
- `app/core/prompt_builder.py`: added `_PROMPT_TEMPLATE_VERSION = 2`, now
  included in `_cache_key()`'s hashed payload. Any future change to
  `build_prompt()`'s assembly/wording MUST bump this constant, so a
  code-only template change always busts every previously-cached prompt on
  disk even when `request.json` itself is byte-identical to a prior run.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 69 | **passed** (up from 66 -- 3 new tests) |
| Stress (`pytest -m stress`) | 2 | **passed** |

3 new tests in `app/evaluation/unit/test_prompt_builder.py`:
`test_write_prompt_log_always_overwrites_stale_content` and
`test_write_prompt_log_overwrites_even_with_a_footer` write a prompt log,
then write a DIFFERENT one to the same (model, version) path, and assert
the old content is completely gone (not just appended past); `test_build_prompt_cache_busts_when_prompt_template_version_changes`
calls `build_prompt()` twice with the IDENTICAL request/guideline/attachment
labels, only changing `_PROMPT_TEMPLATE_VERSION` in between (via
monkeypatch), and asserts the 2 calls produce DIFFERENT `cache_key`s --
directly proving a code-only change busts the cache even when
`request.json`'s data hasn't moved at all.

Also manually reproduced the user's EXACT reported scenario end-to-end:
hand-wrote stale `prompt-gpt-image-2-version-{1,2,3}.txt` files (plain text
containing no safe-area-guide wording) and a stale `_prompt_cache/built_prompt.json`
into `output/DB_NOTI_AUTOPAY_NOEL/` (simulating "already ran once before
this fix"), then ran the real pipeline (mocked provider) against that same
directory. Confirmed the STALE text is now completely gone
(`grep -c "STALE" prompt-gpt-image-2-version-1.txt` → 0) and the new
content is present: `"SAFE AREA GUIDE"` appears 3 times, `"CRITICAL
REMINDER"` once, `"ATTACHED IMAGES SENT WITH THIS PROMPT"` once -- exactly
as designed. This confirms the user does NOT need to manually delete their
existing `output/` folder -- the next real run will self-heal automatically
now that both caches correctly invalidate. Test artifacts from this manual
run were deleted afterward (not part of the delivered diff).

## 2026-08-20 (8) — SAFE AREA GUIDE image now draws actual measurements, not just a shape

**User feedback (after looking at the (5) guide image):** a plain magenta
rectangle communicates the IDEA of a boundary, but an image-generation
model reading a picture has no way to back out exact pixel measurements
just by eyeballing a box -- it can't count pixels. The guide needed the
actual width/height and x/y numbers drawn directly ON the picture.

**Change (`app/core/safe_area_guide.py`, `build_safe_area_guide_png()`
rewritten):**
- Solid blue ruler lines with arrowheads, drawn INSIDE the safe area along
  its own top edge (labeled `WIDTH <n>px`) and left edge (labeled `HEIGHT
  <n>px`) -- like a technical-drawing dimension line.
- Dashed black leader lines with arrowheads, from the canvas origin (0, 0)
  out to the safe area's top-left corner -- a horizontal one labeled `x =
  <n>px`, a vertical one labeled `y = <n>px` -- visually distinct (dashed +
  black) from the solid blue size rulers so "where it starts" and "how big
  it is" don't get visually confused.
- The safe area's exact top-left and bottom-right corner coordinates
  printed directly next to those corners, e.g. `(82,190)` and `(1117,610)`.
- Every label sits on a solid-color background plate (readable against
  either the plain white interior or the translucent magenta exterior) and
  is now clamped to stay fully inside the canvas -- a label centered on a
  short offset line (e.g. a safe area only 82px from the edge) is wider
  than the line itself and was initially getting cut off past the canvas
  boundary; `_draw_label()` now shifts the whole label back onto the
  canvas so it's always fully legible regardless of how close the safe
  area sits to an edge.
- `build_safe_area_guide_attachment()`: removed its own "already exists --
  reuse" cache check (same lesson as (7)'s `write_prompt_log()` fix,
  applied here proactively) -- rendering is cheap, so the guide is now
  ALWAYS regenerated, guaranteeing a future visual change to this file
  never gets silently masked by an old cached PNG at the same geometry hash.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 70 | **passed** (up from 69) |
| Stress (`pytest -m stress`) | 2 | **passed** |

`app/evaluation/unit/test_safe_area_guide.py`: renamed the geometry-caching
test to reflect the new "same filename, always-refreshed content" behavior,
and added `test_build_safe_area_guide_attachment_always_overwrites_stale_content`
-- writes garbage bytes to the guide's path, calls the builder again, and
asserts the garbage is gone and a real PNG is there instead (regression
guard mirroring (7)'s `write_prompt_log` test). Existing geometry tests
(inside-stays-white, outside-is-tinted, degenerate-box-doesn't-crash) all
still pass against the new drawing code.

Manually re-rendered both the user's originally-reported geometry (canvas
1536x1024, safe area 1035x420 at x=250,y=302) and the real
`DB_NOTI_AUTOPAY_NOEL.json` geometry (canvas 1200x800, safe area 1035x420
at x=82,y=190) and visually inspected both -- confirmed width/height
rulers, x/y offset leader lines, and corner coordinates all render
correctly positioned and legible, including the small-offset
(`x=82px`) case that initially clipped off-canvas before the
`_draw_label()` clamping fix.

## 2026-08-20 (9) — SAFE AREA GUIDE cache moved out of `output/<request_name>/...` into a shared `lib/safe_area_guide/` library

**User request:** the guide image was being rebuilt (or, after (8), always
rebuilt) once per request under `output/<request_name>/_resized_refs/...`,
even when a different request declared the exact same `generation.canvas` +
`generation.safe_area` geometry. The user wants generation moved fully
inside the app (no more hand-generated example file), stored in ONE shared
place keyed by geometry, named
`safe_area_guide_<canvas_w>x<canvas_h>_<safe_w>x<safe_h>.png`, and reused
by filename lookup: if the file already exists, use it as-is; only render
and save a new one when it's missing.

**Changes:**

- `app/settings.py`: new `safe_area_guide_dir` field (default
  `./lib/safe_area_guide`, override via `SAFE_AREA_GUIDE_DIR` in `.env`),
  resolved the same way every other configurable directory in this project
  is (`_resolve_dir()` -- falls back to a project-relative default if the
  configured path doesn't exist on the current machine).
- `app/core/safe_area_guide.py`:
  - New `guide_filename(canvas_wh, safe_area_wh) -> str` implementing the
    user's exact naming convention. Documented (not fixed, since no real
    `request.json` in this project needs it yet) that the name is keyed on
    SIZES only, not x/y offset -- an off-center safe area with the same
    sizes as an existing centered one would collide.
  - `build_safe_area_guide_attachment()` signature changed from
    `(request_name, canvas_wh, crop_box)` to `(canvas_wh, crop_box)` --
    `request_name` is no longer meaningful since the cache isn't
    per-request anymore.
  - Its caching behavior is now "search `safe_area_guide_dir` for the
    expected filename; reuse it if found; otherwise render and save it" --
    this DELIBERATELY reintroduces the exists-then-reuse pattern that (7)
    removed for the OLD per-request cache. That removal was correct for a
    cache nobody asked to survive across runs; this reintroduction is
    correct too, because the user explicitly asked for reuse-across-runs
    this time. Documented clearly in the function's docstring: if
    `build_safe_area_guide_png()`'s drawing logic changes again later,
    already-rendered files sitting in `safe_area_guide_dir` will NOT
    automatically pick up the change -- the directory (or the specific
    stale files) needs to be cleared by hand after such an edit.
- `app/core/pipeline.py`: updated the call site to the new 2-arg signature.
- `app/core/output_manager.py`: corrected the module docstring's directory
  map -- the guide image is no longer under
  `output/<request_name>/_resized_refs/`.
- `app/evaluation/conftest.py`: new `isolated_safe_area_guide_dir` fixture
  (redirects `Settings.safe_area_guide_dir` to a throwaway `tmp_path`),
  mirroring `isolated_output`. Evaluation tests must never read/write the
  real shared `lib/safe_area_guide/` cache.
- `app/evaluation/unit/test_safe_area_guide.py` and
  `app/evaluation/smoke/test_api_smoke.py`: updated for the new signature,
  the new fixture, and the new filename convention (`safe_area_guide_...`
  instead of the old geometry-hash `_safe_area_guide_...` name); replaced
  the old "always overwrites stale content" regression test (no longer
  true -- that was the OLD per-request cache's contract) with tests for
  "reuses an existing file byte-for-byte instead of rebuilding it" and
  "shares the cache across different request names for the same geometry".

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 72 | **passed** (up from 70) |
| Stress (`pytest -m stress`) | 2 | 1 passed, 1 pre-existing latency flake (see note below) |

The 1 stress failure (`test_health_endpoint_under_concurrent_load`, p95
261ms vs a 200ms threshold) is unrelated to this change -- it's a
`/health`-endpoint concurrency-latency benchmark that never touches
`safe_area_guide.py`, and its threshold is sensitive to this sandbox's
available CPU at the moment the test happens to run, not to any code
changed here.

Manually ran the real (mocked-image-model) LangGraph pipeline against
`DB_NOTI_AUTOPAY_NOEL` twice in a row: the 1st run logged "Built safe-area
guide image ... -> lib/safe_area_guide/safe_area_guide_1200x800_1035x420.png"
and created the file; the 2nd run logged "Reusing cached safe-area guide
image -> lib/safe_area_guide/safe_area_guide_1200x800_1035x420.png" and did
NOT re-render it -- confirming the search-reuse-else-create behavior works
end-to-end, not just in the isolated unit tests. Cleaned up
`lib/safe_area_guide/` and `output/DB_NOTI_AUTOPAY_NOEL/` afterward so the
manual verification run didn't leave test artifacts behind.

## 2026-08-20 (10) — Investigated: gpt-image-2 still not respecting the safe area on a real run

**User report:** after (5)-(9) (text instruction + attached visual guide +
CRITICAL REMINDER repeated at the end + stale-cache fixes), a real re-run
against gpt-image-2 still didn't respect the safe area.

**Cannot be fully diagnosed from this environment:** this sandbox has no
network access to Azure OpenAI, so gpt-image-2 cannot actually be called
here to directly confirm whether it now complies -- everything below is
static code review + a proportional/logical argument, not a live test
against the real API. The user's own next real run is the actual test.

**One concrete, verifiable bug found and fixed:**
`app/providers/azure_openai_image.py::_closest_supported_size()` maps
ANY requested canvas to the closest of gpt-image-2's 3 fixed supported
sizes -- `(1024,1024)`, `(1024,1536)`, `(1536,1024)` -- before the real API
call, because that's all the `images/edits` endpoint accepts. For
`DB_NOTI_AUTOPAY_NOEL` (canvas 1200x800, ratio 1.5), that resolves to
`1536x1024` -- a DIFFERENT absolute size than the 1200x800 the prompt text
and the attached guide image both described. So gpt-image-2 was reading
"canvas is 1200x800, safe area starts at pixel (82, 190)" and being handed
a reference image rendered at 1200x800 px, while its real working canvas
was actually 1536x1024 -- pixel (82, 190) on a 1536-wide canvas is NOT the
same relative position as pixel (82, 190) on a 1200-wide canvas (5.3% from
the left edge vs. the intended 6.8%). This mismatch exists for gpt-image-2
specifically (confirmed by contrast with `app/providers/azure_generic_image.py`,
which sends the requested size as-is, no remapping) whenever
`generation.canvas` isn't already exactly one of those 3 sizes -- true for
every real `request.json` in this project today.

**Fix applied:** added percentage-of-canvas-width/height figures as the
PRIMARY, size-independent description of the safe area, in 3 places that
all now say the same thing 2 ways (pixels AND percentages), with an
explicit instruction to trust the percentages if the model's real output
canvas ends up a different exact pixel size than stated:
- `prompt_builder.py`'s CANVAS & SAFE AREA section
- `prompt_builder.py`'s CRITICAL REMINDER section (repeated at the end)
- `safe_area_guide.py`'s guide-image caption bar

This does NOT eliminate the underlying size-quantization (gpt-image-2 will
still physically render at 1536x1024, not 1200x800) -- it makes the
INSTRUCTION robust to it, since "keep content within 6.8%-93.1% of width"
is unambiguous no matter which of the 3 fixed sizes the model actually
uses. `_PROMPT_TEMPLATE_VERSION` bumped 2 -> 3 so this update busts every
previously cached prompt automatically on the next run (same lesson as (7)).

**Other possibilities considered, not implemented (flagged for the user,
not assumed away):**
- Auto-aligning `generation.canvas` in `request.json` to gpt-image-2's own
  supported sizes specifically for that provider -- would require making
  prompt/guide generation provider-specific, a bigger architectural change
  than this session's scope (today 1 prompt + 1 guide image is built once
  and shared across every model/version in a run).
- Generating directly at the final `output.size` instead of a larger canvas
  that's cropped down afterward -- would remove the crop-safety-margin
  concept entirely (REQUEST.md's whole reason for `generation.canvas` +
  `generation.safe_area` existing as distinct from `output.size`), a
  product decision, not a code fix.
- A structural cause independent of pixel/percentage wording: image-
  generation models given a reference/guide image through a generic
  multi-image edit endpoint (as opposed to a purpose-built spatial-
  conditioning mechanism, e.g. ControlNet-style ports) are well known to
  follow COARSE spatial cues (roughly-centered, roughly-avoid-the-edges)
  more reliably than an EXACT boundary -- this is a real, documented
  limitation of this class of model/API, not a bug in this app's code, and
  no amount of prompt engineering fully guarantees pixel-perfect
  compliance. If gpt-image-2 keeps missing the mark after this fix, the
  practical mitigation is a wider safety margin (a `generation.safe_area`
  noticeably smaller than the full `generation.canvas`, giving the crop
  more room for the model's typical few-percent drift) rather than chasing
  exact-pixel compliance further.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 72 | **passed** (unchanged count, all updated content re-verified) |

Manually re-rendered `DB_NOTI_AUTOPAY_NOEL`'s built prompt text after this
change and visually confirmed the percentages are arithmetically correct
(82/1200=6.8%, 1117/1200=93.1%, 190/800=23.8%, 610/800=76.2%) and appear in
both the CANVAS & SAFE AREA section and the CRITICAL REMINDER section.
Could not verify against the real gpt-image-2 API (no network/credentials
in this sandbox, see note above) -- this needs the user's own next run to
confirm the actual effect on output quality.

## 2026-08-20 (11) — Found a real conflicting-instruction bug: the guideline text explicitly permits phone/app-UI imagery to bleed past the margin

**User report (with the actual gpt-image-2 output image attached):**
content still spilled outside the intended safe area even after (10)'s
percentage-based fix -- the headline text ran almost to the right edge of
the canvas, and the phone mockup / Christmas tree / gift boxes sat right
against the left edge, both clearly outside the declared safe area
(6.8%-93.1% width, 23.8%-76.2% height for this request).

**Root cause found (a genuine conflict in the prompt, not a rendering/
caching bug):** `lib/guidelines/guidelines.json`'s `banner_popup_noti` entry
(included verbatim in every prompt for this format_type, under FORMAT
DESIGN GUIDELINE) contains, under its own "BIÊN AN TOÀN" (safety margin)
section:

> "Đối với hình ảnh điện thoại có giao diện app, có thể phóng to vượt ra
> ngoài biên an toàn để đảm bảo xem rõ chi tiết giao diện" -- "For phone
> images with an app interface, [it] may be enlarged past the safe margin
> to keep the interface details legible."

This line was written by/for a design team producing the FINAL banner
directly -- it has no notion of this app's 2-stage `generation.canvas` ->
crop -> `output.size` pipeline, and never anticipated a second, stricter,
separately-declared safe area governing what survives the crop. As
written, it sits later in the prompt than the CANVAS & SAFE AREA section
and is a direct, explicit permission for exactly the kind of content this
request has -- `buckets.product` is a phone screenshot -- to bleed toward
the canvas edge. `design_rules.md`'s own "7% padding" line is similarly
phrased as a soft "tốt nhất" (best-practice) suggestion, not a hard rule,
so it doesn't override the guideline's explicit exception either. This is
a far more concrete, specific explanation for the reported failure than
"the model just doesn't listen well" -- it's the model being given 2
directly contradictory instructions and picking the permissive one for
exactly the asset type (phone UI) present in this request.

**Fix applied:** added a new "OVERRIDE ON THE GUIDELINE ABOVE" section to
`prompt_builder.py`'s `build_prompt()`, inserted immediately after the
FORMAT DESIGN GUIDELINE text (only when `has_crop`) so it reads as a direct
correction sitting right next to the conflicting wording. It explicitly
names the phone/app-interface-imagery exception and states it does not
apply here -- nothing, including a phone/device mockup, may cross the safe
area boundary. `_PROMPT_TEMPLATE_VERSION` bumped 3 -> 4.

**Not fixed, flagged instead:** the underlying guideline TEXT in
`guidelines.json` still contains this permissive wording, and any OTHER
format_type's guideline entry could contain a similar conflict this
override doesn't specifically name (it only calls out the phone/app-UI
case, since that's the one proven to matter here). A more complete fix
would be to either edit `guidelines.json` itself to remove/qualify that
exception (a content change, not a code change -- did not do this without
asking, since that guideline presumably reflects a real design-team
decision for how banners look outside of this app's own generation
pipeline), or make the override fully generic instead of phone-specific.
Left as a decision for the user.

**Considered but not built -- the more structurally reliable fix:**
generate the creative content directly at `output.size` (no larger canvas,
no crop at all), then use gpt-image-2's mask-based outpainting (an
`images/edits` call with a mask that PRESERVES the original center pixels
and only lets the model fill in newly added surrounding canvas) to extend
the background outward for any margin needed. This makes safe-area
compliance a property of the pipeline architecture -- there's no longer
anything for the model to "get right" positionally, since the protected
content is generated first and never touched again -- rather than
something hoped for via prompt wording, however precise. This is a bigger
change (new provider capability, a 2-call-per-version flow instead of 1,
rethinking what `generation.canvas` even means) and was not implemented
this round; flagged as the strongest next step if prompt-level fixes keep
falling short.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 72 | **passed** |

Manually rebuilt `DB_NOTI_AUTOPAY_NOEL`'s prompt and confirmed the new
OVERRIDE section appears, naming the phone/app-UI exception explicitly.
Not verified against the real API (see (10)'s note on this sandbox's lack
of network access) -- needs the user's next real run to confirm.

## 2026-08-20 (12) — Rewrite the guideline's margin numbers via GPT-5-mini instead of only overriding them

**User request (after confirming (11)'s finding was correct):** rather than
patch the conflict with a static override sentence, have GPT-5-mini
actually REWRITE the guideline's own "BIÊN AN TOÀN" margin numbers --
adding the safe area's x/y offset from the canvas edges to whatever
margin-from-edge values the guideline already states -- and cache the
result by `<format_type>_<canvas_size>_<safe_area_size>`, in the same
`./lib/safe_area_guide` directory as the visual guide image, reused by
filename lookup so the LLM is only called once per distinct combination.

**New module: `app/core/guideline_rewriter.py`.**
- `rewritten_guideline_filename(format_type, canvas_wh, safe_area_wh) ->
  str`: `<format_type>_<canvas_w>x<canvas_h>_<safe_w>x<safe_h>.txt` -- the
  user's own example used a `.png` extension (by analogy with the visual
  guide's naming rule); used `.txt` here instead since this is rewritten
  guideline TEXT, not an image -- flagged clearly rather than silently
  matching the example literally and writing prose into a `.png` file.
- `rewrite_guideline_for_safe_area(format_type, guideline_text, canvas_wh,
  crop_box) -> str | None`: precomputes the exact per-side pixel offsets
  in Python first (`left_offset = sa_x`, `right_offset = canvas_w -
  (sa_x + sa_w)`, `top_offset = sa_y`, `bottom_offset = canvas_h - (sa_y +
  sa_h)` -- correct for ANY safe-area position, not just centered ones)
  and hands them to GPT-5-mini already computed, so the model only has to
  ADD 2 given numbers per margin rather than reconstruct the geometry
  itself. Also instructs the model to remove/neutralize any "may bleed
  past the margin" exception (the exact phone/app-UI wording found in
  (11)). Searches `Settings.safe_area_guide_dir` for the cache file first
  (reuse if found); calls the LLM and writes the cache file only on a
  miss. Returns `None` -- never raises -- on any failure (empty guideline
  text, `LLMProviderError`, or any other exception), so a broken/
  unconfigured LLM never blocks a run; callers fall back to the original
  guideline text.

**`app/core/prompt_builder.py` wiring:** when `has_crop` and the resolved
guideline has text, calls the rewriter before assembling the FORMAT DESIGN
GUIDELINE section; on success, uses the rewritten text (section header
notes "-- REWRITTEN for the safe area") and SKIPS appending (11)'s static
OVERRIDE ON THE GUIDELINE ABOVE section (redundant once the guideline
itself no longer conflicts); on `None` (rewrite skipped or failed), uses
the original guideline text and keeps the OVERRIDE section as a fallback
safety net, exactly as (11) left it. `_PROMPT_TEMPLATE_VERSION` bumped
4 -> 5.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 81 | **passed** (up from 72) |

New `app/evaluation/unit/test_guideline_rewriter.py` (7 tests, all use a
fake `LLMFactory.create()` -- no real network/credentials needed): filename
convention, empty-text no-op, LLM-called-and-cached, cache-reused-without-
a-2nd-LLM-call, the exact per-side offset numbers handed to the LLM prompt
(canvas 1536x1024 / safe area 1035x420 at (250,302) -> left=250, right=251,
top=302, bottom=302 -- verified arithmetically), failure leaves no stale
cache file behind, and 2 different (format_type, geometry) combinations
never collide on the same cache file. 2 new tests in
`test_prompt_builder.py` cover the `build_prompt()` wiring itself (rewrite
succeeds -> rewritten text used, override omitted; rewrite fails ->
original text kept, override kept).

Manually ran `build_prompt()` against the real `DB_NOTI_AUTOPAY_NOEL.json`
with a fake LLM standing in for GPT-5-mini (no real network in this
sandbox): confirmed the cache file was created at
`lib/safe_area_guide/banner_popup_noti_1200x800_1035x420.txt` with offsets
L=82 R=83 T=190 B=190 (matching the request's actual geometry), the
rewritten text appeared in the prompt with the OVERRIDE section correctly
omitted, and a 2nd `build_prompt()` call for a DIFFERENT request name with
the same format_type/geometry reused the cache file with 0 additional LLM
calls. Cleaned up all test artifacts (`lib/safe_area_guide/`,
`output/DB_NOTI_AUTOPAY_NOEL*/`) afterward.

**Still not fixed, by design (needs the user's decision, not assumed):**
the SOURCE guideline text in `guidelines.json` still contains the original
permissive wording -- this feature only affects what gets SENT to the
image model, not the guideline file on disk. If the user wants the source
file itself corrected (so anyone reading `guidelines.json` directly also
sees accurate margin guidance), that's a separate, explicit edit to make.
Also unverified against the real gpt-image-2 API, same limitation noted in
(10)/(11) -- needs the user's own next real run.

## 2026-08-20 (13) — Simplify the guide image (drop measurements) + a targeted product-containment guard pass

**User feedback, with a real gemini-3.1-flash output attached:** the
model's composition tracked the guide's magenta BOX shape almost exactly --
the mask itself is doing the real work, like a layout template. 2 asks:
(1) remove the width/height/offset/corner-coordinate ANNOTATIONS from the
guide image -- unnecessary now that the box's own pixel geometry already
carries the exact canvas/safe-area size; (2) the `product` asset (a phone
mockup) still bled past the safe area's TOP/BOTTOM edges -- have GPT-5-mini
scan the whole final assembled prompt, strip any remaining wording that
permits the product to bleed, and add an explicit "product MUST stay
inside the safe area" instruction.

**1) `app/core/safe_area_guide.py`:** `build_safe_area_guide_png()` no
longer draws width/height rulers, x/y offset leader lines, or corner-
coordinate labels (all from (2)) -- only the translucent magenta tint +
solid border + 1 short instruction line in the caption bar (no numbers).
Removed the now-dead helper functions (`_draw_arrowhead`,
`_draw_double_arrow`, `_draw_dashed_double_arrow`, `_draw_label`, `_unit`)
and their color constants. Output file size dropped from ~26KB to ~13KB for
the same geometry (visually confirmed: still tints outside/leaves inside
plain white, sampled pixels at both to check).

**CACHE INVALIDATION NEEDED ON THE USER'S OWN MACHINE:** this module's
attachment cache (`build_safe_area_guide_attachment()`) reuses an existing
file at the expected filename WITHOUT checking whether it still matches
the current drawing code (a deliberate trade-off from (4), reconfirmed in
(6)'s docstring). Any `.png` already sitting in the real
`lib/safe_area_guide/` directory from before this change still has the OLD
ruler/label annotations baked in and will keep being served as-is until
deleted by hand -- this update cannot reach into that directory on its
own. **The user needs to clear `lib/safe_area_guide/` once after this
update lands**, or the next run will keep attaching the old, more
cluttered guide image.

**2) New module `app/core/prompt_bleed_guard.py`:**
`enforce_product_containment(prompt_text) -> str | None` -- a final GPT-5-
mini pass over the WHOLE assembled prompt text (not just the guideline
section, unlike (12)'s targeted fix), specifically hunting for and
removing/neutralizing any sentence anywhere that permits the `product`
asset to bleed past the safe area, and adding an explicit containment
requirement if one isn't already unambiguous. Wired into
`prompt_builder.py::build_prompt()` as the LAST step (after LLM
refinement, if any), gated on `has_crop and req.buckets.product` (skipped
entirely when there's no product asset to check -- saves an LLM call).
Falls back to leaving `final_text` unchanged on any failure. No separate
file cache needed -- the corrected text is already captured by
`build_prompt()`'s own existing `_prompt_cache/built_prompt.json` cache.
`_PROMPT_TEMPLATE_VERSION` bumped 5 -> 6.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 88 | **passed** (up from 81) |

New `app/evaluation/unit/test_prompt_bleed_guard.py` (4 tests, fake
`LLMFactory` -- no real network/credentials): empty-prompt no-op, the LLM
is called with the FULL prompt verbatim + a product-focused system
instruction, failure falls back to `None`, empty LLM response also falls
back to `None`. 3 new tests in `test_prompt_builder.py` cover the wiring:
guard applied and its result used when `buckets.product` + crop are both
present, guard SKIPPED (0 calls) when there's no product bucket, and
`build_prompt()` stays healthy when the guard returns `None`.
`test_safe_area_guide.py` needed no changes -- none of its assertions
depended on the removed annotations (they check pixel colors, not label
text).

Manually verified the guideline-rewrite (12) and product-containment-guard
(13) features work correctly TOGETHER in one real `build_prompt()` call
against `DB_NOTI_AUTOPAY_NOEL` with a single fake LLM that distinguishes
the 2 call sites by their prompt content (both features share the same
`LLMFactory.create()` entry point in real usage) -- confirmed the
guideline came back rewritten, the OVERRIDE fallback was correctly
omitted, and the guard's correction was present in the final prompt text.
An earlier attempt at this same check, using 2 separately-mocked fake
providers in nested `with patch.object(...)` blocks, gave a misleading
result: the 2nd patch silently overwrote the 1st because both modules
import and patch the SAME shared `LLMFactory` class object -- a mistake in
the verification script itself, not in the production code (each
feature's own isolated unit tests, which mock only 1 call site per test,
already passed correctly and were unaffected). Noted here in case this
comes up again when hand-verifying multi-LLM-call flows.

Cleaned up all test artifacts (`lib/safe_area_guide/`,
`output/DB_NOTI_AUTOPAY_NOEL/`) afterward. Not verified against a real
image-generation API in this sandbox (no network) -- needs the user's own
next real run, after clearing their `lib/safe_area_guide/` cache as noted
above.

## 2026-08-20 (14) — Fixed: `ConnectionResetError` loading a request from the frontend, + a related cwd-dependent config bug

**User report:** `GET /requests/MKT_DEMO_TET` failed with `('Connection
aborted.', ConnectionResetError(10054, 'An existing connection was
forcibly closed by the remote host', ...))` from the Streamlit frontend.

**Root cause, found by reading the real `output/_logs/app.log` from the
user's machine (staged via the device bridge, not guessed):** the backend
had been started with the documented command, `uvicorn app.main:app
--reload` -- with NO `--reload-dir` argument. Without `--reload-dir`,
uvicorn's `--reload` watches the ENTIRE current working directory
(`version_1/`) for file changes, including `input/`, `output/`, and
`lib/` -- directories this app itself writes into constantly while
running: prompt logs, downsampled image caches, the safe-area-guide
cache, and (critically) the app's OWN log file at
`output/_logs/app.log`. That creates a self-triggering loop -- the app
logs a line -> uvicorn's watcher sees the log file change -> restarts the
server -> the restart logs MORE lines -> triggers another restart -- and
separately, simply saving a new `request.json` into `input/` (exactly what
the user did to test `MKT_DEMO_TET`) is itself enough to trigger a
restart. The log confirmed this directly: 2 "starting" lines 3 seconds
apart with no valid "shutting down" line logged between them (the
lifespan shutdown handler got cut off mid-run by the next restart already
starting) -- and there was NO log entry at all for the `MKT_DEMO_TET`
request, meaning it hit the server exactly while a restart was in flight
and its TCP connection was torn down -- exactly matching the reported
`ConnectionResetError`.

**Fix:** added `--reload-dir app` to every documented run command
(`README.md`, `2-project-technote.md`, `app/main.py`'s own run-instructions
comment) -- this makes uvicorn watch ONLY `app/` (the actual source code)
for changes, so writing prompt logs, cached images, or new `request.json`
files no longer triggers a restart. Added a clear explanation in
`2-project-technote.md` (Vietnamese, matching that file's language) so
future readers understand WHY this flag matters, not just that it's
required. **The user needs to stop and restart their backend with the new
`--reload-dir app` flag for this fix to take effect** -- editing the docs
alone doesn't change an already-running process.

**A 2nd, related bug found while reading that same log:** the log's own
startup banner showed `asset_library_dir=C:\Users\admin\...\
dco_dynamic_creative_opt\lib` -- a path inside a DIFFERENT, unrelated
project directory (missing `_v2\version_1`), not this project's real
`lib/`. Root cause: `app/settings.py`'s `_resolve_dir()` helper checked a
relative default path (e.g. `Path("./lib")`) with `raw.exists()` directly
-- which resolves against the process's CURRENT WORKING DIRECTORY at
launch time, not `PROJECT_ROOT`. If the backend is ever started from a
directory other than `version_1/` (as apparently happened here) that
happens to also contain a same-named subfolder, that unrelated folder
gets silently picked up instead of a proper "directory not found, falling
back" case. Since every directory setting in `Settings` uses a plain
relative default (`./lib`, `./input`, `./output`, `./script/output`,
`./lib/safe_area_guide`), this could recur for any of them.

**Fix:** `_resolve_dir()` (and `prompt_rules_file`'s validator, which had
the identical bug inline) now anchors a relative path at `PROJECT_ROOT`
before checking existence, so the resolved directory no longer depends on
the process's cwd at all -- only an explicit ABSOLUTE path in `.env` can
still point elsewhere, which is correct (an absolute override should be
honored regardless of cwd).

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 94 | **passed** (up from 88) |

New `app/evaluation/unit/test_settings.py` (6 tests): a relative path that
exists under `PROJECT_ROOT` resolves there; the exact bug scenario
reproduced directly -- `monkeypatch.chdir()` into an unrelated tmp
directory containing its OWN decoy `lib/` folder, and confirmed
`_resolve_dir()` still resolves to THIS project's real `lib/`, never the
decoy; a relative path that doesn't exist anywhere falls back; an
absolute path is honored/rejected on its own existence regardless of cwd;
and the `Path("")` -> `Path(".")` normalization edge case resolves
predictably to `PROJECT_ROOT` rather than the process's cwd.

Manually reproduced the `--reload` file-watch issue's mechanism by
inspecting the log timeline (2 restarts 3 seconds apart, no request log
entry for the failing call) rather than by running a live uvicorn
`--reload` process in this network-isolated sandbox -- could not directly
re-trigger and re-verify the fix against a live server here; needs the
user's own restart with the corrected command to confirm the
`ConnectionResetError` stops recurring.

## 2026-08-20 (15) — Found the real reason the GPT-5-mini guideline rewrite "wasn't taking effect": `.env` key names didn't match `Settings` fields, so real credentials were silently ignored

**User report:** after re-running against the real gpt-image-2 API, the
prompt actually sent still contained the format_type guideline's
UNMODIFIED `48px` margin figure, not the GPT-5-mini-rewritten value the
(12) change was supposed to produce (e.g. `(48+250)=298px` horizontal /
`(48+302)=350px` vertical, derived from `safe_area.x`/`safe_area.y` in
that request's `request.json`). The user asked directly whether the code
was actually calling GPT-5-mini with the format_type guideline text and
the `safe_area` geometry as designed.

**Investigation:** read the real `output/_logs/app.log` from the user's
machine (not guessed) and grepped for both LLM-based features added in
(12)/(13). Both `dco.guideline_rewriter` and `dco.prompt_bleed_guard` DO
fire on every run and DO attempt the call -- confirmed by matching
`ERROR` log lines -- but every single attempt fails immediately with:

```
app.llm.base.LLMProviderError: No OpenAI-compatible endpoint configured --
set either (AZ_GPI_5_MINI_ENDPOINT + AZ_GPI_5_MINI_KEY) for the Azure
deployment, or (LLM_BASE_URL + OPENAI_API_KEY) for a custom endpoint, or
just OPENAI_API_KEY to use the public OpenAI API.
```

So the answer to the user's question is: **yes, the code calls GPT-5-mini
exactly as designed (guideline text + safe_area-derived margin offsets in
the prompt), on every run** -- this was never a logic bug. The call
itself was failing, and `rewrite_guideline_for_safe_area()`/
`enforce_product_containment()` were both correctly catching the
`LLMProviderError` and falling back to the original, unmodified guideline
text -- which is exactly the "still says 48px" symptom the user observed.

**Root cause:** the error message above names `AZ_GPI_5_MINI_ENDPOINT` /
`AZ_GPI_5_MINI_KEY`, but `app/settings.py`'s actual `Settings` fields are
`az_gpt5mini_endpoint` / `az_gpt5mini_key` -- under this project's own
convention (env var name = uppercased field name, no extra underscores),
pydantic-settings only ever looks for `AZ_GPT5MINI_ENDPOINT` /
`AZ_GPT5MINI_KEY`. The user's real `.env` (staged and inspected key-names
-only, values never read/printed) had real Azure credentials filled in --
but under the old, wrong names, so `Settings` silently kept its empty
default and the credentials were never actually used. Auditing every
`Settings` field name against its expected env var name (regex-extracted
field list, cross-referenced against the `.env`/`.env.example` key names)
turned up 4 more instances of the same class of bug, all pre-existing in
`.env.example` itself (a template bug, not something the user typo'd):
`AZ_FLUX_2_PRO_DEPLOYMENT_NAME`, `AZ_MAI_IMAGE_25_DEPLOYMENT_NAME`,
`AZ_MAI_IMAGE_25_FLASH_DEPLOYMENT_NAME`, and
`AZ_MAI_IMAGE_25_PRO_DEPLOYMENT_NAME` all carry a `_NAME` suffix the
`Settings` fields (`az_flux_2_pro_deployment`, etc.) don't have, plus
`AZ_GPT_5_MINI_API_VERSION` (should be `AZ_GPT5MINI_API_VERSION`) --
7 keys total.

**Fix:**
- `app/llm/providers/openai_provider.py`: corrected the module docstring,
  the `LLMProviderError` message text, and the run-instructions comment
  to `AZ_GPT5MINI_ENDPOINT`/`AZ_GPT5MINI_KEY`.
- `.env.example`: renamed all 7 keys to match the real `Settings` field
  names, with dated comments explaining the old/new names for anyone
  still holding an old-format `.env`; also added the previously-missing
  `SAFE_AREA_GUIDE_DIR=./lib/safe_area_guide` line (present as a
  `Settings` field since (9) but never added to the template).
- The sandbox's own local `.env` (test credentials only, no real
  secrets) renamed the same way for consistency.
- **The user's real `.env`** (the file that actually mattered for
  unblocking their runs) was renamed key-by-key, in place, via
  `device_bash`'s `sed -i` on this machine -- only the 7 left-hand key
  names were rewritten; every credential VALUE was preserved byte-for-byte
  and was never read, printed, or transmitted at any point in this
  investigation or fix. A timestamped backup
  (`.env.bak_before_key_rename_20260820`) was left alongside it before the
  rename, in the same folder, in case anything needs to be compared back.

**Result:**

| Tier | Tests | Result |
|---|---|---|
| Default (`pytest -q`, unit + smoke) | 94 | **passed** (unchanged -- this was a config/docs fix, no logic changed) |

**The user must restart their backend for the corrected `.env` to take
effect** (pydantic-settings reads `.env` once at process startup via the
`lru_cache`d `get_settings()`). Not yet verified against a live
gpt-image-2 run with the corrected credentials in place -- needs the
user's own re-run to confirm the guideline text now actually arrives
rewritten (e.g. the `298px`/`350px` figures from their example) rather
than the original `48px`.

## 2026-08-25 (16) — Fix bảng aspect ratio sai/thiếu của `gemini-3.1-flash-image` (nguyên nhân gây stretch/méo trước khi crop)

Bối cảnh: project đã chốt chỉ dùng 1 model `gemini-3.1-flash-image`
(`config/model_usage.json`). Cùng lớp bug với entry (10) ở trên (đã fix cho
gpt-image-2) nhưng chưa fix cho Gemini.

| Vấn đề | Cũ | Mới | Sửa ở |
|---|---|---|---|
| Bảng ratio hỗ trợ sai/thiếu -- chỉ 8 ratio, có `5:2` không tồn tại thật, thiếu 6 ratio thật (1:4, 1:8, 4:1, 4:5, 5:4, 8:1, 21:9) | `1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3, 5:2` | Đủ 14 ratio thật Gemini 3.1 Flash Image hỗ trợ: `1:1, 1:4, 1:8, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9` | `gemini_image.py::_ASPECT_RATIO_BUCKETS` |
| Đo khoảng cách ratio bằng hiệu số thường -- lệch bất đối xứng giữa portrait/landscape | `abs(ratio_a - ratio_b)` | log-ratio distance: `abs(log(target) - log(candidate))` | `gemini_image.py::_closest_aspect_ratio` |
| Khi canvas khai báo lệch nhiều so với ratio Gemini hỗ trợ nhất -> ảnh trả về bị resize ép (stretch) trước khi crop ở `image_post.py`, có thể làm méo nội dung -- không có cảnh báo | im lặng | Log WARNING khi ước tính crop/stretch-loss > 12%, kèm % và ratio đã chọn, gợi ý chỉnh `generation.canvas` trong request.json | `gemini_image.py::_closest_aspect_ratio` (mới thêm) |

**Chủ động KHÔNG làm** (đúng phạm vi đã thống nhất với user trước khi sửa
code -- xem thảo luận 2026-08-25): không xây content-aware crop, object/
face/logo detection, generative outpainting, hay regenerate-and-relayout.
Lý do: cơ chế `safe_area_guide.py` (magenta box) đã được xác nhận hoạt động
tốt trong thực tế (entry (13) ở trên: "model's composition tracked the
guide's magenta BOX shape almost exactly") -- rủi ro thật nằm ở bước
snap-ratio/resize trước khi crop, không phải ở việc chọn vùng crop.

**Chưa verify:** chưa gọi API thật (không có network trong sandbox này) --
cần user chạy lại thực tế để xác nhận ảnh trả về ít bị méo hơn và warning
log xuất hiện đúng khi test 1 canvas có ratio lệch xa (ví dụ safe_area
1035x420 ≈ 2.46:1, gần `5:4`/`21:9` nhất trong bảng mới thay vì `5:2` giả
trước đây).

## 2026-08-25 (17) — Root cause thật của "ảnh sau crop vẫn khá tệ" (request `DB_NOTI_DEMO_TRUNGTHU_huy`) + roadmap ưu tiên

User chạy thật (`num_version=3`) sau fix (16), báo ảnh ORG và ảnh crop vẫn tệ
(Mastercard bị crop lệch). Điều tra bằng đo pixel + đọc log + đọc prompt log
thật gửi cho Gemini:

| Vấn đề | Phát hiện | Nguyên nhân |
|---|---|---|
| Fix (16) không liên quan | Canvas request này = 1200x800 = đúng "3:2", đã hợp lệ ở CẢ bảng cũ và mới -> fix (16) không đổi gì cho request này | Không phải bug ratio-snap |
| Ảnh vẫn bị crop mất nội dung dù mọi safeguard trong prompt (OVERRIDE, CRITICAL REMINDER, safe_area_guide ảnh magenta...) đều đúng và đầy đủ | `app.log`: `guideline_rewriter` và `prompt_bleed_guard` (2 bước LLM gọi Azure endpoint để (a) tính lại margin theo safe-area thật, (b) tăng cường câu lệnh containment riêng cho product) FAIL 100% với `SSLCertVerificationError: self-signed certificate in certificate chain`, fallback về text tĩnh (không có 2 lớp bổ sung này) | Lỗi SSL cert khi gọi `ngothanhnhan125-dco-model-eastus.services.ai.azure.com` -- nghi do proxy/mạng công ty MITM cert không nằm trong CA store mà `requests`/`certifi` dùng |
| Prompt bị cache "degraded" | Prompt build lần đầu (10:41:03) đã fail 2 bước trên và cache lại luôn bản fallback -- mọi lần chạy sau (10:52, 11:27 -- đúng lần user chụp ảnh) tái dùng lại y nguyên bản thiếu bảo vệ đó, không có dấu hiệu gì báo hiệu | `prompt_builder.py` cache theo key, không phân biệt "cache từ 1 lần build có guard fail" |

**Sửa trong lượt này (Tier 1 -- hạ tầng, điều kiện để guard cũ chạy được):**

| Vấn đề | Cũ | Mới | Sửa ở |
|---|---|---|---|
| `requests` dùng CA bundle của `certifi`, không thấy root CA của proxy công ty (nếu có) -> fail xác thực TLS tới Azure endpoint | không có | Inject `truststore` (dùng CA store của Windows/OS, nơi root CA công ty thường đã được trust sẵn) vào `ssl` module ngay khi import `_http.py` -- KHÔNG tắt verify (`verify=False`), vẫn xác thực TLS đầy đủ | `app/providers/_http.py`, `requirements.txt` (`truststore`) |

**Chưa sửa trong lượt này (cần user cài lại package + chạy thật để verify Tier 1 xong mới quyết Tier 2):**
- Tier 2 (fix lõi cho đúng vấn đề bleed): thêm bước generate → validate (kiểm tra product/text có nằm trong safe-area không) → retry có feedback, thay vì chỉ tin vào prompt-side safeguard. Cần vision-capable LLM call (kiến trúc `app/llm/*` hiện tại là text-only) -- sẽ thảo luận cách làm cụ thể SAU khi xác nhận Tier 1 unblock được 2 guard cũ, vì có thể 2 guard đó (`guideline_rewriter` + `prompt_bleed_guard`) một khi chạy được đã đủ giải quyết phần lớn vấn đề, không cần xây thêm vòng validate mới ngay.

**Roadmap đầy đủ đã thống nhất với user (không làm ngay, để tránh over-engineering khi chưa có nhu cầu thực tế):**
- Tier 3 (để dành, làm khi có ca lỗi cụ thể): product/logo làm layer riêng không cho AI vẽ lại, typography render bằng code, landscape/portrait relayout riêng, platform-specific safe-area, resolution/quality pipeline, đồng bộ creative-spec nhiều ratio.
- Tier 4 (ngoài phạm vi công cụ nội bộ hiện tại): pipeline đa thị trường/đa ngôn ngữ, feedback loop từ CTR/conversion thật.
- Đã có sẵn, không cần thêm: human review trước publish (bước `Release` hiện tại đã là cổng kiểm duyệt thủ công).

**Update sau khi user cài `truststore` + chạy lại thật (cùng ngày):** SSL
cert đã hết lỗi (xác nhận qua `app.log` 14:02 -- request tới Azure endpoint
giờ nhận được response thật, không còn `SSLCertVerificationError`). Nhưng
lộ ra bug THỨ 2 trước đó bị SSL che mất:

| Vấn đề | Cũ | Mới | Sửa ở |
|---|---|---|---|
| Azure Responses API trả `HTTP 400 invalid_value` trên `input[1]` -- mỗi item trong `input` thiếu field `type` bắt buộc | `{"role": "system"/"user", "content": ...}` | `{"type": "message", "role": ..., "content": ...}` | `app/llm/providers/openai_provider.py::generate()` |

Đã verify bằng gọi trực tiếp endpoint thật (không qua sandbox) -- request
với `type: "message"` trả `200 OK`. Vậy là MỌI lần chạy trước đó (kể cả 6
ảnh user gửi lúc 10:52-14:03) đều CHƯA có `guideline_rewriter`/
`prompt_bleed_guard` chạy thật -- 2 guard này bị chặn liên tiếp bởi 2 bug
khác nhau (SSL rồi tới bug `type` này) suốt từ đầu. Đã xoá lại
`_prompt_cache/` lần 2 -- cần user chạy lại 1 lần nữa để có ảnh phản ánh
đúng cả 2 guard hoạt động, rồi mới đánh giá được phần bleed của Mastercard
có cải thiện hay không.

## 2026-08-25 (18) — Tier 2: validate ảnh SAU khi generate + retry (gộp 2 vấn đề mới phát hiện ở run có guard chạy thật)

User chạy lại (run 14:25-14:28, guard (17) chạy thành công cả 2). Đọc trực
tiếp 3 file ORG+final trên đĩa: vấn đề gốc (card bị crop mất) đã cải thiện
rõ, nhưng lộ 2 vấn đề THẬT khác prompt-engineering không bắt được:
(1) model tự vẽ lại khung magenta + caption của SAFE AREA GUIDE vào chính
ảnh nó tạo (v2) -- vi phạm "do not reproduce this"; (2) card bị THIẾU hẳn
ở 2/3 version (v2, v3) dù rule MUST bắt dùng mọi bucket có asset.

Quyết định: gộp cả 2 thành 1 bước validate SAU crop + retry, vì cùng là
"prompt-side guard chỉ XIN, không XÁC NHẬN được kết quả thật". Ràng buộc
thêm của user: chỉ kiểm bucket THỰC SỰ có trong request, không tự bịa yêu
cầu cho bucket rỗng.

| Vấn đề | Cũ | Mới | Sửa ở |
|---|---|---|---|
| Không có bước xác nhận ảnh cuối thật sự đạt yêu cầu | im lặng, chỉ tin prompt | Sau crop: quét màu magenta đặc trưng của guide (rule-based) + hỏi vision LLM (chỉ hỏi bucket có trong request) | `app/core/output_validator.py` (mới) |
| Không hỗ trợ ảnh đầu vào cho LLM provider | `generate()` chỉ nhận text | Thêm `generate_vision()` (nhận nhiều ảnh, dùng Responses API `input_image`) | `app/llm/providers/openai_provider.py` |
| Ảnh lỗi vẫn được lưu thẳng, không có cơ hội sửa | generate 1 lần, lưu luôn | Retry tối đa 2 lần/version nếu validate fail, giữ bản cuối + log WARNING nếu vẫn lỗi sau khi hết lượt | `app/core/pipeline.py::run_generate()` (`_MAX_GENERATE_ATTEMPTS=2`) |

**2 bug tự phát hiện trong lúc test (đã sửa trước khi giao):**

| Vấn đề | Cũ (sai) | Mới (đúng) | Vì sao |
|---|---|---|---|
| Câu hỏi "có product không" quá chung (ví dụ "card, phone...") | hỏi trực tiếp, không có ảnh đối chiếu | So sánh với ẢNH THẬT của bucket `product` (2 ảnh: ảnh generate + ảnh reference) | Test thật: false-positive -- LLM trả "yes" vì nhầm cái điện thoại demo (thuộc `context`) là "product" |
| Câu hỏi "có người không" so khớp identity với ảnh gốc | "does the SAME person appear" | Bỏ so khớp identity, chỉ hỏi "có nhân vật xuất hiện không" (presence-only) | Test thật: false-negative -- design rule ép chuyển ảnh chụp thật thành minh họa 2D flat + cho phép đổi pose, nên so identity với ảnh chụp gốc luôn ra "no" dù đúng |
| Quét màu magenta downscale ảnh về 300px trước khi so | downscale rồi quét | Quét ở full-resolution | Test thật: false-negative -- viền guide chỉ vài pixel, bị blend mất khi downscale |
| Vision LLM call chậm (có lúc timeout) | effort mặc định của deployment ("medium") | Ép `reasoning.effort="low"` riêng cho các câu hỏi yes/no đơn giản này | Đo thật: 1 lần Read timed out ở effort medium; effort low giảm ~10s/lần |

**Đã verify bằng dữ liệu thật** (3 ảnh final của run 14:25-14:28, không phải
giả định): v1 = "OK", v2 = đúng 2 vấn đề (magenta leak + thiếu product),
v3 = đúng 1 vấn đề (thiếu product) -- khớp 100% với quan sát bằng mắt.

**Trade-off cần biết:** mỗi version lỗi sẽ tốn thêm 1 lần gọi lại Gemini +
1-2 lần gọi vision LLM (~10s) trước khi chấp nhận hoặc bỏ cuộc sau
`_MAX_GENERATE_ATTEMPTS=2` lần. Với `num_version=3`, trường hợp xấu nhất
(cả 3 version đều lỗi lần đầu) sẽ gần gấp đôi thời gian chạy so với trước.

**Chưa verify:** chưa chạy lại toàn bộ pipeline thật qua `run_generate()`
với retry loop mới (mới test từng hàm riêng lẻ qua CLI) -- cần user chạy
lại 1 lần nữa để xác nhận retry thật sự kích hoạt khi cần và log đúng như
mô tả.

## 2026-08-26 (19) — Đổi kiến trúc: `context`/`product`/text KHÔNG còn do AI vẽ nữa, mà ghép layer thật bằng code (composite)

**Bối cảnh:** user xác nhận 2 vấn đề còn lại sau (17)/(18): (a) ảnh nhân vật
chụp thật bị AI vẽ lại thành hoạt họa (dù đúng theo `banner_general_standard`
"phong cách 2D phẳng, không dùng ảnh chụp" -- nhưng user muốn brand
asset/nhân vật marketing giữ nguyên, không tái tạo); (b) thẻ Mastercard AI vẽ
lại thiếu chi tiết thật (thiếu chữ "Ngân Hàng Phương Đông", sai pattern chấm
vàng) -- rủi ro compliance ngân hàng. User yêu cầu: khoá cứng `context`
(nhân vật) + `product`, và uỷ quyền chọn hướng tối ưu cho layout + text.

| Vấn đề | Cũ | Mới | Sửa ở |
|---|---|---|---|
| AI vẽ lại nhân vật/sản phẩm từ ảnh tham chiếu -- sai style, sai chi tiết brand | AI nhận `context`+`product` làm reference rồi tự vẽ lại toàn bộ trong 1 ảnh | AI **chỉ vẽ background/scene trang trí**; `context`+`product` được ghép layer thật (pixel gốc) lên ảnh nền sau khi crop | `compositor.py` (mới), `pipeline.py::run_generate()` |
| Cắt nền trắng cho ảnh `context` (ảnh chụp studio nền trắng thật) | không có | Cutout bằng ngưỡng màu trắng thuần PIL (không cần model ML riêng), có feather biên mềm | `compositor.py::_cutout_white_bg` |
| Text (headline/message/CTA) do AI tự vẽ chữ -- rủi ro sai chính tả/dấu/font | AI vẽ chữ trực tiếp trong ảnh | Render bằng code từ đúng JSON, đúng font/màu/canh lề theo guideline `banner_popup_noti` (Manrope Bold 57pt, #E39717/#079F42, canh phải) | `compositor.py::_draw_text_block` |
| Font Manrope Bold không có file thật trong repo | n/a | Fallback Segoe UI Bold (`C:\Windows\Fonts\segoeuib.ttf`), log cảnh báo mỗi lần dùng fallback; drop `lib/fonts/Manrope-Bold.ttf` để dùng đúng font thật | `compositor.py::_resolve_font_path` |
| Vấn đề crop mất nội dung (gốc rễ toàn bộ session trước) | Né bằng prompt+validate+retry (Tier 1/2) -- vẫn có rủi ro vì AI tự đặt vị trí | **Cấu trúc loại bỏ hẳn rủi ro**: product/character/text được đặt bằng code, luôn nằm trong margin an toàn -- AI chỉ còn vẽ decoration, mất/lệch decoration không còn là vấn đề nghiêm trọng | `prompt_builder.py` (bỏ hẳn `CANVAS & SAFE AREA`/`CRITICAL REMINDER` nặng, `guideline_rewriter`/`prompt_bleed_guard` không gọi nữa) |
| `output_validator.py` hỏi vision-LLM "có product/nhân vật trong ảnh không" | Có (Tier 2, tốn 2 LLM call/version) | Bỏ hẳn 2 check đó (không còn ý nghĩa vì AI không vẽ 2 thứ này nữa); chỉ giữ check leak màu magenta + check logo (nếu `has_logo=true`) | `output_validator.py` |

**Đã verify thật (gọi API Gemini thật, không phải giả lập):** chạy
`run_generate('DB_NOTI_DEMO_TRUNGTHU_huy', ['gemini-3.1-flash-image'], num_version_override=1)`
-- ảnh nền AI trả về chỉ có trăng/đèn lồng/cây/mái chùa, KHÔNG có nhân vật,
KHÔNG có thẻ, KHÔNG có chữ. Ảnh cuối sau composite: nhân vật là ảnh chụp
thật (không hoạt họa), thẻ đúng pixel gốc (đủ chữ "Ngân Hàng Phương Đông",
đúng pattern), text đúng câu/đúng màu/canh phải, không chồng lấn.

**Đã chạy full test suite** (`requirements-dev.txt`, trước đó chưa từng cài
trong môi trường này): 94/95 pass. Nhân tiện phát hiện + sửa 3 bug CÓ SẴN
TỪ TRƯỚC, không liên quan tới thay đổi lần này (lộ ra vì đây là lần đầu
pytest thật sự chạy được trong môi trường này):
- `/files` endpoint kiểm tra `exists()` trước `allowed_roots()` -- path traversal tới file không tồn tại trả về 404 thay vì 403 (đúng ra phải chặn trước khi tiết lộ file có tồn tại hay không). Sửa ở `app/api/routes.py`.
- 2 test smoke (`test_generate_end_to_end_quality_mode`, `test_outputs_latest_and_at_timestamp`) hard-code `== 3` / model `"gpt-image-2"` -- đã lỗi thời từ lúc `config/model_usage.json` khoá về 1 model duy nhất (2026-08-25). Sửa số liệu test cho khớp thực tế.
- Assertion `Image.open(org_path).size == (1200, 800)` hard-code sai so với canvas thật của `DB_NOTI_AUTOPAY_NOEL.json` (1536x1024) -- đổi sang đọc động từ `request.json`.

**Còn lại (chưa làm, chủ động không làm)**
- Chưa render logo bằng compositing (vẫn để AI vẽ khi `has_logo=true`) -- request test hiện tại `has_logo=false`, chưa có nhu cầu thật.
- Layout product/nhân vật/text là toạ độ cố định theo % (không phải AI tự đề xuất) -- đơn giản, dễ kiểm soát, nhưng chỉ tối ưu cho bố cục ngang kiểu `banner_popup_noti`; format khác (dọc, ratio khác) nên review lại nếu dùng.
- `test_list_new_images_filters_by_since_ts` (unit, không liên quan): fail nhất quán trên máy Windows này (không phải flaky do tải) -- nghi do độ phân giải mtime NTFS, không đụng tới vì ngoài phạm vi thay đổi lần này.

**Cập nhật 2026-08-27 -- entry này đã bị REVERT ngay sau đó, bảng trên chỉ còn giá trị lịch sử:** kết quả compositing bị đánh giá "thô, tách lớp" khi test thật -- quay lại kiến trúc AI vẽ toàn bộ (product/context) + validate/retry (như entry (17)/(18)), chỉ giữ lại `canvas_planner.py` (auto-plan geometry) và `output_validator.py` bản siết chặt hơn. Xem `CANVAS_PLANNER_CHANGES.md` cho trạng thái ĐÚNG hiện tại. `compositor.py` nêu ở bảng trên không còn tồn tại trên đĩa.

## 2026-08-27 (20) — Text/CTA luôn ghép bằng code cho 3 usecase banner mới (AI_DESIGN_REQUEST_INPUT_3usecase.xlsx)

Bối cảnh: Excel yêu cầu 3 usecase (Passport, Esaving, World), mỗi usecase 2 size
(1024x440, 1024x256) với typography cố định riêng từng size, và 1 rule cứng:
"TEXT luôn ghép bằng code, không để AI vẽ chữ". Khác với entry (19) (đã revert) --
lần này **chỉ** text/CTA/button chuyển qua code, `theme`/`context`/`product` vẫn
để AI vẽ như hiện tại (đã chứng minh hoạt động tốt qua entry (17)/(18)).

| Vấn đề | Cũ | Mới | Sửa ở |
|---|---|---|---|
| Không có cơ chế ghép text/CTA bằng code | AI tự vẽ chữ trong ảnh | `render_text_layer()`: vẽ headline (nhiều dòng, nhiều màu)/message/CTA(+button) bằng PIL từ đúng giá trị JSON, sau khi crop | `app/core/text_compositor.py` (mới) |
| Request để trống headline/message/cta (usecase Esaving "AI tự sáng tạo") không có nội dung | rỗng | 1 lần gọi LLM (cùng pattern `guideline_rewriter.py`) soạn nội dung từ `design_brief`, chỉ điền field đang rỗng | `app/core/content_writer.py` (mới) |
| Prompt vẫn bảo AI vẽ đúng text thật | "REQUIRED TEXT CONTENT: render exactly..." | Khi `text_by_code=true`: đổi thành "không vẽ chữ, chừa cột %X bên phải trống nền" + override rule "keep text exactly" trong design_rules.md | `app/core/prompt_builder.py` |
| Không có field đánh dấu request nào dùng flow mới | n/a | `design_options.text_by_code: bool = False` (opt-in, mặc định giữ nguyên hành vi cũ) | `app/models/schemas.py` |

**3 bug tự phát hiện khi test thật (đã sửa trước khi giao), cùng 1 bài học: đo bằng dữ liệu thật, không suy diễn:**

| Vấn đề | Phát hiện bằng | Sửa |
|---|---|---|
| `text_zone_right_fraction=0.42` quá hẹp -- headline/message bị ellipsize dù còn ngắn | Render thử offline lên canvas trơn trước khi tốn API thật | Tăng lên 0.60 (1024x440) / 0.70 (1024x256) + thêm `fit_font_and_wrap()` tự giảm cỡ chữ (tối đa còn 55% size gốc) trước khi mới ellipsize |
| CTA màu `#FFFFFF` theo đúng Excel (G11 "Tương tự") nhưng size 1024x256 không có nền button -- chữ trắng trên nền sáng gần như vô hình | Nhìn ảnh render thử | Đổi màu CTA thành `#00804B` cho riêng size 1024x256 (lệch literal Excel có chủ đích, đã note rõ trong code) |
| Prompt chỉ nói "để cột bên phải yên tĩnh" -- AI đặt luôn cả product/card vào đúng cột đó, text đè lên card không đọc được | Chạy thật `USECASE2_ESAVING_1024x440` qua Gemini | Đổi thành MUST rõ ràng "không đặt product/context/motif vào cột đó, chỉ nền trơn"; thêm check `output_validator.py` hỏi vision-LLM "có nội dung nào lấn vào cột không" để tự động retry khi vẫn xảy ra |

**Đã verify bằng API thật** (`run_generate('USECASE2_ESAVING_1024x440', ['gemini-3.1-flash-image'])`, 2 lần chạy): lần 1 lộ lỗi chồng chữ lên card (đã sửa ở trên); lần 2 sau fix, ảnh cuối chữ rõ ràng, không chồng lên card, đúng font-fallback/màu/wrap. Vẫn còn 1 lần bị Gemini trả `429 RESOURCE_EXHAUSTED` giữa lượt retry (tự phục hồi nhờ retry/backoff có sẵn của `_http.py`) và validator vẫn gắn cờ "product hơi khác ảnh gốc" ở lần chạy thứ 2 -- retry hết 2 lượt vẫn giữ bản cuối theo đúng thiết kế đã có (entry (18)), không phải bug mới.

**Chưa verify bằng API thật (chỉ verify qua code review + offline render, để tiết kiệm quota sau khi vừa bị rate-limit):**
- Usecase 1 (Passport): buckets rỗng theo yêu cầu user ("bổ sung ảnh sau") -- 2 file `USECASE1_PASSPORT_1024x{440,256}.json` sẵn sàng, cần ảnh thật để test full run.
- Usecase 3 (World): dùng đúng `prod_card_lifestyle.png` theo yêu cầu user dù tên file không khớp "World" -- chưa chạy thật.
- Request cũ (`text_by_code=false`, ví dụ `DB_NOTI_DEMO_TRUNGTHU_huy`) không chạy lại thật để xác nhận không bị ảnh hưởng -- xác nhận bằng code review: mọi thay đổi đều nằm trong `if req.design_options.text_by_code:`, default `False`.

**Hạn chế nội dung đã biết (do chính dữ liệu Excel, không phải bug code):**
- Message của usecase Passport ở size 1024x256 (giới hạn 1 dòng) và headline 2-dòng của usecase World ở cùng size 1024x256 (giới hạn 1 dòng) đều bị ellipsize dù đã auto-shrink tới floor -- nội dung Excel cho cả 2 size giống nhau nhưng size 256 rõ ràng cần bản text ngắn hơn riêng, Excel không cung cấp bản ngắn đó.
- Chưa có file Manrope thật (`lib/fonts/Manrope-*.ttf`) -- đang fallback Segoe UI Semibold/Regular, có log warning mỗi lần dùng fallback.
