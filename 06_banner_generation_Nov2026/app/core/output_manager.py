"""All output path/naming conventions live HERE and nowhere else (REQUEST.md
[2] + the naming convention already in use on disk under `output/`).

Directory layout for 1 request (`<request_name>` == `request_name.json`'s
stem, e.g. `MKT_DEMO_TET`):

    output/<request_name>/<display_size>/<image_model>/<file>.png       FINAL images (already cropped to output.size)
    output/<request_name>/<display_size>/<image_model>/ORG/<file>.png   RAW images exactly as returned by the Image
                                                                          GenAI model, BEFORE crop -- same filename as
                                                                          its cropped counterpart next door, so the 2
                                                                          are trivially paired (2026-08-20, user request)
    output/<request_name>/prompt/prompt-<image_model>-version-<n>.txt  exact final prompt text sent to the model for
                                                                          that (model, version) request (REQUEST.md
                                                                          [3d]) -- ALWAYS (re)written every call, so
                                                                          it always reflects what was actually sent
                                                                          (2026-08-20 (3): used to skip writing if the
                                                                          file already existed, which left stale
                                                                          content on disk after a prompt-template
                                                                          code change -- see prompt_builder.py)
    output/<request_name>/_prompt_cache/built_prompt.json           shared/manual cached prompt-build result
    output/<request_name>/_prompt_cache/built_prompt-v<n>.json      auto/generated-text prompt cache for version n
    output/<request_name>/_text_gen_cache/text_gen.json             per-run generated headline/message/CTA sets
    output/<request_name>/_text_gen_cache/text_gen_prompt.txt        exact filled LLM prompt used to generate them
                                                                          (2026-09-08 user request) -- ALWAYS
                                                                          (re)written whenever `generate_text=true`
                                                                          actually reaches the LLM call, same
                                                                          always-overwrite convention as
                                                                          `prompt/prompt-<model>-version-<n>.txt`
                                                                          above, so it never goes stale
    output/<request_name>/_text_gen_cache/text_gen_result.json      raw per-version headline/message/cta JSON the
                                                                          LLM returned (2026-09-08 user request) --
                                                                          the OUTPUT counterpart of
                                                                          text_gen_prompt.txt's INPUT, ALWAYS
                                                                          (re)written whether that JSON came from a
                                                                          fresh call or a `text_gen.json` cache hit
    output/<request_name>/_technique_selection/technique_selection.json  aggregate auto-selection audit artifact
    output/<request_name>/_technique_selection/batch-*.json         per-batch selector responses and validation status
    output/<request_name>/_resized_refs/<name>                      downsampled attachment images (REQUEST.md [1] step [4])
    output/<request_name>/release/<file>.png                        released images (copies) + `<stem>_clone_<size>_<model>.png`

`usage/usage_log.db` (2026-09-06 user request; schema redesigned twice
since, see `app/core/usage_tracker.py` module docstring's (2) and (3)
notes): a SINGLE shared SQLite database -- deliberately NOT under
`output/` at all, but in its own `usage/` directory, a sibling of `app/`
directly at the project root (`Settings.usage_log_db_file`), so it stays at
1 fixed, predictable location regardless of where `output_dir` points.
Every `run_generate()`/`run_clone()` call upserts exactly 1 row into its
`runs` table (covering the whole run: total tokens, total cost, ...) plus 1
row per version generated in that run into its `run_versions` table --
never overwritten, never split per request/run, and never a fixed/flat set
of columns (a run's `num_version` differs run to run, which is exactly why
per-version data lives in its own table's ROWS instead).

The visual SAFE AREA GUIDE image attached to the model call
(`app/core/safe_area_guide.py`, 2026-08-20, user-reported failure) is NOT
under `output/<request_name>/...` -- since 2026-08-20 (4) (user request) it
lives in a SHARED library cache at `Settings.safe_area_guide_dir`
(`./lib/safe_area_guide` by default), 1 file per (canvas_size,
safe_area_size) combination, named
`safe_area_guide_<canvas_w>x<canvas_h>_<safe_w>x<safe_h>.png`, reused across
every request/run that needs that same geometry instead of each request
rendering its own copy.

`<display_size>/<image_model>/ORG/` is deliberately INSIDE the model
directory (not a top-level `output/<request_name>/ORG/`) so it's obvious
which model produced which raw image, and deliberately NOT added to
`_NON_RESULT_DIR_NAMES` below -- `_iter_result_pngs()` already only
`glob("*.png")`s directly inside `<image_model>/` (non-recursive), so
`ORG/`'s contents are naturally invisible to `list_new_images()` /
`latest_run_images()` / `images_matching_substring()` and therefore to the
webapp's Output gallery and "Select for release" checkboxes -- release only
ever offers the final, cropped image, never the raw one.

`display_size` is always `"WIDTHxHEIGHT"` of the REAL final output pixels
(`CreativeRequest.display_size_slug()` -- i.e. `output.size`, the
`generation.safe_area` size), never the internal `generation.canvas` size.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("output_manager")

# Includes seconds (not just hour+minute): app/evaluation/stress found that
# 2 concurrent /generate runs for the SAME request_name landing in the same
# MINUTE previously produced byte-identical filenames and silently
# overwrote each other's output (see 4-project-evaluation-results.md).
# Minute-only precision is still what's on disk for files generated before
# this fix -- that's fine, every lookup in this module matches on
# mtime/substring, never by parsing this exact format back out of a filename.
TIMESTAMP_FORMAT = "%Y-%m-%d-%H%M%S"

# Directories inside `output/<request_name>/` that are never image "results"
# -- excluded from list_outputs/latest_outputs/outputs_at_timestamp scans.
_NON_RESULT_DIR_NAMES = {
    "prompt",
    "_prompt_cache",
    "_text_gen_cache",
    "_technique_selection",
    "_resized_refs",
    "release",
    "_logs",
}

# When no run is currently active, cluster PNGs by mtime into "runs" -- any
# gap larger than this many seconds starts a new (older) cluster. Generous
# because `mode=all`/`quality` can legitimately take many minutes between
# 1 model finishing and the next one starting (`GENERATE_TIMEOUT_SECONDS` in
# `webapp/utils/api_client.py` budgets up to 30 minutes for a whole run).
_RUN_CLUSTER_GAP_SECONDS = 25 * 60

# Keep enough headroom for the sibling ORG directory and Windows tools that
# still use the traditional MAX_PATH limit (260 characters).
_SAFE_ABSOLUTE_PATH_LENGTH = 230


def now_timestamp_str() -> str:
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def request_output_dir(request_name: str) -> Path:
    return get_settings().output_dir / request_name


def size_model_dir(request_name: str, display_size: str, image_model: str) -> Path:
    return request_output_dir(request_name) / display_size / image_model


def org_dir(request_name: str, display_size: str, image_model: str) -> Path:
    """Where the RAW (pre-crop) image returned by the Image GenAI model is
    saved, alongside the final cropped one in `size_model_dir()` (see module
    docstring)."""
    return size_model_dir(request_name, display_size, image_model) / "ORG"


def org_image_path(request_name: str, display_size: str, image_model: str, filename: str) -> Path:
    """`filename` should be the SAME filename the final cropped image was
    saved as (`unique_image_output_path(...).name`) -- pairs the raw and
    cropped files 1:1 by filename, just in different directories."""
    return org_dir(request_name, display_size, image_model) / filename


def prompt_dir(request_name: str) -> Path:
    return request_output_dir(request_name) / "prompt"


def prompt_file_path(request_name: str, image_model: str, version: int) -> Path:
    """REQUEST.md [3d]: 1 log file per (image_model, version) -- the name
    itself records which model the prompt was actually sent to and which
    version request it was for, e.g. `prompt-gpt-image-2-version-1.txt`
    (2026-08-20, user request -- was previously `prompt-<model>.txt` only,
    with no version in the name)."""
    return prompt_dir(request_name) / f"prompt-{image_model}-version-{version}.txt"


def repair_prompt_file_path(request_name: str, image_model: str, version: int, attempt: int) -> Path:
    """Exact targeted-repair prompt sent after a failed post-crop validation."""
    return prompt_dir(request_name) / f"prompt-{image_model}-version-{version}-repair-{attempt}.txt"


def validation_report_file_path(request_name: str, image_model: str, version: int, attempt: int) -> Path:
    """Human-readable vision-LLM validation log for business review."""
    return (
        request_output_dir(request_name)
        / "_validation"
        / f"validate-{image_model}-version-{version}-attempt-{attempt}.txt"
    )


def validation_final_report_file_path(request_name: str, image_model: str, version: int) -> Path:
    """Which attempt was actually exported, after keep-best."""
    return (
        request_output_dir(request_name)
        / "_validation"
        / f"validate-{image_model}-version-{version}-FINAL.txt"
    )


def prompt_cache_dir(request_name: str) -> Path:
    return request_output_dir(request_name) / "_prompt_cache"


def prompt_cache_file(request_name: str) -> Path:
    return prompt_cache_dir(request_name) / "built_prompt.json"


def versioned_prompt_cache_file(request_name: str, cache_variant: str | None = None) -> Path:
    """Prompt cache path for one materialized version.

    The legacy no-variant path remains unchanged for direct/manual callers.
    Variant names are reduced to filesystem-safe characters because they are
    internal identifiers, not user-controlled path fragments.
    """
    if not cache_variant:
        return prompt_cache_file(request_name)
    safe_variant = "".join(ch for ch in cache_variant if ch.isalnum() or ch in ("-", "_"))
    if not safe_variant:
        return prompt_cache_file(request_name)
    return prompt_cache_dir(request_name) / f"built_prompt-{safe_variant}.json"


def text_gen_cache_dir(request_name: str) -> Path:
    return request_output_dir(request_name) / "_text_gen_cache"


def text_gen_cache_file(request_name: str) -> Path:
    return text_gen_cache_dir(request_name) / "text_gen.json"


def text_gen_prompt_log_file(request_name: str) -> Path:
    """2026-09-08 (user request): a human-readable log of the exact, fully
    filled prompt sent to the LLM to generate per-version headline/message/
    CTA copy -- kept alongside `text_gen.json` in the same `_text_gen_cache/`
    directory so both artifacts for one run's text generation live together."""
    return text_gen_cache_dir(request_name) / "text_gen_prompt.txt"


def text_gen_result_log_file(request_name: str) -> Path:
    """2026-09-08 (user request): a human-readable log of the raw per-version
    headline/message/cta JSON the LLM returned (before request-level
    validation) -- the exact counterpart of `text_gen_prompt_log_file()`
    above so a run's INPUT (prompt) and OUTPUT (generated JSON) sit side by
    side in the same `_text_gen_cache/` directory, whether that output came
    from a fresh LLM call or a reused `text_gen.json` cache hit."""
    return text_gen_cache_dir(request_name) / "text_gen_result.json"


def clear_text_gen_cache(request_name: str) -> None:
    """Keep generated copy scoped to one explicit generate request."""
    cache_dir = text_gen_cache_dir(request_name)
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
        except OSError:
            logger.exception("Could not clear text generation cache %s; its input hash will still prevent stale reuse.", cache_dir)


def technique_selection_dir(request_name: str) -> Path:
    return request_output_dir(request_name) / "_technique_selection"


def technique_selection_file(request_name: str) -> Path:
    return technique_selection_dir(request_name) / "technique_selection.json"


def technique_selection_batch_file(request_name: str, batch_index: int, first_version: int, last_version: int) -> Path:
    return technique_selection_dir(request_name) / (
        f"batch-{batch_index:03d}-versions-{first_version:03d}-{last_version:03d}.json"
    )


def technique_selection_prompt_log_file(request_name: str) -> Path:
    """2026-09-09 (user request): a human-readable log of the exact prompt(s)
    sent to the LLM to select design techniques -- kept alongside
    `technique_selection.json`/the `batch-*.json` artifacts in the same
    `_technique_selection/` directory, mirroring `text_gen.py`'s established
    `text_gen_prompt_log_file()` convention for the (unrelated) copy-writing
    LLM call. A run with more than `technique_selection_batch_size` versions
    issues several sequential batch calls -- ALL of their prompts are logged
    into this one file, one clearly labeled block per batch, in call order."""
    return technique_selection_dir(request_name) / "technique_selection_prompt.txt"


def clear_technique_selection(request_name: str) -> None:
    """Selections are scoped to one explicit Generate run."""
    selection_dir = technique_selection_dir(request_name)
    if selection_dir.exists():
        try:
            shutil.rmtree(selection_dir)
        except OSError:
            logger.exception("Could not clear technique-selection artifacts %s.", selection_dir)


def usage_db_file() -> Path:
    """The single, shared, across-every-request/every-run usage/cost
    SQLite database (2026-09-06 user request, schema redesigned twice since
    -- see `app/core/usage_tracker.py` module docstring's (2)/(3) notes).
    Deliberately NOT scoped by `request_name`, and deliberately NOT under
    `output_dir` -- always this exact fixed path (`usage/usage_log.db`, a
    sibling of `app/` at the project root), upserted into."""
    return get_settings().usage_log_db_file


def resized_refs_dir(request_name: str) -> Path:
    return request_output_dir(request_name) / "_resized_refs"


def release_dir(request_name: str) -> Path:
    return request_output_dir(request_name) / "release"


def image_filename(creative_id: str, display_size: str, image_model: str, timestamp_str: str, version: int, num_version: int) -> str:
    suffix = f"-v{version}" if num_version and num_version > 1 else ""
    return f"{creative_id}_{display_size}_{image_model}_{timestamp_str}{suffix}.png"


def image_output_path(request_name: str, creative_id: str, display_size: str, image_model: str, timestamp_str: str, version: int, num_version: int) -> Path:
    filename = image_filename(creative_id, display_size, image_model, timestamp_str, version, num_version)
    path = size_model_dir(request_name, display_size, image_model) / filename
    if len(str(path.absolute())) <= _SAFE_ABSOLUTE_PATH_LENGTH:
        return path

    # The request directory already identifies the creative, size and model.
    # Use a stable digest in the leaf filename instead of repeating those long
    # values and exceeding Windows MAX_PATH while writing the generated PNG.
    identity = f"{creative_id}|{display_size}|{image_model}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    suffix = f"-v{version}" if num_version and num_version > 1 else ""
    short_filename = f"img-{digest}_{timestamp_str}{suffix}.png"
    return size_model_dir(request_name, display_size, image_model) / short_filename


def unique_image_output_path(request_name: str, creative_id: str, display_size: str, image_model: str, timestamp_str: str, version: int, num_version: int) -> Path:
    """Same as `image_output_path()`, but guarantees the returned path does
    NOT already exist -- appends `-dupN` if it does. Belt-and-suspenders on
    top of the seconds-precision timestamp above: 2 concurrent runs that
    land in the exact same SECOND (or a clock that doesn't advance, e.g. a
    mocked/frozen clock in a test) still can't silently overwrite each
    other's file."""
    path = image_output_path(request_name, creative_id, display_size, image_model, timestamp_str, version, num_version)
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    n = 2
    while True:
        candidate = path.with_name(f"{stem}-dup{n}{suffix}")
        if not candidate.exists():
            logger.warning("Output path %s already existed -- writing to %s instead to avoid overwriting it.", path, candidate)
            return candidate
        n += 1


def clone_release_filename(source_stem: str, display_size: str, image_model: str) -> str:
    return f"{source_stem}_clone_{display_size}_{image_model}.png"


def clone_release_path(request_name: str, source_stem: str, display_size: str, image_model: str) -> Path:
    return release_dir(request_name) / clone_release_filename(source_stem, display_size, image_model)


# ---------------------------------------------------------------------------
# Listing / scanning helpers used by app/api/routes.py
# ---------------------------------------------------------------------------

def _iter_result_pngs(request_name: str):
    base = request_output_dir(request_name)
    if not base.exists():
        return
    for size_dir in base.iterdir():
        if not size_dir.is_dir() or size_dir.name in _NON_RESULT_DIR_NAMES or size_dir.name.startswith("_"):
            continue
        for model_dir in size_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for png in model_dir.glob("*.png"):
                yield size_dir.name, model_dir.name, png


def _image_dict(display_size: str, image_model: str, png: Path) -> dict:
    return {
        "display_size": display_size,
        "image_model": image_model,
        "filename": png.name,
        "path": str(png.resolve()),
        "mtime": png.stat().st_mtime,
    }


def list_new_images(request_name: str, since_ts: float) -> list[dict]:
    out = []
    for display_size, image_model, png in _iter_result_pngs(request_name):
        try:
            mtime = png.stat().st_mtime
        except OSError:
            continue
        if mtime >= since_ts:
            out.append(_image_dict(display_size, image_model, png))
    out.sort(key=lambda d: d["mtime"])
    return out


def latest_run_images(request_name: str) -> tuple[list[dict], str | None]:
    """Cluster every PNG by mtime and return the most recent cluster (the
    "latest run"), plus a human-readable timestamp for that cluster (the
    newest file's mtime)."""
    all_imgs = [
        _image_dict(size, model, png) for size, model, png in _iter_result_pngs(request_name)
    ]
    if not all_imgs:
        return [], None

    all_imgs.sort(key=lambda d: d["mtime"], reverse=True)
    cluster = [all_imgs[0]]
    for img in all_imgs[1:]:
        if cluster[-1]["mtime"] - img["mtime"] <= _RUN_CLUSTER_GAP_SECONDS:
            cluster.append(img)
        else:
            break

    run_ts = datetime.fromtimestamp(all_imgs[0]["mtime"]).strftime(TIMESTAMP_FORMAT)
    cluster.sort(key=lambda d: (d["display_size"], d["image_model"], d["filename"]))
    return cluster, run_ts


def images_matching_substring(request_name: str, needle: str) -> list[dict]:
    needle_lower = needle.strip().lower()
    if not needle_lower:
        return []
    out = [
        _image_dict(size, model, png)
        for size, model, png in _iter_result_pngs(request_name)
        if needle_lower in png.name.lower()
    ]
    out.sort(key=lambda d: (d["display_size"], d["image_model"], d["filename"]))
    return out


@dataclass
class ReleaseCopyResult:
    ok: bool
    source_path: str
    released_path: str | None = None
    error: str | None = None


def release_images(request_name: str, selected_output_paths: list[str]) -> list[ReleaseCopyResult]:
    rel_dir = release_dir(request_name)
    rel_dir.mkdir(parents=True, exist_ok=True)
    results: list[ReleaseCopyResult] = []
    for src_str in selected_output_paths:
        src = Path(src_str)
        try:
            if not src.exists():
                raise FileNotFoundError(f"source image not found: {src}")
            dest = rel_dir / src.name
            shutil.copy2(src, dest)
            results.append(ReleaseCopyResult(ok=True, source_path=str(src), released_path=str(dest.resolve())))
            logger.info("Released %s -> %s", src, dest)
        except Exception as exc:  # noqa: BLE001 -- never crash on 1 bad path
            logger.exception("Failed to release %s", src_str)
            results.append(ReleaseCopyResult(ok=False, source_path=src_str, error=str(exc)))
    return results


def list_released(request_name: str) -> list[str]:
    rel_dir = release_dir(request_name)
    if not rel_dir.exists():
        return []
    return sorted(str(p.resolve()) for p in rel_dir.glob("*.png"))


def clone_files_for_source(request_name: str, since_ts: float = 0.0, source_stem: str = "") -> list[dict]:
    rel_dir = release_dir(request_name)
    if not rel_dir.exists():
        return []
    out = []
    for png in rel_dir.glob("*_clone_*.png"):
        try:
            mtime = png.stat().st_mtime
        except OSError:
            continue
        if mtime < since_ts:
            continue
        if source_stem and not png.name.startswith(f"{source_stem}_clone_"):
            continue
        # `<source_stem>_clone_<display_size>_<image_model>.png`
        rest = png.stem.split("_clone_", 1)[-1]
        parts = rest.rsplit("_", 1)
        display_size = parts[0] if len(parts) == 2 else "-"
        image_model = parts[1] if len(parts) == 2 else "-"
        out.append(
            {
                "display_size": display_size,
                "image_model": image_model,
                "filename": png.name,
                "path": str(png.resolve()),
                "mtime": mtime,
            }
        )
    out.sort(key=lambda d: (d["display_size"], d["image_model"], d["filename"]))
    return out
