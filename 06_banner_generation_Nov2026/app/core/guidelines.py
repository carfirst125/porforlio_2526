"""Load & resolve design guidelines from `lib/guidelines/guidelines.json`
(REQUEST.md [2]).

Lookup is by `format_type` (e.g. `"banner_popup_noti"`). Each entry may list
`general_guide`: other entry names in the SAME file that are always
mandatory alongside it (e.g. `banner_popup_noti` pulls in
`banner_general_standard`). This module resolves the full, deduplicated set
(entry + all of its `general_guide` names, recursively, cycle-safe) and
returns both the combined guideline TEXT (for the prompt) and the resolved
list of guideline IMAGE paths (to attach to the prompt as reference images).

An entry may also list `"font"` (str or list of paths, same `/`-relative-to-
`lib/guidelines/` convention as `"source"`) -- one or more images showing the
actual reference typeface/lettering to match (e.g. Manrope weight samples).
These resolve into their own `ResolvedGuideline.font_image_paths` list, kept
separate from `image_paths` because -- unlike internal guideline/spec
screenshots, which are deliberately never sent to the image model (see
pipeline.py) -- font reference images ARE meant to be attached, so the
pipeline can label and include them alongside theme/context/product while
still running them through the same downsample step (2026-09-08 user
request).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("guidelines")


@dataclass
class ResolvedGuideline:
    format_type: str
    entry_names: list[str] = field(default_factory=list)
    combined_text: str = ""
    image_paths: list[Path] = field(default_factory=list)
    # 2026-09-08 (user request): a guideline entry may list `"font"` --
    # image(s) showing the actual reference typeface (e.g. Manrope weight
    # samples) -- resolved the same way as `source`, but kept in its own
    # list so callers can attach/label them separately from the internal
    # guideline/spec screenshots in `image_paths` (which are deliberately
    # NOT sent to the image model -- see pipeline.py's `_node_prepare_images`).
    font_image_paths: list[Path] = field(default_factory=list)
    missing_entries: list[str] = field(default_factory=list)


def _load_raw() -> dict:
    settings = get_settings()
    path = settings.guidelines_file
    if not path.exists():
        logger.error("guidelines.json not found at %s", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- dump, never crash the pipeline
        logger.exception("Could not parse guidelines.json at %s", path)
        return {}


def _sources_for(entry: dict) -> list[str]:
    src = entry.get("source")
    if not src:
        return []
    return src if isinstance(src, list) else [src]


def _fonts_for(entry: dict) -> list[str]:
    """`"font"`: same shape/convention as `"source"` (str or list of paths,
    leading `/`, relative to `lib/guidelines/`) -- e.g.
    `"font": ["/font/manrope_medium.png", "/font/manrope_semibold.png"]`."""
    fonts = entry.get("font")
    if not fonts:
        return []
    return fonts if isinstance(fonts, list) else [fonts]


def _resolve_source_path(src: str) -> Path:
    """`"/db/page-4.png"` -> `<guidelines.json's dir>/db/page-4.png`. Sources
    in guidelines.json are written with a leading `/`, relative to the
    `lib/guidelines/` folder that guidelines.json itself lives in."""
    settings = get_settings()
    base = settings.guidelines_file.parent
    return base / src.lstrip("/")


def resolve_guideline(format_type: str) -> ResolvedGuideline:
    raw = _load_raw()
    result = ResolvedGuideline(format_type=format_type)

    if not format_type:
        return result

    to_visit = [format_type]
    seen: set[str] = set()
    ordered_names: list[str] = []

    while to_visit:
        name = to_visit.pop(0)
        if name in seen:
            continue
        seen.add(name)
        entry = raw.get(name)
        if entry is None:
            result.missing_entries.append(name)
            logger.warning("guidelines.json has no entry named %r (referenced from format_type=%r)", name, format_type)
            continue
        ordered_names.append(name)
        for extra in entry.get("general_guide") or []:
            if extra not in seen:
                to_visit.append(extra)

    result.entry_names = ordered_names

    text_blocks: list[str] = []
    for name in ordered_names:
        entry = raw[name]
        content = (entry.get("content") or entry.get("short") or "").strip()
        if content:
            text_blocks.append(f"### Guideline: {name}\n{content}")
        for src in _sources_for(entry):
            p = _resolve_source_path(src)
            if p.exists():
                result.image_paths.append(p)
            else:
                logger.warning("Guideline reference image not found: %s (entry=%s)", p, name)
        for src in _fonts_for(entry):
            p = _resolve_source_path(src)
            if p.exists():
                result.font_image_paths.append(p)
            else:
                logger.warning("Guideline font reference image not found: %s (entry=%s)", p, name)

    result.combined_text = "\n\n".join(text_blocks)
    logger.info(
        "Resolved guideline for format_type=%r -> entries=%s, %d reference image(s), %d font reference image(s), %d missing",
        format_type,
        ordered_names,
        len(result.image_paths),
        len(result.font_image_paths),
        len(result.missing_entries),
    )
    return result
