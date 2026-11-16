"""Generate one validated marketing-copy set per requested image version.

Calling the LLM at all is opt-in through ``design_options.generate_text``
alone -- when True, a COMPLETE headline/message/cta set is always generated
for every version (never a partial/field-restricted request), called once
per ``run_generate`` call and cached for that run. Whether a given field's
generated value is actually used in the final design (vs. the static
``text.<field>`` from request.json) is a SEPARATE, independent decision made
per field by ``design_options.generate_text_headline`` /
``generate_text_message`` / ``generate_text_cta_label`` -- see
``_selected_llm_fields()``.

2026-09-08 (user request): the exact, fully filled prompt sent to the LLM is
always logged to ``output/<request_name>/_text_gen_cache/text_gen_prompt.txt``
(see ``_write_prompt_log()``), right next to the existing ``text_gen.json``
cache -- written unconditionally whenever this module builds a prompt,
before the cache-hit check, so it never goes stale on a cache hit.

2026-09-08 (user request, part 2): the raw per-version JSON the LLM
returned is likewise always logged to
``output/<request_name>/_text_gen_cache/text_gen_result.json`` (see
``_write_result_log()``) -- the OUTPUT counterpart of the prompt log above.
Written on BOTH paths that can produce a "result" this call: a cache hit
(the reused ``text_gen.json`` versions) and a fresh LLM call (the
freshly parsed JSON, logged before request-level validation so a
validation failure still leaves the raw LLM output on disk for
debugging).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.core import personas
from app.core.output_manager import text_gen_cache_file, text_gen_prompt_log_file, text_gen_result_log_file
from app.llm.base import LLMProviderError
from app.llm.factory import LLMFactory
from app.logging_conf import get_logger
from app.models.schemas import CreativeRequest, TextContent
from app.settings import PROJECT_ROOT, get_settings

logger = get_logger("text_gen")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_EDGE_DECORATION_RE = re.compile(r"^(?:[-–—•|:]+\s*)|(?:\s*[-–—•|]+)$")
_HEX_COLOR_RE = re.compile(r"(?<!\w)#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?(?:[0-9a-fA-F]{2})?(?!\w)")
_KNOWN_BARE_HEX_RE = re.compile(
    r"(?<![0-9A-Fa-f])#?d?(?:E39717|00804B|545C57|FFFFFF|0A9646|008C55|C5A049|"
    r"DDC37D|E4DABF|FCF3C5|B3F3BB|3DE6A0|0F472F|079F42)(?![0-9A-Fa-f])",
    re.IGNORECASE,
)
_BARE_HEX6_RE = re.compile(r"(?<![0-9A-Fa-f])#?([0-9A-Fa-f]{6})(?![0-9A-Fa-f])")
_CTA_FORBIDDEN_WORD_RE = re.compile(r"(?<!\w)ngay(?!\w)", re.IGNORECASE)


def _contains_spec_color_token(value: str) -> bool:
    """True when copy smuggles a production hex (with or without '#')."""
    if _HEX_COLOR_RE.search(value) or _KNOWN_BARE_HEX_RE.search(value):
        return True
    for match in _BARE_HEX6_RE.finditer(value):
        token = match.group(1)
        if any(ch in token for ch in "ABCDEFabcdef"):
            return True
    return False


def prompt_template_path(format_type: str) -> Path | None:
    specific = PROJECT_ROOT / "prompts" / f"{format_type}_text_gen_prompt_template.txt"
    if specific.exists():
        return specific
    fallback = PROJECT_ROOT / "prompts" / "default_text_gen_prompt_template.txt"
    return fallback if fallback.exists() else None


def _parse_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip()).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise ValueError("text-generation response contains no JSON object")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("text-generation response must be a JSON object")
    return parsed


def _write_prompt_log(request_name: str, prompt_text: str) -> None:
    """2026-09-08 (user request): always (re)write the exact filled prompt
    used for this run's per-version copy generation, alongside `text_gen.json`
    in `_text_gen_cache/` -- written unconditionally, BEFORE the cache-hit
    check below, so the log always reflects the prompt that was actually
    built for this call regardless of whether a fresh LLM call ends up
    happening. Mirrors `prompt_builder.write_prompt_log()`'s established
    always-overwrite convention (2026-08-20 (3)): a log that only writes on
    a cache MISS would go stale and silently lie on every cache HIT."""
    try:
        path = text_gen_prompt_log_file(request_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt_text, encoding="utf-8")
        logger.info("Wrote text-generation prompt log -> %s", path)
    except OSError:
        logger.exception("Could not write text-generation prompt log for request=%s (continuing anyway)", request_name)


def _write_result_log(request_name: str, versions: dict) -> None:
    """2026-09-08 (user request): always (re)write the raw per-version
    headline/message/cta JSON that ends up being used this call -- whether
    it came from a fresh LLM response or a reused `text_gen.json` cache hit
    -- to `_text_gen_cache/text_gen_result.json`, the OUTPUT counterpart of
    `_write_prompt_log()`'s INPUT log. Same always-overwrite convention:
    never skipped just because the file already exists."""
    try:
        path = text_gen_result_log_file(request_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote text-generation result log -> %s", path)
    except OSError:
        logger.exception("Could not write text-generation result log for request=%s (continuing anyway)", request_name)


def _selected_llm_fields(req: CreativeRequest) -> set[str]:
    """Fields whose FINAL value in the design should be the LLM-generated
    copy rather than the static `text.<field>` from request.json.

    2026-09-04 (user request): generation and selection are now separate
    concerns. `generate_creative_texts()` always asks the LLM for a
    complete headline/message/cta set per version whenever
    `design_options.generate_text=True` (see the prompt template itself,
    which has no field-level restriction). This function only decides,
    independently of `generate_text`, which of those generated fields
    actually replace the static request text -- each of the 3
    `generate_text_<field>` flags gates its own field, with no shortcut
    that turns "generate_text=True" alone into "use all 3". A field whose
    flag is False keeps the static `text.<field>` value (used as-is if
    non-empty, otherwise left empty/unused) even if the LLM produced
    something for it.
    """
    fields: set[str] = set()
    if req.design_options.generate_text_headline:
        fields.add("headline")
    if req.design_options.generate_text_message:
        fields.add("message")
    if req.design_options.generate_text_cta_label:
        fields.add("cta_label")
    return fields


def _validated_versions(parsed: dict, num_version: int, req: CreativeRequest) -> dict[int, TextContent]:
    requested = _selected_llm_fields(req)
    versions: dict[int, TextContent] = {}
    for version in range(1, num_version + 1):
        entry = parsed.get(str(version))
        if not isinstance(entry, dict):
            raise ValueError(f"missing text set for version {version}")
        generated = {
            "headline": entry.get("headline"),
            "message": entry.get("message"),
            "cta_label": entry.get("cta", entry.get("cta_label")),
        }
        values = {
            "headline": req.text.headline,
            "message": req.text.message,
            "cta_label": req.text.cta_label,
        }
        for name in requested:
            value = generated[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"version {version} must contain non-empty generated field {name}")
            if _contains_spec_color_token(value):
                raise ValueError(f"version {version} field {name} contains a non-printing hex color token")
            values[name] = value
        # Strip presentation marks occasionally invented by a copy LLM.
        # The image model receives clean copy and handles layout separately.
        for name in requested:
            value = values[name]
            cleaned = value.strip()
            previous = None
            while cleaned != previous:
                previous = cleaned
                cleaned = _EDGE_DECORATION_RE.sub("", cleaned).strip()
            if not cleaned:
                raise ValueError(f"version {version} field {name} contains only decoration")
            values[name] = cleaned
        if "cta_label" in requested:
            cta = values["cta_label"]
            word_count = len(cta.split())
            if not 2 <= word_count <= 4:
                raise ValueError(f"version {version} CTA must contain 2-4 words")
            if _CTA_FORBIDDEN_WORD_RE.search(cta):
                raise ValueError(f"version {version} CTA contains forbidden word 'ngay'")
        versions[version] = TextContent(
            headline=values["headline"].strip(),
            message=values["message"].strip(),
            cta_label=values["cta_label"].strip(),
            cta_deep_link=req.text.cta_deep_link,
        )
    return versions


def generate_creative_texts(
    request_name: str, req: CreativeRequest, num_version: int, usage_tracker=None
) -> dict[int, TextContent] | None:
    """Return complete text for versions 1..N, or ``None`` for safe fallback.

    `usage_tracker`, when passed, records the LLM call below (2026-09-06
    user request) -- this ONE call covers every version at once, so it is
    recorded with `version=None`; skipped entirely on a cache hit, since no
    call happens then."""
    num_version = max(1, num_version)
    path = prompt_template_path(req.format_type)
    if path is None:
        logger.warning("No text-generation prompt template for format_type=%r; using static request text.", req.format_type)
        return None
    try:
        template = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Could not read text-generation template %s; using static request text.", path)
        return None

    settings = get_settings()
    model = settings.llm_text_gen
    # 2026-09-04 (user request): the LLM is now always asked for a COMPLETE
    # headline/message/cta set per version, unconditionally -- the template
    # itself already has no field-level restriction. Which of those 3
    # generated fields actually end up in the final design (vs. the static
    # `text.<field>`) is decided afterward, per version, by
    # `_selected_llm_fields()` inside `_validated_versions()` -- not at
    # generation time -- so toggling `generate_text_headline/message/
    # cta_label` never requires a fresh LLM call or busts this cache.
    persona_block = personas.copy_tone_block(req)
    filled_prompt = (
        template.replace("{{DESIGN_BRIEF}}", req.design_brief or "(không có design brief)")
        .replace("{{NUM_VERSIONS}}", str(num_version))
        .replace("{{PERSONA_DIRECTION}}", persona_block or "(không dùng chân dung chi tiêu thẻ)")
    )
    if "{{PERSONA_DIRECTION}}" not in template and persona_block:
        filled_prompt = filled_prompt.rstrip() + "\n\n" + persona_block + "\n"
    _write_prompt_log(request_name, filled_prompt)
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "design_brief": req.design_brief,
                "format_type": req.format_type,
                "num_version": num_version,
                "template": template,
                "model": model,
                "audience_persona": req.audience.persona,
                "persona_direction": persona_block,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = text_gen_cache_file(request_name)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key:
                _write_result_log(request_name, cached["versions"])
                return _validated_versions(cached["versions"], num_version, req)
        except Exception:  # noqa: BLE001 - corrupt content-addressed cache should regenerate
            logger.exception("Ignoring invalid text-generation cache %s.", cache_path)

    usage_sink: dict = {} if usage_tracker is not None else None
    try:
        provider = LLMFactory.create(settings)
        raw = provider.generate(filled_prompt, model=model, usage_sink=usage_sink).strip()
        parsed = _parse_json(raw)
        _write_result_log(request_name, parsed)
        versions = _validated_versions(parsed, num_version, req)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"cache_key": cache_key, "versions": parsed}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Generated and cached %d distinct text set(s) for request=%s.", num_version, request_name)
        return versions
    except (LLMProviderError, json.JSONDecodeError, ValueError, OSError):
        logger.exception("Could not generate complete per-version text for request=%s; using static request text.", request_name)
        return None
    except Exception:  # noqa: BLE001 - optional copy enhancement must not abort image generation
        logger.exception("Unexpected per-version text-generation failure for request=%s; using static request text.", request_name)
        return None
    finally:
        if usage_tracker is not None:
            usage_tracker.record_llm(usage_sink, stage="text_gen_generate_creative_texts", version=None, fallback_model=model)
