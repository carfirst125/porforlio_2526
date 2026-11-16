"""LLM-assisted design-technique selection, once per creative version.

The selector reads the complete schema-v2 catalog, asks the configured text
LLM to choose a coherent set of techniques per group, validates exact catalog
keys deterministically, and safely falls back to empty selections. It never
reads `design_options.technique_selection.manual_techniques`; auto mode
deliberately ignores that manual field completely, even as a fallback.

2026-09-09 (user request): `design_options.technique_selection.auto_config`
(only consulted when `mode == "auto"`) now lets each request configure its
OWN per-group maximums (`max_background_techniques` /
`max_product_techniques` / `max_decoration_techniques`, see
`_effective_group_max()`) instead of one catalog-wide uniform
`selection_policy.max_per_group`, and `selection_constraints` -- a list of
extra guidance files, resolved relative to `ASSET_LIBRARY_DIR` same as every
other lib-relative path in this project -- whose full text (with any
`{{dominant_color}}` placeholder filled in) is folded into the selection
prompt per version (`_resolve_selection_constraints_text()`,
`_version_context()`).

2026-09-09 (2) (user request): `design_options.technique_selection.observe_brand`
(list of brand codes, default `["general"]`) scopes which BRAND(s)'
techniques the auto selector may even see/choose from. Because
`_build_selection_prompt()` shares ONE `"catalog"` payload across an entire
batch of versions, the shared payload is filtered down to the UNION of every
version's own `observe_brand` in that batch (`_effective_observe_brands()`,
`_filter_catalog_payload_by_brands()`); each version's OWN `observe_brand`
list is then also placed in its per-version context (`_version_context()`)
so the LLM knows which subset of that shared, unioned catalog actually
applies to THAT version, and `_validate_selection()` independently
re-checks every selected technique's `brand` against that specific
version's `observe_brand` as a deterministic safety net -- a technique that
leaked into the shared catalog only because of a SIBLING version's broader
`observe_brand` must never be accepted for a version that doesn't allow
that brand.

2026-09-14 (user request): `app/core/design_techniques.py::load_catalog()`
was rewritten to load ONE catalog file PER BRAND,
`config/design_techniques_<brand>.json` (e.g. `design_techniques_general.
json`, `design_techniques_acb.json`), instead of the old fixed pair
`design_techniques.json` (general only) + `design_techniques_ref.json`
(every other brand merged together) -- adding a brand is now just dropping
in a new file, no shared ref file to edit. `select_techniques_for_versions()`
now passes `load_catalog()` the exact union of every version's
`observe_brand` in the batch (computed via `_effective_observe_brands()`),
so ONLY the brand files actually needed this run are ever read from disk --
the `_filter_catalog_payload_by_brands()`/`_validate_selection()` per-version
filtering described above still runs on top of that, unchanged, since a
batch's union can still be broader than any single version's own
`observe_brand`.

2026-09-10 (user request): each `auto_config.max_<group>_techniques` field
(only meaningful when `mode == "auto"`) may now be EITHER the original `int`
count OR a literal list of exact catalog technique keys for that group --
see `TechniqueAutoConfig` in `app/models/schemas.py` for the full contract.
When it's a list, that group is no longer selected by the LLM at all
(`_effective_group_max()` reports an effective max of 0 for it, and
`_build_selection_prompt()` tells the LLM why); the given keys are instead
applied directly, after validation, by `_apply_fixed_techniques()` --
called on every produced `TechniqueSelection` (success, corrected, AND
empty-safe-fallback alike) right before it's recorded, so a fixed group is
honored even if the LLM call for the OTHER groups fails outright. Mixing is
per-group: e.g. `max_background_techniques` can be a fixed list while
`max_product_techniques`/`max_decoration_techniques` stay plain ints and are
still auto-selected as before.

2026-09-11 (user request): added a hard rule to `_build_selection_prompt()`
telling the selector LLM to never pick, or combine, techniques that would
increase the risk of a person/hand/limb anatomy error (extra/duplicated
limbs, malformed fingers/joints, obscured grip) or make skin/face look
plastic/waxy/CGI -- part of a broader natural-human-realism pass that also
touched `app/core/prompt_builder.py` (new unconditional `HUMAN REALISM AND
ANATOMY` prompt section), `lib/guidelines/guidelines.json` (new
`banner_art_direction` -> `HUMAN QUALITY` guideline, wired into the
banner-family entries via `general_guide`), and
`lib/prompts/mkt_techniques_sel.txt` (new numbered rule #5). The user
believed all four already existed; a repo-wide search found none of them
present in this directory, so this is the first real implementation.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core import design_techniques, output_manager
from app.core.asset_resolver import ResolvedBucketAssets
from app.core.guidelines import ResolvedGuideline
from app.llm.factory import LLMFactory
from app.logging_conf import get_logger
from app.models.schemas import CreativeRequest
from app.settings import get_settings

logger = get_logger("technique_selector")

_GROUPS = ("background", "decoration", "product")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# 2026-09-09 (user request): `design_options.technique_selection.auto_config.
# selection_constraints` files may contain a `{{dominant_color}}` placeholder
# (real example: `lib/prompts/mkt_techniques_sel.txt`) -- filled the same way
# `app/core/prompt_builder.py::_theme_instruction()` already treats
# `design_options.dominant_color`: the literal value when set, otherwise a
# descriptive fallback (never a hardcoded literal color) so the LLM still
# gets a sensible, non-empty instruction.
_DOMINANT_COLOR_PLACEHOLDER_RE = re.compile(r"\{\{\s*dominant_color\s*\}\}", re.IGNORECASE)
_SYSTEM_INSTRUCTION = (
    "You are a Senior Marketing Design Director and art-direction expert for professional banking campaigns. "
    "Select a restrained, coherent set of production techniques that strengthens the brief, respects supplied "
    "assets and brand constraints, and avoids visual clutter. Reason about compatibility, prerequisites, visual "
    "hierarchy, medium, audience, occasion, final aspect ratio, and the relationship between every selected "
    "technique. Never select an irrelevant technique merely to fill a category. Return strict JSON only."
)

_OCCASION_TERMS: dict[str, tuple[str, ...]] = {
    "tet": ("tết", "tet", "lunar new year", "lunar_new_year"),
    "lunar_new_year": ("tết", "tet", "lunar new year", "lunar_new_year"),
    "noel": ("noel", "christmas", "giáng sinh"),
    "christmas": ("noel", "christmas", "giáng sinh"),
    "new_year": ("new year", "năm mới", "tết dương", "01/01"),
    "solar_new_year": ("new year", "năm mới", "tết dương", "01/01"),
    "international_womens_day": ("8/3", "08/03", "8-3", "women's day", "quốc tế phụ nữ"),
    "08_03": ("8/3", "08/03", "8-3", "quốc tế phụ nữ"),
    "vietnamese_womens_day": ("20/10", "20-10", "phụ nữ việt nam"),
    "20_10": ("20/10", "20-10", "phụ nữ việt nam"),
    "international_childrens_day": ("1/6", "01/06", "children's day", "quốc tế thiếu nhi"),
    "01_06": ("1/6", "01/06", "quốc tế thiếu nhi"),
    "mid_autumn": ("trung thu", "mid-autumn", "mid autumn"),
    "vietnam_national_day": ("2/9", "02/09", "quốc khánh"),
    "02_09": ("2/9", "02/09", "quốc khánh"),
    "reunification_day": ("30/4", "30/04", "giải phóng miền nam", "thống nhất đất nước"),
    "30_04": ("30/4", "30/04", "giải phóng miền nam", "thống nhất đất nước"),
}

_DEVICE_TECHNIQUES = {
    "product_pop_out_frame",
    "callout_pointer_badge",
    "live_app_ui_mockup",
    "feature_icon_cluster",
}
_PRODUCT_REQUIRED_TECHNIQUES = {
    "product_pop_out_frame",
    "callout_pointer_badge",
    "live_app_ui_mockup",
    "feature_icon_cluster",
    "product_lineup_strip",
    "hero_pedestal_stage",
    "controlled_contact_shadow",
    "premium_reflection_glow",
    "product_scale_hierarchy",
}
_AI_TEXT_TECHNIQUES = {
    "outlined_headline_typography",
    "numeric_ribbon_badge",
    "seasonal_corner_tag",
}


class TechniqueRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")

    background: str = ""
    decoration: str = ""
    product: str = ""


class TechniqueSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    background: list[str] = Field(default_factory=list)
    decoration: list[str] = Field(default_factory=list)
    product: list[str] = Field(default_factory=list)
    cohesion_summary: str = ""
    rationale: TechniqueRationale = Field(default_factory=TechniqueRationale)

    def flattened_keys(self, group_order: tuple[str, ...] = _GROUPS) -> list[str]:
        keys: list[str] = []
        for group in group_order:
            keys.extend(getattr(self, group))
        return keys


class TechniqueBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    versions: dict[str, TechniqueSelection]


def empty_selection(reason: str = "No technique could be proven safe and relevant.") -> TechniqueSelection:
    return TechniqueSelection(
        background=[],
        decoration=[],
        product=[],
        cohesion_summary=reason,
        rationale=TechniqueRationale(background=reason, decoration=reason, product=reason),
    )


def _parse_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip()).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            raise ValueError("technique-selection response contains no JSON object")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("technique-selection response must be a JSON object")
    return parsed


def _request_haystack(req: CreativeRequest) -> str:
    values = [
        req.creative_id,
        req.format_type,
        req.design_brief,
        req.text.headline,
        req.text.message,
        req.text.cta_label,
        *req.buckets.theme,
        *req.buckets.context,
        *req.buckets.product,
        req.reference,
    ]
    return " ".join(values).lower()


def _occasion_matches(occasions: tuple[str, ...], haystack: str) -> bool:
    if not occasions or "generic" in occasions:
        return True
    if "explicit_seasonal_campaign" in occasions:
        return any(term in haystack for terms in _OCCASION_TERMS.values() for term in terms)
    for occasion in occasions:
        terms = _OCCASION_TERMS.get(occasion, (occasion.replace("_", " "),))
        if any(term in haystack for term in terms):
            return True
    return False


def _has_numeric_offer(req: CreativeRequest) -> bool:
    visible_copy = " ".join((req.design_brief, req.text.headline, req.text.message, req.text.cta_label))
    return bool(re.search(r"(?:\d|%|₫|vnd|đồng)", visible_copy, flags=re.IGNORECASE))


def _normalize_lib_relative(value: str) -> str:
    """Strip a leading `/`, `\\`, or `./` so a `selection_constraints` entry
    resolves the SAME way whether it's written `"prompts/x.txt"` or
    `"/prompts/x.txt"` -- unlike plain `pathlib` `/` joining (where an
    absolute-looking right-hand side silently DISCARDS the base directory
    entirely), every `selection_constraints` path is always meant to be
    relative to `ASSET_LIBRARY_DIR`, same convention as `buckets.*`/
    `reference`/`guidelines` (`app/core/asset_resolver.py::_resolve_one()`)."""
    cleaned = value.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def _resolve_selection_constraint(value: str) -> tuple[Path | None, str | None]:
    base = get_settings().asset_library_dir
    cleaned = _normalize_lib_relative(value)
    if not cleaned:
        return None, f"empty selection_constraints entry: {value!r}"
    p = (base / cleaned).resolve()
    if p.exists() and p.is_file():
        return p, None
    return None, f"file not found under ASSET_LIBRARY_DIR ({base}): {value!r} (resolved to {cleaned!r})"


def _fill_dominant_color_placeholder(text: str, dominant_color: str) -> str:
    value = (dominant_color or "").strip()
    replacement = value or "OCB's approved brand theme color (no explicit dominant_color set for this request)"
    return _DOMINANT_COLOR_PLACEHOLDER_RE.sub(replacement, text)


def _resolve_selection_constraints_text(req: CreativeRequest) -> str:
    """Best-effort: load and concatenate every `selection_constraints` file's
    content for this request (auto mode only), with `{{dominant_color}}`
    filled in. Never raises -- a missing/unreadable file is skipped and
    logged, same "optional enhancement must not abort the pipeline"
    convention as every other file lookup in this project."""
    cfg = req.design_options.technique_selection
    if cfg.mode != "auto" or not cfg.auto_config.selection_constraints:
        return ""
    blocks: list[str] = []
    for raw_path in cfg.auto_config.selection_constraints:
        path, error = _resolve_selection_constraint(raw_path)
        if path is None:
            logger.warning("Could not resolve technique selection_constraints entry: %s", error)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("Could not read technique selection_constraints file %s", path)
            continue
        filled = _fill_dominant_color_placeholder(text, req.design_options.dominant_color).strip()
        if filled:
            blocks.append(f"--- selection constraints from {raw_path} ---\n{filled}")
    return "\n\n".join(blocks)


def _persona_selection_constraints_text(req: CreativeRequest) -> str:
    from app.core import personas

    return personas.technique_constraints_block(req)


def _combined_selection_constraints_text(req: CreativeRequest) -> str:
    return "\n\n".join(
        block
        for block in (
            _resolve_selection_constraints_text(req),
            _persona_selection_constraints_text(req),
        )
        if block
    )


def _raw_group_max(req: CreativeRequest) -> dict[str, int | list[str]]:
    """The raw, un-normalized per-group `auto_config.max_*_techniques`
    values for this request -- each either an `int` count or a literal list
    of fixed catalog keys (2026-09-10 user request). Manual mode has no
    per-group config at all, so every group reports `0` (informational
    only; `_effective_group_max()` ignores this in manual mode anyway)."""
    cfg = req.design_options.technique_selection
    if cfg.mode != "auto":
        return {group: 0 for group in _GROUPS}
    auto = cfg.auto_config
    return {
        "background": auto.max_background_techniques,
        "decoration": auto.max_decoration_techniques,
        "product": auto.max_product_techniques,
    }


def _effective_group_max(req: CreativeRequest, catalog_max_per_group: int) -> dict[str, int]:
    """Per-GROUP maximum technique count the LLM selector must respect for
    this request/version.

    2026-09-09 (user request): when `technique_selection.mode == "auto"`,
    each request configures its OWN background/product/decoration maximums
    via `auto_config` -- these REPLACE the catalog's single, uniform
    `selection_policy.max_per_group` as the effective per-group cap. Manual
    mode (or a request somehow reaching this without auto_config) keeps the
    catalog's uniform value unchanged.

    2026-09-10 (user request): a group whose raw config value is a LIST
    (fixed techniques, see `_effective_fixed_techniques()`) is no longer
    selected by the LLM at all -- its effective max here is always `0`, so
    the selector is required to return an empty array for that group; the
    fixed keys are applied separately by `_apply_fixed_techniques()`."""
    cfg = req.design_options.technique_selection
    if cfg.mode != "auto":
        return {group: catalog_max_per_group for group in _GROUPS}
    return {
        group: 0 if isinstance(value, list) else max(0, int(value))
        for group, value in _raw_group_max(req).items()
    }


def _effective_fixed_techniques(req: CreativeRequest) -> dict[str, list[str]]:
    """Per-GROUP list of exact catalog technique keys fixed directly by
    `auto_config.max_<group>_techniques` (2026-09-10 user request) -- empty
    list for a group still using the plain-`int` auto-selection behavior, or
    for any group when `mode != "auto"`."""
    return {
        group: list(value) if isinstance(value, list) else []
        for group, value in _raw_group_max(req).items()
    }


def _apply_fixed_techniques(
    selection: TechniqueSelection,
    req: CreativeRequest,
    catalog: design_techniques.TechniqueCatalog,
) -> TechniqueSelection:
    """Override each group that `auto_config.max_<group>_techniques` fixed
    to a literal key list (2026-09-10 user request) with that exact list,
    replacing whatever the LLM selector produced (which was told to leave
    that group empty via `_effective_group_max()` returning 0 for it, but
    this override is deterministic and applies regardless).

    Each fixed key is validated against the loaded catalog the same way an
    LLM-selected key would be in `_validate_selection()`: it must exist,
    belong to exactly this group, and satisfy this version's
    `observe_brand`. An invalid key is dropped with a warning rather than
    failing the request -- a mistyped/disallowed key in a fixed list must
    never break generation, same "optional enhancement, never fatal"
    convention as every other best-effort helper in this module. Groups with
    no fixed override are returned unchanged (no-op when nothing is fixed)."""
    fixed = _effective_fixed_techniques(req)
    if not any(fixed.values()):
        return selection
    observe_brands = set(_effective_observe_brands(req))
    updates: dict[str, list[str]] = {}
    for group, keys in fixed.items():
        if not keys:
            continue
        valid_keys: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            definition = catalog.techniques.get(key)
            if definition is None:
                logger.warning(
                    "Ignoring unknown technique %r fixed via auto_config.max_%s_techniques for request=%s",
                    key, group, req.creative_id,
                )
                continue
            if definition.group != group:
                logger.warning(
                    "Ignoring technique %r fixed for group %r (catalog says it belongs to group %r) for request=%s",
                    key, group, definition.group, req.creative_id,
                )
                continue
            if definition.brand not in observe_brands:
                logger.warning(
                    "Ignoring technique %r fixed for group %r: brand %r not in observe_brand=%s for request=%s",
                    key, group, definition.brand, sorted(observe_brands), req.creative_id,
                )
                continue
            seen.add(key)
            valid_keys.append(key)
        updates[group] = valid_keys
    if not updates:
        return selection
    rationale_updates = {
        group: (
            f"Fixed directly by request config (auto_config.max_{group}_techniques list override); "
            "not chosen by the selector LLM."
        )
        for group in updates
    }
    return selection.model_copy(
        update={
            **updates,
            "rationale": selection.rationale.model_copy(update=rationale_updates),
        }
    )


def _effective_observe_brands(req: CreativeRequest) -> list[str]:
    """Brand(s) whose techniques THIS version's auto selector may see/choose
    from (2026-09-09 (2), user request). `TechniqueSelectionConfig`'s own
    validator already normalizes an empty/missing value to `["general"]`, so
    this just re-reads that already-normalized field -- kept as its own
    helper (mirroring `_effective_group_max()`) so every call site (prompt
    filtering, per-version context, validation) shares one definition."""
    return list(req.design_options.technique_selection.observe_brand) or ["general"]


def _filter_catalog_payload_by_brands(
    catalog_payload: dict[str, Any], allowed_brands: set[str]
) -> dict[str, Any]:
    """Restrict a `TechniqueCatalog.selector_payload()` dict's
    `design_techniques` list to only entries whose `brand` is in
    `allowed_brands` -- e.g. `observe_brand=["acb"]` shows the LLM ONLY acb
    techniques; `["general","acb"]` shows both. `incompatible_with`/
    `works_well_with` references inside the remaining entries are left as-is
    (purely descriptive text in the prompt; deterministic validation in
    `_validate_selection()` is what actually enforces compatibility)."""
    filtered = dict(catalog_payload)
    filtered["design_techniques"] = [
        item
        for item in catalog_payload.get("design_techniques", [])
        if item.get("brand") in allowed_brands
    ]
    return filtered


def _validate_selection(
    selection: TechniqueSelection,
    req: CreativeRequest,
    catalog: design_techniques.TechniqueCatalog,
) -> list[str]:
    errors: list[str] = []
    policy = catalog.selection_policy
    min_per_group = max(0, int(policy.get("min_per_group", 0)))
    catalog_max_per_group = max(min_per_group, int(policy.get("max_per_group", 3)))
    group_max = _effective_group_max(req, catalog_max_per_group)
    catalog_max_total = max(0, int(policy.get("max_total_per_version", 6)))
    # A request's own per-group caps (auto_config.max_*_techniques) take
    # priority over the catalog's generic max_total_per_version -- raise the
    # effective ceiling to their sum so a legitimate full per-group selection
    # (e.g. 3+3+3=9) is never rejected purely because it exceeds a catalog
    # default sized for the OLD uniform 0-3-per-group policy.
    max_total = max(catalog_max_total, sum(group_max.values()))

    all_keys: list[str] = []
    for group in _GROUPS:
        keys = getattr(selection, group)
        max_per_group = max(min_per_group, group_max.get(group, catalog_max_per_group))
        if not min_per_group <= len(keys) <= max_per_group:
            errors.append(f"group {group!r} must contain {min_per_group}-{max_per_group} keys, got {len(keys)}")
        if len(keys) != len(set(keys)):
            errors.append(f"group {group!r} contains duplicate keys")
        for key in keys:
            definition = catalog.techniques.get(key)
            if definition is None:
                errors.append(f"unknown technique key {key!r}")
            elif definition.group != group:
                errors.append(f"technique {key!r} belongs to group {definition.group!r}, not {group!r}")
        all_keys.extend(keys)

    if len(all_keys) != len(set(all_keys)):
        errors.append("a technique key appears in more than one group")
    if len(all_keys) > max_total:
        errors.append(f"selection has {len(all_keys)} techniques, exceeding max_total_per_version={max_total}")

    selected = set(all_keys)
    for key in selected:
        definition = catalog.techniques.get(key)
        if definition is None:
            continue
        conflicts = selected.intersection(definition.incompatible_with)
        if conflicts:
            errors.append(f"technique {key!r} conflicts with {sorted(conflicts)}")

    haystack = _request_haystack(req)
    has_product = bool(req.buckets.product)
    has_context = bool(req.buckets.context) or req.design_options.generate_person
    device_terms = ("phone", "smartphone", "device", "mobile", "app", "ui", "điện thoại", "ứng dụng", "màn hình")
    has_device_context = has_product and any(term in haystack for term in device_terms)
    observe_brands = _effective_observe_brands(req)

    for key in selected:
        definition = catalog.techniques.get(key)
        if definition is None:
            continue
        # 2026-09-09 (2) (user request): deterministic safety net -- the
        # shared batch-level catalog payload is filtered to the UNION of
        # every version's observe_brand (`_filter_catalog_payload_by_brands`
        # in `_build_selection_prompt`), so a technique allowed for a
        # SIBLING version in the same batch could still show up here for a
        # version whose own observe_brand doesn't include that brand.
        if definition.brand not in observe_brands:
            errors.append(
                f"technique {key!r} has brand {definition.brand!r} not in this version's "
                f"observe_brand={observe_brands!r}"
            )
        if not _occasion_matches(definition.occasion, haystack):
            errors.append(f"technique {key!r} requires occasion {list(definition.occasion)!r}, not found in request")
        if key in _AI_TEXT_TECHNIQUES and req.design_options.text_by_code:
            errors.append(f"technique {key!r} requires image-model text but text_by_code=true")
        if key == "outlined_headline_typography" and not req.text.headline.strip():
            errors.append("outlined_headline_typography requires a non-empty headline")
        if key == "numeric_ribbon_badge" and not _has_numeric_offer(req):
            errors.append("numeric_ribbon_badge requires an approved numeric offer")
        if key == "character_cutout_ground_shadow" and not has_context:
            errors.append("character_cutout_ground_shadow requires a supplied or permitted person/context")
        if key == "real_person_presenting_card_or_phone" and not has_context:
            errors.append("real_person_presenting_card_or_phone requires generate_person or a context person")
        if key == "layered_depth_overlap" and not (has_product or has_context):
            errors.append("layered_depth_overlap requires a foreground product or context subject")
        if key in _PRODUCT_REQUIRED_TECHNIQUES and not has_product:
            errors.append(f"technique {key!r} requires a product asset or product description")
        if key in _DEVICE_TECHNIQUES and not has_device_context:
            errors.append(f"technique {key!r} requires clear phone/device/app evidence in the request")
        if key == "product_lineup_strip" and len(req.buckets.product) < 2:
            errors.append("product_lineup_strip requires at least two declared product assets/descriptions")

    decorative_limit = req.design_options.num_decorative_elements
    if decorative_limit is not None and len(selection.decoration) > decorative_limit:
        errors.append(
            f"decoration selection count {len(selection.decoration)} exceeds num_decorative_elements={decorative_limit}"
        )
    return sorted(set(errors))


def _validate_batch(
    parsed: TechniqueBatchResponse,
    version_ids: list[int],
    version_reqs: dict[int, CreativeRequest],
    catalog: design_techniques.TechniqueCatalog,
    prior_signatures: set[tuple[str, ...]],
) -> list[str]:
    expected = {str(version) for version in version_ids}
    actual = set(parsed.versions)
    errors: list[str] = []
    if actual != expected:
        errors.append(f"versions keys must be exactly {sorted(expected)}, got {sorted(actual)}")
        return errors

    batch_signatures: set[tuple[str, ...]] = set()
    for version in version_ids:
        selection = parsed.versions[str(version)]
        errors.extend(f"version {version}: {error}" for error in _validate_selection(selection, version_reqs[version], catalog))
        signature = tuple(selection.flattened_keys())
        # Empty sets are allowed to repeat: sometimes no technique is the
        # only brief-safe decision. Non-empty exact duplicates are corrected.
        if signature and (signature in prior_signatures or signature in batch_signatures):
            errors.append(f"version {version}: non-empty technique set exactly duplicates an earlier version")
        batch_signatures.add(signature)
    return errors


def _version_context(req: CreativeRequest) -> dict[str, Any]:
    # 2026-09-09 (user request): auto mode's own per-group maximums and
    # resolved selection_constraints text are per-VERSION (each version's
    # req can carry its own dominant_color/auto_config), so both are folded
    # into THIS version's context rather than shared across the whole batch.
    catalog_default_max = 3
    return {
        "creative_id": req.creative_id,
        "format_type": req.format_type,
        "design_brief": req.design_brief,
        "final_text": req.text.model_dump(),
        "buckets": req.buckets.model_dump(),
        "reference": req.reference,
        "flags": {
            "generate_person": req.design_options.generate_person,
            "has_logo": req.design_options.has_logo,
            "text_by_code": req.design_options.text_by_code,
            "dominant_color": req.design_options.dominant_color,
            "num_decorative_elements": req.design_options.num_decorative_elements,
        },
        "geometry": {
            "canvas": req.canvas_wh(),
            "crop_box": req.crop_box(),
            "final_output": req.final_output_wh(),
        },
        # Per-group maximum technique count THIS version must respect --
        # from `design_options.technique_selection.auto_config` when
        # mode=="auto" (overrides the catalog's generic uniform default);
        # unused/informational otherwise. A group fixed via a literal key
        # list (see `technique_selection_fixed_groups` below) always reports
        # 0 here (2026-09-10 user request).
        "technique_selection_max_per_group": _effective_group_max(req, catalog_default_max),
        # 2026-09-10 (user request): groups whose `auto_config.max_<group>_
        # techniques` is a literal list of catalog keys rather than a count
        # -- these are applied directly by `_apply_fixed_techniques()` after
        # this LLM call returns, NOT selected by the LLM. Listed here (keys
        # only, group -> fixed keys) purely so the LLM understands WHY that
        # group's max is 0 and never tries to fill it anyway. Empty dict
        # when no group is fixed for this version.
        "technique_selection_fixed_groups": {
            group: keys for group, keys in _effective_fixed_techniques(req).items() if keys
        },
        # Full text content of every `auto_config.selection_constraints` file
        # declared for THIS version, `{{dominant_color}}` already filled in.
        # Empty string when no constraints are declared or mode != "auto".
        "technique_selection_constraints": _combined_selection_constraints_text(req),
        # 2026-09-09 (2) (user request): brand(s) THIS version may select
        # techniques from. The shared "catalog" payload above is already the
        # UNION of every version's observe_brand in this batch -- this field
        # tells the LLM which subset of that shared catalog is actually valid
        # for THIS specific version (also independently re-checked in
        # `_validate_selection()`).
        "technique_selection_observe_brand": _effective_observe_brands(req),
    }


def _build_selection_prompt(
    version_ids: list[int],
    version_reqs: dict[int, CreativeRequest],
    catalog: design_techniques.TechniqueCatalog,
    guideline: ResolvedGuideline,
    assets: ResolvedBucketAssets,
    prior_summary: list[dict[str, Any]],
) -> str:
    payload = {
        "task": (
            "Select one coherent art-direction technique set for each requested version. Select 0-3 exact keys "
            "from each group and never force a technique when none is relevant. The brief and locked-asset rules "
            "are mandatory. Prefer fewer techniques when sufficient and obey max_total_per_version."
        ),
        "output_contract": {
            "format": "strict JSON object with one 'versions' object and no other top-level keys",
            "version_shape": {
                "background": "array of exact catalog keys",
                "decoration": "array of exact catalog keys",
                "product": "array of exact catalog keys",
                "cohesion_summary": "short explanation",
                "rationale": {"background": "short reason", "decoration": "short reason", "product": "short reason"},
            },
            "required_version_ids": [str(version) for version in version_ids],
        },
        "hard_rules": [
            "Zero selected keys in any or all groups is valid and preferred over an irrelevant choice.",
            "Use exact catalog keys only; aliases are informational and must not be returned.",
            "Obey preconditions, avoid_when, occasion, visual_density, incompatible_with, and request flags.",
            "Do not select a technique that changes, redraws, duplicates, crops, or invents a supplied asset.",
            "2026-09-11 (user request): never select, or combine, a technique whose effect would increase the risk "
            "of a person/hand/limb anatomy error (e.g. duplicated or extra hands/arms/legs, distorted or "
            "malformed fingers/joints, obscured or ambiguous hand-to-product grip) or that would make a person's "
            "skin or face look plastic, waxy, overly smooth, or CGI/mannequin-like. When a technique's own "
            "description or tags do not clearly rule this out for a version that includes a person, prefer the "
            "safer alternative or omit that group entirely.",
            "Keep each non-empty version set distinct when another equally relevant combination exists; key overlap is allowed.",
            "The techniques must work as one composition, not as unrelated visual effects.",
            "Each version's own `technique_selection_max_per_group` (background/decoration/product) is a HARD per-group "
            "cap for that version -- never exceed it, even if the catalog would otherwise allow more.",
            "When a version's `technique_selection_constraints` is non-empty, it is MANDATORY selection guidance for "
            "that version: only select techniques that satisfy every rule stated there, in addition to every other "
            "rule in this prompt. Audience-persona blocks in that field describe lifestyle mood only; they never "
            "override format_type color, layout, or locked assets.",
            "The shared 'catalog' below is the UNION of every technique brand allowed across ALL versions in this "
            "batch. Each version's own `technique_selection_observe_brand` (in that version's context) lists which "
            "'brand' values are actually allowed FOR THAT VERSION -- never select a technique whose 'brand' is not "
            "in that specific version's own observe_brand list, even though it may legally appear for another "
            "version in this same batch.",
            "When a version's `technique_selection_fixed_groups` lists a group (background/decoration/product), "
            "that group's techniques have ALREADY been fixed directly by the request config and will be applied "
            "outside this selection step -- you MUST return an empty array for that group for that version (its "
            "`technique_selection_max_per_group` is already 0 for the same reason). Do not select anything for it "
            "and do not repeat those fixed keys in another group either.",
        ],
        "catalog": _filter_catalog_payload_by_brands(
            catalog.selector_payload(),
            set().union(*(_effective_observe_brands(version_reqs[version]) for version in version_ids)),
        ),
        "shared_context": {
            "guideline": guideline.combined_text,
            "resolved_asset_filenames": {
                "theme": [path.name for path in assets.theme],
                "context": [path.name for path in assets.context],
                "product": [path.name for path in assets.product],
                "reference": assets.reference.name if assets.reference else "",
            },
            "bucket_text_descriptions": {
                "theme": assets.theme_text,
                "context": assets.context_text,
                "product": assets.product_text,
            },
        },
        "versions": {str(version): _version_context(version_reqs[version]) for version in version_ids},
        "prior_selection_summary": prior_summary,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _write_prompt_log(request_name: str, blocks: list[str]) -> None:
    """2026-09-09 (user request): always (re)write the exact prompt(s) used to
    select techniques for this run -- to `_technique_selection/
    technique_selection_prompt.txt`, alongside the JSON artifacts. Called
    again after every batch (not just once at the end) so the log always
    reflects every batch actually built so far, even if a later batch's LLM
    call fails or the run is interrupted -- same always-overwrite-with-
    current-state convention as `app/core/text_gen.py::_write_prompt_log()`."""
    try:
        path = output_manager.technique_selection_prompt_log_file(request_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        logger.info("Wrote technique-selection prompt log -> %s", path)
    except OSError:
        logger.exception(
            "Could not write technique-selection prompt log for request=%s (continuing anyway)", request_name
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        logger.exception("Could not write technique-selection artifact %s", path)


# ── Technique-selection content-addressed cache ───────────────────────────────
# Cache key = SHA256(request_sha256 + ":" + catalog_sha256).
# Invalidated automatically when the creative brief OR catalog changes.
# Location: output/_technique_selection_cache/<key[:2]>/<key>.json
# (global/shared across all requests so identical briefs share one entry).

def _selection_cache_key(request_sha256: str, catalog_sha256: str) -> str:
    """Stable 64-hex cache key; changes iff request brief OR catalog changes."""
    return hashlib.sha256(f"{request_sha256}:{catalog_sha256}".encode()).hexdigest()


def _selection_cache_path(cache_key: str) -> Path:
    return get_settings().output_dir / "_technique_selection_cache" / cache_key[:2] / f"{cache_key}.json"


def _load_selection_cache(
    cache_key: str, version_ids: list[int]
) -> "dict[int, TechniqueSelection] | None":
    """Return cached TechniqueSelection objects if valid and complete, else None (never raises)."""
    try:
        path = _selection_cache_path(cache_key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_versions = data.get("versions", {})
        result: dict[int, TechniqueSelection] = {}
        for version in version_ids:
            raw = cached_versions.get(str(version))
            if raw is None:
                return None  # version count changed → cache miss
            result[version] = TechniqueSelection.model_validate(raw)
        return result
    except Exception:
        logger.debug("Technique selection cache miss/corrupt for key=%s…", cache_key[:16])
        return None


def _save_selection_cache(
    cache_key: str,
    selections: "dict[int, TechniqueSelection]",
    catalog_sha256: str,
    request_sha256: str,
) -> None:
    """Persist final selections to disk cache (never raises)."""
    try:
        payload = {
            "cache_key": cache_key,
            "catalog_sha256": catalog_sha256,
            "request_sha256": request_sha256,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "versions": {str(v): sel.model_dump() for v, sel in selections.items()},
        }
        _write_json(_selection_cache_path(cache_key), payload)
    except Exception:
        logger.debug("Could not write technique selection cache key=%s…", cache_key[:16])

# ── End cache helpers ─────────────────────────────────────────────────────────


def _request_hash(version_reqs: dict[int, CreativeRequest]) -> str:
    payload = {str(version): req.model_dump(mode="json") for version, req in version_reqs.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def select_techniques_for_versions(
    request_name: str,
    version_reqs: dict[int, CreativeRequest],
    guideline: ResolvedGuideline,
    assets: ResolvedBucketAssets,
    usage_tracker=None,
) -> dict[int, TechniqueSelection]:
    """Select validated techniques for every version, batched sequentially.

    This is best-effort and never raises. Failed batches receive empty safe
    selections, which is always preferable to violating the creative brief.

    `usage_tracker`, when passed, records EVERY attempt that actually calls
    the LLM below (2026-09-06 user request) -- up to 2 per batch (initial +
    correction retry), each its own billable call. A batch covers several
    versions at once, so these are recorded with `version=None`; the exact
    version range is in each record's `notes`.
    """
    # Enforce auto-mode isolation again inside the public selector boundary,
    # not only in the LangGraph loader. This keeps direct callers from
    # accidentally making the manual list affect prompt context or hashes.
    version_reqs = {
        version: req.model_copy(
            update={
                "design_options": req.design_options.model_copy(update={"techniques": []})
            }
        )
        if req.design_options.technique_selection.mode == "auto" and req.design_options.techniques
        else req
        for version, req in version_reqs.items()
    }
    settings = get_settings()
    # 2026-09-14 (user request): only load the `design_techniques_<brand>.
    # json` file(s) actually needed this batch -- the union of every
    # version's own `observe_brand` (never empty, `_effective_observe_brands()`
    # already normalizes a missing/empty value to `["general"]`), rather
    # than the old always-merge-everything behavior. The per-version
    # `_filter_catalog_payload_by_brands()`/`_validate_selection()` brand
    # checks further down still narrow this shared, unioned catalog down to
    # each version's OWN `observe_brand` -- unchanged.
    requested_brands: set[str] = set()
    for req in version_reqs.values():
        requested_brands.update(_effective_observe_brands(req))
    catalog = design_techniques.load_catalog(sorted(requested_brands) or ["general"])
    version_ids = sorted(version_reqs)
    batch_size = max(1, settings.technique_selection_batch_size)
    selections: dict[int, TechniqueSelection] = {}
    records: dict[str, Any] = {}
    prior_summary: list[dict[str, Any]] = []
    prior_signatures: set[tuple[str, ...]] = set()

    if not catalog.techniques or catalog.validation_errors:
        error = (
            "Technique catalog is missing, malformed, or contains no valid definitions."
            if not catalog.techniques
            else "Technique catalog failed validation: " + "; ".join(catalog.validation_errors)
        )
        logger.error("%s Using empty safe fallback for request=%s.", error, request_name)
        for version in version_ids:
            selection = empty_selection(error)
            selections[version] = selection
            records[str(version)] = {
                **selection.model_dump(),
                "status": "empty_safe_fallback",
                "errors": [error],
            }
        _write_json(
            output_manager.technique_selection_file(request_name),
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "empty_safe_fallback",
                "catalog_sha256": catalog.sha256,
                "request_sha256": _request_hash(version_reqs),
                "batch_size": batch_size,
                "versions": records,
            },
        )
        return selections

    # ── Cache lookup ──────────────────────────────────────────────────────────
    _req_sha = _request_hash(version_reqs)
    _cache_key = _selection_cache_key(_req_sha, catalog.sha256)
    if settings.technique_selection_cache_enabled:
        _cached = _load_selection_cache(_cache_key, version_ids)
        if _cached is not None:
            logger.info(
                "Technique selection cache HIT (key=%s…) for request=%s — skipping LLM call.",
                _cache_key[:16], request_name,
            )
            _cache_records = {
                str(v): {**sel.model_dump(), "status": "cache_hit", "cache_key": _cache_key[:16]}
                for v, sel in _cached.items()
            }
            _write_json(
                output_manager.technique_selection_file(request_name),
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "cache_hit",
                    "cache_key": _cache_key,
                    "catalog_sha256": catalog.sha256,
                    "request_sha256": _req_sha,
                    "versions": _cache_records,
                },
            )
            return _cached
    # ── End cache lookup ──────────────────────────────────────────────────────

    try:
        provider = LLMFactory.create(settings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not create Technique Selector LLM provider; using empty safe fallback.")
        for version in version_ids:
            # Catalog loaded fine above (only the LLM provider failed), so a
            # group fixed via a literal key list (2026-09-10 user request)
            # can still be honored even though nothing else got selected.
            selection = _apply_fixed_techniques(empty_selection(str(exc)), version_reqs[version], catalog)
            selections[version] = selection
            records[str(version)] = {**selection.model_dump(), "status": "empty_safe_fallback", "errors": [str(exc)]}
        _write_json(
            output_manager.technique_selection_file(request_name),
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "empty_safe_fallback",
                "catalog_sha256": catalog.sha256,
                "request_sha256": _request_hash(version_reqs),
                "batch_size": batch_size,
                "versions": records,
            },
        )
        return selections

    batch_records: list[dict[str, Any]] = []
    prompt_log_blocks: list[str] = []
    for batch_index, offset in enumerate(range(0, len(version_ids), batch_size), start=1):
        batch_versions = version_ids[offset : offset + batch_size]
        prompt = _build_selection_prompt(batch_versions, version_reqs, catalog, guideline, assets, prior_summary)
        prompt_log_blocks.append(
            f"===== batch {batch_index} -- versions {batch_versions} =====\n{prompt}"
        )
        _write_prompt_log(request_name, prompt_log_blocks)
        raw_responses: list[str] = []
        validation_errors: list[str] = []
        accepted: TechniqueBatchResponse | None = None
        accepted_attempt = 0

        for attempt in range(1, 3):
            try:
                call_prompt = prompt
                if attempt == 2:
                    call_prompt += (
                        "\n\nCORRECTION REQUIRED. The previous response was invalid. Return a corrected strict JSON "
                        "response for the same catalog and version IDs only.\nVALIDATION ERRORS:\n- "
                        + "\n- ".join(validation_errors)
                        + "\nPREVIOUS RESPONSE:\n"
                        + (raw_responses[-1] if raw_responses else "(call failed before producing a response)")
                    )
                usage_sink: dict = {} if usage_tracker is not None else None
                raw_response = provider.generate(
                    call_prompt, system=_SYSTEM_INSTRUCTION, model=settings.llm_model_name, usage_sink=usage_sink
                )
                if usage_tracker is not None:
                    usage_tracker.record_llm(
                        usage_sink,
                        stage="technique_selector",
                        version=None,
                        fallback_model=settings.llm_model_name,
                        notes=f"batch={batch_index} versions={batch_versions} attempt={attempt}",
                    )
                raw_responses.append(raw_response)
                parsed = TechniqueBatchResponse.model_validate(_parse_json(raw_response))
                validation_errors = _validate_batch(parsed, batch_versions, version_reqs, catalog, prior_signatures)
                if not validation_errors:
                    accepted = parsed
                    accepted_attempt = attempt
                    break
            except Exception as exc:  # noqa: BLE001 -- selection is optional, fallback must remain available
                logger.exception(
                    "Technique selection failed request=%s batch=%d attempt=%d", request_name, batch_index, attempt
                )
                validation_errors = [f"{type(exc).__name__}: {exc}"]

        batch_status = "auto_selected" if accepted_attempt == 1 else "corrected" if accepted is not None else "empty_safe_fallback"
        if accepted is None:
            logger.warning(
                "Technique selector could not produce a valid result for versions=%s; using empty safe fallback: %s",
                batch_versions,
                "; ".join(validation_errors),
            )
            for version in batch_versions:
                selections[version] = empty_selection()
        else:
            for version in batch_versions:
                selections[version] = accepted.versions[str(version)]

        # 2026-09-10 (user request): apply any group fixed via a literal
        # key list in `auto_config.max_<group>_techniques` -- the LLM above
        # was already told (via `technique_selection_max_per_group`/
        # `technique_selection_fixed_groups`) to leave that group empty, but
        # this override is deterministic and runs regardless of whether the
        # LLM call succeeded, was corrected, or fell back to empty, so a
        # fixed group is always honored.
        for version in batch_versions:
            selections[version] = _apply_fixed_techniques(selections[version], version_reqs[version], catalog)

        for version in batch_versions:
            selection = selections[version]
            signature = tuple(selection.flattened_keys())
            if signature:
                prior_signatures.add(signature)
            prior_summary.append(
                {
                    "version": version,
                    "background": selection.background,
                    "decoration": selection.decoration,
                    "product": selection.product,
                }
            )
            records[str(version)] = {
                **selection.model_dump(),
                "status": batch_status,
                "batch_index": batch_index,
                "validation_errors": validation_errors if accepted is None else [],
            }

        batch_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "batch_index": batch_index,
            "version_ids": batch_versions,
            "status": batch_status,
            "accepted_attempt": accepted_attempt,
            "validation_errors": validation_errors if accepted is None else [],
            "raw_responses": raw_responses,
            "selections": {str(version): records[str(version)] for version in batch_versions},
        }
        batch_records.append(batch_payload)
        _write_json(
            output_manager.technique_selection_batch_file(
                request_name, batch_index, batch_versions[0], batch_versions[-1]
            ),
            batch_payload,
        )

    statuses = {record["status"] for record in records.values()}
    overall_status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
    # ── Cache save ────────────────────────────────────────────────────────────
    if settings.technique_selection_cache_enabled and overall_status not in ("empty_safe_fallback",):
        _save_selection_cache(_cache_key, selections, catalog.sha256, _req_sha)
    # ── End cache save ────────────────────────────────────────────────────────
    _write_json(
        output_manager.technique_selection_file(request_name),
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": overall_status,
            "selector_provider": settings.llm_provider,
            "selector_model": settings.llm_model_name,
            "catalog_schema_version": catalog.schema_version,
            "catalog_sha256": catalog.sha256,
            "request_sha256": _request_hash(version_reqs),
            "effective_num_version": len(version_ids),
            "batch_size": batch_size,
            "batches": [
                {
                    "batch_index": item["batch_index"],
                    "version_ids": item["version_ids"],
                    "status": item["status"],
                    "accepted_attempt": item["accepted_attempt"],
                }
                for item in batch_records
            ],
            "versions": records,
        },
    )
    return selections
