"""Load `config/design_techniques_<brand>.json` catalog file(s) and resolve
`design_options.techniques` (REQUEST.md background-design techniques, see
`version_1/6-design-style-guide.md` for where these technique
descriptions/prompt wording originally came from -- silhouette watermark,
textured surface background, radial spotlight/framing, ...).

2026-09-14 (user request): ONE catalog file PER BRAND, named
`config/design_techniques_<brand>.json` -- e.g. `design_techniques_general.
json` ("general" = techniques not tied to any specific bank/brand),
`design_techniques_acb.json`, `design_techniques_ocb_db.json`, and any
future `design_techniques_<new_brand>.json`. This replaces the previous
fixed pair (`design_techniques.json` for "general" only +
`design_techniques_ref.json` merging every OTHER brand together) -- adding
a brand no longer means editing a shared ref file, just dropping in a new
`design_techniques_<brand>.json`. `load_catalog(brands=[...])` loads and
merges exactly the requested brands' files (normally the union of every
version's `design_options.technique_selection.observe_brand` in the current
batch, see `app/core/technique_selector.py`); `load_catalog()` with no
`brands` argument (used by `resolve_techniques()` below, which has no
brand-scoping concept) auto-discovers and merges EVERY
`design_techniques_*.json` file present under `config/`.

Each file's shape (unchanged):

    {
      "schema_version": "2.0",
      "selection_policy": {...},
      "group_definitions": {...},
      "design_techniques": {
        "silhoutte_watermark": {
          "group": "background",
          "brand": "general",
          "label": "Silhouette Watermark",
          "description": "...",
          "prompt_instruction": "...",
          ... (other fields, e.g. opacity_range_pct/keywords, are carried
              but not currently used by the prompt builder)
        },
        ...
      }
    }

2026-09-02 (user request): `design_options.techniques` (a list of keys, e.g.
`["silhoutte_watermark"]`) is resolved here into the matching entries'
`description` + `prompt_instruction`, which `app/core/prompt_builder.py`
folds into a "DESIGN TECHNIQUES" section of the final image-gen prompt. An
empty `design_options.techniques` (the default) resolves to nothing --
behavior is unchanged for every existing request.json that doesn't set it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("design_techniques")


@dataclass
class ResolvedTechnique:
    key: str
    label: str = ""
    description: str = ""
    prompt_instruction: str = ""


@dataclass(frozen=True)
class TechniqueDefinition:
    """One validated selector-facing entry from catalog schema v2.

    2026-09-09 (user request): `brand` tags which brand this technique
    belongs to -- `"general"` for brand-agnostic techniques, or a specific
    bank code (e.g. `"acb"`, `"ocb_db"`, and future additions) otherwise.

    2026-09-14 (user request): each brand's entries now live in their OWN
    file, `config/design_techniques_<brand>.json` (see this module's
    docstring and `load_catalog()`) -- an entry's `brand` field should match
    the brand of the file it was loaded from (`load_catalog()` uses the
    requested/discovered brand as the `default_brand` when the entry itself
    omits `"brand"`), but the field itself is still what every filtering
    check (`app/core/technique_selector.py`'s `observe_brand` filtering)
    actually reads -- not the filename.
    """

    key: str
    group: str
    label: str
    description: str
    keywords: tuple[str, ...] = ()
    prompt_instruction: str = ""
    tags: tuple[str, ...] = ()
    occasion: tuple[str, ...] = ("generic",)
    visual_density: str = "medium"
    preconditions: tuple[str, ...] = ()
    avoid_when: tuple[str, ...] = ()
    incompatible_with: tuple[str, ...] = ()
    works_well_with: tuple[str, ...] = ()
    selection_notes: str = ""
    aliases: tuple[str, ...] = ()
    brand: str = "general"

    def selector_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "group": self.group,
            "brand": self.brand,
            "label": self.label,
            "description": self.description,
            "keywords": list(self.keywords),
            "prompt_instruction": self.prompt_instruction,
            "tags": list(self.tags),
            "occasion": list(self.occasion),
            "visual_density": self.visual_density,
            "preconditions": list(self.preconditions),
            "avoid_when": list(self.avoid_when),
            "incompatible_with": list(self.incompatible_with),
            "works_well_with": list(self.works_well_with),
            "selection_notes": self.selection_notes,
        }


@dataclass(frozen=True)
class TechniqueCatalog:
    schema_version: str = ""
    selection_policy: dict[str, Any] = field(default_factory=dict)
    group_definitions: dict[str, str] = field(default_factory=dict)
    techniques: dict[str, TechniqueDefinition] = field(default_factory=dict)
    sha256: str = ""
    validation_errors: tuple[str, ...] = ()

    def selector_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_policy": self.selection_policy,
            "group_definitions": self.group_definitions,
            "design_techniques": [item.selector_payload() for item in self.techniques.values()],
        }


_GROUPS = {"background", "decoration", "product"}
# 2026-09-09 (user request): some brand catalog files also define a 4th
# "text" group (text display style) that the operational selector
# (`app/core/technique_selector.py::TechniqueSelection`, still
# background/decoration/product only) doesn't support yet. Entries in that
# group are skipped SILENTLY when loading ANY brand file (2026-09-14: no
# longer just the old "ref" file -- applies uniformly to every
# `design_techniques_<brand>.json`) -- this is an expected, documented gap
# (see a brand file's own `schema_notes`), not a config bug, so it must
# never add a `validation_errors` entry (which would blank out ALL
# technique selection for every request, general included).
_UNSUPPORTED_GROUPS = {"text"}

# 2026-09-14 (user request): a brand code (from `observe_brand`, i.e. from
# untrusted request.json content) must be a safe filename fragment before
# it's ever joined into `design_techniques_<brand>.json` -- this is the same
# allow-list a brand-file's own `"brand"` field is expected to use
# (`config/design_techniques_general.json`'s `metadata_contract.brand`
# examples: "general", "acb", "vpb", "vcb"). Anything else (path separators,
# "..", empty string, ...) is rejected outright rather than silently
# stripped, so a malformed brand can never resolve to an unintended file.
_BRAND_RE = re.compile(r"^[A-Za-z0-9_]+$")
# Files this module auto-discovers when `load_catalog()` is called with no
# explicit `brands` (i.e. `brands=None`, used by `resolve_techniques()`
# below): every `config/design_techniques_<brand>.json`, `<brand>` captured
# so it can be threaded through as `default_brand` exactly like an
# explicitly-requested brand would be.
_BRAND_FILE_RE = re.compile(r"^design_techniques_([A-Za-z0-9_]+)\.json$")

_REQUIRED_TECHNIQUE_FIELDS = {
    "group",
    "label",
    "description",
    "keywords",
    "prompt_instruction",
    "tags",
    "occasion",
    "visual_density",
    "preconditions",
    "avoid_when",
    "incompatible_with",
    "works_well_with",
    "selection_notes",
}


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item.strip())


def _parse_technique_entries(
    raw_catalog: dict[str, Any],
    *,
    default_brand: str,
    skip_groups_silently: frozenset[str] = frozenset(),
) -> tuple[dict[str, TechniqueDefinition], list[str], set[str]]:
    """Shared parsing logic for one `design_techniques` mapping -- called once
    per brand file by `load_catalog()`. `skip_groups_silently` lets a brand
    file's not-yet-supported "text" group be dropped without being treated
    as a validation error (see `_UNSUPPORTED_GROUPS`) -- the 3rd
    return value is the set of keys dropped that way, which `load_catalog()`
    still treats as "known" (but non-selectable) so that some OTHER
    technique's `incompatible_with`/`works_well_with` reference to one of
    them isn't wrongly flagged as an unknown-reference validation error."""
    definitions: dict[str, TechniqueDefinition] = {}
    validation_errors: list[str] = []
    skipped_keys: set[str] = set()
    for key, entry in raw_catalog.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            logger.warning("Skipping malformed technique entry %r", key)
            validation_errors.append(f"malformed technique entry {key!r}")
            continue
        group = str(entry.get("group") or "")
        if group in skip_groups_silently:
            skipped_keys.add(key)
            continue
        missing_fields = sorted(_REQUIRED_TECHNIQUE_FIELDS.difference(entry))
        if missing_fields:
            validation_errors.append(f"technique {key!r} is missing required metadata: {missing_fields}")
        if group not in _GROUPS:
            logger.warning("Skipping technique %r with unsupported group=%r", key, group)
            validation_errors.append(f"technique {key!r} has unsupported group {group!r}")
            continue
        if not str(entry.get("description") or "").strip():
            validation_errors.append(f"technique {key!r} has an empty description")
        if not str(entry.get("prompt_instruction") or "").strip():
            validation_errors.append(f"technique {key!r} has an empty prompt_instruction")
        visual_density = str(entry.get("visual_density") or "medium")
        if visual_density not in {"low", "medium", "high"}:
            validation_errors.append(f"technique {key!r} has invalid visual_density {visual_density!r}")
        definitions[key] = TechniqueDefinition(
            key=key,
            group=group,
            label=str(entry.get("label") or key),
            description=str(entry.get("description") or ""),
            keywords=_as_tuple(entry.get("keywords")),
            prompt_instruction=str(entry.get("prompt_instruction") or ""),
            tags=_as_tuple(entry.get("tags")),
            occasion=_as_tuple(entry.get("occasion")) or ("generic",),
            visual_density=visual_density,
            preconditions=_as_tuple(entry.get("preconditions")),
            avoid_when=_as_tuple(entry.get("avoid_when")),
            incompatible_with=_as_tuple(entry.get("incompatible_with")),
            works_well_with=_as_tuple(entry.get("works_well_with")),
            selection_notes=str(entry.get("selection_notes") or ""),
            aliases=_as_tuple(entry.get("aliases")),
            brand=str(entry.get("brand") or default_brand),
        )
    if skipped_keys:
        # Second pass, now that `skipped_keys` is fully known regardless of
        # dict iteration order: drop any `incompatible_with`/`works_well_with`
        # reference that points at one of the silently-skipped (unsupported
        # group, e.g. "text") keys. Those keys never become a real
        # TechniqueDefinition, so a reference to one is not actionable for
        # the operational selector -- stripping it here (rather than leaving
        # a dangling reference) keeps every remaining reference resolvable to
        # a real, selectable technique, which `test_design_techniques.py`'s
        # `test_schema_v2_catalog_is_complete_and_referentially_valid` relies
        # on (`set(item.incompatible_with) <= keys`, same for works_well_with).
        for key, definition in list(definitions.items()):
            new_incompatible = tuple(k for k in definition.incompatible_with if k not in skipped_keys)
            new_works_well = tuple(k for k in definition.works_well_with if k not in skipped_keys)
            if new_incompatible != definition.incompatible_with or new_works_well != definition.works_well_with:
                definitions[key] = dataclasses.replace(
                    definition, incompatible_with=new_incompatible, works_well_with=new_works_well
                )
    return definitions, validation_errors, skipped_keys


def _sanitize_brand(brand: str) -> str | None:
    """Returns `brand` unchanged if it's safe to interpolate into
    `design_techniques_<brand>.json` (see `_BRAND_RE`), else `None`. `brand`
    is untrusted -- it comes from a request.json's own
    `design_options.technique_selection.observe_brand` list -- so this must
    reject anything that isn't a plain alphanumeric/underscore token
    (blocks path traversal like `"../secrets"` or an empty string) rather
    than trying to clean it up."""
    candidate = str(brand).strip()
    return candidate if _BRAND_RE.match(candidate) else None


def _discover_all_brand_files(config_dir: Path) -> list[tuple[str, Path]]:
    """`brands=None` fallback for `load_catalog()` (used by
    `resolve_techniques()`, which has no brand-scoping concept): every
    `config/design_techniques_<brand>.json` file present, sorted by brand
    code for a deterministic merge order. `design_techniques-bk0609.json`
    (a hyphen-separated backup, not underscore) and any other file that
    doesn't match the `design_techniques_<brand>.json` pattern are not
    picked up here -- intentional, same as before this file was ever a
    glob target."""
    found: list[tuple[str, Path]] = []
    if not config_dir.is_dir():
        return found
    for path in sorted(config_dir.glob("design_techniques_*.json")):
        match = _BRAND_FILE_RE.match(path.name)
        if match:
            found.append((match.group(1), path))
    return found


def _all_discoverable_technique_keys(config_dir: Path) -> set[str]:
    """Every technique key defined in ANY `config/design_techniques_<brand>.
    json` file on disk, regardless of which brand(s) the current
    `load_catalog()` call actually loads/exposes.

    2026-09-14 (self-discovered while verifying the per-brand-file split):
    a technique's `incompatible_with`/`works_well_with` list is sometimes a
    deliberate CROSS-BRAND reference -- e.g. `design_techniques_general.
    json`'s `numeric_ribbon_badge` names `design_techniques_acb.json`'s
    `giant_oversized_stat_graphic` as incompatible, and that pair is
    symmetric (acb's `giant_oversized_stat_graphic` names
    `numeric_ribbon_badge` right back). That's legitimate authored content,
    not a typo -- but a request with `observe_brand: ["general"]` (the
    default) only ever loads `design_techniques_general.json`, so without
    this wider universe the reference would look "unknown" and fail
    `load_catalog()`'s referential-integrity check, which
    `select_techniques_for_versions()` treats as fatal (empty safe fallback
    for the WHOLE batch, see `app/core/technique_selector.py`) -- i.e. every
    ordinary general-only request would be permanently broken by a
    perfectly valid cross-brand annotation it can't see. This function is a
    cheap, keys-only pre-scan of every design_techniques_<brand>.json file
    present (loaded brands or not) purely so `load_catalog()` can tell a
    genuine unknown-reference typo (key not defined ANYWHERE) apart from a
    valid reference to a technique that simply belongs to a brand this call
    didn't happen to load."""
    keys: set[str] = set()
    for _brand, path in _discover_all_brand_files(config_dir):
        try:
            document = json.loads(path.read_bytes().decode("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        raw_catalog = document.get("design_techniques") if isinstance(document, dict) else None
        if isinstance(raw_catalog, dict):
            keys.update(k for k in raw_catalog if isinstance(k, str))
    return keys


def _order_general_first(brand_files: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """`"general"` (when present in `brand_files`) is always processed
    FIRST -- it plays the role the old always-present `design_techniques.
    json` used to play: on a duplicate technique key across brand files,
    whichever file is processed first wins (see `load_catalog()`), and the
    FIRST file processed also supplies `schema_version`/`selection_policy`
    for the merged catalog. Every other brand keeps its original relative
    order (the caller's requested order, or alphabetical from
    `_discover_all_brand_files()`)."""
    general = [item for item in brand_files if item[0] == "general"]
    others = [item for item in brand_files if item[0] != "general"]
    return general + others


def load_catalog(brands: list[str] | None = None) -> TechniqueCatalog:
    """Load the complete schema-v2 catalog used by the auto selector, merged
    from one file PER BRAND.

    2026-09-14 (user request): replaces the old fixed pair of
    `config/design_techniques.json` (general only) +
    `config/design_techniques_ref.json` (every other brand merged together)
    with one file per brand, `config/design_techniques_<brand>.json`.

    `brands`: the exact set of brand codes to load (normally the union of
    every version's `design_options.technique_selection.observe_brand` in
    the current batch -- see `app/core/technique_selector.py`, which is the
    only caller that passes this explicitly). Each is sanitized
    (`_sanitize_brand()`) before being turned into a file path; an invalid
    brand code is skipped with a warning rather than raising. `brands=None`
    (the default; used by `resolve_techniques()` below, which resolves
    `design_options.techniques` -- a plain key list with no brand-scoping
    concept) instead auto-discovers and merges EVERY
    `design_techniques_<brand>.json` file present under `config/`
    (`_discover_all_brand_files()`), matching this module's old behavior of
    a single, always-fully-merged catalog for that call path.

    A brand whose file is missing, unparsable, or has no object-valued
    `design_techniques` mapping is skipped with a warning -- optional and
    never fatal on its own, same convention as the old ref-file handling.
    The catalog as a whole is empty (not an error) only when NOT ONE
    requested/discovered brand file could be loaded, so auto selection can
    safely fall back to no techniques instead of aborting image generation.
    """
    settings = get_settings()
    if brands is None:
        brand_files = _discover_all_brand_files(settings.design_techniques_dir)
    else:
        brand_files = []
        seen: set[str] = set()
        for raw_brand in brands:
            brand = _sanitize_brand(raw_brand)
            if brand is None:
                logger.warning(
                    "Ignoring invalid observe_brand entry %r (must be a non-empty alphanumeric/underscore token)",
                    raw_brand,
                )
                continue
            if brand in seen:
                continue
            seen.add(brand)
            brand_files.append((brand, settings.design_techniques_file_for_brand(brand)))
    brand_files = _order_general_first(brand_files)

    definitions: dict[str, TechniqueDefinition] = {}
    validation_errors: list[str] = []
    known_skipped_keys: set[str] = set()
    schema_version = ""
    policy: dict[str, Any] | None = None
    group_definitions: dict[str, str] = {}
    combined_bytes = b""
    any_file_loaded = False

    for brand, path in brand_files:
        if not path.exists():
            logger.warning(
                "config/design_techniques_%s.json not found at %s -- no techniques for brand=%r this run.",
                brand, path, brand,
            )
            continue
        try:
            raw_bytes = path.read_bytes()
            document = json.loads(raw_bytes.decode("utf-8"))
        except Exception:  # noqa: BLE001
            logger.exception(
                "Could not parse config/design_techniques_%s.json at %s -- no techniques for brand=%r this run.",
                brand, path, brand,
            )
            continue
        raw_catalog = document.get("design_techniques") if isinstance(document, dict) else None
        if not isinstance(raw_catalog, dict):
            logger.warning(
                "%s has no object-valued design_techniques catalog -- skipping it (brand=%r)", path, brand
            )
            continue

        any_file_loaded = True
        combined_bytes += raw_bytes
        file_definitions, file_errors, file_skipped_keys = _parse_technique_entries(
            raw_catalog, default_brand=brand, skip_groups_silently=frozenset(_UNSUPPORTED_GROUPS)
        )
        for key, definition in file_definitions.items():
            if key in definitions:
                logger.warning(
                    "Skipping technique %r from config/design_techniques_%s.json: key already defined with brand=%r",
                    key, brand, definitions[key].brand,
                )
                validation_errors.append(
                    f"duplicate technique key {key!r} in design_techniques_{brand}.json ignored "
                    f"(already defined with brand {definitions[key].brand!r})"
                )
                continue
            definitions[key] = definition
        validation_errors.extend(file_errors)
        known_skipped_keys |= file_skipped_keys

        # First successfully-loaded file (with "general" pulled to the front
        # by `_order_general_first()` when present) supplies schema_version/
        # selection_policy for the merged catalog -- same role the old
        # always-present `design_techniques.json` played; every other file's
        # own schema_version/selection_policy is not authoritative for the
        # merge (only its techniques + group_definitions contribute).
        if schema_version == "":
            schema_version = str(document.get("schema_version") or "")
        if policy is None and isinstance(document.get("selection_policy"), dict):
            policy = document.get("selection_policy")
        file_group_definitions = document.get("group_definitions")
        if isinstance(file_group_definitions, dict):
            for k, v in file_group_definitions.items():
                if k in _GROUPS:
                    group_definitions.setdefault(str(k), str(v))

    if not any_file_loaded:
        logger.warning("No design_techniques_<brand>.json file could be loaded for brands=%r", brands)
        return TechniqueCatalog()

    if not isinstance(policy, dict):
        policy = {"min_per_group": 0, "max_per_group": 3, "max_total_per_version": 6}
        validation_errors.append("selection_policy must be an object")
    try:
        min_per_group = int(policy.get("min_per_group", 0))
        max_per_group = int(policy.get("max_per_group", 3))
        max_total = int(policy.get("max_total_per_version", 6))
        if min_per_group != 0 or max_per_group < min_per_group or max_per_group > 3 or max_total < 0:
            validation_errors.append(
                "selection_policy must use min_per_group=0, max_per_group<=3, and a non-negative max_total_per_version"
            )
    except (TypeError, ValueError):
        validation_errors.append("selection_policy numeric limits must be integers")

    # `known_reference_keys` additionally includes every brand file's
    # SILENTLY skipped "text"-group keys (`known_skipped_keys`) -- those
    # techniques aren't selectable (no TechniqueDefinition exists for them),
    # but some OTHER technique's `incompatible_with`/`works_well_with` may
    # still legitimately name one (e.g. a background technique that "works
    # well with" a text-display-style technique). Since dropping that whole
    # group is an expected, documented limitation (not a config bug), a
    # reference to one of its keys must not be flagged as an
    # unknown-reference error -- `definitions.get(...)` naturally returns
    # None for these below, so the symmetry check right after still safely
    # skips them.
    #
    # It ALSO includes every key discoverable in ANY design_techniques_
    # <brand>.json file on disk (`_all_discoverable_technique_keys()`), not
    # just the brand(s) this call loaded -- see that function's docstring
    # for why a scoped load (e.g. `observe_brand: ["general"]`, the default)
    # must not treat a deliberate, real cross-brand reference as an unknown
    # one just because the referenced brand's file wasn't requested this run.
    known_keys = set(definitions)
    known_reference_keys = known_keys | known_skipped_keys | _all_discoverable_technique_keys(
        settings.design_techniques_dir
    )
    for key, definition in definitions.items():
        for relation_name, references in (
            ("incompatible_with", definition.incompatible_with),
            ("works_well_with", definition.works_well_with),
        ):
            unknown = sorted(set(references).difference(known_reference_keys))
            if unknown:
                validation_errors.append(f"technique {key!r} has unknown {relation_name} references: {unknown}")
        for conflict in definition.incompatible_with:
            other = definitions.get(conflict)
            if other is not None and key not in other.incompatible_with:
                validation_errors.append(f"conflict {key!r} <-> {conflict!r} is not symmetric")
    deduplicated_errors = tuple(sorted(set(validation_errors)))
    if deduplicated_errors:
        logger.error("Technique catalog validation failed: %s", "; ".join(deduplicated_errors))
    return TechniqueCatalog(
        schema_version=schema_version,
        selection_policy=policy,
        group_definitions=group_definitions,
        techniques=definitions,
        sha256=hashlib.sha256(combined_bytes).hexdigest(),
        validation_errors=deduplicated_errors,
    )


def _load_raw() -> dict:
    """Returns the merged `design_techniques` mapping (`{key: {...}}`) across
    every `config/design_techniques_<brand>.json` file present (via
    `load_catalog()` with no `brands` filter -- `design_options.techniques`
    is a plain key list with no brand-scoping concept, see this module's
    docstring). Never raises -- a missing/broken file just means
    `design_options.techniques` resolves to nothing (logged), same soft-fail
    convention as `app/core/model_usage.py::load_model_usage()`."""
    return {
        key: {
            "label": item.label,
            "description": item.description,
            "prompt_instruction": item.prompt_instruction,
        }
        for key, item in load_catalog().techniques.items()
    }


def resolve_techniques(technique_keys: list[str]) -> tuple[list[ResolvedTechnique], list[str]]:
    """Returns `(resolved, unknown_keys)`. A key in `technique_keys` that has
    no matching entry in any `config/design_techniques_<brand>.json` file is
    skipped (not fatal -- REQUEST.md: never let an optional enhancement
    crash the pipeline) but reported back in `unknown_keys` so the caller
    can log it (helps catch a typo like `"silhouette_watermark"` vs. the
    config's own `"silhoutte_watermark"` spelling)."""
    if not technique_keys:
        return [], []
    catalog = _load_raw()
    resolved: list[ResolvedTechnique] = []
    unknown: list[str] = []
    for key in technique_keys:
        entry = catalog.get(key)
        if entry is None:
            unknown.append(key)
            continue
        resolved.append(
            ResolvedTechnique(
                key=key,
                label=str(entry.get("label") or key),
                description=str(entry.get("description") or ""),
                prompt_instruction=str(entry.get("prompt_instruction") or ""),
            )
        )
    if unknown:
        logger.warning(
            "design_options.techniques references unknown technique key(s) %s -- not found in "
            "any config/design_techniques_<brand>.json file, ignored.",
            unknown,
        )
    return resolved, unknown
