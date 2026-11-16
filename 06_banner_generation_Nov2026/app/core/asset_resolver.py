"""Resolve `buckets.{theme,context,product,brand_text}` + `reference` values from
`request.json` into real files under `ASSET_LIBRARY_DIR` (`.env`, defaults to
`version_1/lib/`).

Values in `request.json` are relative to `ASSET_LIBRARY_DIR`, e.g.
`"Bucket_1/theme_promotion.jpg"` -> `<ASSET_LIBRARY_DIR>/Bucket_1/theme_promotion.jpg`.

2026-09-04 (user request): a `theme`/`context`/`product` bucket entry may now
ALSO be a free-text description instead of an image path, e.g.
`"theme": "theme_supermarket: không gian xanh lá gradient..."` -- the user
wants to describe the desired background/character/product in words rather
than attach a reference image. `split_bucket_values()` below classifies each
raw bucket string as an image-path-looking value (has a known image
extension, e.g. `.png`/`.jpg`) vs. a free-text description (no image
extension) -- purely syntactic, no filesystem access, so it stays consistent
with `app/core/prompt_builder.py`'s existing "declared vs. resolved" split
(a bucket entry counts as a locked/attached ASSET as soon as it is declared
with an image-looking value, whether or not the file actually resolves on
disk -- see that module's `has_theme_assets`/`has_context_assets`/
`has_product_assets`). Only image-looking values are ever passed to
`_resolve_one()` / attached to the image model; text descriptions are never
treated as "file not found" errors and are instead folded into the prompt as
creative direction (`prompt_builder.py`'s `_theme_instruction()` /
`_person_instruction()` / `_product_instruction()`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.logging_conf import get_logger
from app.models.schemas import CreativeRequest, ResolvedAsset
from app.settings import get_settings

logger = get_logger("asset_resolver")

# Recognized image file extensions -- a bucket value is treated as an
# image-path reference only when it ends in one of these; anything else
# (no extension, or a colon-delimited description like
# "theme_supermarket: ...") is treated as a free-text description instead.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def looks_like_image_path(value: str) -> bool:
    """Purely syntactic check -- never touches the filesystem, so a typo'd
    or missing file still counts as "meant to be an image" (existing
    behavior: `_resolve_one()` below reports that as a real "file not
    found" error) instead of silently being reinterpreted as descriptive
    text."""
    return Path(value.strip()).suffix.lower() in _IMAGE_EXTENSIONS


def split_bucket_values(values: list[str]) -> tuple[list[str], list[str]]:
    """Split one bucket's raw string list into `(image_like_values,
    text_descriptions)`, preserving each sublist's original order."""
    images: list[str] = []
    texts: list[str] = []
    for value in values:
        (images if looks_like_image_path(value) else texts).append(value)
    return images, texts


@dataclass
class ResolvedBucketAssets:
    theme: list[Path]
    context: list[Path]
    product: list[Path]
    brand_text: list[Path]
    reference: Path | None
    guideline_extra: Path | None
    all_resolved: list[ResolvedAsset]
    # Free-text descriptions found in each bucket (no image attached for
    # these -- see module docstring). Empty by default so every other call
    # site that only cares about the resolved image paths is unaffected.
    theme_text: list[str] = field(default_factory=list)
    context_text: list[str] = field(default_factory=list)
    product_text: list[str] = field(default_factory=list)
    brand_text_text: list[str] = field(default_factory=list)


def _resolve_one(value: str) -> tuple[Path | None, str | None]:
    base = get_settings().asset_library_dir
    p = (base / value).resolve()
    if p.exists() and p.is_file():
        return p, None
    return None, f"file not found under ASSET_LIBRARY_DIR ({base}): {value}"


def resolve_assets(req: CreativeRequest) -> ResolvedBucketAssets:
    all_resolved: list[ResolvedAsset] = []
    labels = ("theme", "context", "product", "brand_text")
    resolved_paths: dict[str, list[Path]] = {label: [] for label in labels}
    resolved_texts: dict[str, list[str]] = {label: [] for label in labels}

    for label in labels:
        values = getattr(req.buckets, label)
        count = len(values)
        for i, value in enumerate(values, start=1):
            if not looks_like_image_path(value):
                # Free-text description -- not a file to resolve, and not
                # an error: `ok=True` here means "usable as-is", it's just
                # not an image attachment.
                resolved_texts[label].append(value)
                all_resolved.append(
                    ResolvedAsset(label=label, value=value, index=i, count=count, ok=True, path="", error=None, kind="text")
                )
                continue
            path, error = _resolve_one(value)
            ok = path is not None
            if ok:
                resolved_paths[label].append(path)
            else:
                logger.warning("Could not resolve %s bucket asset %r: %s", label, value, error)
            all_resolved.append(
                ResolvedAsset(label=label, value=value, index=i, count=count, ok=ok, path=str(path) if path else "", error=error, kind="image")
            )

    reference_path: Path | None = None
    if req.reference:
        path, error = _resolve_one(req.reference)
        if path:
            reference_path = path
        else:
            logger.warning("Could not resolve reference asset %r: %s", req.reference, error)
        all_resolved.append(
            ResolvedAsset(label="reference", value=req.reference, ok=path is not None, path=str(path) if path else "", error=error)
        )

    guideline_extra: Path | None = None
    if req.guidelines:
        path, error = _resolve_one(req.guidelines)
        if path:
            guideline_extra = path
        else:
            logger.warning(
                "request.json 'guidelines' field %r does not resolve under ASSET_LIBRARY_DIR (%s) -- "
                "continuing without it, format_type-driven guidelines (REQUEST.md [2]) still apply.",
                req.guidelines,
                error,
            )

    return ResolvedBucketAssets(
        theme=resolved_paths["theme"],
        context=resolved_paths["context"],
        product=resolved_paths["product"],
        brand_text=resolved_paths["brand_text"],
        reference=reference_path,
        guideline_extra=guideline_extra,
        all_resolved=all_resolved,
        theme_text=resolved_texts["theme"],
        context_text=resolved_texts["context"],
        product_text=resolved_texts["product"],
        brand_text_text=resolved_texts["brand_text"],
    )
