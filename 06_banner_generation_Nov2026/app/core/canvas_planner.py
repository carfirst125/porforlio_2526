"""Plan generation geometry from the requested FINAL output size.

The image model should work in a native aspect ratio that is as close as
possible to ``output.size``.  The request's hand-written
``generation.canvas`` remains useful as input metadata, but it is not a safe
source of truth for models with a fixed aspect-ratio list: a 3:2 canvas for an
ultra-wide final banner forces the model to compose inside a thin strip and
throws most generated pixels away.

This module keeps the policy size-independent:

* choose the nearest supported ratio using symmetric log-ratio distance;
* build the smallest native-ratio canvas that fully contains ``output.size``;
* center the final output rectangle in that canvas;
* crop only the small ratio mismatch after generation.

No output size (including 1035x420) is special-cased.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.models.schemas import (
    CanvasSpec,
    CreativeRequest,
    CropAreaSpec,
    CropSpec,
    GenerationSpec,
    SafeAreaSpec,
)

RatioBucket = tuple[str, int, int]

# Gemini 3.1 Flash Image's documented native ratios.  Keep this single list
# shared with ``app/providers/gemini_image.py`` so provider selection and
# pipeline geometry cannot drift apart again.
GEMINI_31_FLASH_IMAGE_RATIOS: tuple[RatioBucket, ...] = (
    ("1:1", 1, 1),
    ("1:4", 1, 4),
    ("1:8", 1, 8),
    ("2:3", 2, 3),
    ("3:2", 3, 2),
    ("3:4", 3, 4),
    ("4:1", 4, 1),
    ("4:3", 4, 3),
    ("4:5", 4, 5),
    ("5:4", 5, 4),
    ("8:1", 8, 1),
    ("9:16", 9, 16),
    ("16:9", 16, 9),
    ("21:9", 21, 9),
)

_MODEL_RATIO_BUCKETS: dict[str, tuple[RatioBucket, ...]] = {
    "gemini-3.1-flash-image": GEMINI_31_FLASH_IMAGE_RATIOS,
}

CONTENT_SAFE_INSET_FRACTION = 0.05


@dataclass(frozen=True)
class CanvasPlan:
    output_wh: tuple[int, int]
    canvas_wh: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    ratio_label: str | None
    crop_loss: float

    @property
    def has_crop(self) -> bool:
        return self.canvas_wh != self.output_wh


def content_safe_box(
    crop_box: tuple[int, int, int, int],
    inset_fraction: float = CONTENT_SAFE_INSET_FRACTION,
) -> tuple[int, int, int, int]:
    """Return a proportional inner box for important advertising content.

    ``crop_box`` is still the exact delivered output. The inner box adds a
    small, size-independent safety buffer so minor model placement errors do
    not put text/products on the post-crop edge.
    """
    x, y, width, height = crop_box
    if not 0 <= inset_fraction < 0.5:
        raise ValueError(f"inset_fraction must be in [0, 0.5), got {inset_fraction}")
    inset_x = min(max(2, round(width * inset_fraction)), max(0, (width - 1) // 2))
    inset_y = min(max(2, round(height * inset_fraction)), max(0, (height - 1) // 2))
    return x + inset_x, y + inset_y, width - 2 * inset_x, height - 2 * inset_y


def closest_supported_ratio(
    width: int,
    height: int,
    buckets: tuple[RatioBucket, ...],
) -> RatioBucket:
    """Return the nearest ratio by orientation-symmetric log distance."""
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    target = width / height
    return min(buckets, key=lambda item: abs(math.log(target) - math.log(item[1] / item[2])))


def plan_canvas_for_output(
    output_wh: tuple[int, int],
    buckets: tuple[RatioBucket, ...],
) -> CanvasPlan:
    """Create the smallest supported-ratio canvas containing ``output_wh``."""
    output_w, output_h = output_wh
    label, ratio_w, ratio_h = closest_supported_ratio(output_w, output_h, buckets)

    # Integer cross-products avoid float equality errors for exact matches.
    target_cross = output_w * ratio_h
    native_cross = output_h * ratio_w
    if target_cross == native_cross:
        canvas_w, canvas_h = output_w, output_h
    elif target_cross < native_cross:  # native ratio is wider
        canvas_h = output_h
        canvas_w = math.ceil(output_h * ratio_w / ratio_h)
    else:  # native ratio is taller/narrower
        canvas_w = output_w
        canvas_h = math.ceil(output_w * ratio_h / ratio_w)

    crop_x = (canvas_w - output_w) // 2
    crop_y = (canvas_h - output_h) // 2
    crop_box = (crop_x, crop_y, output_w, output_h)
    crop_loss = 1.0 - (output_w * output_h) / (canvas_w * canvas_h)
    return CanvasPlan(output_wh, (canvas_w, canvas_h), crop_box, label, crop_loss)


def ratio_buckets_for_models(models: list[str]) -> tuple[RatioBucket, ...] | None:
    """Return one shared ratio table, or ``None`` for mixed/unknown models.

    The graph builds one shared prompt for all models.  Geometry is therefore
    auto-planned only when every selected model has the same known ratio
    capabilities; otherwise the request's declared geometry is preserved.
    """
    if not models:
        return None
    tables = [_MODEL_RATIO_BUCKETS.get(model) for model in models]
    if any(table is None for table in tables):
        return None
    first = tables[0]
    return first if all(table == first for table in tables[1:]) else None


def direct_canvas_plan(output_wh: tuple[int, int]) -> CanvasPlan:
    """No-op geometry for providers without a known shared ratio table."""
    width, height = output_wh
    return CanvasPlan(output_wh, output_wh, (0, 0, width, height), None, 0.0)


def plan_canvas_for_model(output_wh: tuple[int, int], model: str) -> CanvasPlan:
    buckets = _MODEL_RATIO_BUCKETS.get(model)
    return plan_canvas_for_output(output_wh, buckets) if buckets else direct_canvas_plan(output_wh)


def plan_request_geometry(req: CreativeRequest, models: list[str]) -> tuple[CreativeRequest, CanvasPlan | None]:
    """Return an in-memory request copy with output-driven geometry.

    The JSON file is never modified.  ``output.size`` and all creative data
    remain unchanged; only the model canvas, safe area, crop box, and stale
    free-text geometry instruction are normalized for the current run.
    """
    buckets = ratio_buckets_for_models(models)
    if buckets is None:
        return req, None

    plan = plan_canvas_for_output(req.final_output_wh(), buckets)
    canvas_w, canvas_h = plan.canvas_wh
    crop_x, crop_y, output_w, output_h = plan.crop_box

    if plan.has_crop:
        instruction = (
            f"Generate on the full {canvas_w}x{canvas_h} canvas ({plan.ratio_label} native ratio). "
            f"Keep every important element fully inside the centered {output_w}x{output_h} safe area at "
            f"({crop_x}, {crop_y}); only background decoration may extend outside because that small outer "
            "strip will be cropped away."
        )
    else:
        instruction = (
            f"Generate directly at the full {output_w}x{output_h} canvas ({plan.ratio_label} native ratio). "
            "This is the final output; keep every important element inside the canvas with safe padding."
        )

    generation = GenerationSpec(
        canvas=CanvasSpec(width=canvas_w, height=canvas_h),
        safe_area=SafeAreaSpec(
            width=output_w,
            height=output_h,
            position="center",
            x=crop_x,
            y=crop_y,
        ),
        instruction=instruction,
    )
    output = req.output.model_copy(
        update={
            "crop": CropSpec(
                source="generation.canvas",
                area=CropAreaSpec(x=crop_x, y=crop_y, width=output_w, height=output_h),
            )
        }
    )
    return req.model_copy(update={"generation": generation, "output": output}), plan
