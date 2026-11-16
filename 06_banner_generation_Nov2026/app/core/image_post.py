"""Post-process a raw image returned by an Image GenAI model (REQUEST.md
[1] step [6]): crop the `generation.safe_area` / `output.crop.area` box out
of the generated canvas, then hard-match the result to `output.size`.

2026-08-20 correction: this used to crop `design_options.protect_area`
(centered) out of the canvas -- that field no longer exists in the real
request.json schema (see `app/models/schemas.py` module docstring). The
crop box is now an explicit `(x, y, width, height)` from
`CreativeRequest.crop_box()`, which is NOT necessarily centered (REQUEST.md's
`generation.safe_area` carries its own `x`/`y` offset).
"""
from __future__ import annotations

import io

from PIL import Image, ImageChops, ImageDraw

from app.logging_conf import get_logger

logger = get_logger("image_post")

SUPERSAMPLE_FACTOR = 4


def build_rounded_mask(size: tuple[int, int], radius_px: int) -> Image.Image:
    """Tạo mask alpha L có hình chữ nhật bo tròn 4 góc bằng supersampling + LANCZOS."""
    w, h = size
    scale = SUPERSAMPLE_FACTOR
    big_w, big_h = w * scale, h * scale
    r = max(0, radius_px * scale)

    mask = Image.new("L", (big_w, big_h), 0)
    draw = ImageDraw.Draw(mask)

    draw.rectangle([r, 0, big_w - r, big_h], fill=255)
    draw.rectangle([0, r, big_w, big_h - r], fill=255)

    draw.ellipse([0, 0, 2 * r, 2 * r], fill=255)
    draw.ellipse([big_w - 2 * r, 0, big_w, 2 * r], fill=255)
    draw.ellipse([0, big_h - 2 * r, 2 * r, big_h], fill=255)
    draw.ellipse([big_w - 2 * r, big_h - 2 * r, big_w, big_h], fill=255)

    return mask.resize((w, h), Image.LANCZOS)


def apply_rounded_corners(img: Image.Image, radius_px: int) -> Image.Image:
    """Trả về bản sao RGBA với 4 góc được bo tròn trong suốt."""
    rgba = img.convert("RGBA")
    mask = build_rounded_mask(rgba.size, radius_px)
    orig_alpha = rgba.getchannel("A")
    combined_alpha = ImageChops.darker(mask, orig_alpha)
    rgba.putalpha(combined_alpha)
    return rgba


def crop_canvas_to_output(
    image_bytes: bytes,
    canvas_wh: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    final_wh: tuple[int, int],
    corner_radius: int | None = None,
) -> bytes:
    """`canvas_wh` is the size the model was asked to generate
    (`generation.canvas`). `crop_box` is `(x, y, width, height)` in canvas
    pixel coordinates (`CreativeRequest.crop_box()` -- from `output.crop.area`
    if present, else `generation.safe_area`). `final_wh` is `output.size`,
    the real pixel size that must end up on disk.

    If the model returned an image at a different size than `canvas_wh`, it
    is resized to the canvas first. The crop box is clamped to the canvas
    bounds (defensive -- REQUEST.md: never crash on a malformed/hand-edited
    request.json). If the cropped region doesn't come out at EXACTLY
    `final_wh` (it always should when `crop_box`'s width/height match
    `output.size`, but a hand-edited request.json is never fully trusted),
    it is resized to match `final_wh` exactly as a last step.

    For Popup banners (720x1280), corner radius defaults to 60px unless overridden."""
    canvas_w, canvas_h = canvas_wh
    box_x, box_y, box_w, box_h = crop_box
    final_w, final_h = final_wh

    with Image.open(io.BytesIO(image_bytes)) as img:
        img.load()
        img = img.convert("RGBA")

        if img.size != (canvas_w, canvas_h):
            logger.warning(
                "Model returned image at %s, expected canvas %s -- resizing to the requested canvas before crop.",
                img.size,
                (canvas_w, canvas_h),
            )
            img = img.resize((canvas_w, canvas_h), Image.LANCZOS)

        left = max(0, min(box_x, canvas_w))
        top = max(0, min(box_y, canvas_h))
        right = max(left, min(box_x + box_w, canvas_w))
        bottom = max(top, min(box_y + box_h, canvas_h))

        if right <= left or bottom <= top:
            logger.warning(
                "crop box %s collapsed to an empty/degenerate region on canvas %s -- falling back to the full canvas.",
                crop_box,
                (canvas_w, canvas_h),
            )
            cropped = img
        elif (left, top, right, bottom) == (0, 0, canvas_w, canvas_h):
            cropped = img
        else:
            cropped = img.crop((left, top, right, bottom))
            logger.info("Cropped box=%s from canvas=%s", (left, top, right, bottom), (canvas_w, canvas_h))

        if cropped.size != (final_w, final_h):
            logger.warning(
                "Cropped region %s does not match declared output.size %s -- resizing to match exactly.",
                cropped.size,
                (final_w, final_h),
            )
            cropped = cropped.resize((final_w, final_h), Image.LANCZOS)

        # Tu dong ap dung bo goc cho popup 720x1280
        effective_radius = corner_radius
        if effective_radius is None and (final_w, final_h) == (720, 1280):
            effective_radius = 60

        if effective_radius and effective_radius > 0:
            logger.info("Applying corner radius %dpx to output %s", effective_radius, (final_w, final_h))
            cropped = apply_rounded_corners(cropped, effective_radius)

        out = io.BytesIO()
        cropped.save(out, format="PNG", optimize=True)
        return out.getvalue()

