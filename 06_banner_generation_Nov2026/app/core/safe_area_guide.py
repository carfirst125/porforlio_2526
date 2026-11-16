"""Renders a visual SAFE AREA GUIDE image and attaches it to the Image
GenAI model call alongside the text instruction in `prompt_builder.py`'s
"CANVAS & SAFE AREA" section (REQUEST.md [3]: "phải đặt trong safe area").

2026-08-20 (user-reported failure): a real run declared `generation.canvas`
1536x1024 with `generation.safe_area` 1035x420 centered at (250, 302), the
text instruction spelled out those exact pixel coordinates, and gpt-image-2
STILL placed headline text past the safe-area edge -- it got cut off by the
crop. Bare pixel coordinates in a paragraph of text are notoriously
unreliable for image-generation models to follow precisely; they have no
robust way to translate "x=250, y=302" into an actual spatial position
without a corresponding image to look at. Attaching an actual same-size
picture that marks the boundary gives the model something to literally
match visually -- this module builds that picture.

2026-08-20 (2) (user follow-up, after looking at the first version of this
guide): a plain magenta rectangle communicates the IDEA of a boundary, but
an image-generation model reading a picture cannot reliably back out exact
pixel MEASUREMENTS just by looking at a box -- it has no ruler. So this
version draws the actual numbers directly ON the picture, next to
ruler-style dimension lines/arrows: the safe area's width and height along
its own top/left edges, and its x/y offset from the canvas origin as
dashed leader lines from the canvas corner -- the same way a technical
drawing/blueprint would annotate a shape, not just outline it.

2026-08-20 (3) (user follow-up, re-run still showed neither the new prompt
wording nor the attached-images list): root-caused to a stale-cache bug in
`write_prompt_log()` / `prompt_builder.py`'s `_cache_key()` (see those
files' own history notes) -- not a bug in this module, but this module's
OWN geometry-keyed cache had the identical bug class, so it was made
always-overwrite here too as a proactive fix.

2026-08-20 (4) (user request): moved this module's cache from a PER-REQUEST
location (`output/<request_name>/_resized_refs/_safe_area_guide_<geometry-
hash>.png`, always rebuilt) to a SHARED LIBRARY cache at
`Settings.safe_area_guide_dir` (`./lib/safe_area_guide` by default), 1 file
per DISTINCT (canvas_size, safe_area_size) combination, named
`safe_area_guide_<canvas_w>x<canvas_h>_<safe_w>x<safe_h>.png` (see
`guide_filename()`), reused across every request/run that needs that same
geometry instead of each request paying the render cost again. This
DELIBERATELY re-introduces the exists-then-reuse pattern removed in
2026-08-20 (3) above -- an informed, requested trade-off this time (see
`build_safe_area_guide_attachment()`'s own docstring for the staleness
caveat this re-creates).

2026-08-20 (5) (user follow-up, "image model vẫn chưa thực hiện theo" even
after (1)-(4)): while investigating, found that
`app/providers/azure_openai_image.py::_closest_supported_size()` maps
gpt-image-2's requested canvas to the closest of only 3 fixed sizes
((1024,1024)/(1024,1536)/(1536,1024)) before actually calling the API --
so whenever `generation.canvas` isn't already one of those 3 sizes, the
ABSOLUTE pixel numbers on this guide image (and in `prompt_builder.py`'s
text) can describe a canvas gpt-image-2 isn't really using. Added
percentage-of-width/height figures to the caption here (and to the prompt
sections in `prompt_builder.py`) as the authoritative, size-independent
description, alongside the pixel numbers kept as a secondary, human-
readable cross-check. This does NOT fix the size-quantization itself (that
would need a bigger change -- see `4-project-evaluation-results.md`'s
2026-08-20 (10) entry for the full writeup and the options considered);
it only makes the safe-area INSTRUCTION robust to it.

2026-08-20 (6) (user follow-up, with a real gemini-3.1-flash output
attached): the model's composition matched the guide's magenta BOX almost
exactly -- the mask shape itself is doing the real work, essentially like a
layout template the model traces. The measurement annotations added in (2)
(width/height rulers, x/y offset leader lines, corner-coordinate labels)
turned out to be unnecessary complexity: the box's own pixel geometry
already IS the exact canvas/safe-area size, so there's nothing left for a
number overlay to communicate that the shape doesn't already show. Removed
all of it -- `build_safe_area_guide_png()` now draws only the tint + the
solid border + 1 short instruction line in the caption bar, no numbers.
`_draw_arrowhead()` / `_draw_double_arrow()` / `_draw_dashed_double_arrow()`
/ `_draw_label()` / `_unit()` and their now-unused color constants were
removed as dead code along with the annotations that used them.

IMPORTANT -- cache invalidation after this change: this module's own
attachment cache (`build_safe_area_guide_attachment()`, see its docstring)
reuses an existing file at the expected filename WITHOUT checking whether
it still matches the current drawing code. Any `.png` already sitting in
`Settings.safe_area_guide_dir` from BEFORE this change still has the old
ruler/label annotations baked in and will keep being served as-is until
it's deleted by hand -- this update does not, and cannot, reach into a
user's already-populated cache directory. Clear
(or delete the specific stale files in) that directory once after
deploying this change.
"""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from app.core.image_prep import PreparedImage
from app.logging_conf import get_logger

logger = get_logger("safe_area_guide")

# Bright magenta: chosen because it collides with essentially no real OCB
# brand color or natural banner content (OCB's palette is green/gold/white/
# dark-gray -- see design_rules.md), so the guide's own marker lines are
# unambiguous and impossible to mistake for actual design elements to keep.
_BOX_COLOR = (255, 0, 220)          # safe-area border + outside tint
_GUIDE_LINE_WIDTH = 6
_OUTSIDE_TINT_ALPHA = 70  # translucent wash over the discard zone, out of 255
_CAPTION_BAR_HEIGHT = 34


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow < 10.1 -- load_default() doesn't take a size argument yet.
        return ImageFont.load_default()


def build_safe_area_guide_png(canvas_wh: tuple[int, int], crop_box: tuple[int, int, int, int]) -> bytes:
    """Returns PNG bytes, same pixel size as `canvas_wh`:

    - the region OUTSIDE `crop_box` (discarded after crop) washed with a
      translucent magenta tint, `crop_box` itself outlined in solid magenta;
    - a short caption bar restating the rule in plain words, with NO
      numeric measurements -- 2026-08-20 (6): a real run showed the model's
      composition tracking the magenta BOX SHAPE itself almost exactly, so
      the earlier width/height ruler lines, x/y offset leader lines, and
      corner-coordinate labels (2026-08-20 (2)) turned out to be
      unnecessary complexity layered on top of a mask that already encodes
      the exact geometry through its own pixel dimensions.
    """
    canvas_w, canvas_h = canvas_wh
    box_x, box_y, box_w, box_h = crop_box
    box_x = max(0, min(box_x, canvas_w))
    box_y = max(0, min(box_y, canvas_h))
    box_w = max(0, min(box_w, canvas_w - box_x))
    box_h = max(0, min(box_h, canvas_h - box_y))
    box_x2, box_y2 = box_x + box_w, box_y + box_h

    base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Tint the whole canvas, then punch a fully transparent hole over the
    # safe area so only the OUTSIDE (discarded-after-crop) zone stays tinted.
    draw.rectangle([0, 0, canvas_w, canvas_h], fill=_BOX_COLOR + (_OUTSIDE_TINT_ALPHA,))
    if box_w > 0 and box_h > 0:
        draw.rectangle([box_x, box_y, box_x2, box_y2], fill=(0, 0, 0, 0))
        draw.rectangle([box_x, box_y, box_x2 - 1, box_y2 - 1], outline=_BOX_COLOR + (255,), width=_GUIDE_LINE_WIDTH)

    guide = Image.alpha_composite(base, overlay)
    gdraw = ImageDraw.Draw(guide)

    caption = "SAFE AREA GUIDE -- keep ALL content strictly inside the magenta box. Tinted zone outside it is DISCARDED after crop."
    gdraw.rectangle([0, 0, canvas_w, _CAPTION_BAR_HEIGHT], fill=(0, 0, 0, 220))
    gdraw.text((8, 8), caption, fill=(255, 255, 255, 255), font=_font(16))

    out = io.BytesIO()
    guide.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


def guide_filename(canvas_wh: tuple[int, int], safe_area_wh: tuple[int, int]) -> str:
    """`safe_area_guide_<canvas_w>x<canvas_h>_<safe_w>x<safe_h>.png` -- e.g.
    `safe_area_guide_1536x1024_1035x420.png` (2026-08-20 (4), user's exact
    naming convention).

    NOTE (flagged, not silently swallowed): this name is keyed on SIZES
    only, not the safe area's x/y OFFSET. In every real request.json seen
    in this project so far, `position="center"` -- so a given (canvas_size,
    safe_area_size) pair always implies the SAME x/y (there's only one way
    to center a box of a given size on a given canvas), and the name is
    unambiguous. If a FUTURE request.json ever declares an off-center
    `generation.safe_area` (an explicit non-center x/y) with the SAME sizes
    as an existing centered one, this naming scheme would incorrectly reuse
    the wrong (centered) cached guide. None of the real request.json files
    in this project do that today -- if you add one that does, widen this
    filename to also encode x/y, or ask before relying on this cache for
    that request."""
    canvas_w, canvas_h = canvas_wh
    sa_w, sa_h = safe_area_wh
    return f"safe_area_guide_{canvas_w}x{canvas_h}_{sa_w}x{sa_h}.png"


def build_safe_area_guide_attachment(canvas_wh: tuple[int, int], crop_box: tuple[int, int, int, int]) -> PreparedImage | None:
    """Looks up (or builds, if missing) the guide PNG for this
    (canvas_size, safe_area_size) in the SHARED library cache
    `Settings.safe_area_guide_dir` (`./lib/safe_area_guide` by default,
    2026-08-20 (4) user request) and returns it as a `PreparedImage` ready
    to append to `app/core/pipeline.py`'s attachment list -- `None` on any
    failure (never lets a rendering bug fail the whole run; the text
    instruction in `prompt_builder.py` still applies on its own).

    Unlike the per-request cache this replaced, this one DOES reuse an
    existing file across requests/runs (by design, per the user's explicit
    ask -- 1 shared image per geometry instead of every request re-rendering
    its own copy). This intentionally re-introduces the same staleness
    trade-off flagged and removed in the 2026-08-20 (3)/(4) history just
    above `build_safe_area_guide_png()`: if this function's drawing logic
    changes again in the future, files already sitting in
    `safe_area_guide_dir` will NOT automatically pick up the change -- clear
    that directory (or delete the specific stale files) after any future
    edit to `build_safe_area_guide_png()`."""
    from app.settings import get_settings

    try:
        out_dir = get_settings().safe_area_guide_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        box_x, box_y, box_w, box_h = crop_box
        dest = out_dir / guide_filename(canvas_wh, (box_w, box_h))

        if dest.exists():
            logger.info("Reusing cached safe-area guide image -> %s", dest)
        else:
            dest.write_bytes(build_safe_area_guide_png(canvas_wh, crop_box))
            logger.info("Built safe-area guide image (canvas=%s, crop_box=%s) -> %s", canvas_wh, crop_box, dest)

        return PreparedImage(label="safe_area_guide", source_path=dest, resized_path=dest, mime_type="image/png")
    except Exception:  # noqa: BLE001 -- guide image is a best-effort aid, never fatal
        logger.exception("Could not build/find safe-area guide image for canvas=%s crop_box=%s -- continuing without it.", canvas_wh, crop_box)
        return None


# ---------------------------------------------------------------------------
# Chạy độc lập file này (xem thử guide image trông như thế nào):
#   python -m app.core.safe_area_guide --canvas 1536x1024 --box 250,302,1035,420 --out /tmp/guide.png
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas", required=True, help="WIDTHxHEIGHT, e.g. 1536x1024")
    parser.add_argument("--box", required=True, help="x,y,width,height, e.g. 250,302,1035,420")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cw, ch = (int(v) for v in args.canvas.lower().split("x"))
    bx, by, bw, bh = (int(v) for v in args.box.split(","))
    png_bytes = build_safe_area_guide_png((cw, ch), (bx, by, bw, bh))
    with open(args.out, "wb") as f:
        f.write(png_bytes)
    print(f"Wrote {args.out} ({len(png_bytes)} bytes)")
