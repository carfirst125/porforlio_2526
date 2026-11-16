"""Unit tests -- app/core/safe_area_guide.py (2026-08-20, user-reported
failure: gpt-image-2 placed content outside the declared safe_area despite
an explicit text instruction with pixel coordinates; this module renders an
actual picture of the boundary to attach alongside that text)."""
from __future__ import annotations

import io

from PIL import Image

from app.core import safe_area_guide
from app.core.image_prep import PreparedImage


def test_guide_png_is_same_size_as_canvas():
    png_bytes = safe_area_guide.build_safe_area_guide_png((1536, 1024), (250, 302, 1035, 420))
    with Image.open(io.BytesIO(png_bytes)) as img:
        assert img.size == (1536, 1024)
        assert img.format == "PNG"


def test_guide_marks_inside_vs_outside_the_safe_area():
    """A pixel well inside the safe area must NOT be the magenta tint used
    outside it; a pixel well outside must be tinted (or on/near the
    magenta border)."""
    canvas_wh = (1536, 1024)
    crop_box = (250, 302, 1035, 420)  # x, y, w, h
    png_bytes = safe_area_guide.build_safe_area_guide_png(canvas_wh, crop_box)
    with Image.open(io.BytesIO(png_bytes)) as img:
        img = img.convert("RGB")

        # Center of the safe area -- well clear of both the caption bar (top
        # 30px) and the border line -- must stay plain white (untinted).
        inside_x, inside_y = 250 + 1035 // 2, 302 + 420 // 2
        assert img.getpixel((inside_x, inside_y)) == (255, 255, 255)

        # Bottom-right corner of the canvas -- well outside the safe area
        # and outside the caption bar -- must be tinted magenta-ish (not
        # plain white), since it's in the "will be cropped away" zone.
        outside_x, outside_y = canvas_wh[0] - 5, canvas_wh[1] - 5
        r, g, b = img.getpixel((outside_x, outside_y))
        assert (r, g, b) != (255, 255, 255)
        assert r > g  # magenta wash: red channel clearly stronger than green


def test_guide_handles_a_crop_box_that_touches_the_canvas_edge_without_crashing():
    # Degenerate/edge-hugging box -- must clamp, never raise.
    png_bytes = safe_area_guide.build_safe_area_guide_png((800, 600), (0, 0, 800, 600))
    with Image.open(io.BytesIO(png_bytes)) as img:
        assert img.size == (800, 600)


def test_guide_filename_uses_canvas_and_safe_area_sizes():
    """2026-08-20 (4), user's exact naming convention:
    `safe_area_guide_<canvas_w>x<canvas_h>_<safe_w>x<safe_h>.png`."""
    name = safe_area_guide.guide_filename((1536, 1024), (1035, 420))
    assert name == "safe_area_guide_1536x1024_1035x420.png"


def test_build_safe_area_guide_attachment_returns_prepared_image(isolated_safe_area_guide_dir):
    result = safe_area_guide.build_safe_area_guide_attachment((1536, 1024), (250, 302, 1035, 420))
    assert isinstance(result, PreparedImage)
    assert result.label == "safe_area_guide"
    assert result.mime_type == "image/png"
    assert result.resized_path.exists()
    assert result.resized_path.name == "safe_area_guide_1536x1024_1035x420.png"
    assert result.resized_path.parent == isolated_safe_area_guide_dir


def test_build_safe_area_guide_attachment_uses_the_same_filename_for_same_geometry(isolated_safe_area_guide_dir):
    """Same (canvas, crop_box) on a 2nd call must resolve to the SAME
    filename; a crop_box with a genuinely different SIZE must get its OWN
    file -- never silently collide with unrelated geometry.

    NOTE: the filename is keyed on (canvas_size, safe_area_size) only, NOT
    x/y offset (see `guide_filename()`'s own docstring for why, and its
    documented limitation) -- so this test deliberately varies WIDTH, not
    just position, to stay a valid check of that."""
    first = safe_area_guide.build_safe_area_guide_attachment((1536, 1024), (250, 302, 1035, 420))
    second = safe_area_guide.build_safe_area_guide_attachment((1536, 1024), (250, 302, 1035, 420))
    assert first.resized_path == second.resized_path

    different = safe_area_guide.build_safe_area_guide_attachment((1536, 1024), (0, 0, 900, 420))
    assert different.resized_path != first.resized_path


def test_build_safe_area_guide_attachment_reuses_an_existing_file_instead_of_rebuilding(isolated_safe_area_guide_dir):
    """2026-08-20 (4), user's explicit new request: unlike the per-request
    cache this replaced (which had to always-overwrite, see the 2026-08-20
    (3) history note above `build_safe_area_guide_attachment()`), this
    SHARED library cache must reuse a file that's already there by name --
    "nếu có rồi thì đem ra sử dụng luôn". Deliberately re-introduces the
    staleness trade-off (also documented in that same docstring): existing
    content at the expected filename is treated as authoritative and is
    NOT regenerated, even if it no longer matches what
    `build_safe_area_guide_png()` would currently produce."""
    first = safe_area_guide.build_safe_area_guide_attachment((1536, 1024), (250, 302, 1035, 420))
    marker = b"NOT A REAL PNG -- PRE-EXISTING FILE, MUST BE REUSED AS-IS"
    first.resized_path.write_bytes(marker)

    second = safe_area_guide.build_safe_area_guide_attachment((1536, 1024), (250, 302, 1035, 420))
    assert second.resized_path == first.resized_path
    assert second.resized_path.read_bytes() == marker  # reused, NOT rebuilt


def test_build_safe_area_guide_attachment_shares_the_cache_across_different_request_names(isolated_safe_area_guide_dir):
    """The cache is keyed ONLY on (canvas_size, safe_area_size) now -- 2
    unrelated requests asking for the same geometry must resolve to the
    exact same file on disk (that's the whole point of moving this out of
    `output/<request_name>/...` and into a shared library directory)."""
    for_req_a = safe_area_guide.build_safe_area_guide_attachment((1536, 1024), (250, 302, 1035, 420))
    for_req_b = safe_area_guide.build_safe_area_guide_attachment((1536, 1024), (250, 302, 1035, 420))
    assert for_req_a.resized_path == for_req_b.resized_path
    assert for_req_a.resized_path.parent == isolated_safe_area_guide_dir
