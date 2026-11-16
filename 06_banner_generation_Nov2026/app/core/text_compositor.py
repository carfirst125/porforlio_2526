"""Renders headline/message/CTA(+button) directly onto the FINAL (post-crop)
image via PIL, from the exact JSON text -- never left to the image model to
draw. Used only when `design_options.text_by_code=True` (the 3 promo-banner
use cases in `AI_DESIGN_REQUEST_INPUT_3usecase.xlsx`); see
`app/core/text_style_profiles.py` for the fixed typography data this reads.

2026-08-27: a near-identical `compositor.py` existed briefly for entry (19)'s
full image-compositing experiment and was deleted when that architecture was
reverted (product/context stayed AI-drawn; only text was ever a hard
requirement to move off the image model). This module is a fresh, narrower
rewrite -- text/CTA/button only, nothing about product or character layers.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.core.text_style_profiles import ButtonSpec, TextStyleProfile, profile_for
from app.logging_conf import get_logger
from app.models.schemas import CreativeRequest
from app.settings import get_settings

logger = get_logger("text_compositor")

# Manrope isn't in this repo yet (no `lib/fonts/Manrope-*.ttf`) -- fall back
# to the closest bundled Windows weight, with a warning every time so a real
# Manrope drop-in is easy to notice is still needed.
_MANROPE_FILENAMES = {
    "SemiBold": "Manrope-SemiBold.ttf",
    "Medium": "Manrope-Medium.ttf",
}
_WINDOWS_FALLBACK_FONTS = {
    "SemiBold": r"C:\Windows\Fonts\seguisb.ttf",
    "Medium": r"C:\Windows\Fonts\segoeui.ttf",
}

_PT_TO_PX = 96 / 72  # PIL/Pillow draws fonts in px; the Excel gives pt sizes at a typical 96dpi screen


def _resolve_font_path(weight: str) -> str:
    fonts_dir = get_settings().asset_library_dir / "fonts"
    manrope_path = fonts_dir / _MANROPE_FILENAMES.get(weight, f"Manrope-{weight}.ttf")
    if manrope_path.exists():
        return str(manrope_path)
    fallback = _WINDOWS_FALLBACK_FONTS.get(weight, _WINDOWS_FALLBACK_FONTS["Medium"])
    logger.warning(
        "Manrope %s not found at %s -- falling back to %s. Drop the real Manrope .ttf files into %s to fix this.",
        weight, manrope_path, fallback, fonts_dir,
    )
    return fallback


def _font(weight: str, size_pt: int) -> ImageFont.FreeTypeFont:
    path = _resolve_font_path(weight)
    size_px = max(1, round(size_pt * _PT_TO_PX))
    return ImageFont.truetype(path, size_px)


def wrap_and_fit(text: str, font: ImageFont.FreeTypeFont, max_width: int, max_lines: int) -> tuple[list[str], bool]:
    """Greedy word-wrap `text` to fit within `max_width` px, capped at
    `max_lines`. Returns `(lines, truncated)` -- `truncated` is True when it
    didn't all fit and the last line was ellipsized (also logged, never
    silent)."""
    words = text.split()
    if not words:
        return [], False

    lines: list[str] = []
    current = words[0]
    idx = 1
    while idx < len(words):
        word = words[idx]
        candidate = f"{current} {word}"
        if font.getlength(candidate) <= max_width:
            current = candidate
            idx += 1
            continue
        lines.append(current)
        current = word
        idx += 1
        if len(lines) == max_lines:
            break
    else:
        lines.append(current)
        return lines, False

    # Hit `max_lines` with words still left over -- ellipsize the last
    # accepted line rather than silently dropping the remainder.
    leftover = " ".join([current] + words[idx:])
    logger.warning("Text %r exceeds max_lines=%d at this width -- truncating (dropped: %r).", text, max_lines, leftover)
    truncated_line = lines[-1]
    while font.getlength(truncated_line + "…") > max_width and " " in truncated_line:
        truncated_line = truncated_line.rsplit(" ", 1)[0]
    lines[-1] = truncated_line + "…"
    return lines, True


def wrap_multiline(text: str, font: ImageFont.FreeTypeFont, max_width: int, max_lines: int) -> tuple[list[str], bool]:
    """Like `wrap_and_fit`, but treats an explicit `\\n` in `text` (e.g. a
    request author deliberately writing a headline as 2 named lines) as a
    hard line break rather than letting word-wrap decide where lines fall
    -- each `\\n`-separated segment is still word-wrapped/ellipsized on its
    own if it's too wide, within whatever line budget remains. Returns
    `(lines, truncated)`, same contract as `wrap_and_fit`."""
    lines: list[str] = []
    truncated = False
    paragraphs = text.split("\n")
    for i, paragraph in enumerate(paragraphs):
        if len(lines) >= max_lines:
            logger.warning(
                "Text %r has %d explicit line(s) but max_lines=%d -- dropping remaining line(s): %r.",
                text, len(paragraphs), max_lines, paragraphs[i:],
            )
            truncated = True
            break
        sub_lines, sub_truncated = wrap_and_fit(paragraph, font, max_width, max_lines - len(lines))
        lines.extend(sub_lines)
        truncated = truncated or sub_truncated
    return lines[:max_lines], truncated


def fit_font_and_wrap(
    weight: str, nominal_size_pt: int, text: str, max_width: int, max_lines: int, floor_ratio: float = 0.55
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Auto-shrink: tries `nominal_size_pt` first (the Excel-specified
    size), then progressively smaller sizes down to `floor_ratio` of it,
    picking the largest size that fits `text` in `max_lines` at
    `max_width` with no ellipsis. Falls back to the floor size's
    (possibly still-truncated) result if nothing fits -- `wrap_multiline`
    already logs that case. This exists because the Excel gives ONE fixed
    point size per role regardless of how long the actual headline/message
    copy turns out to be (verified against real usecase copy: a 25+
    character Vietnamese headline does not fit on 1 line at 53pt in a
    1024px-wide banner) -- shrinking beats silently cutting real marketing
    copy short."""
    steps = 6
    floor_size = max(1, round(nominal_size_pt * floor_ratio))
    sizes = [nominal_size_pt] + [
        round(nominal_size_pt - (nominal_size_pt - floor_size) * i / steps) for i in range(1, steps + 1)
    ]

    font = lines = None
    for size_pt in sizes:
        font = _font(weight, size_pt)
        lines, truncated = wrap_multiline(text, font, max_width, max_lines)
        if not truncated:
            if size_pt != nominal_size_pt:
                logger.info("Shrunk %r from %dpt to %dpt to fit %d line(s) at %dpx wide.", text, nominal_size_pt, size_pt, max_lines, max_width)
            return font, lines
    logger.warning("Could not fit %r within %d line(s) at %dpx even at floor size %dpt -- rendering truncated.", text, max_lines, max_width, floor_size)
    return font, lines


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # noqa: E203


def _draw_button(draw: ImageDraw.ImageDraw, *, right_x: int, top_y: int, cta_text: str, button: ButtonSpec) -> int:
    """Draws the CTA button so its right edge is at `right_x`, top at
    `top_y`. Returns the button's bottom y (for layout bookkeeping)."""
    available_text_w = button.max_width - 2 * button.padding_left_right
    font, lines = fit_font_and_wrap(button.font_weight, button.font_size, cta_text, available_text_w, max_lines=1)
    cta_text = lines[0] if lines else cta_text
    text_w = font.getlength(cta_text)
    ascent, descent = font.getmetrics()
    text_h = ascent + descent

    btn_w = min(button.max_width, int(text_w) + 2 * button.padding_left_right)
    btn_h = text_h + button.padding_top + button.padding_bottom
    left_x = right_x - btn_w

    radius = btn_h // 2
    draw.rounded_rectangle(
        [left_x, top_y, left_x + btn_w, top_y + btn_h],
        radius=radius,
        fill=_hex_to_rgb(button.fill_color),
    )
    text_x = left_x + (btn_w - text_w) / 2
    text_y = top_y + button.padding_top
    draw.text((text_x, text_y), cta_text, font=font, fill=_hex_to_rgb(button.text_color))
    return top_y + btn_h


def _draw_plain_cta(draw: ImageDraw.ImageDraw, *, right_x: int, top_y: int, cta_text: str, profile: TextStyleProfile, zone_w: int, text_color: str) -> int:
    font, lines = fit_font_and_wrap(profile.cta_font_weight, profile.cta_font_size, cta_text, zone_w, max_lines=1)
    cta_text = lines[0] if lines else cta_text
    text_w = font.getlength(cta_text)
    ascent, descent = font.getmetrics()
    draw.text((right_x - text_w, top_y), cta_text, font=font, fill=_hex_to_rgb(text_color))
    return top_y + ascent + descent


def _region_is_dark(img: Image.Image, box: tuple[int, int, int, int]) -> bool:
    """True if the average brightness of `img` inside `box` (x0, y0, x1,
    y1) is dark -- used to pick a light-on-dark vs dark-on-light text color
    so message/CTA text stays legible regardless of which `dominant_color`
    a request chose (2026-08-27: verified against a real render that a
    request-chosen dark blue background made the fixed dark-gray message
    color nearly unreadable -- fixed colors alone can't work for every
    possible background, so this measures the ACTUAL generated pixels)."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 <= x0 or y1 <= y0:
        return False
    region = img.crop((x0, y0, x1, y1)).convert("L")
    # 1 histogram-based average instead of per-pixel Python loop -- fast
    # even for a large region.
    histogram = region.histogram()
    total_pixels = sum(histogram)
    if total_pixels == 0:
        return False
    avg_luminance = sum(value * count for value, count in enumerate(histogram)) / total_pixels
    return avg_luminance < 128


def render_text_layer(image_bytes: bytes, req: CreativeRequest, output_wh: tuple[int, int]) -> bytes:
    """Draws headline/message/CTA(+button) onto `image_bytes` per the
    profile for `output_wh`. Returns new PNG bytes unchanged on any error
    or when there's no matching profile/no text to draw at all -- this must
    never be able to fail a whole (model, version) result."""
    profile = profile_for(output_wh)
    if profile is None:
        logger.warning("No text style profile for output size %s -- skipping code-rendered text.", output_wh)
        return image_bytes

    headline = req.text.headline.strip()
    message = req.text.message.strip()
    cta_label = req.text.cta_label.strip()
    if not headline and not message and not cta_label:
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        canvas_w, canvas_h = img.size
        zone_w = int(canvas_w * profile.text_zone_right_fraction)
        right_x = canvas_w - max(24, int(canvas_w * 0.03))  # small outer margin so text never touches the edge
        cursor_y = max(24, int(canvas_h * 0.06))
        zone_is_dark = _region_is_dark(img, (canvas_w - zone_w, 0, canvas_w, canvas_h))
        message_color = profile.message_color_dark_bg if zone_is_dark else profile.message_color
        plain_cta_color = profile.cta_text_color_dark_bg if zone_is_dark else profile.cta_text_color

        headline_lines: list[str] = []
        if headline:
            hfont, headline_lines = fit_font_and_wrap(profile.headline_font_weight, profile.headline_font_size, headline, zone_w, profile.headline_max_lines)
            for i, line in enumerate(headline_lines):
                color = profile.headline_colors[min(i, len(profile.headline_colors) - 1)]
                line_w = hfont.getlength(line)
                ascent, descent = hfont.getmetrics()
                draw.text((right_x - line_w, cursor_y), line, font=hfont, fill=_hex_to_rgb(color))
                cursor_y += ascent + descent

        show_message = bool(message) and (
            not profile.message_only_if_headline_is_one_line or len(headline_lines) <= 1
        )
        if show_message:
            cursor_y += profile.message_gap_below_headline_px
            mfont, message_lines = fit_font_and_wrap(profile.message_font_weight, profile.message_font_size, message, zone_w, profile.message_max_lines)
            for line in message_lines:
                line_w = mfont.getlength(line)
                ascent, descent = mfont.getmetrics()
                draw.text((right_x - line_w, cursor_y), line, font=mfont, fill=_hex_to_rgb(message_color))
                cursor_y += ascent + descent

        if cta_label:
            cta_words = cta_label.split()
            if len(cta_words) > profile.cta_max_words:
                logger.warning("CTA label %r exceeds %d words -- rendering as-is anyway.", cta_label, profile.cta_max_words)
            cursor_y += 12
            if profile.button is not None:
                _draw_button(draw, right_x=right_x, top_y=cursor_y, cta_text=cta_label, button=profile.button)
            else:
                _draw_plain_cta(draw, right_x=right_x, top_y=cursor_y, cta_text=cta_label, profile=profile, zone_w=zone_w, text_color=plain_cta_color)

        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:  # noqa: BLE001 -- code-rendered text is critical for these use cases, but a bug here must not lose the generated image entirely
        logger.exception("Could not render code-based text layer for output_wh=%s -- returning the image without text.", output_wh)
        return image_bytes


# ---------------------------------------------------------------------------
# Chạy độc lập file này (test render lên 1 ảnh nền trơn, không cần gọi model):
#   python -m app.core.text_compositor --request USECASE1_PASSPORT_1024x440 --out /tmp/test.png
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.core.pipeline import load_request_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    req = load_request_json(args.request)
    w, h = req.final_output_wh()
    blank = Image.new("RGB", (w, h), (245, 255, 242))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    result = render_text_layer(buf.getvalue(), req, (w, h))
    Path(args.out).write_bytes(result)
    print(f"Wrote {args.out} ({len(result)} bytes)")
