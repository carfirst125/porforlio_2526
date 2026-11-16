"""Small hex-color helpers used to derive a gradient TINT/SHADE pair from a
single `design_options.dominant_color` value (REQUEST.md background-design
techniques, see `version_1/6-design-style-guide.md`: "Radial gradient
background from {TINT_HEX} at the center to {SHADE_HEX} at the edges").

TINT = a slightly LIGHTER version of the dominant color (used near the
center/around the product). SHADE = a slightly DARKER version of the SAME
color (used toward the outer edges). This is the standard color-theory
meaning of "tint"/"shade".

2026-09-16 (user-reported bug: generated banners kept coming out with a
plain OCB green/mint background regardless of the `dominant_color` set in
that request's `design_options`): root cause was NOT this module -- it was
never actually wired into the live prompt. `tint_shade_from_dominant()` had
been written and unit-tested (see `2026-09-02` note below) but had NO
caller anywhere in `app/`; the `{{TINT_HEX}}`/`{{SHADE_HEX}}` placeholders
it was meant to fill do not exist in the current `prompts/design_rules.md`
(that file was rewritten since -- it now only lists the fixed OCB brand
palette, no gradient placeholders at all). Meanwhile
`config/design_techniques_ocb_db.json`'s BACKGROUND-group technique entries
(`ocb_dark_glow_radial_field`, `ocb_mint_soft_gradient_field`,
`luxury_gold_radial_vignette_orbit_scene`, `festive_night_skyline_photo_wash`,
`architecture_photo_spotlight_overlay`) hardcoded literal colors ("deep
green", "light mint", "brand green wash", "gold-to-champagne") directly in
their `description`/`prompt_instruction` text -- a much more concrete,
directive instruction than the soft "{dominant_color} as the dominant
palette anchor" clause in `prompt_builder.py::_theme_instruction()`, so the
image model followed the hardcoded catalog color instead of the request's
actual `dominant_color` whenever one of those techniques got selected.

Fixed by (1) actually wiring this module in --
`fill_color_placeholders()` below is now called from
`prompt_builder.py::_design_techniques_text()` on every resolved
technique's `description`/`prompt_instruction` before it is folded into the
final prompt, and (2) rewriting those 5 background entries in
`design_techniques_ocb_db.json` to reference `{{dominant_color}}` /
`{{tint_hex}}` / `{{shade_hex}}` placeholders instead of a hardcoded color
name, while keeping each tier's gradient/glow SHAPE (radial glow, flat
gradient, concentric vignette, photo wash, spotlight beam) unchanged.

2026-09-02 (user request): originally written so
`app/core/prompt_builder.py::_design_rules_text()` would call
`tint_shade_from_dominant()` to fill those 2 placeholders whenever
`design_options.dominant_color` is set, so the image model gets concrete,
already-computed hex values instead of having to invent a gradient itself.
That specific call site no longer exists (see the 2026-09-16 note above for
where this module is actually wired in today).

2026-09-02 (2) (user-reported bug, real generated banners reviewed): the
first version of this function mixed the dominant color toward pure
white/black in RGB space (65%/35%). 2 real problems came out of that:

1. Banner #1: the mix stayed close to the dominant hue, but was otherwise
   fine -- the user pointed at this one as the CORRECT amount of variation
   ("chỉ được đổi nhẹ", only a slight change).
2. Banner #2: the SAME formula, on a different (apparently darker)
   `dominant_color`, produced a SHADE_HEX that read as an almost completely
   different, near-black/near-unrelated color ("màu đã sai quá xa" -- user's
   words) -- RGB-space mixing toward black desaturates/darkens very
   aggressively once the starting color is already mid-to-dark, so a fixed
   35% black mix could swing wildly depending on the starting lightness.

Fixed by switching to HSL space and shifting LIGHTNESS ONLY by a small,
FIXED number of percentage points (not a ratio of the distance to
white/black) -- hue and saturation stay the same as `dominant_color`, so
TINT_HEX/SHADE_HEX always visually read as "the same color, just a bit
lighter/darker", regardless of how light or dark the original dominant
color already was.
"""
from __future__ import annotations

import colorsys
import re

from app.logging_conf import get_logger

logger = get_logger("color_utils")

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

# Used whenever `dominant_color` is missing/unparsable -- a neutral
# light-gray/dark-gray pair so the gradient rule in design_rules.md still
# reads sensibly instead of leaking a broken hex or an empty placeholder
# into the prompt sent to the image model.
DEFAULT_TINT_HEX = "#F5F5F5"
DEFAULT_SHADE_HEX = "#4D4D4D"

# How far TINT/SHADE move away from `dominant_color`'s own lightness, in HSL
# percentage points (0..1 scale, e.g. 0.14 == 14 percentage points). Small on
# purpose -- see the 2026-09-02 (2) note above: the user explicitly asked
# for TINT_HEX/SHADE_HEX to stay visually "xoay quanh" (revolve around) the
# dominant color, only a slight change, never a different-looking color.
DEFAULT_TINT_LIGHTNESS_DELTA = 0.14
DEFAULT_SHADE_LIGHTNESS_DELTA = 0.14

# Hard floor/ceiling so an already very light/dark dominant color still
# produces a *visible* (but still close) gradient step instead of clipping
# to flat white/black.
_MIN_LIGHTNESS = 0.10
_MAX_LIGHTNESS = 0.92

# Saturation is nudged slightly toward the tint direction (softer) / kept
# put for the shade direction (a fully unchanged saturation on a much
# darker lightness already reads as "deeper", not "different").
_TINT_SATURATION_FACTOR = 0.92


def hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    """`"#0F472F"` / `"0F472F"` -> `(15, 71, 47)`. Returns `None` for
    anything that isn't a valid 6-digit hex color (never raises -- REQUEST.md:
    never let an optional enhancement crash the pipeline over bad input)."""
    if not hex_color:
        return None
    match = _HEX_RE.match(hex_color.strip())
    if not match:
        return None
    raw = match.group(1)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, round(c))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def mix_hex(hex_color: str, target_rgb: tuple[int, int, int], ratio: float) -> str | None:
    """Linearly interpolate `hex_color` toward `target_rgb` by `ratio`
    (0 = unchanged, 1 = exactly `target_rgb`). Returns `None` when
    `hex_color` isn't parsable.

    NOTE: no longer used by `tint_shade_from_dominant()` below (see this
    module's docstring, 2026-09-02 (2)) -- kept as a small standalone
    utility in case a future caller wants a plain RGB-space mix."""
    base = hex_to_rgb(hex_color)
    if base is None:
        return None
    ratio = max(0.0, min(1.0, ratio))
    mixed = tuple(base[i] + (target_rgb[i] - base[i]) * ratio for i in range(3))
    return rgb_to_hex(mixed)


def hex_to_hsl(hex_color: str) -> tuple[float, float, float] | None:
    """`"#0F472F"` -> `(h, l, s)` each in `0..1` (Python's `colorsys` uses
    HLS ordering -- this wrapper returns `(hue, lightness, saturation)`, the
    more common reading order, to avoid mixing the two up at call sites).
    Returns `None` when `hex_color` isn't a valid 6-digit hex color."""
    rgb = hex_to_rgb(hex_color)
    if rgb is None:
        return None
    r, g, b = (c / 255.0 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, l, s


def hsl_to_hex(h: float, l: float, s: float) -> str:
    """Inverse of `hex_to_hsl()` -- `h`/`l`/`s` each in `0..1`."""
    l = max(0.0, min(1.0, l))
    s = max(0.0, min(1.0, s))
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return rgb_to_hex((r * 255.0, g * 255.0, b * 255.0))


def tint_shade_from_dominant(
    dominant_color: str,
    tint_lightness_delta: float = DEFAULT_TINT_LIGHTNESS_DELTA,
    shade_lightness_delta: float = DEFAULT_SHADE_LIGHTNESS_DELTA,
) -> tuple[str, str]:
    """Derive `(TINT_HEX, SHADE_HEX)` from a single dominant/theme color --
    the SAME hue/saturation as `dominant_color`, only lightness nudged up
    (TINT) or down (SHADE) by a small fixed amount, so both stay visually
    close to the original color ("xoay quanh dominant color, chỉ đổi nhẹ",
    2026-09-02 user request -- see this module's docstring for the earlier,
    too-aggressive RGB-mix version this replaced).

    Falls back to a neutral light/dark gray pair (logged as a warning, never
    raised) when `dominant_color` is empty or not a valid hex string --
    mirrors `_design_rules_text()`'s existing "Theme Color" fallback for the
    plain `{{DOMINANT_COLOR}}` placeholder."""
    hsl = hex_to_hsl(dominant_color)
    if hsl is None:
        if dominant_color and dominant_color.strip():
            logger.warning(
                "dominant_color=%r is not a valid hex color -- falling back to default TINT/SHADE (%s/%s)",
                dominant_color,
                DEFAULT_TINT_HEX,
                DEFAULT_SHADE_HEX,
            )
        return DEFAULT_TINT_HEX, DEFAULT_SHADE_HEX

    hue, lightness, saturation = hsl
    tint_l = max(_MIN_LIGHTNESS, min(_MAX_LIGHTNESS, lightness + tint_lightness_delta))
    shade_l = max(_MIN_LIGHTNESS, min(_MAX_LIGHTNESS, lightness - shade_lightness_delta))

    tint_hex = hsl_to_hex(hue, tint_l, saturation * _TINT_SATURATION_FACTOR)
    shade_hex = hsl_to_hex(hue, shade_l, saturation)
    return tint_hex, shade_hex


# Matches `technique_selector.py::_DOMINANT_COLOR_PLACEHOLDER_RE` (kept as a
# separate constant there instead of importing this one -- that call site's
# fallback wording is tuned for a technique-SELECTION prompt, this one's for
# the final image-gen prompt; see `fill_color_placeholders()` below).
_DOMINANT_COLOR_PLACEHOLDER_RE = re.compile(r"\{\{\s*dominant_color\s*\}\}", re.IGNORECASE)
_TINT_HEX_PLACEHOLDER_RE = re.compile(r"\{\{\s*tint_hex\s*\}\}", re.IGNORECASE)
_SHADE_HEX_PLACEHOLDER_RE = re.compile(r"\{\{\s*shade_hex\s*\}\}", re.IGNORECASE)

# Fallback text substituted for a bare `{{dominant_color}}` placeholder when
# the request sets no `design_options.dominant_color` -- never a hardcoded
# literal color (that would just reintroduce the same bug for the no-color
# case), a descriptive phrase instead so the sentence around it still reads
# sensibly.
DOMINANT_COLOR_FALLBACK_PHRASE = "OCB's approved brand theme color (no explicit dominant_color set for this request)"


def fill_color_placeholders(text: str, dominant_color: str) -> str:
    """Fill `{{dominant_color}}` / `{{tint_hex}}` / `{{shade_hex}}`
    placeholders that may appear inside a design-technique's `description`
    or `prompt_instruction` text (see the BACKGROUND-group entries in
    `config/design_techniques_ocb_db.json`) with values derived from THIS
    request's `design_options.dominant_color`.

    Called from `app/core/prompt_builder.py::_design_techniques_text()` on
    every resolved technique before it is folded into the final image-gen
    prompt -- this is what makes a technique's own color wording (e.g. a
    background gradient's hue) actually track the request's dominant color
    instead of a color name baked into the catalog file. Text with no
    placeholder in it (the common case -- most techniques, e.g. every entry
    in `design_techniques_general.json`, don't reference color at all) is
    returned unchanged.

    Never raises -- `tint_shade_from_dominant()` already falls back to a
    neutral gray pair and logs a warning for an empty/invalid
    `dominant_color`, so this always produces a usable substitution."""
    if not text:
        return text
    value = (dominant_color or "").strip()
    dominant_replacement = value or DOMINANT_COLOR_FALLBACK_PHRASE
    tint_hex, shade_hex = tint_shade_from_dominant(value)
    filled = _DOMINANT_COLOR_PLACEHOLDER_RE.sub(dominant_replacement, text)
    filled = _TINT_HEX_PLACEHOLDER_RE.sub(tint_hex, filled)
    filled = _SHADE_HEX_PLACEHOLDER_RE.sub(shade_hex, filled)
    return filled
