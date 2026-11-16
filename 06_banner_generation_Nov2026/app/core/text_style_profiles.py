"""Fixed typography specs for the 2 promo-banner output sizes defined in
`AI_DESIGN_REQUEST_INPUT_3usecase.xlsx` (1024x440 and 1024x256) -- plain
data, not a general-purpose typography DSL, because the Excel gives an
exact, closed spec per size rather than something a request author should
be tuning per-request.

Used by `app/core/text_compositor.py` (rendering) and `app/core/
prompt_builder.py` (to tell the image model which rectangle to leave
visually calm so the composited text doesn't collide with AI-drawn content).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ButtonSpec:
    fill_color: str
    max_width: int
    padding_left_right: int
    padding_top: int
    padding_bottom: int
    text_color: str
    font_weight: str
    font_size: int


@dataclass(frozen=True)
class TextStyleProfile:
    output_wh: tuple[int, int]

    headline_font_weight: str
    headline_font_size: int
    headline_colors: tuple[str, ...]  # 1 color per line, cycles/holds on the last if more lines than colors
    headline_max_lines: int

    message_font_weight: str
    message_font_size: int
    message_color: str
    message_color_dark_bg: str  # used instead of `message_color` when the actual generated background behind the text zone turns out dark (e.g. a request-chosen dark `dominant_color`)
    message_max_lines: int
    message_only_if_headline_is_one_line: bool
    message_gap_below_headline_px: int

    # CTA text always renders (Manrope SemiBold 39, white, max 5 words) --
    # `button` is `None` when this size has no button container at all
    # ("Không có button" for 1024x256): the CTA text then draws plain,
    # right-aligned, no filled background behind it.
    cta_font_weight: str
    cta_font_size: int
    cta_text_color: str
    cta_text_color_dark_bg: str  # only used when `button is None` (no fill behind the CTA text) and the background is dark
    cta_max_words: int
    button: ButtonSpec | None

    text_align: str  # "right" -- matches this project's existing banner_popup_noti convention
    text_zone_right_fraction: float  # reserved column width, as a fraction of canvas width, from the right edge


# 2026-08-27: the Excel gives no explicit point size for the "message"
# role at either output size -- only "Manrope Medium" with no number.
# Chosen as roughly half the headline size (53pt), a common type-hierarchy
# ratio; not from the source spec, flagged here so it's easy to find and
# correct once the user has a real preference.
_MESSAGE_FONT_SIZE_ASSUMED = 28

_BUTTON_1024x440 = ButtonSpec(
    fill_color="#00804B",
    max_width=412,
    padding_left_right=40,
    padding_top=12,
    padding_bottom=16,
    text_color="#FFFFFF",
    font_weight="SemiBold",
    font_size=39,
)

TEXT_STYLE_PROFILES: dict[tuple[int, int], TextStyleProfile] = {
    (1024, 440): TextStyleProfile(
        output_wh=(1024, 440),
        headline_font_weight="SemiBold",
        headline_font_size=53,
        headline_colors=("#00804B", "#E39717"),
        headline_max_lines=3,
        message_font_weight="Medium",
        message_font_size=_MESSAGE_FONT_SIZE_ASSUMED,
        message_color="#545C57",
        message_color_dark_bg="#F0F3F1",
        message_max_lines=2,
        message_only_if_headline_is_one_line=False,
        message_gap_below_headline_px=3,
        cta_font_weight="SemiBold",
        cta_font_size=39,
        cta_text_color="#FFFFFF",
        cta_text_color_dark_bg="#FFFFFF",
        cta_max_words=5,
        button=_BUTTON_1024x440,
        text_align="right",
        text_zone_right_fraction=0.60,
    ),
    (1024, 256): TextStyleProfile(
        output_wh=(1024, 256),
        headline_font_weight="SemiBold",
        headline_font_size=53,
        headline_colors=("#00804B",),
        headline_max_lines=1,
        message_font_weight="Medium",
        message_font_size=_MESSAGE_FONT_SIZE_ASSUMED,
        message_color="#545C57",
        message_color_dark_bg="#F0F3F1",
        message_max_lines=1,
        message_only_if_headline_is_one_line=True,
        message_gap_below_headline_px=3,
        cta_font_weight="SemiBold",
        cta_font_size=39,
        # 2026-08-27: Excel's row 11/G11 ("Tương tự") literally says the CTA
        # text color stays #FFFFFF even though this size has NO button fill
        # behind it (G12 "Không có button") -- tried that literally first,
        # then visually confirmed via `_test_render/` it's genuinely
        # unreadable (near-invisible white-on-light-background). Overridden
        # to the headline's green for a LIGHT background -- a real
        # legibility defect, not a style preference. On a DARK background
        # (`cta_text_color_dark_bg`), the original literal white is exactly
        # right again, so both are kept and picked at render time based on
        # the actual generated background color.
        cta_text_color="#00804B",
        cta_text_color_dark_bg="#FFFFFF",
        cta_max_words=5,
        button=None,
        text_align="right",
        text_zone_right_fraction=0.70,
    ),
}


def profile_for(output_wh: tuple[int, int]) -> TextStyleProfile | None:
    return TEXT_STYLE_PROFILES.get(output_wh)
