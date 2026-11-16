"""Rewrites a FORMAT DESIGN GUIDELINE's margin/frame-size wording so it
correctly describes the SAFE AREA inside a larger generation canvas, instead
of the guideline's own original assumption that its frame IS the final
delivered banner (REQUEST.md's `generation.canvas` + `generation.safe_area`
+ crop pipeline did not exist yet when `lib/guidelines/guidelines.json` was
written).

2026-08-20 (user-reported + confirmed): `banner_popup_noti`'s own guideline
text says "giữ khoảng cách ... đến 4 cạnh của banner một khoảng ... ít nhất
48px" (keep >=48px from the banner's 4 edges) and explicitly "Đối với hình
ảnh điện thoại có giao diện app, có thể phóng to vượt ra ngoài biên an toàn"
(phone/app-UI imagery MAY extend past that margin) -- both statements are
about the OLD assumed "banner" (== what this app now calls the safe area),
not the larger `generation.canvas` the model is actually asked to fill.
Left as-is, this directly contradicts `prompt_builder.py`'s own CANVAS &
SAFE AREA section and was a very plausible cause of a real run where a
phone mockup bled toward the canvas edge (see
`4-project-evaluation-results.md`'s 2026-08-20 (11) entry for the full
investigation).

Instead of only patching this with a static override sentence
(`prompt_builder.py`'s OVERRIDE ON THE GUIDELINE ABOVE section, still kept
as a fallback for when this rewrite is skipped or fails), the user asked
for the guideline's margin numbers to be REWRITTEN directly: every
"distance from edge" value in the original text should have the matching
side's canvas-to-safe-area pixel offset ADDED to it (so a `48px` margin
against the old frame becomes `48 + left_offset` px measured from the new
canvas's left edge, etc.), and any "may bleed past the margin" exception
should be removed/neutralized. The arithmetic is precomputed here in Python
(not left to the LLM) so the model only ever adds 2 already-correct numbers
together per side, rather than reconstructing the geometry itself.

Cached the SAME way, and in the SAME shared directory, as the visual guide
image (`safe_area_guide.py` / `Settings.safe_area_guide_dir`) -- 1
rewritten guideline per (format_type, canvas_size, safe_area_size)
combination, looked up by filename and reused across every request/run
that needs it, so the LLM is only called once per distinct combination
(the user's explicit ask, "mục đích là tiết kiệm thời gian gọi LLM").
"""
from __future__ import annotations

from app.llm.base import LLMProviderError
from app.llm.factory import LLMFactory
from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("guideline_rewriter")

_REWRITE_SYSTEM_INSTRUCTION = (
    "You are a precise technical editor. You are given a design guideline (in Vietnamese) that describes a "
    "banner's frame size and safe margins, written for a frame that is now a SAFE AREA sitting inside a larger "
    "generation canvas. Your ONLY job is to renumber the margin/safe-zone/frame-edge distance statements to be "
    "correct for the new, larger canvas, using the exact per-side pixel offsets you are given (simple addition "
    "only -- do not invent new numbers), and to remove or neutralize any exception that allows content to cross "
    "that margin. Every other instruction (fonts, sizes unrelated to margins, colors, image-composition rules, "
    "display-location notes, etc.) must be preserved exactly as written, unchanged, in the same language, "
    "structure, and section numbering as the original. Output ONLY the rewritten guideline text -- no preamble, "
    "no explanation, no markdown code fences."
)


def rewritten_guideline_filename(format_type: str, canvas_wh: tuple[int, int], safe_area_wh: tuple[int, int]) -> str:
    """`<format_type>_<canvas_w>x<canvas_h>_<safe_w>x<safe_h>.txt` -- mirrors
    `safe_area_guide.py::guide_filename()`'s naming convention (same
    per-user request), but with the `format_type` as the prefix instead of
    the fixed `safe_area_guide` label, and a `.txt` extension since this is
    rewritten TEXT, not an image (the user's own example used `.png` by
    analogy with the visual guide's naming rule -- kept the directory and
    `<name>_<canvas>_<safe_area>` pattern exactly as asked, but used `.txt`
    here since a PNG extension on guideline prose would be misleading on
    disk).

    NOTE: like `guide_filename()`, this is keyed on SIZES only, not the
    safe area's x/y offset -- see that function's docstring for the same
    caveat (no real request.json today declares an off-center safe area
    with the same sizes as an existing centered one)."""
    canvas_w, canvas_h = canvas_wh
    sa_w, sa_h = safe_area_wh
    safe_format_type = (format_type or "unknown_format").strip() or "unknown_format"
    return f"{safe_format_type}_{canvas_w}x{canvas_h}_{sa_w}x{sa_h}.txt"


def rewrite_guideline_for_safe_area(
    format_type: str,
    guideline_text: str,
    canvas_wh: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> str | None:
    """Returns the safe-area-adjusted guideline text, or `None` if there's
    nothing to rewrite (empty `guideline_text`) or the rewrite couldn't be
    completed (LLM call failed, no credentials configured, cache dir
    unwritable, ...) -- callers must fall back to the original
    `guideline_text` on `None`, exactly like every other best-effort step
    in this app (REQUEST.md: dump the exception, never crash the run)."""
    if not guideline_text or not guideline_text.strip():
        return None

    canvas_w, canvas_h = canvas_wh
    sa_x, sa_y, sa_w, sa_h = crop_box
    left_offset = sa_x
    right_offset = canvas_w - (sa_x + sa_w)
    top_offset = sa_y
    bottom_offset = canvas_h - (sa_y + sa_h)

    try:
        out_dir = get_settings().safe_area_guide_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / rewritten_guideline_filename(format_type, canvas_wh, (sa_w, sa_h))

        if dest.exists():
            logger.info("Reusing cached safe-area-adjusted guideline -> %s", dest)
            return dest.read_text(encoding="utf-8")

        user_prompt = (
            f"ORIGINAL GUIDELINE TEXT (format_type={format_type!r}):\n{guideline_text}\n\n"
            "GEOMETRY: this guideline's own frame/margin numbers above were written assuming its frame IS the "
            f"final banner. That frame now corresponds to a SAFE AREA of {sa_w}x{sa_h}px sitting inside a "
            f"larger {canvas_w}x{canvas_h}px generation canvas, offset from the canvas's own edges by exactly: "
            f"left={left_offset}px, right={right_offset}px, top={top_offset}px, bottom={bottom_offset}px.\n\n"
            "TASK:\n"
            "1. For every 'keep at least Npx from the edge' (or equivalent margin/safe-zone) statement above, "
            "add the matching offset for that side to N (left offset for a left-edge margin, right offset for "
            "a right-edge margin, top offset for a top-edge margin, bottom offset for a bottom-edge margin) and "
            "state the new total instead of N. If a single statement covers all 4 edges with one number, restate "
            "it per-edge with each edge's own new number (the 4 offsets above are not equal in general).\n"
            "2. Remove or rewrite any exception that allows some element (including phone/device mockups or "
            "app-interface screenshots) to extend past that margin -- state instead that nothing may cross the "
            "safe-area boundary, no exceptions.\n"
            "3. Leave every other instruction unchanged: fonts, unrelated sizes, colors, image-composition "
            "rules, display-location notes, etc.\n"
            "Output ONLY the rewritten guideline text."
        )

        provider = LLMFactory.create(get_settings())
        rewritten = provider.generate(user_prompt, system=_REWRITE_SYSTEM_INSTRUCTION).strip()
        if not rewritten:
            raise ValueError("LLM returned an empty rewritten guideline")

        dest.write_text(rewritten, encoding="utf-8")
        logger.info(
            "Built safe-area-adjusted guideline (format_type=%s, canvas=%s, safe_area=%s, offsets L=%d R=%d T=%d B=%d) -> %s",
            format_type, canvas_wh, (sa_w, sa_h), left_offset, right_offset, top_offset, bottom_offset, dest,
        )
        return rewritten
    except LLMProviderError:
        logger.exception("LLM guideline rewrite failed for format_type=%s -- falling back to the original guideline text.", format_type)
        return None
    except Exception:  # noqa: BLE001 -- best-effort enhancement, never fatal to the run
        logger.exception("Could not build/find safe-area-adjusted guideline for format_type=%s -- falling back to the original guideline text.", format_type)
        return None


# ---------------------------------------------------------------------------
# Chạy độc lập file này (in ra guideline đã được viết lại cho 1 format_type):
#   python -m app.core.guideline_rewriter --format-type banner_popup_noti --canvas 1536x1024 --box 250,302,1035,420
#
# Yêu cầu trước khi chạy: .env có cấu hình LLM (AZ_GPT5MINI_ENDPOINT/KEY hoặc
# provider tương ứng với `llm_provider`), và format_type phải tồn tại trong
# lib/guidelines/guidelines.json.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.core.guidelines import resolve_guideline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format-type", required=True)
    parser.add_argument("--canvas", required=True, help="WIDTHxHEIGHT, e.g. 1536x1024")
    parser.add_argument("--box", required=True, help="x,y,width,height, e.g. 250,302,1035,420")
    args = parser.parse_args()

    cw, ch = (int(v) for v in args.canvas.lower().split("x"))
    bx, by, bw, bh = (int(v) for v in args.box.split(","))

    guideline = resolve_guideline(args.format_type)
    result = rewrite_guideline_for_safe_area(args.format_type, guideline.combined_text, (cw, ch), (bx, by, bw, bh))
    print(result if result is not None else "(rewrite failed or nothing to rewrite -- see logs above)")
