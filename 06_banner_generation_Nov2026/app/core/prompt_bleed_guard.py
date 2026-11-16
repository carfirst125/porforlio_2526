"""Final LLM-based safety pass over the FULLY ASSEMBLED prompt text, right
before it's returned by `build_prompt()`: scans the whole prompt for --
and strips -- any lingering wording that would permit the PRODUCT asset
(e.g. a phone/device mockup) to bleed past the safe area, and makes sure an
explicit "product MUST stay inside the safe area" instruction is present.

2026-08-20 (user request, after (11)/(12) already targeted the ONE known
conflicting sentence found in `guidelines.json`'s banner_popup_noti entry):
a real gemini-3.1-flash run STILL showed the product (a phone mockup)
bleeding past the safe area's top/bottom edges even after the guideline-
rewrite fix in `guideline_rewriter.py`. Rather than keep hunting by hand
for every remaining possible source of permissive wording (design_rules.md,
a future guideline entry, a design_brief someone writes later, ...), this
asks GPT-5-mini to scan the WHOLE final prompt text as one pass -- the one
thing actually sent to the image model -- and correct it directly, targeted
specifically at the PRODUCT asset (the user's exact ask).

No separate file cache here, unlike `guideline_rewriter.py`'s shared,
cross-request cache: this runs as the LAST step inside `build_prompt()`,
so its output is already captured by that function's own
`_prompt_cache/built_prompt.json` cache -- caching it a 2nd time would be
redundant.
"""
from __future__ import annotations

from app.llm.base import LLMProviderError
from app.llm.factory import LLMFactory
from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("prompt_bleed_guard")

_SYSTEM_INSTRUCTION = (
    "You are a precise technical editor reviewing a FULL prompt about to be sent to an image-generation AI "
    "model. The prompt describes a generation canvas and a safe area inside it; the PRODUCT asset (e.g. a "
    "phone/device mockup) must be fully contained within that safe area. Your ONLY job: "
    "(1) find and remove or rewrite any sentence anywhere in the prompt that permits the product asset to "
    "extend, bleed past, or exceed the safe area / margin / canvas boundary, for ANY stated reason (including "
    "'to show interface detail clearly'); "
    "(2) if the prompt does not already contain an unambiguous statement that the product asset MUST be fully "
    "contained within the safe area, add one clearly. "
    "Preserve every other sentence, section, and the overall structure and language exactly as written -- do "
    "not paraphrase, compress, reorder, or otherwise change anything unrelated to the product asset's "
    "containment. Output ONLY the corrected prompt text -- no preamble, no explanation, no markdown code fences."
)


def enforce_product_containment(prompt_text: str) -> str | None:
    """Returns the corrected prompt text, or `None` if there's nothing to
    check (`prompt_text` empty) or the correction pass couldn't be
    completed (LLM call failed, empty/unusable response, ...) -- callers
    must fall back to the original `prompt_text` unchanged on `None`
    (REQUEST.md: an optional enhancement must never crash, or silently
    leave the run worse off than before the enhancement existed)."""
    if not prompt_text or not prompt_text.strip():
        return None

    try:
        provider = LLMFactory.create(get_settings())
        corrected = provider.generate(prompt_text, system=_SYSTEM_INSTRUCTION).strip()
        if not corrected:
            raise ValueError("LLM returned an empty corrected prompt")
        logger.info("Product-containment guard pass applied (%d -> %d chars).", len(prompt_text), len(corrected))
        return corrected
    except LLMProviderError:
        logger.exception("LLM product-bleed guard call failed -- keeping the prompt text unchanged.")
        return None
    except Exception:  # noqa: BLE001 -- best-effort enhancement, never fatal to the run
        logger.exception("Could not run the product-bleed guard pass -- keeping the prompt text unchanged.")
        return None


# ---------------------------------------------------------------------------
# Chạy độc lập file này (in ra prompt sau khi được rà soát/sửa lại):
#   python -m app.core.prompt_bleed_guard --file /tmp/some_prompt.txt
#
# Yêu cầu trước khi chạy: .env có cấu hình LLM (AZ_GPT5MINI_ENDPOINT/KEY
# hoặc provider tương ứng với `llm_provider`).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Path to a .txt file containing the full prompt to check.")
    args = parser.parse_args()

    text = open(args.file, encoding="utf-8").read()
    result = enforce_product_containment(text)
    print(result if result is not None else "(guard pass failed or nothing to check -- see logs above)")
