"""Drafts missing `text.headline`/`text.message`/`text.cta_label` from
`design_brief` when a request opts into `design_options.text_by_code=True`
but leaves one of those fields as an empty string -- the JSON-authoring
equivalent of the Excel's "AI tự sáng tạo" cells (usecase 2, Esaving). Only
ever fills in fields that are already empty; a field the request author
actually wrote is never touched.

Same call pattern as `app/core/guideline_rewriter.py` /
`app/core/prompt_bleed_guard.py`: one `LLMFactory` text call, best-effort,
never fatal to the run -- on any failure this returns the request's own
(still-empty) text unchanged rather than crashing or inventing content
outside the LLM path.
"""
from __future__ import annotations

import json

from app.llm.base import LLMProviderError
from app.llm.factory import LLMFactory
from app.logging_conf import get_logger
from app.core.text_style_profiles import TextStyleProfile, profile_for
from app.models.schemas import CreativeRequest, TextContent
from app.settings import get_settings

logger = get_logger("content_writer")

_SYSTEM_INSTRUCTION = (
    "You are a Vietnamese banking marketing copywriter. You are given a design brief and must draft ONLY the "
    "requested missing fields for a promo banner. Output STRICT JSON with exactly the requested keys and "
    "nothing else -- no markdown fences, no explanation. Keep copy in Vietnamese, concise, on-brand for OCB."
)


def _build_prompt(req: CreativeRequest, profile: TextStyleProfile | None, missing_fields: list[str]) -> str:
    constraints = []
    if "headline" in missing_fields:
        max_lines = profile.headline_max_lines if profile else 2
        constraints.append(f'"headline": a short, attention-grabbing headline, at most {max_lines} short lines worth of text.')
    if "message" in missing_fields:
        max_lines = profile.message_max_lines if profile else 1
        constraints.append(f'"message": a supporting sentence describing the offer, at most {max_lines} short line(s).')
    if "cta_label" in missing_fields:
        max_words = profile.cta_max_words if profile else 5
        constraints.append(f'"cta_label": a call-to-action button label, at most {max_words} words, imperative mood.')

    return (
        f"DESIGN BRIEF:\n{req.design_brief}\n\n"
        f"CREATIVE ID: {req.creative_id}\n\n"
        "Draft the following JSON fields:\n" + "\n".join(f"- {c}" for c in constraints) + "\n\n"
        'Respond with a single JSON object containing only these keys, e.g. {"headline": "...", "message": "..."}.'
    )


def draft_missing_text(req: CreativeRequest, usage_tracker=None) -> TextContent:
    """Returns a `TextContent` with empty `headline`/`message`/`cta_label`
    fields filled in from `design_brief` via the LLM -- fields that already
    have a value are passed through untouched. Returns `req.text` unchanged
    (still empty where it was empty) if there's nothing missing, or if the
    LLM call fails for any reason.

    `usage_tracker`, when passed, records this call (2026-09-06 user
    request) -- this drafts text for the WHOLE request, not one version, so
    it is recorded with `version=None`."""
    missing_fields = [
        name
        for name, value in (
            ("headline", req.text.headline),
            ("message", req.text.message),
            ("cta_label", req.text.cta_label),
        )
        if not value.strip()
    ]
    if not missing_fields:
        return req.text

    profile = profile_for(req.final_output_wh())
    prompt = _build_prompt(req, profile, missing_fields)
    settings = get_settings()
    usage_sink: dict = {} if usage_tracker is not None else None

    try:
        provider = LLMFactory.create(settings)
        raw = provider.generate(prompt, system=_SYSTEM_INSTRUCTION, usage_sink=usage_sink).strip()
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()
        drafted = json.loads(raw)
    except (LLMProviderError, json.JSONDecodeError, ValueError):
        logger.exception("Could not draft missing text fields %s for request=%r -- leaving them empty.", missing_fields, req.creative_id)
        return req.text
    except Exception:  # noqa: BLE001 -- best-effort enhancement, never fatal to the run
        logger.exception("Unexpected error drafting text fields %s for request=%r -- leaving them empty.", missing_fields, req.creative_id)
        return req.text
    finally:
        if usage_tracker is not None:
            usage_tracker.record_llm(usage_sink, stage="content_writer_draft_missing_text", version=None, fallback_model=settings.llm_model_name)

    updated = req.text.model_copy(
        update={field: str(drafted[field]).strip() for field in missing_fields if isinstance(drafted.get(field), str) and drafted[field].strip()}
    )
    logger.info("Drafted missing text fields %s for request=%r via LLM.", missing_fields, req.creative_id)
    return updated


# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   python -m app.core.content_writer --request USECASE2_ESAVING_1024x440
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.core.pipeline import load_request_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    req = load_request_json(args.request)
    result = draft_missing_text(req)
    print(result.model_dump_json(indent=2))
