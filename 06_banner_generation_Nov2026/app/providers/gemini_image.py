"""Gemini image-generation models on Vertex AI (`gemini-3.1-flash-image`,
`gemini-3-pro-image`), called via the `:generateContent` REST endpoint
already fully-qualified in `.env` (`GCP_GEMINI_31_FLASH_IMAGE_ENDPOINT` /
`GCP_GEMINI_3_PRO_IMAGE_ENDPOINT`).

NOTE: exact request/response field names for image-output Gemini models can
shift between API versions -- this follows the documented `generateContent`
multimodal-in/multimodal-out shape (`contents[].parts[]` with `inlineData`
for images in, `generationConfig.responseModalities` to ask for an image
out, `candidates[].content.parts[].inlineData.data` for the image out,
base64-encoded). Adjust here if GCP's actual response shape differs.
"""
from __future__ import annotations

import base64

from app.core.canvas_planner import GEMINI_31_FLASH_IMAGE_RATIOS, closest_supported_ratio
from app.logging_conf import get_logger
from app.providers._http import post_json
from app.providers.base import ImageAttachment, ProviderError

logger = get_logger("providers.gemini_image")

# 2026-08-25: full, real list of aspect ratios `gemini-3.1-flash-image`
# actually supports (per Gemini's own supported-ratio table) -- the previous
# 8-entry list was stale/wrong (missing 1:4, 1:8, 4:1, 4:5, 5:4, 8:1, 21:9;
# had a bogus "5:2" that Gemini does not support at all).
_ASPECT_RATIO_BUCKETS = GEMINI_31_FLASH_IMAGE_RATIOS

# Above this estimated crop/stretch loss, the requested canvas ratio doesn't
# map well onto any ratio Gemini actually supports -- log a warning so it's
# visible without silently degrading output quality.
_CROP_LOSS_WARN_THRESHOLD = 0.12


def _closest_aspect_ratio(width: int, height: int) -> str:
    """Nearest Gemini-supported ratio to `width/height`, by LOG-ratio
    distance (symmetric between portrait/landscape -- e.g. 2:1 and 1:2 are
    equally "far" from 1:1, which plain `abs(a - b)` difference is not)."""
    target = width / height
    label, ratio_w, ratio_h = closest_supported_ratio(width, height, _ASPECT_RATIO_BUCKETS)
    ratio = ratio_w / ratio_h

    crop_loss = 1.0 - min(target / ratio, ratio / target)
    if crop_loss > _CROP_LOSS_WARN_THRESHOLD:
        logger.warning(
            "Requested canvas ratio %.3f (%dx%d) has no close Gemini-supported match -- nearest is %s "
            "(ratio %.3f, ~%.0f%% area loss on resize-to-canvas). Consider adjusting generation.canvas "
            "in the request.json closer to a supported ratio.",
            target, width, height, label, ratio, crop_loss * 100,
        )
    return label


def generate(
    *,
    endpoint: str,
    api_key: str,
    prompt: str,
    attachments: list[ImageAttachment],
    width: int,
    height: int,
    usage_sink: dict | None = None,
) -> bytes:
    """`usage_sink`, when passed, is filled in with this call's token usage
    from Gemini's own `usageMetadata` (2026-09-06 user request -- see
    `app/core/usage_tracker.py`). Return type (`bytes`) is unchanged;
    callers that pass no sink see no behavior change."""
    if not endpoint or not api_key:
        raise ProviderError("Gemini endpoint/API key not configured in .env")

    parts: list[dict] = [{"text": prompt}]
    for att in attachments:
        parts.append({"inlineData": {"mimeType": att.mime_type, "data": base64.b64encode(att.data).decode("ascii")}})

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": _closest_aspect_ratio(width, height)},
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    payload = post_json(endpoint, headers=headers, json_body=body)

    if usage_sink is not None:
        usage_meta = payload.get("usageMetadata") or {}
        usage_sink["provider"] = "gemini"
        usage_sink["input_tokens"] = usage_meta.get("promptTokenCount", 0)
        usage_sink["output_tokens"] = usage_meta.get("candidatesTokenCount", 0)

    try:
        candidates = payload["candidates"]
        for cand in candidates:
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
        raise KeyError("no inlineData part found in any candidate")
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"Unexpected Gemini response shape: {exc}; payload keys={list(payload.keys())}") from exc
