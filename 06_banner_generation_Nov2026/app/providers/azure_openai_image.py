"""`gpt-image-2` on Azure OpenAI (`AZ_GPT_IMAGE_2_ENDPOINT`, already the full
`.../openai/v1/images/generations` URL in `.env`).

Uses `images/edits` (multipart, multiple `image[]` files) when reference
images are attached -- gpt-image-* models accept multiple input images on
this endpoint for image-guided generation -- and falls back to plain
`images/generations` (JSON, text-only) when there are none.
"""
from __future__ import annotations

import base64

from app.providers._http import post_json, post_multipart
from app.providers.base import ImageAttachment, ProviderError

# gpt-image-* only accepts a small fixed set of sizes (or "auto"). Map the
# request's arbitrary WxH canvas to the closest supported size -- the exact
# final pixels are still enforced by `app/core/image_post.py` afterwards.
_SUPPORTED_SIZES = [(1024, 1024), (1024, 1536), (1536, 1024)]


def _closest_supported_size(width: int, height: int) -> str:
    if width == height:
        return "1024x1024"
    target_ratio = width / height
    best = min(_SUPPORTED_SIZES, key=lambda wh: abs((wh[0] / wh[1]) - target_ratio))
    return f"{best[0]}x{best[1]}"


def _extension(mime_type: str) -> str:
    return "png" if "png" in mime_type else "jpg"


def _edits_url(generations_url: str) -> str:
    if "/images/generations" in generations_url:
        return generations_url.replace("/images/generations", "/images/edits")
    return generations_url.rstrip("/") + "/edits"


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
    from the images API's own `usage` object -- `input_tokens_details.
    {text_tokens,image_tokens}` maps onto `model_profiles.py`'s split text/
    image pricing shape for this model (2026-09-06 user request). Return
    type (`bytes`) is unchanged; callers that pass no sink see no behavior
    change."""
    if not endpoint or not api_key:
        raise ProviderError("gpt-image-2 endpoint/API key not configured in .env")

    headers = {"api-key": api_key}
    size = _closest_supported_size(width, height)

    if attachments:
        url = _edits_url(endpoint)
        files = [
            ("image[]", (f"ref_{i}.{_extension(att.mime_type)}", att.data, att.mime_type))
            for i, att in enumerate(attachments)
        ]
        data = {"model": "gpt-image-2", "prompt": prompt, "size": size, "n": "1"}
        payload = post_multipart(url, headers=headers, data=data, files=files)
    else:
        url = endpoint
        headers = {**headers, "Content-Type": "application/json"}
        body = {"model": "gpt-image-2", "prompt": prompt, "size": size, "n": 1}
        payload = post_json(url, headers=headers, json_body=body)

    if usage_sink is not None:
        usage = payload.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        usage_sink["provider"] = "azure-openai"
        usage_sink["input_tokens"] = usage.get("input_tokens", 0)
        usage_sink["output_tokens"] = usage.get("output_tokens", 0)
        usage_sink["input_text_tokens"] = details.get("text_tokens", 0)
        usage_sink["input_image_tokens"] = details.get("image_tokens", 0)
        usage_sink["output_image_tokens"] = usage.get("output_tokens", 0)

    try:
        b64 = payload["data"][0]["b64_json"]
        return base64.b64decode(b64)
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"Unexpected gpt-image-2 response shape: {exc}; payload keys={list(payload.keys())}") from exc
