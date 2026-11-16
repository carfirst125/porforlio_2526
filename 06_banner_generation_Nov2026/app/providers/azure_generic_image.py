"""Shared client for the newer Azure-hosted image models that don't (yet)
have officially documented request/response shapes at the time this was
written: MAI-Image-2.5 / -Flash / -Pro, and FLUX.2-Pro. All 4 are called the
same way from `.env`'s endpoints (`AZ_MAI_IMAGE_25*_ENDPOINT`,
`AZ_FLUX_2_PRO_ENDPOINT`) with only the deployment/model name differing.

Follows the common "OpenAI images-compatible" JSON convention (`prompt`,
`size`, `n`, base64 image(s) in `response_format=b64_json`, reference images
passed as base64 in an `image`/`images` field) that most of these
Azure-preview image endpoints use. ADJUST HERE if the real provider docs for
these deployments specify a different request/response shape once
available -- this is the single place that would need to change.
"""
from __future__ import annotations

import base64

from app.providers._http import post_json
from app.providers.base import ImageAttachment, ProviderError


def generate(
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str,
    prompt: str,
    attachments: list[ImageAttachment],
    width: int,
    height: int,
    usage_sink: dict | None = None,
) -> bytes:
    """`usage_sink`, when passed, is filled in with this call's token usage
    IF the response reports a `usage` object in the shared OpenAI-images-
    compatible shape (2026-09-06 user request). MAI-2.5-* deployments are
    expected to report it; FLUX.2-pro is billed per-megapixel instead (see
    `model_profiles.estimate_cost()`) and may report no usage at all -- an
    empty/missing `usage` object here is normal and left as an empty sink,
    not an error. Return type (`bytes`) is unchanged; callers that pass no
    sink see no behavior change."""
    if not endpoint or not api_key:
        raise ProviderError(f"{deployment} endpoint/API key not configured in .env")

    headers = {"api-key": api_key, "Content-Type": "application/json"}
    body: dict = {
        "model": deployment,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "n": 1,
        "response_format": "b64_json",
    }
    if api_version:
        body["api_version"] = api_version
    if attachments:
        body["images"] = [base64.b64encode(att.data).decode("ascii") for att in attachments]

    payload = post_json(endpoint, headers=headers, json_body=body)

    if usage_sink is not None:
        usage = payload.get("usage") or {}
        if usage:
            details = usage.get("input_tokens_details") or {}
            usage_sink["provider"] = "azure-generic"
            usage_sink["input_tokens"] = usage.get("input_tokens", 0)
            usage_sink["output_tokens"] = usage.get("output_tokens", 0)
            usage_sink["input_text_tokens"] = details.get("text_tokens", 0)
            usage_sink["input_image_tokens"] = details.get("image_tokens", 0)
            usage_sink["output_image_tokens"] = usage.get("output_tokens", 0)

    try:
        b64 = payload["data"][0]["b64_json"]
        return base64.b64decode(b64)
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"Unexpected {deployment} response shape: {exc}; payload keys={list(payload.keys())}") from exc
