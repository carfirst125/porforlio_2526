"""Shared types for image-generation provider clients (`app/providers/*.py`).

Every provider module exposes 1 function with this signature:

    generate(prompt: str, attachments: list[ImageAttachment], width: int, height: int) -> bytes

...returning raw PNG/JPEG bytes of ONE generated image, or raising
`ProviderError` on failure. `app/core/pipeline.py` is the only caller and
always wraps it in try/except (REQUEST.md: never let a provider failure
crash the whole run -- 1 failed model/version becomes 1 `ok: false` result
entry, the rest of the run continues).
"""
from __future__ import annotations

from dataclasses import dataclass


class ProviderError(RuntimeError):
    """Raised by any `app/providers/*.py` client on a failed generation
    call (HTTP error, malformed response, missing credentials, ...)."""


@dataclass
class ImageAttachment:
    label: str
    mime_type: str
    data: bytes
