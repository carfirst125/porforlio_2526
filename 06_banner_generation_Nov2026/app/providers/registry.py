"""Maps a `request.json`/`model_usage.json` model name string (e.g.
`"gpt-image-2"`) to the provider client that implements it, with that
provider's endpoint/key/deployment pulled from `Settings`.

`available_providers()` is every model name this registry KNOWS ABOUT
(used for the `mode=custom` picker + `GET /models`), independent of whether
its `.env` credentials happen to be filled in right now -- a missing
credential surfaces as a per-call `ProviderError` (caught by the pipeline,
turned into an `ok: false` result), not a hidden entry.
"""
from __future__ import annotations

from typing import Callable

from app.providers import azure_generic_image, azure_openai_image, gemini_image
from app.providers.base import ImageAttachment, ProviderError
from app.settings import Settings

GenerateFn = Callable[[str, list[ImageAttachment], int, int, Settings, "dict | None"], bytes]


def _gemini_31_flash(
    prompt: str, attachments: list[ImageAttachment], width: int, height: int, s: Settings, usage_sink: dict | None = None
) -> bytes:
    return gemini_image.generate(
        endpoint=s.gcp_gemini_31_flash_image_endpoint,
        api_key=s.gcp_vertex_api_key,
        prompt=prompt,
        attachments=attachments,
        width=width,
        height=height,
        usage_sink=usage_sink,
    )


def _gemini_3_pro(
    prompt: str, attachments: list[ImageAttachment], width: int, height: int, s: Settings, usage_sink: dict | None = None
) -> bytes:
    return gemini_image.generate(
        endpoint=s.gcp_gemini_3_pro_image_endpoint,
        api_key=s.gcp_vertex_api_key,
        prompt=prompt,
        attachments=attachments,
        width=width,
        height=height,
        usage_sink=usage_sink,
    )


def _gpt_image_2(
    prompt: str, attachments: list[ImageAttachment], width: int, height: int, s: Settings, usage_sink: dict | None = None
) -> bytes:
    return azure_openai_image.generate(
        endpoint=s.az_gpt_image_2_endpoint,
        api_key=s.az_gpt_image_2_key,
        prompt=prompt,
        attachments=attachments,
        width=width,
        height=height,
        usage_sink=usage_sink,
    )


def _mai_image_25(
    prompt: str, attachments: list[ImageAttachment], width: int, height: int, s: Settings, usage_sink: dict | None = None
) -> bytes:
    return azure_generic_image.generate(
        endpoint=s.az_mai_image_25_endpoint,
        api_key=s.az_mai_image_25_key,
        deployment=s.az_mai_image_25_deployment,
        api_version=s.az_mai_image_25_api_version,
        prompt=prompt,
        attachments=attachments,
        width=width,
        height=height,
        usage_sink=usage_sink,
    )


def _mai_image_25_flash(
    prompt: str, attachments: list[ImageAttachment], width: int, height: int, s: Settings, usage_sink: dict | None = None
) -> bytes:
    return azure_generic_image.generate(
        endpoint=s.az_mai_image_25_flash_endpoint,
        api_key=s.az_mai_image_25_flash_key,
        deployment=s.az_mai_image_25_flash_deployment,
        api_version=s.az_mai_image_25_flash_api_version,
        prompt=prompt,
        attachments=attachments,
        width=width,
        height=height,
        usage_sink=usage_sink,
    )


def _mai_image_25_pro(
    prompt: str, attachments: list[ImageAttachment], width: int, height: int, s: Settings, usage_sink: dict | None = None
) -> bytes:
    return azure_generic_image.generate(
        endpoint=s.az_mai_image_25_pro_endpoint,
        api_key=s.az_mai_image_25_pro_key,
        deployment=s.az_mai_image_25_pro_deployment,
        api_version=s.az_mai_image_25_api_version,
        prompt=prompt,
        attachments=attachments,
        width=width,
        height=height,
        usage_sink=usage_sink,
    )


def _flux_2_pro(
    prompt: str, attachments: list[ImageAttachment], width: int, height: int, s: Settings, usage_sink: dict | None = None
) -> bytes:
    return azure_generic_image.generate(
        endpoint=s.az_flux_2_pro_endpoint,
        api_key=s.az_flux_2_pro_key,
        deployment=s.az_flux_2_pro_deployment,
        api_version=s.az_flux_2_pro_api_version,
        prompt=prompt,
        attachments=attachments,
        width=width,
        height=height,
        usage_sink=usage_sink,
    )


_REGISTRY: dict[str, GenerateFn] = {
    "gemini-3.1-flash-image": _gemini_31_flash,
    "gemini-3-pro-image": _gemini_3_pro,
    "gpt-image-2": _gpt_image_2,
    "mai-image-2.5": _mai_image_25,
    "mai-image-2.5-flash": _mai_image_25_flash,
    "mai-image-2.5-pro": _mai_image_25_pro,
    "flux.2-pro": _flux_2_pro,
}


def available_providers() -> list[str]:
    return list(_REGISTRY.keys())


def generate_image(
    model_name: str,
    prompt: str,
    attachments: list[ImageAttachment],
    width: int,
    height: int,
    settings: Settings,
    usage_sink: dict | None = None,
) -> bytes:
    """`usage_sink`, when passed, is filled in with this call's raw token
    usage (whatever fields the underlying provider's own API response
    reports -- see each `app/providers/*.py generate()`), for
    `app/core/usage_tracker.py` to turn into a cost (2026-09-06 user
    request). Additive/optional -- omitting it is unchanged behavior."""
    fn = _REGISTRY.get(model_name)
    if fn is None:
        raise ProviderError(f"Unknown image model {model_name!r} -- not registered in app/providers/registry.py")
    return fn(prompt, attachments, width, height, settings, usage_sink)


# ---------------------------------------------------------------------------
# Chạy độc lập file này -- test 1 image model thật, không cần chạy cả app
# (đúng nhu cầu gốc "chạy thử nghiệm" từng model trong REQUEST.md):
#   python -m app.providers.registry --model gpt-image-2 --prompt "a red apple on a white background" \
#       --width 512 --height 512 --out test.png [--image ref1.png --image ref2.jpg]
#
# Yêu cầu trước khi chạy:
#   - `.env` đã có endpoint/key thật cho --model được chọn (xem .env.example)
#   - `--model` phải là 1 trong: xem `available_providers()` ở trên
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from app.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=available_providers())
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--out", default="test_output.png")
    parser.add_argument("--image", action="append", default=[], help="reference image path, can repeat")
    args = parser.parse_args()

    atts = [
        ImageAttachment(label="ref", mime_type="image/png" if p.lower().endswith("png") else "image/jpeg", data=Path(p).read_bytes())
        for p in args.image
    ]
    result_bytes = generate_image(args.model, args.prompt, atts, args.width, args.height, get_settings())
    Path(args.out).write_bytes(result_bytes)
    print(f"Wrote {len(result_bytes)} bytes to {args.out}")
