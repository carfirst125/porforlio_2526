"""One-off environment check -- run before starting the backend for the
first time (or after editing `.env`) to catch a missing/misconfigured value
early, instead of discovering it mid-`/generate` call. Does NOT make any
real network call by default (safe to run anytime, no API cost); pass
`--ping` to also do a lightweight reachability check per configured image
model endpoint (DNS + TCP connect only, no request body).

This is a preparation script (`script/`), not runtime app logic -- nothing
here is imported by `app/`.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import model_usage  # noqa: E402
from app.providers.registry import available_providers  # noqa: E402
from app.settings import get_settings  # noqa: E402

_ENDPOINT_KEY_PAIRS = {
    "gemini-3.1-flash-image": ("gcp_gemini_31_flash_image_endpoint", "gcp_vertex_api_key"),
    "gemini-3-pro-image": ("gcp_gemini_3_pro_image_endpoint", "gcp_vertex_api_key"),
    "gpt-image-2": ("az_gpt_image_2_endpoint", "az_gpt_image_2_key"),
    "mai-image-2.5": ("az_mai_image_25_endpoint", "az_mai_image_25_key"),
    "mai-image-2.5-flash": ("az_mai_image_25_flash_endpoint", "az_mai_image_25_flash_key"),
    "mai-image-2.5-pro": ("az_mai_image_25_pro_endpoint", "az_mai_image_25_pro_key"),
    "flux.2-pro": ("az_flux_2_pro_endpoint", "az_flux_2_pro_key"),
}


def _tcp_reachable(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True, "reachable"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    ping = "--ping" in sys.argv
    settings = get_settings()
    problems: list[str] = []

    print(f"environment           = {settings.environment}")
    print(f"input_dir              = {settings.input_dir} (exists={settings.input_dir.exists()})")
    print(f"output_dir             = {settings.output_dir} (exists={settings.output_dir.exists()})")
    print(f"asset_library_dir      = {settings.asset_library_dir} (exists={settings.asset_library_dir.exists()})")
    print(f"guidelines_file        = {settings.guidelines_file} (exists={settings.guidelines_file.exists()})")
    print(f"model_usage_file       = {settings.model_usage_file} (exists={settings.model_usage_file.exists()})")
    print()

    if not settings.guidelines_file.exists():
        problems.append("lib/guidelines/guidelines.json is missing")
    if not settings.model_usage_file.exists():
        problems.append("config/model_usage.json is missing")

    usage, usage_error = model_usage.load_model_usage()
    if usage_error:
        problems.append(f"config/model_usage.json failed to parse: {usage_error}")

    print(f"llm_provider           = {settings.llm_provider!r}")
    print(f"use_llm_prompt_refinement = {settings.use_llm_prompt_refinement}")
    if settings.use_llm_prompt_refinement and not (settings.az_gpt5mini_endpoint and settings.az_gpt5mini_key) and not settings.openai_api_key:
        problems.append("use_llm_prompt_refinement=true but no LLM endpoint/key is configured (prompt refinement will be skipped, not fatal)")
    print()

    print("Image model credentials:")
    for name in available_providers():
        endpoint_attr, key_attr = _ENDPOINT_KEY_PAIRS.get(name, (None, None))
        endpoint = getattr(settings, endpoint_attr, "") if endpoint_attr else ""
        key = getattr(settings, key_attr, "") if key_attr else ""
        configured = bool(endpoint and key)
        status = "OK" if configured else "NOT CONFIGURED"
        line = f"  {name:<24} {status}"
        if configured and ping:
            reachable, detail = _tcp_reachable(endpoint)
            line += f"  ping={'ok' if reachable else 'FAIL: ' + detail}"
            if not reachable:
                problems.append(f"{name}: endpoint not reachable ({detail})")
        print(line)
        if not configured:
            models_using_it = [
                mode for mode, models in usage.items() if not mode.startswith("_") and mode != "all" and name in models
            ]
            if models_using_it:
                problems.append(f"{name} is used by model_usage.json modes {models_using_it} but has no endpoint/key configured")

    print()
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("No problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   cd version_1 && python script/check_env.py            # không gọi network
#   cd version_1 && python script/check_env.py --ping     # + kiểm tra TCP reachability
#
# Yêu cầu trước khi chạy: `.env` đã tồn tại (xem .env.example), chạy từ
# thư mục version_1/ để import `app.*` resolve đúng.
# ---------------------------------------------------------------------------
