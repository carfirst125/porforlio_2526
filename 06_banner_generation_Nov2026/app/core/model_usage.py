"""Load `config/model_usage.json` (REQUEST.md [4]) and resolve a run
`mode` into the concrete list of image model names to run.

`model_usage.json` shape:

    {
      "default": ["gemini-3.1-flash-image"],
      "quality": ["gpt-image-2", ...],
      "eco": ["gemini-3.1-flash-image", ...],
      "custom": [],
      "all": ["quality", "eco"],
      "_note_...": "... any key starting with '_' is a comment, ignored ..."
    }

`"all"` is special: its value is a list of OTHER group names (not model
names) to union together, in order, de-duplicated.
"""
from __future__ import annotations

import json

from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("model_usage")


def load_model_usage() -> tuple[dict, str | None]:
    """Returns `(model_usage_dict, error_message_or_None)`. Never raises --
    a broken/missing file just means every mode below resolves to 0 models,
    surfaced to the webapp via `error_message` (see `app/api/routes.py::models`,
    which the Generate page shows prominently instead of a silent failure)."""
    path = get_settings().model_usage_file
    if not path.exists():
        msg = f"config/model_usage.json not found at {path}"
        logger.error(msg)
        return {}, msg
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not parse config/model_usage.json")
        return {}, f"{type(exc).__name__}: {exc}"

    # Strip "_note..." style comment keys.
    cleaned = {k: v for k, v in raw.items() if not k.startswith("_")}
    return cleaned, None


def resolve_models(mode: str, custom_models: list[str] | None, model_usage: dict | None = None) -> list[str]:
    if model_usage is None:
        model_usage, _ = load_model_usage()

    mode = (mode or "default").strip().lower()

    if mode == "custom":
        return [m for m in (custom_models or []) if m]

    if mode == "all":
        resolved: list[str] = []
        for group_name in model_usage.get("all", []):
            for m in model_usage.get(group_name, []):
                if m not in resolved:
                    resolved.append(m)
        return resolved

    return list(model_usage.get(mode, []))
