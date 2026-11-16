"""Load `config/model_profiles.json` -- the single consolidated source for
every GenAI model's supported aspect ratios AND its input/output token
pricing (2026-09-06, user request: this file replaces the old, pricing-less
`config/model_ratio.json`, which nothing in `app/` actually read as JSON --
its ratio table had been hard-coded directly in
`app/core/canvas_planner.py::GEMINI_31_FLASH_IMAGE_RATIOS` instead).

`model_profiles.json` shape:

    {
      "models": {
        "<profile_key>": {
          "type": "llm" | "image",
          "provider": "...", "service": "...",
          "pricing": {... one of several shapes, see `estimate_cost()` ...},
          "supported_aspect_ratios": ["1:1", ...],
          "notes": "..."
        },
        ...
      },
      "metadata": {"currency": "USD", ...}
    }

`app/core/usage_tracker.py` is the only other module that should import this
one -- it calls `estimate_cost()` once per recorded GenAI call to turn raw
token counts (or image pixel dimensions, for the megapixel-billed FLUX
models) into a cost in `metadata.currency`.

Model-name mismatch note: every OTHER part of `app/` (request.json,
`app/providers/registry.py`, `config/model_usage.json`) identifies an image
model by its REGISTRY name (e.g. `"gemini-3.1-flash-image"`,
`"mai-image-2.5-flash"`, `"flux.2-pro"`), while `model_profiles.json` itself
uses the pricing source's own display name (e.g. `"gemini-3.1-flash"`,
`"MAI-2.5-flash"`, `"FLUX.2-pro"`). `_MODEL_ALIASES` below bridges the two so
callers can always look a profile up by the SAME registry name used
everywhere else in the app -- no other module needs to know the difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json

from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("model_profiles")

# Registry/runtime model name -> config/model_profiles.json's own key, for
# every case where the two differ. A name already matching a profile key
# exactly (e.g. "gpt-image-2", "gpt-5-mini") needs no entry here.
_MODEL_ALIASES: dict[str, str] = {
    "gemini-3.1-flash-image": "gemini-3.1-flash",
    "gemini-3-pro-image": "gemini-3-pro",
    "mai-image-2.5": "MAI-2.5",  # no profile defined for this exact deployment yet -- see module note in estimate_cost()
    "mai-image-2.5-flash": "MAI-2.5-flash",
    "mai-image-2.5-pro": "MAI-2.5-pro",
    "flux.2-pro": "FLUX.2-pro",
    "flux.2-flex": "FLUX.2-flex",
    "flux.2-flex-pro": "FLUX.2-pro",
}


@dataclass(frozen=True)
class ModelProfile:
    key: str
    type: str = ""
    provider: str = ""
    service: str = ""
    pricing: dict[str, Any] = field(default_factory=dict)
    supported_aspect_ratios: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ModelProfileCatalog:
    models: dict[str, ModelProfile] = field(default_factory=dict)
    currency: str = "USD"
    as_of: str = ""

    def get(self, model_name: str) -> ModelProfile | None:
        """Look up by registry name first, falling back to the alias map."""
        if model_name in self.models:
            return self.models[model_name]
        alias = _MODEL_ALIASES.get(model_name)
        if alias and alias in self.models:
            return self.models[alias]
        return None


def _as_ratio_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item.strip())


def load_catalog() -> ModelProfileCatalog:
    """Never raises -- a missing/broken file just means every `estimate_cost()`
    call reports `priced=False` (soft-fail, same convention as
    `app/core/model_usage.py::load_model_usage()` /
    `app/core/design_techniques.py::load_catalog()`)."""
    path = get_settings().model_profiles_file
    if not path.exists():
        logger.warning("config/model_profiles.json not found at %s", path)
        return ModelProfileCatalog()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("Could not parse config/model_profiles.json at %s", path)
        return ModelProfileCatalog()

    raw_models = document.get("models")
    if not isinstance(raw_models, dict):
        logger.error("config/model_profiles.json has no object-valued 'models' key")
        return ModelProfileCatalog()

    models: dict[str, ModelProfile] = {}
    for key, entry in raw_models.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            logger.warning("Skipping malformed model_profiles.json entry %r", key)
            continue
        pricing = entry.get("pricing")
        models[key] = ModelProfile(
            key=key,
            type=str(entry.get("type") or ""),
            provider=str(entry.get("provider") or ""),
            service=str(entry.get("service") or ""),
            pricing=pricing if isinstance(pricing, dict) else {},
            supported_aspect_ratios=_as_ratio_tuple(entry.get("supported_aspect_ratios")),
            notes=str(entry.get("notes") or ""),
        )

    metadata = document.get("metadata")
    currency = "USD"
    as_of = ""
    if isinstance(metadata, dict):
        currency = str(metadata.get("currency") or "USD")
        as_of = str(metadata.get("as_of") or "")

    return ModelProfileCatalog(models=models, currency=currency, as_of=as_of)


def get_profile(model_name: str, catalog: ModelProfileCatalog | None = None) -> ModelProfile | None:
    catalog = catalog if catalog is not None else load_catalog()
    return catalog.get(model_name)


@dataclass(frozen=True)
class CostEstimate:
    input_rate: float | None
    output_rate: float | None
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str
    priced: bool
    notes: str = ""


def estimate_cost(
    model_name: str,
    *,
    input_tokens: int | None = 0,
    output_tokens: int | None = 0,
    input_text_tokens: int | None = None,
    input_image_tokens: int | None = None,
    output_image_tokens: int | None = None,
    width: int | None = None,
    height: int | None = None,
    catalog: ModelProfileCatalog | None = None,
) -> CostEstimate:
    """Turn raw usage into a cost, handling every pricing shape currently in
    `config/model_profiles.json`:

    1. Per-megapixel (FLUX.2-*: `per_mp_usd`, no token pricing at all) --
       needs `width`/`height`, ignores every token count.
    2. Split text/image token pricing (`gpt-image-2`, `MAI-2.5-*`):
       `text_input_per_1m_tokens_usd` + `image_input_per_1m_tokens_usd` +
       `image_output_per_1m_tokens_usd`. Uses the `input_text_tokens`/
       `input_image_tokens`/`output_image_tokens` breakdown when the caller
       has it (from the provider's own usage payload); otherwise charges the
       single combined `input_tokens`/`output_tokens` count against whichever
       one of those rates is defined.
    3. Image-output token pricing (`gemini-3-pro`): `input_per_1m_tokens_usd`
       + `output_per_1m_image_tokens_usd`.
    4. Plain LLM token pricing (`gpt-5-mini`, `gemini-3.1-flash`, ...):
       `input_per_1m_tokens_usd` + `output_per_1m_tokens_usd`.

    Returns `priced=False` (all costs 0.0) when no profile is found for
    `model_name`, or its `pricing` object matches none of the shapes above --
    this is a normal, expected outcome (e.g. `config/model_profiles.json` has
    no entry for the bare `mai-image-2.5` deployment, only its `-flash`/`-pro`
    variants) and callers must never treat it as an error.
    """
    catalog = catalog if catalog is not None else load_catalog()
    profile = catalog.get(model_name)
    if profile is None:
        return CostEstimate(None, None, 0.0, 0.0, 0.0, catalog.currency, False, f"no pricing profile found for model {model_name!r}")

    pricing = profile.pricing

    per_mp = pricing.get("per_mp_usd")
    if per_mp is not None and width and height:
        megapixels = (width * height) / 1_000_000
        rate = float(per_mp)
        total = megapixels * rate
        return CostEstimate(None, rate, 0.0, total, total, catalog.currency, True, f"billed per megapixel ({megapixels:.3f} MP)")

    image_output_rate = pricing.get("image_output_per_1m_tokens_usd")
    if image_output_rate is not None:
        image_input_rate = pricing.get("image_input_per_1m_tokens_usd")
        text_input_rate = pricing.get("text_input_per_1m_tokens_usd")
        input_cost = 0.0
        if input_text_tokens is not None or input_image_tokens is not None:
            input_cost += (input_text_tokens or 0) / 1_000_000 * float(text_input_rate or 0)
            input_cost += (input_image_tokens or 0) / 1_000_000 * float(image_input_rate or 0)
            effective_input_rate = image_input_rate if image_input_rate is not None else text_input_rate
        else:
            effective_input_rate = image_input_rate if image_input_rate is not None else text_input_rate
            input_cost = (input_tokens or 0) / 1_000_000 * float(effective_input_rate or 0)
        out_tokens = output_image_tokens if output_image_tokens is not None else output_tokens
        output_cost = (out_tokens or 0) / 1_000_000 * float(image_output_rate)
        total = input_cost + output_cost
        return CostEstimate(effective_input_rate, image_output_rate, input_cost, output_cost, total, catalog.currency, True)

    output_image_token_rate = pricing.get("output_per_1m_image_tokens_usd")
    if output_image_token_rate is not None:
        input_rate = pricing.get("input_per_1m_tokens_usd")
        input_cost = (input_tokens or 0) / 1_000_000 * float(input_rate or 0)
        output_cost = (output_tokens or 0) / 1_000_000 * float(output_image_token_rate)
        total = input_cost + output_cost
        return CostEstimate(input_rate, output_image_token_rate, input_cost, output_cost, total, catalog.currency, True)

    input_rate = pricing.get("input_per_1m_tokens_usd")
    output_rate = pricing.get("output_per_1m_tokens_usd")
    if input_rate is not None and output_rate is not None:
        input_cost = (input_tokens or 0) / 1_000_000 * float(input_rate)
        output_cost = (output_tokens or 0) / 1_000_000 * float(output_rate)
        total = input_cost + output_cost
        return CostEstimate(input_rate, output_rate, input_cost, output_cost, total, catalog.currency, True)

    return CostEstimate(
        None, None, 0.0, 0.0, 0.0, catalog.currency, False,
        f"unrecognized pricing shape for model {model_name!r}: keys={sorted(pricing)}",
    )


# ---------------------------------------------------------------------------
# Chạy độc lập file này (in mọi model profile đã load được, hữu ích để kiểm
# tra model_profiles.json parse đúng và alias resolve đúng trước khi chạy cả
# app):
#   python -m app.core.model_profiles
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    catalog = load_catalog()
    print(f"currency={catalog.currency!r} as_of={catalog.as_of!r} models={len(catalog.models)}")
    for name in sorted(catalog.models):
        profile = catalog.models[name]
        print(f"- {name}: type={profile.type} pricing={profile.pricing}")
    print()
    print("Alias resolution check (registry name -> profile):")
    for registry_name in _MODEL_ALIASES:
        profile = catalog.get(registry_name)
        print(f"- {registry_name!r} -> {profile.key if profile else None!r}")
