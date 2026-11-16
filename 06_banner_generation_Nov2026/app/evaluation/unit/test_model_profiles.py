"""Unit tests -- `app/core/model_profiles.py` (2026-09-06 user request: cost
tracking uses `config/model_profiles.json` -- the consolidated ratio+pricing
file -- as its pricing source). Exercises catalog loading against the REAL
project file, alias resolution for every registry model name, and every
pricing shape `estimate_cost()` currently handles.
"""
from __future__ import annotations

from app.core import model_profiles


def test_load_catalog_loads_real_project_file():
    catalog = model_profiles.load_catalog()
    assert catalog.currency == "USD"
    assert catalog.as_of == "2026-09"
    # Every model documented in config/model_profiles.json as of this test.
    assert set(catalog.models) >= {
        "gemini-3.1-flash", "gemini-3-pro", "gpt-image-2",
        "MAI-2.5-flash", "MAI-2.5-pro", "FLUX.2-flex", "FLUX.2-pro",
        "gpt-5-mini",
    }


class _StubSettings:
    """`Settings.model_profiles_file` is a read-only computed property (like
    every other `*_file` property in `app/settings.py`) -- it can't be
    monkeypatched via `setattr()` on the real pydantic model, so these 2
    tests stub out `model_profiles.get_settings()` itself instead."""

    def __init__(self, path):
        self.model_profiles_file = path


def test_load_catalog_missing_file_soft_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(model_profiles, "get_settings", lambda: _StubSettings(tmp_path / "does_not_exist.json"))
    catalog = model_profiles.load_catalog()
    assert catalog.models == {}
    assert catalog.currency == "USD"  # safe default, never raises


def test_load_catalog_malformed_json_soft_fails(monkeypatch, tmp_path):
    bad = tmp_path / "model_profiles.json"
    bad.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(model_profiles, "get_settings", lambda: _StubSettings(bad))
    catalog = model_profiles.load_catalog()
    assert catalog.models == {}


def test_alias_resolution_for_every_registry_image_model():
    """Every image model name used elsewhere in app/ (request.json,
    app/providers/registry.py, config/model_usage.json) must resolve to
    SOME profile via `_MODEL_ALIASES`, except the one documented gap: the
    bare `mai-image-2.5` deployment has no profile of its own yet (only its
    -flash/-pro variants do)."""
    catalog = model_profiles.load_catalog()
    expected_none = {"mai-image-2.5"}
    for registry_name in model_profiles._MODEL_ALIASES:
        profile = catalog.get(registry_name)
        if registry_name in expected_none:
            assert profile is None, f"{registry_name!r} was expected to have NO profile (documented gap)"
        else:
            assert profile is not None, f"{registry_name!r} should resolve to a profile via alias"


def test_get_profile_direct_key_no_alias_needed():
    catalog = model_profiles.load_catalog()
    assert catalog.get("gpt-image-2") is not None
    assert catalog.get("gpt-5-mini") is not None


def test_estimate_cost_unknown_model_is_unpriced():
    result = model_profiles.estimate_cost("totally-unknown-model-xyz", input_tokens=100, output_tokens=50)
    assert result.priced is False
    assert result.total_cost == 0.0
    assert result.input_rate is None and result.output_rate is None
    assert "no pricing profile" in result.notes


def test_estimate_cost_mai_image_25_bare_deployment_is_unpriced():
    """Documented gap: config/model_profiles.json has no entry for the bare
    `mai-image-2.5` deployment, only `-flash`/`-pro`."""
    result = model_profiles.estimate_cost("mai-image-2.5", input_tokens=100, output_tokens=50)
    assert result.priced is False
    assert result.total_cost == 0.0


def test_estimate_cost_plain_llm_shape():
    """gpt-5-mini: input_per_1m_tokens_usd + output_per_1m_tokens_usd."""
    result = model_profiles.estimate_cost("gpt-5-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert result.priced is True
    assert result.input_rate == 0.25
    assert result.output_rate == 2.00
    assert result.input_cost == 0.25
    assert result.output_cost == 2.00
    assert result.total_cost == 2.25
    assert result.currency == "USD"


def test_estimate_cost_flux_per_megapixel_shape_ignores_token_counts():
    """FLUX.2-pro: per_mp_usd only, no token pricing at all -- billed purely
    from width/height regardless of what token counts are passed in (they
    should be entirely ignored)."""
    result = model_profiles.estimate_cost(
        "flux.2-pro", input_tokens=999_999, output_tokens=999_999, width=1000, height=1000,
    )
    assert result.priced is True
    assert result.input_rate is None
    assert result.output_rate == 0.03
    assert result.input_cost == 0.0
    assert abs(result.total_cost - 0.03) < 1e-9  # exactly 1.0 MP * $0.03/MP


def test_estimate_cost_flux_without_dimensions_is_unpriced():
    """No width/height means the per-megapixel shape can't be computed --
    must degrade to unpriced rather than silently charging $0."""
    result = model_profiles.estimate_cost("flux.2-pro", input_tokens=100, output_tokens=50)
    assert result.priced is False
    assert result.total_cost == 0.0


def test_estimate_cost_gemini_3_pro_image_output_token_shape():
    """gemini-3-pro: input_per_1m_tokens_usd + output_per_1m_image_tokens_usd
    (note the reversed word order vs. gpt-image-2's key -- this is the shape
    that must NOT be confused with the split text/image shape below)."""
    result = model_profiles.estimate_cost("gemini-3-pro-image", input_tokens=1_000_000, output_tokens=1_000_000)
    assert result.priced is True
    assert result.input_rate == 2.00
    assert result.output_rate == 12.00
    assert result.input_cost == 2.00
    assert result.output_cost == 12.00
    assert result.total_cost == 14.00


def test_estimate_cost_split_text_image_shape_with_breakdown():
    """gpt-image-2: text_input_per_1m_tokens_usd + image_input_per_1m_tokens_usd
    + image_output_per_1m_tokens_usd, using the provider's own breakdown when
    available (input_text_tokens/input_image_tokens/output_image_tokens)."""
    result = model_profiles.estimate_cost(
        "gpt-image-2",
        input_text_tokens=1_000_000,
        input_image_tokens=1_000_000,
        output_image_tokens=1_000_000,
    )
    assert result.priced is True
    # input_cost = 1M text @ $5 + 1M image @ $8 = $13
    assert abs(result.input_cost - 13.0) < 1e-9
    assert result.output_cost == 30.0
    assert abs(result.total_cost - 43.0) < 1e-9


def test_estimate_cost_split_text_image_shape_without_breakdown_falls_back_to_combined():
    """When the provider's usage payload only reported combined
    input_tokens/output_tokens (no text/image breakdown), the split shape
    still produces a cost using whichever per-image rate is defined."""
    result = model_profiles.estimate_cost("gpt-image-2", input_tokens=1_000_000, output_tokens=1_000_000)
    assert result.priced is True
    assert result.output_cost == 30.0
    assert result.total_cost > 0


def test_estimate_cost_returns_catalog_currency():
    catalog = model_profiles.load_catalog()
    result = model_profiles.estimate_cost("gpt-5-mini", input_tokens=0, output_tokens=0, catalog=catalog)
    assert result.currency == catalog.currency == "USD"
