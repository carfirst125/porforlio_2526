"""Unit tests -- app/core/model_usage.py (REQUEST.md [4] mode resolution)."""
from __future__ import annotations

from app.core.model_usage import load_model_usage, resolve_models


def test_load_real_model_usage_file_has_no_error():
    usage, error = load_model_usage()
    assert error is None
    assert "default" in usage
    # keys starting with "_" (comment keys) must be stripped
    assert all(not k.startswith("_") for k in usage)


def test_resolve_default_mode():
    usage = {"default": ["model-a"], "quality": ["model-b"], "eco": ["model-a"], "custom": [], "all": ["quality", "eco"]}
    assert resolve_models("default", None, usage) == ["model-a"]


def test_resolve_all_mode_unions_and_dedupes_in_order():
    usage = {"default": ["a"], "quality": ["b", "a"], "eco": ["a", "c"], "custom": [], "all": ["quality", "eco"]}
    assert resolve_models("all", None, usage) == ["b", "a", "c"]


def test_resolve_custom_mode_uses_provided_list():
    usage = {"default": [], "quality": [], "eco": [], "custom": [], "all": []}
    assert resolve_models("custom", ["x", "y"], usage) == ["x", "y"]


def test_resolve_custom_mode_ignores_usage_json_custom_key():
    usage = {"default": [], "quality": [], "eco": [], "custom": ["should-be-ignored"], "all": []}
    assert resolve_models("custom", ["x"], usage) == ["x"]


def test_resolve_unknown_mode_returns_empty():
    usage = {"default": ["a"], "quality": [], "eco": [], "custom": [], "all": []}
    assert resolve_models("bogus-mode", None, usage) == []


class _FakeSettingsWithPath:
    """`model_usage_file` is a read-only computed property on the real
    `Settings` -- can't be monkeypatched on the instance, so these 2 tests
    swap `app.core.model_usage.get_settings` itself for a tiny stub exposing
    just the 1 attribute that function reads."""

    def __init__(self, model_usage_file):
        self.model_usage_file = model_usage_file


def test_load_model_usage_missing_file_reports_error(monkeypatch, tmp_path):
    import app.core.model_usage as model_usage_module

    fake_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(model_usage_module, "get_settings", lambda: _FakeSettingsWithPath(fake_path))
    usage, error = load_model_usage()
    assert usage == {}
    assert error is not None and "not found" in error


def test_load_model_usage_broken_json_reports_error(monkeypatch, tmp_path):
    import app.core.model_usage as model_usage_module

    bad_file = tmp_path / "model_usage.json"
    bad_file.write_text("{not valid json,,,", encoding="utf-8")
    monkeypatch.setattr(model_usage_module, "get_settings", lambda: _FakeSettingsWithPath(bad_file))
    usage, error = load_model_usage()
    assert usage == {}
    assert error is not None
