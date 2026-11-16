"""Unit tests -- app/core/guidelines.py (REQUEST.md [2] guideline lookup +
`general_guide` recursion), against the project's real `lib/guidelines/guidelines.json`."""
from __future__ import annotations

from app.core.guidelines import resolve_guideline


def test_banner_popup_noti_pulls_in_general_guide():
    g = resolve_guideline("banner_popup_noti")
    assert "banner_popup_noti" in g.entry_names
    assert "banner_general_standard" in g.entry_names
    assert g.missing_entries == []
    assert g.combined_text  # non-empty


def test_fb_post_banner_pulls_in_ocb_logo():
    g = resolve_guideline("fb_post_banner")
    assert "ocb_logo" in g.entry_names


def test_corrected_banner_rules_include_exact_brand_palette_without_forced_2d():
    from app.core.prompt_builder import _prepare_guideline_for_generation

    g = resolve_guideline("banner_popup_noti")
    prepared = _prepare_guideline_for_generation(g.combined_text).lower()

    assert "#F5FFF2" in g.combined_text
    assert "#B7F3B7" in g.combined_text
    assert "#E39717" in g.combined_text
    assert "Select the medium from the active design brief" in g.combined_text
    assert "Apply these stroke weights only to outlines within individual illustrated subjects" in g.combined_text
    assert "flat 2D" not in g.combined_text
    for artifact_word in ("border", "inset panel", "safe-area rectangle", "ruler", "dimension label"):
        assert artifact_word not in prepared


def test_banner_card_uses_its_actual_source_dimensions():
    import json

    from app.settings import get_settings

    data = json.loads(get_settings().guidelines_file.read_text(encoding="utf-8"))
    assert data["banner_card"]["display_size"] == "1035x384"
    assert data["banner_card"]["ratio"] == "345:128"


def test_every_documented_guideline_source_resolves():
    import json

    from app.settings import get_settings

    path = get_settings().guidelines_file
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) >= 17
    for entry in data.values():
        for source in ([entry.get("source")] if isinstance(entry.get("source"), str) else entry.get("source", [])):
            if source:
                assert (path.parent / source.lstrip("/")).is_file(), source


def test_mkt_banner_card_passport_resolves_font_reference_images():
    """2026-09-08 (user request): guideline entries may list `"font"` --
    typeface/lettering reference image(s) -- resolved into their own
    `font_image_paths` list (kept separate from `image_paths`, whose
    internal spec screenshots are never attached to the image model)."""
    g = resolve_guideline("mkt_banner_card_passport_1024x440")
    assert "mkt_banner_card_passport_1024x440" in g.entry_names
    assert g.missing_entries == []
    assert len(g.font_image_paths) == 2
    names = {p.name for p in g.font_image_paths}
    assert names == {"manrope_medium.png", "manrope_semibold.png"}
    for p in g.font_image_paths:
        assert p.is_file()


def test_format_type_without_font_field_has_empty_font_image_paths():
    g = resolve_guideline("banner_popup_noti")
    assert g.font_image_paths == []


def test_every_documented_guideline_font_resolves():
    import json

    from app.settings import get_settings

    path = get_settings().guidelines_file
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data.values():
        fonts = entry.get("font")
        for font in ([fonts] if isinstance(fonts, str) else fonts or []):
            assert (path.parent / font.lstrip("/")).is_file(), font


def test_unknown_format_type_reports_missing_not_crash():
    g = resolve_guideline("this_format_type_does_not_exist")
    assert g.entry_names == []
    assert g.missing_entries == ["this_format_type_does_not_exist"]
    assert g.combined_text == ""


def test_empty_format_type_returns_empty_result():
    g = resolve_guideline("")
    assert g.entry_names == []
    assert g.missing_entries == []


def test_general_guide_cycle_does_not_infinite_loop(monkeypatch, tmp_path):
    """2 entries pointing `general_guide` at each other must resolve once
    each, not hang -- guards `resolve_guideline`'s cycle-safety."""
    import json

    import app.core.guidelines as guidelines_module

    cyclic = {
        "a": {"content": "A", "general_guide": ["b"]},
        "b": {"content": "B", "general_guide": ["a"]},
    }
    fixture = tmp_path / "guidelines.json"
    fixture.write_text(json.dumps(cyclic), encoding="utf-8")

    class _FakeSettings:
        guidelines_file = fixture

    monkeypatch.setattr(guidelines_module, "get_settings", lambda: _FakeSettings())

    g = resolve_guideline("a")
    assert set(g.entry_names) == {"a", "b"}
    assert "A" in g.combined_text and "B" in g.combined_text
