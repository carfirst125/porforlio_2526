"""Unit tests -- app/models/schemas.py. No network, no filesystem, no
external service -- pure function/model behavior only (see
3-project-evaluation.md's "unit" tier definition).

2026-08-20: rewritten for the corrected request.json schema (nested
`generation.canvas`/`generation.safe_area` + `output.size` as a
`{width, height}` object) -- see `app/models/schemas.py` module docstring
and `4-project-evaluation-results.md` for why the previous flat
`design_options.protect_area` schema was wrong and has been replaced."""
from __future__ import annotations

import pytest

from app.models.schemas import CreativeRequest, parse_wh, wh_slug


def test_parse_wh_comma_form():
    assert parse_wh("1035,420") == (1035, 420)


def test_parse_wh_x_form_and_whitespace():
    assert parse_wh(" 640 x 960 ") == (640, 960)


def test_parse_wh_empty_raises():
    with pytest.raises(ValueError):
        parse_wh("")


def test_parse_wh_malformed_raises():
    with pytest.raises(ValueError):
        parse_wh("1035")


def test_wh_slug():
    assert wh_slug(1035, 420) == "1035x420"


def _base_request(**overrides) -> dict:
    base = {
        "creative_id": "TEST_ID",
        "format_type": "banner_popup_noti",
        "design_brief": "test brief",
        "buckets": {
            "theme": "Bucket_1/a.png",
            "context": [],
            "product": "Bucket_3/b.png",
            "brand_text": "Bucket_4/brand.png",
        },
        "text": {"headline": "H", "message": "M", "cta_label": "C", "cta_deep_link": "d"},
        "design_options": {"generate_person": False, "has_logo": True},
        "num_version": 3,
        "generation": {
            "canvas": {"width": 1200, "height": 800},
            "safe_area": {"width": 1035, "height": 420, "position": "center", "x": 82, "y": 190},
            "instruction": "Keep everything inside the safe area.",
        },
        "output": {
            "size": {"width": 1035, "height": 420},
            "crop": {"source": "generation.canvas", "area": {"x": 82, "y": 190, "width": 1035, "height": 420}},
            "clone": [],
        },
    }
    base.update(overrides)
    return base


def test_buckets_normalize_string_to_list():
    req = CreativeRequest.model_validate(_base_request())
    assert req.buckets.theme == ["Bucket_1/a.png"]
    assert req.buckets.context == []
    assert req.buckets.product == ["Bucket_3/b.png"]
    assert req.buckets.brand_text == ["Bucket_4/brand.png"]


def test_buckets_accept_list_of_multiple_images():
    req = CreativeRequest.model_validate(_base_request(buckets={"theme": ["a.png", "b.png"], "context": [], "product": "c.png"}))
    assert req.buckets.theme == ["a.png", "b.png"]


def test_canvas_and_final_output_with_explicit_crop():
    req = CreativeRequest.model_validate(_base_request())
    assert req.canvas_wh() == (1200, 800)
    assert req.final_output_wh() == (1035, 420)  # output.size -- the real pixels written to disk
    assert req.has_crop() is True
    assert req.crop_box() == (82, 190, 1035, 420)
    assert req.display_size_slug() == "1035x420"


def test_crop_box_falls_back_to_safe_area_when_output_crop_omitted():
    """`output.crop` is redundant with `generation.safe_area` in every real
    file seen so far -- a request.json is allowed to omit it."""
    raw = _base_request()
    del raw["output"]["crop"]
    req = CreativeRequest.model_validate(raw)
    assert req.crop_box() == (82, 190, 1035, 420)  # == generation.safe_area's own box


def test_no_crop_when_canvas_equals_output_size():
    """A square banner with no background bleed: canvas == safe_area == output.size."""
    req = CreativeRequest.model_validate(
        _base_request(
            generation={
                "canvas": {"width": 1200, "height": 1200},
                "safe_area": {"width": 1200, "height": 1200, "position": "center", "x": 0, "y": 0},
                "instruction": "",
            },
            output={"size": {"width": 1200, "height": 1200}, "clone": []},
        )
    )
    assert req.canvas_wh() == (1200, 1200)
    assert req.final_output_wh() == (1200, 1200)
    assert req.has_crop() is False
    assert req.display_size_slug() == "1200x1200"


def test_missing_output_size_rejected():
    bad = _base_request()
    del bad["output"]["size"]
    with pytest.raises(Exception):
        CreativeRequest.model_validate(bad)


def test_missing_generation_rejected():
    bad = _base_request()
    del bad["generation"]
    with pytest.raises(Exception):
        CreativeRequest.model_validate(bad)


def test_stray_legacy_protect_area_field_is_ignored_not_crashed():
    """A leftover `design_options.protect_area` from the old (wrong) schema
    must never crash validation -- it's simply ignored (`extra='ignore'`)."""
    raw = _base_request()
    raw["design_options"]["protect_area"] = "1035,420"
    req = CreativeRequest.model_validate(raw)
    assert not hasattr(req.design_options, "protect_area")
    assert req.final_output_wh() == (1035, 420)  # still comes from output.size, not the stray field


def test_text_accepts_explicit_display_line_lists_without_changing_string_consumers():
    req = CreativeRequest.model_validate(
        _base_request(text={"headline": ["Dòng một", "Dòng hai"], "message": ["Lợi ích"], "cta_label": "Mở ngay"})
    )
    assert req.text.headline == "Dòng một\nDòng hai"
    assert req.text.message == "Lợi ích"


def test_new_design_options_are_opt_in():
    req = CreativeRequest.model_validate(_base_request())
    assert req.design_options.generate_text is False
    assert req.design_options.generate_text_headline is False
    assert req.design_options.generate_text_message is False
    assert req.design_options.generate_text_cta_label is False
    assert req.design_options.num_decorative_elements is None
    assert req.design_options.technique_selection.mode == "manual"
    assert req.design_options.technique_selection.manual_techniques == []
    assert req.design_options.technique_selection.auto_config.max_background_techniques == 3
    assert req.design_options.technique_selection.auto_config.max_product_techniques == 3
    assert req.design_options.technique_selection.auto_config.max_decoration_techniques == 3
    assert req.design_options.technique_selection.auto_config.selection_constraints == []
    assert req.design_options.techniques == []


def test_design_techniques_accept_explicit_keys():
    raw = _base_request()
    raw["design_options"]["techniques"] = ["radial_spotlight", "textured_surface"]
    req = CreativeRequest.model_validate(raw)
    assert req.design_options.techniques == ["radial_spotlight", "textured_surface"]


def test_technique_selection_mode_accepts_auto_and_rejects_unknown_value():
    """2026-09-09: `technique_selection_mode` + flat `techniques` was replaced
    by the nested `design_options.technique_selection` object -- see
    `TechniqueSelectionConfig` in app/models/schemas.py."""
    raw = _base_request()
    raw["design_options"]["technique_selection"] = {"mode": "auto"}
    assert CreativeRequest.model_validate(raw).design_options.technique_selection.mode == "auto"

    raw["design_options"]["technique_selection"] = {"mode": "sometimes"}
    with pytest.raises(Exception):
        CreativeRequest.model_validate(raw)


def test_technique_selection_manual_mode_uses_manual_techniques():
    raw = _base_request()
    raw["design_options"]["technique_selection"] = {
        "mode": "manual",
        "manual_techniques": ["radial_spotlight", "textured_surface"],
    }
    cfg = CreativeRequest.model_validate(raw).design_options.technique_selection
    assert cfg.mode == "manual"
    assert cfg.manual_techniques == ["radial_spotlight", "textured_surface"]


def test_technique_selection_auto_config_per_group_max_and_constraints():
    raw = _base_request()
    raw["design_options"]["technique_selection"] = {
        "mode": "auto",
        "auto_config": {
            "max_background_techniques": 2,
            "max_product_techniques": 1,
            "max_decoration_techniques": 0,
            "selection_constraints": ["prompts/mkt_techniques_sel.txt", "/prompts/other.txt"],
        },
    }
    cfg = CreativeRequest.model_validate(raw).design_options.technique_selection
    assert cfg.mode == "auto"
    assert cfg.auto_config.max_background_techniques == 2
    assert cfg.auto_config.max_product_techniques == 1
    assert cfg.auto_config.max_decoration_techniques == 0
    assert cfg.auto_config.selection_constraints == ["prompts/mkt_techniques_sel.txt", "/prompts/other.txt"]


def test_technique_selection_manual_techniques_accepts_single_string():
    raw = _base_request()
    raw["design_options"]["technique_selection"] = {"mode": "manual", "manual_techniques": "radial_spotlight"}
    cfg = CreativeRequest.model_validate(raw).design_options.technique_selection
    assert cfg.manual_techniques == ["radial_spotlight"]


def test_num_decorative_elements_rejects_negative_values():
    raw = _base_request()
    raw["design_options"]["num_decorative_elements"] = -1
    with pytest.raises(Exception):
        CreativeRequest.model_validate(raw)


def test_audience_defaults_to_no_persona():
    req = CreativeRequest.model_validate(_base_request())
    assert req.audience.persona is None


def test_audience_blank_persona_normalizes_to_none():
    raw = _base_request()
    raw["audience"] = {"persona": "  "}
    assert CreativeRequest.model_validate(raw).audience.persona is None


def test_audience_persona_is_optional_string():
    raw = _base_request()
    raw["audience"] = {"persona": "digital_shopaholic"}
    assert CreativeRequest.model_validate(raw).audience.persona == "digital_shopaholic"


def test_generate_request_body_persona_override_defaults_none():
    from app.models.schemas import GenerateRequestBody

    body = GenerateRequestBody(request_name="DB_AUTOBILL_1024x440_MASS")
    assert body.persona_override is None
