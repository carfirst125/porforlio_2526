"""Unit tests -- app/core/prompt_builder.py helpers that don't need the full
LangGraph pipeline (the full `build_prompt()` assembly is already exercised
end-to-end by the real request.json in app/evaluation/smoke/)."""
from __future__ import annotations

import json
from pathlib import Path

from app.core.image_prep import PreparedImage
from app.core.prompt_builder import (
    _db_text_color_lock,
    _human_realism_instruction,
    _person_instruction,
    _visible_copy_whitelist_instruction,
    format_attached_images_note,
)


def test_db_mass_text_color_lock_is_explicit_and_forbids_black():
    lock = _db_text_color_lock("ocb_omni_110926_mass_aff_1024x440")
    assert "#00804B" in lock
    assert "#E39717" in lock
    assert "#545C57" in lock
    assert "Never render the headline or message in black" in lock
    assert "Never paint a hex token, typeface name, or font weight as letters" in lock


def test_visible_copy_whitelist_forbids_font_and_hex_spec_as_copy():
    text = _visible_copy_whitelist_instruction()
    assert "VISIBLE COPY WHITELIST" in text
    assert "Manrope" in text
    assert "dE39717" in text
    assert "gọn nhẹ E39717" in text
    assert "Do not add any extra line of copy" in text


def test_build_prompt_includes_visible_copy_whitelist(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    built = prompt_builder_module.build_prompt(
        "REQ_VISIBLE_COPY",
        _minimal_req(),
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )
    assert built.text.count("VISIBLE COPY WHITELIST") == 1
    assert "dE39717" in built.text

    data = _minimal_req().model_dump()
    data["format_type"] = "ocb_omni_110926_mass_1024x440"
    db_built = prompt_builder_module.build_prompt(
        "REQ_VISIBLE_COPY_DB",
        CreativeRequest.model_validate(data),
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )
    assert db_built.text.count("VISIBLE COPY WHITELIST") == 2


def test_db_aff_text_color_lock_uses_approved_gold_palette():
    lock = _db_text_color_lock("ocb_omni_110926_aff_1024x440")
    assert all(color in lock for color in ("#C5A049", "#DDC37D", "#E4DABF", "#FCF3C5"))
    assert "Never render the headline or message in black" in lock


def test_db_popup_aff_color_lock_uses_approved_gold_and_pill_gradient():
    lock = _db_text_color_lock("ocb_omni_popup_aff_720x1280")
    assert all(color in lock for color in ("#C5A049", "#DDC37D", "#E4DABF", "#FAF0D7", "#E1C364", "#D2A046", "#553C00"))
    assert "pill-shaped" in lock


def test_db_popup_mass_color_lock_uses_approved_green_and_orange_palette():
    lock = _db_text_color_lock("ocb_omni_popup_mass_720x1280")
    assert all(color in lock for color in ("#E39717", "#0A9646", "#545C57", "#008C55", "#00804B", "#FFFFFF"))
    assert "pill-shaped" in lock


def test_build_prompt_no_person_omits_hand_holding_anatomy(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    data = _minimal_req().model_dump()
    data["design_options"]["generate_person"] = False
    built = prompt_builder_module.build_prompt(
        "REQ_NO_HANDS",
        CreativeRequest.model_validate(data),
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )

    assert "NO PERSON / NO HANDS" in built.text
    assert "HUMAN REALISM AND ANATOMY" not in built.text
    assert "fingers must wrap around the phone" not in built.text
    assert "do not invent a floating or bodiless hand" in built.text.lower() or "Do not invent a floating or bodiless hand" in built.text
    assert "do not invent any new person, character, hand, arm, wrist" in built.text


def test_build_prompt_with_person_keeps_anatomy_block(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    data = _minimal_req().model_dump()
    data["design_options"]["generate_person"] = True
    built = prompt_builder_module.build_prompt(
        "REQ_WITH_PERSON_ANATOMY",
        CreativeRequest.model_validate(data),
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )

    assert "HUMAN REALISM AND ANATOMY" in built.text
    assert "NO PERSON / NO HANDS" not in built.text
    instruction = _human_realism_instruction()

    assert "look straight at the viewer/camera" in instruction
    assert "eyes must stay directed at the viewer" in instruction
    assert "never down at the product" in instruction
    assert "their eyes must be looking at that same phone/screen" not in instruction


def test_ai_generated_person_is_vietnamese_happy_and_campaign_appropriate():
    data = _minimal_req().model_dump()
    data["design_options"]["generate_person"] = True
    req = type(_minimal_req()).model_validate(data)

    instruction = _person_instruction(req, has_context_assets=False)

    assert "naturally Vietnamese-looking person" in instruction
    assert "choose their age" in instruction
    assert "refined context-appropriate clothing" in instruction
    assert "genuine happiness and enthusiasm about this program" in instruction

    described_character = _person_instruction(
        _minimal_req(), has_context_assets=False, context_texts=["A young digital-banking customer"]
    )
    assert "naturally Vietnamese-looking person" in described_character


def test_generated_person_rule_does_not_override_locked_context_identity():
    instruction = _person_instruction(_minimal_req(), has_context_assets=True)

    assert "LOCKED CONTEXT ASSET" in instruction
    assert "naturally Vietnamese-looking person" not in instruction


def _minimal_req():
    """Smallest valid `CreativeRequest` for exercising `build_prompt()`
    directly, without going through the full LangGraph pipeline."""
    from app.models.schemas import CreativeRequest

    return CreativeRequest.model_validate(
        {
            "creative_id": "UNIT_TEST_REQ",
            "format_type": "does_not_exist_in_guidelines_json",
            "design_brief": "Unit test brief.",
            "buckets": {"theme": [], "context": [], "product": []},
            "reference": "",
            "text": {"headline": "H", "message": "M", "cta_label": "", "cta_deep_link": ""},
            "guidelines": "",
            "num_version": 1,
            "design_options": {"generate_person": False, "has_logo": False},
            "generation": {
                "canvas": {"width": 1000, "height": 1000},
                "safe_area": {"width": 1000, "height": 1000, "position": "center", "x": 0, "y": 0},
                "instruction": "",
            },
            "output": {"size": {"width": 1000, "height": 1000}, "clone": []},
        }
    )


def _minimal_req_with_crop():
    """Same as `_minimal_req()` but with a real crop (canvas larger than
    output.size) -- needed to exercise the `has_crop` branches of
    `build_prompt()` (CANVAS & SAFE AREA section, guideline rewrite,
    OVERRIDE fallback, CRITICAL REMINDER)."""
    from app.models.schemas import CreativeRequest

    return CreativeRequest.model_validate(
        {
            "creative_id": "UNIT_TEST_REQ_CROP",
            "format_type": "banner_popup_noti",
            "design_brief": "Unit test brief.",
            "buckets": {"theme": [], "context": [], "product": []},
            "reference": "",
            "text": {"headline": "H", "message": "M", "cta_label": "", "cta_deep_link": ""},
            "guidelines": "",
            "num_version": 1,
            "design_options": {"generate_person": False, "has_logo": False},
            "generation": {
                "canvas": {"width": 1536, "height": 1024},
                "safe_area": {"width": 1035, "height": 420, "position": "center", "x": 250, "y": 302},
                "instruction": "",
            },
            "output": {"size": {"width": 1035, "height": 420}, "clone": []},
        }
    )


def _minimal_req_with_crop_and_product():
    """Same as `_minimal_req_with_crop()` but with a non-empty `buckets.product`
    -- needed to exercise `prompt_bleed_guard`'s wiring in `build_prompt()`,
    which is gated on `has_crop AND req.buckets.product`."""
    from app.models.schemas import CreativeRequest

    return CreativeRequest.model_validate(
        {
            "creative_id": "UNIT_TEST_REQ_CROP_PRODUCT",
            "format_type": "does_not_exist_in_guidelines_json",
            "design_brief": "Unit test brief.",
            "buckets": {"theme": [], "context": [], "product": "some_product.png"},
            "reference": "",
            "text": {"headline": "H", "message": "M", "cta_label": "", "cta_deep_link": ""},
            "guidelines": "",
            "num_version": 1,
            "design_options": {"generate_person": False, "has_logo": False},
            "generation": {
                "canvas": {"width": 1536, "height": 1024},
                "safe_area": {"width": 1035, "height": 420, "position": "center", "x": 250, "y": 302},
                "instruction": "",
            },
            "output": {"size": {"width": 1035, "height": 420}, "clone": []},
        }
    )


def _minimal_req_with_locked_assets():
    from app.models.schemas import CreativeRequest

    data = _minimal_req().model_dump()
    data["buckets"]["context"] = "licensed_person.jpg"
    data["buckets"]["product"] = "approved_card.png"
    return CreativeRequest.model_validate(data)


def _empty_guideline():
    from app.core.guidelines import ResolvedGuideline

    return ResolvedGuideline(format_type="does_not_exist_in_guidelines_json")


def _guideline_with_text(text: str, format_type: str = "banner_popup_noti"):
    from app.core.guidelines import ResolvedGuideline

    return ResolvedGuideline(format_type=format_type, entry_names=[format_type], combined_text=text)


def _pi(label: str, resized_name: str, source_name: str | None = None) -> PreparedImage:
    return PreparedImage(
        label=label,
        source_path=Path(f"/fake/src/{source_name or resized_name}"),
        resized_path=Path(f"/fake/resized/{resized_name}"),
        mime_type="image/png",
    )


def test_format_attached_images_note_empty_list():
    note = format_attached_images_note([])
    assert "none" in note.lower()


def test_build_prompt_locks_supplied_person_and_product_assets(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    built = prompt_builder_module.build_prompt(
        "REQ_LOCKED_ASSETS",
        _minimal_req_with_locked_assets(),
        _empty_guideline(),
        ["context", "product"],
        use_llm_refinement=False,
    )

    assert "LOCKED CONTEXT ASSET" in built.text
    assert "not automatically a person asset" in built.text
    assert "LOCKED PRODUCT ASSET" in built.text
    assert "Do NOT mirror, flip" in built.text
    assert "redraw, cartoonize" in built.text
    assert "A supplied photograph must remain photographic" in built.text
    assert "Never vary or reinterpret" not in built.text  # num_version == 1


def test_build_prompt_places_brand_text_as_locked_text_artwork(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    data = _minimal_req().model_dump()
    data["buckets"]["brand_text"] = ["Bucket_4/FLEXIText02.png"]
    req = CreativeRequest.model_validate(data)

    built = prompt_builder_module.build_prompt(
        "REQ_BRAND_TEXT",
        req,
        _empty_guideline(),
        ["brand_text"],
        use_llm_refinement=False,
    )

    assert "LOCKED BRAND-TEXT ASSET" in built.text
    assert "Place it exactly once inside the designated text area" in built.text
    assert "Do not redraw, retype, recolor, crop, distort, restyle, split apart, or duplicate it" in built.text


def test_build_prompt_treats_theme_as_flexible_background_reference(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    data = _minimal_req_with_locked_assets().model_dump()
    data["buckets"]["theme"] = "campaign_background.jpg"
    built = prompt_builder_module.build_prompt(
        "REQ_THEME_REFERENCE",
        CreativeRequest.model_validate(data),
        _empty_guideline(),
        ["theme", "context", "product"],
        use_llm_refinement=False,
    )

    assert "CREATIVE BACKGROUND REFERENCE" in built.text
    assert "NOT pixel-locked" in built.text
    assert "reinterpret, improve, extend, simplify, or recompose" in built.text
    assert "final headline/message/CTA only from REQUIRED TEXT CONTENT" in built.text
    assert "commercial hybrid with natural depth" in built.text
    assert "flat 2D" not in built.text
    assert "LOCKED CONTEXT ASSET" in built.text
    assert "LOCKED PRODUCT ASSET" in built.text


def test_build_prompt_no_theme_allows_creative_depth_and_grounds_product(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    data = _minimal_req_with_locked_assets().model_dump()
    data["design_options"]["dominant_color"] = "#8FCF4F"
    built = prompt_builder_module.build_prompt(
        "REQ_NO_THEME_CREATIVE",
        CreativeRequest.model_validate(data),
        _empty_guideline(),
        ["context", "product"],
        use_llm_refinement=False,
    )

    assert "original, polished commercial campaign environment" in built.text
    assert "does not require a plain or uniformly colored background" in built.text
    assert "Do not default to a plain gradient with small icons scattered over it" in built.text
    assert "shared lighting, scale, perspective, overlap, rhythm, and negative space" in built.text
    assert "local light/dark contrast" in built.text
    assert "soft contact shadow or reflection" in built.text
    assert "without changing the product artwork" in built.text
    assert "Do NOT invent photographic scenery" not in built.text
    assert "minimal, flat, and 2D" not in built.text


def test_build_prompt_applies_optional_decorative_limit_without_changing_default(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    default_prompt = prompt_builder_module.build_prompt(
        "REQ_DECOR_DEFAULT", _minimal_req(), _empty_guideline(), [], use_llm_refinement=False
    ).text
    data = _minimal_req().model_dump()
    data["design_options"]["num_decorative_elements"] = 2
    limited_prompt = prompt_builder_module.build_prompt(
        "REQ_DECOR_TWO", CreativeRequest.model_validate(data), _empty_guideline(), [], use_llm_refinement=False
    ).text

    assert "DECORATIVE DETAIL LIMIT" not in default_prompt
    assert "no more than 2 purely decorative" in limited_prompt
    assert "maximum, not a target" in limited_prompt
    assert "supplied theme/context/product assets" in limited_prompt


def test_build_prompt_applies_selected_technique_as_subordinate_optional_direction(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    default_prompt = prompt_builder_module.build_prompt(
        "REQ_TECHNIQUE_DEFAULT", _minimal_req(), _empty_guideline(), [], use_llm_refinement=False
    ).text
    data = _minimal_req_with_locked_assets().model_dump()
    data["design_options"]["techniques"] = ["radial_spotlight"]
    selected_prompt = prompt_builder_module.build_prompt(
        "REQ_TECHNIQUE_SELECTED",
        CreativeRequest.model_validate(data),
        _empty_guideline(),
        ["context", "product"],
        use_llm_refinement=False,
    ).text

    assert "DESIGN TECHNIQUES" not in default_prompt
    assert "Radial Spotlight / Halo + Framing" in selected_prompt
    assert "supporting methods, not extra content requirements" in selected_prompt
    assert "never override the request JSON" in selected_prompt
    assert "Never redraw, duplicate, replace, crop, distort" in selected_prompt


def test_guideline_cleanup_removes_safe_area_specs_and_rejects_forced_flat_style():
    from app.core.prompt_builder import _prepare_guideline_for_generation

    source = (
        "3. TITLE\n- Font: Manrope Bold\n\n"
        "4. BIÊN AN TOÀN\n- trái 48px, trên 60px\n\n"
        "5. HÌNH ẢNH\n- Keep campaign imagery\n\n"
        "6. VỊ TRÍ HIỂN THỊ\n- dạng thẻ (card) và popup\n\n"
        "### Guideline: banner_general_standard\n"
        "(4) Phong cách: bắt buộc sử dụng hình vẽ minh họa dạng 2D phẳng (flat design); "
        "nghiêm cấm sử dụng hình ảnh chụp, hình ảnh tả thực hoặc hình ảnh 3D. "
        "(5) Đường viền: quy định 0px, 3px và 6px.\n\n"
        "GEOMETRY\nleft=0px, top=12px"
    )
    cleaned = _prepare_guideline_for_generation(source)

    assert "Font: Manrope Bold" in cleaned
    assert "Keep campaign imagery" in cleaned
    assert "48px" not in cleaned
    assert "60px" not in cleaned
    assert "GEOMETRY" not in cleaned
    assert "dạng thẻ" not in cleaned
    assert "popup" not in cleaned
    assert "chọn chất liệu hình ảnh phù hợp với design brief" in cleaned
    assert "bắt buộc sử dụng hình vẽ minh họa" not in cleaned
    assert "flat design" not in cleaned
    assert "Đường viền" not in cleaned


def test_guideline_cleanup_removes_frame_and_export_production_specs():
    from app.core.prompt_builder import _prepare_guideline_for_generation

    source = (
        "1. FRAME\n- Landscape 1035x420px\n\n"
        "2. EXPORT\n- File size below 100KB\n\n"
        "3. TITLE OR SLOGAN\n- Manrope Bold 57pt\n\n"
        "4. SAFE AREA\n- Keep 48px from all edges\n\n"
        "5. IMAGERY\n- Use a continuous full-bleed background"
    )
    cleaned = _prepare_guideline_for_generation(source)

    assert "1035x420" not in cleaned
    assert "100KB" not in cleaned
    assert "48px" not in cleaned
    assert "Manrope Bold 57pt" in cleaned
    assert "continuous full-bleed background" in cleaned


def test_format_attached_images_note_lists_label_and_resized_filename():
    """2026-08-20 (2) (user request): "note thêm ... tên các bức ảnh nào đã
    được đính kèm gởi image model" -- every attached image's actual on-disk
    filename must appear, not just its generic bucket label."""
    images = [
        _pi("theme", "db_theme_01__123.jpg", "db_theme_01.png"),
        _pi("product", "db_phone__123.png"),  # same name before/after downsample
        _pi("safe_area_guide", "_safe_area_guide_abc123.png"),
    ]
    note = format_attached_images_note(images)

    assert "db_theme_01__123.jpg" in note
    assert "db_phone__123.png" in note
    assert "_safe_area_guide_abc123.png" in note
    # order preserved, numbered 1..N
    assert note.index("1.") < note.index("db_theme_01__123.jpg")
    assert note.index("2.") < note.index("db_phone__123.png")
    assert note.index("3.") < note.index("_safe_area_guide_abc123.png")


def test_format_attached_images_note_shows_original_source_when_downsampled():
    images = [_pi("theme", "db_theme_01__123.jpg", "db_theme_01.png")]
    note = format_attached_images_note(images)
    assert "downsampled from source: db_theme_01.png" in note


def test_format_attached_images_note_omits_source_note_when_names_match():
    """The safe-area-guide image (and anything else with resized_path ==
    source_path) shouldn't get a misleading "downsampled from source: X"
    footnote pointing at itself."""
    images = [_pi("safe_area_guide", "_safe_area_guide_abc123.png")]
    note = format_attached_images_note(images)
    assert "downsampled from source" not in note


def test_write_prompt_log_always_overwrites_stale_content(isolated_output):
    """2026-08-20 (3) (user-reported bug): a re-run after prompt_builder.py's
    template changed still showed the OLD prompt text in the saved
    prompt-<model>-version-<n>.txt log, because the old code skipped writing
    whenever that exact path already existed on disk. The log's whole job
    is to accurately reflect what was actually just sent -- it must be
    overwritten every call, never left stale."""
    from app.core.prompt_builder import write_prompt_log

    path1 = write_prompt_log("REQ", "gpt-image-2", 1, "OLD prompt text (stale, from an earlier run)")
    assert "OLD prompt text" in path1.read_text(encoding="utf-8")

    path2 = write_prompt_log("REQ", "gpt-image-2", 1, "NEW prompt text (fresh, has the safe-area-guide wording)")
    assert path1 == path2
    content = path2.read_text(encoding="utf-8")
    assert "NEW prompt text" in content
    assert "OLD prompt text" not in content  # must be a real overwrite, not an append


def test_write_prompt_log_overwrites_even_with_a_footer(isolated_output):
    from app.core.prompt_builder import write_prompt_log

    path = write_prompt_log("REQ", "gpt-image-2", 1, "first prompt", "ATTACHED IMAGES SENT WITH THIS PROMPT (in order): 1. [theme] old.jpg")
    assert "old.jpg" in path.read_text(encoding="utf-8")

    write_prompt_log("REQ", "gpt-image-2", 1, "second prompt", "ATTACHED IMAGES SENT WITH THIS PROMPT (in order): 1. [theme] new.jpg")
    content = path.read_text(encoding="utf-8")
    assert "new.jpg" in content
    assert "old.jpg" not in content


def test_build_prompt_cache_busts_when_prompt_template_version_changes(isolated_output, monkeypatch):
    """2026-08-20 (3) (user-reported bug): `request.json` unchanged between
    2 runs must NOT be enough, by itself, to guarantee the cached prompt
    text is still up to date -- a CODE-only change to build_prompt()'s
    template (e.g. adding the SAFE AREA GUIDE wording) must also bust the
    cache, via `_PROMPT_TEMPLATE_VERSION`. This simulates exactly that: same
    req/guideline/attachment_labels both times, only the template version
    constant changes in between -- the 2nd call must NOT reuse the 1st
    call's cache entry."""
    import app.core.prompt_builder as prompt_builder_module
    from app.core.output_manager import prompt_cache_file

    req = _minimal_req()
    guideline = _empty_guideline()

    monkeypatch.setattr(prompt_builder_module, "_PROMPT_TEMPLATE_VERSION", 1)
    prompt_builder_module.build_prompt("REQ", req, guideline, [], use_llm_refinement=False)
    cache_after_v1 = json.loads(prompt_cache_file("REQ").read_text(encoding="utf-8"))

    monkeypatch.setattr(prompt_builder_module, "_PROMPT_TEMPLATE_VERSION", 2)
    prompt_builder_module.build_prompt("REQ", req, guideline, [], use_llm_refinement=False)
    cache_after_v2 = json.loads(prompt_cache_file("REQ").read_text(encoding="utf-8"))

    # Same request.json data both times -- if the template version weren't
    # part of the cache key, this would be the SAME cache_key both times
    # (a stale-cache reuse, exactly the bug reported).
    assert cache_after_v1["cache_key"] != cache_after_v2["cache_key"]


def test_build_prompt_cache_busts_when_guideline_text_changes(isolated_output):
    """Editing guidelines.json must take effect without editing request.json."""
    import app.core.prompt_builder as prompt_builder_module
    from app.core.output_manager import prompt_cache_file

    req = _minimal_req()
    first = _guideline_with_text("OLD BRAND RULE")
    second = _guideline_with_text("NEW BRAND RULE")

    prompt_builder_module.build_prompt("REQ_GUIDELINE_CACHE", req, first, [], use_llm_refinement=False)
    cache_before = json.loads(prompt_cache_file("REQ_GUIDELINE_CACHE").read_text(encoding="utf-8"))
    prompt_builder_module.build_prompt("REQ_GUIDELINE_CACHE", req, second, [], use_llm_refinement=False)
    cache_after = json.loads(prompt_cache_file("REQ_GUIDELINE_CACHE").read_text(encoding="utf-8"))

    assert cache_before["cache_key"] != cache_after["cache_key"]
    assert "NEW BRAND RULE" in cache_after["text"]
    assert "OLD BRAND RULE" not in cache_after["text"]


def test_build_prompt_cache_busts_when_guideline_font_paths_change(isolated_output):
    """2026-09-08 (user request): a guideline's `font` field resolves into
    paths, not text, so it would never show up in `guideline_text` -- the
    cache key must hash `font_image_paths` directly or an edit to which
    font-reference images an entry lists would silently keep serving a
    prompt built before that edit."""
    import app.core.prompt_builder as prompt_builder_module
    from app.core.guidelines import ResolvedGuideline
    from app.core.output_manager import prompt_cache_file

    req = _minimal_req()
    first = ResolvedGuideline(
        format_type="mkt_banner_card_passport_1024x440",
        entry_names=["mkt_banner_card_passport_1024x440"],
        font_image_paths=[Path("/tmp/manrope_medium.png")],
    )
    second = ResolvedGuideline(
        format_type="mkt_banner_card_passport_1024x440",
        entry_names=["mkt_banner_card_passport_1024x440"],
        font_image_paths=[Path("/tmp/manrope_medium.png"), Path("/tmp/manrope_semibold.png")],
    )

    prompt_builder_module.build_prompt("REQ_FONT_CACHE", req, first, [], use_llm_refinement=False)
    cache_before = json.loads(prompt_cache_file("REQ_FONT_CACHE").read_text(encoding="utf-8"))
    prompt_builder_module.build_prompt("REQ_FONT_CACHE", req, second, [], use_llm_refinement=False)
    cache_after = json.loads(prompt_cache_file("REQ_FONT_CACHE").read_text(encoding="utf-8"))

    assert cache_before["cache_key"] != cache_after["cache_key"]


def test_bucket_section_labels_font_attachment():
    """`_bucket_section()` must give a `guideline_font` attachment its own
    descriptive title (not fall back to the raw label string) so the model
    is told to match the typeface, not treat it as a design asset or
    literal text-content requirement."""
    from app.core.prompt_builder import _bucket_section

    section = _bucket_section(["theme", "guideline_font"], _minimal_req())
    assert "FONT REFERENCE" in section
    assert "match" in section.lower()


def test_build_prompt_strips_guideline_geometry_instead_of_rewriting_it(isolated_output):
    """Generation prompts must not expose safe-zone geometry to the image model."""
    import app.core.prompt_builder as prompt_builder_module

    req = _minimal_req_with_crop()
    guideline = _guideline_with_text(
        "### Guideline: banner_popup_noti\n"
        "4. BIÊN AN TOÀN\n- Giữ khoảng cách 48px\n\n"
        "5. HÌNH ẢNH\n- Giữ bố cục sạch và hiện đại"
    )
    built = prompt_builder_module.build_prompt("REQ_CROP_GEOMETRY_STRIPPED", req, guideline, [], use_llm_refinement=False)

    assert "Giữ bố cục sạch và hiện đại" in built.text
    assert "BIÊN AN TOÀN" not in built.text
    assert "48px" not in built.text
    assert "REWRITTEN" not in built.text


def test_build_prompt_uses_positive_continuous_background_wording(isolated_output):
    import app.core.prompt_builder as prompt_builder_module

    built = prompt_builder_module.build_prompt(
        "REQ_CLEAN_TRIM", _minimal_req_with_crop(), _empty_guideline(), [], use_llm_refinement=False
    )

    assert "CANVAS AND FINAL TRIM" in built.text
    assert "one cohesive campaign environment across every pixel" in built.text
    assert "7% padding" not in built.text.lower()
    assert "design boundary" not in built.text.lower()
    assert "full-bleed background (no exceptions)" in built.text.lower()
    assert "do not draw a decorative border" in built.text.lower()
    assert "must reach and continue naturally across all four canvas edges" in built.text.lower()


def test_build_prompt_with_product_keeps_clean_trim_wording(isolated_output):
    import app.core.prompt_builder as prompt_builder_module

    built = prompt_builder_module.build_prompt(
        "REQ_PRODUCT_CLEAN_TRIM",
        _minimal_req_with_crop_and_product(),
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )

    assert "LOCKED PRODUCT ASSET" in built.text
    assert "safe area" not in built.text.lower()
    assert "FINAL COMPOSITION CHECK" in built.text


def test_build_prompt_omits_visual_layout_guide_vocabulary(isolated_output):
    import app.core.prompt_builder as prompt_builder_module

    built = prompt_builder_module.build_prompt(
        "REQ_NO_LAYOUT_GUIDE_WORDS",
        _minimal_req_with_crop_and_product(),
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )

    assert built.text
    assert "full-bleed background (no exceptions)" in built.text.lower()
    assert "do not draw a decorative border" in built.text.lower()
    assert "must reach and continue naturally across all four canvas edges" in built.text.lower()


def test_cropped_prompt_skips_llm_refinement_even_when_requested(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module

    class _ProviderMustNotRun:
        def generate(self, *_args, **_kwargs):
            raise AssertionError("cropped prompt must not be rewritten by an LLM")

    monkeypatch.setattr(prompt_builder_module.LLMFactory, "create", lambda _settings: _ProviderMustNotRun())
    built = prompt_builder_module.build_prompt(
        "REQ_CROP_NO_REFINEMENT",
        _minimal_req_with_crop(),
        _empty_guideline(),
        [],
        use_llm_refinement=True,
    )

    assert built.llm_refined is False
    assert "CANVAS AND FINAL TRIM" in built.text


def test_build_prompt_omits_persona_direction_when_audience_unset(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    built = prompt_builder_module.build_prompt(
        "REQ_NO_PERSONA",
        _minimal_req(),
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )
    assert "PERSONA DIRECTION" not in built.text


def test_build_prompt_includes_persona_direction_and_busts_cache(isolated_output, monkeypatch):
    import app.core.prompt_builder as prompt_builder_module
    from app.core.output_manager import prompt_cache_file
    from app.models.schemas import CreativeRequest

    monkeypatch.setattr(prompt_builder_module, "_design_rules_text", lambda: "")
    data = _minimal_req().model_dump()
    data["audience"] = {"persona": "digital_shopaholic"}
    data["design_options"]["generate_person"] = True
    req = CreativeRequest.model_validate(data)
    built = prompt_builder_module.build_prompt(
        "REQ_PERSONA",
        req,
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )
    assert "PERSONA DIRECTION" in built.text
    assert "digital_shopaholic" in built.text
    assert "Tín đồ mua sắm số" in built.text
    assert "matching this persona talent" in built.text

    cache_with = json.loads(prompt_cache_file("REQ_PERSONA").read_text(encoding="utf-8"))
    prompt_builder_module.build_prompt(
        "REQ_PERSONA",
        _minimal_req(),
        _empty_guideline(),
        [],
        use_llm_refinement=False,
    )
    cache_without = json.loads(prompt_cache_file("REQ_PERSONA").read_text(encoding="utf-8"))
    assert cache_with["cache_key"] != cache_without["cache_key"]
