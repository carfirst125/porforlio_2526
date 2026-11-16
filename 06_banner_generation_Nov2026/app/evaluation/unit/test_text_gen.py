from __future__ import annotations

import json

import pytest

from app.core import text_gen
from app.models.schemas import CreativeRequest


def _request(**design_option_overrides) -> CreativeRequest:
    """`generate_text=True` by default (LLM call happens); the 3
    `generate_text_<field>` selection flags default to False like the real
    schema default -- pass e.g. `generate_text_headline=True` to also mark
    that field's generated copy as the one to use, matching the
    generate_text/selection split described in `text_gen._selected_llm_fields()`."""
    return CreativeRequest.model_validate(
        {
            "creative_id": "TEXT_GEN_TEST",
            "format_type": "banner_popup_noti",
            "design_brief": "Quảng bá thẻ, hoàn tiền 3%.",
            "buckets": {},
            "text": {"headline": "Fallback", "message": "Fallback message", "cta_label": "Mở ngay", "cta_deep_link": "deep"},
            "num_version": 2,
            "design_options": {"generate_text": True, **design_option_overrides},
            "generation": {
                "canvas": {"width": 100, "height": 100},
                "safe_area": {"width": 100, "height": 100, "x": 0, "y": 0},
            },
            "output": {"size": {"width": 100, "height": 100}},
        }
    )


def _request_full_selection() -> CreativeRequest:
    """`_request()` with all 3 fields selected -- i.e. the old backward-
    compatible "generate_text=True uses all 3" shortcut, now expressed
    explicitly since that shortcut is no longer implicit (2026-09-04)."""
    return _request(generate_text_headline=True, generate_text_message=True, generate_text_cta_label=True)


class _Provider:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        assert "2" in prompt
        return self.response


def test_generates_complete_text_for_each_version_once(isolated_output, monkeypatch, tmp_path):
    template = tmp_path / "template.txt"
    template.write_text("{{DESIGN_BRIEF}} / {{NUM_VERSIONS}}", encoding="utf-8")
    monkeypatch.setattr(text_gen, "prompt_template_path", lambda _: template)
    provider = _Provider(json.dumps({
        "1": {"headline": "H1", "message": "M1", "cta": "Xem thêm"},
        "2": {"headline": "H2", "message": "M2", "cta": "Tìm hiểu"},
    }))
    monkeypatch.setattr(text_gen.LLMFactory, "create", staticmethod(lambda settings: provider))

    first = text_gen.generate_creative_texts("REQ", _request_full_selection(), 2)
    second = text_gen.generate_creative_texts("REQ", _request_full_selection(), 2)

    assert provider.calls == 1
    assert first[1].headline == "H1"
    assert first[2].message == "M2"
    assert first[2].cta_deep_link == "deep"
    assert second[2].cta_label == "Tìm hiểu"


def test_writes_prompt_log_alongside_text_gen_cache(isolated_output, monkeypatch, tmp_path):
    """2026-09-08 (user request): the exact filled LLM prompt used to
    generate the per-version copy must be logged to
    `_text_gen_cache/text_gen_prompt.txt`, next to `text_gen.json`."""
    from app.core.output_manager import text_gen_cache_file, text_gen_prompt_log_file

    template = tmp_path / "template.txt"
    template.write_text("BRIEF=[{{DESIGN_BRIEF}}] N=[{{NUM_VERSIONS}}]", encoding="utf-8")
    monkeypatch.setattr(text_gen, "prompt_template_path", lambda _: template)
    provider = _Provider(json.dumps({
        "1": {"headline": "H1", "message": "M1", "cta": "C1"},
        "2": {"headline": "H2", "message": "M2", "cta": "C2"},
    }))
    monkeypatch.setattr(text_gen.LLMFactory, "create", staticmethod(lambda settings: provider))

    text_gen.generate_creative_texts("REQ_PROMPT_LOG", _request_full_selection(), 2)

    log_path = text_gen_prompt_log_file("REQ_PROMPT_LOG")
    assert log_path.parent == text_gen_cache_file("REQ_PROMPT_LOG").parent  # same _text_gen_cache/ dir
    content = log_path.read_text(encoding="utf-8")
    assert "Quảng bá thẻ, hoàn tiền 3%." in content
    assert "N=[2]" in content


def test_text_gen_includes_persona_tone_and_busts_cache(isolated_output, monkeypatch, tmp_path):
    from app.core.output_manager import text_gen_prompt_log_file

    template = tmp_path / "template.txt"
    template.write_text("BRIEF=[{{DESIGN_BRIEF}}] N=[{{NUM_VERSIONS}}] P=[{{PERSONA_DIRECTION}}]", encoding="utf-8")
    monkeypatch.setattr(text_gen, "prompt_template_path", lambda _: template)
    provider = _Provider(json.dumps({
        "1": {"headline": "H1", "message": "M1", "cta": "C1"},
        "2": {"headline": "H2", "message": "M2", "cta": "C2"},
    }))
    monkeypatch.setattr(text_gen.LLMFactory, "create", staticmethod(lambda settings: provider))

    plain = _request_full_selection()
    with_persona = plain.model_copy(
        update={"audience": plain.audience.model_copy(update={"persona": "digital_shopaholic"})}
    )
    text_gen.generate_creative_texts("REQ_PERSONA_COPY", plain, 2)
    assert provider.calls == 1
    text_gen.generate_creative_texts("REQ_PERSONA_COPY", with_persona, 2)
    assert provider.calls == 2
    content = text_gen_prompt_log_file("REQ_PERSONA_COPY").read_text(encoding="utf-8")
    assert "digital_shopaholic" in content or "Tín đồ mua sắm số" in content


def test_prompt_log_is_overwritten_not_appended_on_a_second_call(isolated_output, monkeypatch, tmp_path):
    from app.core.output_manager import text_gen_prompt_log_file

    template = tmp_path / "template.txt"
    template.write_text("{{DESIGN_BRIEF}} / {{NUM_VERSIONS}}", encoding="utf-8")
    monkeypatch.setattr(text_gen, "prompt_template_path", lambda _: template)
    provider = _Provider(json.dumps({
        "1": {"headline": "H1", "message": "M1", "cta": "C1"},
        "2": {"headline": "H2", "message": "M2", "cta": "C2"},
    }))
    monkeypatch.setattr(text_gen.LLMFactory, "create", staticmethod(lambda settings: provider))

    text_gen.generate_creative_texts("REQ_PROMPT_LOG2", _request_full_selection(), 2)
    first_content = text_gen_prompt_log_file("REQ_PROMPT_LOG2").read_text(encoding="utf-8")

    template.write_text("CHANGED {{DESIGN_BRIEF}} / {{NUM_VERSIONS}}", encoding="utf-8")
    text_gen.generate_creative_texts("REQ_PROMPT_LOG2", _request_full_selection(), 2)
    second_content = text_gen_prompt_log_file("REQ_PROMPT_LOG2").read_text(encoding="utf-8")

    assert second_content.startswith("CHANGED")
    assert second_content != first_content


def test_writes_result_log_with_raw_llm_json_on_fresh_call(isolated_output, monkeypatch, tmp_path):
    """2026-09-08 (user request, part 2): the raw per-version JSON returned
    by the LLM must be logged to `_text_gen_cache/text_gen_result.json`,
    the OUTPUT counterpart of `text_gen_prompt.txt`."""
    from app.core.output_manager import text_gen_result_log_file

    template = tmp_path / "template.txt"
    template.write_text("{{DESIGN_BRIEF}} / {{NUM_VERSIONS}}", encoding="utf-8")
    monkeypatch.setattr(text_gen, "prompt_template_path", lambda _: template)
    raw_response = {
        "1": {"headline": "H1", "message": "M1", "cta": "C1"},
        "2": {"headline": "H2", "message": "M2", "cta": "C2"},
    }
    provider = _Provider(json.dumps(raw_response))
    monkeypatch.setattr(text_gen.LLMFactory, "create", staticmethod(lambda settings: provider))

    text_gen.generate_creative_texts("REQ_RESULT_LOG", _request_full_selection(), 2)

    logged = json.loads(text_gen_result_log_file("REQ_RESULT_LOG").read_text(encoding="utf-8"))
    assert logged == raw_response


def test_result_log_reflects_cached_versions_on_a_cache_hit(isolated_output, monkeypatch, tmp_path):
    """A cache hit still (re)writes the result log -- with the REUSED cached
    versions, not stale content from before -- since no fresh LLM call
    happens on that path to write it otherwise."""
    from app.core.output_manager import text_gen_result_log_file

    template = tmp_path / "template.txt"
    template.write_text("{{DESIGN_BRIEF}} / {{NUM_VERSIONS}}", encoding="utf-8")
    monkeypatch.setattr(text_gen, "prompt_template_path", lambda _: template)
    raw_response = {
        "1": {"headline": "H1", "message": "M1", "cta": "Xem thêm"},
        "2": {"headline": "H2", "message": "M2", "cta": "Tìm hiểu"},
    }
    provider = _Provider(json.dumps(raw_response))
    monkeypatch.setattr(text_gen.LLMFactory, "create", staticmethod(lambda settings: provider))

    text_gen.generate_creative_texts("REQ_RESULT_LOG2", _request_full_selection(), 2)
    assert provider.calls == 1

    # Corrupt the on-disk result log to prove the 2nd (cache-hit) call
    # actually rewrites it rather than leaving it untouched.
    log_path = text_gen_result_log_file("REQ_RESULT_LOG2")
    log_path.write_text("STALE", encoding="utf-8")

    text_gen.generate_creative_texts("REQ_RESULT_LOG2", _request_full_selection(), 2)
    assert provider.calls == 1  # still a cache hit -- no 2nd LLM call

    logged = json.loads(log_path.read_text(encoding="utf-8"))
    assert logged == raw_response


def test_incomplete_version_response_falls_back_instead_of_returning_empty_text(isolated_output, monkeypatch, tmp_path):
    template = tmp_path / "template.txt"
    template.write_text("{{DESIGN_BRIEF}} / {{NUM_VERSIONS}}", encoding="utf-8")
    monkeypatch.setattr(text_gen, "prompt_template_path", lambda _: template)
    provider = _Provider('{"1": {"headline": "H1", "message": "M1", "cta": "C1"}}')
    monkeypatch.setattr(text_gen.LLMFactory, "create", staticmethod(lambda settings: provider))

    assert text_gen.generate_creative_texts("REQ", _request(), 2) is None


def test_generated_copy_strips_presentational_dashes_and_bullets():
    parsed = {
        "1": {"headline": "— Hoàn tiền 3%", "message": "• Mở thẻ miễn phí |", "cta": "- Mở thẻ"}
    }
    result = text_gen._validated_versions(parsed, 1, _request_full_selection())
    assert result[1].headline == "Hoàn tiền 3%"
    assert result[1].message == "Mở thẻ miễn phí"
    assert result[1].cta_label == "Mở thẻ"


def test_field_level_generation_preserves_fixed_copy():
    req = _request().model_copy(
        update={
            "text": _request().text.model_copy(
                update={"headline": "Headline đã duyệt", "message": "", "cta_label": "Mở ngay"}
            ),
            "design_options": _request().design_options.model_copy(
                update={"generate_text": False, "generate_text_message": True}
            ),
        }
    )
    parsed = {"1": {"headline": "Không được dùng", "message": "Sinh lời an tâm", "cta": "Không được dùng"}}
    result = text_gen._validated_versions(parsed, 1, req)
    assert result[1].headline == "Headline đã duyệt"
    assert result[1].message == "Sinh lời an tâm"
    assert result[1].cta_label == "Mở ngay"


def test_generated_copy_rejects_non_printing_hex_color_tokens():
    parsed = {
        "1": {"headline": "Hoàn tiền #E39717", "message": "Mở thẻ miễn phí", "cta": "Mở ngay"},
        "2": {"headline": "Hoàn tiền mỗi ngày", "message": "Thanh toán tiện lợi", "cta": "Đăng ký"},
    }
    with pytest.raises(ValueError, match="hex color"):
        text_gen._validated_versions(parsed, 2, _request_full_selection())


def test_generated_copy_rejects_bare_hex_glued_onto_phrase():
    parsed = {
        "1": {"headline": "Quản lý hóa đơn", "message": "gọn nhẹ E39717", "cta": "Tìm hiểu thêm"},
        "2": {"headline": "Hoàn tiền mỗi ngày", "message": "Thanh toán tiện lợi", "cta": "Đăng ký"},
    }
    with pytest.raises(ValueError, match="hex color"):
        text_gen._validated_versions(parsed, 2, _request_full_selection())


def test_generated_cta_rejects_forbidden_ngay():
    parsed = {"1": {"headline": "Lợi ích linh hoạt", "message": "Sinh lời hiệu quả", "cta": "Khám phá ngay"}}

    with pytest.raises(ValueError, match="forbidden word 'ngay'"):
        text_gen._validated_versions(parsed, 1, _request_full_selection())


def test_generated_cta_requires_two_to_four_words():
    parsed = {"1": {"headline": "Lợi ích linh hoạt", "message": "Sinh lời hiệu quả", "cta": "Xem"}}

    with pytest.raises(ValueError, match="2-4 words"):
        text_gen._validated_versions(parsed, 1, _request_full_selection())
