from app.core.output_validator import (
    ValidationCheck,
    ValidationReport,
    _CONTEXT_REFERENCE_QUESTION,
    _PRODUCT_REFERENCE_QUESTION,
    _meta_check,
    format_validation_report_text,
)


def _fail(name: str) -> ValidationCheck:
    return _meta_check(name, status="fail", question="q?", answer="no", problem_message=f"{name} failed")


def test_containment_alone_is_warning_and_keeps_image():
    report = ValidationReport(checks=[_fail("content_containment")])
    assert report.repair_mode == "none"
    assert report.score == 1
    assert "GIỮ ẢNH" in report.decision_vi


def test_two_warnings_trigger_edit():
    extra = ValidationCheck(
        name="extra_warning",
        status="fail",
        severity="warning",
        family="layout",
        title_vi="cảnh báo thêm",
        fail_vi="x",
        question="",
        answer="no",
        problem_message="extra",
    )
    report = ValidationReport(checks=[_fail("content_containment"), extra])
    assert report.repair_mode == "edit"
    assert report.score == 2


def test_layout_blocker_triggers_edit():
    report = ValidationReport(checks=[_fail("logo_presence")])
    assert report.repair_mode == "edit"
    assert report.score == 10


def test_concept_blocker_triggers_regenerate_even_with_warnings():
    report = ValidationReport(checks=[_fail("product_reference"), _fail("content_containment")])
    assert report.repair_mode == "regenerate"
    assert report.score == 11


def test_clean_report_keeps_image():
    report = ValidationReport(checks=[_meta_check("guide_color_leak", status="pass", answer="not detected")])
    assert report.repair_mode == "none"
    assert report.problems == []
    assert report.score == 0


def test_format_validation_report_is_readable_for_business():
    report = ValidationReport(checks=[_fail("content_containment")])
    text = format_validation_report_text(
        report, request_name="DB_AUTOBILL_1024x440_AFF", model="gemini-3.1-flash-image", version=1, attempt=1
    )
    assert "BÁO CÁO VALIDATE ẢNH" in text
    assert "DB_AUTOBILL_1024x440_AFF" in text
    assert "content_containment" in text
    assert "Câu hỏi gửi vision LLM" in text
    assert "Trả lời vision/check: no" in text
    assert "repair_mode : none" in text
    assert "Ý nghĩa nghiệp vụ" in text


def test_write_validation_report_file(tmp_path):
    from app.core.output_validator import write_validation_report_file

    path = tmp_path / "validate-demo-version-1-attempt-1.txt"
    write_validation_report_file(
        ValidationReport(checks=[_fail("logo_presence")]),
        path,
        request_name="DEMO",
        model="gemini-3.1-flash-image",
        version=1,
        attempt=1,
    )
    saved = path.read_text(encoding="utf-8")
    assert "SỬA TRÊN ẢNH CŨ" in saved
    assert "logo_presence" in saved


def test_concept_regenerate_prompt_keeps_original_and_lists_failures():
    from app.core.generate_repair_graph import _concept_regenerate_prompt

    text = _concept_regenerate_prompt("ORIGINAL BANNER PROMPT", ["product asset missing or changed"])
    assert text.startswith("ORIGINAL BANNER PROMPT")
    assert "PREVIOUS ATTEMPT FAILED THESE CONCEPT CHECKS" in text
    assert "product asset missing or changed" in text
    assert "TARGETED LAYOUT REPAIR" not in text


def test_product_and_context_questions_do_not_steal_containment():
    """Sát mép is a warning check. Identity checks must not fail for the same reason
    or Autobill-like usable crops would trigger a full regenerate."""
    assert "cropped by an image edge" not in _PRODUCT_REFERENCE_QUESTION
    assert "inside the banner edges" not in _CONTEXT_REFERENCE_QUESTION
    assert "separate containment warning" in _PRODUCT_REFERENCE_QUESTION
    assert "separate containment warning" in _CONTEXT_REFERENCE_QUESTION
