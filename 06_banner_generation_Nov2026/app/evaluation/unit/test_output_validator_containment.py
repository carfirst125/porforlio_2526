from app.core.output_validator import _TECHNICAL_ARTIFACT_QUESTION, _content_containment_question
from app.models.schemas import CreativeRequest


def _req_with_text() -> CreativeRequest:
    return CreativeRequest.model_validate(
        {
            "creative_id": "UNIT_CONTAINMENT",
            "format_type": "ocb_omni_110926_mass_1024x440",
            "design_brief": "Unit.",
            "buckets": {"theme": [], "context": [], "product": ["Điện thoại với tính năng thanh toán hóa đơn tự động."]},
            "reference": "",
            "text": {
                "headline": "TỰ ĐỘNG THANH TOÁN\nMỘT CHẠM DỄ DÀNG",
                "message": "Tặng 10.000 OMNI coin",
                "cta_label": "",
                "cta_deep_link": "",
            },
            "guidelines": "",
            "num_version": 1,
            "design_options": {"generate_person": False, "has_logo": False},
            "generation": {
                "canvas": {"width": 1024, "height": 440},
                "safe_area": {"width": 1024, "height": 440, "position": "center", "x": 0, "y": 0},
                "instruction": "",
            },
            "output": {"size": {"width": 1024, "height": 440}, "clone": []},
        }
    )


def test_containment_question_includes_exact_required_text_and_edge_rules():
    import json

    req = _req_with_text()
    question = _content_containment_question(req)
    dumped = json.dumps([req.text.headline, req.text.message], ensure_ascii=False)

    assert dumped in question
    assert "product corner" in question
    assert "touches an edge" in question


def test_technical_artifact_question_flags_whole_banner_frame_not_scene_objects():
    q = _TECHNICAL_ARTIFACT_QUESTION
    assert "WHOLE banner" in q
    assert "gold or metallic line" in q
    assert "48px" in q
    assert "E39717" in q
    assert "WITH OR WITHOUT" in q
    assert "real objects inside the scene" in q
    assert "Google, TikTok, Meta" in q
    assert "frosted-glass container" not in q
    assert "ANY inset rectangle" not in q
