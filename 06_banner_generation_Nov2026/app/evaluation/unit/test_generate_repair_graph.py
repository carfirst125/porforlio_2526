"""Unit tests for the generate → validate → repair LangGraph.

These mock the image provider and validator so they prove routing / keep-best
without calling Azure or Gemini.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.core.generate_repair_graph import (
    _route_after_generate,
    _route_after_validate,
    build_generate_repair_graph,
    run_generate_and_repair,
)
from app.core.output_validator import ValidationReport, _meta_check
from app.models.schemas import CreativeRequest
from app.providers.base import ImageAttachment


class _FakeUsage:
    def record_image(self, *args, **kwargs) -> None:
        return None


def _req() -> CreativeRequest:
    return CreativeRequest.model_validate(
        {
            "creative_id": "GRAPH_REPAIR_TEST",
            "format_type": "banner_popup_noti",
            "design_brief": "Test request for generate-repair graph routing.",
            "buckets": {"theme": [], "context": [], "product": []},
            "text": {"headline": "Ngân hàng số", "message": "Nhanh và an toàn", "cta_label": "Khám phá"},
            "design_options": {"generate_person": False, "has_logo": False},
            "num_version": 1,
            "generation": {
                "canvas": {"width": 1200, "height": 800},
                "safe_area": {"width": 1035, "height": 420, "position": "center", "x": 82, "y": 190},
                "instruction": "",
            },
            "output": {
                "size": {"width": 1035, "height": 420},
                "crop": {"source": "generation.canvas", "area": {"x": 82, "y": 190, "width": 1035, "height": 420}},
                "clone": [],
            },
        }
    )


def _fail_report(*names: str) -> ValidationReport:
    return ValidationReport(
        checks=[_meta_check(name, status="fail", problem_message=f"{name} failed") for name in names]
    )


def _run(monkeypatch, fake_validate_report, fake_generate=None):
    import app.core.generate_repair_graph as graph_mod

    graph_mod._GRAPH = None
    req = _req()
    calls = {"n": 0, "prompts": []}

    def default_generate(model, prompt, attachments, width, height, settings, usage_sink=None):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        if usage_sink is not None:
            usage_sink["prompt_tokens"] = 0
        return f"raw-{calls['n']}".encode()

    def fake_crop(raw_bytes, canvas_wh, crop_box, final_wh):
        return b"crop-" + raw_bytes

    monkeypatch.setattr("app.core.generate_repair_graph.generate_image", fake_generate or default_generate)
    monkeypatch.setattr("app.core.generate_repair_graph.crop_canvas_to_output", fake_crop)
    monkeypatch.setattr("app.core.generate_repair_graph.validate_output_report", fake_validate_report)
    monkeypatch.setattr("app.core.generate_repair_graph._write_repair_prompt_log", lambda *a, **k: None)
    monkeypatch.setattr("app.core.generate_repair_graph._write_validation_report", lambda *a, **k: None)
    monkeypatch.setattr("app.core.generate_repair_graph._write_final_validation_summary", lambda *a, **k: None)

    result = run_generate_and_repair(
        request_name="GRAPH_REPAIR_TEST",
        model="gemini-3.1-flash-image",
        version=1,
        req=req,
        prompt="first prompt",
        attachments=[ImageAttachment("theme", "image/jpeg", b"theme")],
        canvas_wh=req.canvas_wh(),
        crop_box=req.crop_box(),
        usage_tracker=_FakeUsage(),
        settings=SimpleNamespace(),
        max_attempts=2,
    )
    return result, calls


def test_route_after_validate_uses_repair_mode():
    assert _route_after_validate({"repair_mode": "none", "attempt": 1, "max_attempts": 2}) == "finalize"
    assert _route_after_validate({"repair_mode": "edit", "attempt": 1, "max_attempts": 2}) == "repair"
    assert _route_after_validate({"repair_mode": "regenerate", "attempt": 1, "max_attempts": 2}) == "regenerate"
    assert _route_after_validate({"repair_mode": "edit", "attempt": 2, "max_attempts": 2}) == "finalize"
    assert _route_after_validate({"problems": ["x"], "attempt": 1, "max_attempts": 2}) == "finalize"


def test_route_after_generate_keeps_best_on_provider_failure():
    assert _route_after_generate({"generate_failed": False, "best_cropped": b"x"}) == "validate"
    assert _route_after_generate({"generate_failed": True, "best_cropped": b"x"}) == "finalize"
    assert _route_after_generate({"generate_failed": True}) == "validate"


def test_compiled_graph_has_conditional_validate_edge():
    compiled = build_generate_repair_graph()
    drawable = compiled.get_graph()
    node_ids = {node.id for node in drawable.nodes.values()}
    assert {"generate", "validate", "repair", "regenerate", "finalize"}.issubset(node_ids)
    edge_pairs = {(edge.source, edge.target) for edge in drawable.edges}
    assert ("generate", "validate") in edge_pairs
    assert ("generate", "finalize") in edge_pairs
    assert ("repair", "generate") in edge_pairs
    assert ("regenerate", "generate") in edge_pairs
    assert ("validate", "repair") in edge_pairs
    assert ("validate", "regenerate") in edge_pairs
    assert ("validate", "finalize") in edge_pairs


def test_containment_warning_alone_does_not_repair(monkeypatch):
    def fake_validate_report(image_bytes, req, usage_tracker=None, version=None):
        return _fail_report("content_containment")

    result, calls = _run(monkeypatch, fake_validate_report)
    assert calls["n"] == 1
    assert result["cropped_bytes"] == b"crop-raw-1"
    assert result["problems"] == ["content_containment failed"]


def test_layout_blocker_edits_then_pass(monkeypatch):
    def fake_validate_report(image_bytes, req, usage_tracker=None, version=None):
        if image_bytes == b"crop-raw-1":
            return _fail_report("logo_presence")
        return ValidationReport()

    result, calls = _run(monkeypatch, fake_validate_report)
    assert calls["n"] == 2
    assert "TARGETED LAYOUT REPAIR" in calls["prompts"][1]
    assert result["cropped_bytes"] == b"crop-raw-2"
    assert result["problems"] == []


def test_concept_blocker_regenerates_from_original_prompt(monkeypatch):
    def fake_validate_report(image_bytes, req, usage_tracker=None, version=None):
        if image_bytes == b"crop-raw-1":
            return _fail_report("product_reference")
        return ValidationReport()

    result, calls = _run(monkeypatch, fake_validate_report)
    assert calls["n"] == 2
    assert calls["prompts"][0] == "first prompt"
    assert "first prompt" in calls["prompts"][1]
    assert "PREVIOUS ATTEMPT FAILED THESE CONCEPT CHECKS" in calls["prompts"][1]
    assert "TARGETED LAYOUT REPAIR" not in calls["prompts"][1]
    assert result["cropped_bytes"] == b"crop-raw-2"
    assert result["problems"] == []


def test_keep_best_reverts_when_repair_regresses(monkeypatch):
    def fake_validate_report(image_bytes, req, usage_tracker=None, version=None):
        if image_bytes == b"crop-raw-1":
            return _fail_report("logo_presence")
        return _fail_report("logo_presence", "technical_artifact")

    result, calls = _run(monkeypatch, fake_validate_report)
    assert calls["n"] == 2
    assert result["cropped_bytes"] == b"crop-raw-1"
    assert result["raw_bytes"] == b"raw-1"
    assert result["problems"] == ["logo_presence failed"]


def test_repair_provider_failure_keeps_first_image(monkeypatch):
    req = _req()
    calls = {"n": 0}

    def fake_generate(model, prompt, attachments, width, height, settings, usage_sink=None):
        calls["n"] += 1
        if usage_sink is not None:
            usage_sink["prompt_tokens"] = 0
        if calls["n"] == 1:
            return b"raw-1"
        raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")

    def fake_crop(raw_bytes, canvas_wh, crop_box, final_wh):
        return b"crop-" + raw_bytes

    def fake_validate_report(image_bytes, req, usage_tracker=None, version=None):
        return _fail_report("logo_presence")

    import app.core.generate_repair_graph as graph_mod

    graph_mod._GRAPH = None
    monkeypatch.setattr("app.core.generate_repair_graph.generate_image", fake_generate)
    monkeypatch.setattr("app.core.generate_repair_graph.crop_canvas_to_output", fake_crop)
    monkeypatch.setattr("app.core.generate_repair_graph.validate_output_report", fake_validate_report)
    monkeypatch.setattr("app.core.generate_repair_graph._write_repair_prompt_log", lambda *a, **k: None)
    monkeypatch.setattr("app.core.generate_repair_graph._write_validation_report", lambda *a, **k: None)
    monkeypatch.setattr("app.core.generate_repair_graph._write_final_validation_summary", lambda *a, **k: None)

    result = run_generate_and_repair(
        request_name="GRAPH_REPAIR_TEST",
        model="gemini-3.1-flash-image",
        version=1,
        req=req,
        prompt="first prompt",
        attachments=[ImageAttachment("theme", "image/jpeg", b"theme")],
        canvas_wh=req.canvas_wh(),
        crop_box=req.crop_box(),
        usage_tracker=_FakeUsage(),
        settings=SimpleNamespace(),
        max_attempts=2,
    )
    assert calls["n"] == 2
    assert result["cropped_bytes"] == b"crop-raw-1"
    assert result["raw_bytes"] == b"raw-1"
    assert result["problems"] == ["logo_presence failed"]
