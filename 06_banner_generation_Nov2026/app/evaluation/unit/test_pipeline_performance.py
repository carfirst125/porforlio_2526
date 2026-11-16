from __future__ import annotations

import io
import threading
from types import SimpleNamespace

from PIL import Image

from app.core import pipeline
from app.core.asset_resolver import ResolvedBucketAssets
from app.core.guidelines import ResolvedGuideline
from app.core.output_validator import ValidationReport, _meta_check
from app.core.prompt_builder import BuiltPrompt
from app.models.schemas import CreativeRequest


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 80, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def _request(num_version: int) -> CreativeRequest:
    return CreativeRequest.model_validate(
        {
            "creative_id": "PERF_TEST",
            "format_type": "perf_test",
            "design_brief": "Performance regression test.",
            "buckets": {"theme": [], "context": [], "product": []},
            "text": {"headline": "H", "message": "M", "cta_label": "C"},
            "num_version": num_version,
            "design_options": {
                "generate_person": False,
                "has_logo": False,
                "technique_selection": {"mode": "manual", "manual_techniques": []},
            },
            "generation": {
                "canvas": {"width": 64, "height": 64},
                "safe_area": {"width": 64, "height": 64, "position": "center", "x": 0, "y": 0},
            },
            "output": {"size": {"width": 64, "height": 64}, "clone": []},
        }
    )


def _patch_graph(monkeypatch, req: CreativeRequest) -> None:
    state = {
        "req": req,
        "built_prompt": BuiltPrompt(text="base prompt", cache_key="base"),
        "prepared_images": [],
        "guideline": ResolvedGuideline(format_type="perf_test", combined_text=""),
        "resolved_assets": ResolvedBucketAssets(
            theme=[], context=[], product=[], brand_text=[], reference=None, guideline_extra=None, all_resolved=[]
        ),
    }

    class FakeGraph:
        def invoke(self, _state):
            return state

    monkeypatch.setattr(pipeline, "_get_graph", lambda: FakeGraph())
    monkeypatch.setattr(pipeline.output_manager, "clear_technique_selection", lambda _name: None)
    monkeypatch.setattr(pipeline.UsageTracker, "save", lambda self, num_version=None: None)


def _patch_repair_graph(monkeypatch, fake_generate, fake_validate_report) -> None:
    import app.core.generate_repair_graph as graph_mod

    graph_mod._GRAPH = None
    monkeypatch.setattr(graph_mod, "generate_image", fake_generate)
    monkeypatch.setattr(graph_mod, "validate_output_report", fake_validate_report)
    monkeypatch.setattr(graph_mod, "_write_repair_prompt_log", lambda *a, **k: None)
    monkeypatch.setattr(graph_mod, "_write_validation_report", lambda *a, **k: None)
    monkeypatch.setattr(graph_mod, "_write_final_validation_summary", lambda *a, **k: None)


def test_versions_run_concurrently_and_results_remain_ordered(monkeypatch, isolated_output):
    req = _request(3)
    _patch_graph(monkeypatch, req)
    barrier = threading.Barrier(3)

    def fake_generate(_model, _prompt, _attachments, width, height, _settings, usage_sink=None):
        barrier.wait(timeout=3)
        return _png(width, height)

    monkeypatch.setattr(pipeline, "get_settings", lambda: SimpleNamespace(generation_max_workers=3))
    _patch_repair_graph(monkeypatch, fake_generate, lambda *_a, **_k: ValidationReport())

    results = pipeline.run_generate("PERF_TEST", ["fake-model"])

    assert [result["version"] for result in results] == [1, 2, 3]
    assert all(result["ok"] for result in results)


def test_containment_warning_does_not_spend_a_second_image_call(monkeypatch, isolated_output):
    """New validate policy: a single warning must not pay for another image gen."""
    req = _request(1)
    _patch_graph(monkeypatch, req)
    calls = {"n": 0}

    def fake_generate(_model, _prompt, _attachments, width, height, _settings, usage_sink=None):
        calls["n"] += 1
        return _png(width, height)

    def fake_validate(*_a, **_k):
        return ValidationReport(
            checks=[_meta_check("content_containment", status="fail", problem_message="clipped")]
        )

    monkeypatch.setattr(pipeline, "get_settings", lambda: SimpleNamespace(generation_max_workers=1))
    _patch_repair_graph(monkeypatch, fake_generate, fake_validate)

    results = pipeline.run_generate("PERF_TEST", ["fake-model"])

    assert results[0]["ok"] is True
    assert calls["n"] == 1


def test_layout_blocker_runs_targeted_edit(monkeypatch, isolated_output):
    req = _request(1)
    _patch_graph(monkeypatch, req)
    calls = {"n": 0, "prompts": []}

    def fake_generate(_model, prompt, _attachments, width, height, _settings, usage_sink=None):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        return _png(width, height)

    def fake_validate(*_a, **_k):
        if calls["n"] == 1:
            return ValidationReport(
                checks=[_meta_check("logo_presence", status="fail", problem_message="logo missing")]
            )
        return ValidationReport()

    monkeypatch.setattr(pipeline, "get_settings", lambda: SimpleNamespace(generation_max_workers=1))
    _patch_repair_graph(monkeypatch, fake_generate, fake_validate)

    results = pipeline.run_generate("PERF_TEST", ["fake-model"])

    assert results[0]["ok"] is True
    assert calls["n"] == 2
    assert "TARGETED LAYOUT REPAIR" in calls["prompts"][1]
