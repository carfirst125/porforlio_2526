from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.core import output_manager, pipeline, technique_selector
from app.core.prompt_builder import BuiltPrompt
from app.core.asset_resolver import ResolvedBucketAssets
from app.core.guidelines import ResolvedGuideline
from app.models.schemas import CreativeRequest


def _request(
    mode: str = "auto",
    techniques: list[str] | None = None,
    auto_config: dict | None = None,
) -> CreativeRequest:
    """`techniques` here means `design_options.technique_selection.manual_techniques`
    -- kept as a parameter name for minimal diff against the pre-2026-09-09
    call sites below; it has no effect at all when `mode="auto"` (ignored
    completely, per `TechniqueSelectionConfig`'s contract)."""
    technique_selection: dict = {"mode": mode, "manual_techniques": techniques or []}
    if auto_config is not None:
        technique_selection["auto_config"] = auto_config
    return CreativeRequest.model_validate(
        {
            "creative_id": "AUTO_TECHNIQUE_TEST",
            "format_type": "banner_popup_noti",
            "design_brief": "Premium digital banking campaign with a restrained blue visual system.",
            "buckets": {"theme": [], "context": [], "product": []},
            "text": {"headline": "Ngân hàng số", "message": "Nhanh và an toàn", "cta_label": "Khám phá"},
            "design_options": {
                "generate_person": False,
                "has_logo": False,
                "technique_selection": technique_selection,
            },
            "num_version": 1,
            "generation": {
                "canvas": {"width": 1200, "height": 800},
                "safe_area": {"width": 1035, "height": 420, "position": "center", "x": 82, "y": 190},
                "instruction": "Keep required content in the safe area.",
            },
            "output": {
                "size": {"width": 1035, "height": 420},
                "crop": {"source": "generation.canvas", "area": {"x": 82, "y": 190, "width": 1035, "height": 420}},
                "clone": [],
            },
        }
    )


def _context():
    return (
        ResolvedGuideline(format_type="banner_popup_noti", combined_text="Keep hierarchy clear."),
        ResolvedBucketAssets(
            theme=[], context=[], product=[], brand_text=[], reference=None, guideline_extra=None, all_resolved=[]
        ),
    )


def _patch_runtime(monkeypatch, tmp_path: Path, provider, batch_size: int = 10):
    selector_settings = SimpleNamespace(
        llm_provider="fake",
        llm_model_name="fake-selector",
        technique_selection_batch_size=batch_size,
        technique_selection_cache_enabled=False,
    )
    monkeypatch.setattr(technique_selector, "get_settings", lambda: selector_settings)
    monkeypatch.setattr(technique_selector.LLMFactory, "create", lambda _settings: provider)
    monkeypatch.setattr(output_manager, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path))


class _EmptyProvider:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        payload = json.loads(prompt)
        versions = {
            version: {"background": [], "decoration": [], "product": [], "cohesion_summary": "No forced effects."}
            for version in payload["output_contract"]["required_version_ids"]
        }
        return json.dumps({"versions": versions})


def test_selector_batches_more_than_ten_versions_and_writes_artifacts(monkeypatch, tmp_path):
    provider = _EmptyProvider()
    _patch_runtime(monkeypatch, tmp_path, provider, batch_size=10)
    guideline, assets = _context()
    requests = {version: _request() for version in range(1, 12)}

    selected = technique_selector.select_techniques_for_versions("REQ", requests, guideline, assets)

    assert len(provider.prompts) == 2
    assert all(selection.flattened_keys() == [] for selection in selected.values())
    artifact = json.loads(output_manager.technique_selection_file("REQ").read_text(encoding="utf-8"))
    assert artifact["effective_num_version"] == 11
    assert [batch["version_ids"] for batch in artifact["batches"]] == [list(range(1, 11)), [11]]
    assert len(list(output_manager.technique_selection_dir("REQ").glob("batch-*.json"))) == 2


def test_selector_batches_exactly_twenty_and_twentyone_versions(monkeypatch, tmp_path):
    # Section 16 test plan: 20 versions -> 2 calls, 21 versions -> 3 calls
    # (the batching loop is a plain fixed-size slice over sorted version IDs,
    # so this also guards against an off-by-one at the exact multiple-of-10
    # boundary, not just the 11-version case already covered above).
    for total_versions, expected_batch_count in ((20, 2), (21, 3)):
        provider = _EmptyProvider()
        _patch_runtime(monkeypatch, tmp_path, provider, batch_size=10)
        guideline, assets = _context()
        requests = {version: _request() for version in range(1, total_versions + 1)}

        selected = technique_selector.select_techniques_for_versions(
            f"REQ_{total_versions}", requests, guideline, assets
        )

        assert len(provider.prompts) == expected_batch_count
        assert set(selected) == set(range(1, total_versions + 1))
        artifact = json.loads(
            output_manager.technique_selection_file(f"REQ_{total_versions}").read_text(encoding="utf-8")
        )
        assert len(artifact["batches"]) == expected_batch_count
        all_ids = [v for batch in artifact["batches"] for v in batch["version_ids"]]
        assert sorted(all_ids) == list(range(1, total_versions + 1))
        assert len(all_ids) == len(set(all_ids))  # no version appears in two batches


def test_selector_uses_custom_batch_size_from_settings(monkeypatch, tmp_path):
    provider = _EmptyProvider()
    _patch_runtime(monkeypatch, tmp_path, provider, batch_size=5)
    guideline, assets = _context()
    requests = {version: _request() for version in range(1, 12)}

    technique_selector.select_techniques_for_versions("REQ_CUSTOM_BATCH", requests, guideline, assets)

    assert len(provider.prompts) == 3  # 5 + 5 + 1
    artifact = json.loads(output_manager.technique_selection_file("REQ_CUSTOM_BATCH").read_text(encoding="utf-8"))
    assert [batch["version_ids"] for batch in artifact["batches"]] == [
        list(range(1, 6)),
        list(range(6, 11)),
        [11],
    ]


def test_selector_writes_prompt_log_with_one_block_per_batch(monkeypatch, tmp_path):
    provider = _EmptyProvider()
    _patch_runtime(monkeypatch, tmp_path, provider, batch_size=10)
    guideline, assets = _context()
    requests = {version: _request() for version in range(1, 12)}  # 11 versions -> 2 batches

    technique_selector.select_techniques_for_versions("REQ_PROMPT_LOG", requests, guideline, assets)

    log_path = output_manager.technique_selection_prompt_log_file("REQ_PROMPT_LOG")
    assert log_path.parent == output_manager.technique_selection_dir("REQ_PROMPT_LOG")
    assert log_path.name == "technique_selection_prompt.txt"
    text = log_path.read_text(encoding="utf-8")
    assert "===== batch 1 -- versions" in text
    assert "===== batch 2 -- versions" in text
    # The logged prompt is the real payload sent to the LLM (not a summary).
    assert '"output_contract"' in text


def test_selector_prompt_never_contains_manual_list_in_auto_mode(monkeypatch, tmp_path):
    provider = _EmptyProvider()
    _patch_runtime(monkeypatch, tmp_path, provider)
    guideline, assets = _context()

    technique_selector.select_techniques_for_versions(
        "IGNORE_MANUAL",
        {1: _request(mode="auto", techniques=["MANUAL_SENTINEL_MUST_BE_IGNORED"])},
        guideline,
        assets,
    )

    assert "MANUAL_SENTINEL_MUST_BE_IGNORED" not in provider.prompts[0]


def test_selector_retries_once_after_invalid_key(monkeypatch, tmp_path):
    class RetryProvider:
        def __init__(self):
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs) -> str:
            self.calls += 1
            key = ["not_in_catalog"] if self.calls == 1 else []
            return json.dumps({"versions": {"1": {"background": key, "decoration": [], "product": []}}})

    provider = RetryProvider()
    _patch_runtime(monkeypatch, tmp_path, provider)
    guideline, assets = _context()
    selected = technique_selector.select_techniques_for_versions("RETRY", {1: _request()}, guideline, assets)

    assert provider.calls == 2
    assert selected[1].flattened_keys() == []
    artifact = json.loads(output_manager.technique_selection_file("RETRY").read_text(encoding="utf-8"))
    assert artifact["versions"]["1"]["status"] == "corrected"


def test_selector_falls_back_to_empty_after_two_failed_calls(monkeypatch, tmp_path):
    class FailingProvider:
        def __init__(self):
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs) -> str:
            self.calls += 1
            raise RuntimeError("offline")

    provider = FailingProvider()
    _patch_runtime(monkeypatch, tmp_path, provider)
    guideline, assets = _context()
    selected = technique_selector.select_techniques_for_versions("FAIL", {1: _request()}, guideline, assets)

    assert provider.calls == 2
    assert selected[1].flattened_keys() == []
    artifact = json.loads(output_manager.technique_selection_file("FAIL").read_text(encoding="utf-8"))
    assert artifact["versions"]["1"]["status"] == "empty_safe_fallback"


def test_pipeline_boundary_auto_ignores_manual_techniques(monkeypatch):
    auto_req = _request(mode="auto", techniques=["radial_spotlight"])
    monkeypatch.setattr(pipeline, "load_request_json", lambda _name: auto_req)
    monkeypatch.setattr(pipeline, "plan_request_geometry", lambda req, _models: (req, None))

    loaded = pipeline._node_load_request({"request_name": "REQ", "models": []})["req"]

    assert loaded.design_options.technique_selection.mode == "auto"
    assert loaded.design_options.techniques == []


def test_pipeline_boundary_manual_preserves_explicit_techniques(monkeypatch):
    manual_req = _request(mode="manual", techniques=["radial_spotlight"])
    monkeypatch.setattr(pipeline, "load_request_json", lambda _name: manual_req)
    monkeypatch.setattr(pipeline, "plan_request_geometry", lambda req, _models: (req, None))

    loaded = pipeline._node_load_request({"request_name": "REQ", "models": []})["req"]

    assert loaded.design_options.techniques == ["radial_spotlight"]


def test_pipeline_persona_override_sets_or_clears_without_touching_json(monkeypatch):
    req = _request()
    req = req.model_copy(update={"audience": req.audience.model_copy(update={"persona": "experience_seeker"})})
    monkeypatch.setattr(pipeline, "load_request_json", lambda _name: req)
    monkeypatch.setattr(pipeline, "plan_request_geometry", lambda loaded, _models: (loaded, None))

    kept = pipeline._node_load_request({"request_name": "REQ", "models": []})["req"]
    assert kept.audience.persona == "experience_seeker"

    cleared = pipeline._node_load_request(
        {"request_name": "REQ", "models": [], "persona_override": ""}
    )["req"]
    assert cleared.audience.persona is None

    replaced = pipeline._node_load_request(
        {"request_name": "REQ", "models": [], "persona_override": "digital_shopaholic"}
    )["req"]
    assert replaced.audience.persona == "digital_shopaholic"


def test_versioned_prompt_cache_paths_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setattr(output_manager, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path))
    assert output_manager.versioned_prompt_cache_file("REQ", "v1").name == "built_prompt-v1.json"
    assert output_manager.versioned_prompt_cache_file("REQ", "v2").name == "built_prompt-v2.json"
    assert output_manager.versioned_prompt_cache_file("REQ", "v1") != output_manager.versioned_prompt_cache_file("REQ", "v2")


def test_effective_group_max_uses_catalog_uniform_in_manual_mode():
    req = _request(mode="manual", techniques=["radial_spotlight"])
    assert technique_selector._effective_group_max(req, catalog_max_per_group=3) == {
        "background": 3,
        "decoration": 3,
        "product": 3,
    }


def test_effective_group_max_uses_request_auto_config_in_auto_mode():
    req = _request(
        mode="auto",
        auto_config={
            "max_background_techniques": 2,
            "max_product_techniques": 1,
            "max_decoration_techniques": 0,
        },
    )
    assert technique_selector._effective_group_max(req, catalog_max_per_group=3) == {
        "background": 2,
        "decoration": 0,
        "product": 1,
    }


def test_fill_dominant_color_placeholder_uses_literal_value_when_set():
    text = "Use {{dominant_color}} as the theme color."
    assert technique_selector._fill_dominant_color_placeholder(text, "#B8E8B0") == (
        "Use #B8E8B0 as the theme color."
    )


def test_fill_dominant_color_placeholder_falls_back_when_empty():
    filled = technique_selector._fill_dominant_color_placeholder("{{dominant_color}}", "")
    assert filled == "OCB's approved brand theme color (no explicit dominant_color set for this request)"


def test_resolve_selection_constraints_text_empty_when_not_auto_mode(monkeypatch, tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "sel.txt").write_text("Use {{dominant_color}}.", encoding="utf-8")
    monkeypatch.setattr(
        technique_selector, "get_settings", lambda: SimpleNamespace(asset_library_dir=tmp_path)
    )
    req = _request(mode="manual", auto_config={"selection_constraints": ["prompts/sel.txt"]})
    assert technique_selector._resolve_selection_constraints_text(req) == ""


def test_resolve_selection_constraints_text_loads_content_and_fills_placeholder(monkeypatch, tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "sel.txt").write_text("Use {{dominant_color}}.", encoding="utf-8")
    monkeypatch.setattr(
        technique_selector, "get_settings", lambda: SimpleNamespace(asset_library_dir=tmp_path)
    )
    req = _request(
        mode="auto",
        auto_config={"selection_constraints": ["/prompts/sel.txt"]},  # leading slash, like the real sample
    )
    req = req.model_copy(update={"design_options": req.design_options.model_copy(update={"dominant_color": "#213887"})})

    text = technique_selector._resolve_selection_constraints_text(req)

    assert "Use #213887." in text
    assert "/prompts/sel.txt" in text  # names the source file it came from


def test_resolve_selection_constraints_text_skips_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        technique_selector, "get_settings", lambda: SimpleNamespace(asset_library_dir=tmp_path)
    )
    req = _request(mode="auto", auto_config={"selection_constraints": ["prompts/does_not_exist.txt"]})
    assert technique_selector._resolve_selection_constraints_text(req) == ""


def test_combined_selection_constraints_append_persona_block():
    req = _request(mode="auto")
    req = req.model_copy(update={"audience": req.audience.model_copy(update={"persona": "family_first_planner"})})
    text = technique_selector._combined_selection_constraints_text(req)
    assert "audience persona family_first_planner" in text
    assert "format_type color/layout still wins" in text
    assert technique_selector._resolve_selection_constraints_text(req) == ""


def test_ocb_omni_selection_constraint_lives_under_lib_prompts():
    path = Path(__file__).resolve().parents[3] / "lib" / "prompts" / "ocb_omni_110926_techniques_sel.txt"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "Aff:" in text
    assert "Mass/Mass Aff" in text
    assert "{{dominant_color}}" in text


def test_run_generate_materializes_auto_selection_before_version_prompt(monkeypatch):
    req = _request(mode="auto")
    guideline, assets = _context()
    graph_state = {
        "req": req,
        "built_prompt": BuiltPrompt(text="base", cache_key="base"),
        "prepared_images": [],
        "guideline": guideline,
        "resolved_assets": assets,
    }

    class FakeGraph:
        def invoke(self, _state):
            return graph_state

    captured: dict = {}

    def fake_select(**kwargs):
        captured["selector"] = kwargs
        return {
            1: technique_selector.TechniqueSelection(
                background=["radial_spotlight"], decoration=[], product=[]
            )
        }

    def fake_build(_name, version_req, _guideline, _labels, **kwargs):
        captured["prompt_req"] = version_req
        captured["prompt_kwargs"] = kwargs
        return BuiltPrompt(text="version prompt", cache_key="version")

    monkeypatch.setattr(pipeline, "_get_graph", lambda: FakeGraph())
    monkeypatch.setattr(pipeline, "select_techniques_for_versions", fake_select)
    monkeypatch.setattr(pipeline, "build_prompt", fake_build)
    monkeypatch.setattr(
        pipeline.output_manager,
        "clear_text_gen_cache",
        lambda _name: (_ for _ in ()).throw(AssertionError("text cache must survive across generate runs")),
    )
    monkeypatch.setattr(pipeline.output_manager, "clear_technique_selection", lambda _name: None)
    monkeypatch.setattr(pipeline, "get_settings", lambda: SimpleNamespace())

    assert pipeline.run_generate("REQ", []) == []
    assert captured["selector"]["assets"] is assets
    assert captured["prompt_req"].design_options.techniques == ["radial_spotlight"]
    assert captured["prompt_kwargs"]["cache_variant"] == "v1"
