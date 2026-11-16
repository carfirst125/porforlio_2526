"""Shared fixtures for `app/evaluation/{unit,smoke,stress}` -- see
`3-project-evaluation.md` for the methodology these tests implement.
"""
from __future__ import annotations

import io
import json

import pytest
from PIL import Image


@pytest.fixture
def isolated_output(monkeypatch, tmp_path):
    """Redirects `Settings.output_dir` to a throwaway `tmp_path` for the
    duration of 1 test, so evaluation tests NEVER write into the real
    `output/` directory the webapp/CLI use."""
    from app.settings import get_settings

    settings = get_settings()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    monkeypatch.setattr(settings, "output_dir", out_dir)
    return out_dir


@pytest.fixture
def isolated_safe_area_guide_dir(monkeypatch, tmp_path):
    """Redirects `Settings.safe_area_guide_dir` to a throwaway `tmp_path`
    (2026-08-20 (4): the safe-area-guide cache moved from per-request,
    under `output/<request_name>/...`, to a SHARED library directory,
    `./lib/safe_area_guide` by default -- tests need their OWN isolated
    fixture for it now, separate from `isolated_output`, so evaluation runs
    never read/write the real `lib/safe_area_guide/` cache on disk)."""
    from app.settings import get_settings

    settings = get_settings()
    guide_dir = tmp_path / "safe_area_guide"
    monkeypatch.setattr(settings, "safe_area_guide_dir", guide_dir)
    return guide_dir


def _fake_png_bytes(width: int, height: int, color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def two_requests_isolated_input(monkeypatch, tmp_path):
    """Stress tier only: redirects `Settings.input_dir` to a throwaway
    `tmp_path` containing 2 DIFFERENT request.json files, so the
    cross-talk-between-requests test in `app/evaluation/stress/` doesn't
    depend on there being 2+ real files under the project's own `input/`
    directory (only `DB_NOTI_AUTOPAY_NOEL.json` is guaranteed to exist there
    -- see `4-project-evaluation-results.md` 2026-08-20). Both requests
    reference REAL asset files already under `lib/` (`Settings.
    asset_library_dir` is left untouched, still pointing at the real
    project `lib/`), so `resolve_assets()` still resolves real images."""
    import shutil

    from app.settings import get_settings

    settings = get_settings()
    real_input_dir = settings.input_dir

    in_dir = tmp_path / "input"
    in_dir.mkdir()

    # 1st request: exact copy of the real, current DB_NOTI_AUTOPAY_NOEL.json (has a crop).
    shutil.copy2(real_input_dir / "DB_NOTI_AUTOPAY_NOEL.json", in_dir / "DB_NOTI_AUTOPAY_NOEL.json")

    # 2nd request: synthetic, reuses real lib/ assets, exercises the NO-crop path
    # (canvas == safe_area == output.size) for extra coverage.
    second = {
        "creative_id": "STRESS_SECOND_REQ",
        "format_type": "fb_post_banner",
        "design_brief": "Synthetic 2nd request for the stress cross-talk test (not a real production request).",
        "buckets": {"theme": "Bucket_1/theme_promotion.jpg", "context": [], "product": "Bucket_3/prod_card_lifestyle.png"},
        "reference": "",
        "text": {"headline": "H", "message": "M", "cta_label": "C", "cta_deep_link": ""},
        "guidelines": "",
        "num_version": 3,
        "design_options": {"generate_person": False, "has_logo": False},
        "generation": {
            "canvas": {"width": 1200, "height": 1200},
            "safe_area": {"width": 1200, "height": 1200, "position": "center", "x": 0, "y": 0},
            "instruction": "",
        },
        "output": {"size": {"width": 1200, "height": 1200}, "clone": []},
    }
    (in_dir / "STRESS_SECOND_REQ.json").write_text(json.dumps(second, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(settings, "input_dir", in_dir)
    return ["DB_NOTI_AUTOPAY_NOEL", "STRESS_SECOND_REQ"]


@pytest.fixture
def fake_generate_image(monkeypatch):
    """Monkeypatches image-provider entry points with an instant, offline
    fake -- returns a solid-color PNG at the requested size instead of
    hitting a real Gemini/Azure endpoint. Used by every smoke/stress test so
    the evaluation suite runs with zero network calls and zero API cost.

    Patches both `pipeline.generate_image` (clone path) and
    `generate_repair_graph.generate_image` (generate → validate → repair graph).
    """
    call_log: list[dict] = []

    def _fake(model_name, prompt, attachments, width, height, settings, usage_sink=None):
        call_log.append({"model": model_name, "width": width, "height": height, "n_attachments": len(attachments), "prompt_len": len(prompt)})
        assert prompt, "prompt must never be empty when calling a provider"
        # 2026-09-06: real providers fill `usage_sink` in with their own
        # reported token counts (see app/providers/*.py) -- the fake reports
        # a small fixed amount so usage_tracker.py's plumbing is exercised by
        # any test that inspects it, without depending on real API usage.
        if usage_sink is not None:
            usage_sink["provider"] = "fake"
            usage_sink["input_tokens"] = 100
            usage_sink["output_tokens"] = 50
        return _fake_png_bytes(width, height)

    import app.core.generate_repair_graph as generate_repair_module
    import app.core.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "generate_image", _fake)
    monkeypatch.setattr(generate_repair_module, "generate_image", _fake)
    return call_log
