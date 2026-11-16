"""Smoke tier -- app boots, every endpoint webapp/utils/api_client.py calls
responds with a valid shape, using a REAL request.json + REAL bucket/
guideline assets already in this project (input/DB_NOTI_AUTOPAY_NOEL.json) with only
the outbound image-model HTTP call mocked (`fake_generate_image` fixture --
see 3-project-evaluation.md for why network calls are out of scope here).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(isolated_output, isolated_safe_area_guide_dir):
    from app.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_models_lists_providers_and_usage(client):
    r = client.get("/models")
    body = r.json()
    assert r.status_code == 200
    assert "gpt-image-2" in body["available_providers"]
    assert body["model_usage_error"] is None
    assert "default" in body["model_usage"]


def test_list_requests_includes_real_fixture_request(client):
    r = client.get("/requests")
    assert "DB_NOTI_AUTOPAY_NOEL" in r.json()["requests"]


def test_get_request_resolves_all_bucket_assets(client):
    r = client.get("/requests/DB_NOTI_AUTOPAY_NOEL")
    body = r.json()
    assert body["ok"] is True
    assert len(body["resolved_assets"]) >= 1
    assert all(a["ok"] for a in body["resolved_assets"]), body["resolved_assets"]


def test_get_request_missing_file_reports_ok_false_not_500(client):
    r = client.get("/requests/DOES_NOT_EXIST_ABC123")
    assert r.status_code == 200  # never a raw 500 -- REQUEST.md: dump, don't crash
    assert r.json()["ok"] is False


def test_generate_end_to_end_quality_mode(client, fake_generate_image):
    r = client.post("/generate", json={"request_name": "DB_NOTI_AUTOPAY_NOEL", "mode": "quality", "custom_models": []})
    body = r.json()
    assert r.status_code == 200
    assert body["ok"] is True
    # 2026-08-25 (config/model_usage.json, "_note_2026-08-25_gpt-image-2_removed_from_quality"):
    # "quality" mode was locked down to 1 model (gemini-3.1-flash-image) x
    # this fixture's num_version=1 -- this used to assert 3 (gpt-image-2 was
    # part of "quality" before that change, and left stale here after it).
    assert len(body["results"]) == 1
    assert all(res["ok"] for res in body["results"])
    # Vision checks still run against this fixture's fake solid-color image.
    # A missing product is a concept blocker → one regenerate; containment
    # alone is a warning → keep. Either way the provider was invoked.
    assert 1 <= len(fake_generate_image) <= 2

    # 2026-08-20 (user request): every final image must have a raw (pre-crop)
    # sibling of the SAME filename under an ORG/ subfolder next to it, and
    # (since DB_NOTI_AUTOPAY_NOEL declares a real crop, generation.canvas ->
    # output.size) the raw one must actually be the LARGER, uncropped canvas
    # size -- read from the real request.json rather than hardcoded, so this
    # doesn't silently go stale again if the fixture request is edited (as
    # happened before: this used to hardcode (1200, 800) while the real
    # request.json's canvas had already been changed to 1536x1024).
    from PIL import Image

    from app.core import output_manager
    from app.core.canvas_planner import plan_request_geometry
    from app.core.pipeline import load_request_json

    validation_reports = list(
        (output_manager.request_output_dir("DB_NOTI_AUTOPAY_NOEL") / "_validation").glob(
            "validate-*-attempt-*.txt"
        )
    )
    assert validation_reports, "expected a human-readable validation txt for nghiệp vụ"
    report_text = validation_reports[0].read_text(encoding="utf-8")
    assert "BÁO CÁO VALIDATE ẢNH" in report_text
    assert "QUYẾT ĐỊNH" in report_text
    final_reports = list(
        (output_manager.request_output_dir("DB_NOTI_AUTOPAY_NOEL") / "_validation").glob(
            "validate-*-FINAL.txt"
        )
    )
    assert final_reports, "expected FINAL validation summary for nghiệp vụ"
    assert "ẢNH ĐƯỢC XUẤT" in final_reports[0].read_text(encoding="utf-8")

    fixture_req = load_request_json("DB_NOTI_AUTOPAY_NOEL")
    planned_req, _ = plan_request_geometry(fixture_req, ["gemini-3.1-flash-image"])
    for res in body["results"]:
        final_path = Path(res["path"])
        org_path = output_manager.org_image_path("DB_NOTI_AUTOPAY_NOEL", res["display_size"], res["image_model"], final_path.name)
        assert org_path.exists(), f"missing raw (pre-crop) image at {org_path}"
        assert org_path.parent == final_path.parent / "ORG"
        assert Image.open(org_path).size == planned_req.canvas_wh()  # auto-planned native canvas -- NOT cropped
        assert Image.open(final_path).size == fixture_req.final_output_wh()  # output.size -- cropped

    # 2026-08-20 (2) (user request, REQUEST.md [3d]): the exact prompt text
    # sent to the model must be logged once per (model, version) under
    # output/<request_name>/prompt/prompt-<model>-version-<n>.txt -- assert
    # this for all 3 (gpt-image-2, version 1..3) results of this run, and
    # that the saved text is the real non-empty prompt (not an empty stub).
    for res in body["results"]:
        prompt_path = output_manager.prompt_file_path("DB_NOTI_AUTOPAY_NOEL", res["image_model"], res["version"])
        assert prompt_path.exists(), f"missing prompt log at {prompt_path}"
        assert prompt_path.name == f"prompt-{res['image_model']}-version-{res['version']}.txt"
        saved_text = prompt_path.read_text(encoding="utf-8")
        assert len(saved_text) > 100  # a real assembled prompt, not an empty/stub file

        # 2026-08-20 (2) (user request): the saved log must also list, by
        # actual filename, every image that was attached to the model call
        # -- not just the generic bucket labels already inside the prompt
        # text itself.
        assert "ATTACHED IMAGES SENT WITH THIS PROMPT" in saved_text
        assert "[theme]" in saved_text
        assert "[product]" in saved_text

        # The output-driven canvas planner leaves only a small ratio-mismatch
        # crop here. Text geometry is sufficient, so do not attach technical
        # guide/spec images that the model could copy into the artwork.
        assert "CRITICAL REMINDER" in saved_text
        assert "[safe_area_guide]" not in saved_text
        assert "[guideline]" not in saved_text


def test_generate_unknown_mode_with_no_custom_models_fails_cleanly(client):
    r = client.post("/generate", json={"request_name": "DB_NOTI_AUTOPAY_NOEL", "mode": "custom", "custom_models": []})
    body = r.json()
    assert body["ok"] is False
    assert "No models resolved" in body["error"]


def test_outputs_latest_and_at_timestamp(client, fake_generate_image):
    # 2026-08-25 (config/model_usage.json): "quality" mode is 1 model
    # (gemini-3.1-flash-image) x this fixture's num_version=1 -- these used
    # to assert 3 / match on "gpt-image-2" from before that change, left
    # stale here after it.
    gen = client.post("/generate", json={"request_name": "DB_NOTI_AUTOPAY_NOEL", "mode": "quality", "custom_models": []})
    assert gen.json()["ok"] is True

    latest = client.get("/outputs/DB_NOTI_AUTOPAY_NOEL/latest")
    assert latest.json()["ok"] is True
    assert len(latest.json()["images"]) == 1

    at_ts = client.get("/outputs/DB_NOTI_AUTOPAY_NOEL/at-timestamp", params={"ts": "gemini-3.1-flash-image"})
    assert len(at_ts.json()["images"]) == 1


def test_files_serves_generated_image_and_blocks_outside_paths(client, fake_generate_image):
    gen = client.post("/generate", json={"request_name": "DB_NOTI_AUTOPAY_NOEL", "mode": "quality", "custom_models": []})
    first_path = gen.json()["results"][0]["path"]

    ok = client.get("/files", params={"path": first_path})
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/png"

    blocked = client.get("/files", params={"path": "/etc/passwd"})
    assert blocked.status_code == 403


def test_release_and_clone_from_release_round_trip(client, fake_generate_image):
    gen = client.post("/generate", json={"request_name": "DB_NOTI_AUTOPAY_NOEL", "mode": "quality", "custom_models": []})
    first_path = gen.json()["results"][0]["path"]

    rel = client.post("/release", json={"request_name": "DB_NOTI_AUTOPAY_NOEL", "selected_output_paths": [first_path]})
    assert rel.json()["ok"] is True
    released_path = rel.json()["released_paths"][0]

    got = client.get("/release/DB_NOTI_AUTOPAY_NOEL")
    assert released_path in got.json()["released"]

    clone = client.post(
        "/clone-from-release",
        json={"request_name": "DB_NOTI_AUTOPAY_NOEL", "released_image_path": released_path, "display_sizes": ["256,256"], "image_model": "gpt-image-2"},
    )
    assert clone.json()["ok"] is True
    assert clone.json()["results"][0]["ok"] is True

    status = client.get("/release/DB_NOTI_AUTOPAY_NOEL/clone-status", params={"since_ts": 0, "source_stem": ""})
    assert len(status.json()["files"]) == 1
