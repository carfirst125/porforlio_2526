"""Unit tests -- app/core/output_manager.py naming/listing conventions.
Uses the `isolated_output` fixture (app/evaluation/conftest.py) so nothing
here ever touches the real output/ directory."""
from __future__ import annotations

import os
import time

from app.core import output_manager


def test_image_filename_no_suffix_for_single_version():
    name = output_manager.image_filename("ID", "512x512", "gpt-image-2", "2026-08-19-1000", version=1, num_version=1)
    assert name == "ID_512x512_gpt-image-2_2026-08-19-1000.png"


def test_image_filename_has_version_suffix_when_multiple_versions():
    name = output_manager.image_filename("ID", "512x512", "gpt-image-2", "2026-08-19-1000", version=2, num_version=3)
    assert name == "ID_512x512_gpt-image-2_2026-08-19-1000-v2.png"


def test_image_output_path_shortens_leaf_when_windows_path_would_be_too_long(monkeypatch, tmp_path):
    long_root = tmp_path / ("very-long-output-root-" * 6)
    monkeypatch.setattr(output_manager, "request_output_dir", lambda request_name: long_root / request_name)
    path = output_manager.image_output_path(
        "JSON_V3_USECASE1_PASSPORT_MASS_1024x256",
        "JSON_V3_USECASE1_PASSPORT_MASS_1024x256",
        "1024x256",
        "gemini-3.1-flash-image",
        "2026-08-28-155413",
        1,
        3,
    )
    assert path.name.startswith("img-")
    assert path.name.endswith("_2026-08-28-155413-v1.png")


def test_unique_image_output_path_avoids_collision(isolated_output):
    """Regression test for the race app/evaluation/stress found: 2 writers
    targeting the identical (request, size, model, timestamp, version) must
    never resolve to the same path."""
    first = output_manager.unique_image_output_path("REQ", "ID", "512x512", "model-a", "2026-08-20-000000", 1, 1)
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"first")

    second = output_manager.unique_image_output_path("REQ", "ID", "512x512", "model-a", "2026-08-20-000000", 1, 1)
    assert second != first
    assert not second.exists()
    second.write_bytes(b"second")

    assert first.read_bytes() == b"first"  # first writer's file must survive untouched
    assert second.name.endswith("-dup2.png")


def test_prompt_file_path_includes_model_and_version():
    """2026-08-20 (2) (user request, REQUEST.md [3d]): filename must record
    BOTH which model the prompt was sent to AND which version request it
    was for -- was previously `prompt-<model>.txt` only (no version), which
    silently collapsed every version onto 1 shared file."""
    p1 = output_manager.prompt_file_path("REQ", "gpt-image-2", 1)
    p2 = output_manager.prompt_file_path("REQ", "gpt-image-2", 2)
    assert p1.name == "prompt-gpt-image-2-version-1.txt"
    assert p2.name == "prompt-gpt-image-2-version-2.txt"
    assert p1 != p2  # different versions of the SAME model must not collide
    assert p1.parent == output_manager.prompt_dir("REQ")


def test_org_dir_is_nested_under_the_model_directory():
    """2026-08-20 (user request): raw (pre-crop) images live in an `ORG/`
    subfolder INSIDE `<display_size>/<image_model>/`, not a top-level dir."""
    d = output_manager.org_dir("REQ", "512x512", "gpt-image-2")
    assert d == output_manager.size_model_dir("REQ", "512x512", "gpt-image-2") / "ORG"


def test_org_image_path_reuses_the_final_images_filename():
    """The raw and final (cropped) files must share the same filename --
    that's what pairs them 1:1 across the 2 directories."""
    p = output_manager.org_image_path("REQ", "512x512", "gpt-image-2", "ID_512x512_gpt-image-2_2026-08-20-000000.png")
    assert p.name == "ID_512x512_gpt-image-2_2026-08-20-000000.png"
    assert p.parent == output_manager.org_dir("REQ", "512x512", "gpt-image-2")


def test_org_subfolder_contents_are_invisible_to_listings(isolated_output):
    """ORG/ must never surface in list_new_images/latest_run_images/
    images_matching_substring -- the webapp's Output gallery and "Select for
    release" checkboxes must only ever offer the final cropped image."""
    d = output_manager.size_model_dir("REQ", "512x512", "gpt-image-2")
    d.mkdir(parents=True)
    (d / "ID_512x512_gpt-image-2_2026-08-20-000000.png").write_bytes(b"final")

    org_dir = output_manager.org_dir("REQ", "512x512", "gpt-image-2")
    org_dir.mkdir(parents=True)
    (org_dir / "ID_512x512_gpt-image-2_2026-08-20-000000.png").write_bytes(b"raw")

    images = output_manager.list_new_images("REQ", 0)
    assert len(images) == 1
    assert images[0]["path"] == str((d / "ID_512x512_gpt-image-2_2026-08-20-000000.png").resolve())

    assert len(output_manager.images_matching_substring("REQ", "2026-08-20-000000")) == 1


def test_clone_release_filename_convention():
    name = output_manager.clone_release_filename("SOURCE_STEM", "512x268", "gemini-3-pro-image")
    assert name == "SOURCE_STEM_clone_512x268_gemini-3-pro-image.png"


def test_list_new_images_filters_by_since_ts(isolated_output):
    d = output_manager.size_model_dir("REQ", "512x512", "model-a")
    d.mkdir(parents=True)
    old = d / "old.png"
    old.write_bytes(b"x")
    cutoff = time.time()
    time.sleep(0.05)
    new = d / "new.png"
    new.write_bytes(b"x")

    images = output_manager.list_new_images("REQ", cutoff)
    names = {img["filename"] for img in images}
    assert "new.png" in names
    assert "old.png" not in names


def test_iter_result_pngs_excludes_non_result_dirs(isolated_output):
    req_dir = output_manager.request_output_dir("REQ")
    (req_dir / "512x512" / "model-a").mkdir(parents=True)
    (req_dir / "512x512" / "model-a" / "img.png").write_bytes(b"x")
    (req_dir / "release").mkdir(parents=True)
    (req_dir / "release" / "released.png").write_bytes(b"x")
    (req_dir / "prompt").mkdir(parents=True)
    (req_dir / "prompt" / "prompt-model-a.txt").write_text("x")

    images = output_manager.list_new_images("REQ", 0)
    assert len(images) == 1
    assert images[0]["filename"] == "img.png"


def test_images_matching_substring_case_insensitive(isolated_output):
    d = output_manager.size_model_dir("REQ", "512x512", "gpt-image-2")
    d.mkdir(parents=True)
    (d / "REQ_512x512_gpt-image-2_2026-08-19-1000-v1.png").write_bytes(b"x")

    assert len(output_manager.images_matching_substring("REQ", "GPT-IMAGE-2")) == 1
    assert len(output_manager.images_matching_substring("REQ", "2026-08-19")) == 1
    assert len(output_manager.images_matching_substring("REQ", "no-such-substring")) == 0


def test_release_images_copies_and_reports_missing_source(isolated_output, tmp_path):
    src_dir = output_manager.size_model_dir("REQ", "512x512", "model-a")
    src_dir.mkdir(parents=True)
    src = src_dir / "img.png"
    src.write_bytes(b"hello")

    results = output_manager.release_images("REQ", [str(src), str(tmp_path / "missing.png")])
    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]
    assert len(ok_results) == 1
    assert len(fail_results) == 1
    assert os.path.exists(ok_results[0].released_path)
