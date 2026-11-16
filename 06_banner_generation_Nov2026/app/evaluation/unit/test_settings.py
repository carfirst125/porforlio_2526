"""Unit tests -- app/settings.py's `_resolve_dir()` helper.

2026-08-20 (self-discovered while investigating a user-reported
`ConnectionResetError`, see `2-project-technote.md`'s run instructions for
the actual root cause of that specific bug -- an unrelated uvicorn
`--reload` issue): while reading the app's own startup log during that
investigation, `asset_library_dir` had resolved to a `lib/` folder inside a
completely different, unrelated project directory -- because the backend
process had been launched from that directory (which happened to also
contain its own `lib/` subfolder) instead of `version_1/`, and
`_resolve_dir()`'s old `raw.exists()` check resolved a relative default
path against the process's CURRENT WORKING DIRECTORY rather than
`PROJECT_ROOT`. These tests pin down the fix: a relative default/`.env`
value must resolve consistently, independent of the process's cwd at
launch time."""
from __future__ import annotations

from pathlib import Path

from app.settings import PROJECT_ROOT, _resolve_dir


def test_relative_path_that_exists_under_project_root_resolves_there():
    # "lib" is a real directory under PROJECT_ROOT in this repo.
    result = _resolve_dir(Path("./lib"), PROJECT_ROOT / "does_not_matter")
    assert result == (PROJECT_ROOT / "lib").resolve()


def test_relative_path_resolution_does_not_depend_on_process_cwd(tmp_path, monkeypatch):
    """The exact bug found: launching the process from an unrelated
    directory that HAPPENS to also contain a same-named subfolder must NOT
    cause that unrelated folder to be picked up."""
    unrelated_dir = tmp_path / "some_other_project"
    decoy = unrelated_dir / "lib"
    decoy.mkdir(parents=True)

    monkeypatch.chdir(unrelated_dir)
    try:
        result = _resolve_dir(Path("./lib"), PROJECT_ROOT / "does_not_matter")
    finally:
        pass  # monkeypatch restores the original cwd automatically on teardown

    # Must resolve to THIS project's real lib/, never the decoy that only
    # exists relative to the directory the process happened to start in.
    assert result == (PROJECT_ROOT / "lib").resolve()
    assert result != decoy.resolve()


def test_relative_path_that_does_not_exist_anywhere_falls_back(tmp_path):
    fallback = tmp_path / "fallback_dir"
    result = _resolve_dir(Path("./this_directory_should_never_exist_xyz"), fallback)
    assert result == fallback.resolve()


def test_absolute_path_that_exists_is_honored_regardless_of_cwd(tmp_path, monkeypatch):
    real_dir = tmp_path / "explicit_override"
    real_dir.mkdir()
    monkeypatch.chdir(tmp_path)  # cwd irrelevant -- the path is absolute

    result = _resolve_dir(real_dir, PROJECT_ROOT / "does_not_matter")
    assert result == real_dir.resolve()


def test_absolute_path_that_does_not_exist_falls_back(tmp_path):
    fallback = tmp_path / "fallback_dir"
    missing_absolute = tmp_path / "this_absolute_path_does_not_exist"
    result = _resolve_dir(missing_absolute, fallback)
    assert result == fallback.resolve()


def test_empty_string_path_resolves_to_project_root_not_process_cwd():
    """`Path("")` normalizes to `Path(".")` (a pre-existing quirk, not
    introduced by this fix), which is always truthy and always exists --
    so this can't fall back to `fallback` the way a genuinely-missing path
    does. What matters for the bug this module fixes: it must resolve
    against PROJECT_ROOT, not whatever the process's cwd happens to be."""
    fallback = PROJECT_ROOT / "lib"
    assert _resolve_dir(Path(""), fallback) == PROJECT_ROOT.resolve()
