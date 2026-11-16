"""One-off/pre-batch check: validate every `input/*.json` against the
`CreativeRequest` schema, resolve every bucket/reference asset, and look up
its guideline -- without generating any image. Run this before a big batch
of `run_cli.py` calls (or in CI) to catch a broken request.json / missing
asset file / typo'd format_type up front, instead of discovering it deep
into a `mode=all` run.

This is a preparation/validation script (`script/`), not runtime app logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.asset_resolver import resolve_assets  # noqa: E402
from app.core.guidelines import resolve_guideline  # noqa: E402
from app.models.schemas import CreativeRequest  # noqa: E402
from app.settings import get_settings  # noqa: E402


def _check_one(path) -> list[str]:
    problems: list[str] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"invalid JSON: {exc}"]

    try:
        req = CreativeRequest.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        return [f"schema validation failed: {exc}"]

    assets = resolve_assets(req)
    for a in assets.all_resolved:
        if not a.ok:
            problems.append(f"asset not found -- {a.label}={a.value!r}: {a.error}")

    guideline = resolve_guideline(req.format_type)
    if guideline.missing_entries:
        problems.append(f"format_type={req.format_type!r} -- guideline entries not found: {guideline.missing_entries}")

    try:
        req.canvas_wh()
        req.final_output_wh()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not resolve canvas/output size: {exc}")

    return problems


def main() -> int:
    input_dir = get_settings().input_dir
    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"No *.json files found under {input_dir}")
        return 1

    total_problems = 0
    for f in files:
        problems = _check_one(f)
        status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
        print(f"{f.stem:<30} {status}")
        for p in problems:
            print(f"    - {p}")
        total_problems += len(problems)

    print(f"\n{len(files)} request file(s) checked, {total_problems} total problem(s).")
    return 1 if total_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   cd version_1 && python script/validate_requests.py
#
# Yêu cầu trước khi chạy: `.env` + `input/`, `lib/`, `config/model_usage.json`,
# `lib/guidelines/guidelines.json` đã có (không gọi network, không tốn API).
# ---------------------------------------------------------------------------
