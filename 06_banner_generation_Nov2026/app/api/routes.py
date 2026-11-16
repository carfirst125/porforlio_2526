"""FastAPI routes (REQUEST.md [6]). This is the ONLY HTTP surface the
Streamlit webapp talks to (`webapp/utils/api_client.py`) -- every endpoint
name/shape here is dictated by that client + the view modules that call it
(`webapp/views/generate.py`, `webapp/views/release.py`,
`webapp/views/home.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core import model_usage, output_manager, personas, pipeline
from app.core.asset_resolver import resolve_assets
from app.logging_conf import get_logger
from app.models.schemas import (
    CloneFromReleaseBody,
    CloneRequestBody,
    CreativeRequest,
    GenerateRequestBody,
    ReleaseRequestBody,
    parse_wh,
    wh_slug,
)
from app.providers.registry import available_providers
from app.settings import get_settings

logger = get_logger("api")
router = APIRouter()


# ---------------------------------------------------------------------------
# Health / models
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    settings = get_settings()
    return {"ok": True, "app": settings.app_name, "environment": settings.environment}


@router.get("/models")
def models():
    usage, error = model_usage.load_model_usage()
    return {
        "available_providers": available_providers(),
        "model_usage": usage,
        "model_usage_error": error,
    }


# ---------------------------------------------------------------------------
# Requests (input/*.json)
# ---------------------------------------------------------------------------

@router.get("/personas")
def list_personas(request_name: str | None = Query(default=None)):
    """Catalog of the 8 RB card-spend personas, with a per-product `suggested`
    flag so the webapp can mark (not hide) a weak product×persona fit."""
    family = personas.product_family_for(request_name or "")
    items = []
    for persona in personas.load_catalog():
        payload = persona.as_api_dict()
        payload["suggested"] = (
            True if family is None else family in persona.suggested_product_families
        )
        items.append(payload)
    return {"ok": True, "personas": items, "product_family": family}


@router.get("/requests")
def list_requests():
    input_dir = get_settings().input_dir
    if not input_dir.exists():
        return {"requests": []}
    names = sorted(p.stem for p in input_dir.glob("*.json"))
    return {"requests": names}


@router.get("/requests/{request_name}")
def get_request(request_name: str):
    path = get_settings().input_dir / f"{request_name}.json"
    if not path.exists():
        return {"ok": False, "error": f"request file not found: {path}"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not parse %s", path)
        return {"ok": False, "error": f"invalid JSON: {exc}"}

    try:
        req = CreativeRequest.model_validate(raw)
        resolved = resolve_assets(req)
        resolved_assets = [a.model_dump() for a in resolved.all_resolved]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not validate/resolve assets for %s", path)
        return {"ok": True, "request": raw, "resolved_assets": [], "schema_error": str(exc)}

    return {"ok": True, "request": raw, "resolved_assets": resolved_assets}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@router.get("/outputs/{request_name}")
def list_outputs(request_name: str, since_ts: float = Query(0.0)):
    images = output_manager.list_new_images(request_name, since_ts)
    return {"ok": True, "images": images}


@router.get("/outputs/{request_name}/latest")
def latest_outputs(request_name: str):
    images, run_ts = output_manager.latest_run_images(request_name)
    return {"ok": True, "images": images, "run_timestamp": run_ts}


@router.get("/outputs/{request_name}/at-timestamp")
def outputs_at_timestamp(request_name: str, ts: str = Query(...)):
    images = output_manager.images_matching_substring(request_name, ts)
    return {"ok": True, "images": images}


# ---------------------------------------------------------------------------
# Generate / Clone
# ---------------------------------------------------------------------------

@router.post("/generate")
def generate(body: GenerateRequestBody):
    logger.info("POST /generate request_name=%s mode=%s custom_models=%s", body.request_name, body.mode, body.custom_models)
    usage, usage_error = model_usage.load_model_usage()
    if usage_error:
        return {"ok": False, "error": f"config/model_usage.json error: {usage_error}"}

    models_to_run = model_usage.resolve_models(body.mode, body.custom_models, usage)
    if not models_to_run:
        return {"ok": False, "error": f"No models resolved for mode={body.mode!r} -- check model_usage.json / custom_models."}

    try:
        results = pipeline.run_generate(
            body.request_name,
            models_to_run,
            num_version_override=body.num_version_override,
            generate_person_override=body.generate_person_override,
            persona_override=body.persona_override,
        )
        return {"ok": True, "results": results}
    except Exception as exc:  # noqa: BLE001 -- REQUEST.md: dump, never 500 silently
        logger.exception("POST /generate crashed for request_name=%s", body.request_name)
        return {"ok": False, "error": str(exc)}


@router.post("/clone")
def clone(body: CloneRequestBody):
    logger.info("POST /clone request_name=%s source=%s model=%s", body.request_name, body.source_image_path, body.select_model)
    try:
        req_path = get_settings().input_dir / f"{body.request_name}.json"
        raw = json.loads(req_path.read_text(encoding="utf-8"))
        req = CreativeRequest.model_validate(raw)
        display_sizes = req.output.clone
        if not display_sizes:
            return {"ok": False, "error": f"request '{body.request_name}' has no output.clone sizes configured."}
        results = pipeline.run_clone(body.request_name, body.source_image_path, body.select_model, display_sizes)
        return {"ok": True, "results": results}
    except Exception as exc:  # noqa: BLE001
        logger.exception("POST /clone crashed for request_name=%s", body.request_name)
        return {"ok": False, "error": str(exc)}


@router.post("/clone-from-release")
def clone_from_release(body: CloneFromReleaseBody):
    logger.info(
        "POST /clone-from-release request_name=%s source=%s sizes=%s model=%s",
        body.request_name,
        body.released_image_path,
        body.display_sizes,
        body.image_model,
    )
    try:
        results = pipeline.run_clone(body.request_name, body.released_image_path, body.image_model, body.display_sizes)
        return {"ok": True, "results": results}
    except Exception as exc:  # noqa: BLE001
        logger.exception("POST /clone-from-release crashed for request_name=%s", body.request_name)
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

@router.post("/release")
def release(body: ReleaseRequestBody):
    logger.info("POST /release request_name=%s count=%d", body.request_name, len(body.selected_output_paths))
    results = output_manager.release_images(body.request_name, body.selected_output_paths)
    released_paths = [r.released_path for r in results if r.ok and r.released_path]
    failed = [{"source_path": r.source_path, "error": r.error} for r in results if not r.ok]
    return {"ok": True, "released_paths": released_paths, "failed": failed}


@router.get("/release/{request_name}")
def get_released(request_name: str):
    return {"ok": True, "released": output_manager.list_released(request_name)}


@router.get("/release/{request_name}/clone-status")
def clone_status(request_name: str, since_ts: float = Query(0.0), source_stem: str = Query("")):
    files = output_manager.clone_files_for_source(request_name, since_ts=since_ts, source_stem=source_stem)
    return {"ok": True, "files": files}


# ---------------------------------------------------------------------------
# File serving (`webapp/utils/api_client.py::image_url`) -- backend re-serves
# any file it manages so the webapp process never needs filesystem access to
# the backend's disk (see comment at the top of `api_client.py`).
# ---------------------------------------------------------------------------

def _allowed_roots() -> list[Path]:
    settings = get_settings()
    return [
        settings.output_dir.resolve(),
        settings.input_dir.resolve(),
        settings.asset_library_dir.resolve(),
        settings.guidelines_file.parent.resolve(),
    ]


@router.get("/files")
def get_file(path: str = Query(...)):
    # 2026-08-26: the allowed-directory check must run BEFORE the
    # existence check -- otherwise a path-traversal attempt for a file
    # that happens not to exist (e.g. `/etc/passwd` on a Windows host)
    # returns 404 instead of 403, leaking whether the path exists at all
    # to a caller who was never authorized to ask.
    target = Path(path).resolve()
    if not any(target == root or root in target.parents for root in _allowed_roots()):
        raise HTTPException(status_code=403, detail="path is outside the served directories")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")
    return FileResponse(target)
