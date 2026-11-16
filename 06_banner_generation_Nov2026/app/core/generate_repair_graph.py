"""LangGraph for the generate → validate → repair loop (REQUEST.md step [6]).

This is the part of the pipeline that is actually a graph: validation can
pass, fail-and-repair, or fail-and-keep-best. The linear setup steps
(load request, assets, guideline, downsample, assemble prompt) stay in
`pipeline.py` because they have no branches.

State carries the current attempt plus the best image seen so far so a
repair that regresses never replaces a better original.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.core import output_manager
from app.core.image_post import crop_canvas_to_output
from app.core.output_validator import ValidationReport, validate_output_report, write_validation_report_file
from app.logging_conf import get_logger
from app.models.schemas import CreativeRequest
from app.providers.base import ImageAttachment
from app.providers.registry import generate_image
from app.providers._http import fail_fast_on_quota_exhaustion
from app.settings import Settings

logger = get_logger("generate_repair_graph")

# 2026-08-25 (entry (18)): max attempts per (model, version) -- 1 initial +
# at most one follow-up. Follow-up is an edit (layout blocker / 2+ warnings)
# or a regenerate from the original prompt (concept blocker). A single
# warning such as "too close to the edge" does not spend another image call.
_MAX_GENERATE_ATTEMPTS = 2


class GenerateRepairState(TypedDict, total=False):
    request_name: str
    model: str
    version: int
    req: CreativeRequest
    prompt: str
    original_prompt: str
    attachments: list[ImageAttachment]
    original_attachments: list[ImageAttachment]
    canvas_wh: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    usage_tracker: Any
    settings: Settings
    attempt: int
    max_attempts: int
    raw_bytes: bytes | None
    cropped_bytes: bytes | None
    problems: list[str]
    repair_mode: str
    validation_score: int
    best_raw: bytes | None
    best_cropped: bytes | None
    best_problems: list[str]
    best_score: int
    best_attempt: int
    generate_failed: bool


class GenerateRepairResult(TypedDict):
    raw_bytes: bytes
    cropped_bytes: bytes
    problems: list[str]


def _mime_for_image_bytes(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/png"


def _build_targeted_repair(
    req: CreativeRequest,
    failed_raw_bytes: bytes,
    original_attachments: list[ImageAttachment],
    problems: list[str],
) -> tuple[str, list[ImageAttachment]]:
    """Build a second-pass edit request, preserving the first design.

    The failed full canvas is attachment #1 so content lost by the final crop
    remains available. Technical safe-area guides are deliberately omitted.
    """
    canvas_w, canvas_h = req.canvas_wh()
    crop_x, crop_y, crop_w, crop_h = req.crop_box()
    reusable = [att for att in original_attachments if att.label != "safe_area_guide"]
    repair_attachments = [
        ImageAttachment(label="failed_full_canvas", mime_type=_mime_for_image_bytes(failed_raw_bytes), data=failed_raw_bytes),
        *reusable,
    ]
    attachment_lines = ["1. PREVIOUS FULL-CANVAS DESIGN TO REPAIR"]
    attachment_lines.extend(f"{index}. ORIGINAL {att.label.upper()} REFERENCE" for index, att in enumerate(reusable, start=2))
    # `text_by_code` requests never want the model drawing text at all (see
    # prompt_builder.py's TEXT section) -- the repair pass must repeat that
    # instruction instead of demanding exact text, or it would reintroduce
    # AI-drawn text into a request that composites text by code afterward.
    text_requirement = (
        "REQUIRED TEXT: none -- do NOT render any headline, message, CTA button, or other text anywhere; "
        "text is added separately after generation."
        if req.design_options.text_by_code
        else f"REQUIRED TEXT (exact, complete, no truncation): {[value for value in (req.text.headline, req.text.message, req.text.cta_label) if value]!r}"
    )
    logo_requirement = (
        "- has_logo=false: remove every standalone/corner/watermark/repeated OCB logo. Keep branding only when "
        "it is printed inside the unchanged boundary of a supplied product/context asset; never extract or duplicate it.\n"
        if not req.design_options.has_logo
        else "- has_logo=true: keep the required standalone logo complete, accurate and unobscured.\n"
    )

    prompt = (
        "TARGETED LAYOUT REPAIR — do not create a new concept.\n"
        "The FIRST attached image is the previous full-canvas banner. Preserve its background, campaign idea, "
        "colors, typography, decorative styling, and overall composition as closely as possible. Make only the "
        "minimum layout changes required to fix the failed validation below.\n\n"
        f"FAILED CHECKS:\n- {'; '.join(problems)}\n\n"
        f"OUTPUT GEOMETRY:\n- Recreate the full canvas at {canvas_w}x{canvas_h}.\n"
        f"- Delivery post-processing keeps {crop_w}x{crop_h} beginning at ({crop_x}, {crop_y}).\n"
        "- Move and/or uniformly scale important content so every required element remains complete and "
        "comfortably visible in the delivered banner.\n"
        "- Preserve one continuous campaign scene, with its scenery, colors, lighting, and decoration flowing "
        "naturally throughout the complete canvas.\n\n"
        f"{text_requirement}\n\n"
        "LOCKED ASSETS:\n"
        "- Any supplied product must keep its exact logo, text, artwork, width-to-height aspect ratio, original "
        "portrait/landscape and front/back orientation, reading direction, material and realistic appearance. Do "
        "not mirror, flip, stretch, squash, rotate into the opposite orientation, redraw, redesign or replace it. "
        "Use the original supplied product pixels/appearance as the foreground asset; keep its source camera angle "
        "and silhouette, and do not invent a new perspective, side face, thickness, casing or frame.\n"
        "- A context attachment is not automatically a person: preserve what it actually contains. If it is only "
        "an icon/object/symbol, do not invent a photographed person or family. If it contains a real person, keep "
        "the same identity, pose, clothing and photographic appearance.\n"
        "- Hex colors, font names, pixel values and layout instructions are non-printing metadata. If the "
        "previous banner shows a spec line such as 'dE39717 Manrope Semi bold', '#E39717', 'E39717', "
        "'Manrope Semibold', a px size, or padding — including a hex glued onto a phrase like "
        "'gọn nhẹ E39717' — DELETE that code. Keep only the required advertising words above, filled in "
        "the approved colors. Do not leave an extra gold/orange spec line between headline and message.\n"
        f"{logo_requirement}"
        "- Recover any part visible in the previous FULL canvas but lost by the delivered crop by moving or "
        "scaling it inward; do not merely erase the clipped content.\n\n"
        # 2026-09-11 (user report, DB_FLEXI_1024x440_AFF output): the repair
        # pass had no anti-border guard at all, unlike the first-pass prompt
        # in prompt_builder.py (see its "FULL-BLEED BACKGROUND" section
        # above, added 2026-09-03 for the exact same defect). The clean v1
        # image for this request had no border; the repair pass then drew a
        # thin gold decorative line around the entire canvas that nobody
        # asked for. Repeating the same unconditional guard here so every
        # repair pass carries the same guarantee as the original generation.
        "FULL-BLEED BACKGROUND (no exceptions): The background -- its color, texture, lighting, and scenery -- "
        "must reach and continue naturally across all four canvas edges with zero gap. Do NOT draw a decorative "
        "border, outline, framed panel, rounded card/mat, vignette ring, or any other enclosing line or shape "
        "around the banner's content, no matter how premium, on-brand, or intentional it may look -- this applies "
        "regardless of color, opacity, or thickness (a crisp solid gold/metallic line is just as forbidden as a "
        "faint one). If the previous full-canvas design attached above already has such a border/outline/frame, "
        "REMOVE it and extend the background naturally to the edge instead of preserving it.\n\n"
        "Return one clean finished campaign banner whose background remains visually uninterrupted across the "
        "entire image.\n\n"
        "ATTACHMENTS IN ORDER:\n" + "\n".join(attachment_lines)
    )
    return prompt, repair_attachments


def _write_repair_prompt_log(request_name: str, model: str, version: int, attempt: int, prompt: str) -> None:
    path = output_manager.repair_prompt_file_path(request_name, model, version, attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    logger.info("Wrote targeted repair prompt log -> %s", path)


def _write_validation_report(state: GenerateRepairState, report: ValidationReport) -> None:
    path = output_manager.validation_report_file_path(
        state["request_name"], state["model"], state["version"], state["attempt"]
    )
    write_validation_report_file(
        report,
        path,
        request_name=state["request_name"],
        model=state["model"],
        version=state["version"],
        attempt=state["attempt"],
    )
    logger.info("Wrote validation report -> %s", path)


def _write_final_validation_summary(state: GenerateRepairState) -> None:
    """One file nghiệp vụ can open last: which attempt was actually exported."""
    delivered_attempt = state.get("best_attempt") or state.get("attempt") or 1
    path = output_manager.validation_final_report_file_path(
        state["request_name"], state["model"], state["version"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    problems = state.get("best_problems") or state.get("problems") or []
    lines = [
        "ẢNH ĐƯỢC XUẤT (tóm tắt cho nghiệp vụ)",
        "=" * 72,
        f"Request            : {state['request_name']}",
        f"Model              : {state['model']}",
        f"Version            : {state['version']}",
        f"Lần gen được giữ   : attempt {delivered_attempt}",
        f"Điểm keep-best     : {state.get('best_score', state.get('validation_score', 0))}",
        "",
        "File validate chi tiết từng lần gen:",
    ]
    max_attempts = int(state.get("attempt") or delivered_attempt)
    for attempt in range(1, max_attempts + 1):
        detail = output_manager.validation_report_file_path(
            state["request_name"], state["model"], state["version"], attempt
        )
        marker = "  ← ẢNH NÀY ĐƯỢC XUẤT" if attempt == delivered_attempt else ""
        exists = "có" if detail.exists() else "không có (lần này không gen/validate)"
        lines.append(f"- attempt {attempt}: {detail.name} ({exists}){marker}")
    lines.append("")
    if problems:
        lines.append("Lỗi/cảnh báo còn lại trên ảnh được giữ:")
        lines.extend(f"- {item}" for item in problems)
    else:
        lines.append("Ảnh được giữ không còn lỗi validate.")
    lines.append("")
    lines.append("Nếu có attempt khác, đó là lần sửa hoặc gen lại đã bị loại vì điểm kém hơn,")
    lines.append("hoặc không chạy vì ảnh attempt 1 chỉ có cảnh báo nhỏ (giữ luôn).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote final validation summary -> %s", path)


def _concept_regenerate_prompt(original_prompt: str, problems: list[str]) -> str:
    """Same original prompt, plus why the previous canvas was rejected.

    Do not attach the failed image: the object/concept was wrong.
    """
    failed = "\n".join(f"- {item}" for item in problems) or "- (no detail)"
    return (
        f"{original_prompt.rstrip()}\n\n"
        "PREVIOUS ATTEMPT FAILED THESE CONCEPT CHECKS — generate a NEW banner from this prompt. "
        "Do not reuse or edit the previous image (it is not attached).\n"
        f"{failed}\n"
    )


def _node_generate(state: GenerateRepairState) -> dict:
    canvas_wh = state["canvas_wh"]
    image_usage_sink: dict = {}
    try:
        if int(state.get("attempt") or 1) > 1 and state.get("best_cropped"):
            with fail_fast_on_quota_exhaustion():
                raw_bytes = generate_image(
                    state["model"],
                    state["prompt"],
                    state["attachments"],
                    canvas_wh[0],
                    canvas_wh[1],
                    state["settings"],
                    usage_sink=image_usage_sink,
                )
        else:
            raw_bytes = generate_image(
                state["model"],
                state["prompt"],
                state["attachments"],
                canvas_wh[0],
                canvas_wh[1],
                state["settings"],
                usage_sink=image_usage_sink,
            )
    except Exception as exc:
        # Repair (attempt 2+) must not discard a usable first image when the
        # provider 429s / times out. Keep-best already lives in state.
        if state.get("best_cropped"):
            logger.warning(
                "model=%s version=%d: generate attempt %d/%d failed (%s) — keeping the best earlier image instead of aborting.",
                state["model"],
                state["version"],
                state["attempt"],
                state["max_attempts"],
                exc,
            )
            return {"generate_failed": True}
        raise
    state["usage_tracker"].record_image(
        image_usage_sink,
        stage="image_generation",
        model=state["model"],
        version=state["version"],
        width=canvas_wh[0],
        height=canvas_wh[1],
        notes=f"attempt={state['attempt']}/{state['max_attempts']}",
    )
    cropped_bytes = crop_canvas_to_output(raw_bytes, canvas_wh, state["crop_box"], state["req"].final_output_wh())
    return {"raw_bytes": raw_bytes, "cropped_bytes": cropped_bytes, "generate_failed": False}


def _route_after_generate(state: GenerateRepairState) -> Literal["validate", "finalize"]:
    if state.get("generate_failed") and state.get("best_cropped"):
        return "finalize"
    return "validate"


def _node_validate(state: GenerateRepairState) -> dict:
    report = validate_output_report(
        state["cropped_bytes"],
        state["req"],
        usage_tracker=state["usage_tracker"],
        version=state["version"],
    )
    try:
        _write_validation_report(state, report)
    except Exception:  # noqa: BLE001 -- logging must never block generate
        logger.exception("Could not write validation report (continuing anyway)")

    problems = report.problems
    updates: dict[str, Any] = {
        "problems": problems,
        "repair_mode": report.repair_mode,
        "validation_score": report.score,
    }
    best_cropped = state.get("best_cropped")
    best_score = state.get("best_score")
    if best_cropped is None or best_score is None or report.score < best_score:
        updates["best_raw"] = state["raw_bytes"]
        updates["best_cropped"] = state["cropped_bytes"]
        updates["best_problems"] = problems
        updates["best_score"] = report.score
        updates["best_attempt"] = state["attempt"]
    logger.info(
        "model=%s version=%d attempt %d/%d validate: repair_mode=%s score=%d blockers=%d warnings=%d",
        state["model"],
        state["version"],
        state["attempt"],
        state["max_attempts"],
        report.repair_mode,
        report.score,
        len(report.blockers),
        len(report.warnings),
    )
    return updates


def _route_after_validate(state: GenerateRepairState) -> Literal["repair", "regenerate", "finalize"]:
    """Keep on pass / single warning; edit layout blockers; regenerate concept blockers."""
    mode = state.get("repair_mode") or "none"
    if mode == "none" or state["attempt"] >= state["max_attempts"]:
        return "finalize"
    if mode == "regenerate":
        return "regenerate"
    return "repair"


def _node_repair(state: GenerateRepairState) -> dict:
    problems = state["problems"]
    # Always rebuild from the ORIGINAL request assets plus the failed canvas.
    # Using the previous repair's attachment list would nest failed_full_canvas.
    prompt, attachments = _build_targeted_repair(
        state["req"],
        state["raw_bytes"],
        state["original_attachments"],
        problems,
    )
    try:
        _write_repair_prompt_log(
            state["request_name"], state["model"], state["version"], state["attempt"] + 1, prompt
        )
    except Exception:  # noqa: BLE001 -- logging must never block repair
        logger.exception("Could not write targeted repair prompt log (continuing anyway)")
    logger.warning(
        "Validation layout problems for model=%s version=%d (attempt %d/%d): %s -- targeted edit of the failed full canvas.",
        state["model"],
        state["version"],
        state["attempt"],
        state["max_attempts"],
        "; ".join(problems),
    )
    return {
        "prompt": prompt,
        "attachments": attachments,
        "attempt": state["attempt"] + 1,
    }


def _node_regenerate(state: GenerateRepairState) -> dict:
    """Concept/object is wrong — do not edit a broken canvas; start over."""
    original_prompt = state.get("original_prompt") or state["prompt"]
    original_attachments = state["original_attachments"]
    problems = list(state.get("problems") or [])
    prompt = _concept_regenerate_prompt(original_prompt, problems)
    try:
        _write_repair_prompt_log(
            state["request_name"], state["model"], state["version"], state["attempt"] + 1, prompt
        )
    except Exception:  # noqa: BLE001 -- logging must never block regenerate
        logger.exception("Could not write regenerate prompt log (continuing anyway)")
    logger.warning(
        "Validation concept problems for model=%s version=%d (attempt %d/%d): %s -- regenerating from the original prompt plus failed checks, not editing the failed image.",
        state["model"],
        state["version"],
        state["attempt"],
        state["max_attempts"],
        "; ".join(problems),
    )
    return {
        "prompt": prompt,
        "attachments": original_attachments,
        "attempt": state["attempt"] + 1,
    }


def _node_finalize(state: GenerateRepairState) -> dict:
    cropped_bytes = state.get("cropped_bytes")
    best_cropped = state.get("best_cropped")
    updates: dict[str, Any] = {}
    if best_cropped is not None and best_cropped != cropped_bytes:
        logger.warning(
            "model=%s version=%d: later attempt scored worse — reverting to attempt %s (score=%s, %d problem(s)): %s",
            state["model"],
            state["version"],
            state.get("best_attempt"),
            state.get("best_score"),
            len(state.get("best_problems") or []),
            "; ".join(state.get("best_problems") or []),
        )
        updates = {
            "raw_bytes": state.get("best_raw"),
            "cropped_bytes": best_cropped,
            "problems": list(state.get("best_problems") or []),
        }
    elif state.get("best_problems"):
        logger.warning(
            "model=%s version=%d keeping attempt %s with remaining problem(s): %s",
            state["model"],
            state["version"],
            state.get("best_attempt") or state.get("attempt"),
            "; ".join(state["best_problems"]),
        )
    try:
        merged = {**state, **updates}
        _write_final_validation_summary(merged)
    except Exception:  # noqa: BLE001 -- logging must never block export
        logger.exception("Could not write final validation summary (continuing anyway)")
    return updates


def build_generate_repair_graph():
    graph = StateGraph(GenerateRepairState)
    graph.add_node("generate", _node_generate)
    graph.add_node("validate", _node_validate)
    graph.add_node("repair", _node_repair)
    graph.add_node("regenerate", _node_regenerate)
    graph.add_node("finalize", _node_finalize)

    graph.set_entry_point("generate")
    graph.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"validate": "validate", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"repair": "repair", "regenerate": "regenerate", "finalize": "finalize"},
    )
    graph.add_edge("repair", "generate")
    graph.add_edge("regenerate", "generate")
    graph.add_edge("finalize", END)
    return graph.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_generate_repair_graph()
    return _GRAPH


def run_generate_and_repair(
    *,
    request_name: str,
    model: str,
    version: int,
    req: CreativeRequest,
    prompt: str,
    attachments: list[ImageAttachment],
    canvas_wh: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    usage_tracker: Any,
    settings: Settings,
    max_attempts: int = _MAX_GENERATE_ATTEMPTS,
) -> GenerateRepairResult:
    """Run generate → validate → (repair → generate)* → keep-best.

    Raises whatever the image provider raises; the caller isolates that
    per (model, version) so one failure cannot abort the rest of the run.
    """
    final_state: GenerateRepairState = _get_graph().invoke(
        {
            "request_name": request_name,
            "model": model,
            "version": version,
            "req": req,
            "prompt": prompt,
            "original_prompt": prompt,
            "attachments": attachments,
            "original_attachments": attachments,
            "canvas_wh": canvas_wh,
            "crop_box": crop_box,
            "usage_tracker": usage_tracker,
            "settings": settings,
            "attempt": 1,
            "max_attempts": max_attempts,
            "problems": [],
            "best_problems": [],
            "generate_failed": False,
        }
    )
    raw_bytes = final_state.get("raw_bytes")
    cropped_bytes = final_state.get("cropped_bytes")
    if not raw_bytes or not cropped_bytes:
        raise RuntimeError(
            f"generate-repair graph produced no image for model={model} version={version}"
        )
    return GenerateRepairResult(
        raw_bytes=raw_bytes,
        cropped_bytes=cropped_bytes,
        problems=list(final_state.get("problems") or []),
    )
