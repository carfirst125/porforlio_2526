"""LangGraph workflow implementing REQUEST.md [1]'s numbered steps
`[1]` read request -> `[2]` guideline lookup -> `[3]`/`[4]` prompt + image
downsampling -> `[5]` final prompt assembly. Those five steps are linear, so
they stay in this module's setup graph.

Step `[6]` (generate, crop, validate, targeted repair / regenerate, keep-best)
is a real state machine with conditional edges and a loop -- that lives in
`app/core/generate_repair_graph.py` and runs once per (model, version) pair
so one failure stays isolated (REQUEST.md IMPORTANT NOTE: dump exceptions,
never let one blow up the whole run).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.core import output_manager
from app.core import text_gen
from app.core.asset_resolver import ResolvedBucketAssets, resolve_assets
from app.core.canvas_planner import CanvasPlan, plan_canvas_for_model, plan_request_geometry
from app.core.content_writer import draft_missing_text
from app.core.generate_repair_graph import (
    _MAX_GENERATE_ATTEMPTS,
    run_generate_and_repair,
)
from app.core.guidelines import ResolvedGuideline, resolve_guideline
from app.core.image_post import crop_canvas_to_output
from app.core.image_prep import PreparedImage, prepare_attachments
from app.core.prompt_builder import BuiltPrompt, build_prompt, format_attached_images_note, write_prompt_log
from app.core.technique_selector import empty_selection, select_techniques_for_versions
from app.core.text_compositor import render_text_layer
from app.core.usage_tracker import UsageTracker
from app.logging_conf import get_logger
from app.models.schemas import AudienceConfig, CreativeRequest, GenerateResult, parse_wh, wh_slug
from app.providers.base import ImageAttachment
from app.providers.registry import generate_image
from app.settings import get_settings

logger = get_logger("pipeline")


class PipelineState(TypedDict, total=False):
    request_name: str
    models: list[str]
    generate_person_override: bool | None
    persona_override: str | None
    req: CreativeRequest
    canvas_plan: CanvasPlan
    resolved_assets: ResolvedBucketAssets
    guideline: ResolvedGuideline
    prepared_images: list[PreparedImage]
    built_prompt: BuiltPrompt
    error: str | None
    usage_tracker: Any  # app.core.usage_tracker.UsageTracker | None -- 2026-09-06 user request (cost tracking)


def load_request_json(request_name: str) -> CreativeRequest:
    path = get_settings().input_dir / f"{request_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"request file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CreativeRequest.model_validate(raw)


# ---------------------------------------------------------------------------
# Graph nodes -- REQUEST.md [1]/[2]/[3]/[4]
# ---------------------------------------------------------------------------

def _node_load_request(state: PipelineState) -> dict:
    logger.info("[1/5] Reading request.json for %r", state["request_name"])
    req = load_request_json(state["request_name"])
    person_override = state.get("generate_person_override")
    if person_override is not None and req.design_options.generate_person != person_override:
        logger.info("Overriding design_options.generate_person -> %s for this run only.", person_override)
        req = req.model_copy(
            update={"design_options": req.design_options.model_copy(update={"generate_person": person_override})}
        )
    if "persona_override" in state and state.get("persona_override") is not None:
        persona_value = (state.get("persona_override") or "").strip() or None
        current = (req.audience.persona or None)
        if persona_value != current:
            logger.info("Overriding audience.persona -> %r for this run only.", persona_value)
            req = req.model_copy(update={"audience": AudienceConfig(persona=persona_value)})
    # 2026-09-09 (user request): resolve the new `design_options.
    # technique_selection` object into the internal, derived `techniques`
    # field every downstream consumer already reads uniformly (prompt_builder,
    # technique_selector) -- keeps that plumbing unchanged by this schema
    # migration. Manual mode uses `manual_techniques` verbatim and completely
    # ignores `auto_config` ("Không quan tâm auto_config khi mode là manual" --
    # user's own words); auto mode completely ignores `manual_techniques`
    # (never even as a fallback) and leaves `techniques` empty here -- it is
    # filled later, per version, once `select_techniques_for_versions()` runs
    # (main flow below), the SAME "manual list intentionally removed/deferred
    # at the boundary" isolation this project already relied on before this
    # migration (prevents a stray manual list from affecting prompt caches,
    # refinement, fallback, or validation while auto selection is enabled).
    selection_cfg = req.design_options.technique_selection
    if selection_cfg.mode == "auto":
        if req.design_options.techniques:
            logger.info(
                "Ignoring %d manual technique key(s) (design_options.techniques) because "
                "technique_selection.mode=auto.",
                len(req.design_options.techniques),
            )
        if selection_cfg.manual_techniques:
            logger.info(
                "Ignoring %d technique_selection.manual_techniques key(s) because "
                "technique_selection.mode=auto.",
                len(selection_cfg.manual_techniques),
            )
        req = req.model_copy(
            update={
                "design_options": req.design_options.model_copy(update={"techniques": []})
            }
        )
    else:  # manual
        req = req.model_copy(
            update={
                "design_options": req.design_options.model_copy(
                    update={"techniques": list(selection_cfg.manual_techniques)}
                )
            }
        )
    req, plan = plan_request_geometry(req, state.get("models", []))
    if plan is None:
        logger.info("Keeping declared request geometry (selected models do not share a known ratio table).")
        return {"req": req}
    logger.info(
        "Auto-planned geometry from output=%s -> native ratio=%s canvas=%s crop=%s (discard %.1f%%).",
        plan.output_wh,
        plan.ratio_label,
        plan.canvas_wh,
        plan.crop_box,
        plan.crop_loss * 100,
    )
    return {"req": req, "canvas_plan": plan}


def _node_resolve_assets(state: PipelineState) -> dict:
    logger.info("[2/5] Resolving bucket assets (theme/context/product/reference)")
    return {"resolved_assets": resolve_assets(state["req"])}


def _node_load_guideline(state: PipelineState) -> dict:
    req: CreativeRequest = state["req"]
    logger.info("[3/5] Looking up design guideline for format_type=%r", req.format_type)
    return {"guideline": resolve_guideline(req.format_type)}


def _node_prepare_images(state: PipelineState) -> dict:
    logger.info("[4/5] Downsampling attachment images (REQUEST.md [1] step [4])")
    req: CreativeRequest = state["req"]
    assets: ResolvedBucketAssets = state["resolved_assets"]
    guideline: ResolvedGuideline = state["guideline"]

    labeled: list[tuple[str, Path]] = []
    labeled += [("theme", p) for p in assets.theme]
    labeled += [("context", p) for p in assets.context]
    labeled += [("product", p) for p in assets.product]
    labeled += [("brand_text", p) for p in assets.brand_text]
    if assets.reference:
        labeled.append(("reference", assets.reference))
    if assets.guideline_extra:
        labeled.append(("guideline_extra", assets.guideline_extra))
    # Do not send internal guideline/spec screenshots to the image model.
    # Their resolved text is already included in the prompt, while the
    # screenshots contain borders, rulers, margins and style examples that
    # models have repeatedly copied into the final banner. An explicitly
    # supplied request-level `guideline_extra` remains supported above.
    #
    # 2026-09-08 (user request): a guideline entry's `"font"` field IS meant
    # to be attached, unlike its `source` screenshots above -- it names
    # actual typeface/lettering reference image(s) (e.g. Manrope weight
    # samples) the model should match, not an internal spec page. Appended
    # here (same `labeled` list) so it goes through the identical downsample
    # step below as every other attachment ("giảm phân giải như các hình khác").
    labeled += [("guideline_font", p) for p in guideline.font_image_paths]

    prepared = prepare_attachments(state["request_name"], labeled)

    # Never attach a visual safe-area rectangle. Across both initial and
    # repaired generations, models reproduced it as a pale/rounded inset
    # panel even when told it was not artwork. Text geometry + post-crop
    # validation + targeted repair now provide containment without that leak.
    if req.has_crop():
        logger.info("Not attaching visual safe-area guide; using invisible text padding and post-crop repair.")

    return {"prepared_images": prepared}


def _node_build_prompt(state: PipelineState) -> dict:
    logger.info("[5/5] Assembling final prompt (design brief + guideline + design rules + assets)")
    req: CreativeRequest = state["req"]
    prepared: list[PreparedImage] = state["prepared_images"]
    labels_in_order = [p.label for p in prepared]
    if req.design_options.technique_selection.mode == "auto":
        # The selector must see final per-version copy before a prompt is
        # assembled. Defer prompt construction instead of creating a stale,
        # technique-free base cache that will never be sent to an image model.
        logger.info("Deferring prompt assembly until auto techniques are selected per version.")
        built = BuiltPrompt(text="", cache_key="", attachments_order=labels_in_order)
    else:
        built = build_prompt(
            state["request_name"], req, state["guideline"], labels_in_order, usage_tracker=state.get("usage_tracker")
        )
    return {"built_prompt": built}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("load_request", _node_load_request)
    graph.add_node("resolve_assets", _node_resolve_assets)
    graph.add_node("load_guideline", _node_load_guideline)
    graph.add_node("prepare_images", _node_prepare_images)
    graph.add_node("build_prompt", _node_build_prompt)

    graph.set_entry_point("load_request")
    graph.add_edge("load_request", "resolve_assets")
    graph.add_edge("resolve_assets", "load_guideline")
    graph.add_edge("load_guideline", "prepare_images")
    graph.add_edge("prepare_images", "build_prompt")
    graph.add_edge("build_prompt", END)
    return graph.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _load_attachment_bytes(prepared: list[PreparedImage]) -> list[ImageAttachment]:
    attachments = []
    for p in prepared:
        try:
            data = p.resized_path.read_bytes()
        except OSError:
            logger.exception("Could not read prepared attachment %s -- skipping.", p.resized_path)
            continue
        attachments.append(ImageAttachment(label=p.label, mime_type=p.mime_type, data=data))
    return attachments


# ---------------------------------------------------------------------------
# REQUEST.md [1] step [6] -- generate + crop + save, per (model, version)
# ---------------------------------------------------------------------------

def run_generate(
    request_name: str,
    models: list[str],
    num_version_override: int | None = None,
    generate_person_override: bool | None = None,
    persona_override: str | None = None,
) -> list[dict]:
    """Runs the LangGraph pipeline once to build the shared prompt, then
    calls every requested image model (each `num_version` times) against
    that same prompt. Returns a list of `GenerateResult`-shaped dicts, 1 per
    (model, version) pair -- ALWAYS, even on failure (never raises).

    Saves 2 image files per successful (model, version): the final cropped
    image at `output_manager.size_model_dir(...)`, and the RAW image exactly
    as the model returned it (before `crop_canvas_to_output()`) under that
    same directory's `ORG/` subfolder, same filename -- see `app/core/
    output_manager.py` module docstring (2026-08-20, user request).

    Also logs the exact prompt text sent, once per (model, version), to
    `output_manager.prompt_file_path(...)` -- attempted for EVERY (model,
    version) pair regardless of whether the model call itself succeeds
    (REQUEST.md [3d], 2026-08-20 user request)."""
    settings = get_settings()
    # 2026-09-06 (user request): one UsageTracker per run_generate() call --
    # every LLM/image GenAI call made below (prompt refinement, per-version
    # copy, technique selection, up to 8 vision presence-checks per (model,
    # version), and the image generation calls themselves) is recorded into
    # it, then written once as a CSV under output/<request_name>/_usage/
    # right before this function returns (see app/core/usage_tracker.py).
    usage_tracker = UsageTracker(request_name)
    # Keep generated copy across explicit runs. text_gen's content-addressed
    # cache key already includes the brief, format, version count, template,
    # and model, so changed inputs regenerate while identical reruns avoid a
    # redundant copy-writing LLM call.
    # A manual run must not leave a stale auto-selection artifact looking
    # current, so selection artifacts are scoped to every explicit run.
    output_manager.clear_technique_selection(request_name)

    try:
        state: PipelineState = _get_graph().invoke(
            {
                "request_name": request_name,
                "models": models,
                "usage_tracker": usage_tracker,
                "generate_person_override": generate_person_override,
                "persona_override": persona_override,
            }
        )
    except Exception as exc:  # noqa: BLE001 -- REQUEST.md: dump, never crash
        logger.exception("Pipeline setup (steps 1-5) failed for request=%r", request_name)
        return [
            GenerateResult(ok=False, image_model=m, error=f"pipeline setup failed: {exc}").model_dump()
            for m in models
        ]

    req: CreativeRequest = state["req"]
    built_prompt: BuiltPrompt = state["built_prompt"]
    attachments = _load_attachment_bytes(state["prepared_images"])

    # 2026-08-20 (2) (user request): footer appended to every prompt log
    # file below, naming exactly which image FILES were attached alongside
    # the prompt text -- same attachments for every model/version in this
    # run (built once by the graph), so computed once here.
    attached_images_note = format_attached_images_note(state["prepared_images"])

    canvas_wh = req.canvas_wh()
    crop_box = req.crop_box()
    final_wh = req.final_output_wh()
    display_size = wh_slug(*final_wh)
    num_version = max(1, num_version_override if num_version_override is not None else req.num_version)
    # Keep every downstream prompt/selector context aligned with the actual
    # number of versions requested by the API override.
    if req.num_version != num_version:
        req = req.model_copy(update={"num_version": num_version})

    labels_in_order = [prepared.label for prepared in state["prepared_images"]]
    guideline: ResolvedGuideline = state["guideline"]

    # Three deliberately separate copy modes:
    # - static text (default): preserves the existing prompt exactly;
    # - auto_generate_text: fills only missing fields once for all versions;
    # - generate_text: the SOLE gate for calling the per-version copy LLM
    #   (`text_gen.generate_creative_texts()`) -- it always asks for a
    #   complete headline/message/cta set per version; which of those 3
    #   fields actually get used (vs. the static `text.<field>`) is then
    #   decided per field by generate_text_headline/message/cta_label inside
    #   `text_gen._selected_llm_fields()`, not here. 2026-09-04 (user
    #   request): the 3 field-level flags no longer independently trigger
    #   this LLM call by themselves -- when `generate_text=False`, no
    #   per-version copy LLM call happens at all, regardless of those flags.
    version_texts = None
    shared_prompt_needs_rebuild = req.num_version != state["req"].num_version
    if req.design_options.generate_text:
        version_texts = text_gen.generate_creative_texts(request_name, req, num_version, usage_tracker=usage_tracker)
        if version_texts is None:
            logger.warning("Per-version text generation unavailable for request=%s; using static JSON text.", request_name)
    elif req.design_options.text_by_code or req.design_options.auto_generate_text:
        req = req.model_copy(update={"text": draft_missing_text(req, usage_tracker=usage_tracker)})
        shared_prompt_needs_rebuild = True

    # Materialize a complete request for every creative version first. The
    # technique selector deliberately runs only after final text resolution,
    # so its art-direction decision sees the exact copy used by that version.
    version_reqs: dict[int, CreativeRequest] = {
        version: req for version in range(1, num_version + 1)
    }
    version_prompts: dict[int, BuiltPrompt] = {}
    if version_texts is not None:
        for version in range(1, num_version + 1):
            version_reqs[version] = req.model_copy(update={"text": version_texts[version]})

    is_auto_technique_selection = req.design_options.technique_selection.mode == "auto"
    if is_auto_technique_selection:
        try:
            selections = select_techniques_for_versions(
                request_name=request_name,
                version_reqs=version_reqs,
                guideline=guideline,
                assets=state["resolved_assets"],
                usage_tracker=usage_tracker,
            )
        except Exception as exc:  # noqa: BLE001 -- selector must never abort image generation
            logger.exception("Unexpected Technique Selector failure; continuing with empty safe selections.")
            selections = {
                version: empty_selection(f"Unexpected selector failure: {exc}")
                for version in version_reqs
            }
        for version, version_req in version_reqs.items():
            selected_keys = selections[version].flattened_keys()
            version_reqs[version] = version_req.model_copy(
                update={
                    "design_options": version_req.design_options.model_copy(
                        update={"techniques": selected_keys}
                    )
                }
            )

    # Auto selection and generated per-version copy both make the prompt
    # version-specific. Give each one an isolated cache so a later version
    # cannot overwrite or accidentally reuse another version's prompt.
    if is_auto_technique_selection or version_texts is not None:
        for version, version_req in version_reqs.items():
            version_prompts[version] = build_prompt(
                request_name,
                version_req,
                guideline,
                labels_in_order,
                cache_variant=f"v{version}",
                usage_tracker=usage_tracker,
                version=version,
            )
    elif shared_prompt_needs_rebuild:
        built_prompt = build_prompt(request_name, req, guideline, labels_in_order, usage_tracker=usage_tracker)

    def _generate_version(model: str, version: int) -> dict:
        timestamp_str = output_manager.now_timestamp_str()
        active_req = version_reqs[version]
        active_prompt = version_prompts.get(version, built_prompt)

        # 2026-08-20 (user request, REQUEST.md [3d]): log the EXACT prompt
        # text being sent, once per (model, version) -- filename records
        # which model it was sent to, e.g. `prompt-gpt-image-2-version-1.txt`.
        # Logged BEFORE the actual model call so the sent prompt is captured
        # on disk even if `generate_image()` itself then fails.
        try:
            write_prompt_log(request_name, model, version, active_prompt.text, attached_images_note)
        except Exception:  # noqa: BLE001
            logger.exception("Could not write prompt log for model=%s version=%d (continuing anyway)", model, version)

        try:
            # generate → validate → edit or regenerate → keep-best is a
            # LangGraph with conditional edges (see generate_repair_graph.py).
            # Prompt-side guards can only ask the model for compliance;
            # this graph is the confirmation step. Always keeps the best
            # attempt rather than failing the whole result.
            repaired = run_generate_and_repair(
                request_name=request_name,
                model=model,
                version=version,
                req=active_req,
                prompt=active_prompt.text,
                attachments=attachments,
                canvas_wh=canvas_wh,
                crop_box=crop_box,
                usage_tracker=usage_tracker,
                settings=settings,
                max_attempts=_MAX_GENERATE_ATTEMPTS,
            )
            raw_bytes = repaired["raw_bytes"]
            cropped_bytes = repaired["cropped_bytes"]

            final_bytes = cropped_bytes
            if active_req.design_options.text_by_code:
                final_bytes = render_text_layer(final_bytes, active_req, final_wh)

            output_manager.size_model_dir(request_name, display_size, model).mkdir(parents=True, exist_ok=True)
            out_path = output_manager.unique_image_output_path(
                request_name, req.creative_id, display_size, model, timestamp_str, version, num_version
            )
            out_path.write_bytes(final_bytes)

            # 2026-08-20 (user request): also archive the RAW image exactly as the
            # model returned it (pre-crop), same filename, under an ORG/ subfolder
            # next to the final cropped one -- never lets a failure here fail the
            # (model, version) result, since the final cropped image is what matters.
            try:
                org_dir = output_manager.org_dir(request_name, display_size, model)
                org_dir.mkdir(parents=True, exist_ok=True)
                org_path = output_manager.org_image_path(request_name, display_size, model, out_path.name)
                org_path.write_bytes(raw_bytes)
                logger.info("Saved raw (pre-crop) image -> %s", org_path)
            except Exception:  # noqa: BLE001 -- archival save only, must not fail the result
                logger.exception("Could not save raw (pre-crop) image to ORG/ for model=%s version=%d (continuing -- final cropped image was saved fine)", model, version)

            logger.info("OK  model=%s version=%d -> %s", model, version, out_path)
            return GenerateResult(
                ok=True, image_model=model, version=version, display_size=display_size, path=str(out_path.resolve()), filename=out_path.name
            ).model_dump()
        except Exception as exc:  # noqa: BLE001 -- REQUEST.md: dump exception, keep going
            logger.exception("FAILED model=%s version=%d for request=%s", model, version, request_name)
            return GenerateResult(
                ok=False, image_model=model, version=version, display_size=display_size, error=str(exc)
            ).model_dump()

    results: list[dict] = []
    for model in models:
        logger.info("=== Generating with model=%s (canvas=%s, versions=%d) ===", model, canvas_wh, num_version)
        versions = range(1, num_version + 1)
        worker_count = min(settings.generation_max_workers, num_version)
        if worker_count == 1:
            model_results = [_generate_version(model, version) for version in versions]
        else:
            logger.info("Generating %d versions concurrently with %d workers.", num_version, worker_count)
            # executor.map preserves input order even though versions finish
            # independently, keeping the API response stable (v1, v2, ...).
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dco-version") as executor:
                model_results = list(executor.map(lambda version: _generate_version(model, version), versions))
        results.extend(model_results)
    usage_tracker.save(num_version=num_version)
    return results


def run_clone(request_name: str, source_image_path: str, image_model: str, display_sizes: list[str]) -> list[dict]:
    """REQUEST.md [4] `--mode=clone`: take an already-finished design
    (`source_image_path`) and re-render it at each size in `display_sizes`
    (`"W,H"` strings) using `image_model`, saved into
    `output/<request_name>/release/` (REQUEST.md naming convention already
    on disk: `<source_stem>_clone_<display_size>_<image_model>.png`)."""
    settings = get_settings()
    source = Path(source_image_path)
    results: list[dict] = []
    # 2026-09-06 (user request): `run_clone()` is a separate top-level run
    # from `run_generate()` -- its own image-generation calls (1 per
    # `display_sizes` entry) get their own UsageTracker/run row rather than
    # sharing one with `run_generate()`.
    usage_tracker = UsageTracker(request_name)

    if not source.exists():
        error = f"source image not found: {source}"
        logger.error(error)
        return [{"ok": False, "display_size": s, "image_model": image_model, "error": error} for s in display_sizes]

    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        logger.exception("Could not read source clone image %s", source)
        return [{"ok": False, "display_size": s, "image_model": image_model, "error": str(exc)} for s in display_sizes]

    clone_prompt = (
        "You are given ONE reference image: an already-finished marketing banner design. Recreate the SAME "
        "visual design -- identical text/headline/message/CTA, identical logo, identical product and character "
        "-- adapted to a NEW canvas size. Re-compose the layout as needed to fit the new aspect ratio (e.g. "
        "repositioning elements, extending or cropping background) but do not change any text content, brand "
        "colors, logo, the identity of the product/character shown, or their original photographic versus "
        "illustrated medium. Output must be one finished raster banner "
        "following the generation-canvas/final-region geometry appended below, with no added borders or "
        "letterboxing."
    )
    attachments = [ImageAttachment(label="source_design", mime_type="image/png", data=source_bytes)]
    source_stem = source.stem

    for clone_index, size_str in enumerate(display_sizes, start=1):
        try:
            width, height = parse_wh(size_str)
            slug = wh_slug(width, height)
            logger.info("Cloning %s -> %s with model=%s", source.name, slug, image_model)

            plan = plan_canvas_for_model((width, height), image_model)
            canvas_w, canvas_h = plan.canvas_wh
            crop_x, crop_y, crop_w, crop_h = plan.crop_box
            geometry_note = (
                f" Generate the full image at {canvas_w}x{canvas_h} ({plan.ratio_label} native ratio), keeping "
                f"all important content inside the centered {crop_w}x{crop_h} final region at "
                f"({crop_x}, {crop_y}); only background may extend outside that region."
                if plan.has_crop
                else f" Generate directly at {width}x{height}; this is the final canvas."
            )
            clone_usage_sink: dict = {}
            raw_bytes = generate_image(
                image_model, clone_prompt + geometry_note, attachments, canvas_w, canvas_h, settings,
                usage_sink=clone_usage_sink,
            )
            usage_tracker.record_image(
                clone_usage_sink,
                stage="clone_generation",
                model=image_model,
                # `run_clone()` has no "version" concept -- reuse that slot
                # to number each `display_sizes` entry (1-based, in request
                # order) so it still gets its own row in the `run_versions`
                # table instead of being lumped into `shared_cost_usd`
                # (2026-09-06 (2) user request).
                version=clone_index,
                width=canvas_w,
                height=canvas_h,
                notes=f"source={source_stem} target_size={slug}",
            )
            final_bytes = crop_canvas_to_output(raw_bytes, plan.canvas_wh, plan.crop_box, (width, height))

            out_path = output_manager.clone_release_path(request_name, source_stem, slug, image_model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(final_bytes)

            logger.info("OK  clone %s -> %s", slug, out_path)
            results.append({"ok": True, "display_size": slug, "image_model": image_model, "path": str(out_path.resolve()), "filename": out_path.name})
        except Exception as exc:  # noqa: BLE001
            logger.exception("FAILED clone size=%s model=%s for request=%s", size_str, image_model, request_name)
            results.append({"ok": False, "display_size": size_str, "image_model": image_model, "error": str(exc)})

    usage_tracker.save(num_version=len(display_sizes))
    return results
