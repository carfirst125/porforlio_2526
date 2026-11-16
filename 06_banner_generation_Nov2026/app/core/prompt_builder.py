"""Assemble the final prompt sent to an Image GenAI model (REQUEST.md [1]
step [3] & [5], [3a]-[3d]).

Built once per materialized version and cached under
`output/<request_name>/_prompt_cache/`. Manual/static requests retain the
legacy `built_prompt.json`; version-specific text or auto-selected techniques
use `built_prompt-v<n>.json`. Every image model for the same version therefore
reuses the exact same prompt text instead of rebuilding it (REQUEST.md
[3d]: "chỉ nên xuất 1 lần ... lần sau chạy với model khác ... không cần tạo
lại prompt mà đọc ở đây ra"). The cache key is a hash of every INPUT that
could change the prompt's content (design_brief, buckets, text, guideline,
design_options, num_version, attached-image labels) PLUS `_PROMPT_TEMPLATE_VERSION`
below -- editing `request.json` invalidates it automatically, and so does a
CODE change to this file's prompt-assembly template even when `request.json`
itself is untouched (2026-08-20 (3), user-reported bug: after the SAFE AREA
GUIDE feature was added, a request the user had already run once before kept
serving the OLD cached prompt text with none of the new wording, because
nothing about `request.json`'s own data had changed -- see
`_PROMPT_TEMPLATE_VERSION` below and `write_prompt_log()`'s docstring for
the matching fix on the saved `.txt` log file).

2026-08-26: entry (19)'s "AI draws background only, composite the real
context/product/text on top" architecture was tried and REVERTED (see
`4-project-evaluation-results.md` entry (19)'s follow-up note) -- the
composited result looked visibly crude/pasted-on ("thô", user's word) next
to the AI-drawn illustration background, a real style-mismatch problem
compositing didn't actually solve. Back to the model drawing everything in
one shot, guided by the CANVAS & SAFE AREA / CRITICAL REMINDER wording plus
`guideline_rewriter.py` / `prompt_bleed_guard.py` below.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core import color_utils, design_techniques, personas
from app.core.asset_resolver import split_bucket_values
from app.core.guidelines import ResolvedGuideline
from app.core.output_manager import versioned_prompt_cache_file
from app.core.text_style_profiles import profile_for
from app.llm.base import LLMProviderError
from app.llm.factory import LLMFactory
from app.logging_conf import get_logger
from app.models.schemas import CreativeRequest
from app.settings import get_settings

logger = get_logger("prompt_builder")

# Bump this constant every time `build_prompt()`'s assembly logic below
# changes wording/sections/behavior -- it is baked into the cache key
# (`_cache_key()`) specifically so a CODE-only change (no edit to
# request.json itself) still invalidates every previously-cached prompt on
# disk. Without this, `request.json` hashing to the identical value as a
# prior run would silently keep serving the OLD prompt text forever, no
# matter how much the template itself changed (2026-08-20 (3), real bug hit
# by the user: the SAFE AREA GUIDE + CRITICAL REMINDER wording never showed
# up in a re-run against an already-cached request).
_PROMPT_TEMPLATE_VERSION = 55

_REFINE_SYSTEM_INSTRUCTION = (
    "You are a prompt engineer preparing the FINAL instruction that will be sent, verbatim, to an "
    "image-generation AI model together with several reference images. Rewrite the prompt you are given so it "
    "is concise, unambiguous, and clearly structured with short sections/headers -- but you MUST preserve "
    "every constraint, requirement, MUST/NOT-MUST rule, exact text string (headline/message/CTA), pixel size, "
    "and attached-image reference in the input. Do not invent new requirements. Do not soften or drop any "
    "existing requirement. Do not add commentary before or after the prompt. Output ONLY the rewritten prompt."
)


@dataclass
class BuiltPrompt:
    text: str
    cache_key: str
    attachments_order: list[str] = field(default_factory=list)
    llm_refined: bool = False


def _design_rules_text() -> str:
    path = get_settings().prompt_rules_file
    if not path.exists():
        logger.warning("prompts/design_rules.md not found at %s -- continuing without it.", path)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Could not read design_rules.md at %s", path)
        return ""


def _db_text_color_lock(format_type: str) -> str:
    """Return the non-negotiable DB typography palette beside visible copy.

    The same values also exist in the long format guideline, but image models
    may overlook a deeply nested color note. Keeping this short lock directly
    beside REQUIRED TEXT makes the per-segment palette explicit at render time.
    """
    if format_type == "ocb_omni_110926_aff_1024x440":
        return (
            "DB TEXT-COLOR LOCK (mandatory): This is the AFF format. Headline text must use ONLY the approved "
            "light-gold palette #C5A049, #DDC37D, and/or #E4DABF; message text must be #FCF3C5. CTA label must "
            "be #FFFFFF on a #00804B button. Never render the headline or message in black, generic dark gray, "
            "or any unapproved color. Hex values are invisible production instructions, never visible copy. "
            "Never paint a hex token, typeface name, or font weight as letters on the banner. "
            "Wrong: appending the code after a phrase (e.g. 'gọn nhẹ E39717' or 'dE39717 Manrope'). "
            "Right: the required Vietnamese words only, filled IN those colors."
        )
    if format_type in {"ocb_omni_110926_mass_1024x440", "ocb_omni_110926_mass_aff_1024x440"}:
        return (
            "DB TEXT-COLOR LOCK (mandatory): This is a MASS/MASS_AFF format. Render headline line 1 in #00804B "
            "and headline line 2 in #E39717; if the headline occupies only one line, use #00804B and use #E39717 "
            "only for one intentional key phrase when helpful. Render the message in #545C57. CTA label must be "
            "#FFFFFF on a #00804B button. Never render the headline or message in black or an unapproved color. "
            "Hex values are invisible production instructions, never visible copy. "
            "Never paint a hex token, typeface name, or font weight as letters on the banner. "
            "Wrong: appending the code after a phrase (e.g. 'gọn nhẹ E39717' or 'dE39717 Manrope'). "
            "Right: the required Vietnamese words only, filled IN those colors."
        )
    if format_type == "ocb_omni_popup_aff_720x1280":
        return (
            "DB POPUP TEXT-COLOR LOCK (mandatory): This is the AFF Popup format (720x1280). Headline text must use ONLY "
            "the approved light-gold palette #C5A049, #DDC37D, and/or #E4DABF; message text must be #FAF0D7. "
            "CTA button is pill-shaped with gradient from #E1C364 to #D2A046, and CTA label text must be #553C00. "
            "Never render headline or message in black, generic gray, or unapproved colors. "
            "Hex values are invisible production instructions, never visible copy. "
            "Never paint a hex token, typeface name, or font weight as letters on the banner. "
            "Wrong: appending the code after a phrase (e.g. 'gọn nhẹ E39717' or 'dE39717 Manrope'). "
            "Right: the required Vietnamese words only, filled IN those colors."
        )
    if format_type in {"ocb_omni_popup_mass_720x1280", "ocb_omni_popup_mass_aff_720x1280"}:
        return (
            "DB POPUP TEXT-COLOR LOCK (mandatory): This is a MASS/MASS_AFF Popup format (720x1280). "
            "Prominent headline must be #E39717; secondary headline must be #0A9646; message must be #545C57. "
            "CTA button is pill-shaped with gradient from #008C55 to #00804B, and CTA label text must be #FFFFFF. "
            "Never render headline or message in black or unapproved colors. "
            "Hex values are invisible production instructions, never visible copy. "
            "Never paint a hex token, typeface name, or font weight as letters on the banner. "
            "Wrong: appending the code after a phrase (e.g. 'gọn nhẹ E39717' or 'dE39717 Manrope'). "
            "Right: the required Vietnamese words only, filled IN those colors."
        )
    return ""


def _visible_copy_whitelist_instruction() -> str:
    """Stop the model from painting typography spec as a fake copy line.

    Seen on DB_BH_AN_NINH_MANG MASS v3: guideline `Manrope Semibold` + `#E39717`
    was printed as the gold line `dE39717 Manrope Semi bold` between headline
    and message. Hex/font names in the format guideline are fill/style only.
    """
    return (
        "VISIBLE COPY WHITELIST (mandatory): Readable lettering on this banner may be ONLY "
        "(1) the exact Headline, Message, and CTA label from REQUIRED TEXT CONTENT, and "
        "(2) lettering already printed on a supplied product/phone/asset. "
        "Do not add any extra line of copy between headline and message. "
        "Forbidden as visible letters, even as a gold/orange accent line: typeface names "
        "(Manrope, Bai Jamjuree), weights (Semibold, Semi bold, Medium, Bold), hex color "
        "tokens with or without '#' (00804B, E39717, dE39717, 545C57, FFFFFF), pixel sizes, "
        "padding, and field names (text.headline, text.message, text.cta_label). "
        "Never glue a hex onto an advertising phrase (wrong: 'gọn nhẹ E39717'; right: 'gọn nhẹ' in orange). "
        "If a guideline or FONT REFERENCE mentions a font or hex, use it only as the style "
        "and fill of the required wording -- never as the wording itself. Ignore any specimen "
        "words printed on a font-sample attachment."
    )


def _decorative_limit_instruction(limit: int | None) -> str:
    if limit is None:
        return ""
    return (
        f"DECORATIVE DETAIL LIMIT: Use no more than {limit} purely decorative supporting motif group(s). "
        "This limit does not include supplied theme/context/product assets, required text, the continuous "
        "background itself, or an essential visual metaphor that communicates the offer. Decorative details "
        "A motif group may contain a few small related accents arranged as one visual unit; do not interpret this "
        "limit as a reason to leave the background empty. Details must be purposeful, visually grouped, subtle, "
        "and subordinate to the communication hierarchy; do not "
        "add filler icons or scatter unrelated objects. This is a maximum, not a target -- use fewer or none "
        "when the design is stronger without them."
    )


def _design_techniques_text(technique_keys: list[str], dominant_color: str = "") -> str:
    """Resolve optional art-direction modules without weakening core rules.

    2026-09-16 (user-reported bug, real generated banners reviewed): several
    BACKGROUND-group entries in `config/design_techniques_ocb_db.json`
    (e.g. `ocb_dark_glow_radial_field`, `ocb_mint_soft_gradient_field`)
    hardcoded a literal color ("deep green", "light mint") in their
    `description`/`prompt_instruction` instead of following the request's
    own `design_options.dominant_color` -- a much more concrete/directive
    instruction than the soft "use {dominant_color} as anchor" clause in
    `_theme_instruction()` below, so the image model followed the hardcoded
    catalog color instead. Those entries were rewritten to reference
    `{{dominant_color}}`/`{{tint_hex}}`/`{{shade_hex}}` placeholders instead
    of a fixed color name; `color_utils.fill_color_placeholders()` fills
    them in here, per-resolved-technique, with THIS request's actual
    dominant color before the text is folded into the prompt. Techniques
    with no such placeholder (the common case) are returned unchanged."""
    resolved, _unknown = design_techniques.resolve_techniques(technique_keys)
    if not resolved:
        return ""
    lines = [
        "OPTIONAL ART DIRECTION: The techniques below form one coherent art-direction set for this version. "
        "Combine only their compatible aspects into a unified composition; do not demonstrate them as separate "
        "effects or add content merely to prove a technique was used. Apply only the compatible, contextually "
        "appropriate aspects of the selected "
        "techniques below. They are supporting methods, not extra content requirements. Use them with restraint "
        "and keep the product and required message as the clear visual priorities. These techniques never override "
        "the request JSON, required text, supplied-asset fidelity, person/logo rules, safe area, full-bleed background, "
        "decorative limit, or any MUST/MUST NOT rule. Never redraw, duplicate, replace, crop, distort, or invent a "
        "new view/UI/variant of a supplied product, person, or logo to perform a technique. If a technique conflicts "
        "with those constraints or does not suit the brief, omit the conflicting part rather than forcing it."
    ]
    for technique in resolved:
        description = color_utils.fill_color_placeholders(technique.description, dominant_color)
        instruction = color_utils.fill_color_placeholders(technique.prompt_instruction, dominant_color)
        lines.append(f"- {technique.label}: {description}\n  Rendering guidance: {instruction}")
    return "\n".join(lines)


def _cache_key(req: CreativeRequest, guideline: ResolvedGuideline, attachment_labels: list[str]) -> str:
    payload = {
        "creative_id": req.creative_id,
        "format_type": req.format_type,
        "design_brief": req.design_brief,
        "buckets": req.buckets.model_dump(),
        "reference": req.reference,
        "text": req.text.model_dump(),
        "guidelines_field": req.guidelines,
        "design_options": req.design_options.model_dump(),
        "audience": req.audience.model_dump(),
        "persona_direction": personas.prompt_direction(req),
        "num_version": req.num_version,
        "generation": req.generation.model_dump(),
        "output": req.output.model_dump(),
        "guideline_entries": guideline.entry_names,
        # The guideline file is editable independently of request.json. Hash
        # its resolved text so corrected brand rules invalidate old prompts.
        "guideline_text": guideline.combined_text,
        # 2026-09-08 (user request): a guideline's `font` field is resolved
        # into paths, not text, so it would otherwise never appear in this
        # hash -- editing which font-reference images an entry lists (or
        # adding/removing the field) must still invalidate a stale cached
        # prompt built before that edit.
        "guideline_font_paths": [str(p) for p in guideline.font_image_paths],
        "attachment_labels": attachment_labels,
        "prompt_template_version": _PROMPT_TEMPLATE_VERSION,
        # Hash resolved catalog content as well as requested keys so editing
        # any config/design_techniques_<brand>.json file invalidates a stale
        # prompt automatically.
        "resolved_techniques": [
            {
                "key": technique.key,
                "description": technique.description,
                "prompt_instruction": technique.prompt_instruction,
            }
            for technique in design_techniques.resolve_techniques(req.design_options.techniques)[0]
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _join_descriptions(texts: list[str]) -> str:
    """Join 1+ free-text bucket descriptions into one readable clause for
    the prompt. Most requests supply exactly 1 description per bucket; a
    2nd+ entry in the same bucket is rare but supported (mirrors how
    multiple image paths in one bucket are already supported)."""
    cleaned = [t.strip() for t in texts if t.strip()]
    if len(cleaned) <= 1:
        return cleaned[0] if cleaned else ""
    return "; ".join(f"({i}) {t}" for i, t in enumerate(cleaned, start=1))


def _logo_instruction(req: CreativeRequest, any_asset_has_logo_hint: bool) -> str:
    if req.design_options.has_logo:
        return (
            "LOGO: `design_options.has_logo` is TRUE. Include the OCB logo in the design, following the logo "
            "rules in the Design Rules section below (correct color, clear-space, not cropped/obscured)."
        )
    return (
        "LOGO: `design_options.has_logo` is FALSE. Do NOT create or place any standalone OCB logo, brand lockup, "
        "corner logo, watermark, or repeated logo anywhere in the banner. A logo already printed on an approved "
        "product/context asset must remain only as an inseparable part of that original asset: preserve it inside "
        "the asset, but never extract, duplicate, enlarge, relocate, or recreate it elsewhere. This strict false "
        "flag overrides any general brand guideline or design convention that normally suggests adding a logo."
    )


def _person_instruction(req: CreativeRequest, has_context_assets: bool, context_texts: list[str] | None = None) -> str:
    persona = personas.resolve_for(req)
    if persona is not None and req.design_options.generate_person:
        generated_person_direction = (
            " Whenever you generate a human character yourself, create a naturally Vietnamese-looking person "
            f"matching this persona talent: {persona.talent} Stay within that brief for age, clothing, and "
            "styling. Place them in the campaign color field and layout — do not invent a photoreal street, "
            "café, Starbucks, airport, night city, or white studio unless a theme IMAGE or a specific "
            "(non-generic) theme description already supplies that place. Their facial expression, posture, "
            "and lived experience must communicate genuine happiness and enthusiasm about this program -- "
            "warm, energetic, and aspirational, but still natural rather than an exaggerated or artificial grin."
        )
    else:
        generated_person_direction = (
            " Whenever you generate a human character yourself, create a naturally Vietnamese-looking person. "
            "Creatively choose their age, refined context-appropriate clothing, and overall styling to fit the "
            "campaign, target audience, and program. Their facial expression, posture, and lived experience in the "
            "scene must communicate genuine happiness and enthusiasm about this program -- warm, energetic, and "
            "aspirational, but still natural rather than an exaggerated or artificial grin."
        )
    if has_context_assets and context_texts:
        # 2026-09-04 (2) (user request, with a real example: an attached
        # `context` photo of unspecified people PLUS a text entry in the
        # SAME bucket describing them as "sub_couple_young ... smart casual,
        # túi đồ"): when a bucket mixes an image WITH a text description,
        # the text is an EDIT instruction for that image, not just a mood
        # accent -- the user explicitly wants freedom to rotate/tilt the
        # subject and adapt their pose/clothing/attire to match the
        # described audience/context, while still requiring the actual
        # attached subject(s) to be used (same identity/face), never
        # swapped for a different or invented person. This intentionally
        # replaces the unconditional "preserve clothing/pose/expression"
        # lock below for this mixed case only -- that stricter lock still
        # applies unchanged whenever no text accompanies the image (see the
        # plain `has_context_assets` branch just below).
        return (
            "CONTEXT ASSET + TEXT EDIT DIRECTION: The image(s) attached in the `context` bucket show the actual "
            "subject(s) that must appear in this design. `buckets.context` also includes this text description of "
            f"the target audience/context: \"{_join_descriptions(context_texts)}\". Treat the text as an edit "
            "instruction for the attached image(s): you may adapt the subject(s)' pose, clothing/attire, "
            "expression, and camera angle -- including rotating horizontally/vertically or tilting them for a "
            "natural 3D perspective -- as needed so they fit that description and the surrounding scene. If the "
            "attached image contains a real person, they must remain recognizably the SAME person (same identity/"
            "face/photographic realism), never replaced, swapped, or invented; only their pose, attire, and framing "
            "may change. If the attached image contains only icons, objects, symbols, or scenery rather than a "
            "person, preserve that content's own identity the same way and use the text only to guide how it is "
            "staged, posed, or combined into the scene. Never invent an entirely separate additional person beyond "
            "what the attached image(s) and this description together call for."
        )
    if has_context_assets:
        return (
            "LOCKED CONTEXT ASSET: A `context` image is not automatically a person asset. Inspect what the "
            "supplied image actually contains and preserve that content exactly. If it contains a real person, "
            "preserve that same person's identity, appearance, clothing, pose, expression, and photographic "
            "realism without replacement or stylization. If it contains only icons, objects, symbols, or scenery, "
            "preserve those elements and do NOT convert the idea into a photographed person or family. Never "
            "invent additional people when `generate_person` is false, even if the design brief mentions a human "
            "idea such as family, customers, lifestyle, or future goals; express that idea symbolically instead."
        )
    if context_texts:
        # 2026-09-04 (user request): `buckets.context` may describe the
        # intended character/scene in words instead of attaching an image
        # -- checked BEFORE `generate_person` below, the same way an
        # attached context image already overrides that flag: a supplied
        # description (like a supplied image) is an intentional creative
        # input, not the "invent a random person" case `generate_person`
        # guards against.
        return (
            "CHARACTER/CONTEXT FROM TEXT DESCRIPTION: No `context` image was supplied; instead, `buckets.context` "
            f"describes the intended character or context element in words: \"{_join_descriptions(context_texts)}\". "
            "Generate that character or scene element yourself, following the description's audience, appearance, "
            "styling, pose, and energy as closely as possible, consistent with the design_brief and target "
            "audience. Unlike an attached context image, this description is creative direction, not a locked, "
            "pixel-exact asset -- you have freedom over exact rendering details as long as the result matches the "
            "description. If the description does not clearly call for a human character, render the described "
            "object/scene/symbol instead of inventing an unnecessary person."
            + generated_person_direction
        )
    if req.design_options.generate_person:
        return (
            "CHARACTER: No `context` image or text description was supplied, and `design_options.generate_person` "
            "is TRUE -- generate an appropriate character yourself, consistent with the design_brief and target "
            "audience."
            + generated_person_direction
        )
    return (
        "CHARACTER: No `context` image or text description was supplied, and `design_options.generate_person` is "
        "FALSE -- do NOT invent a human character for this design. Do not invent a hand, arm, wrist, fingers, "
        "or cropped limb holding the phone or product either. Focus on theme/background and product only; the "
        "phone/product stands or floats on its own, without a holder."
    )


def _no_person_no_hands_instruction() -> str:
    """2026-09-29 (user request): generate_person=false still produced a
    bodiless / floating hand holding the phone (same leak as de14). HUMAN
    REALISM AND ANATOMY plus 'tay cầm điện thoại' were always in the prompt,
    so the model treated a gripped phone as the default even with no person."""
    return (
        "NO PERSON / NO HANDS: generate_person is false and no context person was supplied. Do not draw any "
        "human, face, body, hand, arm, wrist, fingers, or cropped limb. Do not invent a floating or bodiless "
        "hand holding a phone, card, or product. The phone/product is a free-standing hero — it may sit, float, "
        "or rest in the scene on its own, without being gripped."
    )


def _human_realism_instruction() -> str:
    """2026-09-11 (user request): the user reported generated designs needed
    "Nhân vật tự nhiên" (natural-looking people) and better hand-anatomy
    control -- no plastic/waxy/mannequin skin, correct hand/arm count and
    attachment, controlled finger/joint/grip/overlap/occlusion, no hiding
    anatomy errors behind the product, and premium wardrobe regardless of
    style. The user believed this section (named "HUMAN REALISM AND
    ANATOMY") already existed in this file -- it did not (verified via
    repo-wide search); this is the first implementation. Unconditional and
    placed near the very end of the assembled prompt (see call site below),
    the same way FINAL COMPOSITION CHECK is deliberately restated at the end
    for emphasis -- applies whenever a person is expected in the design
    (`generate_person=true` or a supplied context person/text). When no
    person is expected, `_no_person_no_hands_instruction()` is used instead."""
    return (
        "HUMAN REALISM AND ANATOMY (applies to every person who appears in this design, whether generated or "
        "already present in a supplied asset):\n"
        "- PHOTOREAL COMMERCIAL LOOK: Render every person the way a real commercial advertising photograph "
        "would -- never plastic-looking or overly-smooth skin, never a waxy face, never a mannequin, doll, or "
        "CGI/3D-render feel.\n"
        "- SKIN, EYES, HAIR, EXPRESSION: Keep light natural skin micro-texture and natural tonal variation across "
        "the skin (not perfectly flat or uniform). Eyes should show a moderate, natural catchlight (never glassy "
        "or overexposed). Hair must show individual strands/locks, not a smooth solid mass. Smiles and "
        "expressions must look natural and relaxed, not stiff or artificially symmetric.\n"
        "- DEFAULT HAND/ARM COUNT: Unless the design brief explicitly calls for something else, every person has "
        "exactly two hands and two arms. Each hand must connect correctly, at a plausible angle, to that SAME "
        "person's own wrist and arm -- never an extra, missing, misplaced, or mismatched hand.\n"
        "- JOINTS, GRIP, OVERLAP, OCCLUSION: Carefully render wrists, fingers (five per hand, natural finger "
        "joints and curvature), elbows/shoulders, and the way hands grip or hold the product/card -- fingers must "
        "wrap around it with plausible physical contact. Handle overlap and occlusion (a hand in front of the "
        "product, a hand behind hair, one hand over the other, etc.) consistently with the scene's lighting and "
        "depth, never producing fused, see-through, or disconnected limbs.\n"
        "- PHONE DISPLAY (when a person holds a smartphone): the person holds exactly ONE phone -- do not invent "
        "a second phone. The phone screen must display realistic Vietnamese-language content (correct Vietnamese "
        "diacritics, natural UI text) relevant to the OCB OMNI app; never show garbled text, English placeholders, "
        "lorem ipsum, or unreadable glyphs on the screen. Fingers must wrap around the phone edge with realistic "
        "pressure and natural curvature; re-check for extra, missing, webbed, or fused fingers before finishing.\n"
        "- NEVER HIDE ERRORS BEHIND THE PRODUCT: Do not use the product, a card, or any other object to mask an "
        "anatomy mistake (a missing finger, a malformed joint, a warped hand). Do not invent a detached, "
        "bodiless hand solely to hold a card or product -- any hand holding something must remain a continuous, "
        "real part of a specific visible person's body.\n"
        "- WARDROBE: Follow the design brief for wardrobe (casual or business), but always render it as refined "
        "and premium -- through fabric quality, cut/fit, and styling -- regardless of style. Accessories are "
        "allowed but should be used sparingly and must never distract from the person or the product.\n"
        "- EYE GAZE / LOOK DIRECTION (strict): Every person must look straight at the viewer/camera with clear, "
        "natural direct eye contact. This remains mandatory even when the person is holding, pointing at, or "
        "presenting a product, card, phone, or screen -- their hands may interact with the product, but their eyes "
        "must stay directed at the viewer, never down at the product. Do NOT render a vague, unfocused, sideward, "
        "middle-distance, or horizon/background gaze."
    )


def _phone_screen_instruction() -> str:
    """2026-09-29 (user request): Vietnamese phone screen content rule.
    Addresses two failure modes observed in practice:
    (1) over-specified instructions → model generates an oversimplified / blank screen;
    (2) under-specified → model defaults to English placeholders or garbled diacritics.
    Fix: OCB brand-anchored style examples + explicit negatives + visual-hierarchy reminder.
    Updated same day: phone screen should look like the real OCB Banking app with OCB colors."""
    return (
        "SMARTPHONE SCREEN CONTENT (applies whenever a smartphone is visible in the design):\n"
        "The phone screen must look like a real, polished screen from the OCB Banking mobile app -- "
        "beautiful, on-brand, and realistic, not a blank or generic demo placeholder. "
        "VISUAL STYLE: Use OCB's brand palette on-screen -- primary green (#00853E) for the top "
        "navigation bar, header background, and primary action buttons; white (#FFFFFF) card "
        "surfaces and body backgrounds; gold (#D4A017) for highlights, badges, or reward indicators; "
        "dark gray (#1A1A1A) for primary body text. The overall look should feel premium and clean, "
        "consistent with a high-quality Vietnamese banking app. "
        "LANGUAGE: All on-screen text must be in correct Vietnamese with full diacritics and tone "
        "marks (ă â đ ê ô ơ ư, tones: à á ả ã ạ etc.). Write UI labels the way the real OCB app "
        "does, for example -- navigation: 'Trang chủ', 'Giao dịch', 'Ví OCB', 'Khuyến mãi'; "
        "buttons/CTAs: 'Xác nhận', 'Tiếp tục', 'Đặt ngay', 'Thanh toán ngay'; "
        "status lines: 'Giao dịch thành công', 'Số dư khả dụng', 'Hoàn tiền thành công'. "
        "CONTEXT MATCH: Reflect the banner's campaign in the on-screen content -- for a transport/"
        "ride campaign show a booking confirmation or route screen; for a cashback/payment campaign "
        "show a payment-success or rewards screen; for a savings/deposit campaign show an "
        "account-summary or interest-rate screen. "
        "FORBIDDEN on the screen: English words or phrases, lorem ipsum, Latin placeholder text, "
        "random alphanumeric strings, garbled glyphs, or any text that does not look like real "
        "Vietnamese. "
        "VISUAL HIERARCHY: The screen must have natural mobile UI density -- top status bar, "
        "a branded header, card sections, body text, and a CTA button -- NOT one oversized word "
        "or number filling the whole display."
    )


def _product_instruction(has_product_assets: bool, product_texts: list[str] | None = None) -> str:
    if not has_product_assets:
        if product_texts:
            # 2026-09-04 (user request): `buckets.product` may describe the
            # intended product visual in words instead of attaching an
            # image -- a creative-generation task, unlike the locked/
            # pixel-exact rules below that apply to a real product asset.
            return (
                "PRODUCT FROM TEXT DESCRIPTION: No `product` image was supplied; instead, `buckets.product` "
                f"describes the intended product visual in words: \"{_join_descriptions(product_texts)}\". "
                "Generate a polished, realistic commercial hero product visual that matches this description as "
                "closely as possible, consistent with the design_brief and brand context. Since no reference "
                "image was supplied, you have creative freedom over exact rendering details, but the result must "
                "still read as one real, coherent, physically plausible product (not an abstract icon or generic "
                "clip-art) and must remain the visual hero of the composition, grounded with natural lighting, "
                "scale, and a soft contact shadow or reflection consistent with the scene."
            )
        return "LOCKED PRODUCT ASSET: No `product` image was supplied; do not invent a product."
    extra_text_clause = (
        # 2026-09-04 (2) (user request): a bucket text entry alongside a
        # product image is an EDIT instruction for staging/context around
        # that product (e.g. added surrounding objects, setting, camera
        # angle) -- unlike theme/context, the product's OWN artwork stays
        # fully locked (see the rules above): this only ever adds
        # surrounding presentation, never changes the product's design.
        f"\nADDITIONAL PRODUCT DIRECTION (text, alongside the attached product image(s) above): "
        f"\"{_join_descriptions(product_texts)}\" -- treat this as an edit instruction for the staging, setting, "
        "surrounding objects, or framing around the locked product asset; it never changes the product image's "
        "own design/artwork, which must stay exactly as supplied."
        if product_texts
        else ""
    )
    # 2026-08-27 (6) (user request, with a reference example): a locked
    # product asset placed dead flat/frontal with no shadow or light reads
    # as a pasted-on sticker, not a real product shot -- the user's
    # reference banner instead gives the card a soft radial light/glow
    # behind it and a subtle shadow/reflection, making it read as the hero
    # subject sitting IN the scene. This is about LIGHTING/PRESENTATION
    # only, never the product's own design (still fully locked below).
    return (
        "LOCKED PRODUCT ASSET: Every image in the `product` bucket is an approved commercial asset that must "
        "remain recognizably exact. Preserve its logo/wordmark, printed text, colors, artwork, material, shape, "
        "proportions, original width-to-height aspect ratio, portrait/landscape orientation, front/back orientation, "
        "reading direction, and photographic realism. Do NOT mirror, flip, rotate a portrait product into landscape "
        "or a landscape product into portrait, stretch, squash, redraw, cartoonize, illustrate, stylize, simplify, "
        "redesign, deform, or replace it. Use the supplied product itself as the foreground asset rather than "
        "generating a look-alike. You are free -- and encouraged -- to choose a deliberate camera angle for visual "
        "impact: a gentle 3D-style tilt, rotation, or perspective view of the whole unchanged asset (as if the same "
        "physical object were photographed from a slightly different angle) is a normal, desirable hero-shot choice, "
        "not a last resort. The only hard limit is inventing geometry the source never showed: do not add a new "
        "side/face, a different thickness/depth, a different casing, or otherwise contradict the reference's real "
        "shape and proportions. Make it read as the scene's hero subject, not a flat pasted sticker. Visually ground "
        "and separate it from the background using the most contextually appropriate combination of local "
        "light/dark contrast, soft directional or rim lighting around it, a restrained gradient or light halo "
        "behind it, and a soft contact shadow or reflection beneath it. Match every effect to the scene's light "
        "source, depth, surface, and perspective; keep the treatment tasteful and harmonious. Apply these effects "
        "to the surrounding scene/presentation only, without changing the product artwork. Do not draw a hard, "
        "uniform outline, border, sticker edge, or harsh spotlight around the product."
        + extra_text_clause
    )


def _theme_instruction(
    has_theme_assets: bool,
    dominant_color: str = "",
    tolerance_percent: int | None = None,
    theme_texts: list[str] | None = None,
) -> str:
    tolerance_clause = (
        f" Supporting shades and gradients must remain within approximately {tolerance_percent}% of this anchor "
        "in hue, saturation, and brightness; use only small contrast accents needed for legibility and product fidelity."
        if dominant_color and tolerance_percent is not None
        else ""
    )
    color_clause_no_image = (
        f"Use {dominant_color} as the dominant palette anchor -- choose that color (or the closest match) "
        "from OCB's approved main/accent palette in the general design rules below. It should lead the "
        f"overall color balance, but it does not require a plain or uniformly colored background.{tolerance_clause}"
        if dominant_color
        else "Use OCB's approved palette and standard background gradient as the color foundation."
    )
    if not has_theme_assets and theme_texts:
        # 2026-09-04 (user request): `buckets.theme` may describe the
        # intended background/environment in words instead of attaching a
        # reference image -- unlike the "no theme at all" branch below,
        # this description IS the campaign's background direction, so it
        # takes the same creative-authority role an attached theme image
        # would (mood/composition/motifs), just expressed in words.
        return (
            "BACKGROUND DIRECTION FROM TEXT DESCRIPTION: No `theme` image was supplied; instead, `buckets.theme` "
            f"describes the intended background/environment in words: \"{_join_descriptions(theme_texts)}\". "
            "Create an original, polished commercial campaign background that faithfully realizes this "
            "description -- its setting, mood, key visual elements, and spatial arrangement -- reinterpreted into "
            f"a full-canvas advertising scene; the description is creative direction, not literal wording to "
            f"render as visible text. {color_clause_no_image} Any decorative motifs must specifically reflect "
            "THIS product's identity and the described environment, integrated through shared lighting, scale, "
            "perspective, overlap, rhythm, and negative space so the result reads as one designed campaign scene, "
            "not a collection of pasted stickers. Keep the background supportive rather than busy, preserve clear "
            "text space, and make the supplied product the visual hero."
        )
    if not has_theme_assets:
        # 2026-08-28 regression fix: making the no-theme path a mandatory
        # plain gradient with minimal flat 2D icons removed too much creative
        # freedom. Repeated generations became rigid split layouts, scattered
        # sticker-like motifs, and a product that looked pasted on. With no
        # theme asset, the brief and product identity should still drive a
        # polished campaign environment with lighting, depth, and spatial
        # relationships. `dominant_color` anchors the palette; it does not
        # force a flat visual medium.
        # 2026-08-27 (2) (user follow-up): each request may now name its OWN
        # dominant color (`design_options.dominant_color`) instead of always
        # the one fixed OCB gradient -- still required to read as an OCB
        # brand color (the approved main/accent palettes in the general
        # design rules below), just not hardcoded to the single default.
        color_clause = (
            f"Use {dominant_color} as the dominant palette anchor -- choose that color (or the closest match) "
            "from OCB's approved main/accent palette in the general design rules below. It should lead the "
            f"overall color balance, but it does not require a plain or uniformly colored background.{tolerance_clause}"
            if dominant_color
            else "Use OCB's approved palette and standard background gradient as the color foundation."
        )
        # 2026-08-27 (3) (user follow-up): a real generated result used
        # generic icons (shopping cart, checklist, globe) that didn't tie
        # back to WHICH specific product/audience the banner targets -- the
        # user pointed out the product itself already signals a clear theme
        # (e.g. a card literally named/styled "Passport" implies a travel/
        # international motif) that the icon choice should follow instead
        # of defaulting to generic finance clip-art.
        return (
            "BACKGROUND DIRECTION: No `theme` image was supplied. Create an original, polished commercial "
            "campaign environment from the design brief, product identity, and target audience. "
            f"{color_clause} Select the visual medium that best fits this campaign; it may use refined gradients, "
            "tonal lighting, subtle texture, atmospheric depth, spatial layering, abstract forms, or restrained "
            "realistic/illustrative scene elements when they strengthen the idea. Do not default to a plain "
            "gradient with small icons scattered over it. Any decorative motifs must specifically reflect THIS "
            "product's identity and use-case (e.g. a travel/international card may use passport-stamp or world-map "
            "language, not generic unrelated finance icons) and must be integrated through shared lighting, scale, "
            "perspective, overlap, rhythm, and negative space so the result reads as one designed campaign scene, "
            "not a collection of pasted stickers. Keep the background supportive rather than busy, preserve clear "
            "text space, and make the supplied product the visual hero."
        )
    # 2026-08-27 (4) (user request): `dominant_color` now applies even when
    # a real `theme` image IS supplied -- each of the 3 promo-banner use
    # cases names its own dominant color per campaign (e.g. green for
    # Passport, dark green for Esaving, cream/gold for World).
    # 2026-08-27 (5) (user follow-up, real run): the first wording here
    # ("shift its palette toward that color") was too soft -- a real
    # generated banner still came out in the theme reference's OWN color
    # (a generic light-green promo background) instead of the requested
    # dominant color. Made this an explicit override: the theme is source
    # material for mood/composition/motifs ONLY, never for color.
    dominant_color_clause = (
        f" COLOR OVERRIDE: {dominant_color} is this campaign's dominant color and takes priority over the theme "
        "reference's own colors -- do not literally reuse the theme image's palette. Recolor the entire "
        "background/scene to be anchored on that dominant color (staying within OCB's approved main/accent "
        f"palette in the general design rules below).{tolerance_clause} Only borrow the theme reference's mood, composition, and "
        "motifs, never its colors."
        if dominant_color
        else ""
    )
    extra_text_clause = (
        # 2026-09-04 (2) (user request, with a real example: an attached
        # `theme` store-interior photo PLUS a text entry in the SAME bucket
        # describing it as a specific "Uniqlo" store with its branding
        # clearly visible): a bucket text entry alongside an image is an
        # EDIT instruction for that image, not just a mood accent -- it may
        # add concrete required elements/branding/signage the base image
        # doesn't already show, blended naturally into the same scene.
        f"\nADDITIONAL BACKGROUND DIRECTION (text, alongside the attached theme image(s) above): "
        f"\"{_join_descriptions(theme_texts)}\" -- treat this as an edit instruction for the attached theme "
        "reference: incorporate the specific elements, branding, signage, or details it describes into the "
        "background scene, reinterpreting or extending the reference as needed to fit them in naturally, in "
        "addition to keeping the reference's own overall mood, composition, and setting."
        if theme_texts
        else ""
    )
    return (
        "CREATIVE BACKGROUND REFERENCE: The image attached in the `theme` bucket supplies the campaign's "
        "background direction, mood, palette, atmosphere, and relevant motifs. It is NOT pixel-locked: you may "
        "reinterpret, improve, extend, simplify, or recompose it into a polished full-canvas advertising "
        f"background.{dominant_color_clause} Treat it as background input only, not as a source of additional "
        "wording or people; render the final headline/message/CTA only from REQUIRED TEXT CONTENT below. Infer "
        "the most suitable medium from the brief and references, favoring a clean modern commercial "
        "hybrid with natural depth, lighting, gradients, and restrained realistic details whenever photographic "
        "people or products are supplied. Decorative styling must remain coherent with the complete banner."
        + extra_text_clause
    )


def _bucket_section(labels_in_order: list[str], req: CreativeRequest) -> str:
    lines = ["ATTACHED REFERENCE IMAGES (in the exact order they are attached to this message):"]
    label_titles = {
        "theme": "CREATIVE BACKGROUND REFERENCE (may be enhanced/recomposed; preserve its essential direction)",
        "context": "LOCKED CONTEXT MARKETING ASSET (inspect actual content; never assume it is a person)",
        "product": "LOCKED PRODUCT MARKETING ASSET (preserve exact design, text, logo, material, and realism)",
        "brand_text": "LOCKED BRAND-TEXT ASSET (finished text artwork; place once in the text zone unchanged)",
        "reference": "REFERENCE (style guide only, not a literal asset to reuse)",
        "guideline_extra": "EXTRA GUIDELINE IMAGE (request.json `guidelines` field)",
        "guideline": "FORMAT GUIDELINE REFERENCE",
        "guideline_font": "FONT REFERENCE (typeface/letterform sample -- match this exact lettering style/weight for any rendered text; it is not a design asset and its own depicted words are not the required text)",
        "safe_area_guide": "SAFE AREA GUIDE (marks the exact crop boundary -- NOT a design asset, match your composition to it)",
    }
    if not labels_in_order:
        lines.append("(none -- no reference images are attached to this request)")
    for i, label in enumerate(labels_in_order, start=1):
        lines.append(f"  {i}. {label_titles.get(label, label)}")
    return "\n".join(lines)


_SAFE_AREA_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s*)?(?:BIÊN AN TOÀN|SAFE AREA|SAFE ZONE)\s*:?[ \t]*$",
    re.IGNORECASE,
)
_DISPLAY_PLACEMENT_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s*)?(?:VỊ TRÍ HIỂN THỊ|DISPLAY (?:LOCATION|PLACEMENT)|PLACEMENT)\s*:?[ \t]*$",
    re.IGNORECASE,
)
_PRODUCTION_SPEC_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s*)?(?:FRAME|EXPORT)\s*:?[ \t]*$",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+|\d+[.)]\s+)")


def _prepare_guideline_for_generation(guideline_text: str) -> str:
    """Remove internal layout-spec material and scope style to AI-made art.

    Safe-area geometry is already described by the canvas section. Repeating
    guideline margin tables (or a leaked ``GEOMETRY`` appendix from an older
    rewrite cache) encourages the image model to draw rulers and labels.
    """
    kept: list[str] = []
    skipping_nonvisual_section = False
    for line in guideline_text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        normalized = lowered.lstrip("-* ")
        if stripped.upper() == "GEOMETRY" or normalized.startswith("this guideline's own frame/margin numbers"):
            break
        if (
            _SAFE_AREA_HEADING_RE.match(line)
            or _DISPLAY_PLACEMENT_HEADING_RE.match(line)
            or _PRODUCTION_SPEC_HEADING_RE.match(line)
        ):
            skipping_nonvisual_section = True
            continue
        if skipping_nonvisual_section:
            if _SECTION_HEADING_RE.match(line):
                skipping_nonvisual_section = False
            else:
                continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    cleaned = cleaned.replace(
        "(4) Phong cách: bắt buộc sử dụng hình vẽ minh họa dạng 2D phẳng (flat design); "
        "nghiêm cấm sử dụng hình ảnh chụp, hình ảnh tả thực hoặc hình ảnh 3D.",
        "(4) Phong cách: chọn chất liệu hình ảnh phù hợp với design brief và các ảnh tham chiếu; ưu tiên phong "
        "cách quảng cáo sạch, hiện đại, chuyên nghiệp, có chiều sâu và ánh sáng tự nhiên hài hòa với "
        "người/sản phẩm.",
    )
    # Stroke-width numbers are production notes, not artwork; exact values
    # have appeared as visible labels in generated banners.
    cleaned = re.sub(r"\s*\(5\) Đường viền:.*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    return cleaned


def build_prompt(
    request_name: str,
    req: CreativeRequest,
    guideline: ResolvedGuideline,
    attachment_labels_in_order: list[str],
    use_llm_refinement: bool | None = None,
    cache_variant: str | None = None,
    usage_tracker=None,
    version: int | None = None,
) -> BuiltPrompt:
    """`attachment_labels_in_order` must exactly match the order images will
    actually be attached to the model call (built by `app/core/pipeline.py`),
    so the prompt's numbered list stays accurate.

    `usage_tracker`/`version`, when passed, record the LLM refinement call
    below (2026-09-06 user request) -- skipped entirely on a cache hit,
    since no call happens then."""
    cache_key = _cache_key(req, guideline, attachment_labels_in_order)

    cache_path = versioned_prompt_cache_file(request_name, cache_variant)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key:
                logger.info("Reusing cached prompt for request=%s (cache key matches)", request_name)
                return BuiltPrompt(
                    text=cached["text"],
                    cache_key=cache_key,
                    attachments_order=attachment_labels_in_order,
                    llm_refined=cached.get("llm_refined", False),
                )
        except Exception:  # noqa: BLE001
            logger.exception("Could not read prompt cache at %s -- rebuilding.", cache_path)

    canvas_w, canvas_h = req.canvas_wh()
    final_w, final_h = req.final_output_wh()
    has_crop = req.has_crop()
    # 2026-09-04 (user request): a bucket entry may be a free-text
    # description instead of an image path (see `Buckets`' docstring in
    # `app/models/schemas.py`) -- split each bucket into its image-like
    # values and its text descriptions so `has_*_assets` below still means
    # "a real image is attached" (unchanged meaning, drives the LOCKED
    # ASSET wording + attachment list), while the text values are folded
    # into the prompt separately as creative direction.
    theme_images, theme_texts = split_bucket_values(req.buckets.theme)
    context_images, context_texts = split_bucket_values(req.buckets.context)
    product_images, product_texts = split_bucket_values(req.buckets.product)
    brand_text_images, brand_text_texts = split_bucket_values(req.buckets.brand_text)
    has_theme_assets = bool(theme_images)
    has_context_assets = bool(context_images)
    has_product_assets = bool(product_images)
    has_brand_text_assets = bool(brand_text_images)

    text = req.text
    sections: list[str] = []

    def _format_text_field(label: str, value: str, empty_note: str) -> str:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if not lines:
            return f"- {label}: {empty_note}"
        if len(lines) == 1:
            return f"- {label}: {lines[0]}"
        numbered = "; ".join(f"line {index}: {line}" for index, line in enumerate(lines, start=1))
        return f"- {label} (render as exactly {len(lines)} separate lines; do not merge): {numbered}"

    sections.append(
        "TASK: Generate ONE finished marketing banner/creative image for OCB (a Vietnamese bank), following "
        "every instruction below exactly. Output must be one finished raster banner at the requested canvas size."
    )

    sections.append(f"DESIGN BRIEF:\n{req.design_brief.strip() or '(none provided)'}")

    persona_block = personas.prompt_direction(
        req, has_theme_assets=has_theme_assets, theme_texts=theme_texts
    )
    if persona_block:
        sections.append(persona_block)

    sections.append(
        "REQUEST CONTROL FLAGS (strict JSON contract; these override suggestions inferred from the brief, brand "
        "guidelines, or reference imagery):\n"
        f"- generate_person={str(req.design_options.generate_person).lower()}: "
        + ("a new person may be generated only when needed." if req.design_options.generate_person else "do not invent any new person, character, hand, arm, wrist, or cropped limb holding a phone/product.")
        + "\n"
        f"- has_logo={str(req.design_options.has_logo).lower()}: "
        + ("include the requested standalone logo." if req.design_options.has_logo else "do not add any standalone or repeated logo; embedded product branding stays embedded only.")
        + "\n"
        f"- populated assets: theme={len(theme_images)} image(s) + {len(theme_texts)} text description(s), "
        f"context={len(context_images)} image(s) + {len(context_texts)} text description(s), "
        f"product={len(product_images)} image(s) + {len(product_texts)} text description(s). "
        f"brand_text={len(brand_text_images)} image(s) + {len(brand_text_texts)} text description(s). "
        "Use only the supplied asset roles and never invent a missing product/context asset."
    )

    if has_crop:
        sa_x, sa_y, sa_w, sa_h = req.crop_box()
        trim_left = sa_x
        trim_top = sa_y
        trim_right = canvas_w - (sa_x + sa_w)
        trim_bottom = canvas_h - (sa_y + sa_h)
        sections.append(
            "CANVAS AND FINAL TRIM:\n"
            f"- Create one continuous finished campaign scene across the complete {canvas_w}x{canvas_h} canvas.\n"
            f"- Delivery post-processing removes {trim_left}px from the left, {trim_right}px from the right, "
            f"{trim_top}px from the top, and {trim_bottom}px from the bottom, producing {final_w}x{final_h}.\n"
            "- Arrange all required wording, people, products, and logos comfortably within the portion that "
            "remains after trimming. Give them natural breathing room and use a smaller scale when necessary.\n"
            "- Continue the same campaign scenery, colors, lighting, and decoration naturally throughout the "
            "complete canvas. The trim is post-processing information only and is not part of the artwork."
        )
    else:
        sections.append(
            f"CANVAS: Generate the image at exactly {canvas_w}x{canvas_h} px. This is also the final output size. "
            "Create one continuous campaign scene across the complete canvas and give all required content "
            "comfortable breathing room."
        )

    # 2026-09-03 (user report, real da21c73170 output): a "Passport"/premium
    # travel-card banner came back with a thin gold decorative line framing
    # the ENTIRE canvas -- not asked for by any brief or guideline, and a
    # direct violation of banner_general_standard's "extend naturally through
    # every edge" rule. The existing anti-border wording (below, in
    # `_product_instruction`) only ever scoped "no outline/border" to the
    # PRODUCT asset, never to the full canvas, so the model was free to add a
    # premium-looking frame around the whole scene. Stated here, once,
    # unconditionally, and explicitly covers solid/opaque decorative borders
    # (not just the pale/translucent safe-area look the validator already
    # knew to catch) since a crisp brand-colored line is what actually showed
    # up.
    sections.append(
        "FULL-BLEED BACKGROUND (no exceptions): The campaign background -- its color, texture, lighting, and "
        "scenery -- must reach and continue naturally across all four canvas edges with zero gap. Do NOT draw a "
        "decorative border, outline, framed panel, rounded card/mat, vignette ring, or any other enclosing "
        "line or shape around the banner's content, no matter how premium, on-brand, or intentional it may look "
        "-- this applies regardless of color, opacity, or thickness (a crisp solid gold/metallic line is just as "
        "forbidden as a faint one). This rule covers the ENTIRE canvas and is separate from the local lighting/"
        "shadow/glow treatment allowed around the product itself elsewhere in this prompt."
    )

    sections.append(_bucket_section(attachment_labels_in_order, req))

    if has_brand_text_assets:
        sections.append(
            "LOCKED BRAND-TEXT PLACEMENT: Each attached `brand_text` image is finished, approved text artwork, "
            "not a product object and not a style reference. Place it exactly once inside the designated text area "
            "at the TOP of the text hierarchy, before any generated headline, message, or CTA. Preserve its supplied "
            "wording, lettering, colors, "
            "artwork, transparency, and aspect ratio. Only uniform scaling and positioning are allowed. Do not redraw, "
            "retype, recolor, crop, distort, restyle, split apart, or duplicate it. Do not generate the product name "
            "or tagline again as separate visible text when either already appears in the brand-text asset; generated "
            "headline, message, and CTA must add new information rather than paraphrase or repeat that artwork. Keep the remaining "
            "headline, message, and CTA visually balanced around it without overlap."
        )

    # 2026-08-27: the 3 promo-banner use cases in
    # AI_DESIGN_REQUEST_INPUT_3usecase.xlsx require headline/message/CTA to
    # always be rendered by code (`app/core/text_compositor.py`) from the
    # exact JSON values, never drawn by the image model -- so instead of
    # handing the model the real text, tell it to leave a calm, text-free
    # reserved column where that code-rendered text will land afterward.
    if req.design_options.text_by_code:
        profile = profile_for(req.final_output_wh())
        # 2026-08-27: verified against a real generation (USECASE2_ESAVING)
        # that the earlier, softer wording ("visually calm column") was not
        # enough -- the model placed the actual product card INSIDE that
        # column, and the code-rendered text ended up drawn on top of it,
        # unreadable. This must be an explicit placement MUST, not a vague
        # mood request: nothing (theme motifs, context, product) may occupy
        # that column at all, empty background only.
        zone_note = (
            f" Reserve the right {int(profile.text_zone_right_fraction * 100)}% of the canvas width as an EMPTY "
            "background-only column -- text will be drawn there afterward, on top of whatever is there, so "
            "nothing may overlap it. Do NOT place the product, the person/context, or any decorative motif/icon "
            "inside that column, even partially; keep them entirely within the remaining left portion of the "
            "canvas. Only plain background color/gradient/texture may extend into the reserved column."
            if profile is not None
            else " Reserve a column on one side of the canvas as an EMPTY background-only area for text to be added afterward -- no product, person, or decorative content may enter it."
        )
        sections.append(
            "TEXT: Do NOT render any headline, message, CTA button, wordmark, or other text anywhere in this "
            "image -- no exceptions, even if other instructions below mention text." + zone_note
        )
    else:
        sections.append(
            "REQUIRED TEXT CONTENT (render exactly as given, do not paraphrase or translate; leave out any field "
            "that is empty):\n"
            + _format_text_field("Headline", text.headline, "(none -- do not render a headline)") + "\n"
            + _format_text_field("Message", text.message, "(none -- do not render a message line)") + "\n"
            f"- CTA button label: {text.cta_label or '(none -- do not render a CTA button)'}\n"
            "TYPOGRAPHY COMPOSITION: Treat the required wording as one professionally composed text block with "
            "clear hierarchy, balanced line lengths, natural line breaks, consistent alignment, and comfortable "
            "spacing relative to the hero visual. Scale it to fit the available negative space without crowding, "
            "stretching, clipping, or leaving an awkward isolated word. Render ONLY the exact supplied wording: "
            "do not add bullets, leading dashes, em/en dashes, divider bars, numbering, labels, quotation marks, "
            "or any other prefix/suffix decoration before or after a headline, message, or CTA. NON-PRINTING "
            "METADATA: Hex color values, font names, pixel measurements, layout dimensions, field names, and "
            "instructions are production metadata only. Never render them as visible text. In particular, any "
            "hash-prefixed color token from the art-direction metadata must stay invisible and must not be appended "
            "to the advertising copy."
        )
        text_color_lock = _db_text_color_lock(req.format_type)
        if text_color_lock:
            sections.append(text_color_lock)
        sections.append(_visible_copy_whitelist_instruction())

    sections.append(_logo_instruction(req, has_context_assets))
    sections.append(
        _theme_instruction(
            has_theme_assets,
            req.design_options.dominant_color,
            req.design_options.dominant_color_tolerance_percent,
            theme_texts,
        )
    )
    sections.append(_person_instruction(req, has_context_assets, context_texts))
    sections.append(_product_instruction(has_product_assets, product_texts))
    # 2026-08-27 (5) (user follow-up, real run of a brief listing many
    # distinct visual elements -- airport lounge, plane, golf, hotel,
    # restaurant, yacht): the result looked like separate stickers pasted
    # in, not one scene -- items didn't share lighting/ground/perspective
    # and felt disconnected. General rule (applies whenever a brief lists
    # multiple items, not specific to any one request): compose them as
    # ONE cohesive scene around the product as the visual hero, the way a
    # real campaign banner would (e.g. a shared ground plane / radial
    # arrangement / consistent light source), not a scattered collage.
    sections.append(
        "MULTI-ELEMENT COMPOSITION: When the design brief lists several visual elements, preserve its breadth of "
        "meaning through visual hierarchy instead of either showing every item at equal weight or deleting most of "
        "them. Select one strong, product-relevant environment or metaphor as the primary campaign idea. Then select "
        "a small coherent set of the remaining relevant ideas (normally two to four) and translate them into "
        "secondary cues -- these can be as simple as small integrated icons or silhouettes, or as rich as textured "
        "surfaces, layered scenic vignettes, embossed patterns, reflections, atmospheric depth, or environmental "
        "details, whichever best serves the campaign's visual richness. Group related cues into one composed visual "
        "unit. Omit only redundant, contradictory, or low-value examples. Supporting motifs must stay subordinate to "
        "the product and message in emphasis (never competing for primary attention), but they do not need to be "
        "minimal -- favor a polished, richly detailed scene over a sparse one, as long as everything shares one "
        "lighting/depth/perspective system. Avoid isolated outline icons floating around the product, abrupt split-screen "
        "joins, hard seams, and unrelated mini-scenes. Connect every selected cue through one continuous atmosphere, "
        "shared lighting, depth, surface, or motion path. Compose the selected elements as ONE cohesive scene, not "
        "a scattered collage of separately-pasted stickers. Give every element the same light source, shadow "
        "direction, and a shared ground/surface or perspective so they visually belong together. Keep the "
        "product as the clear visual hero (largest, most central, or most emphasized), with the other elements "
        "arranged around it in a way that supports it -- e.g. a radial/orbital arrangement, a shared receding "
        "ground plane, or a layered depth arrangement -- rather than placed independently with no relationship "
        "to each other or to the product. The background should have intentional depth, tonal variation, and a clear "
        "visual path toward the product; avoid both empty flat color fields and decorative clutter."
    )

    if req.generation.instruction.strip():
        sections.append(
            "REQUEST-SPECIFIC ART DIRECTION (mandatory; apply without drawing guides, measurements, or labels):\n"
            + req.generation.instruction.strip()
        )

    sections.append(
        "STYLE AND ASSET COHERENCE OVERRIDE (higher priority than every style/guideline rule below): Choose the "
        "background medium from the design brief and theme reference, and keep the result polished, commercial, "
        "and visually coherent with the supplied assets. "
        "Attached `context` people and `product` assets remain locked. A supplied photograph must remain "
        "photographic and a supplied product must retain its real commercial appearance and exact brand "
        "identity. Integrate them with natural lighting, shadows, scale, and spacing without transforming their "
        "content or visual medium."
    )

    # 2026-08-20 (7) (user request, after confirming (6)'s diagnosis was
    # right): rather than only patch the conflict with a static override
    # sentence, rewrite the guideline's own margin numbers via GPT-5-mini
    # (`app/core/guideline_rewriter.py`) so the FORMAT DESIGN GUIDELINE
    # section itself is correct for the actual generation canvas, instead
    # of contradicting the CANVAS & SAFE AREA section above it. Cached
    # (shared across every request/run with the same format_type + canvas +
    # safe-area size, per the user's ask) so the LLM is called once per
    # distinct combination, not once per run.
    # Do not rewrite guideline geometry for the generation canvas. The
    # rewritten prose exposed frame/safe-zone dimensions to the image model,
    # which it repeatedly visualized as a pale rounded content box. Strip
    # non-creative geometry from the original guideline instead.
    guideline_text_for_prompt = _prepare_guideline_for_generation(guideline.combined_text)

    if guideline_text_for_prompt:
        sections.append(f"FORMAT DESIGN GUIDELINE (format_type={req.format_type!r}):\n{guideline_text_for_prompt}")
    else:
        sections.append(
            f"FORMAT DESIGN GUIDELINE: no guideline entry found for format_type={req.format_type!r} in "
            "guidelines.json -- use general good banner-design judgment consistent with the Design Rules below."
        )

    # 2026-08-20 (6) (self-discovered while investigating a user report that
    # gpt-image-2 still wasn't respecting the safe area after every prior
    # fix): the guideline above was written assuming its own margin/edge
    # rules ARE the final word on layout -- e.g. `banner_popup_noti`'s own
    # guidelines.json entry says "hình ảnh điện thoại có giao diện app, có
    # thể phóng to vượt ra ngoài biên an toàn" (phone/app-UI imagery MAY
    # extend past its safe margin). That sentence was written for a design
    # team producing the final banner directly, with no notion of THIS app's
    # 2-stage generation.canvas -> crop -> output.size pipeline -- it never
    # anticipated a stricter, separately-defined safe area governing what
    # survives the crop. This override is only needed as a FALLBACK now
    # (2026-08-20 (7)): when the rewrite above succeeded, the guideline text
    # itself no longer conflicts, so repeating this warning would be
    # redundant noise; it's kept for when the rewrite is skipped (no
    # guideline text) or fails (LLM call error -- REQUEST.md: never let an
    # optional enhancement silently leave the run WORSE off than before it
    # existed).
    rules_text = _design_rules_text()
    if rules_text:
        sections.append(f"DESIGN RULES (MUST / NOT MUST -- these override anything else if they conflict):\n{rules_text}")

    decorative_limit = _decorative_limit_instruction(req.design_options.num_decorative_elements)
    if decorative_limit:
        sections.append(decorative_limit)

    techniques_text = _design_techniques_text(req.design_options.techniques, req.design_options.dominant_color)
    if techniques_text:
        sections.append(f"DESIGN TECHNIQUES (optional supporting art direction):\n{techniques_text}")

    if req.design_options.text_by_code:
        sections.append(
            "OVERRIDE ON THE DESIGN RULES ABOVE (higher priority): ignore the rule about keeping provided text "
            "exactly as specified / not truncating text -- for THIS request, render NO text of any kind at all. "
            "Every other design rule above (locked context/product assets, no cropping people/products/logos, "
            "etc.) still applies unchanged."
        )

    sections.append(
        "BACKGROUND COVERAGE: Use one cohesive campaign environment across every pixel of the canvas. Continue "
        "its scenery, color transitions, texture, lighting, and decorative details naturally and uninterrupted "
        "through all outer areas of the image."
    )

    if req.num_version and req.num_version > 1:
        sections.append(
            f"CREATIVE VARIATION: This exact prompt will be sent to you independently {req.num_version} separate "
            "times to produce distinct creative versions. Treat every call as a fresh opportunity to explore a "
            "genuinely different layout, composition, or decorative motif -- while strictly keeping the same "
            "required assets, text content, and design rules above unchanged. Never vary or reinterpret the "
            "appearance, pose, expression, medium, identity, or design of a supplied person/product asset, EXCEPT "
            "where a CONTEXT ASSET + TEXT EDIT DIRECTION / ADDITIONAL BACKGROUND OR PRODUCT DIRECTION above "
            "explicitly calls for a specific pose/clothing/staging adaptation -- apply that same adaptation "
            "identically across every one of these calls, so the versions stay consistent with EACH OTHER, not "
            "just with the original asset."
        )

    # 2026-09-29 (user request): HUMAN REALISM AND ANATOMY must NOT run when
    # generate_person is false -- its grip/hand language was itself causing a
    # floating hand holding the phone. Use the no-hands lock instead.
    expects_person = bool(
        req.design_options.generate_person or has_context_assets or context_texts
    )
    if expects_person:
        sections.append(_human_realism_instruction())
    else:
        sections.append(_no_person_no_hands_instruction())

    # 2026-09-29 (user request): phone screen Vietnamese-language rule,
    # also placed near the end for emphasis -- applies unconditionally since
    # any banner may include a smartphone mockup.
    sections.append(_phone_screen_instruction())

    if has_context_assets or has_product_assets or has_brand_text_assets:
        sections.append(
            "FINAL LOCKED-ASSET REMINDER: Attached `context`, `product`, and `brand_text` images are the actual "
            "approved marketing assets being advertised, not references to redraw. Preserve their original "
            "realistic medium and identity exactly. Keep all surrounding background and decorative treatment "
            "commercially polished and visually harmonious with them."
        )

    final_text_color_lock = _db_text_color_lock(req.format_type)
    if final_text_color_lock and not req.design_options.text_by_code:
        sections.append("FINAL " + final_text_color_lock)
        sections.append("FINAL " + _visible_copy_whitelist_instruction())

    # 2026-08-20: restated one last time at the very end of the prompt, on
    # purpose -- this is the single MOST common failure mode reported for
    # this app (content spilling past the safe area and getting cropped
    # off), and repeating a hard constraint at both the start and the end
    # of a long, multi-section prompt is measurably more effective than
    # stating it only once in the middle.
    if has_crop:
        sections.append(
            "FINAL COMPOSITION CHECK:\n"
            f"All required wording and supplied advertising assets must remain complete and comfortably visible "
            f"in the delivered {final_w}x{final_h} banner after trimming. Supplied people must stay photographic "
            "and supplied products must keep their exact commercial identity. Return only the finished campaign artwork."
        )

    raw_prompt = "\n\n".join(sections)

    final_text = raw_prompt
    llm_refined = False
    refinement_requested = get_settings().use_llm_prompt_refinement if use_llm_refinement is None else use_llm_refinement
    # Cropped requests use deliberately minimal, positive trim wording. Do
    # not pass it through a second LLM: paraphrasing previously reintroduced
    # safe-zone/container vocabulary and made the image model visualize an
    # inset box. Non-cropped requests retain the existing refinement option.
    should_refine = refinement_requested and not has_crop
    if refinement_requested and has_crop:
        logger.info("Skipping prompt refinement for cropped request to preserve artifact-safe trim wording.")
    if should_refine:
        usage_sink: dict = {} if usage_tracker is not None else None
        try:
            provider = LLMFactory.create(get_settings())
            final_text = provider.generate(raw_prompt, system=_REFINE_SYSTEM_INSTRUCTION, usage_sink=usage_sink)
            llm_refined = True
        except LLMProviderError:
            logger.exception("LLM prompt refinement failed -- falling back to the un-refined prompt.")
        except Exception:  # noqa: BLE001 -- REQUEST.md: never let an optional step crash the pipeline
            logger.exception("Unexpected error during LLM prompt refinement -- falling back to the un-refined prompt.")
        finally:
            if usage_tracker is not None:
                usage_tracker.record_llm(
                    usage_sink, stage="prompt_refinement", version=version, fallback_model=get_settings().llm_model_name
                )

    # 2026-08-20 (8) (user request, with a real gemini-3.1-flash output
    # showing the product/phone mockup bleeding past the safe area's top and
    # bottom edges even after (7)'s guideline rewrite): a final, targeted
    # LLM pass over the WHOLE assembled prompt text -- the actual thing
    # about to be sent to the image model -- specifically hunting for and
    # removing any remaining product-bleed permission, and adding an
    # explicit containment requirement if one isn't already unambiguous.
    # Runs LAST (after refinement, on whatever `final_text` ended up being)
    # so it has the final word regardless of whether refinement changed
    # anything. Only worth calling when there's a product asset attached
    # and a safe area for it to bleed out of.
    # The former LLM containment pass was intentionally removed here. It
    # reintroduced repeated safe-zone/frame terminology after the clean
    # prompt had been assembled, which correlated with visible inset boxes.

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"cache_key": cache_key, "text": final_text, "llm_refined": llm_refined, "raw_prompt": raw_prompt},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        logger.exception("Could not write prompt cache at %s (continuing -- not fatal)", cache_path)

    return BuiltPrompt(text=final_text, cache_key=cache_key, attachments_order=attachment_labels_in_order, llm_refined=llm_refined)


def format_attached_images_note(prepared_images: list) -> str:
    """Human-readable footer block listing exactly which image FILES were
    attached and sent to the model alongside the prompt text -- this is
    local bookkeeping appended to the saved log only, it is NEVER sent to
    the model itself (2026-08-20, user request: "note thêm ... tên các bức
    ảnh nào đã được đính kèm gởi image model" -- the log previously showed
    only generic labels like "1. THEME (background)" inside the prompt
    text, with no actual filename, so there was no way to tell from the log
    alone exactly which file on disk had been sent).

    `prepared_images` is `list[app.core.image_prep.PreparedImage]` (typed
    loosely here to avoid a needless import -- every element just needs
    `.label`, `.resized_path`, `.source_path`)."""
    if not prepared_images:
        return "ATTACHED IMAGES SENT WITH THIS PROMPT: (none)"
    lines = ["ATTACHED IMAGES SENT WITH THIS PROMPT (in order, as actually attached to the model call):"]
    for i, p in enumerate(prepared_images, start=1):
        if p.resized_path.name == p.source_path.name:
            lines.append(f"  {i}. [{p.label}] {p.resized_path.name}")
        else:
            lines.append(f"  {i}. [{p.label}] {p.resized_path.name}  (downsampled from source: {p.source_path.name})")
    return "\n".join(lines)


def write_prompt_log(request_name: str, image_model: str, version: int, prompt_text: str, attached_images_note: str | None = None) -> Path:
    """ALWAYS (over)writes `prompt/prompt-<model>-version-<n>.txt` with the
    EXACT prompt text (+ attached-images footer) actually being sent for
    THIS (model, version) call right now.

    2026-08-20 (3) (user-reported bug): this used to skip writing whenever
    the target file already existed on disk, on the theory that REQUEST.md
    [3d]'s "chỉ nên xuất 1 lần" meant the log itself should never be
    touched again. In practice that made the log LIE: the user re-ran a
    request after the SAFE AREA GUIDE + attached-images-footer features were
    added, `build_prompt()` correctly rebuilt fresh prompt text (see
    `_PROMPT_TEMPLATE_VERSION` above), but this function silently kept the
    STALE `.txt` file from an earlier run with the SAME (model, version)
    pair on disk -- so the log the user actually opened to check never
    reflected any of the new wording, even though the real prompt sent to
    the model already had it. [3d]'s actual intent -- don't waste an LLM
    refinement call re-building the prompt TEXT for every model -- is
    already handled by `build_prompt()`'s own `_prompt_cache/built_prompt.json`
    cache; that's the expensive step worth reusing. Writing a small local
    `.txt` log file is cheap and its whole purpose is to always be an
    accurate record of what was actually sent -- so it is unconditionally
    (re)written every call, never skipped.

    `attached_images_note` (see `format_attached_images_note()`) is
    appended as a footer AFTER the actual prompt text -- it is NOT part of
    `prompt_text` itself and is never sent to the model, it only makes the
    saved log file on disk self-contained (2026-08-20 (2), user request)."""
    from app.core.output_manager import prompt_dir, prompt_file_path

    prompt_dir(request_name).mkdir(parents=True, exist_ok=True)
    path = prompt_file_path(request_name, image_model, version)
    content = prompt_text
    if attached_images_note:
        content = f"{prompt_text}\n\n{'=' * 80}\n{attached_images_note}\n"
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote prompt log %s", path)
    return path
