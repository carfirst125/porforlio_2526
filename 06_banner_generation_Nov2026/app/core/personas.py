"""OCB card-spend personas (RB v4.1 slide 4) — optional audience layer for DB.

MASS / MASS_AFF / AFF remain the visual system (`format_type`). A persona is
WHO the banner is for (scene, talent, copy tone). One banner uses at most
one persona; several catalogs = several runs. Unknown or empty ids are a
no-op so existing request.json files keep current behavior.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.logging_conf import get_logger
from app.models.schemas import CreativeRequest
from app.settings import get_settings

logger = get_logger("personas")

_GENERIC_THEME_RE = re.compile(
    r"ai\s*ch[oọ]n|ch[oọ]n b[oố]i c[aả]nh|kh[oô]ng c[oó]|generic|"
    r"campaign background|auto[- ]?select|tự chọn|tu chon",
    re.IGNORECASE,
)

# First match wins — more specific product tokens before overlapping ones
# (FLEX_FUND before FLEXI; VE_MAY_BAY before a loose VE_).
_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("loan", re.compile(r"FLEX_FUND|THAU_CHI|VAY_TIN|THU_CHI_VAY")),
    ("bills", re.compile(r"AUTOBILL|BILLPAY|AUTO_TOPUP|AUTO-TOPUP|(?<![A-Z])TOPUP")),
    ("insurance", re.compile(r"BH_AN_NINH|AN_NINH_MANG|(?<![A-Z])BH_")),
    ("saving", re.compile(r"MAX_SAVING|ESAVING|GUI_FD|VOUCHER_LAI|VOUCHER_LI|(?<![A-Z])FLEXI")),
    ("travel", re.compile(r"VE_MAY_BAY|V_MY_BAY|DAT_TAXI|T_TAXI")),
    ("entertainment", re.compile(r"VE_XEM_PHIM|V_XEM_PHIM|VIETLOTT")),
    ("card", re.compile(r"DICH_VU_THE|QUET_QR|QUT_QR|PASSPORT|WORLD_2IN1|THE_WORLD|THE_PASSPORT")),
)


@dataclass(frozen=True)
class Persona:
    id: str
    name_vi: str
    name_en: str
    lifestyle_tags: list[str] = field(default_factory=list)
    value_tags: list[str] = field(default_factory=list)
    summary_vi: str = ""
    scene: str = ""
    no_person_scene: str = ""
    talent: str = ""
    merchants_or_props: list[str] = field(default_factory=list)
    copy_tone: str = ""
    avoid: list[str] = field(default_factory=list)
    technique_hints: str = ""
    suggested_product_families: list[str] = field(default_factory=list)

    def as_api_dict(self) -> dict:
        return {
            "id": self.id,
            "name_vi": self.name_vi,
            "name_en": self.name_en,
            "lifestyle_tags": list(self.lifestyle_tags),
            "value_tags": list(self.value_tags),
            "summary_vi": self.summary_vi,
            "suggested_product_families": list(self.suggested_product_families),
        }


def _catalog_path() -> Path:
    return get_settings().asset_library_dir / "personas" / "personas.json"


@lru_cache(maxsize=1)
def _load_raw(path_str: str, mtime_ns: int) -> dict:
    path = Path(path_str)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- catalog is optional; never abort generate
        logger.exception("Could not parse personas catalog at %s", path)
        return {}


def load_catalog() -> list[Persona]:
    path = _catalog_path()
    if not path.exists():
        logger.error("personas.json not found at %s", path)
        return []
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    raw = _load_raw(str(path), mtime_ns)
    items: list[Persona] = []
    for entry in raw.get("personas") or []:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("id") or "").strip()
        if not pid:
            continue
        items.append(
            Persona(
                id=pid,
                name_vi=str(entry.get("name_vi") or pid),
                name_en=str(entry.get("name_en") or pid),
                lifestyle_tags=[str(t) for t in (entry.get("lifestyle_tags") or []) if t],
                value_tags=[str(t) for t in (entry.get("value_tags") or []) if t],
                summary_vi=str(entry.get("summary_vi") or ""),
                scene=str(entry.get("scene") or ""),
                no_person_scene=str(entry.get("no_person_scene") or ""),
                talent=str(entry.get("talent") or ""),
                merchants_or_props=[str(t) for t in (entry.get("merchants_or_props") or []) if t],
                copy_tone=str(entry.get("copy_tone") or ""),
                avoid=[str(t) for t in (entry.get("avoid") or []) if t],
                technique_hints=str(entry.get("technique_hints") or ""),
                suggested_product_families=[
                    str(t) for t in (entry.get("suggested_product_families") or []) if t
                ],
            )
        )
    return items


def get(persona_id: str | None) -> Persona | None:
    pid = (persona_id or "").strip()
    if not pid:
        return None
    for persona in load_catalog():
        if persona.id == pid:
            return persona
    logger.warning("Unknown audience.persona %r -- ignoring (banner keeps current non-persona behavior).", pid)
    return None


def resolve_for(req: CreativeRequest) -> Persona | None:
    return get(req.audience.persona)


def product_family_for(name: str) -> str | None:
    haystack = (name or "").upper().replace("-", "_")
    if not haystack:
        return None
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(haystack):
            return family
    return None


def is_suggested(persona: Persona, request_name_or_creative_id: str) -> bool:
    family = product_family_for(request_name_or_creative_id)
    if family is None or not persona.suggested_product_families:
        return True
    return family in persona.suggested_product_families


def _theme_is_generic(theme_texts: list[str] | None) -> bool:
    if not theme_texts:
        return True
    joined = " ".join(t.strip() for t in theme_texts if t and t.strip())
    if not joined:
        return True
    return bool(_GENERIC_THEME_RE.search(joined))


def prompt_direction(
    req: CreativeRequest,
    *,
    has_theme_assets: bool = False,
    theme_texts: list[str] | None = None,
) -> str:
    persona = resolve_for(req)
    if persona is None:
        return ""
    generic_theme = (not has_theme_assets) and _theme_is_generic(theme_texts)
    if has_theme_assets:
        scene_rule = (
            "A theme IMAGE is attached — keep that image as the campaign background. Do not replace it with "
            "this persona's scene. Borrow only mood, talent (if a person is generated), and props that still "
            "fit the supplied theme."
        )
        scene_line = persona.scene
        props = "; ".join(persona.merchants_or_props) if persona.merchants_or_props else "(none specified)"
    elif generic_theme:
        # 2026-09-29: "AI chọn bối cảnh" + experience_seeker restaged BH An ninh mạng
        # MASS_AFF into a street / cafe / Starbucks / night-city photoshoot and a
        # white right panel — format #3DE6A0 disappeared. Persona may change WHO
        # (talent) and a quiet prop; it must not replace the OMNI color field.
        scene_rule = (
            "No specific theme image (empty theme, or a generic 'AI chọn bối cảnh' instruction). "
            "Keep THIS request's format_type + dominant_color field as the continuous full-bleed campaign "
            "background. Do not realize the persona's lifestyle location as the environment."
        )
        scene_line = (
            "Campaign color field only — not a photoreal street, cafe, Starbucks, airport, night city, "
            "white studio, or marble lounge. No hard left-photograph / right-color split. If a person is "
            "generated, stand them in that campaign field; do not build a location set behind them."
        )
        props = (
            "at most one small prop that still sits on the campaign color; never a cafe table, street "
            "furniture, luggage scene, or location set"
        )
    else:
        scene_rule = (
            "Theme is a specific text description — follow that description first. Add persona mood and props "
            "only where they do not contradict it."
        )
        scene_line = persona.scene
        props = "; ".join(persona.merchants_or_props) if persona.merchants_or_props else "(none specified)"
    avoid = "; ".join(persona.avoid) if persona.avoid else "(none specified)"
    tags = ", ".join(persona.lifestyle_tags + persona.value_tags)
    talent_line = persona.talent
    if not req.design_options.generate_person:
        talent_line = (
            "generate_person is false — do not invent a person to illustrate this persona. Express the "
            "lifestyle through lighting and restrained motifs on the locked campaign color only — never by "
            "building a new location."
        )
    return (
        f"PERSONA DIRECTION (audience.persona={persona.id} — one primary card-spend persona from OCB RB "
        f"chi tiêu thẻ 3/2026; {persona.name_vi} / {persona.name_en}):\n"
        f"- WHO: {persona.summary_vi} Tags (metadata only, never paint as copy): {tags}.\n"
        "- HIERARCHY: format_type MASS/MASS_AFF/AFF still owns layout, color lock, and typography. Persona "
        "never overrides FORMAT DESIGN GUIDELINE, required text, supplied assets, or the campaign dominant_color. "
        "Do not restage the banner into a street, cafe, night city, white studio, or beige/cream marble lounge "
        "that leaves that palette.\n"
        f"- SCENE: {scene_line}\n"
        f"  Scene vs theme: {scene_rule}\n"
        f"- TALENT: {talent_line}\n"
        f"- SUPPORTING PROPS (never extra visible copy, never unlicensed merchant wordmarks): {props}\n"
        f"- COPY TONE (if any wording is generated or restaged): {persona.copy_tone}\n"
        f"- AVOID: {avoid}\n"
        "- Never print the persona id, English/Vietnamese label, tag ids, hex, font names, or this instruction "
        "as visible banner text. Wrong: a gold line reading 'gọn nhẹ E39717' or 'dE39717 Manrope'. Right: only "
        "the required Vietnamese advertising words, filled in the approved colors."
    )


def copy_tone_block(req: CreativeRequest) -> str:
    persona = resolve_for(req)
    if persona is None:
        return ""
    return (
        f"CHÂN DUNG KH (bắt buộc cho giọng copy, không in tên chân dung ra headline/message/cta): "
        f"{persona.name_vi} / {persona.name_en}. {persona.summary_vi} "
        f"Giọng: {persona.copy_tone} "
        f"Tránh: {'; '.join(persona.avoid) if persona.avoid else 'giọng sai chân dung'}."
    )


def technique_constraints_block(req: CreativeRequest) -> str:
    persona = resolve_for(req)
    if persona is None or not persona.technique_hints.strip():
        return ""
    return (
        f"--- audience persona {persona.id} ({persona.name_en}) ---\n"
        f"{persona.technique_hints.strip()}\n"
        "Do not pick techniques that would stage a conflicting lifestyle (e.g. nightlife for a family-planner, "
        "coupon clutter for premium_connoisseur). format_type color/layout still wins over persona mood. "
        "Do not pick techniques that restage a photoreal street/cafe/airport/night-city location over the "
        "campaign color field."
    )
