from __future__ import annotations

from app.core import personas
from app.models.schemas import CreativeRequest
from app.settings import PROJECT_ROOT


def _req(**overrides) -> CreativeRequest:
    raw = {
        "creative_id": "DB_AUTOBILL_1024x440_MASS",
        "format_type": "ocb_omni_110926_mass_1024x440",
        "design_brief": "Thanh toán hóa đơn tự động.",
        "buckets": {"theme": [], "context": [], "product": []},
        "text": {"headline": "H", "message": "M", "cta_label": "C"},
        "design_options": {"generate_person": True, "has_logo": False},
        "generation": {
            "canvas": {"width": 1024, "height": 440},
            "safe_area": {"width": 1024, "height": 440, "x": 0, "y": 0},
        },
        "output": {"size": {"width": 1024, "height": 440}},
    }
    raw.update(overrides)
    return CreativeRequest.model_validate(raw)


def test_catalog_has_exactly_eight_personas_from_slide_4():
    catalog = personas.load_catalog()
    ids = [p.id for p in catalog]
    assert ids == [
        "experience_seeker",
        "digital_shopaholic",
        "lifestyle_upgrader",
        "family_first_planner",
        "value_savvy_family",
        "future_builder",
        "premium_connoisseur",
        "health_serenity_seeker",
    ]


def test_unknown_or_blank_persona_is_a_noop():
    assert personas.get("") is None
    assert personas.get(None) is None
    assert personas.get("not_a_real_persona") is None
    assert personas.prompt_direction(_req()) == ""
    assert personas.copy_tone_block(_req()) == ""
    assert personas.technique_constraints_block(_req()) == ""


def test_generic_theme_keeps_campaign_field_and_does_not_restage_location():
    req = _req(audience={"persona": "experience_seeker"})
    text = personas.prompt_direction(req, has_theme_assets=False, theme_texts=["AI chọn bối cảnh"])
    assert "PERSONA DIRECTION" in text
    assert "experience_seeker" in text
    assert "Realize THIS persona's scene" not in text
    assert "café terrace" not in text
    assert "airport lounge" not in text
    assert "Keep THIS request's format_type + dominant_color field" in text
    assert "Do not realize the persona's lifestyle location" in text
    assert "No hard left-photograph / right-color split" in text
    assert "never a cafe table" in text
    assert "street, cafe, night city, white studio" in text
    assert "Never print the persona id" in text
    assert "gọn nhẹ E39717" in text


def test_prompt_direction_keeps_theme_image_and_skips_talent_when_no_person():
    req = _req(
        audience={"persona": "family_first_planner"},
        design_options={"generate_person": False, "has_logo": False},
    )
    text = personas.prompt_direction(req, has_theme_assets=True, theme_texts=[])
    assert "keep that image as the campaign background" in text
    assert "do not invent a person" in text


def test_premium_stays_light_and_still_forbids_hex_as_copy():
    req = _req(
        audience={"persona": "premium_connoisseur"},
        design_options={"generate_person": False, "has_logo": False},
    )
    text = personas.prompt_direction(req, has_theme_assets=False, theme_texts=[])
    assert "Keep THIS request's format_type + dominant_color field" in text
    assert "beige/cream marble lounge" in text
    assert "Realize THIS persona's scene" not in text
    assert "quiet luxury space" not in text
    assert "lounge chair" not in text
    assert "gallery-like negative space" not in text
    assert "NO-PERSON STILL LIFE" not in text
    assert "BACKGROUND CONTINUITY" not in text
    assert "stemmed glass" not in text
    assert "gọn nhẹ E39717" in text
    assert "do not invent a person" in text


def test_theme_image_still_keeps_supplied_background_and_may_dump_persona_mood():
    req = _req(audience={"persona": "experience_seeker"})
    text = personas.prompt_direction(req, has_theme_assets=True, theme_texts=["AI chọn bối cảnh"])
    assert "keep that image as the campaign background" in text
    assert "Realize THIS persona's scene" not in text
    assert "café terrace" in text


def test_product_family_mapping_for_db_usecases():
    assert personas.product_family_for("DB_AUTOBILL_1024x440_MASS") == "bills"
    assert personas.product_family_for("DB_POPUP_AUTO_TOPUP_720x1280_AFF") == "bills"
    assert personas.product_family_for("DB_BH_AN_NINH_MANG_1024x440_MASS") == "insurance"
    assert personas.product_family_for("DB_FLEXI_1024x440_MASS") == "saving"
    assert personas.product_family_for("DB_FLEX_FUND_1024x440_AFF") == "loan"
    assert personas.product_family_for("DB_VE_MAY_BAY_1024x440_MASS") == "travel"
    assert personas.product_family_for("DB_VE_XEM_PHIM_1024x440_AFF") == "entertainment"
    assert personas.product_family_for("DB_QUET_QR_1024x440_MASS") == "card"
    assert personas.product_family_for("UNKNOWN_DEMO") is None


def test_autobill_suggests_digital_shopaholic_not_premium():
    shop = personas.get("digital_shopaholic")
    premium = personas.get("premium_connoisseur")
    assert shop is not None and premium is not None
    assert personas.is_suggested(shop, "DB_AUTOBILL_1024x440_MASS") is True
    assert personas.is_suggested(premium, "DB_AUTOBILL_1024x440_MASS") is False


def test_existing_request_json_still_validates_without_audience_field():
    path = PROJECT_ROOT / "input" / "DB_AUTOBILL_1024x440_MASS.json"
    req = CreativeRequest.model_validate_json(path.read_text(encoding="utf-8"))
    assert req.audience.persona is None
