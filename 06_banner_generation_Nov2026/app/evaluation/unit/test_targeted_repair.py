from app.core.generate_repair_graph import _build_targeted_repair
from app.core.pipeline import load_request_json
from app.providers.base import ImageAttachment


def test_targeted_repair_uses_failed_raw_first_and_drops_visual_guide():
    req = load_request_json("DB_NOTI_DEMO_TRUNGTHU_huy")
    original = [
        ImageAttachment("theme", "image/jpeg", b"theme"),
        ImageAttachment("product", "image/png", b"product"),
        ImageAttachment("safe_area_guide", "image/png", b"guide"),
    ]
    failed_png = b"\x89PNG\r\n\x1a\nfailed"

    prompt, attachments = _build_targeted_repair(
        req,
        failed_png,
        original,
        ["required text or advertising asset is clipped or too close to an output edge"],
    )

    assert [item.label for item in attachments] == ["failed_full_canvas", "theme", "product"]
    assert attachments[0].data == failed_png
    assert "do not create a new concept" in prompt
    assert "Move and/or uniformly scale" in prompt
    assert "inner content-safe region" not in prompt
    assert "safe-area" not in prompt.lower()
    assert "panel" not in prompt.lower()
    assert "rectangle" not in prompt.lower()
    assert "frame" not in prompt.lower()
    assert "safe_area_guide" not in prompt
    assert req.text.headline in prompt
