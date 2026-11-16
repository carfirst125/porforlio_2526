"""Create the 18 request JSONs from AI_DESIGN_REQUEST_INPUT_3usecase_v4."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"

CASES = {
    1: {
        "name": "PASSPORT",
        "brief": (
            "Thiết kế banner quảng bá thẻ OCB Mastercard Passport. Nhấn mạnh hoàn tiền đến 3%, mở thẻ online "
            "miễn phí, phí ngoại tệ chỉ 1,2%. Phong cách hiện đại, quốc tế, phù hợp khách hàng thường xuyên mua "
            "sắm online và giao dịch quốc tế."
        ),
        "buckets": {"theme": [], "context": [], "product": "AIsquad_images/usecase1sp.png"},
        "headline": "Hoàn tiền đến 3% mỗi ngày",
        "message": "",
        "cta": "Mở thẻ Ngay",
        "flags": (False, True, False),
    },
    2: {
        "name": "ESAVING",
        "brief": (
            "Thiết kế banner quảng bá sản phẩm Tiền gửi có kỳ hạn OCB dành cho khách hàng mong muốn tối ưu dòng "
            "tiền nhàn rỗi, tích lũy tài sản an toàn và sinh lời hiệu quả. Hình ảnh thể hiện sự tăng trưởng tài "
            "chính vững chắc với các biểu tượng như cây tài lộc phát triển, đồng tiền vàng, biểu đồ tăng trưởng, "
            "két sắt hiện đại, ngôi nhà, gia đình và các mục tiêu tài chính trong tương lai."
        ),
        "buckets": {
            "theme": [],
            "context": "AIsquad_images/usecase2ngucanh.png",
            "product": "AIsquad_images/usecase2sp.png",
        },
        "headline": "",
        "message": "",
        "cta": "",
        "flags": (True, True, True),
    },
    3: {
        "name": "WORLD",
        "brief": (
            "Thiết kế banner quảng bá thẻ OCB Mastercard World 2in1 dành cho nhóm khách hàng ưu tiên có thu nhập "
            "cao, thường xuyên công tác, du lịch quốc tế và sử dụng các dịch vụ cao cấp. Hình ảnh thể hiện phong "
            "cách sống đẳng cấp với các biểu tượng như phòng chờ sân bay hạng thương gia, máy bay, sân golf, khách "
            "sạn 5 sao, nhà hàng cao cấp, du thuyền và các điểm đến quốc tế nổi tiếng. Làm nổi bật các đặc quyền "
            "vượt trội như hoàn tiền đến 10% ẩm thực, hoàn tiền du lịch 5%, phòng chờ sân bay không giới hạn quy "
            "đổi, ưu đãi golf, bảo hiểm du lịch toàn cầu và chi tiêu quốc tế với phí ngoại tệ cạnh tranh."
        ),
        "buckets": {"theme": [], "context": [], "product": "AIsquad_images/usecase3sp.png"},
        "headline": "OCB MASTERCARD WORLD 2IN1\nMở khóa thế giới đặc quyền",
        "message": "Hoàn tiền đến 10% | Tổng hoàn tiền 60 triệu/năm",
        "cta": "Sở hữu ngay",
        "flags": (False, False, False),
    },
}

TIERS = {
    "TIER_AFF": "#0F472F",
    "MASS_AFF": "#3DE6A0",
    "MASS": "#B3F3BB",
}


def art_direction(case_no: int, tier: str, width: int, height: int) -> str:
    is_tier = tier == "TIER_AFF"
    if width == 1024 and height == 440:
        layout = (
            "Use a professional two-zone composition: a clean typography block on the left and the supplied hero "
            "product/context composition on the right. Keep about 64px outer padding; the text zone may use about "
            "420px and may extend by at most 40px when necessary."
        )
        headline = "Manrope Semibold, visually equivalent to 40–52px, 8–12 words, exactly two balanced lines."
        message = "Manrope Medium, visually equivalent to 24–28px, maximum 10 words on one line, 8–14px below headline."
        cta = (
            "Use a restrained #00804B CTA button no wider than 412px. CTA text is white Manrope Semibold, maximum "
            "5 words, with generous horizontal and vertical padding."
        )
    else:
        layout = (
            "Use the approved low-banner two-zone composition with about 40px outer padding. "
            + (
                "Place the typography block on the left (up to about 524px) and the supplied hero visual on the right."
                if is_tier
                else "Place the supplied hero visual on the left (about 420px) and the typography block on the right (about 524px)."
            )
        )
        headline = "Manrope Semibold, visually equivalent to 36–44px, concise 6–10 words with compact balanced line breaks."
        message = (
            "Manrope Medium, visually equivalent to 24–28px, maximum 10 words on one line, 6–10px below headline; "
            "show it only when the headline fits cleanly without crowding."
        )
        cta = "Do not render a CTA button or CTA label at this 1024x256 size."

    if case_no == 3:
        headline_color = "Use two coordinated headline tones chosen from #76540C, #DDC37D, and #E39717."
        message_color = "Use #76540C for the message."
    elif is_tier:
        headline_color = "Use two coordinated headline tones chosen from #C5A049, #DDC37D, and #E4DABF."
        message_color = "Use #FCF3C5 for the message."
    else:
        headline_color = "Use #00804B for the first headline line and #E39717 for the second; at 256px prefer #00804B when one line."
        message_color = "Use #545C57 for the message."

    return " ".join((layout, headline, headline_color, message, message_color, cta))


def build(case_no: int, tier: str, width: int, height: int) -> dict:
    case = CASES[case_no]
    gen_headline, gen_message, gen_cta = case["flags"]
    is_short = height == 256
    creative_id = f"JSON_V3_USECASE{case_no}_{case['name']}_{tier}_{width}x{height}"
    return {
        "creative_id": creative_id,
        "format_type": f"banner_promo_card_{width}x{height}",
        "design_brief": case["brief"],
        "buckets": case["buckets"],
        "reference": "",
        "text": {
            "headline": case["headline"],
            "message": case["message"],
            "cta_label": "" if is_short else case["cta"],
            "cta_deep_link": "",
        },
        "guidelines": "",
        "num_version": 3,
        "design_options": {
            "generate_person": False,
            "has_logo": False,
            "text_by_code": False,
            "auto_generate_text": False,
            "generate_text": False,
            "generate_text_headline": gen_headline,
            "generate_text_message": gen_message,
            "generate_text_cta_label": gen_cta and not is_short,
            "num_decorative_elements": 2,
            "dominant_color": "#FCF3C5" if case_no == 3 else TIERS[tier],
            "dominant_color_tolerance_percent": 15,
        },
        "generation": {
            "canvas": {"width": width, "height": height},
            "safe_area": {"width": width, "height": height, "position": "center", "x": 0, "y": 0},
            "instruction": art_direction(case_no, tier, width, height),
        },
        "output": {"size": {"width": width, "height": height}, "clone": []},
    }


def main() -> None:
    for case_no in CASES:
        for tier in TIERS:
            for width, height in ((1024, 440), (1024, 256)):
                payload = build(case_no, tier, width, height)
                path = INPUT / f"{payload['creative_id']}.json"
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
