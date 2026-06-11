"""
Advisor Profile Recall — 2 nodes:

1. advisor_profile_recall_node:
   Chèn vào giữa domain_detector và field_extractor.
   - Nếu user đã có profile cho domain này → pre-fill collected_info,
     hiện tin nhắn xác nhận lại thông tin, set awaiting_profile_confirm=True → END.
   - Nếu chưa có → pass-through (empty dict) → route sang field_extractor.

2. advisor_profile_update_node:
   Gọi khi user phản hồi tin nhắn xác nhận (awaiting_profile_confirm=True).
   - Dùng LLM trích xuất updates từ câu trả lời user.
   - Merge vào collected_info.
   - Nếu user xác nhận OK / không thay đổi → collected_info complete → route sang advisor_retrieve.
   - Nếu user cập nhật 1 số field → merge → route sang advisor_retrieve (đủ thông tin).
   - Nếu user hỏi lại hoàn toàn khác → reset (route sang field_extractor để hỏi lại).
"""
import json
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.history.advisor_profile_store import get_profile_store
from src.knowledge_graph.field_definitions import DOMAIN_LABELS, DOMAIN_FIELDS
from src.llm import get_llm, parse_json

# ── Field display labels ──────────────────────────────────────────────────────
_FIELD_LABELS = {
    "thu_nhap_hang_thang":  "Thu nhập hàng tháng",
    "muc_chi_tieu_chu_yeu": "Chi tiêu chủ yếu",
    "uu_tien_quyen_loi":    "Quyền lợi ưu tiên",
    "co_the_hien_tai":      "Thẻ tín dụng hiện tại",
    "tuoi":                 "Độ tuổi",
    "tinh_trang_gia_dinh":  "Tình trạng gia đình",
    "muc_dich_bao_hiem":    "Mục đích bảo hiểm",
    "ngan_sach_hang_thang": "Ngân sách hàng tháng",
    "muc_dich_vay":         "Mục đích vay",
    "so_tien_can_vay":      "Số tiền cần vay",
    "tai_san_the_chap":     "Tài sản thế chấp",
    "so_tien_gui":          "Số tiền gửi",
    "thoi_han_gui":         "Thời hạn gửi",
    "muc_tieu":             "Mục tiêu tiết kiệm",
    "loai_san_pham":        "Sản phẩm quan tâm",
}

# ── Update extraction prompt ──────────────────────────────────────────────────
UPDATE_EXTRACT_PROMPT = """Người dùng đang phản hồi sau khi bot nhắc lại thông tin đã lưu.

Thông tin hiện đang lưu:
{current_profile}

Câu trả lời của người dùng: "{user_response}"

Nhiệm vụ: Xác định xem người dùng có muốn cập nhật thông tin nào không.

Trả về JSON với 2 keys:
- "confirmed": true nếu người dùng đồng ý/xác nhận (dù có hay không có cập nhật), false nếu muốn hỏi lại từ đầu
- "updates": object chứa các field được cập nhật (chỉ những field thay đổi), hoặc {{}} nếu không thay đổi gì

Fields hợp lệ: {valid_fields}

Ví dụ:
- "OK đúng rồi" → {{"confirmed": true, "updates": {{}}}}
- "Vẫn vậy nhé" → {{"confirmed": true, "updates": {{}}}}
- "Thu nhập tôi bây giờ 70 triệu rồi" → {{"confirmed": true, "updates": {{"thu_nhap_hang_thang": "70 triệu"}}}}
- "Tôi muốn vay 5 tỷ thay vì 4 tỷ, còn lại vẫn như cũ" → {{"confirmed": true, "updates": {{"so_tien_can_vay": "5 tỷ"}}}}
- "Thôi để tôi hỏi lại từ đầu" → {{"confirmed": false, "updates": {{}}}}

Chỉ output JSON, không giải thích thêm."""


def _format_profile_for_confirm(domain: str, profile: dict) -> str:
    """Tạo tin nhắn xác nhận thông tin đã lưu."""
    domain_label = DOMAIN_LABELS.get(domain, domain)
    lines = [
        f"Mình thấy lần trước bạn đã tư vấn về **{domain_label}** và cung cấp các thông tin sau:\n"
    ]
    for field, value in profile.items():
        label = _FIELD_LABELS.get(field, field)
        lines.append(f"- {label}: **{value}**")

    lines.append(
        "\nThông tin này vẫn còn đúng không ạ? 😊\n"
        "- Nếu **vẫn vậy**, bạn chỉ cần nhắn _\"OK\"_ hoặc _\"Đúng rồi\"_ là mình tư vấn ngay!\n"
        "- Nếu có **thay đổi**, bạn cứ nói cho mình biết nhé (ví dụ: _\"thu nhập tôi bây giờ 70 triệu\"_)."
    )
    return "\n".join(lines)


# ── Node 1: Profile Recall ────────────────────────────────────────────────────

def advisor_profile_recall_node(state: ChatState) -> dict:
    """
    Kiểm tra profile đã lưu cho user+domain.
    - Có profile → trả về confirm message, set awaiting_profile_confirm=True
    - Không có   → trả về dict rỗng (route tiếp sang field_extractor)
    """
    domain = state.get("advisor_domain") or "general"
    user_id = state.get("user_id") or "UID0000"

    profile_store = get_profile_store()
    saved_profile = profile_store.load_profile(user_id, domain)

    if not saved_profile:
        logger.info(
            f"Profile recall: no saved profile for user={user_id}, domain={domain} → "
            "proceeding to field_extractor."
        )
        return {}  # route sang field_extractor

    logger.info(
        f"Profile recall: found saved profile for user={user_id}, domain={domain} | "
        f"fields={list(saved_profile.keys())}"
    )

    confirm_msg = _format_profile_for_confirm(domain, saved_profile)

    return {
        "collected_info": saved_profile,
        "required_fields": DOMAIN_FIELDS.get(domain, {}),
        "missing_fields": [],       # tạm thời — update node sẽ xác nhận
        "awaiting_profile_confirm": True,
        "turn_count": 0,
        "messages": [AIMessage(content=confirm_msg)],
    }


# ── Node 2: Profile Update (xử lý phản hồi xác nhận) ────────────────────────

def advisor_profile_update_node(state: ChatState) -> dict:
    """
    Xử lý câu trả lời của user sau khi bot hiển thị profile cũ.

    - User xác nhận (OK/đúng) → dùng profile hiện tại → route sang advisor_retrieve
    - User cập nhật 1 số field → merge updates → route sang advisor_retrieve
    - User muốn hỏi lại từ đầu → reset → route sang field_extractor
    """
    domain = state.get("advisor_domain") or "general"
    user_id = state.get("user_id") or "UID0000"
    current_profile: dict = dict(state.get("collected_info") or {})
    required_fields: dict = state.get("required_fields") or DOMAIN_FIELDS.get(domain, {})

    # Lấy câu trả lời mới nhất của user
    messages = state.get("messages", [])
    user_response = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_response = m.content.strip()
            break

    if not user_response:
        # Không có response → giữ nguyên profile, tiến tới retrieval
        logger.warning("Profile update: no user response found, proceeding with saved profile.")
        return {
            "collected_info": current_profile,
            "missing_fields": [],
            "awaiting_profile_confirm": False,
        }

    # ── Dùng LLM để trích xuất updates ──────────────────────────────────────
    valid_fields = list(required_fields.keys())
    current_profile_text = "\n".join(
        f"- {_FIELD_LABELS.get(k, k)}: {v}"
        for k, v in current_profile.items()
    )

    prompt = UPDATE_EXTRACT_PROMPT.format(
        current_profile=current_profile_text,
        user_response=user_response,
        valid_fields=", ".join(valid_fields),
    )

    confirmed = True
    updates: dict = {}
    try:
        llm = get_llm(temperature=0.0, num_ctx=4096)
        response = llm.invoke(prompt)
        result = parse_json(response.content)
        confirmed = result.get("confirmed", True)
        raw_updates = result.get("updates", {})
        # Chỉ lấy updates hợp lệ (key thuộc required_fields, value non-empty)
        updates = {
            k: str(v).strip()
            for k, v in raw_updates.items()
            if k in required_fields and v and str(v).strip() not in ("null", "", "None")
        }
        logger.info(
            f"Profile update: confirmed={confirmed}, updates={updates} | "
            f"user_response='{user_response[:80]}'"
        )
    except Exception as e:
        logger.warning(f"Profile update extraction failed: {e} — assuming confirmed, no updates")
        confirmed = True

    # ── Nếu user muốn hỏi lại từ đầu ────────────────────────────────────────
    if not confirmed:
        logger.info("Profile update: user wants to restart → resetting profile.")
        return {
            "collected_info": {},
            "missing_fields": list(required_fields.keys()),
            "required_fields": required_fields,
            "awaiting_profile_confirm": False,
            "turn_count": 0,
        }

    # ── Merge updates vào profile ─────────────────────────────────────────────
    if updates:
        current_profile.update(updates)
        # Cập nhật lại file lưu trữ
        profile_store = get_profile_store()
        profile_store.update_profile(user_id, domain, updates)
        logger.info(f"Profile merged with updates: {list(updates.keys())}")

    # ── Kiểm tra các field còn thiếu sau update ───────────────────────────────
    missing = [f for f in required_fields if f not in current_profile]

    # Tạo transition message
    if updates:
        updated_labels = ", ".join(_FIELD_LABELS.get(k, k) for k in updates)
        ack = (
            f"Mình đã cập nhật thông tin của bạn ({updated_labels}). "
            "Để mình tìm sản phẩm phù hợp nhất cho bạn ngay nhé! ⏳"
        )
    else:
        ack = "Thông tin vẫn vậy, để mình tư vấn ngay cho bạn! ⏳"

    return {
        "collected_info": current_profile,
        "missing_fields": missing,
        "required_fields": required_fields,
        "awaiting_profile_confirm": False,
        "turn_count": 1,
        "messages": [AIMessage(content=ack)],
    }


def route_after_profile_recall(state: ChatState) -> str:
    """Sau recall: nếu đã có profile → END (chờ user confirm), không → field_extractor."""
    if state.get("awaiting_profile_confirm"):
        return "__end__"
    return "advisor_field_extractor"


def route_after_profile_update(state: ChatState) -> str:
    """
    Sau update:
    - Còn missing fields (user reset) → advisor_collect_info
    - Đủ thông tin → advisor_retrieve
    """
    missing = state.get("missing_fields") or []
    if missing:
        return "advisor_collect_info"
    return "advisor_retrieve"
