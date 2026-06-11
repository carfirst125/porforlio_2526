"""Greeting/Farewell handler node.

Sub-classifies GREETING_FAREWELL thành 2 loại trước khi gọi LLM:
  - FAREWELL: tạm biệt, bye, hẹn gặp lại... → LLM dùng FAREWELL_PROMPT
  - GREETING : chào hỏi, hi, xin chào...    → LLM dùng GREETING_PROMPT

Pre-classify bằng keyword để tránh LLM luôn generate kiểu chào hỏi.
"""
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.llm import get_llm

# ── Farewell keywords ────────────────────────────────────────────────────────
_FAREWELL_KEYWORDS = [
    "tạm biệt", "bye", "goodbye", "good bye", "chào nhé", "thôi nhé",
    "hẹn gặp lại", "gặp lại sau", "hẹn gặp", "thôi mình đi", "mình đi nhé",
    "bái bai", "bai bai", "ciao", "see you", "tạm biệt bạn",
    "ok bye", "ok, bye", "oke bye", "nhé bye",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
FAREWELL_PROMPT = """Bạn là nhân viên tư vấn thân thiện, chuyên nghiệp của ngân hàng VIB.

Khách hàng vừa nhắn lời tạm biệt: "{message}"

Hãy trả lời lời tạm biệt: cảm ơn khách hàng đã liên hệ với VIB, chúc khách hàng vui vẻ/thuận lợi, và nhắn rằng bất cứ lúc nào cần hỗ trợ thêm thì cứ nhắn tin.

Yêu cầu: Ngắn gọn (2-3 câu), thân thiện, ấm áp, bằng tiếng Việt."""

GREETING_PROMPT = """Bạn là nhân viên tư vấn thân thiện, chuyên nghiệp của ngân hàng VIB.

Khách hàng vừa nhắn: "{message}"

Hãy chào lại lịch sự, giới thiệu ngắn bạn là trợ lý AI của VIB, và hỏi khách hàng cần hỗ trợ gì hôm nay.

Yêu cầu: Ngắn gọn (2-3 câu), thân thiện, bằng tiếng Việt. Không liệt kê dịch vụ dài dòng."""

# ── Static fallbacks (khi LLM lỗi) ──────────────────────────────────────────
STATIC_GREETING = (
    "Xin chào! Mình là trợ lý AI của ngân hàng VIB. "
    "Mình có thể giúp bạn tìm hiểu thông tin sản phẩm hoặc tư vấn sản phẩm phù hợp. "
    "Bạn cần hỗ trợ gì hôm nay ạ? 😊"
)
STATIC_FAREWELL = (
    "Rất vui được phục vụ bạn! "
    "Bất cứ khi nào cần hỗ trợ, bạn cứ nhắn tin cho mình nhé. "
    "Chúc bạn một ngày vui vẻ! 👋"
)


def _is_farewell(text: str) -> bool:
    """Kiểm tra xem tin nhắn có phải lời tạm biệt không bằng keyword matching."""
    lower = text.lower().strip()
    return any(kw in lower for kw in _FAREWELL_KEYWORDS)


def greeting_node(state: ChatState) -> dict:
    messages = state.get("messages", [])
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    farewell = _is_farewell(last_user_msg)
    prompt_template = FAREWELL_PROMPT if farewell else GREETING_PROMPT
    static_fallback = STATIC_FAREWELL if farewell else STATIC_GREETING
    sub_type = "FAREWELL" if farewell else "GREETING"

    try:
        llm = get_llm(temperature=0.3)
        prompt = prompt_template.format(message=last_user_msg)
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"Greeting LLM error: {e}")
        answer = static_fallback

    logger.info(f"Greeting node [{sub_type}] → {answer[:60]}")
    return {"messages": [AIMessage(content=answer)]}
