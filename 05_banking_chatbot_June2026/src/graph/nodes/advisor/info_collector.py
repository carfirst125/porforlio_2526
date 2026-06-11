"""
Advisor Node 3: Multi-turn info collection.

turn_count=0: First visit from field_extractor.
  - collected_info already pre-filled by field_extractor
  - If missing_fields is empty → all info extracted from initial question → proceed to retrieval
  - If missing_fields non-empty → ask first missing field with intro message

turn_count>0: Subsequent visits from user answering questions.
  - Store last user message as answer to first missing field
  - Ask next missing field, or transition to retrieval if done
"""
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.knowledge_graph.field_definitions import DOMAIN_LABELS
from config.settings import settings

INTRO_TEMPLATES = {
    "credit_card": "Để tư vấn thẻ tín dụng phù hợp nhất, mình cần hỏi thêm một vài thông tin nhé! 😊",
    "insurance":   "Để tư vấn gói bảo hiểm phù hợp, mình cần tìm hiểu thêm về nhu cầu của bạn nhé!",
    "loan":        "Để tư vấn gói vay phù hợp nhất, mình cần hỏi thêm một số thông tin nhé!",
    "savings":     "Để tư vấn sản phẩm tiết kiệm tối ưu, mình cần hỏi thêm một vài thông tin nhé!",
    "general":     "Để tư vấn sản phẩm phù hợp nhất, mình cần hỏi thêm một vài thông tin nhé!",
}

TRANSITION_MSG = (
    "Cảm ơn bạn đã cung cấp thông tin! "
    "Mình sẽ tìm kiếm sản phẩm phù hợp nhất cho bạn ngay nhé... ⏳"
)
MAX_TURNS_MSG = (
    "Cảm ơn bạn! Dựa trên thông tin bạn đã cung cấp, "
    "mình sẽ tư vấn sản phẩm phù hợp nhất cho bạn ngay nhé..."
)
ALL_PRE_FILLED_MSG = (
    "Mình đã hiểu nhu cầu của bạn rồi! "
    "Để mình tìm kiếm sản phẩm phù hợp nhất cho bạn ngay nhé... ⏳"
)


def advisor_collect_info_node(state: ChatState) -> dict:
    messages      = state.get("messages", [])
    collected_info = dict(state.get("collected_info") or {})
    required_fields: dict = state.get("required_fields") or {}
    turn_count    = state.get("turn_count", 0)
    max_turns     = state.get("max_turns", settings.max_advisor_turns)
    domain        = state.get("advisor_domain") or "general"

    # Current missing fields
    missing = [f for f in required_fields if f not in collected_info]

    # ── turn_count == 0: First visit from field_extractor ───────────────────
    if turn_count == 0:
        if not missing:
            # All info pre-extracted from initial question → proceed immediately
            logger.info("All fields pre-filled from initial question, proceeding to retrieval.")
            return {
                "collected_info": collected_info,
                "missing_fields": [],
                "turn_count": 1,
                "messages": [AIMessage(content=ALL_PRE_FILLED_MSG)],
            }
        else:
            # Ask first missing field with intro
            intro = INTRO_TEMPLATES.get(domain, INTRO_TEMPLATES["general"])
            first_q = required_fields[missing[0]]
            opening = f"{intro}\n\n{first_q}"
            logger.info(f"First question (turn=0): {missing[0]}")
            return {
                "collected_info": collected_info,
                "missing_fields": missing,
                "turn_count": 1,
                "messages": [AIMessage(content=opening)],
            }

    # ── turn_count > 0: Subsequent visits — user answered previous question ──
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content.strip()
            break

    if missing and last_user_msg:
        field_to_store = missing[0]
        collected_info[field_to_store] = last_user_msg
        missing = [f for f in required_fields if f not in collected_info]
        logger.info(f"Stored [{field_to_store}] = '{last_user_msg[:60]}'")

    turn_count += 1

    if missing and turn_count <= max_turns:
        # Ask next missing field
        next_q = required_fields[missing[0]]
        logger.info(f"Asking field: {missing[0]} (turn={turn_count})")
        return {
            "collected_info": collected_info,
            "missing_fields": missing,
            "turn_count": turn_count,
            "messages": [AIMessage(content=next_q)],
        }
    else:
        # All collected or max turns reached
        transition = TRANSITION_MSG if not missing else MAX_TURNS_MSG
        if missing:
            logger.info(f"Max turns reached with partial info: {list(collected_info.keys())}")
        return {
            "collected_info": collected_info,
            "missing_fields": [],
            "turn_count": turn_count,
            "messages": [AIMessage(content=transition)],
        }


def route_after_collect_info(state: ChatState) -> str:
    """Proceed to retrieval if all fields collected, else wait for next user turn."""
    missing = state.get("missing_fields") or []
    if missing:
        return "__end__"
    return "advisor_retrieve"
