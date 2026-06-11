"""
Cache Check Node — chỉ áp dụng cho intent KHÔNG phải PRODUCT_CONSULT.

Tìm câu hỏi tương tự trong lịch sử hội thoại của tất cả users.
Nếu similarity >= threshold → trả về cached answer ngay, không cần xử lý tiếp.
Nếu không → pass through để xử lý bình thường.

Cache áp dụng cho: GREETING_FAREWELL | PERSONAL_UNRELATED | PRODUCT_INFO_QA
KHÔNG áp dụng cho: PRODUCT_CONSULT | CUSTOMER_FEEDBACK
  - PRODUCT_CONSULT: tư vấn phụ thuộc profile KH cụ thể
  - CUSTOMER_FEEDBACK: phản hồi cá nhân, cần xử lý theo ngữ cảnh
"""
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.history.conversation_store import get_store
from config.settings import settings


def cache_check_node(state: ChatState) -> dict:
    """
    Kiểm tra cache trước khi xử lý (chỉ cho non-advisor intents).
    Node này chỉ được gọi sau khi intent đã được phân loại là
    GREETING_FAREWELL | PERSONAL_UNRELATED | PRODUCT_INFO_QA.
    """
    messages = state.get("messages", [])
    user_question = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_question = m.content
            break

    if not user_question:
        return {"cache_hit": False}

    store = get_store()
    cached = store.search_similar(user_question)

    if cached:
        logger.info(
            f"[CACHE HIT in graph] sim={cached['similarity']:.3f} | "
            f"matched: '{cached['matched_question'][:60]}'"
        )
        return {
            "messages": [AIMessage(content=cached["answer"])],
            "cache_hit": True,
            "cache_similarity": cached["similarity"],
            # Giữ intent để response có thể trả về đúng thông tin
            "intent": cached.get("intent") or state.get("intent"),
            "advisor_domain": cached.get("advisor_domain") or None,
        }

    logger.debug(f"[CACHE MISS in graph] proceeding to normal processing.")
    return {"cache_hit": False, "cache_similarity": None}


def route_after_cache_check(state: ChatState) -> str:
    """
    Nếu cache hit → END (answer đã có trong messages).
    Nếu cache miss → route tiếp theo dựa vào intent.
    """
    if state.get("cache_hit"):
        return "__end__"

    intent = state.get("intent", "PRODUCT_INFO_QA")
    route_map = {
        "GREETING_FAREWELL":  "greeting",
        "PERSONAL_UNRELATED": "personal_unrelated",
        "PRODUCT_INFO_QA":    "rag_rewrite",
        "CUSTOMER_FEEDBACK":  "customer_feedback",
    }
    dest = route_map.get(intent, "rag_rewrite")
    logger.debug(f"Cache miss → {dest}")
    return dest
