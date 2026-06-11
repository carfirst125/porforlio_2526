"""LangGraph state definition for the chatbot."""
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    # ── Core ────────────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    session_id: str
    user_id: Optional[str]          # dùng để load/save advisor profile per user

    # ── Intent ──────────────────────────────────────────────────────────────
    # GREETING_FAREWELL | PERSONAL_UNRELATED | PRODUCT_INFO_QA | PRODUCT_CONSULT
    intent: Optional[str]

    # ── RAG fields ──────────────────────────────────────────────────────────
    rewritten_query: Optional[str]
    retrieved_docs: Optional[list]   # list[{content, score_rank}]

    # ── Advisor fields ──────────────────────────────────────────────────────
    # credit_card | insurance | loan | savings | general
    advisor_domain: Optional[str]
    # {field_name: question_to_ask}  — set once by field_extractor
    required_fields: Optional[dict]
    # {field_name: user_answer}  — filled incrementally
    collected_info: Optional[dict]
    # fields not yet collected
    missing_fields: Optional[list]
    # True khi bot đã hiển thị profile cũ và đang chờ user xác nhận/cập nhật
    awaiting_profile_confirm: Optional[bool]

    # ── Cache ────────────────────────────────────────────────────────────────
    # Set bởi cache_check_node nếu tìm thấy câu hỏi tương tự trong lịch sử
    cache_hit: Optional[bool]
    cache_similarity: Optional[float]

    # ── Session management ───────────────────────────────────────────────────
    turn_count: int
    max_turns: int
