"""RAG Node 1: Query rewrite — clarify and enrich the user's question."""
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.llm import get_llm, parse_json

REWRITE_PROMPT = """Bạn đang hỗ trợ tìm kiếm thông tin trong tài liệu sản phẩm ngân hàng VIB.

Lịch sử hội thoại:
{history}

Câu hỏi gốc của khách hàng: "{question}"

Viết lại câu hỏi để tối ưu cho tìm kiếm ngữ nghĩa trong vectorstore:
- Làm rõ ý hỏi, bổ sung context từ lịch sử nếu cần
- Thêm từ khóa chuyên ngành ngân hàng VIB liên quan
- Giữ nguyên ngôn ngữ tiếng Việt

Gợi ý từ khóa theo danh mục (bổ sung nếu phù hợp với câu hỏi):
- Thẻ tín dụng: lãi suất, hạn mức tín dụng, phí thường niên, hoàn tiền cashback, dặm bay, điều kiện mở thẻ, hồ sơ, miễn lãi, dư nợ, thẻ phụ
- Vay: lãi suất vay, thời hạn vay, hồ sơ vay, tài sản thế chấp, tài sản đảm bảo, tín chấp, điều kiện vay, giải ngân, trả góp
- Tiết kiệm: sổ tiết kiệm, kỳ hạn, lãi suất tiết kiệm, gửi tiết kiệm, tiền gửi có kỳ hạn, rút trước hạn
- Bảo hiểm: quyền lợi bảo hiểm chính, tử vong thương tật toàn bộ vĩnh viễn, tích lũy tài khoản, bảo hiểm nhân thọ liên kết đơn vị, quy tắc điều khoản bảo hiểm, phí bảo hiểm, hợp đồng bảo hiểm
- Thông tin chung: sản phẩm VIB, dịch vụ ngân hàng, tư vấn khách hàng

Lưu ý: Nếu câu hỏi hỏi số liệu cụ thể (ví dụ: lãi suất tiết kiệm kỳ hạn X tháng), hãy viết lại để tìm kiếm cả thông tin chung về kỳ hạn và lãi suất của loại sản phẩm đó.

Trả về JSON: {{"rewritten": "câu hỏi đã viết lại"}}"""


def _get_recent_history(messages: list, n: int = 4) -> str:
    recent = messages[-n * 2 :] if messages else []
    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"KH: {m.content[:800]}")
        elif isinstance(m, AIMessage):
            lines.append(f"Bot: {m.content[:800]}")
    return "\n".join(lines) if lines else "(không có)"


def rag_rewrite_node(state: ChatState) -> dict:
    messages = state.get("messages", [])
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    history = _get_recent_history(messages[:-1])
    prompt = REWRITE_PROMPT.format(history=history, question=last_user_msg)

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(prompt)
        result = parse_json(response.content)
        rewritten = result.get("rewritten", last_user_msg)
    except Exception as e:
        logger.error(f"Query rewrite error: {e}")
        rewritten = last_user_msg

    logger.info(f"Query rewrite: '{last_user_msg[:150]}' → '{rewritten[:200]}'")
    return {"rewritten_query": rewritten}
