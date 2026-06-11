"""
Customer Feedback Node.

Xử lý phản hồi của khách hàng về:
- Câu trả lời của chatbot (không hiểu, trả lời sai, không đủ thông tin...)
- Sản phẩm/dịch vụ của VIB (hài lòng, không hài lòng, khiếu nại...)

Phân loại sentiment:
- NEGATIVE: KH bực bội, không hài lòng, phàn nàn → trả lời nhỏ nhẹ, lịch sự,
  thể hiện cầu thị, sẽ ghi nhận và cải thiện.
- POSITIVE: KH khen ngợi, hài lòng, cảm ơn (về sản phẩm/câu trả lời) →
  cảm ơn chân thành, khiêm tốn.
- NEUTRAL/UNCLEAR: Phản hồi trung tính, yêu cầu thêm thông tin →
  ghi nhận và hỏi thêm chi tiết.
"""
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.llm import get_llm, parse_json

FEEDBACK_PROMPT = """Bạn là nhân viên chăm sóc khách hàng của ngân hàng VIB, lịch sự và chuyên nghiệp.

LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{history}

PHẢN HỒI MỚI NHẤT CỦA KHÁCH HÀNG:
"{message}"

Nhiệm vụ:
1. Phân tích sentiment của phản hồi: NEGATIVE / POSITIVE / NEUTRAL
2. Xác định đối tượng phản hồi: về chatbot/câu trả lời hay về sản phẩm/dịch vụ VIB
3. Tạo câu trả lời phù hợp theo hướng dẫn bên dưới

HƯỚNG DẪN PHẢN HỒI:

Nếu NEGATIVE (bực bội, không hài lòng, phàn nàn, chê bai):
- Xin lỗi chân thành, không phòng thủ
- Đồng cảm với cảm xúc của khách hàng
- Cam kết ghi nhận và cải thiện
- Nếu là về câu trả lời bot: hứa sẽ cố gắng hỗ trợ tốt hơn, mời KH hỏi lại
- Nếu là về sản phẩm VIB: ghi nhận phản hồi, hướng KH đến hotline/chi nhánh nếu cần giải quyết cụ thể
- Giọng điệu: nhỏ nhẹ, chân thành, cầu thị

Nếu POSITIVE (khen ngợi, hài lòng, cảm ơn):
- Cảm ơn chân thành
- Khiêm tốn, không tự mãn
- Hứa tiếp tục phục vụ tốt
- Hỏi thêm KH cần hỗ trợ gì không
- Giọng điệu: ấm áp, vui vẻ, khiêm tốn

Nếu NEUTRAL (phản hồi trung tính, không rõ ý):
- Ghi nhận phản hồi
- Hỏi thêm chi tiết để hiểu rõ hơn
- Sẵn sàng hỗ trợ thêm

Trả về JSON:
{{
  "sentiment": "NEGATIVE" | "POSITIVE" | "NEUTRAL",
  "response": "Câu trả lời bằng tiếng Việt, thân thiện, 2-4 câu"
}}

Chỉ output JSON, không giải thích thêm."""


def _build_history_str(messages: list, max_turns: int = 4) -> str:
    """Format recent conversation history."""
    recent = messages[-max_turns * 2:] if messages else []
    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"KH: {m.content[:500]}")
        elif isinstance(m, AIMessage):
            lines.append(f"Bot: {m.content[:500]}")
    return "\n".join(lines) if lines else "(Không có lịch sử)"


def customer_feedback_node(state: ChatState) -> dict:
    messages = state.get("messages", [])

    # Lấy tin nhắn cuối của user
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    if not last_user_msg:
        return {"messages": [AIMessage(content="Cảm ơn phản hồi của bạn! Mình luôn sẵn sàng hỗ trợ bạn tốt hơn.")]}

    # Lịch sử hội thoại (bỏ tin nhắn cuối vì đã có ở trên)
    history = _build_history_str(messages[:-1])

    prompt = FEEDBACK_PROMPT.format(history=history, message=last_user_msg)

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        result = parse_json(response.content)

        sentiment = result.get("sentiment", "NEUTRAL")
        answer = result.get("response", "").strip()

        if not answer:
            raise ValueError("Empty response from LLM")

        logger.info(
            f"Customer feedback: sentiment={sentiment} | "
            f"message='{last_user_msg[:80]}'"
        )

    except Exception as e:
        logger.warning(f"Customer feedback node error: {e} — using fallback")
        answer = (
            "Cảm ơn bạn đã chia sẻ! Mình xin ghi nhận phản hồi của bạn và sẽ "
            "cố gắng phục vụ tốt hơn. Bạn cần mình hỗ trợ thêm gì không ạ?"
        )

    return {"messages": [AIMessage(content=answer)]}
