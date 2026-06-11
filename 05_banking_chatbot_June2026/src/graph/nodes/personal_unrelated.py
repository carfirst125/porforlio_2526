"""
Personal/Unrelated handler node.
Empathizes with customer's personal share, then naturally redirects
to the most relevant VIB product.
"""
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.llm import get_llm

PERSONAL_PROMPT = """Bạn là nhân viên tư vấn thân thiện, tinh tế của ngân hàng VIB.

Khách hàng vừa chia sẻ: "{message}"

Hãy phản hồi theo 3 bước tự nhiên (KHÔNG liệt kê từng bước, viết thành đoạn văn liền mạch):
1. Đồng cảm ngắn gọn, chân thành với tình huống khách hàng chia sẻ (1 câu).
2. Tự nhiên liên kết sang sản phẩm/dịch vụ của VIB phù hợp NHẤT với tình huống đó.
   - Ví dụ: KH nói "vừa có con đầu lòng" → gợi ý bảo hiểm giáo dục/nhân thọ
   - Ví dụ: KH nói "vừa được tăng lương" → gợi ý thẻ tín dụng hạng cao hoặc gửi tiết kiệm
   - Ví dụ: KH nói "đang tìm mua nhà" → gợi ý gói vay mua nhà
   - Ví dụ: KH nói "đang bận" hoặc không rõ → hỏi chung cần hỗ trợ sản phẩm gì
3. Hỏi khách hàng có muốn tìm hiểu thêm không (1 câu).

Yêu cầu: Tự nhiên, không cứng nhắc, thân thiện, bằng tiếng Việt. Tổng cộng 3-4 câu."""


def personal_unrelated_node(state: ChatState) -> dict:
    messages = state.get("messages", [])
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    try:
        llm = get_llm(temperature=0.4)
        prompt = PERSONAL_PROMPT.format(message=last_user_msg)
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"Personal unrelated LLM error: {e}")
        answer = (
            "Cảm ơn bạn đã chia sẻ! "
            "Nếu bạn cần tư vấn về sản phẩm tài chính phù hợp với tình huống của mình, "
            "mình luôn sẵn sàng hỗ trợ bạn nhé. Bạn có muốn tìm hiểu thêm không ạ?"
        )

    logger.info(f"Personal unrelated → {answer[:60]}")
    return {"messages": [AIMessage(content=answer)]}
