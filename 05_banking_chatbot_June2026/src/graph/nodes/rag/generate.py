"""RAG Node 3: Answer generation from retrieved documents — strict grounding."""
import re
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.llm import get_llm
from src.retrieval.retriever import format_context

GENERATE_PROMPT = """Bạn là chuyên gia tư vấn sản phẩm của ngân hàng VIB.
Nhiệm vụ: Trả lời câu hỏi của khách hàng DỰA HOÀN TOÀN vào nội dung tài liệu được cung cấp dưới đây.

══ QUY TẮC BẮT BUỘC ══
✅ CHỈ sử dụng thông tin CÓ TRONG tài liệu dưới đây — trích dẫn trực tiếp, không diễn giải thêm.
✅ Số liệu (lãi suất, phí, kỳ hạn, hạn mức...) phải lấy NGUYÊN VĂN từ tài liệu, KHÔNG tự tính toán hay quy đổi.
✅ Danh sách sản phẩm/tính năng chỉ liệt kê những gì tài liệu đề cập, không thêm bớt.
✅ Trả lời có cấu trúc rõ ràng, dễ đọc, bằng tiếng Việt thân thiện.
❌ TUYỆT ĐỐI KHÔNG bịa đặt hoặc suy luận ra số liệu, lãi suất, điều kiện, tên sản phẩm không có trong tài liệu.
❌ TUYỆT ĐỐI KHÔNG dùng kiến thức nền (training data) để bổ sung thông tin ngoài tài liệu.
❌ TUYỆT ĐỐI KHÔNG quy đổi/tính toán ra con số mới từ các con số trong tài liệu (ví dụ: không tự đổi %/tháng sang %/năm nếu tài liệu không ghi cả hai).

══ XỬ LÝ KHI KHÔNG TÌM THẤY THÔNG TIN ══
Nếu tài liệu không chứa thông tin trực tiếp trả lời câu hỏi → phải trả lời ĐÚNG theo mẫu sau:
"Xin lỗi, mình chưa tìm thấy thông tin về [chủ đề câu hỏi] trong hệ thống tài liệu hiện tại.
Để được tư vấn chính xác, bạn vui lòng liên hệ:
- Hotline: **1800 8180** (miễn phí, 24/7)
- Website: **vib.com.vn**
- Hoặc đến chi nhánh VIB gần nhất để được hỗ trợ trực tiếp nhé!"

══ TÀI LIỆU THAM KHẢO ══
{context}

══ CÂU HỎI CỦA KHÁCH HÀNG ══
{question}

TRẢ LỜI (chỉ dùng thông tin từ tài liệu trên):"""

NOT_FOUND_RESPONSE = (
    "Xin lỗi, mình chưa tìm thấy thông tin phù hợp trong hệ thống tài liệu hiện tại.\n\n"
    "Để được tư vấn chính xác, bạn vui lòng liên hệ:\n"
    "- Hotline: **1800 8180** (miễn phí, 24/7)\n"
    "- Website: **vib.com.vn**\n"
    "- Hoặc đến chi nhánh VIB gần nhất để được hỗ trợ trực tiếp nhé!"
)


def rag_generate_node(state: ChatState) -> dict:
    docs = state.get("retrieved_docs") or []
    query = state.get("rewritten_query") or ""

    if not query:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                query = m.content
                break

    if not docs:
        logger.warning("No docs retrieved — returning not-found response.")
        return {
            "messages": [AIMessage(content=NOT_FOUND_RESPONSE)],
            "retrieved_docs": [],
        }

    context = format_context(docs)
    prompt = GENERATE_PROMPT.format(context=context, question=query)

    try:
        llm = get_llm(temperature=0.05)
        response = llm.invoke(prompt)
        # Strip <think>...</think> blocks from DeepSeek-R1 before returning
        answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        if not answer:
            answer = NOT_FOUND_RESPONSE
    except Exception as e:
        logger.error(f"RAG generate error: {e}")
        answer = NOT_FOUND_RESPONSE

    logger.info(f"RAG answer generated ({len(answer)} chars)")
    return {
        "messages": [AIMessage(content=answer)],
        "retrieved_docs": docs,
    }
