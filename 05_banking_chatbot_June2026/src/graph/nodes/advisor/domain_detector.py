"""Advisor Node 1: Detect product domain from user message."""
from langchain_core.messages import HumanMessage
from loguru import logger

from src.graph.state import ChatState
from src.llm import get_llm, parse_json

DOMAIN_PROMPT = """Xác định domain sản phẩm ngân hàng từ tin nhắn khách hàng.

Các domain:
- credit_card: thẻ tín dụng, thẻ ghi nợ, thẻ ATM, thẻ ngân hàng
- insurance: bảo hiểm nhân thọ, bảo hiểm sức khỏe, bảo hiểm xe, bảo hiểm tai nạn
- loan: vay mua nhà, vay mua xe, vay tiêu dùng, vay tín chấp, vay thế chấp, vay kinh doanh
- savings: gửi tiết kiệm, tiết kiệm online, đầu tư, tích lũy
- general: không rõ hoặc hỏi chung về nhiều sản phẩm

Tin nhắn: "{message}"

Trả về JSON: {{"domain": "tên_domain"}}"""

VALID_DOMAINS = {"credit_card", "insurance", "loan", "savings", "general"}


def advisor_domain_detector_node(state: ChatState) -> dict:
    messages = state.get("messages", [])
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    try:
        llm = get_llm(temperature=0.0)
        prompt = DOMAIN_PROMPT.format(message=last_user_msg)
        response = llm.invoke(prompt)
        result = parse_json(response.content)
        domain = result.get("domain", "general").lower()
        if domain not in VALID_DOMAINS:
            domain = "general"
    except Exception as e:
        logger.error(f"Domain detector error: {e}")
        domain = "general"

    logger.info(f"Advisory domain: {domain}")
    return {"advisor_domain": domain}
