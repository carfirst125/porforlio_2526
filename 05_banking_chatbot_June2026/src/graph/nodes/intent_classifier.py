"""
Intent Classifier Node.
Classifies user message into one of 5 intents.

v3.3: Dynamic few-shot examples — chọn ví dụ phù hợp nhất với câu hỏi hiện tại
từ example library, thay vì hardcode cố định trong prompt.
"""
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.llm import get_llm, parse_json

VALID_INTENTS = {
    "GREETING_FAREWELL",
    "PERSONAL_UNRELATED",
    "PRODUCT_INFO_QA",
    "PRODUCT_CONSULT",
    "CUSTOMER_FEEDBACK",
}

# Path tới example library
_INTENT_EXAMPLES_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "examples" / "intent_examples.json"
)

# Lazy cache
_intent_examples: list = []


def _get_intent_examples() -> list:
    global _intent_examples
    if not _intent_examples:
        try:
            from src.examples.example_selector import load_examples
            _intent_examples = load_examples(_INTENT_EXAMPLES_PATH)
        except Exception as e:
            logger.warning(f"Cannot load intent examples: {e}")
            _intent_examples = []
    return _intent_examples


# Base prompt (không thay đổi) — ví dụ few-shot được inject động
_BASE_PROMPT = (
    "Ban la he thong phan loai intent cho chatbot ngan hang VIB.\n\n"
    "Phan loai tin nhan cua khach hang vao DUNG MOT trong 5 loai sau:\n\n"

    "- GREETING_FAREWELL: Chao hoi, tam biet, cam on chung. "
    "Ke ca khi KH noi muon hoi hay can tu van nhung CHUA neu ro van de cu the.\n"
    "  Dan hieu: xin chao, hi ban, tam biet, cam on chung chung, chao minh can tu van.\n"
    "  Luu y: 'cam on bot da giai dap, rat huu ich' la CUSTOMER_FEEDBACK (co danh gia), "
    "  khong phai GREETING_FAREWELL.\n\n"

    "- PERSONAL_UNRELATED: KH chia se thong tin ca nhan hoac tinh hinh tai chinh "
    "ca nhan ma KHONG kem cau hoi hay yeu cau tu van san pham cu the.\n"
    "  Dan hieu: vua duoc tang luong, vua ban dat, nam nay thu nhap tang, "
    "dang can tien gap, moi ra truong di lam 6 thang.\n"
    "  Luu y: PERSONAL_UNRELATED du de cap tien/tai chinh -- "
    "mien la KHONG hoi ve san pham ngan hang cu the.\n"
    "  Phan biet: 'Toi co 500 trieu' -> PERSONAL_UNRELATED; "
    "'Toi co 500 trieu muon gui tiet kiem, nen chon ky han nao?' -> PRODUCT_CONSULT.\n\n"

    "- PRODUCT_INFO_QA: Hoi thong tin cu the tra cuu duoc ve san pham/dich vu ngan hang. "
    "Bao gom: lai suat, phi, dieu kien, quy trinh, co nhung loai nao, tai san the chap, quyen loi.\n"
    "  Dan hieu: lai suat the VIB Super Card la bao nhieu, phi thuong nien the Cash Back, "
    "dieu kien mo the VIB Premier Boundless la gi, quy trinh vay mua nha VIB, "
    "VIB co nhung loai the nao, tai san the chap de vay gom loai nao.\n"
    "  Luu y: cau hoi co 'quy trinh', 'dieu kien', 'phi', 'lai suat', 'ho so', "
    "'co nhung loai nao' la PRODUCT_INFO_QA du co nhac san pham cu the hay khong.\n\n"

    "- PRODUCT_CONSULT: Muon duoc tu van CHON san pham phu hop voi ban than, "
    "chua biet nen chon loai nao, can ai giup quyet dinh.\n"
    "  Dan hieu: khong biet chon the nao, nen dung loai nao, tu van cho toi, "
    "loai nao tot hon, loai nao phu hop voi toi, nen chon goi nao.\n"
    "  Phan biet voi PRODUCT_INFO_QA: 'vay mua nha quy trinh nhu the nao' -> PRODUCT_INFO_QA; "
    "'toi muon vay mua nha nen chon goi nao' -> PRODUCT_CONSULT.\n\n"

    "- CUSTOMER_FEEDBACK: Phan hoi, danh gia, gop y, than phien, khen ngoi "
    "ve chatbot HOAC ve san pham/dich vu VIB.\n"
    "  Dan hieu:\n"
    "    (1) Khen/che chatbot: 'bot tra loi khong ro', 'cau tra loi rat hay', "
    "'bot cu hoi lai thong tin toi da cung cap roi'\n"
    "    (2) Feature request cho bot: 'bot co the them tinh nang...', "
    "'chatbot nen co chuc nang...', 'nen co them vi du minh hoa'\n"
    "    (3) So sanh chu quan VIB vs doi thu: 'phi VIB cao hon ngan hang khac', "
    "'tai sao lai suat VIB cao vay so voi noi khac', 'VIB nen canh tranh hon'\n"
    "    (4) Nhan xet trai nghiem: 'toi vua dung thu chatbot lan dau', "
    "'giao dien chat don gian de dung'\n"
    "  Luu y phan biet:\n"
    "    'Tai sao phi VIB cao vay?' (phan nan chu quan) -> CUSTOMER_FEEDBACK\n"
    "    'Phi thuong nien VIB la bao nhieu?' (hoi so lieu) -> PRODUCT_INFO_QA\n"
    "    'Bot co the so sanh san pham khong?' (feature request) -> CUSTOMER_FEEDBACK\n"
    "    'So sanh the Cash Back va Online Plus' (muon bot tu van chon) -> PRODUCT_CONSULT\n\n"

    "Quy tac uu tien khi kho phan biet:\n"
    "1. Co nhan xet/so sanh chu quan ve chatbot hoac VIB vs doi thu "
    "-> CUSTOMER_FEEDBACK (uu tien truoc PRODUCT_INFO_QA hoac PRODUCT_CONSULT)\n"
    "2. Co feature request cho chatbot (them tinh nang, nen co...) "
    "-> CUSTOMER_FEEDBACK (khong phai PRODUCT_CONSULT)\n"
    "3. Chia se tai chinh ca nhan khong kem cau hoi san pham "
    "-> PERSONAL_UNRELATED (khong phai PRODUCT_CONSULT)\n"
    "4. Chao hoi kem can tu van nhung chua neu ro van de "
    "-> GREETING_FAREWELL (khong phai PRODUCT_CONSULT)\n"
    "5. Hoi 'nen chon loai nao', 'loai nao tot hon', 'phu hop voi toi' "
    "-> PRODUCT_CONSULT (khong phai PRODUCT_INFO_QA)\n"
)


def _build_classify_prompt(message: str, history: str, few_shot_examples: list) -> str:
    """Xây dựng full prompt với dynamic few-shot examples."""
    lines = [_BASE_PROMPT]

    if few_shot_examples:
        lines.append("Vi du tham khao (few-shot):\n")
        for ex in few_shot_examples:
            intent = ex.get("intent", "")
            msg = ex.get("message", "")
            note = ex.get("note", "")
            lines.append(f'  Tin nhan: "{msg}"')
            lines.append(f'  -> {intent}  [{note}]')
            lines.append("")

    lines += [
        f"Lich su hoi thoai gan day:\n{history}\n",
        f'Tin nhan moi cua khach hang: "{message}"\n',
        'Tra ve JSON duy nhat khong giai thich:',
        '{"intent": "TEN_INTENT"}',
    ]
    return "\n".join(lines)


def _build_history_str(messages: list, max_turns: int = 4) -> str:
    """Format recent messages as conversation history string."""
    recent = messages[-max_turns * 2:] if messages else []
    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append("KH: " + m.content[:800])
        elif isinstance(m, AIMessage):
            lines.append("Bot: " + m.content[:800])
    return "\n".join(lines) if lines else "(Day la tin nhan dau tien)"


def intent_classifier_node(state: ChatState) -> dict:
    messages = state.get("messages", [])

    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    if not last_user_msg:
        return {"intent": "GREETING_FAREWELL"}

    history = _build_history_str(messages[:-1])

    # Dynamic few-shot: chọn 6 ví dụ phù hợp nhất (≥1 per intent class)
    all_examples = _get_intent_examples()
    few_shot: list = []
    if all_examples:
        try:
            from src.examples.example_selector import select_intent_examples
            few_shot = select_intent_examples(
                message=last_user_msg,
                examples=all_examples,
                n=6,
                min_per_class=1,
            )
        except Exception as e:
            logger.warning(f"Few-shot selection failed: {e}")

    prompt = _build_classify_prompt(last_user_msg, history, few_shot)

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(prompt)
        result = parse_json(response.content)
        intent = result.get("intent", "PRODUCT_INFO_QA").upper()
        if intent not in VALID_INTENTS:
            intent = "PRODUCT_INFO_QA"
    except Exception as e:
        logger.error("Intent classifier error: " + str(e) + " -- defaulting to PRODUCT_INFO_QA")
        intent = "PRODUCT_INFO_QA"

    logger.info("Intent classified: " + intent + " | message: " + last_user_msg[:200])
    return {"intent": intent}
