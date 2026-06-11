"""
Advisor Nodes 4 & 5: Filtered retrieval + LLM recommendation.
"""
import re
from langchain_core.messages import AIMessage
from loguru import logger

from src.graph.state import ChatState
from src.retrieval.retriever import hybrid_retrieve, format_context
from src.knowledge_graph.field_definitions import get_domain_keyword, DOMAIN_LABELS
from src.llm import get_llm
from config.settings import settings

RECOMMEND_PROMPT = """Ban la chuyen gia tu van san pham ngan hang VIB.

THONG TIN KHACH HANG:
{customer_profile}

TAI LIEU SAN PHAM VIB:
{context}

QUY TAC: Chi de xuat san pham co TEN CU THE trong tai lieu.
Giai thich tai sao san pham phu hop voi thong tin khach hang.
TUYET DOI KHONG bia dat ten san pham, lai suat, quyen loi khong co trong tai lieu.

Tra loi bang tieng Viet than thien, co cau truc ro rang."""

NOT_FOUND_TEMPLATE = (
    "Xin loi, minh chua tim thay thong tin ve san pham **{domain_label}** "
    "phu hop voi nhu cau cua ban trong he thong tai lieu hien tai.\n\n"
    "De duoc tu van chinh xac, ban vui long lien he:\n"
    "- Hotline: **1800 8180** (mien phi, 24/7)\n"
    "- Website: **vib.com.vn**\n"
    "- Hoac den chi nhanh VIB gan nhat de gap chuyen vien tu van truc tiep nhe!"
)


def _build_customer_profile(domain: str, collected_info: dict) -> str:
    domain_label = DOMAIN_LABELS.get(domain, domain)
    lines = ["Nhu cau: Tu van " + domain_label]
    field_labels = {
        "thu_nhap_hang_thang":  "Thu nhap hang thang",
        "muc_chi_tieu_chu_yeu": "Chi tieu chu yeu",
        "uu_tien_quyen_loi":    "Quyen loi uu tien",
        "co_the_hien_tai":      "The hien tai",
        "tuoi":                 "Tuoi",
        "tinh_trang_gia_dinh":  "Tinh trang gia dinh",
        "muc_dich_bao_hiem":    "Muc dich bao hiem",
        "ngan_sach_hang_thang": "Ngan sach hang thang",
        "muc_dich_vay":         "Muc dich vay",
        "so_tien_can_vay":      "So tien can vay",
        "tai_san_the_chap":     "Tai san the chap",
        "so_tien_gui":          "So tien gui",
        "thoi_han_gui":         "Thoi han gui",
        "muc_tieu":             "Muc tieu",
        "loai_san_pham":        "San pham quan tam",
    }
    for field, value in collected_info.items():
        label = field_labels.get(field, field)
        lines.append("- " + label + ": " + str(value))
    return "\n".join(lines)


def _docs_are_relevant(domain: str, docs: list) -> bool:
    if not docs:
        return False
    domain_kw = get_domain_keyword(domain).lower().split()
    for doc in docs:
        content = doc.get("content", "").lower()
        if any(kw in content for kw in domain_kw):
            return True
    return False


def advisor_retrieve_node(state: ChatState) -> dict:
    domain = state.get("advisor_domain") or "general"
    collected_info = state.get("collected_info") or {}

    domain_kw = get_domain_keyword(domain)
    profile_text = " ".join(str(v) for v in collected_info.values())
    query = (domain_kw + " " + profile_text).strip()

    docs = hybrid_retrieve(
        query=query,
        top_k=settings.top_k_final,
        domain_hint=domain_kw,
        category_filter=domain,
    )
    logger.info("Advisor retrieved " + str(len(docs)) + " docs for domain=" + domain)
    return {"retrieved_docs": docs}


def advisor_recommend_node(state: ChatState) -> dict:
    docs = state.get("retrieved_docs") or []
    domain = state.get("advisor_domain") or "general"
    collected_info = state.get("collected_info") or {}
    domain_label = DOMAIN_LABELS.get(domain, domain)
    user_id = state.get("user_id") or "UID0000"

    if not docs or not _docs_are_relevant(domain, docs):
        logger.warning("No relevant docs for domain=" + domain)
        answer = NOT_FOUND_TEMPLATE.format(domain_label=domain_label)
        return {"messages": [AIMessage(content=answer)]}

    customer_profile = _build_customer_profile(domain, collected_info)
    context = format_context(docs)

    prompt = RECOMMEND_PROMPT.format(
        customer_profile=customer_profile,
        context=context,
        domain_label=domain_label,
    )

    try:
        llm = get_llm(temperature=0.1)
        response = llm.invoke(prompt)
        # Strip <think>...</think> blocks from DeepSeek-R1 output
        answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        if not answer:
            answer = NOT_FOUND_TEMPLATE.format(domain_label=domain_label)
    except Exception as e:
        logger.error("Recommender error: " + str(e))
        answer = NOT_FOUND_TEMPLATE.format(domain_label=domain_label)

    if collected_info:
        try:
            from src.history.advisor_profile_store import get_profile_store
            store = get_profile_store()
            store.save_profile(user_id, domain, collected_info)
        except Exception as e:
            logger.warning("Failed to save advisor profile: " + str(e))

    logger.info("Recommendation generated (" + str(len(answer)) + " chars)")
    return {"messages": [AIMessage(content=answer)]}
