"""
Advisor Node 2: Extract required fields from KG + pre-fill from initial user question.

Flow:
  1. Get required fields for domain from KG
  2. Use LLM to extract any info already provided in the initial question
  3. Pre-fill collected_info with extracted values
  4. Set missing_fields = only what's NOT yet provided
  5. Set turn_count=0 (signal to info_collector this is first visit)
  -> Always routes to advisor_collect_info (NOT directly to END)
"""
from langchain_core.messages import HumanMessage
from loguru import logger

from src.graph.state import ChatState
from src.knowledge_graph.field_definitions import get_fields, DOMAIN_LABELS
from src.llm import get_llm, parse_json

EXTRACT_PROMPT = """Nhiem vu: Doc cau hoi cua khach hang va dien thong tin vao JSON ben duoi.

Cau hoi: "{question}"

Dien gia tri vao JSON -- moi key co goi y y nghia trong comment:
{json_with_hints}

Quy tac:
- Dien chuoi neu thong tin CO TRONG cau hoi (ke ca ngam hieu ro rang)
- De null neu cau hoi KHONG DE CAP hoac khong chac chan
- "thu nhap 20 trieu" -> thu_nhap_hang_thang: "20 trieu"
- "vay mua nha" -> muc_dich_vay: "mua nha"
- "vay mua xe" -> muc_dich_vay: "mua xe"

Chi output JSON, khong giai thich them."""

FIELD_HINTS = {
    "tuoi":                 "tuoi cua khach hang (nguoi dang hoi)",
    "tinh_trang_gia_dinh":  "co gia dinh/con chua",
    "muc_dich_bao_hiem":    "muc dich mua bao hiem",
    "ngan_sach_hang_thang": "ngan sach/so tien danh hang thang",
    "thu_nhap_hang_thang":  "thu nhap hang thang",
    "muc_chi_tieu_chu_yeu": "chi tieu chu yeu vao linh vuc nao",
    "uu_tien_quyen_loi":    "quyen loi uu tien (cashback, dam bay...)",
    "co_the_hien_tai":      "da co the tin dung chua",
    "muc_dich_vay":         "vay de lam gi",
    "so_tien_can_vay":      "so tien can vay",
    "tai_san_the_chap":     "tai san the chap",
    "so_tien_gui":          "so tien gui tiet kiem",
    "thoi_han_gui":         "thoi han gui",
    "muc_tieu":             "muc tieu gui tiet kiem",
    "loai_san_pham":        "loai san pham quan tam",
}


def _build_extract_prompt(question: str, fields: dict) -> str:
    lines = ["{"]
    for i, k in enumerate(fields):
        hint = FIELD_HINTS.get(k, fields[k][:50])
        comma = "," if i < len(fields) - 1 else ""
        lines.append('  "' + k + '": null' + comma + '  // ' + hint)
    lines.append("}")
    json_with_hints = "\n".join(lines)
    return EXTRACT_PROMPT.format(question=question, json_with_hints=json_with_hints)


def _extract_info_from_question(question: str, fields: dict) -> dict:
    if not question or not fields:
        return {}
    try:
        prompt = _build_extract_prompt(question, fields)
        llm = get_llm(temperature=0.0)
        response = llm.invoke(prompt)

        raw = response.content
        logger.debug("Field extraction raw (last 300): ..." + raw[-300:])

        result = parse_json(raw)
        logger.debug("Parsed extraction: " + str(result))

        extracted = {
            k: str(v).strip()
            for k, v in result.items()
            if v is not None and str(v).strip() not in ("null", "", "None") and k in fields
        }
        logger.info(
            "Pre-extracted: " + str(extracted) +
            " (missing: " + str([f for f in fields if f not in extracted]) + ")"
        )
        return extracted
    except Exception as e:
        logger.warning("Info extraction failed: " + str(e))
        return {}


def advisor_field_extractor_node(state: ChatState) -> dict:
    domain = state.get("advisor_domain") or "general"
    required_fields = get_fields(domain)

    messages = state.get("messages", [])
    initial_question = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            initial_question = m.content
            break

    pre_filled = _extract_info_from_question(initial_question, required_fields)
    missing_fields = [f for f in required_fields if f not in pre_filled]

    logger.info(
        "Field extractor: domain=" + domain +
        ", pre_filled=" + str(list(pre_filled.keys())) +
        ", missing=" + str(missing_fields)
    )

    return {
        "required_fields": required_fields,
        "collected_info": pre_filled,
        "missing_fields": missing_fields,
        "turn_count": 0,
        "max_turns": 8,
    }
