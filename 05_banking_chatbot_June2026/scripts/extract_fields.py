"""
extract_fields.py — Auto-generate field_definitions_draft.py from parquet documents.

Workflow:
  1. Read parquet (same file used by the chatbot)
  2. Detect category for each chunk (reuses loader.detect_category)
  3. Sample representative chunks per category
  4. For each category: call LLM → "what customer info is needed to recommend
     the right product in this category?"
  5. Write field_definitions_draft.py — same structure as field_definitions.py,
     ready for developer review before replacing the production file.

Usage (run from version_3/ directory):
  python scripts/extract_fields.py
  python scripts/extract_fields.py --parquet ../documents_bgem3.parquet
  python scripts/extract_fields.py --max-chunks 30 --output src/knowledge_graph/field_definitions_draft.py
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Allow imports from src/ ────────────────────────────────────────────────────
# Script is run from version_3/, so src/ is on the path after this
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.data.loader import detect_category, VALID_CATEGORIES
from src.llm import get_llm, parse_json


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_PARQUET = settings.parquet_path
DEFAULT_OUTPUT  = "src/knowledge_graph/field_definitions_draft.py"
DEFAULT_MAX_CHUNKS_PER_CAT = 25   # enough context for LLM, not too slow
DEFAULT_MAX_FIELDS = 5            # max fields per domain

# Categories to analyze (exclude "general" — it's a catch-all, not a real product domain)
ANALYZE_CATEGORIES = ["credit_card", "insurance", "loan", "savings"]

# ── LLM prompt ────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """Bạn là chuyên gia phân tích sản phẩm ngân hàng.

Dưới đây là {n_chunks} đoạn tài liệu về sản phẩm "{category_label}" của ngân hàng VIB:

--- TÀI LIỆU ---
{chunks_text}
--- HẾT TÀI LIỆU ---

Nhiệm vụ của bạn:
Phân tích các đoạn tài liệu trên và xác định tối đa {max_fields} thông tin cần thu thập từ khách hàng
để tư vấn chọn đúng sản phẩm {category_label} phù hợp nhất với họ.

Tiêu chí chọn field:
- Field phải PHÂN BIỆT được các sản phẩm khác nhau trong cùng danh mục (ví dụ: thẻ A vs thẻ B)
- Field phải là thông tin khách hàng CÓ THỂ cung cấp dễ dàng qua chat
- Ưu tiên các field xuất hiện nhiều trong tài liệu (điều kiện, đặc điểm sản phẩm...)
- Không hỏi những thứ không liên quan đến việc chọn sản phẩm

Trả về JSON theo đúng format sau (không thêm text nào ngoài JSON):

{{
  "fields": [
    {{
      "field_name": "ten_field_snake_case",
      "label": "Tên field tiếng Việt ngắn gọn",
      "question": "Câu hỏi lịch sự để hỏi khách hàng, có ví dụ gợi ý trong ngoặc đơn"
    }}
  ],
  "domain_label": "Nhãn tiếng Việt của domain này (ví dụ: thẻ tín dụng / thẻ ghi nợ)",
  "domain_keyword": "Từ khóa tìm kiếm cho domain này (ví dụ: thẻ tín dụng VIB)"
}}
"""


# ── Core logic ─────────────────────────────────────────────────────────────────

def load_chunks_by_category(parquet_path: str, max_chunks: int) -> dict[str, list[str]]:
    """
    Load parquet, detect categories, return sampled chunks per category.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path.resolve()}")

    logger.info(f"Loading parquet: {path}")
    df = pd.read_parquet(path)
    texts = df["input"].tolist()
    logger.info(f"Total chunks: {len(texts)}")

    # Detect categories
    logger.info("Detecting categories...")
    categories = [detect_category(t) for t in texts]

    # Count distribution
    dist = {c: categories.count(c) for c in VALID_CATEGORIES}
    logger.info(f"Category distribution: {dist}")

    # Group + sample
    chunks_by_cat: dict[str, list[str]] = {c: [] for c in ANALYZE_CATEGORIES}
    for text, cat in zip(texts, categories):
        if cat in chunks_by_cat:
            chunks_by_cat[cat].append(text)

    sampled: dict[str, list[str]] = {}
    for cat, chunks in chunks_by_cat.items():
        if not chunks:
            logger.warning(f"  No chunks found for category '{cat}'")
            continue
        # Random sample so we get diverse product coverage, not just the first N
        sample = random.sample(chunks, min(max_chunks, len(chunks)))
        sampled[cat] = sample
        logger.info(f"  {cat}: {len(chunks)} chunks → sampled {len(sample)}")

    return sampled


def analyze_category(
    category: str,
    chunks: list[str],
    max_fields: int,
) -> dict:
    """
    Call LLM to analyze chunks for one category.
    Returns parsed dict with 'fields', 'domain_label', 'domain_keyword'.
    """
    CATEGORY_LABELS = {
        "credit_card": "thẻ tín dụng / thẻ ghi nợ",
        "insurance":   "bảo hiểm",
        "loan":        "vay vốn",
        "savings":     "tiết kiệm / đầu tư",
    }

    # Format chunks — truncate each to 400 chars to keep context manageable
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        short = chunk[:400].replace("\n", " ").strip()
        if len(chunk) > 400:
            short += "..."
        formatted.append(f"[{i}] {short}")
    chunks_text = "\n\n".join(formatted)

    prompt = ANALYSIS_PROMPT.format(
        n_chunks=len(chunks),
        category_label=CATEGORY_LABELS.get(category, category),
        chunks_text=chunks_text,
        max_fields=max_fields,
    )

    logger.info(f"  Calling LLM for category '{category}' ({len(chunks)} chunks)...")

    llm = get_llm(temperature=0.0, num_ctx=8192)
    response = llm.invoke(prompt)
    raw = response.content

    result = parse_json(raw)

    # Validate structure
    if "fields" not in result or not isinstance(result["fields"], list):
        logger.warning(f"  LLM returned unexpected structure for '{category}': {raw[:300]}")
        return {}

    # Validate each field has required keys
    valid_fields = []
    for f in result["fields"]:
        if all(k in f for k in ("field_name", "label", "question")):
            # Sanitize field_name to valid Python identifier
            fn = re.sub(r"[^a-z0-9_]", "_", f["field_name"].lower().strip())
            valid_fields.append({
                "field_name": fn,
                "label":      f["label"].strip(),
                "question":   f["question"].strip(),
            })
        else:
            logger.warning(f"  Skipping malformed field: {f}")

    result["fields"] = valid_fields[:max_fields]
    logger.info(f"  → {len(valid_fields)} fields extracted for '{category}'")
    for fld in valid_fields:
        logger.info(f"     • {fld['field_name']}: {fld['label']}")

    return result


def generate_draft_file(
    results: dict[str, dict],
    output_path: str,
    parquet_path: str,
) -> None:
    """
    Write field_definitions_draft.py from analyzed results.
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append('"""')
    lines.append("field_definitions_draft.py — AUTO-GENERATED, DO NOT USE IN PRODUCTION")
    lines.append("")
    lines.append(f"Generated : {now}")
    lines.append(f"Source    : {parquet_path}")
    lines.append(f"Script    : scripts/extract_fields.py")
    lines.append("")
    lines.append("Review this file, make any corrections, then copy the contents")
    lines.append("into src/knowledge_graph/field_definitions.py.")
    lines.append('"""')
    lines.append("")

    # ── DOMAIN_FIELDS ──────────────────────────────────────────────────────────
    lines.append("DOMAIN_FIELDS: dict[str, dict[str, str]] = {")

    for cat in ANALYZE_CATEGORIES:
        if cat not in results or not results[cat].get("fields"):
            logger.warning(f"  No results for '{cat}' — skipping in output")
            continue

        data = results[cat]
        lines.append(f'    "{cat}": {{')

        for fld in data["fields"]:
            fn = fld["field_name"]
            q  = fld["question"].replace('"', '\\"')
            lines.append(f'        "{fn}": (')
            # Word-wrap the question at ~80 chars
            words = q.split()
            current = ""
            first = True
            for word in words:
                if len(current) + len(word) + 1 > 75:
                    prefix = '            "' if first else '            "'
                    lines.append(f'{prefix}{current} "')
                    first = False
                    current = word
                else:
                    current = (current + " " + word).strip()
            if current:
                lines.append(f'            "{current}"')
            lines.append("        ),")

        lines.append("    },")

    # general — always include a minimal catch-all
    lines.append('    "general": {')
    lines.append('        "loai_san_pham": (')
    lines.append('            "Bạn đang quan tâm đến sản phẩm hay dịch vụ nào của VIB ạ? "')
    lines.append('            "(thẻ, vay, bảo hiểm, tiết kiệm, hay dịch vụ khác?)"')
    lines.append("        ),")
    lines.append("    },")
    lines.append("}")
    lines.append("")

    # ── DOMAIN_LABELS ──────────────────────────────────────────────────────────
    lines.append("# Domain display names (for confirm messages)")
    lines.append("DOMAIN_LABELS = {")
    for cat in ANALYZE_CATEGORIES:
        if cat in results and results[cat].get("domain_label"):
            label = results[cat]["domain_label"].replace('"', '\\"')
        else:
            label = cat.replace("_", " ")
        lines.append(f'    "{cat}": "{label}",')
    lines.append('    "general": "sản phẩm ngân hàng",')
    lines.append("}")
    lines.append("")

    # ── DOMAIN_KEYWORDS ────────────────────────────────────────────────────────
    lines.append("# Domain → retrieval bias keywords")
    lines.append("DOMAIN_KEYWORDS = {")
    for cat in ANALYZE_CATEGORIES:
        if cat in results and results[cat].get("domain_keyword"):
            kw = results[cat]["domain_keyword"].replace('"', '\\"')
        else:
            kw = f"{cat.replace('_', ' ')} VIB"
        lines.append(f'    "{cat}": "{kw}",')
    lines.append('    "general": "sản phẩm dịch vụ VIB",')
    lines.append("}")
    lines.append("")

    # ── Helper functions (same as production file) ─────────────────────────────
    lines.append("")
    lines.append("def get_fields(domain: str) -> dict[str, str]:")
    lines.append('    return DOMAIN_FIELDS.get(domain, DOMAIN_FIELDS["general"])')
    lines.append("")
    lines.append("")
    lines.append("def get_domain_keyword(domain: str) -> str:")
    lines.append('    return DOMAIN_KEYWORDS.get(domain, "VIB ngân hàng")')
    lines.append("")

    # Write file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Draft written to: {out.resolve()}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract advisory fields from parquet and generate field_definitions_draft.py"
    )
    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET,
        help=f"Path to parquet file (default: {DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_CAT,
        help=f"Max chunks to sample per category (default: {DEFAULT_MAX_CHUNKS_PER_CAT})",
    )
    parser.add_argument(
        "--max-fields",
        type=int,
        default=DEFAULT_MAX_FIELDS,
        help=f"Max fields per domain (default: {DEFAULT_MAX_FIELDS})",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=ANALYZE_CATEGORIES,
        choices=ANALYZE_CATEGORIES,
        help="Which categories to analyze (default: all 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk sampling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("extract_fields.py — Field extraction from parquet")
    logger.info("=" * 60)
    logger.info(f"Parquet  : {args.parquet}")
    logger.info(f"Output   : {args.output}")
    logger.info(f"Chunks   : up to {args.max_chunks} per category")
    logger.info(f"Fields   : up to {args.max_fields} per domain")
    logger.info(f"Categories: {args.categories}")
    logger.info("=" * 60)

    # Step 1: Load and group chunks
    chunks_by_cat = load_chunks_by_category(args.parquet, args.max_chunks)

    # Step 2: Analyze each category
    results: dict[str, dict] = {}
    for cat in args.categories:
        if cat not in chunks_by_cat or not chunks_by_cat[cat]:
            logger.warning(f"Skipping '{cat}' — no chunks found")
            continue

        logger.info(f"\n{'─'*50}")
        logger.info(f"Analyzing category: {cat}")
        logger.info(f"{'─'*50}")

        try:
            result = analyze_category(
                category=cat,
                chunks=chunks_by_cat[cat],
                max_fields=args.max_fields,
            )
            if result:
                results[cat] = result
        except Exception as e:
            logger.error(f"Failed to analyze '{cat}': {e}")

    if not results:
        logger.error("No categories were successfully analyzed. Exiting.")
        sys.exit(1)

    # Step 3: Write draft file
    logger.info(f"\n{'─'*50}")
    logger.info("Generating draft file...")
    generate_draft_file(results, args.output, args.parquet)

    # Step 4: Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for cat, data in results.items():
        fields = data.get("fields", [])
        logger.info(f"  {cat} ({len(fields)} fields):")
        for f in fields:
            logger.info(f"    - {f['field_name']}: {f['label']}")
    logger.info(f"\nDraft file: {Path(args.output).resolve()}")
    logger.info("Next step : review the file, then copy to src/knowledge_graph/field_definitions.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
