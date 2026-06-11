"""
extract_fields_graphrag.py — GraphRAG-based field extraction (Cách 3).

Khác biệt so với Cách 2 (extract_fields.py):
  Cách 2: LLM đọc raw text → đoán field.
  Cách 3: Trước tiên xây dựng Knowledge Graph từ tài liệu → phân tích cấu trúc
          graph để tìm các thuộc tính THỰC SỰ phân biệt sản phẩm → LLM chỉ
          cần đặt câu hỏi cho khách hàng dựa trên kết quả graph.

Pipeline 3 bước:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Bước 1 — Triplet Extraction                                     │
  │   Mỗi batch chunk → LLM trích ra (product, attribute, value)    │
  │   Ví dụ: (Thẻ VIB Online Plus, yêu_cầu_thu_nhập, ≥5 triệu)    │
  │          (Thẻ VIB Platinum, yêu_cầu_thu_nhập, ≥20 triệu)       │
  ├─────────────────────────────────────────────────────────────────┤
  │ Bước 2 — Build Knowledge Graph (NetworkX)                       │
  │   Product nodes → Attribute nodes → Value nodes                 │
  │   Tìm "discriminative attributes": attribute kết nối ≥2 sản    │
  │   phẩm với giá trị khác nhau → đây mới là field cần hỏi KH     │
  ├─────────────────────────────────────────────────────────────────┤
  │ Bước 3 — Graph-informed LLM call                                │
  │   Truyền cho LLM tóm tắt graph (không phải raw text):          │
  │   "Attribute X phân biệt sản phẩm A (≥5tr), B (≥15tr), C..."  │
  │   → LLM đặt câu hỏi KH chính xác hơn nhiều so với Cách 2       │
  └─────────────────────────────────────────────────────────────────┘

Output:
  - src/knowledge_graph/field_definitions_graphrag.py
  - data/graph/{category}_graph.json  (inspect graph nếu cần)

Usage (run from version_3/):
  python scripts/extract_fields_graphrag.py
  python scripts/extract_fields_graphrag.py --categories credit_card loan
  python scripts/extract_fields_graphrag.py --max-chunks 60 --batch-size 5
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.data.loader import detect_category, VALID_CATEGORIES
from src.llm import get_llm, parse_json


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_PARQUET          = settings.parquet_path
DEFAULT_OUTPUT           = "src/knowledge_graph/field_definitions_graphrag.py"
DEFAULT_GRAPH_DIR        = "data/graph"
DEFAULT_MAX_CHUNKS       = 50    # more chunks than Cách 2 — graph benefits from volume
DEFAULT_BATCH_SIZE       = 4     # chunks per triplet-extraction LLM call
DEFAULT_MAX_FIELDS       = 5
DEFAULT_MIN_PRODUCTS     = 2     # min distinct products an attribute must appear in to be "discriminative"

ANALYZE_CATEGORIES = ["credit_card", "insurance", "loan", "savings"]

CATEGORY_LABELS = {
    "credit_card": "thẻ tín dụng / thẻ ghi nợ",
    "insurance":   "bảo hiểm",
    "loan":        "vay vốn",
    "savings":     "tiết kiệm / đầu tư",
}

# ── Prompts ────────────────────────────────────────────────────────────────────

TRIPLET_PROMPT = """Bạn là chuyên gia phân tích tài liệu ngân hàng.

Đọc các đoạn tài liệu sau về sản phẩm "{category_label}" của VIB:

--- TÀI LIỆU ---
{chunks_text}
--- HẾT TÀI LIỆU ---

Nhiệm vụ: Trích xuất TẤT CẢ các bộ ba quan hệ (sản phẩm, thuộc tính, giá trị) từ tài liệu trên.

Quy tắc:
- subject: Tên sản phẩm cụ thể (ví dụ: "Thẻ VIB Online Plus", "Gói vay mua nhà VIB Smart")
- relation: Loại thuộc tính, dùng tiếng Anh snake_case ngắn gọn:
    yeu_cau_thu_nhap | yeu_cau_tai_san | yeu_cau_do_tuoi | muc_dich |
    han_muc | lai_suat | phi | ky_han | uu_dai | doi_tuong | dieu_kien | tinh_nang
- object: Giá trị cụ thể (ví dụ: "từ 5 triệu/tháng", "18-65 tuổi", "hoàn tiền 5%")

Chỉ trích xuất những gì tài liệu đề cập rõ ràng. Bỏ qua thông tin mơ hồ.

Trả về JSON (không thêm text nào ngoài JSON):
{{
  "triplets": [
    {{"subject": "...", "relation": "...", "object": "..."}},
    ...
  ]
}}"""


FIELD_GEN_PROMPT = """Bạn là chuyên gia tư vấn sản phẩm ngân hàng.

Dựa trên Knowledge Graph về sản phẩm "{category_label}" của VIB, dưới đây là các thuộc tính
PHÂN BIỆT các sản phẩm với nhau (được tự động phát hiện từ graph):

{graph_summary}

Tổng số sản phẩm phát hiện được: {n_products}
Tổng số thuộc tính phân biệt: {n_attrs}

Nhiệm vụ: Dựa trên các thuộc tính phân biệt trên, xác định tối đa {max_fields} thông tin
cần hỏi KHÁCH HÀNG để tư vấn chọn sản phẩm phù hợp nhất.

Tiêu chí:
- Chỉ hỏi thông tin KHÁCH HÀNG có thể cung cấp (thu nhập, nhu cầu, mục tiêu, tình trạng...)
- KHÔNG hỏi thuộc tính cố định của sản phẩm (hạn mức, lãi suất, phí...) — đó là output tư vấn
- Ưu tiên thuộc tính xuất hiện trong nhiều sản phẩm nhất (discriminative score cao)
- Câu hỏi phải thực tế, dễ trả lời qua chat

Trả về JSON (không thêm text nào ngoài JSON):
{{
  "fields": [
    {{
      "field_name": "ten_field_snake_case",
      "label": "Tên field tiếng Việt ngắn gọn",
      "question": "Câu hỏi lịch sự, có ví dụ gợi ý trong ngoặc đơn",
      "based_on_attrs": ["attr1", "attr2"]
    }}
  ],
  "domain_label": "Nhãn tiếng Việt của domain (ví dụ: thẻ tín dụng / thẻ ghi nợ)",
  "domain_keyword": "Từ khóa tìm kiếm cho domain (ví dụ: thẻ tín dụng VIB)"
}}"""


# ── Step 1: Triplet Extraction ─────────────────────────────────────────────────

def extract_triplets_from_batch(
    chunks: list[str],
    category: str,
    llm,
) -> list[dict]:
    """Call LLM on a batch of chunks → list of {subject, relation, object}."""
    formatted = []
    for i, c in enumerate(chunks, 1):
        short = c[:500].replace("\n", " ").strip()
        if len(c) > 500:
            short += "..."
        formatted.append(f"[{i}] {short}")

    prompt = TRIPLET_PROMPT.format(
        category_label=CATEGORY_LABELS.get(category, category),
        chunks_text="\n\n".join(formatted),
    )

    try:
        response = llm.invoke(prompt)
        result = parse_json(response.content)
        triplets = result.get("triplets", [])

        valid = []
        for t in triplets:
            if all(k in t for k in ("subject", "relation", "object")):
                s = t["subject"].strip()
                r = t["relation"].strip().lower().replace(" ", "_")
                o = t["object"].strip()
                if s and r and o:
                    valid.append({"subject": s, "relation": r, "object": o})
        return valid

    except Exception as e:
        logger.warning(f"    Batch failed: {e}")
        return []


def extract_all_triplets(
    chunks: list[str],
    category: str,
    batch_size: int,
) -> list[dict]:
    """Process all chunks in batches, collect all triplets."""
    llm = get_llm(temperature=0.0, num_ctx=6144)
    all_triplets: list[dict] = []

    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    logger.info(f"  Extracting triplets: {len(chunks)} chunks → {len(batches)} batches")

    for idx, batch in enumerate(batches, 1):
        logger.info(f"    Batch {idx}/{len(batches)}...")
        triplets = extract_triplets_from_batch(batch, category, llm)
        all_triplets.extend(triplets)
        logger.info(f"    → {len(triplets)} triplets extracted")

    logger.info(f"  Total triplets for '{category}': {len(all_triplets)}")
    return all_triplets


# ── Step 2: Knowledge Graph Analysis ──────────────────────────────────────────

class KnowledgeGraph:
    """
    Lightweight graph: no external dependency needed.
    Structure: {attribute: {product: [values...]}}
    Plus reverse index: {product: {attribute: [values...]}}
    """

    def __init__(self):
        # attr → {product → [values]}
        self.attr_product_values: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.all_products: set[str] = set()
        self.triplets: list[dict] = []

    def add_triplets(self, triplets: list[dict]) -> None:
        for t in triplets:
            subj = t["subject"]
            rel  = t["relation"]
            obj  = t["object"]

            self.triplets.append(t)
            self.all_products.add(subj)
            # Only store if value not already present (dedup)
            if obj not in self.attr_product_values[rel][subj]:
                self.attr_product_values[rel][subj].append(obj)

    def get_discriminative_attrs(self, min_products: int = 2) -> list[dict]:
        """
        Return attributes sorted by discriminative power:
        - n_products: how many distinct products have this attribute
        - n_values:   how many distinct values exist across all products
        - variation:  std-dev proxy — are the values actually different?

        Only returns attrs that appear in >= min_products products.
        """
        results = []
        for attr, prod_vals in self.attr_product_values.items():
            n_products = len(prod_vals)
            if n_products < min_products:
                continue

            all_values = [v for vals in prod_vals.values() for v in vals]
            n_distinct_values = len(set(all_values))

            # Discriminative score: more products × more distinct values = higher
            score = n_products * n_distinct_values

            results.append({
                "attribute":        attr,
                "n_products":       n_products,
                "n_distinct_values": n_distinct_values,
                "score":            score,
                "product_values":   dict(prod_vals),  # {product: [values]}
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def to_json(self) -> dict:
        return {
            "products":   list(self.all_products),
            "triplets":   self.triplets,
            "attr_index": {
                attr: dict(prod_vals)
                for attr, prod_vals in self.attr_product_values.items()
            },
        }


def build_graph_summary(discriminative_attrs: list[dict], top_n: int = 15) -> str:
    """
    Format top discriminative attributes into human-readable summary for LLM.
    Shows: attribute name, how many products, sample values per product.
    """
    lines = []
    for i, attr in enumerate(discriminative_attrs[:top_n], 1):
        name      = attr["attribute"]
        n_prod    = attr["n_products"]
        score     = attr["score"]
        prod_vals = attr["product_values"]

        lines.append(
            f"{i}. [{name}] — xuất hiện trong {n_prod} sản phẩm "
            f"(discriminative score: {score})"
        )
        for product, values in list(prod_vals.items())[:4]:  # show max 4 products
            vals_str = " | ".join(values[:2])  # show max 2 values per product
            lines.append(f"   • {product}: {vals_str}")
        if len(prod_vals) > 4:
            lines.append(f"   ... và {len(prod_vals) - 4} sản phẩm khác")

    return "\n".join(lines)


# ── Step 3: Graph-informed field generation ────────────────────────────────────

def generate_fields_from_graph(
    category: str,
    graph: KnowledgeGraph,
    max_fields: int,
    min_products: int,
) -> dict:
    """Final LLM call: graph summary → field definitions."""
    discriminative = graph.get_discriminative_attrs(min_products=min_products)

    if not discriminative:
        logger.warning(
            f"  No discriminative attributes found for '{category}' "
            f"(min_products={min_products}). Try --min-products 1."
        )
        return {}

    logger.info(
        f"  Graph: {len(graph.all_products)} products, "
        f"{len(discriminative)} discriminative attrs"
    )
    for a in discriminative[:8]:
        logger.info(
            f"    [{a['score']:3d}] {a['attribute']}: "
            f"{a['n_products']} products, {a['n_distinct_values']} distinct values"
        )

    graph_summary = build_graph_summary(discriminative, top_n=15)

    prompt = FIELD_GEN_PROMPT.format(
        category_label  = CATEGORY_LABELS.get(category, category),
        graph_summary   = graph_summary,
        n_products      = len(graph.all_products),
        n_attrs         = len(discriminative),
        max_fields      = max_fields,
    )

    logger.info(f"  Calling LLM for field generation ('{category}')...")
    llm = get_llm(temperature=0.0, num_ctx=4096)
    response = llm.invoke(prompt)
    result = parse_json(response.content)

    if "fields" not in result or not isinstance(result["fields"], list):
        logger.warning(f"  Unexpected LLM response: {response.content[:300]}")
        return {}

    # Sanitize field_names
    valid_fields = []
    for f in result["fields"]:
        if all(k in f for k in ("field_name", "label", "question")):
            fn = re.sub(r"[^a-z0-9_]", "_", f["field_name"].lower().strip())
            valid_fields.append({
                "field_name":    fn,
                "label":         f["label"].strip(),
                "question":      f["question"].strip(),
                "based_on_attrs": f.get("based_on_attrs", []),
            })

    result["fields"] = valid_fields[:max_fields]
    logger.info(f"  → {len(valid_fields)} fields generated for '{category}'")
    for fld in valid_fields:
        logger.info(f"     • {fld['field_name']}: {fld['label']}")

    return result


# ── Output file generation ─────────────────────────────────────────────────────

def generate_output_file(
    results: dict[str, dict],
    graphs: dict[str, KnowledgeGraph],
    output_path: str,
    graph_dir: str,
    parquet_path: str,
) -> None:
    """Write field_definitions_graphrag.py + per-category graph JSON files."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Save graph JSON files (for inspection) ─────────────────────────────────
    gdir = Path(graph_dir)
    gdir.mkdir(parents=True, exist_ok=True)
    for cat, graph in graphs.items():
        gfile = gdir / f"{cat}_graph.json"
        gfile.write_text(
            json.dumps(graph.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"  Graph saved: {gfile}")

    # ── Write Python file ──────────────────────────────────────────────────────
    lines = []

    lines.append('"""')
    lines.append("field_definitions_graphrag.py — AUTO-GENERATED via GraphRAG (Cách 3)")
    lines.append("")
    lines.append(f"Generated : {now}")
    lines.append(f"Source    : {parquet_path}")
    lines.append(f"Script    : scripts/extract_fields_graphrag.py")
    lines.append("")
    lines.append("Fields were derived from a Knowledge Graph of product triplets,")
    lines.append("not from raw text — attributes that appear in multiple products")
    lines.append("with different values are used as advisory fields.")
    lines.append("")
    lines.append("Graph artifacts: data/graph/{category}_graph.json")
    lines.append("")
    lines.append("Review then copy to src/knowledge_graph/field_definitions.py")
    lines.append('"""')
    lines.append("")

    # ── DOMAIN_FIELDS ──────────────────────────────────────────────────────────
    lines.append("DOMAIN_FIELDS: dict[str, dict[str, str]] = {")

    for cat in ANALYZE_CATEGORIES:
        if cat not in results or not results[cat].get("fields"):
            logger.warning(f"  No results for '{cat}' — skipping")
            continue

        fields = results[cat]["fields"]
        lines.append(f'    "{cat}": {{')

        for fld in fields:
            fn = fld["field_name"]
            q  = fld["question"].replace('"', '\\"')
            attrs = fld.get("based_on_attrs", [])
            if attrs:
                lines.append(f'        # graph attrs: {", ".join(attrs[:3])}')
            lines.append(f'        "{fn}": (')
            # Word-wrap at ~75 chars
            words = q.split()
            current = ""
            first = True
            for word in words:
                if len(current) + len(word) + 1 > 72:
                    lines.append(f'            "{current} "')
                    first = False
                    current = word
                else:
                    current = (current + " " + word).strip()
            if current:
                lines.append(f'            "{current}"')
            lines.append("        ),")

        lines.append("    },")

    # general always included
    lines.append('    "general": {')
    lines.append('        "loai_san_pham": (')
    lines.append('            "Bạn đang quan tâm đến sản phẩm hay dịch vụ nào của VIB ạ? "')
    lines.append('            "(thẻ, vay, bảo hiểm, tiết kiệm, hay dịch vụ khác?)"')
    lines.append("        ),")
    lines.append("    },")
    lines.append("}")
    lines.append("")

    # ── DOMAIN_LABELS ──────────────────────────────────────────────────────────
    lines.append("DOMAIN_LABELS = {")
    for cat in ANALYZE_CATEGORIES:
        if cat in results and results[cat].get("domain_label"):
            label = results[cat]["domain_label"].replace('"', '\\"')
        else:
            label = CATEGORY_LABELS.get(cat, cat)
        lines.append(f'    "{cat}": "{label}",')
    lines.append('    "general": "sản phẩm ngân hàng",')
    lines.append("}")
    lines.append("")

    # ── DOMAIN_KEYWORDS ────────────────────────────────────────────────────────
    lines.append("DOMAIN_KEYWORDS = {")
    for cat in ANALYZE_CATEGORIES:
        if cat in results and results[cat].get("domain_keyword"):
            kw = results[cat]["domain_keyword"].replace('"', '\\"')
        else:
            kw = f"{CATEGORY_LABELS.get(cat, cat)} VIB"
        lines.append(f'    "{cat}": "{kw}",')
    lines.append('    "general": "sản phẩm dịch vụ VIB",')
    lines.append("}")
    lines.append("")

    # ── Helpers ────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("def get_fields(domain: str) -> dict[str, str]:")
    lines.append('    return DOMAIN_FIELDS.get(domain, DOMAIN_FIELDS["general"])')
    lines.append("")
    lines.append("")
    lines.append("def get_domain_keyword(domain: str) -> str:")
    lines.append('    return DOMAIN_KEYWORDS.get(domain, "VIB ngân hàng")')
    lines.append("")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Output written: {out.resolve()}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG-based advisory field extraction (Cách 3)"
    )
    parser.add_argument("--parquet",       default=DEFAULT_PARQUET)
    parser.add_argument("--output",        default=DEFAULT_OUTPUT)
    parser.add_argument("--graph-dir",     default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--max-chunks",    type=int, default=DEFAULT_MAX_CHUNKS,
                        help="Max chunks to sample per category (default: 50)")
    parser.add_argument("--batch-size",    type=int, default=DEFAULT_BATCH_SIZE,
                        help="Chunks per triplet-extraction LLM call (default: 4)")
    parser.add_argument("--max-fields",    type=int, default=DEFAULT_MAX_FIELDS)
    parser.add_argument("--min-products",  type=int, default=DEFAULT_MIN_PRODUCTS,
                        help="Min products an attr must appear in to be discriminative (default: 2)")
    parser.add_argument("--categories",    nargs="+", default=ANALYZE_CATEGORIES,
                        choices=ANALYZE_CATEGORIES)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("extract_fields_graphrag.py — GraphRAG Field Extraction")
    logger.info("=" * 60)
    logger.info(f"Parquet      : {args.parquet}")
    logger.info(f"Max chunks   : {args.max_chunks}/category")
    logger.info(f"Batch size   : {args.batch_size} chunks/LLM call")
    logger.info(f"Min products : {args.min_products} (discriminative threshold)")
    logger.info(f"Categories   : {args.categories}")
    logger.info("=" * 60)

    # ── Load parquet ───────────────────────────────────────────────────────────
    path = Path(args.parquet)
    if not path.exists():
        logger.error(f"Parquet not found: {path.resolve()}")
        sys.exit(1)

    df = pd.read_parquet(path)
    texts = df["input"].tolist()
    logger.info(f"Loaded {len(texts)} chunks from parquet")

    cats = [detect_category(t) for t in texts]
    dist = {c: cats.count(c) for c in VALID_CATEGORIES}
    logger.info(f"Category distribution: {dist}")

    # Group by category
    chunks_by_cat: dict[str, list[str]] = defaultdict(list)
    for text, cat in zip(texts, cats):
        if cat in args.categories:
            chunks_by_cat[cat].append(text)

    # ── Process each category ──────────────────────────────────────────────────
    results: dict[str, dict] = {}
    graphs:  dict[str, KnowledgeGraph] = {}

    for cat in args.categories:
        raw_chunks = chunks_by_cat.get(cat, [])
        if not raw_chunks:
            logger.warning(f"No chunks for '{cat}' — skipping")
            continue

        sample = random.sample(raw_chunks, min(args.max_chunks, len(raw_chunks)))
        logger.info(f"\n{'─'*50}")
        logger.info(f"Category: {cat}  ({len(sample)} chunks sampled from {len(raw_chunks)})")
        logger.info(f"{'─'*50}")

        # Step 1: Extract triplets
        try:
            triplets = extract_all_triplets(sample, cat, args.batch_size)
        except Exception as e:
            logger.error(f"Triplet extraction failed for '{cat}': {e}")
            continue

        if not triplets:
            logger.warning(f"  Zero triplets extracted for '{cat}' — skipping")
            continue

        # Step 2: Build graph
        graph = KnowledgeGraph()
        graph.add_triplets(triplets)
        graphs[cat] = graph

        logger.info(
            f"  Graph built: {len(graph.all_products)} products, "
            f"{len(graph.attr_product_values)} unique attributes"
        )

        # Step 3: Graph-informed field generation
        try:
            result = generate_fields_from_graph(
                category=cat,
                graph=graph,
                max_fields=args.max_fields,
                min_products=args.min_products,
            )
            if result:
                results[cat] = result
        except Exception as e:
            logger.error(f"Field generation failed for '{cat}': {e}")

    if not results:
        logger.error("No categories produced results. Exiting.")
        sys.exit(1)

    # ── Write outputs ──────────────────────────────────────────────────────────
    logger.info(f"\n{'─'*50}")
    logger.info("Writing output files...")
    generate_output_file(results, graphs, args.output, args.graph_dir, args.parquet)

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for cat, data in results.items():
        g = graphs.get(cat)
        g_info = f"{len(g.all_products)} products, {len(g.attr_product_values)} attrs" if g else "n/a"
        fields = data.get("fields", [])
        logger.info(f"  {cat}  [graph: {g_info}]  → {len(fields)} fields:")
        for f in fields:
            logger.info(f"    - {f['field_name']}: {f['label']}")

    logger.info(f"\nPython output : {Path(args.output).resolve()}")
    logger.info(f"Graph JSONs   : {Path(args.graph_dir).resolve()}/{{category}}_graph.json")
    logger.info("Next step     : review output, compare with field_definitions.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
