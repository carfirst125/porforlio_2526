"""
Data loader: parquet → ChromaDB + BM25 corpus.

Parquet schema: {input: str, embedding: str (JSON-encoded list[float])}
Embeddings are pre-computed with bge-m3 (dim=1024).

V3 change: each chunk is tagged with a `category` metadata field
(credit_card | insurance | loan | savings | general) so the retriever
can filter ChromaDB by category for faster, more accurate advisory search.
"""
import json
from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings

# ── Singletons ──────────────────────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_texts: Optional[list[str]] = None
_categories: Optional[list[str]] = None          # parallel list of category per text
_bm25: Optional[BM25Okapi] = None
_bm25_by_category: Optional[dict[str, BM25Okapi]] = None
_texts_by_category: Optional[dict[str, list[str]]] = None


# ── Category detection ───────────────────────────────────────────────────────

# Order matters: more specific categories first.
# Keys map to the ChromaDB metadata field value.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "credit_card": [
        "thẻ tín dụng", "thẻ ghi nợ", "thẻ atm", "thẻ visa", "thẻ mastercard",
        "cashback", "hoàn tiền", "tích điểm", "hạn mức thẻ", "thẻ đen", "thẻ vàng",
        "thẻ platinum", "thẻ online plus", "thẻ travel", "super card",
        "phí thường niên thẻ", "dặm bay",
    ],
    "insurance": [
        "bảo hiểm nhân thọ", "bảo hiểm sức khỏe", "bảo hiểm tai nạn",
        "bảo hiểm xe", "phí bảo hiểm", "quyền lợi bảo hiểm",
        "hợp đồng bảo hiểm", "mức bảo hiểm", "bảo hiểm",
    ],
    "loan": [
        "vay mua nhà", "vay mua xe", "vay tiêu dùng", "vay kinh doanh",
        "vay tín chấp", "vay thế chấp", "lãi suất vay", "khoản vay",
        "dư nợ", "trả nợ", "thế chấp", "tín chấp", "vay vốn", "cho vay",
    ],
    "savings": [
        "tiết kiệm", "lãi suất tiết kiệm", "gửi tiết kiệm",
        "kỳ hạn gửi", "sổ tiết kiệm", "online savings",
        "tích lũy", "lãi suất gửi",
    ],
}

VALID_CATEGORIES = {"credit_card", "insurance", "loan", "savings", "general"}


def detect_category(text: str) -> str:
    """
    Detect product category from chunk text using keyword matching.
    Returns one of: credit_card | insurance | loan | savings | general
    """
    text_lower = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "general"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_embedding(raw: str) -> list[float]:
    """Parse embedding stored as double-encoded JSON string."""
    return json.loads(raw.strip('"'))


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def _build_bm25_indexes(texts: list[str], categories: list[str]) -> None:
    """Build global BM25 + per-category BM25 indexes from text/category lists."""
    global _bm25, _bm25_by_category, _texts_by_category, _texts, _categories

    _texts = texts
    _categories = categories

    logger.info("Building global BM25 index...")
    tokenized_all = [t.lower().split() for t in texts]
    _bm25 = BM25Okapi(tokenized_all)

    logger.info("Building per-category BM25 indexes...")
    texts_by_cat: dict[str, list[str]] = {c: [] for c in VALID_CATEGORIES}
    for text, cat in zip(texts, categories):
        texts_by_cat[cat].append(text)

    bm25_by_cat: dict[str, BM25Okapi] = {}
    for cat, cat_texts in texts_by_cat.items():
        if cat_texts:
            tokenized = [t.lower().split() for t in cat_texts]
            bm25_by_cat[cat] = BM25Okapi(tokenized)
            logger.info(f"  BM25[{cat}]: {len(cat_texts)} docs")
        else:
            logger.warning(f"  BM25[{cat}]: no docs — skipped")

    _bm25_by_category = bm25_by_cat
    _texts_by_category = texts_by_cat
    logger.info("All BM25 indexes ready.")


def load_data(force_reload: bool = False) -> dict:
    """
    Load parquet into ChromaDB (with category metadata) and build BM25 indexes.
    Returns stats dict. Idempotent — skips if already loaded unless force_reload=True.
    """
    global _collection

    client = get_chroma_client()

    # ── Check if already loaded ─────────────────────────────────────────────
    if not force_reload:
        try:
            col = client.get_collection(settings.chroma_collection_name)
            count = col.count()
            if count > 0:
                logger.info(f"ChromaDB already has {count} docs, skipping reload.")
                _collection = col
                if _bm25 is None:
                    _build_bm25_from_collection(col)
                return {"status": "already_loaded", "chunks_loaded": count}
        except Exception:
            pass

    # ── Load parquet ─────────────────────────────────────────────────────────
    parquet_path = Path(settings.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path.resolve()}")

    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Parquet loaded: {len(df)} rows")

    # ── Parse ────────────────────────────────────────────────────────────────
    texts = df["input"].tolist()
    logger.info("Parsing embeddings...")
    embeddings = [_parse_embedding(e) for e in df["embedding"]]
    ids = [f"chunk_{i}" for i in range(len(texts))]

    # ── Detect categories ────────────────────────────────────────────────────
    logger.info("Detecting categories for each chunk...")
    categories = [detect_category(t) for t in texts]
    metadatas = [{"category": cat} for cat in categories]
    cat_counts = {c: categories.count(c) for c in VALID_CATEGORIES if categories.count(c) > 0}
    logger.info(f"Category distribution: {cat_counts}")

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    if force_reload:
        try:
            client.delete_collection(settings.chroma_collection_name)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in range(0, len(texts), batch_size):
        col.add(
            ids=ids[i : i + batch_size],
            documents=texts[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        logger.info(f"  Indexed {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    _collection = col

    # ── BM25 indexes ─────────────────────────────────────────────────────────
    _build_bm25_indexes(texts, categories)

    return {"status": "loaded", "chunks_loaded": len(texts), "category_counts": cat_counts}


def _build_bm25_from_collection(col: chromadb.Collection) -> None:
    """
    Rebuild BM25 indexes from existing ChromaDB collection at startup.
    Reads documents + category metadata.
    """
    logger.info("Rebuilding BM25 from existing ChromaDB collection...")
    result = col.get(include=["documents", "metadatas"])
    texts = result["documents"]
    raw_metas = result.get("metadatas") or []

    # Recover categories from metadata; fallback to re-detection if missing
    if raw_metas and raw_metas[0] and "category" in raw_metas[0]:
        categories = [m.get("category", "general") for m in raw_metas]
        logger.info("Loaded categories from ChromaDB metadata.")
    else:
        logger.warning("No category metadata found — re-detecting categories.")
        categories = [detect_category(t) for t in texts]

    _build_bm25_indexes(texts, categories)
    logger.info(f"BM25 rebuilt with {len(texts)} docs.")


def get_collection() -> chromadb.Collection:
    """Return ChromaDB collection, loading if necessary."""
    global _collection
    if _collection is None:
        load_data()
    return _collection


def get_bm25() -> tuple[BM25Okapi, list[str]]:
    """Return (global BM25 index, all texts), loading if necessary."""
    global _bm25, _texts
    if _bm25 is None:
        load_data()
    return _bm25, _texts


def get_bm25_by_category() -> tuple[dict[str, BM25Okapi], dict[str, list[str]]]:
    """
    Return (bm25_by_category, texts_by_category) dicts.
    Keys: credit_card | insurance | loan | savings | general
    """
    global _bm25_by_category, _texts_by_category
    if _bm25_by_category is None:
        load_data()
    return _bm25_by_category, _texts_by_category


def get_vectorstore_stats() -> dict:
    try:
        col = get_chroma_client().get_collection(settings.chroma_collection_name)
        count = col.count()
        # Count per category via metadata query
        cat_counts = {}
        for cat in VALID_CATEGORIES:
            try:
                r = col.get(where={"category": cat}, include=[])
                cat_counts[cat] = len(r["ids"])
            except Exception:
                pass
        return {"ready": True, "count": count, "category_counts": cat_counts}
    except Exception:
        return {"ready": False, "count": 0, "category_counts": {}}
