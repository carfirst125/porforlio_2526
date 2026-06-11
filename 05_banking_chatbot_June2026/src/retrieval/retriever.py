"""
HybridRetriever: BM25 keyword search + Semantic search (ChromaDB) → RRF fusion.
Query embedding computed on-the-fly via Ollama bge-m3.

V3 change: supports `category_filter` param.
  - Semantic search: uses ChromaDB `where={"category": category_filter}` clause
    to restrict vector search to chunks of that category only.
  - BM25 search: uses per-category BM25 index (built at startup) for the same effect.
  This makes advisory retrieval faster and more precise than searching all chunks.
"""
from typing import Optional
from loguru import logger

from src.data.loader import get_collection, get_bm25, get_bm25_by_category
from src.llm import embed_query
from config.settings import settings


def _rrf_merge(
    semantic_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k_rrf: int = 60,
    sem_weight: float = None,
    bm25_weight: float = None,
) -> list[str]:
    """Reciprocal Rank Fusion — returns texts sorted by merged score."""
    sem_w = sem_weight or settings.semantic_weight
    bm25_w = bm25_weight or settings.bm25_weight

    scores: dict[str, float] = {}

    for rank, (text, _) in enumerate(semantic_results, start=1):
        key = text[:120]
        scores[key] = scores.get(key, 0) + sem_w / (k_rrf + rank)

    for rank, (text, _) in enumerate(bm25_results, start=1):
        key = text[:120]
        scores[key] = scores.get(key, 0) + bm25_w / (k_rrf + rank)

    # Rebuild text map (use first occurrence)
    text_map: dict[str, str] = {}
    for text, _ in [*semantic_results, *bm25_results]:
        key = text[:120]
        if key not in text_map:
            text_map[key] = text

    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [text_map[k] for k in ranked]


def hybrid_retrieve(
    query: str,
    top_k: int = None,
    domain_hint: Optional[str] = None,
    category_filter: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve top-k documents using hybrid BM25 + Semantic search.

    Args:
        query:           Search query (plain text)
        top_k:           Number of results to return
        domain_hint:     Optional keyword to prepend for better domain context
        category_filter: If set (e.g. "credit_card"), restrict search to chunks
                         of that category only — faster and more precise.

    Returns:
        list of {content: str, score_rank: int, category: str}
    """
    k = top_k or settings.top_k_retrieval
    fetch_k = k * 2

    # Optionally enrich query with domain hint
    search_query = f"{domain_hint} {query}".strip() if domain_hint else query

    # ── Semantic search ──────────────────────────────────────────────────────
    collection = get_collection()
    try:
        query_vec = embed_query(search_query)

        # Build optional where clause for category filtering
        where_clause = {"category": category_filter} if category_filter else None

        query_kwargs = dict(
            query_embeddings=[query_vec],
            n_results=fetch_k,
            include=["documents", "distances", "metadatas"],
        )
        if where_clause:
            query_kwargs["where"] = where_clause

        sem_res = collection.query(**query_kwargs)
        sem_docs  = sem_res["documents"][0]
        sem_dists = sem_res["distances"][0]
        sem_metas = sem_res.get("metadatas", [[]])[0]

        # Fallback: if category filter returned 0 results (e.g. collection has no
        # category metadata), retry without the where clause so semantic search
        # always contributes results.
        if where_clause and len(sem_docs) == 0:
            logger.warning(
                f"Semantic search with where={where_clause} returned 0 docs — "
                "falling back to full-collection search."
            )
            fallback_kwargs = dict(query_kwargs)
            fallback_kwargs.pop("where", None)
            sem_res   = collection.query(**fallback_kwargs)
            sem_docs  = sem_res["documents"][0]
            sem_dists = sem_res["distances"][0]
            sem_metas = sem_res.get("metadatas", [[]])[0]

        semantic_results = list(zip(sem_docs, sem_dists))
        # Map text → category from semantic results
        sem_cat_map = {doc[:120]: (meta or {}).get("category", "general")
                       for doc, meta in zip(sem_docs, sem_metas)}
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        semantic_results = []
        sem_cat_map = {}

    # ── BM25 search ──────────────────────────────────────────────────────────
    bm25_results: list[tuple[str, float]] = []
    bm25_cat_map: dict[str, str] = {}
    try:
        tokens = search_query.lower().split()

        if category_filter:
            # Use per-category BM25 index for filtered search
            bm25_by_cat, texts_by_cat = get_bm25_by_category()
            cat_bm25 = bm25_by_cat.get(category_filter)
            cat_texts = texts_by_cat.get(category_filter, [])
            if cat_bm25 and cat_texts:
                scores = cat_bm25.get_scores(tokens)
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                for idx in ranked_indices[:fetch_k]:
                    bm25_results.append((cat_texts[idx], float(scores[idx])))
                    bm25_cat_map[cat_texts[idx][:120]] = category_filter
            else:
                logger.warning(f"No per-category BM25 for '{category_filter}', falling back to global.")
                bm25, texts = get_bm25()
                scores = bm25.get_scores(tokens)
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                for idx in ranked_indices[:fetch_k]:
                    bm25_results.append((texts[idx], float(scores[idx])))
        else:
            # Global BM25 — no filtering
            bm25, texts = get_bm25()
            scores = bm25.get_scores(tokens)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for idx in ranked_indices[:fetch_k]:
                bm25_results.append((texts[idx], float(scores[idx])))
    except Exception as e:
        logger.error(f"BM25 search error: {e}")

    # ── RRF fusion ───────────────────────────────────────────────────────────
    fused = _rrf_merge(semantic_results, bm25_results)
    top_texts = fused[:k]

    logger.debug(
        f"Hybrid retrieve: category_filter={category_filter!r}, "
        f"semantic={len(semantic_results)}, bm25={len(bm25_results)}, fused top-{k}"
    )

    # Attach category to each result (from semantic map, then bm25 map, then filter value)
    results = []
    for i, text in enumerate(top_texts):
        key = text[:120]
        cat = (
            sem_cat_map.get(key)
            or bm25_cat_map.get(key)
            or category_filter
            or "general"
        )
        results.append({"content": text, "score_rank": i + 1, "category": cat})

    return results


def format_context(docs: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved docs into a prompt context string."""
    parts = []
    total = 0
    for i, doc in enumerate(docs, start=1):
        snippet = doc["content"][:1500]
        parts.append(f"[Tài liệu {i}]\n{snippet}")
        total += len(snippet)
        if total > max_chars:
            break
    return "\n\n---\n\n".join(parts)
