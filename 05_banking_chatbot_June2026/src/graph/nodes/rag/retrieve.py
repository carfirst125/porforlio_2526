"""
RAG Node 2: Hybrid retrieval.

V3 change: detects category from the rewritten query so that clearly domain-specific
questions (e.g. about credit cards) search only the relevant category chunks,
while general / ambiguous questions still search all chunks.
"""
from typing import Optional
from loguru import logger

from src.graph.state import ChatState
from src.retrieval.retriever import hybrid_retrieve
from src.data.loader import detect_category, VALID_CATEGORIES
from config.settings import settings


def _infer_category_from_query(query: str) -> Optional[str]:
    """
    Infer a category filter from the query text.
    Returns a category string if the query is clearly domain-specific,
    or None to perform an unfiltered (all-category) search.
    """
    cat = detect_category(query)
    # Only filter when we're confident — skip "general" so broad questions
    # still search across all chunks
    return cat if cat != "general" else None


def rag_retrieve_node(state: ChatState) -> dict:
    query = state.get("rewritten_query") or ""
    if not query:
        from langchain_core.messages import HumanMessage
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                query = m.content
                break

    # Detect category — filter search if clearly domain-specific
    category_filter = _infer_category_from_query(query)

    docs = hybrid_retrieve(
        query,
        top_k=settings.top_k_retrieval,
        category_filter=category_filter,
    )
    logger.info(
        f"RAG retrieved {len(docs)} docs | "
        f"category_filter={category_filter!r} | "
        f"query='{query[:60]}'"
    )
    return {"retrieved_docs": docs}
