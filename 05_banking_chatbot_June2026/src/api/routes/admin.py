"""Admin API routes — data loading and health check."""
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.models.schemas import HealthResponse, LoadDataResponse
from config.settings import settings

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    from src.data.loader import get_vectorstore_stats
    stats = get_vectorstore_stats()
    return HealthResponse(
        status="ok",
        vectorstore_ready=stats["ready"],
        vectorstore_count=stats["count"],
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


@router.post("/load", response_model=LoadDataResponse)
async def load_data(force_reload: bool = False):
    """Load documents from parquet into ChromaDB + BM25. Idempotent."""
    try:
        from src.data.loader import load_data as _load
        result = _load(force_reload=force_reload)
        return LoadDataResponse(
            status=result["status"],
            message=f"Loaded {result['chunks_loaded']} chunks" if result["chunks_loaded"] else result["status"],
            chunks_loaded=result["chunks_loaded"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Load data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    from src.data.loader import get_vectorstore_stats
    from src.knowledge_graph.field_definitions import DOMAIN_FIELDS
    from src.history.conversation_store import get_store
    vs_stats = get_vectorstore_stats()
    history_stats = get_store().get_stats()
    return {
        "vectorstore": vs_stats,
        "history": history_stats,
        "cache_threshold": settings.cache_similarity_threshold,
        "domains": list(DOMAIN_FIELDS.keys()),
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "parquet_path": settings.parquet_path,
    }


@router.get("/history/{user_id}")
async def get_user_history_admin(user_id: str, limit: int = 50):
    """Xem lịch sử của một user (admin view)."""
    from src.history.conversation_store import get_store
    entries = get_store().get_user_history(user_id)
    return {
        "user_id": user_id,
        "total": len(entries),
        "entries": list(reversed(entries))[:limit],
    }


@router.post("/rebuild-history")
async def rebuild_history():
    """
    Xóa toàn bộ ChromaDB history index và rebuild lại từ JSON files hiện tại.

    Dùng khi:
    - Đã xóa/sửa entries trực tiếp trong JSON files và muốn sync ngay vào ChromaDB
    - ChromaDB index bị lệch so với JSON (stale entries còn sót lại)

    Lưu ý: Tác vụ này re-embed toàn bộ entries nên mất thời gian nếu có nhiều lịch sử.
    """
    try:
        from src.history.conversation_store import get_store
        store = get_store()
        # Lấy stats trước khi rebuild
        before = store.get_stats()
        # Force full rebuild
        store.rebuild_history()
        # Stats sau rebuild
        after = store.get_stats()
        return {
            "status": "ok",
            "message": "ChromaDB history rebuilt from JSON files.",
            "before": before,
            "after": after,
        }
    except Exception as e:
        logger.error(f"rebuild-history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
