"""FastAPI application — VIB Chatbot V3"""
import sys
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes.chat import router as chat_router
from src.api.routes.admin import router as admin_router

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level=settings.log_level)
logger.add(
    f"{settings.log_dir}/chatbot_v3.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level,
)


# ── Startup: pre-load data ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== VIB Chatbot V3 starting up ===")
    # 1. Load product documents (parquet → ChromaDB + BM25)
    try:
        from src.data.loader import load_data
        result = load_data()
        logger.info(f"Data layer ready: {result}")
    except Exception as e:
        logger.error(f"Data load failed at startup: {e}. Use POST /admin/load to retry.")
    # 2. Load conversation history (JSON files → ChromaDB history index)
    try:
        from src.history.conversation_store import get_store
        store = get_store()
        store.load_all_history()
        stats = store.get_stats()
        logger.info(f"History loaded: {stats}")
    except Exception as e:
        logger.error(f"History load failed: {e}")
    yield
    logger.info("=== Shutting down ===")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VIB Banking Chatbot API v3",
    description="Intent-based chatbot: GREETING | PERSONAL | RAG | ADVISORY | CUSTOMER_FEEDBACK",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(admin_router)


@app.get("/")
async def root():
    return {
        "message": "VIB Chatbot API v3 is running",
        "docs": "/docs",
        "health": "/admin/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
