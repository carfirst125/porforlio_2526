"""Pydantic schemas for API request/response."""
from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str = "UID0000"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: Optional[str] = None
    advisor_domain: Optional[str] = None
    sources: list[str] = []
    turn_count: int = 0
    collected_info: Optional[dict] = None
    # True nếu câu trả lời lấy từ cache lịch sử
    from_cache: bool = False
    cache_similarity: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    vectorstore_count: int = 0
    llm_model: str
    embedding_model: str


class LoadDataResponse(BaseModel):
    status: str
    message: str
    chunks_loaded: int = 0
