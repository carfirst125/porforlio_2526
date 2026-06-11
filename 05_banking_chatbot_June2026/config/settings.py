"""Centralized configuration — pydantic-settings, overridable via .env"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="deepseek-r1:8b")
    llm_temperature: float = Field(default=0.1)
    llm_num_ctx: int = Field(default=4096)
    ollama_num_gpu: int = Field(default=-1)   # -1=all GPU, 0=CPU, N=N layers

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_model: str = Field(default="bge-m3:latest")
    embedding_dim: int = Field(default=1024)

    # ── Data ────────────────────────────────────────────────────────────────
    # Path relative to version_3/ root (or absolute)
    parquet_path: str = Field(default="../documents_bgem3.parquet")
    chroma_persist_dir: str = Field(default="./data/vectorstore")
    chroma_collection_name: str = Field(default="vib_products_v3")

    # ── Retrieval ───────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(default=6)
    top_k_final: int = Field(default=4)
    bm25_weight: float = Field(default=0.35)
    semantic_weight: float = Field(default=0.65)

    # ── Advisor ─────────────────────────────────────────────────────────────
    max_advisor_turns: int = Field(default=8)

    # ── API ─────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=True)

    # ── Conversation History & Cache ─────────────────────────────────────────
    conversations_dir: str = Field(default="./data/conversations")
    history_collection_name: str = Field(default="conversation_history_v3")

    # Số candidates lấy từ semantic search trước khi re-rank
    cache_top_k: int = Field(default=5)

    # Có dùng LLM equivalence gate để xác nhận trước khi serve cache không.
    #   True  (mặc định): pipeline 3 bước — pre-filter → hybrid → LLM verify.
    #   False (legacy):   pipeline 2 bước — pre-filter → hybrid, dùng threshold cứng.
    cache_llm_verify: bool = Field(default=True)

    # Hybrid similarity threshold (0.0–1.0) — nguồn duy nhất kiểm soát pre-filter.
    #   Khi cache_llm_verify=True:  đặt thấp (0.72) — LLM gate xử lý false positive.
    #   Khi cache_llm_verify=False: đặt cao  (0.85) — threshold là rào chắn duy nhất.
    cache_similarity_threshold: float = Field(default=0.72)

    # Model nhẹ dùng riêng cho cache equivalence gate (binary task, input ngắn).
    #   Để trống ("") → fallback về llm_model (deepseek-r1:8b).
    #   Ví dụ: "qwen2.5:3b" để giảm latency, tiết kiệm VRAM.
    #   KHÔNG dùng model này cho RAG, advisory, intent classifier — chỉ cho gate nhị phân.
    cache_verify_model: str = Field(default="")

    # Model dùng cho LLM-as-judge trong evaluation.
    #   NÊN dùng model KHÔNG phải reasoning (không có <think> tokens) để tránh lỗi
    #   llama-server 500 và timeout dài. Ví dụ: "qwen2.5:7b", "llama3.2:3b".
    #   Để trống ("") → fallback về llm_model (có thể gây lỗi nếu là reasoning model).
    eval_judge_model: str = Field(default="")

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self):
        for d in [self.chroma_persist_dir, self.log_dir, self.conversations_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
