"""LLM + Embedding factory. All nodes import from here."""
import json
import re
from loguru import logger
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.settings import settings


def get_llm(temperature: float = None, num_ctx: int = None) -> ChatOllama:
    """Main LLM — dùng cho tất cả reasoning tasks (intent, RAG, advisory, recommender)."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        num_ctx=num_ctx or settings.llm_num_ctx,
        num_gpu=settings.ollama_num_gpu,
    )


def get_fast_llm(temperature: float = 0.0, num_ctx: int = 2048) -> ChatOllama:
    """
    Lightweight LLM — dùng cho cache equivalence gate (binary yes/no).

    num_ctx=2048 để đủ chỗ cho <think> tokens của deepseek-r1:8b
    (thinking có thể chiếm 200-500 tokens trước khi trả lời).
    Dùng llm_model (deepseek-r1:8b) — bỏ cache_verify_model riêng cho đơn giản.
    """
    model = settings.llm_model
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        num_gpu=settings.ollama_num_gpu,
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        num_gpu=settings.ollama_num_gpu,
    )


def parse_json(text: str) -> dict:
    """
    Extract JSON object from LLM response.
    Handles DeepSeek-R1 thinking tokens (<think>...</think>).

    Strategy: DeepSeek-R1 puts reasoning FIRST then answer LAST.
    → Strip <think> blocks first, then find the LAST valid JSON object.
    """
    # 1. Strip <think>...</think> blocks (DeepSeek-R1 reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Try parsing the whole remaining text as JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3. Find ALL {...} blocks (including multi-line), try from LAST to FIRST
    #    DeepSeek sometimes generates intermediate JSON in reasoning before the real answer
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            result = json.loads(m.group())
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    # 4. Try to find a JSON block between ```json ... ``` markers
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except Exception:
            pass

    logger.warning(f"Could not parse JSON from LLM response: {text[:300]}")
    return {}


def embed_query(query: str) -> list[float]:
    """Embed a single query string using bge-m3."""
    emb = get_embeddings()
    return emb.embed_query(query)
