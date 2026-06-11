"""Chat API routes — với hybrid cache lookup trước khi gọi graph."""
import uuid
from loguru import logger
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from src.api.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def _last_ai_message(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content
    return "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại."


def _extract_sources(docs: list) -> list[str]:
    if not docs:
        return []
    sources = set()
    for doc in docs:
        if isinstance(doc, dict):
            src = doc.get("metadata", {}).get("filename") or doc.get("source", "")
            if src:
                sources.add(src)
    return list(sources)


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from src.graph.main_graph import get_graph
    from src.history.conversation_store import get_store

    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "UID0000"
    message = request.message.strip()
    store = get_store()

    # Cache check đã được chuyển vào trong graph (cache_check_node).
    # Chat route chỉ cần invoke graph và đọc kết quả.

    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "user_id": user_id,
            },
            config=config,
        )

        messages = result.get("messages", [])
        answer = _last_ai_message(messages)
        sources = _extract_sources(result.get("retrieved_docs", []))
        intent = result.get("intent")
        advisor_domain = result.get("advisor_domain")
        collected_info = result.get("collected_info")
        cache_hit = result.get("cache_hit", False)
        cache_similarity = result.get("cache_similarity")

        if cache_hit:
            logger.info(
                f"[CACHE HIT] user={user_id} sim={cache_similarity:.3f} | "
                f"message='{message[:60]}'"
            )

        # ── Save Q&A to history ───────────────────────────────────────────────
        # Bỏ qua nếu:
        #   - Đang trong quá trình hỏi thêm thông tin (advisor collecting)
        #   - Câu trả lời đã từ cache (tránh duplicate entry)
        missing_fields = result.get("missing_fields") or []
        is_collecting = bool(missing_fields)

        if not is_collecting and not cache_hit and answer:
            store.save(
                user_id=user_id,
                question=message,
                answer=answer,
                intent=intent,
                advisor_domain=advisor_domain,
                session_id=session_id,
            )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            intent=intent,
            advisor_domain=advisor_domain,
            sources=sources,
            turn_count=result.get("turn_count", 0),
            collected_info=collected_info,
            from_cache=cache_hit,
            cache_similarity=cache_similarity,
        )

    except Exception as e:
        logger.error(f"Chat error session={session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/new")
async def new_session():
    return {"session_id": str(uuid.uuid4())}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    from src.graph.main_graph import get_graph
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        values = state.values if hasattr(state, "values") else {}
        messages = values.get("messages", [])
        history = []
        for m in messages:
            if isinstance(m, HumanMessage):
                history.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                history.append({"role": "assistant", "content": m.content})
        return {
            "session_id": session_id,
            "history": history,
            "intent": values.get("intent"),
            "advisor_domain": values.get("advisor_domain"),
            "collected_info": values.get("collected_info"),
            "missing_fields": values.get("missing_fields"),
            "turn_count": values.get("turn_count", 0),
        }
    except Exception as e:
        return {"session_id": session_id, "history": [], "error": str(e)}


@router.get("/history/{user_id}")
async def get_user_history(user_id: str, limit: int = 20):
    """Xem lịch sử hội thoại của một user."""
    from src.history.conversation_store import get_store
    store = get_store()
    entries = store.get_user_history(user_id)
    # Trả về mới nhất trước
    return {
        "user_id": user_id,
        "total": len(entries),
        "entries": list(reversed(entries))[:limit],
    }
