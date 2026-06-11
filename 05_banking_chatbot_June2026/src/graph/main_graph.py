"""
Main LangGraph assembly — V3 intent-based architecture.

Routing logic:
  START → priority check:
    ├── awaiting_profile_confirm → advisor_profile_update → retrieve/collect_info
    ├── active advisor session   → advisor_collect_info (loop)
    └── new message → intent_classifier
          ├── PRODUCT_CONSULT   → advisor_domain_detector          [skip cache]
          │       → advisor_profile_recall
          │           ├── [saved profile] → confirm msg → END (chờ user)
          │           └── [no profile]    → field_extractor → collect_info (loop)
          │       → advisor_retrieve → advisor_recommend → END
          └── other intents → cache_check
                  ├── [cache HIT]  → END (trả về cached answer)
                  └── [cache MISS] → greeting / personal_unrelated / rag pipeline
                                   → customer_feedback → END
"""
from loguru import logger
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import ChatState
from src.graph.nodes.intent_classifier import intent_classifier_node
from src.graph.nodes.greeting import greeting_node
from src.graph.nodes.personal_unrelated import personal_unrelated_node
from src.graph.nodes.customer_feedback import customer_feedback_node
from src.graph.nodes.rag.rewrite import rag_rewrite_node
from src.graph.nodes.rag.retrieve import rag_retrieve_node
from src.graph.nodes.rag.generate import rag_generate_node
from src.graph.nodes.cache_check import cache_check_node, route_after_cache_check
from src.graph.nodes.advisor.domain_detector import advisor_domain_detector_node
from src.graph.nodes.advisor.field_extractor import advisor_field_extractor_node
from src.graph.nodes.advisor.info_collector import (
    advisor_collect_info_node,
    route_after_collect_info,
)
from src.graph.nodes.advisor.recommender import advisor_retrieve_node, advisor_recommend_node
from src.graph.nodes.advisor.profile_recall import (
    advisor_profile_recall_node,
    advisor_profile_update_node,
    route_after_profile_recall,
    route_after_profile_update,
)


# ── Routing functions ────────────────────────────────────────────────────────

def route_from_start(state: ChatState) -> str:
    """
    Priority routing:
    1. awaiting_profile_confirm → user đang confirm/update profile đã lưu
    2. active advisor session (missing fields) → tiếp tục hỏi thông tin
    3. otherwise → intent classification
    """
    if state.get("awaiting_profile_confirm"):
        logger.debug("Awaiting profile confirm → advisor_profile_update")
        return "advisor_profile_update"
    if state.get("required_fields") and state.get("missing_fields"):
        logger.debug("Resuming active advisor session → advisor_collect_info")
        return "advisor_collect_info"
    return "intent_classifier"


def route_by_intent(state: ChatState) -> str:
    """
    PRODUCT_CONSULT   → advisor (bỏ qua cache — tư vấn phụ thuộc profile KH).
    CUSTOMER_FEEDBACK → customer_feedback (bỏ qua cache — phản hồi cá nhân).
    Tất cả intent còn lại → cache_check trước.
    """
    intent = state.get("intent", "PRODUCT_INFO_QA")
    if intent == "PRODUCT_CONSULT":
        logger.debug("Intent=PRODUCT_CONSULT → advisor_domain_detector (skip cache)")
        return "advisor_domain_detector"
    if intent == "CUSTOMER_FEEDBACK":
        logger.debug("Intent=CUSTOMER_FEEDBACK → customer_feedback (skip cache)")
        return "customer_feedback"
    logger.debug(f"Intent={intent} → cache_check")
    return "cache_check"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(use_memory: bool = True):
    builder = StateGraph(ChatState)

    # ── Register nodes ───────────────────────────────────────────────────────
    builder.add_node("intent_classifier", intent_classifier_node)
    builder.add_node("cache_check", cache_check_node)
    builder.add_node("greeting", greeting_node)
    builder.add_node("personal_unrelated", personal_unrelated_node)
    builder.add_node("customer_feedback", customer_feedback_node)
    builder.add_node("rag_rewrite", rag_rewrite_node)
    builder.add_node("rag_retrieve", rag_retrieve_node)
    builder.add_node("rag_generate", rag_generate_node)
    builder.add_node("advisor_domain_detector", advisor_domain_detector_node)
    builder.add_node("advisor_profile_recall", advisor_profile_recall_node)   # NEW
    builder.add_node("advisor_profile_update", advisor_profile_update_node)   # NEW
    builder.add_node("advisor_field_extractor", advisor_field_extractor_node)
    builder.add_node("advisor_collect_info", advisor_collect_info_node)
    builder.add_node("advisor_retrieve", advisor_retrieve_node)
    builder.add_node("advisor_recommend", advisor_recommend_node)

    # ── Edges ────────────────────────────────────────────────────────────────

    # START: priority routing
    builder.add_conditional_edges(
        START,
        route_from_start,
        {
            "intent_classifier": "intent_classifier",
            "advisor_collect_info": "advisor_collect_info",
            "advisor_profile_update": "advisor_profile_update",
        },
    )

    # Intent classifier → PRODUCT_CONSULT/CUSTOMER_FEEDBACK bypass cache; others go through cache_check
    builder.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "cache_check": "cache_check",
            "advisor_domain_detector": "advisor_domain_detector",
            "customer_feedback": "customer_feedback",
        },
    )

    # Cache check → HIT: END | MISS: route by intent to actual node
    builder.add_conditional_edges(
        "cache_check",
        route_after_cache_check,
        {
            "__end__": END,
            "greeting": "greeting",
            "personal_unrelated": "personal_unrelated",
            "rag_rewrite": "rag_rewrite",
            "customer_feedback": "customer_feedback",
        },
    )

    # Terminal nodes → END
    builder.add_edge("greeting", END)
    builder.add_edge("personal_unrelated", END)
    builder.add_edge("customer_feedback", END)

    # RAG pipeline
    builder.add_edge("rag_rewrite", "rag_retrieve")
    builder.add_edge("rag_retrieve", "rag_generate")
    builder.add_edge("rag_generate", END)

    # Advisory pipeline
    # domain_detector → profile_recall → (has profile? END : field_extractor)
    builder.add_edge("advisor_domain_detector", "advisor_profile_recall")
    builder.add_conditional_edges(
        "advisor_profile_recall",
        route_after_profile_recall,
        {
            "__end__": END,                              # profile found → show confirm → wait
            "advisor_field_extractor": "advisor_field_extractor",  # no profile → normal flow
        },
    )

    # profile_update → (missing fields? collect_info : retrieve)
    builder.add_conditional_edges(
        "advisor_profile_update",
        route_after_profile_update,
        {
            "advisor_collect_info": "advisor_collect_info",
            "advisor_retrieve": "advisor_retrieve",
        },
    )

    # field_extractor pre-extracts info from initial question → always goes to collect_info
    # collect_info decides: ask first question (if missing) OR proceed to retrieval (if all pre-filled)
    builder.add_edge("advisor_field_extractor", "advisor_collect_info")

    # Info collector: multi-turn loop
    builder.add_conditional_edges(
        "advisor_collect_info",
        route_after_collect_info,
        {
            "__end__": END,                  # still collecting → return question to user
            "advisor_retrieve": "advisor_retrieve",  # done collecting → retrieve
        },
    )

    builder.add_edge("advisor_retrieve", "advisor_recommend")
    builder.add_edge("advisor_recommend", END)

    # ── Compile ──────────────────────────────────────────────────────────────
    if use_memory:
        graph = builder.compile(checkpointer=MemorySaver())
    else:
        graph = builder.compile()

    logger.info("LangGraph V2 compiled successfully.")
    return graph


# ── Singletons ───────────────────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(use_memory=True)
    return _graph
