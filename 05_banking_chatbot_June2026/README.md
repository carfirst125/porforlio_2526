# Project 5 — Bank Customer Service Chatbot (LangGraph · RAG · Multi-turn Advisory)

## Project Overview

This project implements an AI-powered customer service chatbot for a retail bank, built entirely on a local LLM stack with no external API dependency. The system handles product inquiries, personalized multi-turn advisory conversations, customer feedback analysis, and general interactions.

Core design goals:

- **Factual grounding** — all answers are sourced from official product documents; hallucination is unacceptable in a banking context
- **Personalized advisory** — for complex products (loans, insurance, credit cards), the bot gathers customer context across turns and recommends accordingly
- **Data privacy** — all computation runs locally; no customer data leaves the organization
- **Maintainability** — updating the product knowledge base requires only swapping the parquet file and issuing a reload

The architecture centers on a **LangGraph StateGraph** with 15 nodes routing messages across five specialized pipelines based on intent classification. Version 3 (current) adds category-level retrieval filtering, a persistent per-user Advisor Profile Store, and an in-graph cache node.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph Orchestration | LangGraph (StateGraph + MemorySaver) |
| LLM | Ollama → DeepSeek-R1:8b |
| Embeddings | Ollama → bge-m3 (1024-dim) |
| Vector Store (products) | ChromaDB `vib_products_v3` (category metadata) |
| Vector Store (history/cache) | ChromaDB `conversation_history_v3` |
| Keyword Search | rank-bm25 (in-memory, global + per-category) |
| Retrieval Fusion | RRF (65% semantic + 35% BM25) |
| Advisor Profile Store | JSON files per user (`{user_id}_profiles.json`) |
| API Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Config | pydantic-settings + .env |
| Logging | loguru |

---

## Project Structure

```text
05_banking_chatbot_June2026/
└── version_3/
    ├── config/
    │   └── settings.py                    # All configuration (pydantic-settings)
    ├── src/
    │   ├── llm.py                         # LLM + Embedding factory, parse_json
    │   ├── data/
    │   │   └── loader.py                  # Parquet → ChromaDB + BM25 (category metadata)
    │   ├── retrieval/
    │   │   └── retriever.py               # hybrid_retrieve (BM25 + Semantic + RRF + category_filter)
    │   ├── knowledge_graph/
    │   │   └── field_definitions.py       # DOMAIN_FIELDS, DOMAIN_KEYWORDS, DOMAIN_LABELS
    │   ├── history/
    │   │   ├── conversation_store.py      # JSON Q&A history + ChromaDB cache index
    │   │   └── advisor_profile_store.py   # per-user collected_info per domain
    │   ├── graph/
    │   │   ├── state.py                   # ChatState TypedDict
    │   │   ├── main_graph.py              # LangGraph assembly + routing functions
    │   │   └── nodes/
    │   │       ├── intent_classifier.py
    │   │       ├── greeting.py
    │   │       ├── personal_unrelated.py
    │   │       ├── cache_check.py
    │   │       ├── customer_feedback.py
    │   │       ├── rag/
    │   │       │   ├── rewrite.py
    │   │       │   ├── retrieve.py        # detect_category → category_filter
    │   │       │   └── generate.py
    │   │       └── advisor/
    │   │           ├── domain_detector.py
    │   │           ├── profile_recall.py  # profile recall + update nodes
    │   │           ├── field_extractor.py
    │   │           ├── info_collector.py
    │   │           └── recommender.py     # category_filter + save_profile
    │   └── api/
    │       ├── main.py
    │       ├── models/schemas.py
    │       └── routes/
    │           ├── chat.py
    │           └── admin.py
    ├── frontend/
    │   └── app.py                         # Streamlit UI
    ├── evaluation/
    │   ├── data/                          # Test datasets (static, committed)
    │   │   ├── intent_samples.json
    │   │   ├── rag_samples.json
    │   │   ├── cache_pairs.json
    │   │   ├── feedback_samples.json
    │   │   └── advisory_scenarios.json
    │   ├── results/                       # Generated outputs (gitignored)
    │   ├── utils.py
    │   ├── step1_intent_eval.py
    │   ├── step1_cache_eval.py
    │   ├── step1_feedback_eval.py
    │   ├── step2_rag_eval.py
    │   ├── step3_advisory_eval.py
    │   ├── step4_recommend_eval.py
    │   └── run_all.py
    ├── data/
    │   ├── vectorstore/                   # ChromaDB (gitignored — large binary)
    │   └── conversations/                 # Runtime Q&A + profiles (gitignored)
    ├── scripts/
    │   └── check_gpu.py
    ├── requirements.txt
    ├── .env.example
    ├── start_api.bat
    ├── start_streamlit.bat
    ├── VERSION_3_PLAN.md
    ├── technote.md
    ├── userguide.md
    └── evaluation_guideline.md
```

---

## Message Processing Flow

Every incoming message flows through a **LangGraph StateGraph** with three-tier priority routing:

```
User Message + UserID + SessionID
    │
    ▼
[FastAPI POST /chat/]
    │
    └─ LangGraph graph.invoke()
           │
           ▼
       [START Router]  — three-tier priority
           │
           ├─① awaiting_profile_confirm=True
           │       → [advisor_profile_update]
           │
           ├─② active advisor session (missing_fields not empty)
           │       → [advisor_collect_info]  (multi-turn continues)
           │
           └─③ new message
                   → [intent_classifier]
                          │
                   ┌──────┴──────────────────────────────────────┐
                   │                                             │
             PRODUCT_CONSULT                          all other intents
             CUSTOMER_FEEDBACK                         → [cache_check_node]
             (both skip cache)                                │
                   │                              ┌───────────┼───────────┐
                   │                            HIT          MISS        MISS
                   │                         (≥0.80)     GREETING   INFO_QA/PERSONAL
                   │                            END           │              │
                   │                                    [greeting_node]  [rag pipeline]
                   │
                   ├─ PRODUCT_CONSULT
                   │     → [advisor_domain_detector]
                   │     → [advisor_profile_recall]
                   │          ├─ profile found → confirm → END
                   │          └─ no profile   → [field_extractor]
                   │                           → [collect_info loop]
                   │                           → [advisor_retrieve (category_filter=domain)]
                   │                           → [advisor_recommend] → save_profile → END
                   │
                   └─ CUSTOMER_FEEDBACK
                         → [customer_feedback_node]
                              NEGATIVE → apology + improvement commitment
                              POSITIVE → thank + ask if more needed
                              NEUTRAL  → acknowledge + ask for detail
                         → END
```

### Hybrid Retrieval (RAG + Advisor)

Both the RAG pipeline and advisory pipeline use the same hybrid retrieval function:

1. Semantic search via ChromaDB (bge-m3 embeddings), filtered by `category` if applicable
2. BM25 keyword search, using the per-category index when a category is detected
3. RRF fusion: `score = 0.65 / (60 + semantic_rank) + 0.35 / (60 + bm25_rank)`
4. Return top-k documents

Categories: `credit_card`, `insurance`, `loan`, `savings`, `general`

### Advisor Profile Store

After a completed advisory session, the bot persists the collected customer profile (income, goals, preferences) to `data/conversations/{user_id}_profiles.json`. On subsequent sessions for the same domain, the bot recalls the profile, shows a summary, and only re-collects fields that changed — avoiding repetitive questioning across sessions.

---

## API Layer

Run:

```bash
cd version_3
python -m uvicorn src.api.main:app --reload
# or: start_api.bat
```

Startup sequence: load parquet → index into ChromaDB with category metadata → build global + per-category BM25 indexes → bi-directional sync of conversation history (JSON ↔ ChromaDB). Takes ~30–60 seconds on first run.

Endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/admin/health` | Vectorstore status, model info, category counts |
| GET | `/admin/stats` | Chunk count, category breakdown, history stats |
| POST | `/admin/load` | Load parquet → ChromaDB |
| POST | `/admin/load?force_reload=true` | Force reload after parquet update |
| POST | `/admin/rebuild-history` | Clear + rebuild ChromaDB history from JSON files |
| POST | `/chat/` | Main chat endpoint |
| GET | `/chat/history/{user_id}` | Q&A history for a user |
| GET | `/chat/session/{session_id}` | Current session state (debug) |

Chat request/response:

```json
Request:  { "message": "tôi muốn mở thẻ tín dụng", "session_id": "abc-123", "user_id": "UID001" }
Response: { "answer": "...", "intent": "PRODUCT_CONSULT", "advisor_domain": "credit_card",
            "from_cache": false, "cache_similarity": null, "turn_count": 1 }
```

---

## Streamlit Frontend

Run:

```bash
cd version_3
python -m streamlit run frontend/app.py
# or: start_streamlit.bat
```

Access at **http://localhost:8501** · API docs at **http://localhost:8000/docs**

Features: per-user session management via UserID input, intent badges on each response (⚡ Cache, 🎯 Tư vấn, 🔍 Tra cứu, 📝 Phản hồi KH, etc.), sidebar for data load/reload, and a new-chat button.

---

## Evaluation Suite

Six-step evaluation framework covering all pipeline layers:

| Step | Component | Metric | Target |
|---|---|---|---|
| 1a | Intent Classification | Macro F1 | ≥ 0.85 |
| 1b | Cache | False Positive Rate | ≤ 0.05 |
| 1c | Feedback Sentiment | Accuracy | ≥ 0.85 |
| 2 | RAG Retrieval + Generation | Faithfulness | ≥ 0.80 |
| 3 | Advisory Field Collection | Completion Rate | ≥ 0.90 |
| 4 | Advisory Recommendation | Correct Rate | ≥ 0.70 |

Run full evaluation (API must be running):

```bash
cd version_3
python evaluation/run_all.py
```

Run individual steps:

```bash
python evaluation/step1_intent_eval.py
python evaluation/step1_cache_eval.py
python evaluation/step1_feedback_eval.py
python evaluation/step2_rag_eval.py
python evaluation/step3_advisory_eval.py
python evaluation/step4_recommend_eval.py   # uses step3 output as input
```

Results achieved on current build: Intent Macro F1 0.87, Feedback Accuracy 0.89, RAG Faithfulness 0.82, Advisory Completion 0.94, Recommendation Correct Rate 0.75. Cache FPR at 0.07 (WARN — slightly above 0.05 target).

---

## Configuration

All settings via `.env` (copy from `.env.example`):

| Key | Default | Description |
|---|---|---|
| `LLM_MODEL` | `deepseek-r1:8b` | LLM for all tasks (classification, RAG, advisory) |
| `EMBEDDING_MODEL` | `bge-m3:latest` | Embedding model (1024-dim) |
| `OLLAMA_NUM_GPU` | `-1` | -1 = all GPU layers, 0 = CPU only |
| `LLM_NUM_CTX` | `4096` | Context window (increase for long advisory sessions) |
| `CHROMA_COLLECTION_NAME` | `vib_products_v3` | Product knowledge collection |
| `TOP_K_RETRIEVAL` | `6` | Docs fetched for RAG |
| `TOP_K_FINAL` | `4` | Docs fetched for advisory |
| `SEMANTIC_WEIGHT` | `0.65` | RRF semantic weight |
| `BM25_WEIGHT` | `0.35` | RRF BM25 weight |
| `CACHE_SIMILARITY_THRESHOLD` | `0.8` | Threshold for cache hit |
| `MAX_ADVISOR_TURNS` | `8` | Max turns in advisory collection loop |
| `EVAL_JUDGE_MODEL` | `""` | LLM judge model for evaluation (empty = use LLM_MODEL) |

---

## Quick Start

Prerequisites: Python 3.10+, Ollama running with `deepseek-r1:8b` and `bge-m3` pulled, `documents_bgem3.parquet` at `../documents_bgem3.parquet`.

```bash
cd version_3

# Install dependencies
pip install -r requirements.txt

# Configure
copy .env.example .env

# Terminal 1 — start API
python -m uvicorn src.api.main:app --reload

# Terminal 2 — start Streamlit
python -m streamlit run frontend/app.py
```

---

## Notes

- **DeepSeek-R1 think tokens**: The model emits `<think>...</think>` blocks before its answer. `parse_json()` strips these before parsing structured output. The `get_fast_llm()` cache-check function uses `num_ctx=2048` to accommodate reasoning tokens.
- **`EVAL_JUDGE_MODEL`**: For evaluation, a non-reasoning model (e.g. `qwen2.5:7b`) is recommended as judge — reasoning models are slower and can trigger 500 errors in the eval HTTP client.
- **In-graph cache vs. API-level cache**: Moving the cache check inside the graph allows PRODUCT_CONSULT and CUSTOMER_FEEDBACK to bypass it cleanly, and keeps cache logic testable as a first-class graph node.
- **Category filtering vs. re-ranking**: Pre-filtering at retrieval time (ChromaDB `where` clause + per-category BM25) is cheaper and more deterministic than post-hoc re-ranking across the full corpus.
- **JSON files for profiles/history**: Chosen over a database for simplicity and direct inspectability. The bi-directional startup sync (`load_all_history()`) keeps ChromaDB consistent with JSON state across restarts without requiring a separate migration step.
