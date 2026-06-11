# VIB Chatbot V3 — Technical Note

**Version:** 3.2  
**Stack:** LangGraph · DeepSeek-R1 · bge-m3 · ChromaDB · FastAPI · Streamlit  
**Date:** 2026-06

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Layer — Category Metadata & Initialization](#2-data-layer)
3. [Message Processing Pipeline — Step by Step](#3-message-processing-pipeline)
   - 3.1 [START Router — Priority Routing](#31-start-router)
   - 3.2 [Intent Classifier](#32-intent-classifier)
   - 3.3 [Cache Check Node](#33-cache-check-node)
   - 3.4 [GREETING_FAREWELL Pipeline](#34-greeting_farewell-pipeline)
   - 3.5 [PERSONAL_UNRELATED Pipeline](#35-personal_unrelated-pipeline)
   - 3.6 [PRODUCT_INFO_QA — RAG Pipeline](#36-product_info_qa--rag-pipeline)
   - 3.7 [PRODUCT_CONSULT — Advisory Pipeline](#37-product_consult--advisory-pipeline)
   - 3.8 [Advisor Profile Recall & Update](#38-advisor-profile-recall--update)
   - 3.9 [CUSTOMER_FEEDBACK Pipeline](#39-customer_feedback-pipeline)
4. [Hybrid Retrieval — Algorithm Detail](#4-hybrid-retrieval)
5. [Category Filtering](#5-category-filtering)
6. [Advisor Profile Store](#6-advisor-profile-store)
7. [Conversation History & Cache](#7-conversation-history--cache)
8. [LangGraph State & Topology](#8-langgraph-state--topology)
9. [API Layer](#9-api-layer)
10. [Full Data Flow Diagram](#10-full-data-flow-diagram)
11. [Key Configuration](#11-key-configuration)
12. [Extension Points](#12-extension-points)

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                      Streamlit UI                          │
│  UserID · Chat history · Intent badge · Cache badge        │
└──────────────────────┬─────────────────────────────────────┘
                       │ HTTP POST /chat/  {message, session_id, user_id}
                       ▼
┌────────────────────────────────────────────────────────────┐
│                     FastAPI Server                         │
│                                                            │
│  graph.invoke({messages, session_id, user_id})             │
│    └─ LangGraph handles everything (cache, classify, pipeline) │
│                                                            │
│  After graph returns:                                      │
│    ConversationStore.save() — if final answer              │
└──────────────────────┬─────────────────────────────────────┘
                       │
          ┌────────────┼──────────────┐
          ▼            ▼              ▼
     ChromaDB      JSON files    BM25 indexes
  vib_products_v3  conversations/  global + per-category
  conv_history_v3  {uid}_profiles.json
```

**ChromaDB collections:**

| Collection | Contents | Purpose |
|---|---|---|
| `vib_products_v3` | Chunks from parquet, pre-embedded with bge-m3, with `category` metadata | RAG/Advisor retrieval — filter by category |
| `conversation_history_v3` | Answered Q&A (all users) | Cache lookup — avoid repeated LLM calls |

**BM25 indexes (in-memory, rebuilt on every startup):**

| Index | Corpus | Used for |
|---|---|---|
| Global BM25 | All chunks | RAG fallback (category=general) |
| BM25[credit_card] | Chunks category=credit_card | Credit card advisory |
| BM25[insurance] | Chunks category=insurance | Insurance advisory |
| BM25[loan] | Chunks category=loan | Loan advisory |
| BM25[savings] | Chunks category=savings | Savings advisory |

**JSON storage (data/conversations/):**

| File | Contents |
|---|---|
| `{user_id}.json` | Q&A history per user |
| `{user_id}_profiles.json` | Advisor collected_info per domain (V3 new) |

---

## 2. Data Layer

### 2.1 Category Detection

When loading the parquet file, each chunk is assigned a `category` via keyword matching:

```python
CATEGORY_KEYWORDS = {
    "credit_card": ["thẻ tín dụng", "thẻ ghi nợ", "cashback", "hoàn tiền",
                    "tích điểm", "hạn mức thẻ", "super card", "dặm bay", ...],
    "insurance":   ["bảo hiểm nhân thọ", "bảo hiểm sức khỏe", "phí bảo hiểm",
                    "quyền lợi bảo hiểm", "hợp đồng bảo hiểm", ...],
    "loan":        ["vay mua nhà", "vay tiêu dùng", "lãi suất vay",
                    "thế chấp", "tín chấp", "khoản vay", ...],
    "savings":     ["tiết kiệm", "lãi suất tiết kiệm", "gửi tiết kiệm",
                    "kỳ hạn gửi", "sổ tiết kiệm", ...],
}
# No match → "general"
```

Logic: Iterate in order — credit_card → insurance → loan → savings. Return the first matching category, or "general" if none match.

### 2.2 Parquet → ChromaDB (startup)

```
load_data()
  → read parquet (columns: input, embedding)
  → parse embedding: json.loads(raw.strip('"')) → list[float]
  → detect_category(text) → metadata = {"category": cat}
  → batch add to ChromaDB collection vib_products_v3
      collection.add(ids, documents, embeddings, metadatas)
  → build global BM25Okapi + per-category BM25Okapi
  → cache in singletons: _collection, _bm25, _bm25_by_category, _texts_by_category
```

Idempotent: if the collection already contains documents → skip reload (unless `force_reload=true`).

### 2.3 Startup — BM25 Rebuild

On server restart, the ChromaDB collection persists but BM25 indexes (in-memory) are lost. `_build_bm25_from_collection()` re-reads from ChromaDB, re-detects categories (in case metadata is missing), and rebuilds the indexes.

---

## 3. Message Processing Pipeline

### 3.1 START Router

**Location:** `src/graph/main_graph.py` — `route_from_start()`

Three priority tiers:

```
1. awaiting_profile_confirm=True
   → advisor_profile_update
   (user is responding after the bot displayed their saved profile)

2. required_fields NOT NULL AND missing_fields NOT EMPTY
   → advisor_collect_info
   (continue multi-turn info collection in the current session)

3. else
   → intent_classifier
```

### 3.2 Intent Classifier

**Location:** `src/graph/nodes/intent_classifier.py`

```
Input: last user message + history (4 turns × 800 chars)
  ↓
LLM (DeepSeek-R1, temperature=0.0) — few-shot classification
  ↓
Output: intent ∈ {GREETING_FAREWELL, PERSONAL_UNRELATED, PRODUCT_INFO_QA,
                  PRODUCT_CONSULT, CUSTOMER_FEEDBACK}
  Fallback: PRODUCT_INFO_QA if parse fails
```

After the intent is determined, `route_by_intent()` routes to:
- `PRODUCT_CONSULT` → `advisor_domain_detector` (bypass cache)
- `CUSTOMER_FEEDBACK` → `customer_feedback` (bypass cache — response is personal/contextual)
- All other intents → `cache_check`

### 3.3 Cache Check Node

**Location:** `src/graph/nodes/cache_check.py`

Cache check is a **node inside the graph**, applied only to non-PRODUCT_CONSULT, non-CUSTOMER_FEEDBACK intents.

```
Input: last user message
  ↓
ConversationStore.search_similar(question)
  ↓
HIT (score ≥ 0.8):
  → set messages=[AIMessage(cached_answer)], cache_hit=True
  → END (no further processing)

MISS:
  → cache_hit=False
  → route_after_cache_check → greeting / personal_unrelated / rag_rewrite
                             → customer_feedback  (if intent=CUSTOMER_FEEDBACK)
```

**Intents that bypass cache entirely (do not go through cache_check_node):**
- `PRODUCT_CONSULT`: multi-turn advisory depending on the specific customer's profile — cached answers from other users are meaningless.
- `CUSTOMER_FEEDBACK`: responses are personal and depend on conversation context — should not be cached.

### 3.4 GREETING_FAREWELL Pipeline

```
cache_check → MISS
    ↓
greeting_node
    │
    ├─ _is_farewell(text)?  ← keyword matching before calling LLM
    │   Keywords: "tạm biệt", "bye", "ok bye", "ok, bye", "hẹn gặp lại", ...
    │
    ├─ YES (farewell) → FAREWELL_PROMPT → LLM (temp=0.3)
    │       → Thank customer + wish them well (static fallback if LLM fails)
    │
    └─ NO (greeting)  → GREETING_PROMPT → LLM (temp=0.3)
            → Greet + introduce bot + ask how to help
    → END
```

**Why keyword pre-classification:** With a single conditional prompt, the LLM tends to ignore "if farewell then..." instructions and always generates a greeting-style response. Keyword matching is deterministic and does not depend on LLM judgment.

### 3.5 PERSONAL_UNRELATED Pipeline

```
cache_check → MISS
    ↓
personal_unrelated_node → LLM (temp=0.4)
    Prompt in 3 steps:
    1. Brief empathy (1 sentence)
    2. Natural pivot to the most relevant VIB product
    3. Ask if the customer is interested
    → END
```

### 3.9 CUSTOMER_FEEDBACK Pipeline

**Location:** `src/graph/nodes/customer_feedback.py`

This intent **bypasses cache entirely** — responses are personal and context-dependent.

```
(bypass cache)
    ↓
customer_feedback_node
  Input: last user message + last 4 conversation turns
  LLM (temp=0.3) analyzes:
    sentiment ∈ {NEGATIVE, POSITIVE, NEUTRAL}
    subject: about the chatbot/response or about a VIB product
  ↓
  NEGATIVE (frustrated, dissatisfied, complaining):
    → Sincere apology, show empathy
    → Commit to acknowledging and improving
    → For product complaints: direct to hotline/branch if needed
    → Tone: gentle, receptive

  POSITIVE (praise, satisfied):
    → Sincere, humble thanks
    → Ask if the customer needs any further help

  NEUTRAL (neutral, unclear):
    → Acknowledge, ask for more detail
  → END
```

**Examples:**
```
Customer: "Why did you ask me about my loan purpose when I already said I want to buy a house?"
→ CUSTOMER_FEEDBACK, NEGATIVE

Bot: "I sincerely apologize for the inconvenience! You are absolutely right —
     you clearly stated you wanted to buy a house from the start, and I asked again anyway.
     I will take note of this to improve. Would you like me to advise you on a suitable
     home loan package right away?"

Customer: "The bot's answers are very helpful and easy to understand!"
→ CUSTOMER_FEEDBACK, POSITIVE

Bot: "Thank you so much for the kind words! I still have a lot to learn
     and will keep striving to serve you better. Is there anything else I can help you with?"
```

### 3.6 PRODUCT_INFO_QA — RAG Pipeline

```
cache_check → MISS
    ↓
[rag_rewrite_node]
  Input: question + history (4 turns × 800 chars)
  LLM (temp=0.0): clarify question, add domain keywords
  Output: rewritten_query
    ↓
[rag_retrieve_node]
  detect_category(rewritten_query) → category_filter
    category ∈ {credit_card, insurance, loan, savings} → apply filter
    category = "general" → no filter (search entire corpus)
  hybrid_retrieve(query=rewritten_query, top_k=6, category_filter=cat)
  Output: retrieved_docs (top-6, with category field)
    ↓
[rag_generate_node]
  Context = format_context(docs) — max 6000 chars
  LLM (temp=0.05), STRICT grounding:
    ❌ Do not fabricate numbers, rates, products
    → Not found: "not found" + hotline 1800 8180
    → END
```

### 3.7 PRODUCT_CONSULT — Advisory Pipeline

```
(bypass cache)
    ↓
[advisor_domain_detector]
  LLM (temp=0.0)
  Output: domain ∈ {credit_card, insurance, loan, savings, general}
    ↓
[advisor_profile_recall]  ← V3 NEW
  See section 3.8
    ↓ (if no profile)
[advisor_field_extractor]
  get_fields(domain) → required_fields dict
  LLM (temp=0.0, num_ctx=8192): pre-extract info from initial question
  Output: {required_fields, collected_info=pre_filled, missing_fields, turn_count=0}
    ↓
[advisor_collect_info_node]  ← MULTI-TURN LOOP
  turn_count=0:
    missing_fields=[] → "searching now" → advisor_retrieve
    missing_fields!=[] → ask first missing field (with intro) → END
  turn_count>0:
    save last_user_msg → collected_info[missing_fields[0]]
    update missing_fields
    still missing → ask next → END
    all collected or max turns → advisor_retrieve
    ↓
[advisor_retrieve_node]
  query = "{domain_keyword} {collected_info values}"
  hybrid_retrieve(top_k=4, domain_hint=domain_kw, category_filter=domain)
  Output: retrieved_docs (top-4, scoped to domain category)
    ↓
[advisor_recommend_node]
  _docs_are_relevant(domain, docs) → False: NOT_FOUND_TEMPLATE + hotline
  LLM (temp=0.1), STRICT grounding → recommendation
  → save_profile(user_id, domain, collected_info)  ← save profile
  → END
```

### 3.8 Advisor Profile Recall & Update

#### Node 1: `advisor_profile_recall_node`

Inserted between `domain_detector` and `field_extractor`.

```
Input: user_id, domain
  ↓
AdvisorProfileStore.load_profile(user_id, domain)
  ↓
Profile exists:
  → set collected_info=profile, required_fields, missing_fields=[]
  → set awaiting_profile_confirm=True
  → messages=[AIMessage(confirm_message)]
  → END (wait for user response)

No profile:
  → return {}  → route to advisor_field_extractor
```

**Confirm message format:**

```
"I see that last time you consulted about {domain_label} and provided:
 - {field_label}: {value}
 - ...
Is this information still accurate?
- If yes, just say "OK" and I'll give you a recommendation right away!
- If anything has changed, just let me know."
```

#### Node 2: `advisor_profile_update_node`

Called when `awaiting_profile_confirm=True` (highest priority at the START router).

```
Input: current collected_info + user_response (last HumanMessage)
  ↓
LLM (temp=0.0) extracts:
  {
    "confirmed": true/false,
    "updates": {field: new_value, ...}
  }
  ↓
confirmed=False:
  → reset: collected_info={}, missing_fields=all_fields
  → awaiting_profile_confirm=False, turn_count=0
  → route: advisor_collect_info

confirmed=True, updates={}:
  → "Info is the same, let me advise you right away!"
  → awaiting_profile_confirm=False
  → route: advisor_retrieve

confirmed=True, updates={...}:
  → merge updates into collected_info
  → AdvisorProfileStore.update_profile(user_id, domain, updates)
  → "Updated (fields). Let me advise you right away!"
  → awaiting_profile_confirm=False
  → check missing_fields → route: advisor_collect_info or advisor_retrieve
```

---

## 4. Hybrid Retrieval

**Location:** `src/retrieval/retriever.py` — `hybrid_retrieve()`

```
Input: query, top_k, domain_hint (optional), category_filter (optional)

Step 1: Enrich query
  search_query = "{domain_hint} {query}".strip()

Step 2: Semantic Search (ChromaDB)
  query_vec = bge-m3.embed(search_query)
  IF category_filter:
    where_clause = {"category": category_filter}
  results = collection.query(
      query_embeddings=[query_vec],
      n_results=fetch_k,            # top_k × 2
      where=where_clause,
      include=["documents", "distances", "metadatas"]
  )

Step 3: BM25 Keyword Search
  IF category_filter:
    bm25  = _bm25_by_category[category_filter]
    texts = _texts_by_category[category_filter]
  ELSE:
    bm25, texts = global_bm25, all_texts
  scores = bm25.get_scores(tokenize(search_query))
  top_bm25 = argsort(scores)[-fetch_k:]

Step 4: RRF Fusion
  score[doc] += semantic_weight (0.65) / (60 + semantic_rank)
  score[doc] += bm25_weight (0.35)    / (60 + bm25_rank)
  sort descending

Step 5: Return top_k docs
  [{content, score_rank, category}]
```

---

## 5. Category Filtering

### Motivation

V2 searched the entire corpus (~5000+ chunks). When advising on credit cards, chunks about insurance or loans were also returned due to semantic overlap with generic terms like "bank", "VIB", "product".

V3 assigns a `category` metadata tag to every chunk. Both the ChromaDB query and BM25 are filtered — searching only within the correct category.

### Impact

| Scenario | V2 | V3 |
|---|---|---|
| Advisor consulting on cards | Search 5000+ chunks | Search only credit_card chunks |
| RAG query about loan rates | Search 5000+ chunks | Search only loan chunks |
| RAG general question | Search 5000+ chunks | Still searches all (no filter) |

### Where category_filter is applied

```
Indexing (loader.py):
  detect_category(text) → metadatas=[{"category": cat}] → ChromaDB

RAG (rag/retrieve.py):
  detect_category(rewritten_query) → category_filter → hybrid_retrieve(...)

Advisor (advisor/recommender.py):
  domain ("credit_card" / "loan" / ...) → category_filter=domain → hybrid_retrieve(...)
```

---

## 6. Advisor Profile Store

**Location:** `src/history/advisor_profile_store.py`

### Storage format

```
data/conversations/{user_id}_profiles.json
{
  "user_id": "UID001",
  "profiles": {
    "credit_card": {
      "thu_nhap_hang_thang":  "15 million",
      "muc_chi_tieu_chu_yeu": "online shopping, dining",
      "uu_tien_quyen_loi":    "cashback",
      "co_the_hien_tai":      "none",
      "_updated_at":          "2026-06-07T14:00:00"
    },
    "loan": {
      "muc_dich_vay": "home purchase",
      ...
    }
  }
}
```

### API

```python
save_profile(user_id, domain, collected_info)
  # Overwrites the entire profile for the domain
  # Called by: advisor_recommend_node after recommendation is complete

load_profile(user_id, domain) → dict | None
  # Returns field dict (without _updated_at), or None if no profile exists
  # Called by: advisor_profile_recall_node

update_profile(user_id, domain, updates)
  # Merges updates into the existing profile
  # Called by: advisor_profile_update_node when user changes specific fields
```

### Lifecycle

```
1. First advisory session for credit_card:
   → field_extractor → collect_info → retrieve → recommend
   → after recommend: save_profile(uid, "credit_card", collected_info)

2. Next time the same user asks about credit_card:
   → profile_recall: load_profile(uid, "credit_card") → found
   → display confirm message → END

3. User replies "OK" or provides updates:
   → profile_update: merge updates if any → advisor_retrieve (no re-asking)

4. User wants to start over ("no, let me answer again"):
   → profile_update: confirmed=False → field_extractor → fresh collect_info
```

---

## 7. Conversation History & Cache

### Q&A Storage

```
data/conversations/{user_id}.json
{
  "user_id": "UID001",
  "entries": [{
    "id": "uuid4",
    "timestamp": "2026-06-03T19:15:08",
    "question": "lãi suất thẻ VIB Super Card?",
    "answer": "According to the document...",
    "intent": "PRODUCT_INFO_QA",
    "advisor_domain": null,
    "session_id": "abc-123"
  }]
}
```

### ChromaDB History

```
Collection: "conversation_history_v3"
  document : question text
  embedding: bge-m3.embed(question)
  metadata : {answer, user_id, intent, advisor_domain, timestamp, entry_id}
```

### When to save

```python
# chat.py — after graph invoke:
is_collecting = bool(result.get("missing_fields"))  # still collecting info

if not is_collecting and not cache_hit and answer:
    store.save(user_id, question, answer, intent, domain, session_id)
```

Not saved if:
- In the middle of advisor multi-turn (no final answer yet)
- Answer came from cache (avoids duplicates)

### Startup: load_all_history() — Bi-directional Sync

On server startup, `ConversationStore.load_all_history()` performs a **two-way sync** between JSON files and ChromaDB:

```
1. Collect all valid entry IDs from JSON files
   (skip _profiles.json, skip entries missing question/answer)

2. Compare with IDs currently in ChromaDB:
   stale_ids = chroma_ids - json_valid_ids
   → col.delete(ids=stale_ids)   ← remove entries deleted from JSON

3. entries_to_add = json_valid_ids - chroma_ids
   → embed + col.add(...)        ← add new entries not yet indexed
```

**Previously (append-only):** Deleting an entry from JSON and restarting the server left that entry in ChromaDB — the cache still returned results from the deleted entry.

**Now (bi-directional):** Server restart automatically cleans up stale entries.

**Force rebuild without restarting:**

```bash
POST /admin/rebuild-history
# → Clears all ChromaDB history entries
# → Rebuilds from scratch based on current JSON files
# → Returns before/after stats for verification
```

Use this when you need to sync immediately after manually editing JSON files without wanting to restart the server.

---

## 8. LangGraph State & Topology

### ChatState (TypedDict)

```python
class ChatState(TypedDict):
    # ── Core ─────────────────────────────────────────────────────────────
    messages             : Annotated[list, add_messages]
    session_id           : str
    user_id              : Optional[str]        # V3: load/save profile per user

    # ── Intent ───────────────────────────────────────────────────────────
    intent               : Optional[str]

    # ── RAG ──────────────────────────────────────────────────────────────
    rewritten_query      : Optional[str]
    retrieved_docs       : Optional[list]       # [{content, score_rank, category}]

    # ── Advisor ──────────────────────────────────────────────────────────
    advisor_domain       : Optional[str]
    required_fields      : Optional[dict]       # {field_name: question_to_ask}
    collected_info       : Optional[dict]       # {field_name: user_answer}
    missing_fields       : Optional[list]
    awaiting_profile_confirm : Optional[bool]   # V3: True while waiting for user to confirm profile

    # ── Cache ─────────────────────────────────────────────────────────────
    cache_hit            : Optional[bool]
    cache_similarity     : Optional[float]

    # ── Session ───────────────────────────────────────────────────────────
    turn_count           : int
    max_turns            : int                  # default 8
```

### Graph Topology

Nodes (15 nodes):

```
intent_classifier
cache_check
greeting
personal_unrelated
customer_feedback           ← V3 NEW
rag_rewrite
rag_retrieve
rag_generate
advisor_domain_detector
advisor_profile_recall      ← V3 NEW
advisor_profile_update      ← V3 NEW
advisor_field_extractor
advisor_collect_info
advisor_retrieve
advisor_recommend
```

Edges:

```
START → [route_from_start] → intent_classifier
                           → advisor_collect_info    (active advisor session)
                           → advisor_profile_update  (awaiting_profile_confirm)

intent_classifier → [route_by_intent] → cache_check
                                      → advisor_domain_detector (PRODUCT_CONSULT, skip cache)
                                      → customer_feedback       (CUSTOMER_FEEDBACK, skip cache)

cache_check → [route_after_cache_check] → END          (cache HIT)
                                        → greeting
                                        → personal_unrelated
                                        → rag_rewrite
                                        → customer_feedback

greeting → END
personal_unrelated → END
rag_rewrite → rag_retrieve → rag_generate → END

advisor_domain_detector → advisor_profile_recall
advisor_profile_recall → [route_after_profile_recall] → END                     (has profile)
                                                       → advisor_field_extractor (no profile)

advisor_profile_update → [route_after_profile_update] → advisor_collect_info (missing fields)
                                                       → advisor_retrieve      (complete)

advisor_field_extractor → advisor_collect_info
advisor_collect_info → [route_after_collect_info] → END              (still collecting)
                                                   → advisor_retrieve (done)

advisor_retrieve → advisor_recommend → END
```

### MemorySaver

State persists in-memory per `session_id` (thread_id in LangGraph). Server restart → advisor multi-turn state is lost (but JSON history and profiles are not lost — reloaded as needed).

To persist across restarts, use SqliteSaver (see section [Extension Points](#12-extension-points)).

---

## 9. API Layer

### Endpoints

```
GET  /                              — basic health check
GET  /admin/health                  — vectorstore ready, model info, category counts
GET  /admin/stats                   — chunk count, category breakdown, history stats
POST /admin/load                    — load/reload parquet → ChromaDB
POST /admin/load?force_reload=true  — force reload (needed after replacing parquet)
GET  /admin/history/{user_id}       — Q&A history for a user (admin view)
POST /admin/rebuild-history         — clear ChromaDB history + rebuild from JSON files

POST /chat/                         — main chat endpoint
GET  /chat/session/{session_id}     — current session state
GET  /chat/history/{user_id}        — user chat history
POST /chat/new                      — create new session
```

### POST /chat/ — Request/Response

```json
Request:
{
  "message"    : "tôi muốn mở thẻ tín dụng",
  "session_id" : "abc-123",
  "user_id"    : "UID001"
}

Response:
{
  "session_id"       : "abc-123",
  "answer"           : "To recommend a suitable card...",
  "intent"           : "PRODUCT_CONSULT",
  "advisor_domain"   : "credit_card",
  "sources"          : [],
  "turn_count"       : 1,
  "collected_info"   : {"thu_nhap_hang_thang": "15 million"},
  "from_cache"       : false,
  "cache_similarity" : null
}
```

### Processing in chat.py

```python
result = graph.invoke(
    {"messages": [HumanMessage(content=message)], "session_id": session_id, "user_id": user_id},
    config={"configurable": {"thread_id": session_id}},
)

# Save Q&A if there is a final answer and not still collecting
missing_fields = result.get("missing_fields") or []
is_collecting = bool(missing_fields)
if not is_collecting and not cache_hit and answer:
    store.save(user_id, question, answer, intent, advisor_domain, session_id)
```

---

## 10. Full Data Flow Diagram

```
USER MESSAGE
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ LangGraph Graph                                                  │
│                                                                  │
│  START → route_from_start()                                      │
│    ├─ awaiting_profile_confirm → [advisor_profile_update]        │
│    │    ├─ confirmed, no change → advisor_retrieve               │
│    │    ├─ confirmed + updates → merge → advisor_retrieve        │
│    │    └─ reset → advisor_collect_info                          │
│    │                                                             │
│    ├─ active advisor → [advisor_collect_info]                    │
│    │    ├─ missing → ask → END                                   │
│    │    └─ complete → advisor_retrieve → advisor_recommend → END │
│    │                                                             │
│    └─ new message → [intent_classifier]                          │
│         │                                                        │
│         ├─ PRODUCT_CONSULT (no cache)                            │
│         │    → [domain_detector]                                 │
│         │    → [profile_recall]                                  │
│         │         ├─ HAS profile → confirm → END                 │
│         │         └─ NO profile → [field_extractor]             │
│         │              → [collect_info loop]                     │
│         │              → [retrieve(category_filter=domain)]      │
│         │              → [recommend] → save_profile → END        │
│         │                                                        │
│         ├─ CUSTOMER_FEEDBACK (no cache)                          │
│         │    → [customer_feedback]                               │
│         │         NEGATIVE → apologize, empathize, commit        │
│         │         POSITIVE → thank, humble                       │
│         │         NEUTRAL  → acknowledge, ask more               │
│         │    → END                                               │
│         │                                                        │
│         └─ others → [cache_check]                                │
│              ├─ HIT (≥0.8) → END                                 │
│              └─ MISS                                             │
│                   ├─ GREETING → [greeting]                       │
│                   │    _is_farewell()? → FAREWELL/GREETING prompt│
│                   │    → END                                     │
│                   ├─ PERSONAL → [personal_unrelated] → END      │
│                   └─ INFO_QA → [rewrite]                        │
│                              → [retrieve(category_filter=cat)]  │
│                              → [generate] → END                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                    store.save(Q&A) if final answer
                               │
                    JSON + ChromaDB history
```

---

## 11. Key Configuration

| `.env` key | Default | Description |
|---|---|---|
| `LLM_MODEL` | `deepseek-r1:8b` | LLM model |
| `OLLAMA_NUM_GPU` | `-1` | -1=all GPU, 0=CPU only |
| `LLM_NUM_CTX` | `4096` | Context window size |
| `EMBEDDING_MODEL` | `bge-m3:latest` | Embedding model (dim=1024) |
| `CHROMA_COLLECTION_NAME` | `vib_products_v3` | Collection with category metadata |
| `TOP_K_RETRIEVAL` | `6` | Number of docs for RAG |
| `TOP_K_FINAL` | `4` | Number of docs for advisor |
| `SEMANTIC_WEIGHT` | `0.65` | Semantic weight in RRF |
| `BM25_WEIGHT` | `0.35` | BM25 weight in RRF |
| `CACHE_SIMILARITY_THRESHOLD` | `0.8` | Cache hit threshold |
| `CACHE_TOP_K` | `5` | Number of cache lookup candidates |
| `MAX_ADVISOR_TURNS` | `8` | Max turns in advisory session |

---

## 12. Extension Points

### Adding a new advisory domain

1. Add keywords to `_CATEGORY_KEYWORDS` in `src/data/loader.py`.
2. Add to `DOMAIN_FIELDS`, `DOMAIN_KEYWORDS`, `DOMAIN_LABELS` in `field_definitions.py`.
3. Add the domain to `DOMAIN_PROMPT` in `domain_detector.py`.
4. Force reload: `POST /admin/load?force_reload=true`.

### Persist advisor session across restarts

Currently uses `MemorySaver` (in-memory). To persist across restarts:

```python
# main_graph.py
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string("./data/checkpoints.db")
graph = builder.compile(checkpointer=memory)
```

### Improve RAG quality

```
TOP_K_RETRIEVAL=10          # fetch more candidates
SEMANTIC_WEIGHT=0.75        # higher semantic weight for domain-specific corpus
```

### Switch LLM

```
LLM_MODEL=llama3.2:3b       # faster (~3x)
LLM_MODEL=deepseek-r1:14b   # more accurate
```

### Extend Profile Store with TTL

Profiles currently have no expiry. To add a TTL:

```python
# advisor_profile_store.py — load_profile()
from datetime import datetime, timedelta

updated_at = datetime.fromisoformat(profile["_updated_at"])
if datetime.now() - updated_at > timedelta(days=90):
    return None  # profile too old, ask again
```

### Horizontal scaling

- Extract ChromaDB into a separate service (ChromaDB HTTP server mode)
- Replace MemorySaver with a Redis-backed checkpointer
- Load balance FastAPI with `uvicorn --workers N`
