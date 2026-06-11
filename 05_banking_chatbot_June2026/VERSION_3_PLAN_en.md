# VIB Chatbot — Version 3 Architecture Plan

**Last updated:** 2026-06  
**Status:** Implemented & Running

---

## Overview

Version 3 builds on V2 with the following improvements:

1. **Category Filtering** — Each document chunk is labeled with a category; searches are scoped to the relevant category instead of the entire corpus → faster results, less noise.
2. **Advisor Profile Store** — Collected customer information (per user + domain) is persisted to JSON files. On subsequent queries in the same domain, the bot recalls the stored data and asks the customer to confirm or update it, rather than starting over.
3. **Cache Check In-Graph** — Cache lookup is now a LangGraph node (no longer at the API route level). PRODUCT_CONSULT and CUSTOMER_FEEDBACK bypass the cache entirely.
4. **CUSTOMER_FEEDBACK Intent** — New intent that handles customer feedback about bot responses or VIB products, with sentiment analysis (negative / positive / neutral).
5. **Pre-extraction Fix** — `field_extractor` now uses the **last HumanMessage** (not the first) and includes loan-domain examples so `muc_dich_vay` is correctly extracted from the initial question.

---

## Overall Architecture

```
USER MESSAGE + UserID + SessionID
    │
    ▼
[FastAPI POST /chat/]
    │
    └─ LangGraph Graph.invoke()
           │
           ▼
       [START Router]  ← 3-tier priority
           │
           ├─① awaiting_profile_confirm=True
           │       → [advisor_profile_update]  ← user is confirming/updating saved profile
           │
           ├─② required_fields ≠ null AND missing_fields ≠ []
           │       → [advisor_collect_info]     ← continue multi-turn info collection
           │
           └─③ else
                   → [intent_classifier]
                          │
                    ┌─────┴──────────────────────────────────────┐
                    │                                            │
              PRODUCT_CONSULT                         remaining intents
              CUSTOMER_FEEDBACK                        → [cache_check_node]
              (both skip cache)                               │
                    │                           ┌──────────────┼───────────────┐
                    │                          HIT            MISS            MISS
                    ▼                           │          GREETING    PERSONAL/RAG
             ┌──────┴──────┐                  END              │              │
             ▼             ▼                                    ▼              ▼
      PRODUCT_CONSULT  CUSTOMER_FEEDBACK               [greeting_node]   [...pipeline]
             │                │                        _is_farewell()?        │
             ▼                ▼                        → FAREWELL_PROMPT     END
      [advisor_domain_      LLM sentiment              → GREETING_PROMPT
        detector]           analysis                        │
             │              NEGATIVE→apologize              END
             ▼              POSITIVE→thank
      [advisor_profile_      NEUTRAL→ask more
        recall]                  │
             │                  END
        ┌────┴──────┐
    HAS PROFILE  NO PROFILE
        │              │
   confirm msg   [field_extractor]
      → END            │
        │        [collect_info loop]
        │              │ (all collected)
   user reply          ▼
        │        [advisor_retrieve]
        ▼              │
  [profile_update] [advisor_recommend]
        │              │
   has missing?    save profile
        │              │
        ↓YES           END
  [collect_info]
        │
    → END / [advisor_retrieve]

    After graph returns:
    → [ConversationStore.save()] if final answer and not still collecting
       → JSON file (per UserID)
       → ChromaDB history index
```

---

## Key Features

### ① Advisor Profile Store (V3 new)

After the bot completes a recommendation, the full `collected_info` is saved to:

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
      "_updated_at": "2026-06-07T14:00:00"
    },
    "loan": { ... }
  }
}
```

**On the next query in the same domain:**

```
Customer: "I'd like to learn more about credit cards"
→ advisor_profile_recall detects an existing credit_card profile

Bot: "I see that last time you consulted about Credit Cards and provided:
      - Monthly income: 15 million
      - Main spending: online shopping, dining
      - Preferred benefit: cashback
      Is this information still accurate? ..."

Customer: "Same, but my income is now 20 million"
→ advisor_profile_update: merge {thu_nhap_hang_thang: "20 million"}
→ continue to advisor_retrieve with the updated profile
```

### ② Category Filtering (V3)

Every chunk in ChromaDB carries a `category` metadata field:

```
credit_card | insurance | loan | savings | general
```

Both the ChromaDB query and BM25 are filtered by category — avoiding irrelevant documents being returned.

Used in:
- **Indexing time** (`loader.py`): detect_category → assign metadata when loading parquet
- **RAG retrieve**: detect_category from rewritten_query → category_filter
- **Advisor retrieve**: `domain` is used directly as `category_filter`

### ③ Cache Check In-Graph (V3)

Cache is no longer at the API route level — it is now a **node inside the LangGraph graph**:

```
intent_classifier
    │
    ├── PRODUCT_CONSULT → advisor_domain_detector  (SKIP cache entirely)
    └── others → cache_check_node
                    │
                    ├── HIT  (score ≥ 0.8) → END (return cached answer)
                    └── MISS → greeting / personal_unrelated / rag pipeline
```

Why PRODUCT_CONSULT skips cache: the advisory flow is multi-turn and depends on the specific customer's profile — a cached answer from another user is meaningless.

### ④ Hybrid Cache Lookup

```
query_emb = bge-m3.embed(question)
candidates = ChromaDB("conversation_history_v3").query(query_emb, top_k=5)

For each candidate:
  semantic_sim = 1.0 - cosine_distance
  keyword_sim  = Jaccard(query_tokens, cand_tokens)
  hybrid_score = 0.65 × semantic_sim + 0.35 × keyword_sim

IF max(hybrid_score) ≥ 0.8 → Cache HIT → return cached answer
ELSE → Cache MISS → continue processing
```

### ⑤ UserID & Conversation History

- Streamlit accepts a **UserID** (default: `UID0000`); sent with every API request
- After the graph returns a final answer → save Q&A to `data/conversations/{user_id}.json`
- Index the question into the ChromaDB history collection for cache lookup

---

## 5 Intents and Processing Pipelines

### Intent 1: GREETING_FAREWELL
- **Detection**: Greetings, farewells, general thanks, small talk
- **Processing**: Cache check → (HIT: END | MISS: `greeting_node`)
- **Detail**: `_is_farewell()` keyword matching runs before the LLM call — if the message is a farewell, `FAREWELL_PROMPT` is used; otherwise `GREETING_PROMPT`. Prevents the LLM from always responding with a greeting even when the customer says "bye".

### Intent 2: PERSONAL_UNRELATED
- **Detection**: Personal sharing, off-topic conversation unrelated to banking
- **Processing**: Cache check → (HIT: END | MISS: LLM temp=0.4 — empathize → link to a VIB product → ask if interested)

### Intent 3: PRODUCT_INFO_QA → RAG Pipeline

```
[cache_check] → MISS
    ↓
[rag_rewrite]   LLM clarifies the question, adds historical context
    ↓
[rag_retrieve]  detect_category(rewritten_query) → category_filter
                Hybrid: BM25[cat] + ChromaDB[cat] → RRF → top-6 docs
    ↓
[rag_generate]  LLM synthesizes answer, STRICT grounding
                → Not found: "not found" message + hotline 1800 8180
```

### Intent 4: PRODUCT_CONSULT → Advisory Pipeline (Multi-turn + Profile)

```
[advisor_domain_detector]
  LLM classifies: credit_card | insurance | loan | savings | general
    ↓
[advisor_profile_recall]              ← V3 NEW
  Check AdvisorProfileStore[user_id][domain]
    ├── HAS profile → confirm message → END (wait for user to confirm/update)
    └── NO profile  → continue
    ↓
[advisor_field_extractor]
  Get required_fields from Knowledge Graph for the domain
  LLM pre-extracts info from the initial question → collected_info (pre-filled)
  missing_fields = required_fields - pre_filled.keys()
  → ALWAYS proceeds to advisor_collect_info
    ↓
[advisor_collect_info]  ← MULTI-TURN LOOP
  turn_count=0: ask for first missing field (or proceed if none missing)
  turn_count>0: save answer, update missing_fields, ask next field
  → Still missing → END (return question to user)
  → All collected → advisor_retrieve
    ↓
[advisor_retrieve]
  Query = "{domain_keyword} {collected_info values}"
  hybrid_retrieve(category_filter=domain) → top-4 docs
    ↓
[advisor_recommend]
  Relevance check → if irrelevant: not-found template + hotline
  LLM synthesizes recommendation, STRICT grounding
  → save_profile(user_id, domain, collected_info)  ← save profile after completion
```

**Profile Update Flow (when customer already has a profile):**

```
[START] awaiting_profile_confirm=True
    ↓
[advisor_profile_update]
  LLM extracts updates from user's reply
  confirmed=True, updates={}    → use existing profile → advisor_retrieve
  confirmed=True, updates={...} → merge, save → advisor_retrieve
  confirmed=False               → reset → advisor_collect_info
```

### Intent 5: CUSTOMER_FEEDBACK → Feedback Pipeline

- **Detection**: Feedback, reviews, complaints or praise about the chatbot or VIB products/services. Distinguish from GREETING_FAREWELL:
  - `"Thank you"` alone → GREETING_FAREWELL
  - `"Thank you, that answer was very helpful"` → CUSTOMER_FEEDBACK (evaluating quality)
  - `"Why did you ask what I already told you?"` → CUSTOMER_FEEDBACK (complaint)

- **Processing**: **Bypasses cache entirely** → `customer_feedback_node`

```
(skip cache)
    ↓
[customer_feedback_node]
  Input: last user message + last 4 conversation turns
  LLM (temp=0.3) analyzes → JSON {sentiment, response}
    ↓
  NEGATIVE (frustrated, complaining, dissatisfied):
    → Sincere apology, show empathy
    → Commit to acknowledging and improving
    → For product complaints: suggest hotline 1800 8180 for specific resolution
    → Tone: gentle, receptive

  POSITIVE (praise, satisfied):
    → Sincere, humble thanks
    → Ask if the customer needs any further help

  NEUTRAL (neutral, unclear sentiment):
    → Acknowledge, ask for more detail
    → END
```

**Examples:**
```
Customer: "Why did you ask about my loan purpose when I already said I want to buy a house?"
→ NEGATIVE → "I sincerely apologize for the inconvenience! You are absolutely right.
             I will take note of this and work to improve..."

Customer: "The bot's answers are very clear and helpful!"
→ POSITIVE → "Thank you so much for the kind words! I still have a lot to learn and
              will continue striving to serve you better. Is there anything else I can help you with?"
```

- **File**: `src/graph/nodes/customer_feedback.py`
- **Node**: `customer_feedback_node` → added directly to the graph, edge to END

---

## Knowledge Graph — Required Fields per Domain

```
credit_card:
  thu_nhap_hang_thang    — Monthly income
  muc_chi_tieu_chu_yeu   — Main spending areas (dining / online / travel...)
  uu_tien_quyen_loi      — Preferred benefit (cashback / points / 0% instalment...)
  co_the_hien_tai        — Whether they already have a credit card

insurance:
  tuoi                   — Customer's age
  tinh_trang_gia_dinh    — Family situation / dependants
  muc_dich_bao_hiem      — Insurance purpose (health / savings / life / accident)
  ngan_sach_hang_thang   — Monthly budget

loan:
  muc_dich_vay           — Loan purpose (home / car / consumer / business)
  so_tien_can_vay        — Loan amount needed
  thu_nhap_hang_thang    — Monthly income
  tai_san_the_chap       — Collateral assets

savings:
  so_tien_gui            — Deposit amount
  thoi_han_gui           — Term (1/3/6/12 months / long-term)
  muc_tieu               — Goal (safety / high interest / flexibility)

general:
  loai_san_pham          — Product type of interest
```

---

## Data Layer

### Product Knowledge Base
- **Source**: `documents_bgem3.parquet` (chunks, embedding dim=1024, bge-m3)
- **ChromaDB collection**: `vib_products_v3` — pre-computed embeddings + `category` metadata
- **BM25 indexes**: In-memory — 1 global + 1 per category (rebuilt on every startup)

### Conversation History (Cache)
- **ChromaDB collection**: `conversation_history_v3` — index of historical questions
- **JSON files**: `data/conversations/{user_id}.json` — human-readable Q&A history per user
- **Startup sync (bi-directional)**: `load_all_history()` compares JSON ↔ ChromaDB — removes stale ChromaDB entries no longer in JSON, and adds new entries not yet indexed. Use `POST /admin/rebuild-history` to force a full clear + rebuild.

### Advisor Profiles (V3 new)
- **JSON files**: `data/conversations/{user_id}_profiles.json` — collected_info per domain
- **Format**: `{domain: {field: value, _updated_at: ...}}`

---

## Hybrid Retrieval

```
Input: query, top_k, domain_hint (optional), category_filter (optional)

1. Enrich: search_query = "{domain_hint} {query}".strip()

2. Semantic (ChromaDB):
   query_vec = bge-m3.embed(search_query)
   where = {"category": category_filter} IF category_filter
   results = collection.query(query_vec, n=top_k×2, where=where)

3. BM25:
   bm25  = _bm25_by_category[cat] IF cat ELSE global_bm25
   texts = _texts_by_category[cat] IF cat ELSE all_texts
   scores = bm25.get_scores(tokens) → top top_k×2

4. RRF Fusion:
   score[doc] += 0.65 / (60 + semantic_rank)
   score[doc] += 0.35 / (60 + bm25_rank)
   sort descending → top_k docs

5. Return: [{content, score_rank, category}]
```

---

## LangGraph State

```python
class ChatState(TypedDict):
    # ── Core ────────────────────────────────────────────────
    messages          : Annotated[list, add_messages]
    session_id        : str
    user_id           : Optional[str]    # load/save profile per user  ← V3

    # ── Intent ──────────────────────────────────────────────
    intent            : Optional[str]    # 5 intent values

    # ── RAG ─────────────────────────────────────────────────
    rewritten_query   : Optional[str]
    retrieved_docs    : Optional[list]

    # ── Advisor ─────────────────────────────────────────────
    advisor_domain    : Optional[str]    # credit_card | insurance | loan | savings | general
    required_fields   : Optional[dict]   # {field: question_to_ask}
    collected_info    : Optional[dict]   # {field: user_answer}
    missing_fields    : Optional[list]
    awaiting_profile_confirm : Optional[bool]  # True while waiting for user to confirm profile  ← V3

    # ── Cache ────────────────────────────────────────────────
    cache_hit         : Optional[bool]
    cache_similarity  : Optional[float]

    # ── Session ─────────────────────────────────────────────
    turn_count        : int
    max_turns         : int              # default 8
```

**MemorySaver**: State is persisted in-memory per `session_id`. Server restart → advisor session state is lost; JSON history and profiles are not lost.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Graph Orchestration | LangGraph (StateGraph + MemorySaver) |
| LLM | Ollama → DeepSeek-R1:8b |
| Embeddings | Ollama → bge-m3 (1024-dim) |
| Vector Store (products) | ChromaDB `vib_products_v3` (with category metadata) |
| Vector Store (history) | ChromaDB `conversation_history_v3` |
| BM25 | rank-bm25 (in-memory, global + per-category) |
| Knowledge Graph | Dict-based field definitions |
| Advisor Profile Store | JSON files per user (`{user_id}_profiles.json`) |
| API Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| Config | pydantic-settings + .env |
| Logging | loguru |

---

## Directory Structure

```
version_3/
├── VERSION_3_PLAN.md
├── technote.md
├── userguide.md
├── requirements.txt
├── .env / .env.example
├── start_api.bat / start_streamlit.bat
├── config/
│   └── settings.py
├── scripts/
│   └── check_gpu.py
├── src/
│   ├── llm.py
│   ├── data/
│   │   └── loader.py              # Parquet → ChromaDB + BM25 (V3: category metadata)
│   ├── retrieval/
│   │   └── retriever.py           # hybrid_retrieve (V3: category_filter)
│   ├── knowledge_graph/
│   │   └── field_definitions.py   # DOMAIN_FIELDS, DOMAIN_KEYWORDS, DOMAIN_LABELS
│   ├── history/
│   │   ├── conversation_store.py  # JSON + ChromaDB history + cache lookup
│   │   └── advisor_profile_store.py  # ← V3 NEW: persists collected_info per user+domain
│   ├── graph/
│   │   ├── state.py               # ChatState TypedDict (V3: user_id, awaiting_profile_confirm)
│   │   ├── main_graph.py          # LangGraph assembly (V3: profile nodes, updated routing)
│   │   └── nodes/
│   │       ├── intent_classifier.py
│   │       ├── greeting.py
│   │       ├── personal_unrelated.py
│   │       ├── cache_check.py     # ← V3: cache_check_node in-graph (not API-level)
│   │       ├── customer_feedback.py  # ← V3 NEW: CUSTOMER_FEEDBACK intent handler
│   │       ├── rag/
│   │       │   ├── rewrite.py
│   │       │   ├── retrieve.py    # V3: detect_category from query → category_filter
│   │       │   └── generate.py
│   │       └── advisor/
│   │           ├── domain_detector.py
│   │           ├── profile_recall.py  # ← V3 NEW: profile recall + update nodes
│   │           ├── field_extractor.py
│   │           ├── info_collector.py
│   │           └── recommender.py    # V3: category_filter + save_profile after recommend
│   └── api/
│       ├── main.py
│       ├── models/schemas.py
│       └── routes/
│           ├── chat.py
│           └── admin.py
├── frontend/
│   └── app.py
└── data/
    ├── vectorstore/                # ChromaDB (vib_products_v3)
    └── conversations/
        ├── {user_id}.json          # Q&A history per user
        └── {user_id}_profiles.json # ← V3 NEW: advisor profile per user
```

---

## Improvements over Version 2

| V2 Problem | V3 Solution |
|---|---|
| Searches entire corpus for every query | Category filtering: searches only within the relevant category |
| Single global BM25 index | BM25 per-category (credit_card / insurance / loan / savings) |
| Advisor re-asks all fields every session | Profile Recall: recalls previous info, only asks about changes |
| Cache check at API route (before graph) | Cache check is a graph node; PRODUCT_CONSULT + CUSTOMER_FEEDBACK bypass cache |
| `user_id` not in ChatState | `user_id` in state → used to load/save profile per user |
| No profile update node | `advisor_profile_recall` + `advisor_profile_update` nodes added |
| Recommender does not save profile | `advisor_recommend_node` calls `save_profile()` after completion |
| No handler for customer feedback | `CUSTOMER_FEEDBACK` intent + node with sentiment analysis |
| field_extractor used first HumanMessage | Fixed to use last HumanMessage (the current question) |
| Missing loan examples in extract prompt | Added examples: "vay mua nhà" → muc_dich_vay: "mua nhà" |

---

## parse_json — Handling DeepSeek-R1 Output

DeepSeek-R1 generates `<think>...</think>` blocks first, with JSON at the end. `parse_json()` handles this:

```
1. Strip <think>...</think> blocks
2. Try json.loads() on the remaining text
3. Find all {...} blocks → try from the END backwards
4. Look for ```json...``` code block
5. Fail → return {} + log warning
```
