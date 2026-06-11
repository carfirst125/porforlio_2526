# VIB Chatbot V3 — User Guide

## System Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Ollama models pulled:
  ```
  ollama pull deepseek-r1:8b
  ollama pull bge-m3
  ```
- File `documents_bgem3.parquet` located in the parent directory (`../documents_bgem3.parquet`)

---

## Quick Start

### Step 1 — Install dependencies (first time only)
```bash
cd version_3
pip install -r requirements.txt
```

### Step 2 — Create the configuration file
```bash
copy .env.example .env
# Edit .env if needed (model, port, GPU settings...)
```

### Step 3 — Start Ollama
```bash
ollama serve
```

### Step 4 — Start the API server (Terminal 1)
```bash
cd version_3
python -m uvicorn src.api.main:app --reload
# Or: double-click start_api.bat
```

On first startup, the server will:
- Load parquet → index into ChromaDB collection `vib_products_v3`
- Automatically detect the **category** (card / insurance / loan / savings) for each chunk
- Build a global BM25 index + a separate BM25 for each category
- **Sync Q&A history** (bi-directional): remove stale ChromaDB entries no longer in JSON, add new entries → ChromaDB history (for cache lookup)
- This process takes approximately 30–60 seconds

### Step 5 — Start Streamlit (Terminal 2)
```bash
cd version_3
python -m streamlit run frontend/app.py
# Or: double-click start_streamlit.bat
```

Access the UI: **http://localhost:8501**  
API docs: **http://localhost:8000/docs**

---

## Key V3 Features

### Category Filtering

V3 assigns a **category** label to each document chunk in ChromaDB:

| Category | Content type |
|---|---|
| `credit_card` | Credit cards, debit cards, cashback, air miles... |
| `insurance` | Life insurance, health insurance, accident coverage... |
| `loan` | Home loans, consumer loans, secured/unsecured credit... |
| `savings` | Savings accounts, deposit rates, term options... |
| `general` | General VIB information, other services |

When advising on credit cards, the chatbot only searches within `credit_card` chunks — faster and free from noise caused by unrelated documents.

### Advisor Profile — Remembering Customer Information

After completing an advisory session, the chatbot **saves the customer's information** (income, needs, etc.) per user + domain.  
The next time the customer asks about the same topic, the chatbot recalls it and only asks if anything has changed:

```
Customer: "I'd like to learn more about credit cards"
Bot: "I see that last time you consulted about Credit Cards:
      - Monthly income: 15 million
      - Main spending: online shopping, dining
      - Preferred benefit: cashback
      Is this information still accurate? ..."
Customer: "Same, but my income is now 20 million"
Bot: "Updated your income. Let me advise you right away!" → recommendation
```

### Smart Cache

Questions that have been answered before → the chatbot responds instantly, without calling the LLM:
- **Similarity ≥ 0.8** → return cached answer (**⚡ very fast**)
- **Similarity < 0.8** → process normally

The cache uses **hybrid search**: 65% semantic + 35% keyword. Note: the PRODUCT_CONSULT intent (advisory) **does not use cache** — advisory responses depend on the specific customer's profile.

---

## Using the Streamlit Interface

### UserID

- Enter a UserID in the sidebar (default `UID0000`)
- Each UserID has its own chat history and advisory profile
- Changing the UserID resets the chat display and starts a new session

**Data stored per UserID:**
- `data/conversations/{user_id}.json` — Q&A history
- `data/conversations/{user_id}_profiles.json` — advisory information collected

### Sidebar — Controls

| Button | Function |
|---|---|
| Load data | Load parquet into ChromaDB (skipped if already loaded) |
| Reload | Force reload (needed when parquet file is replaced) |
| New Chat | Start a new session, clear the chat display |
| Check connection | Ping the API health check |

### Response Badges

| Badge | Meaning |
|---|---|
| ⚡ Cache (0.92) | Answer from cache — LLM was not called |
| 👋 Greeting | Intent: greeting or farewell |
| 💬 Personal topic | Intent: personal sharing, redirected to a product |
| 🔍 Product info | Intent: product information query → RAG pipeline |
| 🎯 Product advisory | Intent: product recommendation → Advisory pipeline |
| 📝 Customer feedback | Intent: feedback about the chatbot or a VIB product |
| 💳 Credit card | Advisory domain: search in `credit_card` category |
| 🛡️ Insurance | Advisory domain: search in `insurance` category |
| 🏠 Loan | Advisory domain: search in `loan` category |
| 💰 Savings | Advisory domain: search in `savings` category |

---

## 5 Query Types and How the Chatbot Handles Them

### 1. Greetings / Farewells
```
Customer: "Hello"
Bot: "Hi! I'm VIB's AI assistant. How can I help you today?"

Customer: "Thank you, goodbye"
Bot: "Thank you for contacting VIB! Wishing you a great day..."
```

### 2. Personal Sharing
```
Customer: "I just got promoted, my salary went up"
Bot: "Congratulations! With your new salary, you might consider a premium
      VIB credit card to earn better rewards. Would you like to find out more?"
```

### 3. Product Information Query (RAG)
```
Customer: "What is the interest rate for the VIB Super Card?"
→ detect_category("VIB Super Card") = "credit_card"
→ Search ChromaDB category=credit_card + BM25[credit_card]
Bot: [Answer from credit card documents]

Customer: "What are the conditions for a VIB home loan?"
→ detect_category("home loan") = "loan"
→ Search ChromaDB category=loan + BM25[loan]
Bot: [Answer from loan documents]
```

### 5. Customer Feedback (CUSTOMER_FEEDBACK)
```
--- Case 1: Customer is dissatisfied ---
Customer: "Why did you ask about my loan purpose when I already said I want to buy a house?"
→ Detected as CUSTOMER_FEEDBACK, sentiment=NEGATIVE

Bot: "I sincerely apologize for the inconvenience! You are absolutely right — you
      clearly said you wanted to buy a house from the start, and I asked again.
      I will take note of this to improve. Would you like me to advise you on a
      suitable home loan package right away?"

--- Case 2: Customer is satisfied ---
Customer: "The chatbot's answers are very helpful and easy to understand!"
→ CUSTOMER_FEEDBACK, sentiment=POSITIVE

Bot: "Thank you so much for the kind words! I still have a lot to learn
      and will keep striving to serve you better. Is there anything else I can help you with?"

--- Case 3: Customer complains about a product ---
Customer: "I think VIB's interest rates are too high"
→ CUSTOMER_FEEDBACK, sentiment=NEGATIVE (product-related)

Bot: "Thank you for sharing! I understand your concern about the interest rate.
      For specific advice and to find the most suitable loan package, you can
      contact hotline 1800 8180 (free, 24/7) or let me help you find a better option!"
```

### 4. Product Advisory — First Time (Multi-turn)
```
Customer: "I want to open a credit card but don't know which one to choose"
→ domain_detector: "credit_card"
→ profile_recall: no profile found → field_extractor → collect_info

Bot: "To recommend the most suitable card, I need a few more details:
      What is your approximate monthly income?"
Customer: "Around 15 million"
Bot: "What do you mainly spend on?"
Customer: "Online shopping and dining"
Bot: "Which benefit do you prioritize: cashback, points, or 0% instalment?"
Customer: "Cashback"
Bot: "Do you currently have a credit card?"
Customer: "No"
→ advisor_retrieve: search in category=credit_card
→ advisor_recommend: suggest the most suitable product
→ save_profile: save customer information

Bot: "Based on your 15 million income and online shopping/dining spending with
      a preference for cashback, the VIB Online Plus card is a great fit because..."
```

**Note:** If the initial question already includes some information, the chatbot will **automatically extract it** and only ask for the missing fields.

### 4b. Product Advisory — Return Visit (Profile Recall)
```
(Same UserID, asking about credit cards again)

Customer: "Please advise me on a credit card"
→ domain_detector: "credit_card"
→ profile_recall: found an existing credit_card profile for this UserID!

Bot: "I see that last time you consulted about Credit Cards and provided:
      - Monthly income: 15 million
      - Main spending: online shopping, dining
      - Preferred benefit: cashback
      - Current card: none
      Is this information still accurate?
      - If yes, just say "OK" and I'll advise you right away!
      - If anything has changed, please let me know."

--- Case 1: Customer confirms ---
Customer: "Yes, that's correct"
→ profile_update: confirmed=True, no updates
→ advisor_retrieve → recommend immediately (no re-asking)
Bot: [Recommendation based on existing profile]

--- Case 2: Customer updates ---
Customer: "My income is now 25 million, everything else is the same"
→ profile_update: merge {thu_nhap_hang_thang: "25 million"}
→ advisor_retrieve → recommend with updated information
Bot: "Income updated. Let me advise you right away! ⏳"
     [Recommendation with new income]

--- Case 3: Customer wants to start over ---
Customer: "Actually I want to answer from scratch"
→ profile_update: confirmed=False → reset → field_extractor
Bot: [Asks all questions again as if first time]
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/admin/health` | Health check — vectorstore, models, category counts |
| GET | `/admin/stats` | Stats: chunk count, category breakdown, history count |
| POST | `/admin/load` | Load parquet → ChromaDB with category metadata |
| POST | `/admin/load?force_reload=true` | Force reload (needed after replacing parquet) |
| POST | `/admin/rebuild-history` | Clear + rebuild ChromaDB history from JSON (use after manually deleting/editing JSON files without restarting) |
| POST | `/chat/` | Send a message |
| GET | `/chat/history/{user_id}` | User Q&A history |
| GET | `/chat/session/{session_id}` | Session state (debug) |
| GET | `/docs` | Swagger UI |

---

## Directory Structure

```
version_3/
├── config/settings.py              # All configuration (pydantic-settings)
├── src/
│   ├── llm.py                      # LLM + Embedding factory + parse_json
│   ├── data/loader.py              # Parquet → ChromaDB + BM25 (category metadata)
│   ├── retrieval/retriever.py      # Hybrid retrieval (BM25 + Semantic + RRF + category_filter)
│   ├── knowledge_graph/
│   │   └── field_definitions.py    # Domain fields for advisory
│   ├── history/
│   │   ├── conversation_store.py   # JSON Q&A history + ChromaDB cache
│   │   └── advisor_profile_store.py  # ← Persists collected_info per user+domain
│   ├── graph/
│   │   ├── state.py                # ChatState (user_id, awaiting_profile_confirm)
│   │   ├── main_graph.py           # LangGraph assembly + routing
│   │   └── nodes/
│   │       ├── intent_classifier.py
│   │       ├── greeting.py
│   │       ├── personal_unrelated.py
│   │       ├── cache_check.py      # In-graph cache lookup node
│   │       ├── customer_feedback.py  # ← CUSTOMER_FEEDBACK intent handler
│   │       ├── rag/
│   │       │   ├── rewrite.py
│   │       │   ├── retrieve.py     # detect_category → category_filter
│   │       │   └── generate.py
│   │       └── advisor/
│   │           ├── domain_detector.py
│   │           ├── profile_recall.py   # ← Profile recall + update nodes
│   │           ├── field_extractor.py
│   │           ├── info_collector.py
│   │           └── recommender.py      # category_filter + save_profile
│   └── api/
│       ├── main.py
│       └── routes/{chat, admin}.py
├── frontend/app.py                 # Streamlit UI
├── data/
│   ├── vectorstore/                # ChromaDB (vib_products_v3)
│   └── conversations/
│       ├── {user_id}.json          # Q&A history per user
│       └── {user_id}_profiles.json # Advisor profile per user
├── scripts/check_gpu.py
├── requirements.txt
├── .env / .env.example
├── start_api.bat / start_streamlit.bat
├── VERSION_3_PLAN.md / VERSION_3_PLAN_en.md
├── technote.md / technote_en.md
└── userguide.md / userguide_en.md  # This file
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| ❌ Cannot connect to API | Server not running | Run `start_api.bat` |
| ⚠️ ChromaDB not loaded | Parquet not yet indexed | Click "Load data" in the sidebar |
| ⏱️ Very slow responses | Ollama running on CPU | Check `OLLAMA_NUM_GPU=-1`, run `python scripts/check_gpu.py` |
| ❌ Model not found | Model not pulled | `ollama pull deepseek-r1:8b && ollama pull bge-m3` |
| 🔄 No category counts in `/admin/stats` | Using old collection (not `vib_products_v3`) | Force reload: `POST /admin/load?force_reload=true` |
| 🔄 Bot re-asks info already provided in this session | Session state lost (server restart) | Profile is still saved in file — start a new question to trigger recall |
| ⚡ Cache not working | History ChromaDB not loaded | Restart server — `load_all_history()` runs automatically on startup |
| 🔄 Bot doesn't remember profile despite previous advisory | Different UserIDs | Check the UserID in the sidebar — must use the same UserID |
| 🔄 After deleting a JSON entry, the bot still responds from cache | ChromaDB history not yet synced | Restart the server (auto-syncs) or call `POST /admin/rebuild-history` to sync immediately without restarting |

---

## Performance Tuning

**Want faster responses:**
```
LLM_MODEL=llama3.2:3b        # smaller model, ~3x faster
OLLAMA_NUM_GPU=-1             # ensure GPU is being used
```

**Want more accurate answers:**
```
LLM_MODEL=deepseek-r1:14b    # larger model
TOP_K_RETRIEVAL=10           # fetch more docs
```

**Want broader cache matching** (accept more similar questions):
```
CACHE_SIMILARITY_THRESHOLD=0.75
```

**Want stricter cache matching** (only match near-identical questions):
```
CACHE_SIMILARITY_THRESHOLD=0.90
```

---

## VIB Support Hotline
- **1800 8180** (free, 24/7)
- **vib.com.vn**
