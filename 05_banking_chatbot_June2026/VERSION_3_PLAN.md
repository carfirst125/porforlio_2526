# VIB Chatbot — Version 3 Architecture Plan

**Cập nhật lần cuối:** 2026-06  
**Trạng thái:** Implemented & Running

---

## Tổng quan

Version 3 xây dựng trên nền V2 với các cải tiến sau:

1. **Category Filtering** — Gán nhãn category cho mỗi chunk tài liệu, tìm kiếm trong đúng danh mục thay vì toàn bộ corpus → nhanh hơn, ít nhiễu hơn.
2. **Advisor Profile Store** — Lưu thông tin KH đã thu thập (per user + domain) vào file JSON. Lần sau hỏi cùng domain → nhắc lại và cho KH xác nhận/cập nhật thay vì hỏi lại từ đầu.
3. **Cache Check In-Graph** — Cache lookup được đưa vào node trong LangGraph (không phải API-level nữa). PRODUCT_CONSULT và CUSTOMER_FEEDBACK bỏ qua cache.
4. **CUSTOMER_FEEDBACK Intent** — Intent mới xử lý phản hồi của KH về câu trả lời bot hoặc sản phẩm VIB, với sentiment analysis (negative/positive/neutral).
5. **Fix Pre-extraction** — `field_extractor` dùng **last HumanMessage** (không phải first) + bổ sung ví dụ loan domain để extract đúng `muc_dich_vay` từ câu hỏi.

---

## Kiến trúc tổng thể

```
USER MESSAGE + UserID + SessionID
    │
    ▼
[FastAPI POST /chat/]
    │
    └─ LangGraph Graph.invoke()
           │
           ▼
       [START Router]  ← Ưu tiên 3 tầng
           │
           ├─① awaiting_profile_confirm=True
           │       → [advisor_profile_update]  ← user đang confirm/update profile cũ
           │
           ├─② required_fields ≠ null AND missing_fields ≠ []
           │       → [advisor_collect_info]     ← tiếp tục multi-turn thu thập info
           │
           └─③ else
                   → [intent_classifier]
                          │
                    ┌─────┴──────────────────────────────────────┐
                    │                                            │
              PRODUCT_CONSULT                         các intent còn lại
              CUSTOMER_FEEDBACK                        → [cache_check_node]
              (cả 2 skip cache)                               │
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
             │              NEGATIVE→xin lỗi               END
             ▼              POSITIVE→cảm ơn
      [advisor_profile_      NEUTRAL→hỏi thêm
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

    Sau khi graph trả về:
    → [ConversationStore.save()] nếu có final answer và không đang thu thập
       → JSON file (per UserID)
       → ChromaDB history index
```

---

## Tính năng chính

### ① Advisor Profile Store (V3 mới)

Sau khi bot hoàn thành recommendation cho KH, toàn bộ `collected_info` được lưu vào:

```
data/conversations/{user_id}_profiles.json
{
  "user_id": "UID001",
  "profiles": {
    "credit_card": {
      "thu_nhap_hang_thang":  "15 triệu",
      "muc_chi_tieu_chu_yeu": "mua sắm online, ăn uống",
      "uu_tien_quyen_loi":    "cashback",
      "co_the_hien_tai":      "chưa có",
      "_updated_at": "2026-06-07T14:00:00"
    },
    "loan": { ... }
  }
}
```

**Lần sau KH hỏi cùng domain:**

```
KH: "tôi muốn xem thêm về thẻ tín dụng"
→ advisor_profile_recall phát hiện có profile credit_card

Bot: "Mình thấy lần trước bạn đã tư vấn về Thẻ tín dụng và cung cấp:
      - Thu nhập hàng tháng: 15 triệu
      - Chi tiêu chủ yếu: mua sắm online, ăn uống
      - Quyền lợi ưu tiên: cashback
      Thông tin này vẫn còn đúng không ạ? ..."

KH: "vẫn vậy nhưng thu nhập tôi bây giờ 20 triệu rồi"
→ advisor_profile_update: merge {thu_nhap_hang_thang: "20 triệu"}
→ tiếp tục advisor_retrieve với profile đã cập nhật
```

### ② Category Filtering (V3)

Mỗi chunk trong ChromaDB có metadata `category`:

```
credit_card | insurance | loan | savings | general
```

Khi tìm kiếm, cả ChromaDB query và BM25 đều được filter theo category — tránh trả về tài liệu không liên quan.

Dùng cho:
- **Indexing time** (`loader.py`): detect_category → gán metadata khi load parquet
- **RAG retrieve**: detect_category từ rewritten_query → category_filter
- **Advisor retrieve**: dùng trực tiếp `domain` làm `category_filter`

### ③ Cache Check In-Graph (V3)

Cache không còn ở API route mà là một **node trong LangGraph graph**:

```
intent_classifier
    │
    ├── PRODUCT_CONSULT → advisor_domain_detector  (SKIP cache hoàn toàn)
    └── khác → cache_check_node
                    │
                    ├── HIT  (score ≥ 0.8) → END (trả về cached answer)
                    └── MISS → greeting / personal_unrelated / rag pipeline
```

Lý do PRODUCT_CONSULT bỏ qua cache: luồng tư vấn là multi-turn và phụ thuộc profile KH — câu trả lời cached từ user khác không có giá trị.

### ④ Hybrid Cache Lookup

```
query_emb = bge-m3.embed(question)
candidates = ChromaDB("conversation_history_v3").query(query_emb, top_k=5)

Với mỗi candidate:
  semantic_sim = 1.0 - cosine_distance
  keyword_sim  = Jaccard(query_tokens, cand_tokens)
  hybrid_score = 0.65 × semantic_sim + 0.35 × keyword_sim

IF max(hybrid_score) ≥ 0.8 → Cache HIT → return cached answer
ELSE → Cache MISS → xử lý tiếp
```

### ⑤ UserID & Conversation History

- Streamlit nhập **UserID** (default: `UID0000`); gửi kèm mỗi API request
- Sau graph trả về final answer → lưu Q&A vào `data/conversations/{user_id}.json`
- Index câu hỏi vào ChromaDB history collection để cache lookup

---

## 5 Intent và Pipeline xử lý

### Intent 1: GREETING_FAREWELL
- **Nhận diện**: Chào hỏi, tạm biệt, cảm ơn, hỏi thăm chung
- **Xử lý**: Cache check → (HIT: END | MISS: `greeting_node`)
- **Chi tiết**: `_is_farewell()` keyword matching trước khi gọi LLM — nếu là lời tạm biệt dùng `FAREWELL_PROMPT`, ngược lại dùng `GREETING_PROMPT`. Tránh LLM luôn trả lời kiểu chào hỏi ngay cả khi KH nói "bye".

### Intent 2: PERSONAL_UNRELATED
- **Nhận diện**: Chia sẻ cá nhân, tâm sự, chủ đề ngoài ngân hàng
- **Xử lý**: Cache check → (HIT: END | MISS: LLM temp=0.4 — đồng cảm → liên kết sản phẩm → hỏi quan tâm không)

### Intent 3: PRODUCT_INFO_QA → RAG Pipeline

```
[cache_check] → MISS
    ↓
[rag_rewrite]   LLM làm rõ câu hỏi, bổ sung context lịch sử
    ↓
[rag_retrieve]  detect_category(rewritten_query) → category_filter
                Hybrid: BM25[cat] + ChromaDB[cat] → RRF → top-6 docs
    ↓
[rag_generate]  LLM tổng hợp, STRICT grounding
                → Không tìm thấy: "không tìm thấy" + hotline 1800 8180
```

### Intent 4: PRODUCT_CONSULT → Advisory Pipeline (Multi-turn + Profile)

```
[advisor_domain_detector]
  LLM phân loại: credit_card | insurance | loan | savings | general
    ↓
[advisor_profile_recall]              ← V3 MỚI
  Kiểm tra AdvisorProfileStore[user_id][domain]
    ├── CÓ profile → confirm message → END (chờ user confirm/update)
    └── KHÔNG có  → tiếp tục
    ↓
[advisor_field_extractor]
  Lấy required_fields từ Knowledge Graph theo domain
  LLM pre-extract info từ câu hỏi ban đầu → collected_info (pre-filled)
  missing_fields = required_fields - pre_filled.keys()
  → LUÔN đi vào advisor_collect_info
    ↓
[advisor_collect_info]  ← MULTI-TURN LOOP
  turn_count=0: hỏi field đầu tiên còn thiếu (hoặc proceed nếu không thiếu)
  turn_count>0: lưu câu trả lời, cập nhật missing_fields, hỏi tiếp
  → Còn thiếu → END (trả câu hỏi về user)
  → Đủ thông tin → advisor_retrieve
    ↓
[advisor_retrieve]
  Query = "{domain_keyword} {collected_info values}"
  hybrid_retrieve(category_filter=domain) → top-4 docs
    ↓
[advisor_recommend]
  Relevance check → nếu không liên quan: not-found + hotline
  LLM tổng hợp recommendation, STRICT grounding
  → save_profile(user_id, domain, collected_info)  ← lưu profile sau khi xong
```

**Profile Update Flow (khi KH đã có profile):**

```
[START] awaiting_profile_confirm=True
    ↓
[advisor_profile_update]
  LLM trích xuất updates từ câu trả lời user
  confirmed=True, updates={}  → dùng profile cũ → advisor_retrieve
  confirmed=True, updates={...} → merge, lưu → advisor_retrieve
  confirmed=False → reset → advisor_collect_info
```

### Intent 5: CUSTOMER_FEEDBACK → Feedback Pipeline

- **Nhận diện**: Phản hồi, đánh giá, góp ý về chatbot hoặc sản phẩm/dịch vụ VIB. Phân biệt với GREETING_FAREWELL:
  - `"Cảm ơn"` đơn thuần → GREETING_FAREWELL
  - `"Cảm ơn, câu trả lời rất hữu ích"` → CUSTOMER_FEEDBACK (đang đánh giá chất lượng)
  - `"Sao bot hỏi lại điều tôi đã nói?"` → CUSTOMER_FEEDBACK (phàn nàn)

- **Xử lý**: **Bỏ qua cache hoàn toàn** → `customer_feedback_node`

```
(skip cache)
    ↓
[customer_feedback_node]
  Input: last user message + lịch sử 4 turns gần nhất
  LLM (temp=0.3) phân tích → JSON {sentiment, response}
    ↓
  NEGATIVE (bực bội, phàn nàn, không hài lòng):
    → Xin lỗi chân thành, đồng cảm
    → Cam kết ghi nhận và cải thiện
    → Về sản phẩm: đề nghị hotline 1800 8180 nếu cần giải quyết cụ thể
    → Giọng điệu: nhỏ nhẹ, cầu thị

  POSITIVE (khen ngợi, hài lòng):
    → Cảm ơn chân thành, khiêm tốn
    → Hỏi KH còn cần hỗ trợ gì không

  NEUTRAL (trung tính, không rõ cảm xúc):
    → Ghi nhận, hỏi thêm chi tiết
    → END
```

**Ví dụ:**
```
KH: "Sao bạn hỏi mục đích vay trong khi tôi đã nói vay mua nhà rồi?"
→ NEGATIVE → "Mình xin lỗi vì đã làm bạn không hài lòng! Bạn hoàn toàn đúng.
             Mình sẽ ghi nhận để cải thiện..."

KH: "Bot trả lời rất rõ ràng và hữu ích!"
→ POSITIVE → "Cảm ơn bạn rất nhiều! Mình còn nhiều điều cần học hỏi và sẽ
              cố gắng phục vụ tốt hơn. Bạn cần hỗ trợ thêm gì không ạ?"
```

- **File**: `src/graph/nodes/customer_feedback.py`
- **Node**: `customer_feedback_node` → thêm trực tiếp vào graph, edge tới END

---

## Knowledge Graph — Required Fields per Domain

```
credit_card:
  thu_nhap_hang_thang    — Thu nhập hàng tháng
  muc_chi_tieu_chu_yeu   — Chi tiêu chủ yếu (ăn uống / online / du lịch...)
  uu_tien_quyen_loi      — Ưu tiên (cashback / tích điểm / trả góp 0%...)
  co_the_hien_tai        — Đã có thẻ tín dụng chưa

insurance:
  tuoi                   — Tuổi khách hàng
  tinh_trang_gia_dinh    — Gia đình / con cái
  muc_dich_bao_hiem      — Mục đích (SK / tích lũy / nhân thọ / tai nạn)
  ngan_sach_hang_thang   — Ngân sách hàng tháng

loan:
  muc_dich_vay           — Mục đích (nhà / xe / tiêu dùng / kinh doanh)
  so_tien_can_vay        — Số tiền cần vay
  thu_nhap_hang_thang    — Thu nhập hàng tháng
  tai_san_the_chap       — Tài sản thế chấp

savings:
  so_tien_gui            — Số tiền gửi
  thoi_han_gui           — Thời hạn (1/3/6/12 tháng / dài hạn)
  muc_tieu               — Mục tiêu (an toàn / lãi cao / linh hoạt)

general:
  loai_san_pham          — Loại sản phẩm quan tâm
```

---

## Data Layer

### Product Knowledge Base
- **Nguồn**: `documents_bgem3.parquet` (chunks, embedding dim=1024, bge-m3)
- **ChromaDB collection**: `vib_products_v3` — pre-computed embeddings + `category` metadata
- **BM25 indexes**: In-memory — 1 global + 1 per category (xây lại mỗi startup)

### Conversation History (Cache)
- **ChromaDB collection**: `conversation_history_v3` — index câu hỏi lịch sử
- **JSON files**: `data/conversations/{user_id}.json` — readable Q&A history per user
- **Startup sync (bi-directional)**: `load_all_history()` so sánh JSON ↔ ChromaDB — xóa entries ChromaDB không còn trong JSON (stale), add entries mới chưa được index. Dùng `POST /admin/rebuild-history` để force clear + rebuild toàn bộ.

### Advisor Profiles (V3 mới)
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
    awaiting_profile_confirm : Optional[bool]  # True khi chờ user xác nhận profile  ← V3

    # ── Cache ────────────────────────────────────────────────
    cache_hit         : Optional[bool]
    cache_similarity  : Optional[float]

    # ── Session ─────────────────────────────────────────────
    turn_count        : int
    max_turns         : int              # default 8
```

**MemorySaver**: State persist in-memory theo `session_id`. Restart server → mất advisor session state; JSON history và profiles không mất.

---

## Stack Công nghệ

| Layer | Technology |
|---|---|
| Graph Orchestration | LangGraph (StateGraph + MemorySaver) |
| LLM | Ollama → DeepSeek-R1:8b |
| Embeddings | Ollama → bge-m3 (1024-dim) |
| Vector Store (products) | ChromaDB `vib_products_v3` (có category metadata) |
| Vector Store (history) | ChromaDB `conversation_history_v3` |
| BM25 | rank-bm25 (in-memory, global + per-category) |
| Knowledge Graph | Dict-based field definitions |
| Advisor Profile Store | JSON files per user (`{user_id}_profiles.json`) |
| API Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| Config | pydantic-settings + .env |
| Logging | loguru |

---

## Cấu trúc thư mục

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
│   │   └── advisor_profile_store.py  # ← V3 MỚI: lưu collected_info per user+domain
│   ├── graph/
│   │   ├── state.py               # ChatState TypedDict (V3: user_id, awaiting_profile_confirm)
│   │   ├── main_graph.py          # LangGraph assembly (V3: profile nodes, updated routing)
│   │   └── nodes/
│   │       ├── intent_classifier.py
│   │       ├── greeting.py
│   │       ├── personal_unrelated.py
│   │       ├── cache_check.py     # ← V3: cache_check_node in-graph (không phải API-level)
│   │       ├── rag/
│   │       │   ├── rewrite.py
│   │       │   ├── retrieve.py    # V3: detect_category từ query → category_filter
│   │       │   └── generate.py
│   │       └── advisor/
│   │           ├── domain_detector.py
│   │           ├── profile_recall.py  # ← V3 MỚI: recall + update profile nodes
│   │           ├── field_extractor.py
│   │           ├── info_collector.py
│   │           └── recommender.py    # V3: category_filter + save_profile sau recommend
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
        └── {user_id}_profiles.json # ← V3 MỚI: advisor profile per user
```

---

## Điểm cải tiến so với Version 2

| Vấn đề V2 | Giải pháp V3 |
|---|---|
| Tìm kiếm toàn bộ corpus cho mọi query | Category filtering: tìm trong đúng category |
| BM25 chỉ có 1 index global | BM25 per-category (credit_card / insurance / loan / savings) |
| Advisor hỏi lại từ đầu mỗi lần | Profile Recall: nhắc thông tin cũ, chỉ hỏi nếu thay đổi |
| Cache check ở API route (trước graph) | Cache check là node trong graph; PRODUCT_CONSULT + CUSTOMER_FEEDBACK bỏ qua cache |
| `user_id` không có trong ChatState | `user_id` trong state → dùng load/save profile per user |
| Không có node xử lý profile update | `advisor_profile_recall` + `advisor_profile_update` nodes |
| Recommender không lưu profile | `advisor_recommend_node` gọi `save_profile()` sau khi xong |
| Không xử lý phản hồi KH | Intent `CUSTOMER_FEEDBACK` + node với sentiment analysis |
| field_extractor lấy first HumanMessage | Sửa lấy last HumanMessage (câu hỏi hiện tại) |
| Thiếu ví dụ loan trong extract prompt | Thêm examples: "vay mua nhà" → muc_dich_vay: "mua nhà" |

---

## parse_json — Xử lý DeepSeek-R1

DeepSeek-R1 sinh `<think>...</think>` trước, JSON ở cuối. `parse_json()` xử lý:

```
1. Strip <think>...</think> blocks
2. Try json.loads(toàn bộ text còn lại)
3. Tìm tất cả {...} blocks → thử từ CUỐI lên
4. Tìm ```json...``` code block
5. Fail → return {} + log warning
```
