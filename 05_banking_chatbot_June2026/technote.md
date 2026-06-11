# VIB Chatbot V3 — Technical Note

**Version:** 3.3  
**Stack:** LangGraph · DeepSeek-R1 · bge-m3 · ChromaDB · FastAPI · Streamlit  
**Ngày:** 2026-06

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Data Layer — Category Metadata & Khởi tạo](#2-data-layer)
3. [Pipeline xử lý tin nhắn — Chi tiết từng bước](#3-pipeline-xử-lý-tin-nhắn)
   - 3.1 [START Router — Priority Routing](#31-start-router)
   - 3.2 [Intent Classifier](#32-intent-classifier)
   - 3.3 [Cache Check Node](#33-cache-check-node)
   - 3.4 [GREETING_FAREWELL Pipeline](#34-greeting_farewell-pipeline)
   - 3.5 [PERSONAL_UNRELATED Pipeline](#35-personal_unrelated-pipeline)
   - 3.6 [PRODUCT_INFO_QA — RAG Pipeline](#36-product_info_qa--rag-pipeline)
   - 3.7 [PRODUCT_CONSULT — Advisory Pipeline](#37-product_consult--advisory-pipeline)
   - 3.8 [Advisor Profile Recall & Update](#38-advisor-profile-recall--update)
4. [Hybrid Retrieval — Thuật toán chi tiết](#4-hybrid-retrieval)
5. [Category Filtering](#5-category-filtering)
6. [Advisor Profile Store](#6-advisor-profile-store)
7. [Conversation History & Cache](#7-conversation-history--cache)
8. [LangGraph State & Topology](#8-langgraph-state--topology)
9. [API Layer](#9-api-layer)
10. [Sơ đồ toàn bộ luồng dữ liệu](#10-sơ-đồ-toàn-bộ-luồng-dữ-liệu)
11. [Cấu hình quan trọng](#11-cấu-hình-quan-trọng)
12. [Mở rộng](#12-mở-rộng)

---

## 1. Tổng quan kiến trúc

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
│    └─ LangGraph xử lý toàn bộ (cache, classify, pipeline) │
│                                                            │
│  Sau khi graph trả về:                                     │
│    ConversationStore.save() — nếu final answer             │
└──────────────────────┬─────────────────────────────────────┘
                       │
          ┌────────────┼──────────────┐
          ▼            ▼              ▼
     ChromaDB      JSON files    BM25 indexes
  vib_products_v3  conversations/  global + per-category
  conv_history_v3  {uid}_profiles.json
```

**ChromaDB collections:**

| Collection | Nội dung | Mục đích |
|---|---|---|
| `vib_products_v3` | Chunks từ parquet, pre-embedded bge-m3, có `category` metadata | RAG/Advisor retrieval — filter theo category |
| `conversation_history_v3` | Q&A đã trả lời (tất cả users) | Cache lookup — tránh gọi LLM lặp lại |

**BM25 indexes (in-memory, xây lại mỗi startup):**

| Index | Corpus | Dùng cho |
|---|---|---|
| Global BM25 | Tất cả chunks | RAG fallback (category=general) |
| BM25[credit_card] | Chunks category=credit_card | Tư vấn thẻ |
| BM25[insurance] | Chunks category=insurance | Tư vấn bảo hiểm |
| BM25[loan] | Chunks category=loan | Tư vấn vay |
| BM25[savings] | Chunks category=savings | Tư vấn tiết kiệm |

**Lưu trữ JSON (data/conversations/):**

| File | Nội dung |
|---|---|
| `{user_id}.json` | Q&A history per user |
| `{user_id}_profiles.json` | Advisor collected_info per domain (V3 mới) |

---

## 2. Data Layer

### 2.1 Category Detection

Khi load parquet, mỗi chunk được gán `category` bằng keyword matching:

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
# Không khớp → "general"
```

Logic: Duyệt theo thứ tự credit_card → insurance → loan → savings. Trả về category đầu tiên match, hoặc "general".

### 2.2 Parquet → ChromaDB (startup)

```
load_data()
  → đọc parquet (columns: input, embedding)
  → parse embedding: json.loads(raw.strip('"')) → list[float]
  → detect_category(text) → metadata = {"category": cat}
  → batch add vào ChromaDB collection vib_products_v3
      collection.add(ids, documents, embeddings, metadatas)
  → xây global BM25Okapi + per-category BM25Okapi
  → cache vào singletons: _collection, _bm25, _bm25_by_category, _texts_by_category
```

Idempotent: nếu collection đã có docs → skip reload (trừ khi `force_reload=true`).

### 2.3 Startup — Rebuild BM25

Khi server restart, collection ChromaDB vẫn còn nhưng BM25 indexes (in-memory) mất. `_build_bm25_from_collection()` đọc lại từ ChromaDB, re-detect categories (nếu metadata có thể thiếu), xây lại indexes.

---

## 3. Pipeline xử lý tin nhắn

### 3.1 START Router

**Vị trí:** `src/graph/main_graph.py` — `route_from_start()`

Ba tầng ưu tiên:

```
1. awaiting_profile_confirm=True
   → advisor_profile_update
   (user đang phản hồi sau khi bot hiển thị profile đã lưu)

2. required_fields NOT NULL AND missing_fields NOT EMPTY
   → advisor_collect_info
   (tiếp tục multi-turn thu thập thông tin trong session hiện tại)

3. else
   → intent_classifier
```

### 3.2 Intent Classifier

**Vị trí:** `src/graph/nodes/intent_classifier.py`

```
Input: last user message + lịch sử (4 turns × 800 chars)
  ↓
LLM (DeepSeek-R1, temperature=0.0) — few-shot classification
  ↓
Output: intent ∈ {GREETING_FAREWELL, PERSONAL_UNRELATED, PRODUCT_INFO_QA,
                  PRODUCT_CONSULT, CUSTOMER_FEEDBACK}
  Fallback: PRODUCT_INFO_QA nếu parse fail
```

Sau khi có intent, `route_by_intent()` phân luồng:
- `PRODUCT_CONSULT` → `advisor_domain_detector` (bỏ qua cache)
- `CUSTOMER_FEEDBACK` → `customer_feedback` (bỏ qua cache — phản hồi mang tính cá nhân)
- Các intent còn lại → `cache_check`

**Ranh giới PRODUCT_INFO_QA vs PRODUCT_CONSULT (v3.3):**

Boundary quan trọng nhất, hay bị nhầm:

| Câu hỏi | Intent đúng | Lý do |
|---|---|---|
| "Quy trình vay mua nhà tại VIB như thế nào?" | PRODUCT_INFO_QA | Hỏi về process/thông tin cụ thể |
| "Điều kiện để mở thẻ tín dụng VIB là gì?" | PRODUCT_INFO_QA | Hỏi tra cứu thông tin |
| "VIB có những loại thẻ nào?" | PRODUCT_INFO_QA | Hỏi danh sách sản phẩm |
| "Tôi muốn vay mua nhà, nên chọn gói nào?" | PRODUCT_CONSULT | Xin tư vấn chọn sản phẩm |
| "Không biết nên chọn thẻ nào phù hợp" | PRODUCT_CONSULT | Chưa biết chọn, cần recommend |

Rule trong prompt: câu hỏi bắt đầu bằng "quy trình", "điều kiện", "phí", "lãi suất", "có những loại nào" → PRODUCT_INFO_QA dù có nhắc đến sản phẩm cụ thể. Từ khoá PRODUCT_CONSULT: "nen chon loai nao", "tu van cho toi", "recommend", "phu hop voi toi".

### 3.3 Cache Check Node

**Vị trí:** `src/graph/nodes/cache_check.py`

Cache check là **node trong graph**, chỉ áp dụng cho non-PRODUCT_CONSULT intents.

```
Input: last user message
  ↓
ConversationStore.search_similar(question)
  ↓
HIT (score ≥ 0.8):
  → set messages=[AIMessage(cached_answer)], cache_hit=True
  → END (không xử lý thêm)

MISS:
  → cache_hit=False
  → route_after_cache_check → greeting / personal_unrelated / rag_rewrite
                             → customer_feedback  (nếu intent=CUSTOMER_FEEDBACK)
```

**Intents bỏ qua cache hoàn toàn (không qua cache_check_node):**
- `PRODUCT_CONSULT`: tư vấn multi-turn, phụ thuộc profile KH — cached answer từ user khác không có giá trị.
- `CUSTOMER_FEEDBACK`: phản hồi mang tính cá nhân theo ngữ cảnh hội thoại — không nên cache.

### 3.4 GREETING_FAREWELL Pipeline

```
cache_check → MISS
    ↓
greeting_node
    │
    ├─ _is_farewell(text)?  ← keyword matching trước khi gọi LLM
    │   Keywords: "tạm biệt", "bye", "ok bye", "ok, bye", "hẹn gặp lại", ...
    │
    ├─ YES (farewell) → FAREWELL_PROMPT → LLM (temp=0.3)
    │       → Cảm ơn + hẹn gặp lại (static fallback nếu LLM lỗi)
    │
    └─ NO (greeting)  → GREETING_PROMPT → LLM (temp=0.3)
            → Chào + giới thiệu bot + hỏi KH cần gì
    → END
```

**Lý do dùng keyword pre-classification:** LLM với single-prompt có điều kiện hay bỏ qua instruction "nếu là tạm biệt thì..." và luôn generate kiểu chào hỏi. Keyword matching deterministic, không phụ thuộc vào LLM judgment.

### 3.5 PERSONAL_UNRELATED Pipeline

```
cache_check → MISS
    ↓
personal_unrelated_node → LLM (temp=0.4)
    Prompt 3 bước:
    1. Đồng cảm ngắn gọn (1 câu)
    2. Liên kết tự nhiên sang sản phẩm VIB phù hợp nhất
    3. Hỏi KH có quan tâm không
    → END
```

### 3.8 CUSTOMER_FEEDBACK Pipeline

**Vị trí:** `src/graph/nodes/customer_feedback.py`

Intent này **bỏ qua cache** hoàn toàn — phản hồi mang tính cá nhân, phụ thuộc ngữ cảnh.

```
(bỏ qua cache)
    ↓
customer_feedback_node
  Input: last user message + lịch sử hội thoại (4 turns)
  LLM (temp=0.3) phân tích:
    sentiment ∈ {NEGATIVE, POSITIVE, NEUTRAL}
    đối tượng: về chatbot/câu trả lời hay về sản phẩm VIB
  ↓
  NEGATIVE (bực bội, không hài lòng, phàn nàn):
    → Xin lỗi chân thành, đồng cảm
    → Cam kết ghi nhận và cải thiện
    → Về sản phẩm: hướng đến hotline/chi nhánh nếu cần giải quyết
    → Giọng điệu: nhỏ nhẹ, cầu thị

  POSITIVE (khen ngợi, hài lòng):
    → Cảm ơn chân thành, khiêm tốn
    → Hỏi KH còn cần hỗ trợ gì không

  NEUTRAL (trung tính, không rõ):
    → Ghi nhận, hỏi thêm chi tiết
  → END
```

**Ví dụ:**
```
KH: "Tại sao bạn hỏi tôi mục đích vay trong khi tôi đã nói là vay mua nhà?"
→ CUSTOMER_FEEDBACK, NEGATIVE

Bot: "Mình xin lỗi vì đã làm bạn không hài lòng! Bạn hoàn toàn đúng —
     bạn đã nói rõ là vay mua nhà ngay từ đầu mà mình lại hỏi lại.
     Mình sẽ ghi nhận để cải thiện. Bạn có muốn mình tư vấn ngay gói
     vay mua nhà phù hợp với bạn không ạ?"

KH: "Bot trả lời rất hữu ích và dễ hiểu!"
→ CUSTOMER_FEEDBACK, POSITIVE

Bot: "Cảm ơn bạn rất nhiều vì lời khen! Mình còn nhiều điều cần học hỏi
     và sẽ cố gắng phục vụ bạn tốt hơn. Bạn cần mình hỗ trợ thêm gì không ạ?"
```

### 3.6 PRODUCT_INFO_QA — RAG Pipeline

```
cache_check → MISS
    ↓
[rag_rewrite_node]
  Input: câu hỏi + lịch sử (4 turns × 800 chars)
  LLM (temp=0.0): làm rõ câu hỏi, thêm từ khóa chuyên ngành
  Output: rewritten_query
    ↓
[rag_retrieve_node]
  detect_category(rewritten_query) → category_filter
    category ∈ {credit_card, insurance, loan, savings} → filter
    category = "general" → no filter (tìm toàn bộ corpus)
  hybrid_retrieve(query=rewritten_query, top_k=6, category_filter=cat)
  Output: retrieved_docs (top-6, có category field)
    ↓
[rag_generate_node]
  Context = format_context(docs) — tối đa 6000 chars
  LLM (temp=0.05), STRICT grounding (v3.3 — tăng cường):
    ✅ Số liệu (lãi suất, phí, kỳ hạn...) phải lấy NGUYÊN VĂN từ tài liệu
    ✅ Danh sách sản phẩm chỉ liệt kê những gì tài liệu đề cập
    ❌ Không bịa số liệu, không tự tính toán/quy đổi (VD: không tự đổi %/tháng → %/năm)
    ❌ Không dùng kiến thức nền (training data) để bổ sung ngoài tài liệu
    → Không tìm thấy: "không tìm thấy" + hotline 1800 8180
    → END
```

### 3.7 PRODUCT_CONSULT — Advisory Pipeline

```
(bỏ qua cache)
    ↓
[advisor_domain_detector]
  LLM (temp=0.0)
  Output: domain ∈ {credit_card, insurance, loan, savings, general}
    ↓
[advisor_profile_recall]  ← V3 MỚI
  Xem section 3.8
    ↓ (nếu không có profile)
[advisor_field_extractor]
  get_fields(domain) → required_fields dict
  LLM (temp=0.0, num_ctx=8192): pre-extract info từ initial question
  Output: {required_fields, collected_info=pre_filled, missing_fields, turn_count=0}
    ↓
[advisor_collect_info_node]  ← MULTI-TURN LOOP
  turn_count=0:
    missing_fields=[] → "mình tìm ngay" → advisor_retrieve
    missing_fields!=[] → hỏi field đầu tiên còn thiếu (với intro) → END
  turn_count>0:
    lưu last_user_msg → collected_info[missing_fields[0]]
    cập nhật missing_fields
    còn thiếu → hỏi tiếp → END
    hết thiếu hoặc hết turns → advisor_retrieve
    ↓
[advisor_retrieve_node]
  query = "{domain_keyword} {collected_info values}"
  hybrid_retrieve(top_k=4, domain_hint=domain_kw, category_filter=domain)
  Output: retrieved_docs (top-4, chỉ trong category domain)
    ↓
[advisor_recommend_node]
  _docs_are_relevant(domain, docs) → False: NOT_FOUND_TEMPLATE + hotline
  LLM (temp=0.1), STRICT grounding → recommendation
  → save_profile(user_id, domain, collected_info)  ← lưu profile
  → END
```

### 3.8 Advisor Profile Recall & Update

#### Node 1: `advisor_profile_recall_node`

Chèn giữa `domain_detector` và `field_extractor`.

```
Input: user_id, domain
  ↓
AdvisorProfileStore.load_profile(user_id, domain)
  ↓
Profile tồn tại:
  → set collected_info=profile, required_fields, missing_fields=[]
  → set awaiting_profile_confirm=True
  → messages=[AIMessage(confirm_message)]
  → END (chờ user phản hồi)

Không có profile:
  → return {}  → route sang advisor_field_extractor
```

**Confirm message format:**

```
"Mình thấy lần trước bạn đã tư vấn về {domain_label} và cung cấp:
 - {field_label}: {value}
 - ...
Thông tin này vẫn còn đúng không ạ?
- Nếu vẫn vậy, nhắn "OK" là mình tư vấn ngay!
- Nếu có thay đổi, bạn cứ nói cho mình biết."
```

#### Node 2: `advisor_profile_update_node`

Gọi khi `awaiting_profile_confirm=True` (ưu tiên cao nhất ở START router).

```
Input: current collected_info + user_response (last HumanMessage)
  ↓
LLM (temp=0.0) trích xuất:
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
  → "Thông tin vẫn vậy, để mình tư vấn ngay!" 
  → awaiting_profile_confirm=False
  → route: advisor_retrieve

confirmed=True, updates={...}:
  → merge updates vào collected_info
  → AdvisorProfileStore.update_profile(user_id, domain, updates)
  → "Đã cập nhật (fields). Để mình tư vấn ngay!"
  → awaiting_profile_confirm=False
  → Kiểm tra missing_fields → route: advisor_collect_info hoặc advisor_retrieve
```

---

## 4. Hybrid Retrieval

**Vị trí:** `src/retrieval/retriever.py` — `hybrid_retrieve()`

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

### Lý do

V2 tìm kiếm toàn bộ corpus (~5000+ chunks). Khi tư vấn thẻ tín dụng, chunks về bảo hiểm/vay cũng được trả về do semantic overlap với từ chung như "ngân hàng", "VIB", "sản phẩm".

V3 gán `category` metadata cho mỗi chunk. ChromaDB query và BM25 đều được filter — chỉ tìm trong đúng danh mục.

### Kết quả

| Scenario | V2 | V3 |
|---|---|---|
| Advisor tư vấn thẻ | Tìm trong 5000+ chunks | Chỉ tìm trong chunks credit_card |
| RAG hỏi lãi suất vay | Tìm trong 5000+ chunks | Chỉ tìm trong chunks loan |
| RAG câu hỏi chung | Tìm trong 5000+ chunks | Vẫn tìm tất cả (no filter) |

### Nơi dùng category_filter

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

**Vị trí:** `src/history/advisor_profile_store.py`

### Storage format

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
      "_updated_at":          "2026-06-07T14:00:00"
    },
    "loan": {
      "muc_dich_vay": "mua nhà",
      ...
    }
  }
}
```

### API

```python
save_profile(user_id, domain, collected_info)
  # Ghi đè toàn bộ profile của domain
  # Gọi bởi: advisor_recommend_node sau khi recommendation hoàn thành

load_profile(user_id, domain) → dict | None
  # Trả về dict fields (không có _updated_at), hoặc None nếu chưa có
  # Gọi bởi: advisor_profile_recall_node

update_profile(user_id, domain, updates)
  # Merge updates vào profile hiện có
  # Gọi bởi: advisor_profile_update_node khi user thay đổi 1 số field
```

### Lifecycle

```
1. Lần đầu tư vấn credit_card:
   → field_extractor → collect_info → retrieve → recommend
   → recommend xong: save_profile(uid, "credit_card", collected_info)

2. Lần sau hỏi credit_card:
   → profile_recall: load_profile(uid, "credit_card") → tìm thấy
   → hiện confirm message → END

3. User phản hồi "OK" hoặc có update:
   → profile_update: merge updates nếu có → advisor_retrieve (không hỏi lại)

4. User muốn hỏi lại từ đầu ("không, tôi hỏi lại"):
   → profile_update: confirmed=False → field_extractor → collect_info mới
```

---

## 7. Conversation History & Cache

### Lưu trữ Q&A

```
data/conversations/{user_id}.json
{
  "user_id": "UID001",
  "entries": [{
    "id": "uuid4",
    "timestamp": "2026-06-03T19:15:08",
    "question": "lãi suất thẻ VIB Super Card?",
    "answer": "Theo tài liệu...",
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

### Khi nào save?

```python
# chat.py — sau khi graph invoke:
is_collecting = bool(result.get("missing_fields"))  # đang thu thập info

if not is_collecting and not cache_hit and answer:
    store.save(user_id, question, answer, intent, domain, session_id)
```

Không save nếu:
- Đang trong advisor multi-turn (chưa có final answer)
- Câu trả lời từ cache (tránh duplicate)

### Startup: load_all_history() — Bi-directional Sync

Khi server khởi động, `ConversationStore.load_all_history()` thực hiện **sync 2 chiều** giữa JSON files và ChromaDB:

```
1. Thu thập tất cả entry IDs hợp lệ từ JSON files
   (bỏ qua _profiles.json, bỏ qua entries thiếu question/answer)

2. So sánh với IDs đang có trong ChromaDB:
   stale_ids = chroma_ids - json_valid_ids
   → col.delete(ids=stale_ids)   ← xóa entries đã bị xóa khỏi JSON

3. entries_to_add = json_valid_ids - chroma_ids
   → embed + col.add(...)        ← add entries mới chưa được index
```

**Trước đây (append-only):** Khi xóa entry khỏi JSON và restart server, entry đó vẫn còn trong ChromaDB → cache vẫn trả về kết quả từ entry đã xóa.

**Hiện tại (bi-directional):** Restart server tự động dọn sạch stale entries.

**Force rebuild không cần restart:**

```bash
POST /admin/rebuild-history
# → Xóa toàn bộ ChromaDB history
# → Rebuild từ đầu theo JSON files hiện tại
# → Trả về stats trước/sau để verify
```

Dùng khi cần sync ngay sau khi edit JSON thủ công mà không muốn restart server.

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
    awaiting_profile_confirm : Optional[bool]   # V3: True khi chờ user confirm profile

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
customer_feedback           ← V3.2 MỚI
rag_rewrite
rag_retrieve
rag_generate
advisor_domain_detector
advisor_profile_recall      ← V3 MỚI
advisor_profile_update      ← V3 MỚI
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
advisor_profile_recall → [route_after_profile_recall] → END                    (has profile)
                                                       → advisor_field_extractor (no profile)

advisor_profile_update → [route_after_profile_update] → advisor_collect_info (missing fields)
                                                       → advisor_retrieve      (complete)

advisor_field_extractor → advisor_collect_info
advisor_collect_info → [route_after_collect_info] → END              (still collecting)
                                                   → advisor_retrieve (done)

advisor_retrieve → advisor_recommend → END
```

### MemorySaver

State persist in-memory theo `session_id` (thread_id trong LangGraph). Restart server → mất advisor multi-turn state (nhưng JSON history và profiles không mất — load lại khi cần).

Để persist qua restart, dùng SqliteSaver (xem section [Mở rộng](#12-mở-rộng)).

---

## 9. API Layer

### Endpoints

```
GET  /                          — health check cơ bản
GET  /admin/health              — vectorstore ready, model info, category counts
GET  /admin/stats               — chunk count, category breakdown, history stats
POST /admin/load                — load/reload parquet → ChromaDB
POST /admin/load?force_reload=true — force reload (cần sau khi thay parquet mới)
GET  /admin/history/{user_id}   — lịch sử Q&A của user (admin)
POST /admin/rebuild-history     — xóa ChromaDB history + rebuild từ JSON (dùng sau khi edit JSON thủ công)

POST /chat/                     — main chat endpoint
GET  /chat/session/{session_id} — state session hiện tại
GET  /chat/history/{user_id}    — lịch sử chat của user
POST /chat/new                  — tạo session mới
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
  "answer"           : "Để tư vấn thẻ phù hợp...",
  "intent"           : "PRODUCT_CONSULT",
  "advisor_domain"   : "credit_card",
  "sources"          : [],
  "turn_count"       : 1,
  "collected_info"   : {"thu_nhap_hang_thang": "15 triệu"},
  "from_cache"       : false,
  "cache_similarity" : null
}
```

### Xử lý trong chat.py

```python
result = graph.invoke(
    {"messages": [HumanMessage(content=message)], "session_id": session_id, "user_id": user_id},
    config={"configurable": {"thread_id": session_id}},
)

# Lưu Q&A nếu có final answer và không đang thu thập
missing_fields = result.get("missing_fields") or []
is_collecting = bool(missing_fields)
if not is_collecting and not cache_hit and answer:
    store.save(user_id, question, answer, intent, advisor_domain, session_id)
```

---

## 10. Sơ đồ toàn bộ luồng dữ liệu

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
│         │         NEGATIVE → xin lỗi, đồng cảm, cam kết         │
│         │         POSITIVE → cảm ơn, khiêm tốn                  │
│         │         NEUTRAL  → ghi nhận, hỏi thêm                 │
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

## 11. Cấu hình quan trọng

| Key trong `.env` | Default | Ý nghĩa |
|---|---|---|
| `LLM_MODEL` | `deepseek-r1:8b` | Model LLM cho tất cả tasks (RAG, intent, advisory, cache gate) |
| `OLLAMA_NUM_GPU` | `-1` | -1=all GPU, 0=CPU only |
| `LLM_NUM_CTX` | `4096` | Context window cho main LLM |
| `EMBEDDING_MODEL` | `bge-m3:latest` | Model embedding (dim=1024) |
| `CHROMA_COLLECTION_NAME` | `vib_products_v3` | Collection có category metadata |
| `TOP_K_RETRIEVAL` | `6` | Số docs lấy trong RAG |
| `TOP_K_FINAL` | `4` | Số docs lấy trong advisor |
| `SEMANTIC_WEIGHT` | `0.65` | Trọng số semantic trong RRF |
| `BM25_WEIGHT` | `0.35` | Trọng số BM25 trong RRF |
| `CACHE_SIMILARITY_THRESHOLD` | `0.8` | Ngưỡng cache hit |
| `CACHE_TOP_K` | `5` | Số candidates cache lookup |
| `MAX_ADVISOR_TURNS` | `8` | Max lượt hỏi trong advisory |
| `EVAL_JUDGE_MODEL` | `""` | Model dùng cho LLM-as-judge trong eval. Để trống → dùng `LLM_MODEL`. Nên dùng model không phải reasoning để tránh `<think>` token errors |

**Lưu ý về `get_fast_llm()` (v3.3):**
- Dùng `settings.llm_model` (deepseek-r1:8b) — không dùng `cache_verify_model` riêng nữa
- `num_ctx` tăng từ 512 → **2048** để đủ chỗ cho `<think>` tokens của DeepSeek-R1 (thinking chiếm 200–500 tokens trước khi trả lời)
- `cache_verify_model` vẫn còn trong settings nhưng không được `get_fast_llm()` sử dụng

---

## 12. Mở rộng

### Thêm domain tư vấn mới

1. Thêm keywords vào `_CATEGORY_KEYWORDS` trong `src/data/loader.py`.
2. Thêm vào `DOMAIN_FIELDS`, `DOMAIN_KEYWORDS`, `DOMAIN_LABELS` trong `field_definitions.py`.
3. Thêm domain vào `DOMAIN_PROMPT` trong `domain_detector.py`.
4. Force reload: `POST /admin/load?force_reload=true`.

### Persist advisor session qua restart

Hiện tại dùng `MemorySaver` (in-memory). Để persist qua restart:

```python
# main_graph.py
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string("./data/checkpoints.db")
graph = builder.compile(checkpointer=memory)
```

### Tăng chất lượng RAG

```
TOP_K_RETRIEVAL=10          # lấy nhiều candidates hơn
SEMANTIC_WEIGHT=0.75        # tăng trọng số semantic cho corpus chuyên ngành
```

### Thay LLM

```
LLM_MODEL=llama3.2:3b       # nhanh hơn (~3x)
LLM_MODEL=deepseek-r1:14b   # chính xác hơn
```

### Mở rộng Profile Store

Hiện tại profile không có expiry. Để thêm TTL:

```python
# advisor_profile_store.py — load_profile()
from datetime import datetime, timedelta

updated_at = datetime.fromisoformat(profile["_updated_at"])
if datetime.now() - updated_at > timedelta(days=90):
    return None  # profile quá cũ, hỏi lại
```

### Horizontal scaling

- Tách ChromaDB ra service riêng (ChromaDB HTTP server mode)
- Thay MemorySaver bằng Redis-backed checkpointer
- Load balance FastAPI với `uvicorn --workers N`
