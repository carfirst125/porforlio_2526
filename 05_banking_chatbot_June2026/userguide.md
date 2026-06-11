# VIB Chatbot V3 — Hướng dẫn sử dụng

## Yêu cầu hệ thống

- Python 3.10+
- [Ollama](https://ollama.com) đã cài và đang chạy
- Models Ollama đã pull:
  ```
  ollama pull deepseek-r1:8b
  ollama pull bge-m3
  ```
- File `documents_bgem3.parquet` tại thư mục cha (`../documents_bgem3.parquet`)

---

## Khởi động nhanh

### Bước 1 — Cài dependencies (lần đầu)
```bash
cd version_3
pip install -r requirements.txt
```

### Bước 2 — Tạo file cấu hình
```bash
copy .env.example .env
# Chỉnh .env nếu cần (model, port, GPU settings...)
```

### Bước 3 — Chạy Ollama
```bash
ollama serve
```

### Bước 4 — Khởi động API (Terminal 1)
```bash
cd version_3
python -m uvicorn src.api.main:app --reload
# Hoặc: double-click start_api.bat
```

Khi khởi động lần đầu, server sẽ:
- Load parquet → index vào ChromaDB collection `vib_products_v3`
- Tự động phát hiện **category** (thẻ / bảo hiểm / vay / tiết kiệm) cho mỗi chunk
- Xây dựng BM25 index toàn cục + BM25 riêng cho từng category
- **Sync lịch sử Q&A** (bi-directional): xóa entries ChromaDB không còn trong JSON, add entries mới → ChromaDB history (để cache lookup)
- Quá trình này mất khoảng 30–60 giây

### Bước 5 — Khởi động Streamlit (Terminal 2)
```bash
cd version_3
python -m streamlit run frontend/app.py
# Hoặc: double-click start_streamlit.bat
```

Truy cập: **http://localhost:8501**  
API docs: **http://localhost:8000/docs**

---

## Tính năng chính V3

### Category Filtering

V3 gán nhãn **category** cho từng đoạn tài liệu trong ChromaDB:

| Category | Loại nội dung |
|---|---|
| `credit_card` | Thẻ tín dụng, thẻ ghi nợ, cashback, dặm bay... |
| `insurance` | Bảo hiểm nhân thọ, sức khỏe, tai nạn... |
| `loan` | Vay mua nhà, vay tiêu dùng, tín chấp, thế chấp... |
| `savings` | Tiết kiệm, lãi suất gửi, kỳ hạn gửi... |
| `general` | Thông tin chung VIB, dịch vụ khác |

Khi tư vấn thẻ tín dụng, chatbot chỉ tìm trong chunks `credit_card` — nhanh hơn và không bị nhiễu bởi tài liệu không liên quan.

### Advisor Profile — Nhớ thông tin khách hàng

Sau khi tư vấn xong, chatbot **lưu lại thông tin KH** (thu nhập, nhu cầu, v.v.) per user + domain.  
Lần sau KH hỏi cùng chủ đề → chatbot nhắc lại và chỉ hỏi nếu có thay đổi:

```
KH: "Tôi muốn hỏi thêm về thẻ tín dụng"
Bot: "Mình thấy lần trước bạn đã tư vấn về Thẻ tín dụng:
      - Thu nhập hàng tháng: 15 triệu
      - Chi tiêu chủ yếu: mua sắm online, ăn uống
      - Quyền lợi ưu tiên: cashback
      Thông tin này vẫn còn đúng không ạ? ..."
KH: "Vẫn vậy nhưng thu nhập giờ 20 triệu rồi"
Bot: "Đã cập nhật thu nhập. Để mình tư vấn ngay!" → recommendation
```

### Cache thông minh

Câu hỏi đã được trả lời trước đó → chatbot trả lời ngay, không gọi LLM:
- **Similarity ≥ 0.8** → trả về cached answer (**⚡ rất nhanh**)
- **Similarity < 0.8** → xử lý bình thường

Cache dùng **hybrid search**: 65% semantic + 35% keyword. Lưu ý: intent PRODUCT_CONSULT (tư vấn) **không dùng cache** — tư vấn phụ thuộc profile KH cụ thể.

---

## Cách sử dụng Streamlit

### UserID

- Nhập UserID ở sidebar (mặc định `UID0000`)
- Mỗi UserID có lịch sử hội thoại và profile tư vấn riêng
- Đổi UserID → chat history reset, session mới

**Dữ liệu được lưu per UserID:**
- `data/conversations/{user_id}.json` — lịch sử Q&A
- `data/conversations/{user_id}_profiles.json` — thông tin tư vấn đã thu thập

### Sidebar — Chức năng

| Nút | Tác dụng |
|---|---|
| Load data | Load parquet vào ChromaDB (bỏ qua nếu đã load) |
| Reload | Force reload (cần khi thay parquet mới) |
| New Chat | Tạo session mới, xóa lịch sử hiển thị |
| Kiểm tra lại kết nối | Ping API health check |

### Badges trên mỗi câu trả lời

| Badge | Ý nghĩa |
|---|---|
| ⚡ Cache (0.92) | Câu trả lời từ cache — không gọi LLM |
| 👋 Chào hỏi | Intent: chào/tạm biệt |
| 💬 Chủ đề cá nhân | Intent: chia sẻ cá nhân, redirect sang sản phẩm |
| 🔍 Tra cứu thông tin | Intent: hỏi thông tin sản phẩm → RAG pipeline |
| 🎯 Tư vấn sản phẩm | Intent: tư vấn chọn sản phẩm → Advisory pipeline |
| 📝 Phản hồi KH | Intent: feedback về chatbot hoặc sản phẩm VIB |
| 💳 Thẻ tín dụng | Domain tư vấn: tìm trong category `credit_card` |
| 🛡️ Bảo hiểm | Domain tư vấn: tìm trong category `insurance` |
| 🏠 Vay vốn | Domain tư vấn: tìm trong category `loan` |
| 💰 Tiết kiệm | Domain tư vấn: tìm trong category `savings` |

---

## 5 loại câu hỏi và cách chatbot xử lý

### 1. Chào hỏi / Tạm biệt
```
KH: "Xin chào"
Bot: "Chào bạn! Mình là trợ lý AI của VIB. Bạn cần hỗ trợ gì hôm nay?"

KH: "Cảm ơn bạn nhé, tạm biệt"
Bot: "Cảm ơn bạn đã liên hệ với VIB! Chúc bạn ngày vui..."
```

### 2. Chia sẻ cá nhân
```
KH: "Tôi vừa được thăng chức, lương tăng rồi"
Bot: "Chúc mừng bạn! Với mức lương mới, bạn có thể xem xét thẻ tín dụng
      hạng cao của VIB để tích lũy điểm thưởng tốt hơn. Bạn có muốn
      tìm hiểu thêm không?"
```

### 3. Hỏi thông tin sản phẩm (RAG)
```
KH: "Lãi suất thẻ VIB Super Card là bao nhiêu?"
→ detect_category("thẻ VIB Super Card") = "credit_card"
→ Tìm trong ChromaDB category=credit_card + BM25[credit_card]
Bot: [Trả lời từ tài liệu thẻ tín dụng]

KH: "Điều kiện vay mua nhà tại VIB?"
→ detect_category("vay mua nhà") = "loan"
→ Tìm trong ChromaDB category=loan + BM25[loan]
Bot: [Trả lời từ tài liệu vay vốn]
```

**Quy tắc grounding (v3.3):** Số liệu (lãi suất, phí, kỳ hạn, hạn mức) được lấy **nguyên văn từ tài liệu** — bot không tự tính toán hay quy đổi. Nếu tài liệu ghi "%/tháng", bot trả lời "%/tháng", không tự đổi sang "%/năm". Nếu không tìm thấy thông tin trong tài liệu, bot chuyển hướng KH đến hotline 1800 8180.

### 5. Phản hồi khách hàng (CUSTOMER_FEEDBACK)
```
--- Trường hợp 1: KH không hài lòng ---
KH: "Tại sao bạn hỏi tôi mục đích vay trong khi tôi đã nói là vay mua nhà rồi?"
→ Phát hiện CUSTOMER_FEEDBACK, sentiment=NEGATIVE

Bot: "Mình xin lỗi vì đã làm bạn không hài lòng! Bạn hoàn toàn đúng, bạn
      đã nói rõ là vay mua nhà từ đầu mà mình lại hỏi lại. Mình sẽ ghi
      nhận để cải thiện. Bạn có muốn mình tư vấn ngay gói vay mua nhà
      phù hợp với bạn không ạ?"

--- Trường hợp 2: KH hài lòng ---
KH: "Chatbot trả lời rất hữu ích và dễ hiểu!"
→ CUSTOMER_FEEDBACK, sentiment=POSITIVE

Bot: "Cảm ơn bạn rất nhiều vì lời khen! Mình còn nhiều điều cần học hỏi
      và sẽ cố gắng phục vụ tốt hơn. Bạn cần mình hỗ trợ thêm gì không?"

--- Trường hợp 3: KH phàn nàn về sản phẩm ---
KH: "Tôi thấy lãi suất VIB cao quá"
→ CUSTOMER_FEEDBACK, sentiment=NEGATIVE (về sản phẩm)

Bot: "Mình cảm ơn bạn đã chia sẻ! Mình hiểu bạn đang băn khoăn về lãi
      suất. Để được tư vấn cụ thể và tìm gói vay phù hợp nhất, bạn có
      thể liên hệ hotline 1800 8180 (miễn phí, 24/7) hoặc để mình hỗ
      trợ tìm gói tốt hơn cho bạn nhé!"
```

### 4. Tư vấn sản phẩm — Lần đầu (Advisory multi-turn)
```
KH: "Tôi muốn mở thẻ tín dụng nhưng không biết chọn loại nào"
→ domain_detector: "credit_card"
→ profile_recall: chưa có profile → field_extractor → collect_info

Bot: "Để tư vấn thẻ phù hợp, mình cần hỏi thêm:
      Thu nhập hàng tháng của bạn khoảng bao nhiêu?"
KH: "Khoảng 15 triệu"
Bot: "Bạn chi tiêu chủ yếu vào lĩnh vực nào?"
KH: "Mua sắm online và ăn uống"
Bot: "Bạn ưu tiên quyền lợi nào: cashback, tích điểm hay trả góp 0%?"
KH: "Cashback"
Bot: "Bạn đã có thẻ tín dụng nào chưa?"
KH: "Chưa có"
→ advisor_retrieve: tìm trong category=credit_card
→ advisor_recommend: đề xuất sản phẩm phù hợp
→ save_profile: lưu thông tin KH

Bot: "Dựa trên thu nhập 15 triệu và chi tiêu mua sắm online với ưu tiên
      cashback, thẻ VIB Online Plus phù hợp với bạn vì..."
```

**Lưu ý:** Nếu câu hỏi ban đầu đã cung cấp sẵn thông tin, chatbot sẽ **tự extract** và chỉ hỏi những thông tin còn thiếu.

### 4b. Tư vấn sản phẩm — Lần sau (Profile Recall)
```
(Cùng UserID, hỏi lại thẻ tín dụng)

KH: "Cho tôi tư vấn thẻ tín dụng"
→ domain_detector: "credit_card"
→ profile_recall: tìm thấy profile credit_card của UID này!

Bot: "Mình thấy lần trước bạn đã tư vấn về Thẻ tín dụng và cung cấp:
      - Thu nhập hàng tháng: 15 triệu
      - Chi tiêu chủ yếu: mua sắm online, ăn uống
      - Quyền lợi ưu tiên: cashback
      - Thẻ hiện tại: chưa có
      Thông tin này vẫn còn đúng không ạ?
      - Nếu vẫn vậy, nhắn "OK" là mình tư vấn ngay!
      - Nếu có thay đổi, bạn cứ nói cho mình biết."

--- Trường hợp 1: KH xác nhận ---
KH: "OK đúng rồi"
→ profile_update: confirmed=True, no updates
→ advisor_retrieve → recommend ngay (không hỏi lại)
Bot: [Recommendation dựa trên profile cũ]

--- Trường hợp 2: KH cập nhật ---
KH: "Thu nhập tôi giờ 25 triệu rồi, còn lại vẫn vậy"
→ profile_update: merge {thu_nhap_hang_thang: "25 triệu"}
→ advisor_retrieve → recommend với thông tin mới
Bot: "Đã cập nhật thu nhập. Để mình tư vấn ngay! ⏳"
     [Recommendation với thu nhập mới]

--- Trường hợp 3: KH muốn hỏi lại ---
KH: "Thôi tôi muốn hỏi lại từ đầu"
→ profile_update: confirmed=False → reset → field_extractor
Bot: [Hỏi lại từ đầu như lần đầu]
```

---

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/admin/health` | Health check — vectorstore, models, category counts |
| GET | `/admin/stats` | Stats: chunk count, category breakdown, history count |
| POST | `/admin/load` | Load parquet → ChromaDB với category metadata |
| POST | `/admin/load?force_reload=true` | Force reload (cần sau khi thay parquet) |
| POST | `/admin/rebuild-history` | Xóa + rebuild ChromaDB history từ JSON (dùng sau khi xóa/sửa JSON thủ công mà không muốn restart) |
| POST | `/chat/` | Gửi tin nhắn |
| GET | `/chat/history/{user_id}` | Lịch sử Q&A của user |
| GET | `/chat/session/{session_id}` | State session (debug) |
| GET | `/docs` | Swagger UI |

---

## Cấu trúc thư mục

```
version_3/
├── config/settings.py              # Tất cả cấu hình (pydantic-settings)
├── src/
│   ├── llm.py                      # LLM + Embedding factory + parse_json
│   ├── data/loader.py              # Parquet → ChromaDB + BM25 (category metadata)
│   ├── retrieval/retriever.py      # Hybrid retrieval (BM25 + Semantic + RRF + category_filter)
│   ├── knowledge_graph/
│   │   └── field_definitions.py    # Domain fields cho advisory
│   ├── history/
│   │   ├── conversation_store.py   # JSON Q&A history + ChromaDB cache
│   │   └── advisor_profile_store.py  # ← Lưu collected_info per user+domain
│   ├── graph/
│   │   ├── state.py                # ChatState (user_id, awaiting_profile_confirm)
│   │   ├── main_graph.py           # LangGraph assembly + routing
│   │   └── nodes/
│   │       ├── intent_classifier.py
│   │       ├── greeting.py
│   │       ├── personal_unrelated.py
│   │       ├── cache_check.py      # Cache lookup node in-graph
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
├── VERSION_3_PLAN.md
├── technote.md
└── userguide.md                    # File này
```

---

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách fix |
|---|---|---|
| ❌ Không kết nối được API | Server chưa chạy | Chạy `start_api.bat` |
| ⚠️ ChromaDB chưa load | Parquet chưa được index | Nhấn "Load data" trong sidebar |
| ⏱️ Response rất chậm | Ollama chạy CPU | Kiểm tra `OLLAMA_NUM_GPU=-1`, chạy `python scripts/check_gpu.py` |
| ❌ Model not found | Model chưa pull | `ollama pull deepseek-r1:8b && ollama pull bge-m3` |
| 🔄 Không có category counts trong `/admin/stats` | Đang dùng collection cũ (không phải `vib_products_v3`) | Force reload: `POST /admin/load?force_reload=true` |
| 🔄 Bot hỏi lại thông tin đã cung cấp trước đó | Session state mất (server restart) | Profile vẫn lưu trong file — bắt đầu câu hỏi mới để recall |
| ⚡ Cache không hoạt động | History ChromaDB chưa load | Restart server — `load_all_history()` chạy tự động khi startup |
| 🔄 Bot không nhớ profile dù đã tư vấn | UserID khác nhau | Kiểm tra UserID trong sidebar — phải dùng cùng UserID |
| 🔄 Sau khi xóa entry khỏi JSON, bot vẫn trả lời từ cache | ChromaDB history chưa được sync | Restart server (tự sync) hoặc gọi `POST /admin/rebuild-history` để sync ngay không cần restart |
| ❌ Cache node lỗi / trả về sai | Dùng model không phải deepseek-r1:8b cho cache check (biến `CACHE_VERIFY_MODEL` cũ) | Từ v3.3, `get_fast_llm()` luôn dùng `LLM_MODEL` — xóa `CACHE_VERIFY_MODEL` khỏi `.env` nếu có |

---

## Điều chỉnh hiệu năng

**Muốn response nhanh hơn:**
```
LLM_MODEL=llama3.2:3b        # model nhỏ hơn, ~3 lần nhanh hơn
OLLAMA_NUM_GPU=-1             # đảm bảo dùng GPU
```
> ⚠️ Nếu đổi `LLM_MODEL` sang model khác, `get_fast_llm()` (dùng cho cache check) cũng tự đổi theo — không cần cấu hình riêng.

**Muốn câu trả lời chính xác hơn:**
```
LLM_MODEL=deepseek-r1:14b    # model lớn hơn
TOP_K_RETRIEVAL=10           # lấy nhiều docs hơn
```

**Muốn cache hoạt động rộng hơn** (chấp nhận câu hỏi tương tự):
```
CACHE_SIMILARITY_THRESHOLD=0.75
```

**Muốn cache chặt hơn** (chỉ match câu hỏi gần giống):
```
CACHE_SIMILARITY_THRESHOLD=0.90
```

---

## Hotline hỗ trợ VIB
- **1800 8180** (miễn phí, 24/7)
- **vib.com.vn**
