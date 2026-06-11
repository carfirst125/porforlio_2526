# VIB Chatbot V3 — Evaluation Guideline

## Tổng quan

Chatbot V3 có 5 intents và 4 component pipeline chính. Mỗi component cần bộ metrics riêng.
Không thể dùng một metric chung cho toàn bộ hệ thống — lỗi ở tầng nào phải phát hiện ở tầng đó.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION SCOREBOARD                            │
├────────────────────────┬────────────────────┬───────────┬──────────┤
│ Component              │ Metric chính       │ Target    │ Step     │
├────────────────────────┼────────────────────┼───────────┼──────────┤
│ Intent Classification  │ Macro F1           │ ≥ 0.85    │ Step 1a  │
│ Cache                  │ False Positive Rate│ ≤ 0.05    │ Step 1b  │
│ Feedback Sentiment     │ Accuracy           │ ≥ 0.85    │ Step 1c  │
│ RAG Retrieval+Gen      │ Faithfulness       │ ≥ 0.80    │ Step 2   │
│ Advisory Field Collect │ Completion Rate    │ ≥ 0.90    │ Step 3   │
│ Advisory Recommend     │ Correct Rate       │ ≥ 0.70    │ Step 4   │
└────────────────────────┴────────────────────┴───────────┴──────────┘
```

---

## Phần 1 — Quy trình Evaluation

### Nguyên tắc chung

- **Evaluate theo tầng**: Làm Step 1 trước Step 2, Step 2 trước Step 3/4.
  Nếu Intent F1 < 0.85, mọi số liệu tầng sau đều vô nghĩa.
- **Dùng user_id riêng**: Tất cả eval dùng prefix `EVAL_` để không ô nhiễm dữ liệu thật.
- **API phải đang chạy**: Tất cả eval gọi qua HTTP API (`http://localhost:8000`).
- **Ollama phải đang chạy**: Một số eval dùng LLM làm judge.

---

### Step 1a — Intent Classification

**Mục tiêu**: Đo độ chính xác của intent classifier trên 5 intent.

**Metrics:**
- Accuracy tổng thể
- Precision / Recall / F1 theo từng intent
- Confusion matrix (phát hiện cặp intent hay nhầm nhau)

**Test set mẫu** (`evaluation/data/intent_samples.json`):

| Intent | Ví dụ câu | Hay nhầm với |
|---|---|---|
| GREETING_FAREWELL | "xin chào", "tạm biệt" | PERSONAL_UNRELATED |
| PERSONAL_UNRELATED | "lương tôi vừa tăng rồi" | PRODUCT_CONSULT |
| PRODUCT_INFO_QA | "lãi suất thẻ VIB Super Card là bao nhiêu?" | PRODUCT_CONSULT |
| PRODUCT_CONSULT | "tôi muốn mở thẻ nhưng không biết chọn cái nào" | PRODUCT_INFO_QA |
| CUSTOMER_FEEDBACK | "bot hỏi lại thứ tôi đã nói rồi" | PERSONAL_UNRELATED |

**Target:** Macro F1 ≥ 0.85 cho tất cả 5 intent.

**Red flag:** F1 < 0.70 cho bất kỳ intent nào → xem lại prompt của `intent_classifier.py`.

---

### Step 1b — Cache

**Mục tiêu**: Đảm bảo cache không trả sai (false positive) và nhận đúng câu tương tự (true positive).

**Metrics:**
- **True Positive Rate** (paraphrase pairs): Câu tương nghĩa → phải cache hit
- **False Positive Rate** (near-miss pairs): Câu khác nghĩa → không được cache hit
- **Threshold sensitivity**: So sánh kết quả ở threshold 0.75 / 0.80 / 0.85

**2 loại test pairs:**

```
Paraphrase (should_hit=true):
  "lãi suất thẻ VIB Super Card?"
  "thẻ VIB Super Card tính lãi bao nhiêu?"     ← nên hit (≥0.80 similarity)

Near-miss (should_hit=false):
  "lãi suất thẻ VIB Super Card?"
  "hạn mức thẻ VIB Super Card là bao nhiêu?"  ← không được hit (khác nghĩa)
```

**Target:** FPR ≤ 0.05 (≤5% câu khác nghĩa bị nhầm thành cache hit).

**Red flag:** FPR > 0.10 → giảm `CACHE_SIMILARITY_THRESHOLD` hoặc xem lại hybrid scoring.

---

### Step 1c — Customer Feedback Sentiment

**Mục tiêu**: Đo độ chính xác phân loại sentiment trong CUSTOMER_FEEDBACK.

**Metrics:**
- Accuracy tổng thể (3 class: POSITIVE / NEGATIVE / NEUTRAL)
- F1 per class
- Response Quality Score (LLM-as-judge: phản hồi có phù hợp với sentiment không?)

**Ví dụ test:**

| Câu | Sentiment | Phản hồi kỳ vọng |
|---|---|---|
| "Bot hỏi lại info tôi đã nói rồi" | NEGATIVE | Xin lỗi + cải thiện |
| "Chatbot trả lời rất dễ hiểu!" | POSITIVE | Cảm ơn + hỏi tiếp |
| "Tôi vừa dùng thử dịch vụ VIB" | NEUTRAL | Neutral acknowledgment |

**Target:** Accuracy ≥ 0.85.

---

### Step 2 — RAG Pipeline (PRODUCT_INFO_QA)

**Mục tiêu**: Đo chất lượng retrieval + generation cho các câu hỏi thông tin sản phẩm.

**Metrics (LLM-as-judge):**

| Metric | Định nghĩa | Scale |
|---|---|---|
| **Answer Relevance** | Answer có trả lời đúng câu hỏi không? | 1–5 |
| **Faithfulness** | Answer có chứa thông tin bịa đặt ngoài ground truth? | 0/1 |
| **Completeness** | Answer có đầy đủ thông tin cần thiết không? | 1–5 |

**Test set**: Bộ câu hỏi có ground truth answer rút từ tài liệu gốc
(`evaluation/data/rag_samples.json`).

**Target:** Faithfulness ≥ 0.80, Answer Relevance avg ≥ 3.5.

**Note về RAGAS**: Script hỗ trợ flag `--use-ragas` để dùng RAGAS framework
(yêu cầu `pip install ragas`). Mặc định dùng LLM-as-judge (không cần deps thêm).

**Lưu ý quan trọng về Faithfulness (v3.3):**
Judge so sánh bot answer vs `ground_truth_answer` trong file JSON (không phải vs retrieved docs). Ground truth thường là câu trả lời tổng quát (VD: "18–30%/năm"), trong khi bot có thể lấy số liệu cụ thể hơn từ tài liệu. Nếu số liệu cụ thể đó không có trong ground truth, judge sẽ đánh là hallucination.

Khi faithfulness thấp, cần phân biệt 2 trường hợp:
- **Hallucination thật**: bot thêm số liệu không có trong cả tài liệu lẫn ground truth → cần tighten RAG prompt
- **False flag**: bot lấy số liệu từ tài liệu nhưng ground truth quá generic → cần cập nhật ground truth trong `rag_samples.json`

---

### Step 3 — Advisory Pipeline: Field Collection

**Mục tiêu**: Đo advisory pipeline có thu thập đủ thông tin trước khi recommend không.

**Metrics:**
- **Field Completion Rate** = fields collected / fields required
- **Turn Efficiency** = số turns thực tế / số fields cần hỏi
  - Ideal: 1 turn/field + 1 turn opening + 1 turn recommendation
  - Red flag: bot hỏi lại field đã được cung cấp
- **Re-ask Rate** = số lần hỏi lại field đã có / tổng số turns

**Simulation**: Script giả lập user — tự động trả lời theo `user_profile` định sẵn trong
`evaluation/data/advisory_scenarios.json`:

```json
{
  "domain": "credit_card",
  "opening_message": "Tôi muốn mở thẻ tín dụng nhưng chưa biết chọn cái nào",
  "user_profile": {
    "thu_nhap_hang_thang": "15 triệu",
    "muc_chi_tieu_chu_yeu": "mua sắm online và ăn uống",
    "uu_tien_quyen_loi": "cashback",
    "co_the_hien_tai": "chưa có"
  }
}
```

**Target:** Field Completion Rate ≥ 0.90, Re-ask Rate ≤ 0.10.

---

### Step 4 — Advisory Recommendation Quality

**Mục tiêu**: Đo chất lượng recommendation cuối cùng.
Không có "ground truth recommendation" cứng → dùng **LLM-as-judge**.

**Judge prompt:**
```
Cho biết:
- Thông tin KH: {collected_info}
- Sản phẩm được recommend: {recommendation}
- Tài liệu sản phẩm liên quan: {retrieved_context}

Đánh giá recommendation:
1. Sản phẩm gợi ý có phù hợp với thông tin KH không? (Correct/Partially/Incorrect)
2. Lý do recommend có dựa trên tài liệu không? (Grounded/Hallucinated)
3. Có bỏ sót sản phẩm phù hợp hơn không? (Yes/No)
```

**Target:** Correct Rate ≥ 0.70, Hallucination Rate ≤ 0.15.

---

## Phần 2 — Hướng dẫn thực hiện Evaluation

### 2.1 Cài đặt môi trường

```bash
# Đảm bảo đang ở trong version_3/
cd version_3

# Cài thêm dependencies cho eval (nếu chưa có)
pip install requests scikit-learn tabulate colorama --break-system-packages

# (Tùy chọn) Cài RAGAS cho Step 2 full metrics
pip install ragas --break-system-packages
```

**Cấu hình LLM judge (v3.3):**

Eval script (`evaluation/utils.py`) gọi Ollama REST API trực tiếp (`POST /api/chat`) thay vì LangChain ChatOllama. Lý do: tránh bug llama-server 500 "Failed to parse input at pos 0: `<think>`" xảy ra khi ChatOllama xử lý response của reasoning models (DeepSeek-R1).

```
# .env — tùy chọn: dùng model không phải reasoning cho judge (nhanh hơn, ít lỗi hơn)
EVAL_JUDGE_MODEL=qwen2.5:7b    # nếu đã pull model này

# Để trống → dùng LLM_MODEL (deepseek-r1:8b) — vẫn chạy được, chậm hơn
EVAL_JUDGE_MODEL=
```

Nếu judge timeout (>600s/sample) với deepseek-r1:8b, khuyến nghị pull và dùng model không có thinking tokens cho eval.

### 2.2 Khởi động hệ thống

```bash
# Terminal 1 — Ollama
ollama serve

# Terminal 2 — API server (bắt buộc)
cd version_3
python -m uvicorn src.api.main:app --reload

# Chờ server log: "VIB Chatbot V3 starting up" rồi mới chạy eval
```

### 2.3 Kiểm tra kết nối trước khi eval

```bash
curl http://localhost:8000/admin/health
# Kết quả mong muốn: {"status": "ok", "vectorstore_ready": true, ...}
```

---

### 2.4 Chạy từng Step

#### Step 1a — Intent Classification

```bash
cd version_3
python evaluation/step1_intent_eval.py

# Tuỳ chọn:
python evaluation/step1_intent_eval.py --samples evaluation/data/intent_samples.json
python evaluation/step1_intent_eval.py --api-url http://localhost:8000
python evaluation/step1_intent_eval.py --output evaluation/results/step1a_intent.json
```

**Output mẫu:**
```
Intent Classification Report
─────────────────────────────────────────────────────
Intent               Precision  Recall  F1     Support
GREETING_FAREWELL    0.92       0.88    0.90    12
PERSONAL_UNRELATED   0.80       0.83    0.82    12
PRODUCT_INFO_QA      0.85       0.88    0.86    12
PRODUCT_CONSULT      0.87       0.83    0.85    12
CUSTOMER_FEEDBACK    0.92       0.96    0.94    12
─────────────────────────────────────────────────────
Macro F1: 0.87   ✅ PASS (target ≥ 0.85)
```

#### Step 1b — Cache

```bash
python evaluation/step1_cache_eval.py

# Tuỳ chọn:
python evaluation/step1_cache_eval.py --pairs evaluation/data/cache_pairs.json
python evaluation/step1_cache_eval.py --seed-first   # seed cache rồi mới test
python evaluation/step1_cache_eval.py --output evaluation/results/step1b_cache.json
```

**Output mẫu:**
```
Cache Evaluation Report
─────────────────────────────────────────
Paraphrase pairs tested  : 15
  True Positive Rate     : 0.87 (13/15 correctly hit)
Near-miss pairs tested   : 15
  False Positive Rate    : 0.07 (1/15 wrongly hit)
─────────────────────────────────────────
FPR: 0.07  ⚠️  WARN (target ≤ 0.05) — consider increasing threshold
```

#### Step 1c — Feedback Sentiment

```bash
python evaluation/step1_feedback_eval.py

# Tuỳ chọn:
python evaluation/step1_feedback_eval.py --samples evaluation/data/feedback_samples.json
python evaluation/step1_feedback_eval.py --with-quality-check   # LLM judge response quality
```

#### Step 2 — RAG Quality

```bash
python evaluation/step2_rag_eval.py

# Tuỳ chọn:
python evaluation/step2_rag_eval.py --samples evaluation/data/rag_samples.json
python evaluation/step2_rag_eval.py --use-ragas     # dùng RAGAS thay LLM-judge
python evaluation/step2_rag_eval.py --output evaluation/results/step2_rag.json
```

#### Step 3 — Advisory Field Collection

```bash
python evaluation/step3_advisory_eval.py

# Tuỳ chọn:
python evaluation/step3_advisory_eval.py --scenarios evaluation/data/advisory_scenarios.json
python evaluation/step3_advisory_eval.py --output evaluation/results/step3_advisory.json
```

**Output mẫu:**
```
Advisory Pipeline Evaluation
─────────────────────────────────────────────────────────
Scenario         Domain        Fields    Completion  Turns  Re-ask
adv_001          credit_card   4/4       1.00 ✅     6      0
adv_002          loan          4/4       1.00 ✅     7      0
adv_003          insurance     3/4       0.75 ⚠️    5      1
adv_004          savings       3/3       1.00 ✅     5      0
─────────────────────────────────────────────────────────
Avg Completion Rate : 0.94  ✅ PASS (target ≥ 0.90)
Avg Turn Efficiency : 1.52 turns/field
Re-ask Rate         : 0.06  ✅ PASS (target ≤ 0.10)
```

#### Step 4 — Recommendation Quality

```bash
python evaluation/step4_recommend_eval.py

# Tuỳ chọn:
python evaluation/step4_recommend_eval.py --sessions evaluation/results/step3_advisory.json
python evaluation/step4_recommend_eval.py --output evaluation/results/step4_recommend.json
```

**Lưu ý**: Step 4 dùng output của Step 3 làm input. Chạy Step 3 trước.

---

### 2.5 Chạy toàn bộ pipeline (khuyến nghị)

```bash
cd version_3
python evaluation/run_all.py

# Với options:
python evaluation/run_all.py --steps 1a 1b 1c 2 3 4    # tất cả
python evaluation/run_all.py --steps 1a 2               # chỉ intent + RAG
python evaluation/run_all.py --output-dir evaluation/results/run_$(date +%Y%m%d)
python evaluation/run_all.py --fail-fast                # dừng nếu step fail target
```

**Output mẫu (run_all):**
```
╔══════════════════════════════════════════════════════════╗
║          VIB Chatbot V3 — Evaluation Report              ║
║          Run: 2025-01-15 14:30:22                        ║
╠══════════════════════════════════════════════════════════╣
║ Step 1a  Intent Classification   Macro F1  0.87  ✅      ║
║ Step 1b  Cache                   FPR       0.07  ⚠️      ║
║ Step 1c  Feedback Sentiment      Accuracy  0.89  ✅      ║
║ Step 2   RAG Faithfulness        Score     0.82  ✅      ║
║ Step 3   Advisory Completion     Rate      0.94  ✅      ║
║ Step 4   Recommendation Quality  Correct   0.75  ✅      ║
╠══════════════════════════════════════════════════════════╣
║ Overall: 5/6 PASS   1 WARN                              ║
╚══════════════════════════════════════════════════════════╝
```

---

### 2.6 Cấu trúc thư mục evaluation/

```
version_3/evaluation/
├── data/                          # Test datasets
│   ├── intent_samples.json        # ~60 samples (12/intent)
│   ├── rag_samples.json           # ~20 Q&A với ground truth
│   ├── cache_pairs.json           # seed + paraphrase + near-miss pairs
│   ├── feedback_samples.json      # ~30 samples (10/class)
│   └── advisory_scenarios.json   # 5 multi-turn advisory scenarios
├── results/                       # Output (gitignored)
│   └── *.json
├── utils.py                       # API client + reporter + LLM judge
├── step1_intent_eval.py
├── step1_cache_eval.py
├── step1_feedback_eval.py
├── step2_rag_eval.py
├── step3_advisory_eval.py
├── step4_recommend_eval.py
└── run_all.py
```

---

### 2.7 Mở rộng Test Set

Test set mặc định là **mẫu nhỏ** (12–20 samples/step) để chạy nhanh.
Để đánh giá chính xác hơn, bổ sung thêm cases vào file `.json` tương ứng.

**Format thêm intent samples:**
```json
{
  "id": "intent_061",
  "message": "câu hỏi thực tế của khách hàng",
  "expected_intent": "PRODUCT_INFO_QA",
  "note": "ghi chú nếu cần"
}
```

**Format thêm RAG samples:**
```json
{
  "id": "rag_021",
  "question": "câu hỏi về sản phẩm",
  "ground_truth_answer": "câu trả lời đúng từ tài liệu",
  "category": "credit_card"
}
```

---

### 2.8 Khi nào cần chạy lại Evaluation?

| Sự kiện | Steps cần chạy lại |
|---|---|
| Thay đổi prompt `intent_classifier.py` | 1a |
| Thay đổi `CACHE_SIMILARITY_THRESHOLD` | 1b |
| Thay đổi prompt `customer_feedback.py` | 1c |
| Cập nhật `documents_bgem3.parquet` | 2 |
| Thay đổi prompt `generate.py` (RAG generate) | 2 |
| Thay đổi `field_definitions.py` | 3 |
| Thay đổi prompt `recommender.py` | 4 |
| Deploy lên môi trường mới | 1a + 1b + 2 |
| Release mới (full regression) | Tất cả (run_all.py) |
