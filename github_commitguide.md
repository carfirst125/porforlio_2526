# GitHub Commit Guide — Thêm `05_banking_chatbot_June2026`

> **Ngày:** June 2026  
> **Mục tiêu:** Push project `05_banking_chatbot_June2026` lên GitHub, giữ nguyên các project cũ, loại bỏ file lớn.

---

## ⚠️ TÌNH TRẠNG HIỆN TẠI (đọc kỹ trước khi làm)

Chạy lệnh này để kiểm tra:

```bash
cd C:\Users\admin\Documents\05_My_Projects\portfolio_github\porforlio_2526
git status --short | head -20
```

Bạn sẽ thấy **~1113 dòng `D`** — đây là toàn bộ files của các project cũ (01→04) và project mới đang bị **staged deletion** (bị `git rm` nhầm, chưa commit). Đây là vấn đề nghiêm trọng nhất cần xử lý trước.

---

## BƯỚC 0 — Đóng tất cả ứng dụng đang mở repository

Git đang bị lock (`index.lock`). Trước khi chạy bất kỳ lệnh nào:

- Đóng **VS Code** (hoặc terminal VS Code đang mở folder này)
- Đóng **GitKraken / SourceTree / GitHub Desktop** nếu đang mở
- Đóng mọi terminal khác đang `cd` vào thư mục này

Sau đó kiểm tra:

```bash
# Nếu vẫn bị lỗi "index.lock", xóa thủ công:
del C:\Users\admin\Documents\05_My_Projects\portfolio_github\porforlio_2526\.git\index.lock
```

---

## BƯỚC 1 — Unstage toàn bộ staged deletions (khôi phục về HEAD)

Lệnh này **bỏ staged deletions** mà KHÔNG xóa file trên máy, KHÔNG thay đổi commit nào:

```bash
cd C:\Users\admin\Documents\05_My_Projects\portfolio_github\porforlio_2526

git reset HEAD -- .
```

> **Giải thích:** `git reset HEAD -- .` unstage tất cả staged changes trong thư mục hiện tại, đưa index về đúng trạng thái của commit cuối. Files trên disk không bị ảnh hưởng.

Kiểm tra kết quả:

```bash
git status --short | grep "^D" | wc -l
# Kết quả mong đợi: 0 (không còn staged deletions)

git status --short | head -10
# Chỉ còn thấy: M .gitignore, ?? 05_banking_chatbot_June2026/
```

---

## BƯỚC 2 — Kiểm tra .gitignore hoạt động đúng

```bash
# Các file này PHẢI bị ignore (không add vào git):
git check-ignore -v 05_banking_chatbot_June2026/data/vectorstore/chroma.sqlite3
git check-ignore -v 05_banking_chatbot_June2026/.env
git check-ignore -v 05_banking_chatbot_June2026/data/conversations/UID0001.json
git check-ignore -v 05_banking_chatbot_June2026/evaluation/results/run_20260610_222557/summary_20260610_222559.json

# Các file này KHÔNG được bị ignore (cần commit):
git check-ignore -v 05_banking_chatbot_June2026/src/examples/intent_examples.json
git check-ignore -v 05_banking_chatbot_June2026/evaluation/data/advisory_scenarios.json
# → Nếu 2 lệnh trên không output gì = tốt (file sẽ được commit)
```

**Kết quả mong đợi của bước check-ignore:**
- 4 lệnh đầu: in ra rule ignore (tức là file SẼ bị bỏ qua ✓)
- 2 lệnh cuối: không in gì (tức là file SẼ được commit ✓)

---

## BƯỚC 3 — Xem danh sách file sẽ được add

```bash
git add --dry-run 05_banking_chatbot_June2026/
```

Bạn sẽ thấy các file nguồn Python, markdown, requirements.txt, JSON config... **Không** thấy vectorstore, .env, logs.

Nếu thấy file nào không mong muốn, kiểm tra lại `.gitignore` trước khi tiếp tục.

---

## BƯỚC 4 — Stage files

```bash
# Stage .gitignore đã sửa
git add .gitignore

# Stage toàn bộ project mới
git add 05_banking_chatbot_June2026/
```

Kiểm tra lại trước khi commit:

```bash
git status
```

Output mong đợi:
```
Changes to be committed:
  modified:   .gitignore
  new file:   05_banking_chatbot_June2026/.gitignore
  new file:   05_banking_chatbot_June2026/.env.example
  new file:   05_banking_chatbot_June2026/requirements.txt
  new file:   05_banking_chatbot_June2026/src/...
  ... (các file Python, markdown, json config)

nothing to commit (other files unchanged)
```

> ⚠️ Nếu vẫn thấy `D  01_agentic_chatbot_langchain/...` → quay lại Bước 1, chưa unstage thành công.

---

## BƯỚC 5 — Commit

```bash
git commit -m "feat: add 05_banking_chatbot_June2026 - LangGraph RAG banking chatbot"
```

Hoặc message chi tiết hơn:

```bash
git commit -m "feat: add 05_banking_chatbot_June2026

- LangGraph multi-node chatbot with intent classification
- ChromaDB RAG retrieval pipeline
- Advisory profile system with conversation memory
- FastAPI backend + Streamlit frontend
- Evaluation suite (intent, RAG, advisory, cache)
- Update .gitignore: loại bỏ vectorstore, .env, logs, eval results"
```

---

## BƯỚC 6 — Push lên GitHub

```bash
git push origin main
```

Nếu bị lỗi `rejected` (có thể do remote có commit mới hơn):

```bash
git pull origin main --rebase
git push origin main
```

---

## BƯỚC 7 — Verify trên GitHub

Truy cập: https://github.com/carfirst125/porforlio_2526

Kiểm tra:
- [ ] Folder `05_banking_chatbot_June2026/` xuất hiện
- [ ] Folder `data/vectorstore/` **KHÔNG** xuất hiện trong repo
- [ ] File `.env` **KHÔNG** xuất hiện
- [ ] Các project cũ (01→04) vẫn còn nguyên

---

## TÓM TẮT CÁC LỆNH (copy & chạy theo thứ tự)

```bash
cd C:\Users\admin\Documents\05_My_Projects\portfolio_github\porforlio_2526

# Bước 1: Unstage deletions
git reset HEAD -- .

# Bước 4: Stage files mới
git add .gitignore
git add 05_banking_chatbot_June2026/

# Bước 5: Kiểm tra
git status

# Bước 6: Commit
git commit -m "feat: add 05_banking_chatbot_June2026 - LangGraph RAG banking chatbot"

# Bước 7: Push
git push origin main
```

---

## FILES BỊ IGNORE (KHÔNG push lên GitHub)

| Pattern | File/folder bị bỏ qua | Lý do |
|---|---|---|
| `data/vectorstore/` | `data/vectorstore/chroma.sqlite3` (153MB), `*.bin` (21MB mỗi file) | Quá lớn, tự generate được |
| `data/conversations/` | `data/conversations/UID*.json`, `EVAL_*.json` | Runtime data |
| `evaluation/results/` | `evaluation/results/run_*/` | Generated output |
| `.env` | `.env` | Chứa API keys |
| `logs/` | `logs/*.log` | Runtime logs |
| `__pycache__/` | `**/__pycache__/` | Compiled Python |

## FILES ĐƯỢC COMMIT lên GitHub

| Loại | Ví dụ |
|---|---|
| Source code Python | `src/**/*.py`, `config/*.py`, `evaluation/*.py` |
| Config mẫu | `.env.example` |
| Dependencies | `requirements.txt` |
| Docs | `README.md`, `technote.md`, `userguide.md`, `*.md` |
| Eval data (static) | `evaluation/data/*.json`, `src/examples/*.json` |
| Scripts | `scripts/*.py` |
| Batch files | `start_api.bat`, `start_streamlit.bat` |
