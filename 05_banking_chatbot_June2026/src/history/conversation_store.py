"""
ConversationStore — lưu trữ lịch sử hội thoại per UserID + hybrid cache lookup.

Storage:
  - data/conversations/{user_id}.json  → readable history per user
  - ChromaDB collection "conversation_history_v3"  → search index (toàn bộ users)

Cache lookup flow:
  1. Embed câu hỏi mới bằng bge-m3
  2. Semantic search trên ChromaDB history collection → top-K candidates
  3. Với mỗi candidate: tính hybrid score = 0.65 * semantic_sim + 0.35 * keyword_sim (Jaccard)
  4. Nếu best_score >= threshold VÀ LLM gate xác nhận → trả về cached answer
  5. Nếu không → xử lý bình thường qua graph

JSON format per user:
{
  "user_id": "UID001",
  "entries": [
    {
      "id": "uuid",
      "timestamp": "ISO8601",
      "question": "...",
      "answer": "...",
      "intent": "...",
      "advisor_domain": null,
      "session_id": "...",
      "similarity_score": null
    }
  ]
}
"""
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


class ConversationStore:
    """Thread-safe conversation history store with hybrid cache lookup."""

    def __init__(self):
        self._lock = threading.Lock()
        self._collection = None
        self._loaded = False

    # ── ChromaDB collection ─────────────────────────────────────────────────

    def _get_collection(self):
        if self._collection is None:
            from src.data.loader import get_chroma_client
            client = get_chroma_client()
            self._collection = client.get_or_create_collection(
                name=settings.history_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── JSON helpers ────────────────────────────────────────────────────────

    def _json_path(self, user_id: str) -> Path:
        return Path(settings.conversations_dir) / f"{user_id}.json"

    def _load_user_json(self, user_id: str) -> dict:
        path = self._json_path(user_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Cannot read {path}: {e}")
        return {"user_id": user_id, "entries": []}

    def _save_user_json(self, data: dict):
        user_id = data["user_id"]
        path = self._json_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Startup: load all JSON → ChromaDB (bi-directional sync) ────────────

    def load_all_history(self):
        """
        Sync toàn bộ JSON files <-> ChromaDB index (bi-directional).
        Gọi khi startup.
        """
        with self._lock:
            if self._loaded:
                return
            self._sync_history()
            self._loaded = True

    def _sync_history(self):
        """Bi-directional sync: JSON files <-> ChromaDB."""
        col = self._get_collection()
        existing_ids = set(col.get(include=[])["ids"])

        conv_dir = Path(settings.conversations_dir)
        json_files = [
            f for f in conv_dir.glob("*.json")
            if not f.name.endswith("_profiles.json")
        ] if conv_dir.exists() else []

        valid_ids: set = set()
        entries_to_add: list = []

        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                for entry in data.get("entries", []):
                    eid = entry.get("id", "")
                    if not eid:
                        continue
                    if not entry.get("question") or not entry.get("answer"):
                        continue
                    valid_ids.add(eid)
                    if eid not in existing_ids:
                        entries_to_add.append(entry)
            except Exception as e:
                logger.warning(f"Error reading {jf}: {e}")

        stale_ids = existing_ids - valid_ids
        if stale_ids:
            col.delete(ids=list(stale_ids))
            logger.info(f"Removed {len(stale_ids)} stale entries from ChromaDB history.")

        if entries_to_add:
            ids_batch, questions, metadatas, embeddings = [], [], [], []
            for entry in entries_to_add:
                emb = self._embed(entry["question"])
                if emb is None:
                    continue
                ids_batch.append(entry["id"])
                questions.append(entry["question"])
                embeddings.append(emb)
                metadatas.append({
                    "answer": entry["answer"][:2000],
                    "user_id": entry.get("user_id", "unknown"),
                    "intent": entry.get("intent") or "",
                    "advisor_domain": entry.get("advisor_domain") or "",
                    "timestamp": entry.get("timestamp") or "",
                    "entry_id": entry["id"],
                })
            if ids_batch:
                col.add(
                    ids=ids_batch,
                    documents=questions,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                logger.info(f"Added {len(ids_batch)} new history entries to ChromaDB.")
        else:
            logger.info("ChromaDB history is up to date (no new entries to add).")

        logger.info(
            f"History sync complete: {len(valid_ids)} valid entries in JSON, "
            f"{len(stale_ids)} removed, {len(entries_to_add)} added."
        )

    def rebuild_history(self):
        """Xóa toàn bộ ChromaDB history và rebuild lại từ JSON files."""
        with self._lock:
            col = self._get_collection()
            all_ids = col.get(include=[])["ids"]
            if all_ids:
                col.delete(ids=all_ids)
                logger.info(f"Cleared {len(all_ids)} entries from ChromaDB history.")
            self._loaded = False
        self.load_all_history()
        logger.info("ChromaDB history rebuilt from JSON files.")

    # ── Save new Q&A ────────────────────────────────────────────────────────

    def save(
        self,
        user_id: str,
        question: str,
        answer: str,
        intent: Optional[str] = None,
        advisor_domain: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Save Q&A to JSON file và add to ChromaDB index."""
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "intent": intent,
            "advisor_domain": advisor_domain,
            "session_id": session_id,
        }

        with self._lock:
            data = self._load_user_json(user_id)
            data["entries"].append(entry)
            self._save_user_json(data)

            try:
                emb = self._embed(question)
                if emb is not None:
                    col = self._get_collection()
                    col.add(
                        ids=[entry_id],
                        documents=[question],
                        embeddings=[emb],
                        metadatas=[{
                            "answer": answer[:2000],
                            "user_id": user_id,
                            "intent": intent or "",
                            "advisor_domain": advisor_domain or "",
                            "timestamp": entry["timestamp"],
                            "entry_id": entry_id,
                        }],
                    )
            except Exception as e:
                logger.warning(f"Failed to index entry {entry_id} in ChromaDB: {e}")

        logger.info(f"Saved conversation entry for user={user_id}, id={entry_id}")
        return entry_id

    # ── Hybrid cache lookup ──────────────────────────────────────────────────

    # Path tới example library (relative to project root)
    _CACHE_EXAMPLES_PATH = (
        Path(__file__).resolve().parent.parent / "examples" / "cache_equivalence_examples.json"
    )
    _cache_examples: list = []   # lazy-loaded

    # ── Rule-based near-miss detection ──────────────────────────────────────
    # Các nhóm loại thông tin: nếu q_a và q_b thuộc hai nhóm KHÁC NHAU → near-miss
    _ATTR_GROUPS: list = [
        {"lãi suất", "lai suat", "lãi", "interest rate", "tính lãi"},
        {"hạn mức", "han muc", "credit limit", "mức tín dụng"},
        {"phí thường niên", "phi thuong nien", "annual fee"},
        {"phí", "phi", "fee", "phí giao dịch", "phí rút", "phí thanh toán"},
        {"điều kiện", "dieu kien", "yêu cầu", "yeu cau", "đủ điều kiện", "tiêu chuẩn"},
        {"hồ sơ", "ho so", "giấy tờ", "giay to", "tài liệu", "cần gì để"},
        {"quy trình", "quy trinh", "thủ tục", "thu tuc", "các bước", "quy trình"},
        {"quyền lợi", "quyen loi", "benefit", "bảo vệ", "bồi thường"},
        {"thời hạn", "thoi han", "kỳ hạn", "ky han"},
    ]

    # Các nhóm sản phẩm cụ thể: nếu q_a và q_b thuộc hai nhóm KHÁC NHAU → near-miss
    _PRODUCT_GROUPS: list = [
        {"super card"},
        {"cash back"},
        {"online plus"},
        {"premier boundless"},
        {"rewards unlimited"},
        {"family link"},
        {"mua nhà", "mua nha", "vay nhà", "vay nha"},
        {"mua xe", "vay xe"},
        {"tiêu dùng", "tieu dung"},
        {"tín chấp", "tin chap"},
    ]

    # Hành động đối lập: nếu một câu open và câu kia close → near-miss
    _OPEN_ACTIONS: set  = {"mở thẻ", "mo the", "làm thẻ", "lam the", "đăng ký thẻ", "dang ky", "mở tài khoản"}
    _CLOSE_ACTIONS: set = {"đóng thẻ", "dong the", "hủy thẻ", "huy the", "tất toán", "tat toan"}

    _PERIOD_RE = re.compile(r'(\d+)\s*(?:tháng|thang|tuần|tuan|năm|nam)\b')

    @classmethod
    def _rule_near_miss(cls, q_a: str, q_b: str) -> bool:
        """
        Rule-based near-miss detection. Chạy TRƯỚC LLM gate.
        Returns True  → rõ ràng là near-miss, bỏ qua LLM.
        Returns False → không chắc, cho LLM quyết định.
        """
        a = q_a.lower()
        b = q_b.lower()

        # Rule 1: Loại thông tin khác nhau
        a_groups = {i for i, g in enumerate(cls._ATTR_GROUPS) if any(kw in a for kw in g)}
        b_groups = {i for i, g in enumerate(cls._ATTR_GROUPS) if any(kw in b for kw in g)}
        if a_groups and b_groups and not (a_groups & b_groups):
            logger.debug(f"Rule near-miss: attr groups differ {a_groups} vs {b_groups}")
            return True

        # Rule 2: Kỳ hạn / thời gian khác nhau (3 tháng vs 12 tháng)
        a_periods = set(cls._PERIOD_RE.findall(a))
        b_periods = set(cls._PERIOD_RE.findall(b))
        if a_periods and b_periods and a_periods != b_periods:
            logger.debug(f"Rule near-miss: periods differ {a_periods} vs {b_periods}")
            return True

        # Rule 3: Sản phẩm cụ thể khác nhau
        a_prods = {i for i, g in enumerate(cls._PRODUCT_GROUPS) if any(kw in a for kw in g)}
        b_prods = {i for i, g in enumerate(cls._PRODUCT_GROUPS) if any(kw in b for kw in g)}
        if a_prods and b_prods and not (a_prods & b_prods):
            logger.debug(f"Rule near-miss: product groups differ {a_prods} vs {b_prods}")
            return True

        # Rule 4: Hành động đối lập (mở thẻ vs đóng thẻ)
        a_open  = any(kw in a for kw in cls._OPEN_ACTIONS)
        a_close = any(kw in a for kw in cls._CLOSE_ACTIONS)
        b_open  = any(kw in b for kw in cls._OPEN_ACTIONS)
        b_close = any(kw in b for kw in cls._CLOSE_ACTIONS)
        if (a_open and b_close) or (a_close and b_open):
            logger.debug("Rule near-miss: opposite actions")
            return True

        return False  # không chắc → cho LLM quyết định

    @classmethod
    def _get_cache_examples(cls) -> list:
        if not cls._cache_examples:
            try:
                from src.examples.example_selector import load_examples
                cls._cache_examples = load_examples(cls._CACHE_EXAMPLES_PATH)
            except Exception:
                cls._cache_examples = []
        return cls._cache_examples

    @staticmethod
    def _build_equivalence_prompt(q_a: str, q_b: str, examples: list) -> str:
        """
        Prompt với step-by-step attribute extraction + dynamic few-shot examples.
        Force model phân tích rõ ràng LOAI_THONG_TIN và SAN_PHAM trước khi kết luận.
        """
        lines = [
            "Nhiem vu: Phan tich hai cau hoi co hoi ve CUNG MOT thong tin khong.",
            "",
            "Cac buoc phan tich (bat buoc):",
            "Buoc 1 — Loai thong tin can biet:",
            "  Cau A hoi ve: [lai suat / han muc / phi / ho so / dieu kien / thu tuc / quyen loi / ky han / ...]",
            "  Cau B hoi ve: [...]",
            "  -> Neu LOAI THONG TIN khac nhau: ket luan KHONG tuong duong, dung lai.",
            "",
            "Buoc 2 — San pham / doi tuong cu the:",
            "  Cau A ve san pham: [ten the cu the / mua nha / mua xe / ky han so thang / ...]",
            "  Cau B ve san pham: [...]",
            "  -> Neu SAN PHAM / THAM SO khac nhau: ket luan KHONG tuong duong, dung lai.",
            "",
            "Buoc 3 — Chi ket luan TUONG DUONG khi ca LOAI THONG TIN va SAN PHAM deu giong nhau.",
            "",
            "Vi du minh hoa:",
            '  A: "Lai suat the tin dung VIB Super Card la bao nhieu?"',
            '  B: "Han muc the tin dung VIB Super Card la bao nhieu?"',
            "  Buoc 1: A hoi lai suat, B hoi han muc → KHAC loai → KHONG tuong duong",
            "",
            '  A: "Lai suat tiet kiem VIB ky han 12 thang?"',
            '  B: "Lai suat tiet kiem VIB ky han 3 thang?"',
            "  Buoc 1: Deu hoi lai suat → giong. Buoc 2: 12 thang vs 3 thang → KHAC → KHONG tuong duong",
            "",
            '  A: "The VIB Super Card tinh lai suat bao nhieu phan tram?"',
            '  B: "Lai suat the tin dung VIB Super Card la bao nhieu?"',
            "  Buoc 1: Deu hoi lai suat → giong. Buoc 2: Deu ve Super Card → giong. → TUONG DUONG",
            "",
        ]

        if examples:
            lines.append("Vi du them (dynamic few-shot):")
            for ex in examples:
                label = "TUONG DUONG" if ex.get("equivalent") else "KHONG tuong duong"
                note = ex.get("note", "")
                lines.append(f'  A: "{ex["q_a"]}"')
                lines.append(f'  B: "{ex["q_b"]}"')
                lines.append(f'  → {label}  [{note}]')
                lines.append("")

        lines += [
            "Bay gio phan tich:",
            f'  Cau A: "{q_a}"',
            f'  Cau B: "{q_b}"',
            "",
            "Chi tra ve JSON (khong giai thich them):",
            '{"equivalent": true}  hoac  {"equivalent": false}',
            "",
            "JSON:",
        ]
        return "\n".join(lines)

    def search_similar(
        self,
        question: str,
        threshold: float = None,
    ) -> Optional[dict]:
        """
        Tìm câu hỏi tương tự nhất trong lịch sử.

        Khi cache_llm_verify=True (mặc định): pipeline 3 bước
          1. Semantic pre-filter (settings.cache_similarity_threshold)
          2. Hybrid re-rank (0.65 semantic + 0.35 Jaccard)
          3. LLM equivalence gate

        Khi cache_llm_verify=False (legacy): pipeline 2 bước (không có LLM gate)
        """
        pre_thr = threshold if threshold is not None else settings.cache_similarity_threshold
        col = self._get_collection()

        try:
            count = col.count()
        except Exception:
            count = 0

        if count == 0:
            return None

        query_emb = self._embed(question)
        if query_emb is None:
            return None

        k = min(settings.cache_top_k, count)
        try:
            results = col.query(
                query_embeddings=[query_emb],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"History search error: {e}")
            return None

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        if not docs:
            return None

        query_tokens = set(question.lower().split())
        best_score = 0.0
        best_idx = -1

        for i, (doc, dist) in enumerate(zip(docs, dists)):
            semantic_sim = max(0.0, 1.0 - dist)

            cand_tokens = set(doc.lower().split())
            if query_tokens or cand_tokens:
                union = query_tokens | cand_tokens
                intersection = query_tokens & cand_tokens
                keyword_sim = len(intersection) / len(union) if union else 0.0
            else:
                keyword_sim = 0.0

            hybrid = 0.65 * semantic_sim + 0.35 * keyword_sim

            logger.debug(
                f"Cache candidate [{i}]: sem={semantic_sim:.3f}, "
                f"kw={keyword_sim:.3f}, hybrid={hybrid:.3f} | {doc[:60]}"
            )

            if hybrid > best_score:
                best_score = hybrid
                best_idx = i

        if best_score < pre_thr or best_idx < 0:
            logger.debug(f"Cache MISS (pre-filter): best_score={best_score:.3f} < {pre_thr}")
            return None

        meta = metas[best_idx]
        matched_q = docs[best_idx]

        if settings.cache_llm_verify and not self._llm_verify_equivalent(question, matched_q):
            logger.info(
                f"Cache REJECTED by LLM gate: score={best_score:.3f} "
                f"| new=\'{question[:60]}\' vs cached=\'{matched_q[:60]}\'"
            )
            return None

        logger.info(
            f"Cache HIT: score={best_score:.3f} (LLM verified) "
            f"| matched: \'{matched_q[:80]}\'"
        )
        return {
            "answer": meta.get("answer", ""),
            "similarity": round(best_score, 4),
            "matched_question": matched_q,
            "user_id": meta.get("user_id", ""),
            "intent": meta.get("intent"),
            "advisor_domain": meta.get("advisor_domain") or None,
            "timestamp": meta.get("timestamp"),
        }

    def _llm_verify_equivalent(self, q_new: str, q_cached: str) -> bool:
        """
        Kiểm tra hai câu hỏi có yêu cầu cùng thông tin không.

        Pipeline:
          1. Rule-based near-miss check (nhanh, không cần LLM)
             → Nếu phát hiện near-miss rõ ràng: return False ngay
          2. LLM gate với step-by-step attribute extraction prompt
             → Fallback về False nếu lỗi (safe: không serve cache sai)

        num_ctx=2048: đủ chỗ cho <think> block của deepseek-r1:8b.
        """
        # Bước 1: Rule-based pre-check (bắt 80%+ near-miss không cần LLM)
        if self._rule_near_miss(q_new, q_cached):
            logger.info(
                f"Cache REJECTED by rule (near-miss): "
                f"'{q_new[:60]}' vs '{q_cached[:60]}'"
            )
            return False

        # Bước 2: LLM gate cho các trường hợp không chắc chắn
        try:
            from src.llm import get_fast_llm, parse_json
            from src.examples.example_selector import select_cache_examples

            all_examples = self._get_cache_examples()
            selected = select_cache_examples(
                q_a=q_new, q_b=q_cached,
                examples=all_examples,
                n=4, min_true=1, min_false=2,
            )

            prompt = self._build_equivalence_prompt(q_new, q_cached, selected)
            llm = get_fast_llm(temperature=0.0, num_ctx=2048)
            response = llm.invoke(prompt)
            result = parse_json(response.content)
            equivalent = bool(result.get("equivalent", False))
            logger.debug(
                f"LLM equivalence ({llm.model}): {equivalent} "
                f"| '{q_new[:50]}' vs '{q_cached[:50]}'"
            )
            return equivalent
        except Exception as e:
            logger.warning(f"LLM equivalence check failed: {e} — defaulting to MISS")
            return False

    # ── Embedding helper ─────────────────────────────────────────────────────

    @staticmethod
    def _embed(text: str) -> Optional[list]:
        try:
            from src.llm import embed_query
            return embed_query(text)
        except Exception as e:
            logger.warning(f"Embed failed for cache: {e}")
            return None

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_user_history(self, user_id: str) -> list:
        """Trả về danh sách entries của 1 user (từ JSON)."""
        return self._load_user_json(user_id).get("entries", [])

    def get_stats(self) -> dict:
        """Stats: số users, số entries, số entries trong ChromaDB."""
        conv_dir = Path(settings.conversations_dir)
        json_files = list(conv_dir.glob("*.json")) if conv_dir.exists() else []
        total_entries = 0
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                total_entries += len(data.get("entries", []))
            except Exception:
                pass
        try:
            chroma_count = self._get_collection().count()
        except Exception:
            chroma_count = 0
        return {
            "total_users": len(json_files),
            "total_entries": total_entries,
            "chroma_indexed": chroma_count,
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_store: Optional[ConversationStore] = None


def get_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store
