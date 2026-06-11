"""
Shared utilities for evaluation scripts.
- API client (call_chat, new_session)
- LLM-as-judge helper
- Console reporter with color + tables
"""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

# ── Allow imports from src/ ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL     = "http://localhost:8000"
REQUEST_TIMEOUT  = 600   # seconds — DeepSeek-R1 on CPU can take 3-5 min/call; increase further if needed
EVAL_USER_PREFIX = "EVAL_"   # keeps eval data separate from real users

# ANSI colors
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── API Client ─────────────────────────────────────────────────────────────────

def new_session(api_url: str = API_BASE_URL) -> str:
    """Create a new chat session and return session_id."""
    resp = requests.post(f"{api_url}/chat/new", timeout=10)
    resp.raise_for_status()
    return resp.json()["session_id"]


def call_chat(
    message: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    api_url: str = API_BASE_URL,
    retries: int = 3,
    retry_wait: float = 5.0,
) -> dict:
    """
    POST /chat/ — returns ChatResponse as dict.
    Retries up to `retries` times on timeout or connection error.

    Keys: session_id, answer, intent, advisor_domain, sources,
          turn_count, collected_info, from_cache, cache_similarity
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    if user_id is None:
        user_id = f"{EVAL_USER_PREFIX}{uuid.uuid4().hex[:8]}"

    payload = {
        "message": message,
        "session_id": session_id,
        "user_id": user_id,
    }

    last_error: Exception = RuntimeError("No attempts made")
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{api_url}/chat/",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as e:
            last_error = e
            print(f"    ⏱️  Timeout on attempt {attempt}/{retries} — waiting {retry_wait}s...")
            time.sleep(retry_wait)
        except requests.exceptions.ConnectionError as e:
            last_error = e
            print(f"    🔌 Connection error on attempt {attempt}/{retries} — waiting {retry_wait}s...")
            time.sleep(retry_wait)
        except Exception as e:
            # Non-retryable errors (4xx, 5xx, JSON parse error) — raise immediately
            raise

    raise last_error


def get_session_state(session_id: str, api_url: str = API_BASE_URL) -> dict:
    """GET /chat/session/{session_id} — returns session state including missing_fields."""
    resp = requests.get(f"{api_url}/chat/session/{session_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def check_api_health(api_url: str = API_BASE_URL) -> bool:
    """Check if API is up and vectorstore is ready."""
    try:
        resp = requests.get(f"{api_url}/admin/health", timeout=10)
        data = resp.json()
        return resp.status_code == 200 and data.get("vectorstore_ready", False)
    except Exception:
        return False


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

def llm_judge(prompt: str, max_retries: int = 2) -> str:
    """
    Call the local Ollama LLM as an evaluation judge via REST API directly.
    Bypasses LangChain/ChatOllama to reliably pass think=false, which prevents
    the llama-server 500 "Failed to parse input at pos 0: <think>" error that
    occurs with reasoning models (DeepSeek-R1, QwQ, etc.).
    Returns the raw response text with <think> blocks stripped.
    """
    import re as _re
    from config.settings import settings

    # Normalize base_url: strip trailing slash
    base = settings.ollama_base_url.rstrip("/")
    # Use a dedicated judge model if configured (should be non-reasoning to avoid
    # <think> token issues). Falls back to llm_model if not set.
    judge_model = settings.eval_judge_model or settings.llm_model

    payload = {
        "model": judge_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia đánh giá. "
                    "Trả lời NGAY bằng JSON thuần túy, không giải thích thêm, "
                    "không dùng <think> hay bất kỳ tag nào khác."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_gpu": settings.ollama_num_gpu,
        },
    }

    last_err: Exception = RuntimeError("No attempts")
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{base}/api/chat",
                json=payload,
                timeout=600,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            raw = data.get("message", {}).get("content", "") or ""
            # Strip <think>...</think> blocks (DeepSeek-R1 reasoning tokens)
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            return raw
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(3.0 * attempt)
                continue
            raise
    raise last_err


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from any text before passing to a judge prompt."""
    import re as _re
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()


def judge_rag_answer(question: str, answer: str, ground_truth: str) -> dict:
    """
    LLM judge for RAG quality.
    Returns dict with keys: relevance (1-5), faithfulness (0/1),
    completeness (1-5), reasoning.
    """
    # Strip <think> tags from bot answer before embedding in judge prompt
    # (prevents llama-server 500 when the bot answer contains reasoning tokens)
    answer = _strip_think(answer)

    prompt = f"""Bạn là chuyên gia đánh giá chất lượng chatbot ngân hàng.

Câu hỏi  : {question}
Câu trả lời của bot: {answer}
Câu trả lời chuẩn: {ground_truth}

Đánh giá câu trả lời của bot theo 3 tiêu chí sau và trả về JSON:

1. relevance: Câu trả lời có đúng chủ đề câu hỏi không? (1=hoàn toàn lạc đề, 5=hoàn toàn đúng)

2. faithfulness: Bot có BỊA thêm thông tin sai lệch so với câu trả lời chuẩn không?
   - 0 = KHÔNG có hallucination (bot không bịa thông tin)
   - 1 = CÓ hallucination (bot bịa ra số liệu, sự kiện, điều kiện không có trong câu trả lời chuẩn)
   QUAN TRỌNG:
   - Nếu bot nói "không tìm thấy thông tin, vui lòng liên hệ hotline" và câu trả lời chuẩn cũng thừa nhận thông tin không có sẵn → faithfulness=0 (bot đúng, không bịa)
   - Nếu bot nói "không tìm thấy" nhưng câu trả lời chuẩn có nội dung cụ thể → đây là thiếu sót (completeness thấp), KHÔNG phải hallucination → faithfulness=0
   - Chỉ đánh faithfulness=1 khi bot ĐƯA RA thông tin cụ thể (số liệu, tên sản phẩm, điều kiện...) mà câu trả lời chuẩn KHÔNG đề cập hoặc mâu thuẫn

3. completeness: Câu trả lời có đầy đủ như câu trả lời chuẩn không? (1=thiếu nhiều, 5=đầy đủ)

Trả về JSON (không thêm text ngoài JSON):
{{"relevance": <int>, "faithfulness": <int>, "completeness": <int>, "reasoning": "<giải thích ngắn>"}}"""

    raw = llm_judge(prompt)
    from src.llm import parse_json as _parse
    result = _parse(raw)
    # Defaults if parsing fails
    return {
        "relevance":    result.get("relevance", 3),
        "faithfulness": result.get("faithfulness", 1),
        "completeness": result.get("completeness", 3),
        "reasoning":    result.get("reasoning", ""),
    }


def judge_recommendation(
    collected_info: dict,
    recommendation: str,
    domain: str,
) -> dict:
    """
    LLM judge for advisory recommendation quality.
    Returns dict with keys: verdict (Correct/Partially/Incorrect),
    grounded (True/False), reasoning.
    """
    recommendation = _strip_think(recommendation)
    info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
    prompt = f"""Bạn là chuyên gia tư vấn sản phẩm ngân hàng VIB.

Thông tin khách hàng đã thu thập được:
{info_str}

Recommendation của bot về {domain}:
{recommendation}

Đánh giá recommendation theo 2 tiêu chí và trả về JSON:

1. verdict: Recommendation có phù hợp với thông tin KH không?
   - "Correct": Phù hợp hoàn toàn
   - "Partially": Phù hợp nhưng thiếu một số điểm
   - "Incorrect": Không phù hợp hoặc sai hoàn toàn

2. grounded: Recommendation có dựa trên thông tin thực tế (sản phẩm VIB tồn tại,
   điều kiện hợp lý) không? (true/false)

3. reasoning: Giải thích ngắn gọn lý do đánh giá

Trả về JSON:
{{"verdict": "<Correct|Partially|Incorrect>", "grounded": <true|false>, "reasoning": "<giải thích>"}}"""

    raw = llm_judge(prompt)
    from src.llm import parse_json as _parse
    result = _parse(raw)
    return {
        "verdict":   result.get("verdict", "Partially"),
        "grounded":  result.get("grounded", True),
        "reasoning": result.get("reasoning", ""),
    }


def judge_feedback_response(
    user_message: str,
    bot_response: str,
    expected_sentiment: str,
) -> dict:
    """Judge if bot response is appropriate for the feedback sentiment."""
    bot_response = _strip_think(bot_response)
    prompt = f"""Đánh giá phản hồi của chatbot đối với feedback của khách hàng.

Tin nhắn KH: {user_message}
Sentiment thực tế: {expected_sentiment}
Phản hồi của bot: {bot_response}

Đánh giá: Phản hồi của bot có phù hợp với sentiment {expected_sentiment} không?
- Nếu NEGATIVE: bot phải xin lỗi và thể hiện sự cầu tiến
- Nếu POSITIVE: bot phải cảm ơn và khuyến khích
- Nếu NEUTRAL: bot phải ghi nhận và hỏi tiếp

Trả về JSON:
{{"appropriate": <true|false>, "score": <1-5>, "reasoning": "<giải thích>"}}"""

    raw = llm_judge(prompt)
    from src.llm import parse_json as _parse
    result = _parse(raw)
    return {
        "appropriate": result.get("appropriate", True),
        "score":       result.get("score", 3),
        "reasoning":   result.get("reasoning", ""),
    }


# ── Reporter ──────────────────────────────────────────────────────────────────

def status_icon(value: float, target: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= target:
            return f"{GREEN}✅ PASS{RESET}"
        elif value >= target * 0.9:
            return f"{YELLOW}⚠️  WARN{RESET}"
        else:
            return f"{RED}❌ FAIL{RESET}"
    else:  # lower is better (e.g. FPR)
        if value <= target:
            return f"{GREEN}✅ PASS{RESET}"
        elif value <= target * 1.5:
            return f"{YELLOW}⚠️  WARN{RESET}"
        else:
            return f"{RED}❌ FAIL{RESET}"


def print_header(title: str) -> None:
    width = 60
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}{title.center(width)}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}\n")


def print_separator() -> None:
    print("─" * 60)


def save_results(results: dict, output_path: Optional[str]) -> None:
    if not output_path:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{CYAN}Results saved: {out.resolve()}{RESET}")


def require_api(api_url: str = API_BASE_URL) -> None:
    """Exit if API is not reachable or vectorstore not ready."""
    print(f"Checking API at {api_url}...")
    if not check_api_health(api_url):
        print(f"{RED}❌ API not reachable or vectorstore not ready.{RESET}")
        print("   Make sure both Ollama and the API server are running:")
        print("   1. ollama serve")
        print("   2. python -m uvicorn src.api.main:app --reload")
        sys.exit(1)
    print(f"{GREEN}✅ API is up{RESET}\n")


# ── EvalLogger ─────────────────────────────────────────────────────────────────

import logging as _logging


class EvalLogger:
    """
    File-based logger for evaluation runs.

    Usage:
        logger = EvalLogger("evaluation/results/run_20260609/eval.log")
        logger.step_start("1a", "Intent Classification", n_samples=50)
        logger.sample("1a", 1, "intent_001", "PASS", expected="PRODUCT_INFO_QA", predicted="PRODUCT_INFO_QA")
        logger.step_end("1a", metrics={"macro_f1": 0.92}, elapsed=12.3, passed=True)
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a unique logger name to avoid handler accumulation across instances
        self._log = _logging.getLogger(f"eval_{id(self)}")
        self._log.setLevel(_logging.DEBUG)
        self._log.propagate = False

        fh = _logging.FileHandler(str(self.log_path), encoding="utf-8", mode="a")
        fh.setFormatter(_logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        self._log.addHandler(fh)

        self._log.info("=" * 70)
        self._log.info(f"EvalLogger initialized — log: {self.log_path}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)

    def step_start(self, step: str, name: str, n_samples: int = 0) -> None:
        self._log.info("─" * 70)
        self._log.info(f"STEP {step} START  {name}" + (f"  |  n_samples={n_samples}" if n_samples else ""))
        self._log.info("─" * 70)

    def step_end(
        self,
        step: str,
        metrics: dict,
        elapsed: float,
        passed: bool,
    ) -> None:
        status = "PASS" if passed else "FAIL"
        self._log.info(f"STEP {step} END  |  elapsed={elapsed:.1f}s  |  status={status}")
        for k, v in metrics.items():
            if isinstance(v, float):
                self._log.info(f"    {k} = {v:.4f}")
            else:
                self._log.info(f"    {k} = {v}")
        self._log.info("─" * 70)

    def sample(
        self,
        step: str,
        idx: int,
        sample_id,
        status: str,
        **details,
    ) -> None:
        """Log a single sample result. status should be 'PASS', 'FAIL', 'WARN', or 'ERROR'."""
        parts = [f"  {k}={v}" for k, v in details.items()]
        self._log.info(
            f"[{step}][{idx:03d}] id={sample_id!s:<20} {status:<5}" + ("" if not parts else " |" + " |".join(parts))
        )


# ── Step report ────────────────────────────────────────────────────────────────

_STEP_TARGETS = {
    "1a": {"name": "Intent Classification",  "metric": "macro_f1",           "target": 0.85, "unit": "",    "higher": True},
    "1b": {"name": "Cache Evaluation",        "metric": "fpr",                "target": 0.05, "unit": "",    "higher": False},
    "1c": {"name": "Feedback Sentiment",      "metric": "intent_accuracy",    "target": 0.85, "unit": "",    "higher": True},
    "2":  {"name": "RAG Quality",             "metric": "faithfulness_rate",  "target": 0.80, "unit": "",    "higher": True},
    "3":  {"name": "Advisory Field Collection","metric": "avg_completion_rate","target": 0.90, "unit": "",    "higher": True},
    "4":  {"name": "Recommendation Quality",  "metric": "correct_rate",       "target": 0.70, "unit": "",    "higher": True},
}

# Fields to display in the metrics section per step
_STEP_METRIC_KEYS = {
    "1a": ["n_total", "n_valid", "n_errors", "accuracy", "macro_f1"],
    "1b": ["n_paraphrase", "n_near_miss", "tpr", "fpr"],
    "1c": ["n_total", "n_correct", "intent_accuracy"],
    "2":  ["n_samples", "avg_relevance", "faithfulness_rate", "avg_completeness"],
    "3":  ["n_scenarios", "avg_completion_rate", "avg_turn_efficiency", "avg_reask_rate"],
    "4":  ["n_evaluated", "correct_rate", "hallucination_rate", "grounded_rate"],
}

# Sample-level detail keys per step (columns for the sample table)
_STEP_DETAIL_KEYS = {
    "1a": ("details",        ["id", "message", "expected", "predicted", "correct", "is_error"]),
    "1b": None,   # cache step has two sub-tables; handled separately
    "1c": ("details",        ["id", "message", "expected_intent", "predicted_intent", "intent_correct"]),
    "2":  ("details",        ["id", "question", "intent", "scores"]),
    "3":  ("scenarios",      ["id", "domain", "fields_required", "fields_collected", "completion_rate", "turns_taken", "reask_rate"]),
    "4":  ("details",        ["id", "domain", "verdict", "grounded"]),
}




def save_step_report(step, result, elapsed, out_dir, ts):
    """Generate a Markdown summary report for one evaluation step.
    Returns the path of the saved .md file.
    """
    _targets = {
        "1a": {"name": "Intent Classification",    "metric": "macro_f1",           "target": 0.85, "higher": True},
        "1b": {"name": "Cache Evaluation",          "metric": "fpr",                "target": 0.05, "higher": False},
        "1c": {"name": "Feedback Sentiment",        "metric": "intent_accuracy",    "target": 0.85, "higher": True},
        "2":  {"name": "RAG Quality",               "metric": "faithfulness_rate",  "target": 0.80, "higher": True},
        "3":  {"name": "Advisory Field Collection", "metric": "avg_completion_rate","target": 0.90, "higher": True},
        "4":  {"name": "Recommendation Quality",    "metric": "correct_rate",       "target": 0.70, "higher": True},
    }
    _metric_keys = {
        "1a": ["n_total", "n_valid", "n_errors", "accuracy", "macro_f1"],
        "1b": ["n_paraphrase", "n_near_miss", "tpr", "fpr"],
        "1c": ["n_total", "n_correct", "intent_accuracy"],
        "2":  ["n_samples", "avg_relevance", "faithfulness_rate", "avg_completeness"],
        "3":  ["n_scenarios", "avg_completion_rate", "avg_turn_efficiency", "avg_reask_rate"],
        "4":  ["n_evaluated", "correct_rate", "hallucination_rate", "grounded_rate"],
    }
    _detail_keys = {
        "1a": ("details",   ["id", "message", "expected", "predicted", "correct", "is_error"]),
        "1c": ("details",   ["id", "message", "expected_intent", "predicted_intent", "intent_correct"]),
        "2":  ("details",   ["id", "question", "intent", "scores"]),
        "3":  ("scenarios", ["id", "domain", "fields_required", "fields_collected", "completion_rate", "turns_taken", "reask_rate"]),
        "4":  ("details",   ["id", "domain", "verdict", "grounded"]),
    }

    def _cell(v):
        if v is None:
            return "—"
        s = str(v).replace("|", "\\|").replace("\n", " ")
        return (s[:77] + "...") if len(s) > 80 else s

    cfg = _targets.get(step, {"name": step, "metric": "pass", "target": 1.0, "higher": True})
    passed = result.get("pass", False)
    primary_metric = cfg.get("metric")
    target_val = cfg.get("target")
    higher = cfg.get("higher", True)

    pass_str = "✅ PASS" if passed else "❌ FAIL"

    doc = [
        "# Step " + step + " — " + cfg["name"],
        "",
        "| | |",
        "|---|---|",
        "| **Run timestamp** | " + ts + " |",
        "| **Elapsed** | " + f"{elapsed:.1f}s" + " |",
        "| **Status** | " + pass_str + " |",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target | Status |",
        "|--------|-------|--------|--------|",
    ]

    for key in _metric_keys.get(step, []):
        val = result.get(key)
        if val is None:
            continue
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        if key == primary_metric and target_val is not None:
            sym = ">=" if higher else "<="
            tgt = sym + " " + str(target_val)
            ok = (val >= target_val) if higher else (val <= target_val)
            sc = "✅ PASS" if ok else "❌ FAIL"
        else:
            tgt, sc = "—", "—"
        doc.append("| `" + key + "` | " + val_str + " | " + tgt + " | " + sc + " |")

    doc.append("")

    if step == "1a" and "per_intent" in result:
        doc += ["## Per-Intent Metrics", "",
                "| Intent | Precision | Recall | F1 | Support |",
                "|--------|-----------|--------|-----|---------|"]
        for intent, m in result["per_intent"].items():
            doc.append(
                "| `" + intent + "` | " +
                f"{m.get('precision',0):.3f}" + " | " +
                f"{m.get('recall',0):.3f}" + " | " +
                f"{m.get('f1',0):.3f}" + " | " +
                str(m.get('support',0)) + " |"
            )
        doc.append("")

    if step == "1b":
        for sub_key, sub_label in [("paraphrase_details", "Paraphrase Pairs (should hit)"),
                                    ("near_miss_details", "Near-miss Pairs (should NOT hit)")]:
            sub_rows = result.get(sub_key, [])
            if not sub_rows:
                continue
            cols = list(sub_rows[0].keys())
            doc += ["## " + sub_label, "",
                    "| " + " | ".join(cols) + " |",
                    "|" + "|".join("---" for _ in cols) + "|"]
            for r in sub_rows:
                doc.append("| " + " | ".join(_cell(r.get(c)) for c in cols) + " |")
            doc.append("")

    detail_cfg = _detail_keys.get(step)
    if detail_cfg:
        list_key, cols = detail_cfg
        items = result.get(list_key, [])
        if items:
            doc += ["## Sample Results (" + str(len(items)) + " rows)", "",
                    "| " + " | ".join(cols) + " |",
                    "|" + "|".join("---" for _ in cols) + "|"]
            for r in items:
                cells = []
                for c in cols:
                    v = r.get(c)
                    if isinstance(v, dict):
                        v = "; ".join(k + "=" + str(vv) for k, vv in v.items())
                    cells.append(_cell(v))
                doc.append("| " + " | ".join(cells) + " |")
            doc.append("")

    error_samples = result.get("error_samples", [])
    if error_samples:
        doc += ["## Error Samples (" + str(len(error_samples)) + ")", ""]
        for eid in error_samples:
            doc.append("- `" + str(eid) + "`")
        doc.append("")

    report_path = out_dir / ("report_step" + step + "_" + ts + ".md")
    report_path.write_text("\n".join(doc), encoding="utf-8")
    return str(report_path)
