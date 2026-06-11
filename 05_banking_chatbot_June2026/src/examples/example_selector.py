"""
Shared utility: dynamic few-shot example selection.

Dùng Jaccard keyword similarity để chọn n examples phù hợp nhất
từ một example library, tránh phải hardcode examples trong prompt.

Không cần embedding — chỉ dùng keyword overlap để nhanh và nhẹ.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> set[str]:
    """Lowercase + split, loại bỏ dấu câu đơn giản."""
    text = text.lower()
    text = re.sub(r"[?!.,;:\"'()\[\]{}]", " ", text)
    return set(t for t in text.split() if t)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def select_cache_examples(
    q_a: str,
    q_b: str,
    examples: list[dict],
    n: int = 4,
    min_true: int = 1,
    min_false: int = 2,
) -> list[dict]:
    """
    Chọn n ví dụ phù hợp nhất cho equivalence gate.
    Đảm bảo ít nhất min_true ví dụ true và min_false ví dụ false.

    Args:
        q_a, q_b: hai câu hỏi cần so sánh
        examples:  toàn bộ example library
        n:         tổng số ví dụ cần chọn
        min_true:  số ví dụ true tối thiểu
        min_false: số ví dụ false tối thiểu
    """
    query_tokens = _tokenize(q_a) | _tokenize(q_b)

    scored: list[tuple[float, dict]] = []
    for ex in examples:
        ex_tokens = _tokenize(ex.get("q_a", "")) | _tokenize(ex.get("q_b", ""))
        score = _jaccard(query_tokens, ex_tokens)
        scored.append((score, ex))

    scored.sort(key=lambda x: -x[0])

    true_pool  = [ex for _, ex in scored if ex.get("equivalent") is True]
    false_pool = [ex for _, ex in scored if ex.get("equivalent") is False]

    selected: list[dict] = []
    # Fill minimum quotas
    for ex in true_pool[:min_true]:
        selected.append(ex)
    for ex in false_pool[:min_false]:
        selected.append(ex)

    # Fill remaining slots from overall top-ranked (skipping already selected)
    selected_ids = {id(ex) for ex in selected}
    for _, ex in scored:
        if len(selected) >= n:
            break
        if id(ex) not in selected_ids:
            selected.append(ex)
            selected_ids.add(id(ex))

    return selected


def select_intent_examples(
    message: str,
    examples: list[dict],
    n: int = 6,
    min_per_class: int = 1,
) -> list[dict]:
    """
    Chọn n ví dụ few-shot phù hợp nhất cho intent classifier.
    Đảm bảo ít nhất min_per_class ví dụ cho mỗi intent class hiện diện.

    Args:
        message:  câu hỏi của khách hàng
        examples: toàn bộ example library
        n:        tổng số ví dụ
        min_per_class: số ví dụ tối thiểu cho mỗi class (best effort)
    """
    query_tokens = _tokenize(message)

    scored: list[tuple[float, dict]] = []
    for ex in examples:
        ex_tokens = _tokenize(ex.get("message", ""))
        score = _jaccard(query_tokens, ex_tokens)
        scored.append((score, ex))

    scored.sort(key=lambda x: -x[0])

    # Group by intent
    from collections import defaultdict
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for _, ex in scored:
        by_intent[ex.get("intent", "")].append(ex)

    selected: list[dict] = []
    selected_ids: set[int] = set()

    # 1. One best-match per intent class (quota)
    for intent, pool in by_intent.items():
        for ex in pool[:min_per_class]:
            if id(ex) not in selected_ids:
                selected.append(ex)
                selected_ids.add(id(ex))

    # 2. Fill remaining with top-ranked
    for _, ex in scored:
        if len(selected) >= n:
            break
        if id(ex) not in selected_ids:
            selected.append(ex)
            selected_ids.add(id(ex))

    return selected[:n]


def load_examples(json_path: str | Path) -> list[dict]:
    """Load example library từ JSON file."""
    path = Path(json_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("examples", [])
    except Exception:
        return []
