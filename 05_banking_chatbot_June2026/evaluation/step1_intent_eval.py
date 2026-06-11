"""
Step 1a — Intent Classification Evaluation

Gọi API với từng sample trong intent_samples.json, so sánh intent trả về
với expected_intent, tính F1/Precision/Recall theo từng intent.

ERROR handling:
  - Samples bị lỗi (timeout, connection) được tách riêng, KHÔNG tính vào metrics.
  - Report hiển thị 2 tập: "Valid only" (chính) và "All incl. errors" (tham khảo).
  - Pass/Fail dựa trên "valid only" metrics — lỗi infrastructure không nên phạt model.
  - Samples lỗi được list riêng để dễ retry thủ công.

Usage:
  python evaluation/step1_intent_eval.py
  python evaluation/step1_intent_eval.py --samples evaluation/data/intent_samples.json
  python evaluation/step1_intent_eval.py --api-url http://localhost:8000
  python evaluation/step1_intent_eval.py --include-errors-in-metrics   # legacy behavior
  python evaluation/step1_intent_eval.py --output evaluation/results/step1a_intent.json
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils import (
    call_chat, require_api, print_header, print_separator,
    status_icon, save_results, EvalLogger,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET
)

TARGET_MACRO_F1 = 0.85
INTENTS = [
    "GREETING_FAREWELL",
    "PERSONAL_UNRELATED",
    "PRODUCT_INFO_QA",
    "PRODUCT_CONSULT",
    "CUSTOMER_FEEDBACK",
]


def run(
    samples_path: str,
    api_url: str,
    output: str | None,
    include_errors_in_metrics: bool = False,
    logger: EvalLogger | None = None,
) -> dict:
    require_api(api_url)

    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    print_header("Step 1a — Intent Classification")
    print(f"Samples loaded : {len(samples)}")
    print(f"API            : {api_url}")
    print(f"Error policy   : {'counted as wrong' if include_errors_in_metrics else 'excluded from metrics'}\n")

    if logger:
        logger.step_start("1a", "Intent Classification", n_samples=len(samples))

    user_id = f"EVAL_INTENT_{uuid.uuid4().hex[:6]}"

    # ── Run predictions ────────────────────────────────────────────────────────
    details = []       # all samples
    error_items = []   # samples that failed API call

    for i, sample in enumerate(samples, 1):
        msg      = sample["message"]
        expected = sample["expected_intent"]
        ambiguous = sample.get("ambiguous", False)
        sid      = str(uuid.uuid4())
        api_error = None

        try:
            resp      = call_chat(msg, session_id=sid, user_id=user_id, api_url=api_url)
            predicted = (resp.get("intent") or "UNKNOWN").upper()
        except Exception as e:
            predicted = "ERROR"
            api_error = str(e)

        correct = (predicted == expected)
        is_error = (predicted == "ERROR")
        ambig_tag = f" {YELLOW}[ambiguous]{RESET}" if ambiguous else ""

        if is_error:
            print(f"  [{i:3d}] {YELLOW}⚠️  TIMEOUT/ERROR{RESET}  expected={expected}{ambig_tag}")
        else:
            icon = "✅" if correct else "❌"
            print(f"  [{i:3d}] {icon}  expected={expected:<22} got={predicted}{ambig_tag}")

        if logger:
            status = "ERROR" if is_error else ("PASS" if correct else "FAIL")
            logger.sample("1a", i, sample.get("id", f"sample_{i}"), status,
                          expected=expected, predicted=predicted,
                          ambiguous=ambiguous)

        detail = {
            "id":        sample.get("id", f"sample_{i}"),
            "message":   msg,
            "expected":  expected,
            "predicted": predicted,
            "correct":   correct,
            "is_error":  is_error,
            "ambiguous": ambiguous,
            "api_error": api_error,
        }
        details.append(detail)
        if is_error:
            error_items.append(detail)

    # ── Compute metrics on valid samples (exclude ERRORs) ──────────────────────
    valid = [d for d in details if not d["is_error"]]
    all_  = details

    metrics_valid = _compute_metrics(
        [d["expected"] for d in valid],
        [d["predicted"] for d in valid],
    )
    metrics_all = _compute_metrics(
        [d["expected"] for d in all_],
        [d["predicted"] for d in all_],
    )

    primary_metrics = metrics_all if include_errors_in_metrics else metrics_valid

    # ── Print report ───────────────────────────────────────────────────────────
    # Error summary
    if error_items:
        print(f"\n{YELLOW}{'─'*60}{RESET}")
        print(f"{YELLOW}⚠️  {len(error_items)} samples failed (timeout/connection error){RESET}")
        print(f"{YELLOW}   These are EXCLUDED from metrics below.{RESET}")
        print(f"{YELLOW}   Retry tip: run with larger timeout or check server load.{RESET}")
        for e in error_items:
            print(f"   • [{e['id']}] {e['message'][:60]}")
        print(f"{YELLOW}{'─'*60}{RESET}")

    print(f"\n{BOLD}Intent Classification Report  (Valid: {len(valid)}/{len(all_)}){RESET}")
    print_separator()
    print(f"  {'Intent':<24} {'Precision':>9} {'Recall':>9} {'F1':>8} {'Valid/Total':>12}")
    print_separator()

    for intent in INTENTS:
        mv  = metrics_valid["per_intent"].get(intent, {})
        ma  = metrics_all["per_intent"].get(intent, {})
        p   = mv.get("precision", 0.0)
        r   = mv.get("recall", 0.0)
        f1  = mv.get("f1", 0.0)
        sup_valid = mv.get("support", 0)
        sup_all   = ma.get("support", 0)
        color = GREEN if f1 >= 0.85 else (YELLOW if f1 >= 0.70 else RED)
        print(f"  {intent:<24} {p:>9.2f} {r:>9.2f} {color}{f1:>8.2f}{RESET} {sup_valid:>6}/{sup_all:<6}")

    print_separator()
    n_valid   = len(valid)
    n_correct = sum(1 for d in valid if d["correct"])
    acc       = n_correct / n_valid if n_valid else 0.0
    macro_f1  = primary_metrics["macro_f1"]
    status    = status_icon(macro_f1, TARGET_MACRO_F1)

    print(f"\n  Valid samples  : {n_valid}/{len(all_)}  ({len(error_items)} errors excluded)")
    print(f"  Accuracy       : {acc:.3f}  ({n_correct}/{n_valid} correct)")
    print(f"  Macro F1       : {BOLD}{macro_f1:.3f}{RESET}  {status}  (target ≥ {TARGET_MACRO_F1})")

    if error_items:
        macro_f1_with_errors = metrics_all["macro_f1"]
        print(f"  Macro F1 (incl. errors) : {macro_f1_with_errors:.3f}  {YELLOW}[reference only]{RESET}")

    # ── Confusion matrix (valid only) ──────────────────────────────────────────
    if valid:
        print(f"\n{BOLD}Confusion Matrix (valid samples only, row=expected, col=predicted){RESET}")
        _print_confusion_matrix(
            [d["expected"] for d in valid],
            [d["predicted"] for d in valid],
        )

    # ── Misclassification analysis ─────────────────────────────────────────────
    misclassed = [d for d in valid if not d["correct"]]
    if misclassed:
        pairs = _worst_confusion_pairs(
            [d["expected"] for d in valid],
            [d["predicted"] for d in valid],
        )
        print(f"\n{BOLD}Real misclassifications ({len(misclassed)} samples):{RESET}")
        for (true_l, pred_l), count in pairs[:5]:
            print(f"  expected {true_l:<22} → predicted {pred_l:<22} ({count}x)")

        print(f"\n{BOLD}Sample details:{RESET}")
        for d in misclassed:
            ambig_tag = f" {YELLOW}[ambiguous label]{RESET}" if d["ambiguous"] else ""
            print(f"  ❌ [{d['id']}] '{d['message'][:60]}'")
            print(f"       expected={d['expected']}  got={d['predicted']}{ambig_tag}")

    # ── Ambiguous samples ──────────────────────────────────────────────────────
    ambig_samples = [d for d in valid if d.get("ambiguous")]
    if ambig_samples:
        print(f"\n{BOLD}Ambiguous samples in test set ({len(ambig_samples)}):{RESET}")
        print(f"  {YELLOW}These are borderline cases — consider excluding from target metric.{RESET}")
        for d in ambig_samples:
            icon = "✅" if d["correct"] else "❌"
            print(f"  {icon} [{d['id']}] '{d['message'][:60]}'  label={d['expected']}")

    results = {
        "step":          "1a_intent",
        "n_total":       len(all_),
        "n_valid":       n_valid,
        "n_errors":      len(error_items),
        "accuracy":      round(acc, 3),
        "macro_f1":      macro_f1,
        "macro_f1_with_errors": metrics_all["macro_f1"],
        "pass":          macro_f1 >= TARGET_MACRO_F1,
        "per_intent":    primary_metrics["per_intent"],
        "error_samples": [e["id"] for e in error_items],
        "details":       details,
    }
    if logger:
        logger.step_end("1a", metrics={
            "n_total": len(all_), "n_valid": n_valid, "n_errors": len(error_items),
            "accuracy": round(acc, 3), "macro_f1": macro_f1,
        }, elapsed=0, passed=results["pass"])
    save_results(results, output)
    return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: list, y_pred: list) -> dict:
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_intent": {}}

    correct  = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)

    per_intent = {}
    f1s = []
    for intent in INTENTS:
        tp      = sum(t == intent and p == intent for t, p in zip(y_true, y_pred))
        fp      = sum(t != intent and p == intent for t, p in zip(y_true, y_pred))
        fn      = sum(t == intent and p != intent for t, p in zip(y_true, y_pred))
        support = sum(t == intent for t in y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_intent[intent] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "support":   support,
        }
        if support > 0:
            f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "accuracy":   round(accuracy, 3),
        "macro_f1":   round(macro_f1, 3),
        "per_intent": per_intent,
    }


def _print_confusion_matrix(y_true: list, y_pred: list) -> None:
    abbrev = {
        "GREETING_FAREWELL":  "GREET",
        "PERSONAL_UNRELATED": "PERS ",
        "PRODUCT_INFO_QA":    "INFO ",
        "PRODUCT_CONSULT":    "CONS ",
        "CUSTOMER_FEEDBACK":  "FEEDB",
    }
    matrix = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    header = " " * 8 + "  ".join(abbrev.get(l, l[:5]) for l in INTENTS)
    print(f"  {header}")
    for true_l in INTENTS:
        row   = abbrev.get(true_l, true_l[:5])
        cells = []
        for pred_l in INTENTS:
            v = matrix[true_l][pred_l]
            if true_l == pred_l:
                cells.append(f"{GREEN}{v:5d}{RESET}")
            elif v > 0:
                cells.append(f"{RED}{v:5d}{RESET}")
            else:
                cells.append(f"{'0':>5}")
        print(f"  {row}  " + "  ".join(cells))


def _worst_confusion_pairs(y_true: list, y_pred: list) -> list:
    counts = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t != p:
            counts[(t, p)] += 1
    return sorted(counts.items(), key=lambda x: -x[1])


def main():
    import time
    parser = argparse.ArgumentParser(description="Step 1a: Intent Classification Evaluation")
    parser.add_argument("--samples",                    default="evaluation/data/intent_samples.json")
    parser.add_argument("--api-url",                    default="http://localhost:8000")
    parser.add_argument("--output",                     default=None)
    parser.add_argument("--output-dir",                 default="evaluation/results")
    parser.add_argument("--include-errors-in-metrics",  action="store_true")
    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = args.output or str(out_dir / f"step1a_{ts}.json")
    logger = EvalLogger(str(out_dir / f"eval_step1a_{ts}.log"))

    t0 = time.time()
    result = run(args.samples, args.api_url, output, args.include_errors_in_metrics, logger=logger)
    elapsed = time.time() - t0

    from evaluation.utils import save_step_report
    rpt = save_step_report("1a", result, elapsed, out_dir, ts)
    print(f"\nReport : {rpt}")
    print(f"Log    : {out_dir / ('eval_step1a_' + ts + '.log')}")


if __name__ == "__main__":
    main()
