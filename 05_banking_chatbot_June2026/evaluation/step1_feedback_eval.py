"""
Step 1c — Customer Feedback Evaluation

Đánh giá 2 việc:
1. Intent detection accuracy: có classify đúng CUSTOMER_FEEDBACK không?
2. Response quality (--with-quality-check): dùng LLM judge xem phản hồi
   có phù hợp với sentiment không?

Usage:
  python evaluation/step1_feedback_eval.py
  python evaluation/step1_feedback_eval.py --samples evaluation/data/feedback_samples.json
  python evaluation/step1_feedback_eval.py --with-quality-check
  python evaluation/step1_feedback_eval.py --output evaluation/results/step1c_feedback.json
"""

import argparse
import json
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils import (
    call_chat, require_api, print_header, print_separator,
    status_icon, save_results, judge_feedback_response, EvalLogger,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET
)

TARGET_ACCURACY = 0.85


def run(
    samples_path: str,
    api_url: str,
    with_quality: bool,
    output: str | None,
    logger: EvalLogger | None = None,
) -> dict:
    require_api(api_url)

    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    print_header("Step 1c — Customer Feedback Evaluation")
    print(f"Samples: {len(samples)}  |  Quality check: {with_quality}\n")

    if logger:
        logger.step_start("1c", "Feedback Sentiment", n_samples=len(samples))

    user_id = f"EVAL_FB_{uuid.uuid4().hex[:6]}"
    details = []
    intent_correct = 0
    quality_scores = []

    for i, sample in enumerate(samples, 1):
        msg      = sample["message"]
        exp_sent = sample.get("expected_sentiment", "UNKNOWN")
        exp_int  = sample.get("expected_intent", "CUSTOMER_FEEDBACK")
        sid      = str(uuid.uuid4())

        try:
            resp          = call_chat(msg, session_id=sid, user_id=user_id, api_url=api_url)
            pred_intent   = (resp.get("intent") or "UNKNOWN").upper()
            bot_answer    = resp.get("answer", "")
        except Exception as e:
            pred_intent = "ERROR"
            bot_answer  = ""
            print(f"  [{i:3d}] ERROR: {e}")
            if logger:
                logger.sample("1c", i, sample.get("id", f"s{i}"), "ERROR", error=str(e)[:60])

        int_ok   = pred_intent == "CUSTOMER_FEEDBACK"
        int_icon = "✅" if int_ok else "❌"
        if int_ok:
            intent_correct += 1

        print(f"  [{i:3d}] {int_icon}  intent={pred_intent:<22}  sentiment_expected={exp_sent}")
        if logger and pred_intent != "ERROR":
            logger.sample("1c", i, sample.get("id", f"s{i}"),
                          "PASS" if int_ok else "FAIL",
                          predicted_intent=pred_intent, expected_sentiment=exp_sent)

        detail = {
            "id":             sample.get("id"),
            "message":        msg,
            "expected_intent": exp_int,
            "predicted_intent": pred_intent,
            "expected_sentiment": exp_sent,
            "intent_correct": int_ok,
            "bot_answer":     bot_answer[:200],
        }

        if with_quality and int_ok and bot_answer:
            print(f"       Judging response quality...")
            judge = judge_feedback_response(msg, bot_answer, exp_sent)
            detail["quality"] = judge
            quality_scores.append(judge["score"])
            q_icon = "✅" if judge["appropriate"] else "❌"
            print(f"       {q_icon} appropriate={judge['appropriate']}  score={judge['score']}/5  {judge['reasoning'][:60]}")
            if logger:
                logger.info(f"  [1c][quality] id={sample.get('id')} appropriate={judge['appropriate']} score={judge['score']}/5")

        details.append(detail)

    # ── Metrics ────────────────────────────────────────────────────────────────
    n = len(samples)
    accuracy = intent_correct / n if n else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

    # Per-sentiment breakdown
    sentiments = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    per_sent = {}
    for sent in sentiments:
        sent_samples = [d for d in details if d["expected_sentiment"] == sent]
        if sent_samples:
            correct = sum(1 for d in sent_samples if d["intent_correct"])
            per_sent[sent] = {
                "count": len(sent_samples),
                "correct": correct,
                "accuracy": round(correct / len(sent_samples), 3),
            }

    # ── Report ─────────────────────────────────────────────────────────────────
    print_separator()
    print(f"\n{BOLD}Feedback Evaluation Report{RESET}")
    print_separator()
    print(f"  Samples tested            : {n}")
    status = status_icon(accuracy, TARGET_ACCURACY)
    print(f"  Intent accuracy           : {BOLD}{accuracy:.3f}{RESET}  {status}  (target ≥ {TARGET_ACCURACY})")

    if avg_quality is not None:
        q_status = status_icon(avg_quality, 3.5)
        print(f"  Avg response quality      : {BOLD}{avg_quality:.2f}{RESET}/5  {q_status}")

    print(f"\n  Per-sentiment intent accuracy:")
    for sent, m in per_sent.items():
        color = GREEN if m["accuracy"] >= TARGET_ACCURACY else RED
        print(f"    {sent:<12}: {color}{m['accuracy']:.3f}{RESET} ({m['correct']}/{m['count']})")

    wrong = [d for d in details if not d["intent_correct"]]
    if wrong:
        print(f"\n  {RED}Misclassified ({len(wrong)} samples):{RESET}")
        for d in wrong[:5]:
            print(f"    → '{d['message'][:55]}' → got {d['predicted_intent']}")

    results = {
        "step": "1c_feedback",
        "n_total": n,
        "n_correct": intent_correct,
        "intent_accuracy": round(accuracy, 3),
        "avg_quality_score": round(avg_quality, 3) if avg_quality else None,
        "pass": accuracy >= TARGET_ACCURACY,
        "per_sentiment": per_sent,
        "details": details,
    }
    if logger:
        logger.step_end("1c", metrics={
            "n_total": n, "n_correct": intent_correct,
            "intent_accuracy": round(accuracy, 3),
            "avg_quality_score": round(avg_quality, 3) if avg_quality else None,
        }, elapsed=0, passed=results["pass"])
    save_results(results, output)
    return results


def main():
    import time as _time
    parser = argparse.ArgumentParser(description="Step 1c: Feedback Evaluation")
    parser.add_argument("--samples",       default="evaluation/data/feedback_samples.json")
    parser.add_argument("--api-url",       default="http://localhost:8000")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--output",        default=None)
    parser.add_argument("--output-dir",    default="evaluation/results")
    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = args.output or str(out_dir / f"step1c_{ts}.json")
    logger = EvalLogger(str(out_dir / f"eval_step1c_{ts}.log"))

    t0 = _time.time()
    result = run(args.samples, args.api_url, args.quality_check, output, logger=logger)
    elapsed = _time.time() - t0

    from evaluation.utils import save_step_report
    rpt = save_step_report("1c", result, elapsed, out_dir, ts)
    print(f"\nReport : {rpt}")
    print(f"Log    : {out_dir / ('eval_step1c_' + ts + '.log')}")


if __name__ == "__main__":
    main()
