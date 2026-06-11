"""
Step 4 — Advisory Recommendation Quality (LLM-as-judge)

Đánh giá chất lượng recommendation cuối cùng của advisory pipeline.
Input: kết quả từ step3_advisory_eval.py (có collected_info + conversation)
       hoặc chạy lại scenarios để lấy recommendation.

Metrics:
- Correct Rate: % Correct hoặc Partially correct
- Hallucination Rate: % recommendation có thông tin bịa đặt
- Grounded Rate: % dựa trên thông tin thực tế sản phẩm VIB

Usage:
  python evaluation/step4_recommend_eval.py
  python evaluation/step4_recommend_eval.py --step3-results evaluation/results/step3_advisory.json
  python evaluation/step4_recommend_eval.py --scenarios evaluation/data/advisory_scenarios.json
  python evaluation/step4_recommend_eval.py --output evaluation/results/step4_recommend.json
"""

import argparse
import json
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils import (
    call_chat, get_session_state, require_api, print_header, print_separator,
    status_icon, save_results, judge_recommendation, EvalLogger,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET
)

TARGET_CORRECT_RATE       = 0.70
TARGET_HALLUCINATION_RATE = 0.15


def run(
    step3_results_path: str | None,
    scenarios_path: str,
    api_url: str,
    output: str | None,
    logger: EvalLogger | None = None,
) -> dict:
    require_api(api_url)
    print_header("Step 4 — Recommendation Quality (LLM Judge)")

    # Load sessions: either from step3 results or re-run scenarios
    if step3_results_path and Path(step3_results_path).exists():
        print(f"Using step3 results: {step3_results_path}\n")
        sessions = _load_from_step3(step3_results_path)
    else:
        print(f"Re-running scenarios from: {scenarios_path}\n")
        sessions = _collect_recommendations(scenarios_path, api_url)

    if not sessions:
        print(f"{RED}No sessions to evaluate.{RESET}")
        sys.exit(1)

    if logger:
        logger.step_start("4", "Recommendation Quality", n_samples=len(sessions))

    # ── Judge each session ─────────────────────────────────────────────────────
    details = []
    verdicts = []
    grounded_flags = []

    for i, sess in enumerate(sessions, 1):
        sid      = sess.get("id", f"session_{i}")
        domain   = sess.get("domain", "unknown")
        info     = sess.get("collected_info") or {}
        rec      = sess.get("recommendation", "")

        print(f"  [{i}/{len(sessions)}] {sid}  [{domain}]")

        if not rec:
            print(f"    {YELLOW}⚠️  No recommendation found — skipping{RESET}")
            if logger:
                logger.sample("4", i, sid, "SKIP", reason="no_recommendation", domain=domain)
            continue
        if not info:
            print(f"    {YELLOW}⚠️  No collected_info — skipping{RESET}")
            if logger:
                logger.sample("4", i, sid, "SKIP", reason="no_collected_info", domain=domain)
            continue

        print(f"    Collected info: {json.dumps(info, ensure_ascii=False)[:80]}")
        print(f"    Recommendation: {rec[:100]}")
        print(f"    Judging...")

        judgment = judge_recommendation(info, rec, domain)
        verdicts.append(judgment["verdict"])
        grounded_flags.append(judgment["grounded"])

        v_color = GREEN if judgment["verdict"] == "Correct" else (
            YELLOW if judgment["verdict"] == "Partially" else RED
        )
        g_color = GREEN if judgment["grounded"] else RED
        print(f"    → {v_color}{judgment['verdict']}{RESET}  "
              f"grounded={g_color}{judgment['grounded']}{RESET}  "
              f"reason: {judgment['reasoning'][:80]}")

        if logger:
            status = "PASS" if judgment["verdict"] in ("Correct", "Partially") else "FAIL"
            logger.sample("4", i, sid, status,
                          domain=domain, verdict=judgment["verdict"],
                          grounded=judgment["grounded"])

        details.append({
            "id":          sid,
            "domain":      domain,
            "verdict":     judgment["verdict"],
            "grounded":    judgment["grounded"],
            "collected_info": info,
            "recommendation": rec[:500],
            "judgment":    judgment,
        })

    # ── Metrics ────────────────────────────────────────────────────────────────
    n = len(verdicts)
    correct_rate      = sum(1 for v in verdicts if v in ("Correct", "Partially")) / n if n else 0.0
    fully_correct     = sum(1 for v in verdicts if v == "Correct") / n if n else 0.0
    hallucination_rate = sum(1 for g in grounded_flags if not g) / n if n else 0.0
    grounded_rate      = 1.0 - hallucination_rate

    # ── Report ─────────────────────────────────────────────────────────────────
    print_separator()
    print(f"\n{BOLD}Recommendation Quality Summary{RESET}")
    print_separator()
    print(f"  Sessions evaluated         : {n}")

    cr_status  = status_icon(correct_rate, TARGET_CORRECT_RATE)
    hal_status = status_icon(hallucination_rate, TARGET_HALLUCINATION_RATE, higher_is_better=False)

    print(f"  Correct Rate (Correct+Partially): {BOLD}{correct_rate:.3f}{RESET}  {cr_status}  (target ≥ {TARGET_CORRECT_RATE})")
    print(f"  Fully Correct Rate         : {fully_correct:.3f}")
    print(f"  Hallucination Rate         : {BOLD}{hallucination_rate:.3f}{RESET}  {hal_status}  (target ≤ {TARGET_HALLUCINATION_RATE})")
    print(f"  Grounded Rate              : {grounded_rate:.3f}")

    print(f"\n  Verdict breakdown:")
    for verdict in ["Correct", "Partially", "Incorrect"]:
        cnt   = verdicts.count(verdict)
        color = GREEN if verdict == "Correct" else (YELLOW if verdict == "Partially" else RED)
        print(f"    {color}{verdict:<12}{RESET}: {cnt}/{n}")

    results = {
        "step": "4_recommendation",
        "n_evaluated": n,
        "correct_rate": round(correct_rate, 3),
        "fully_correct_rate": round(fully_correct, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "grounded_rate": round(grounded_rate, 3),
        "pass": correct_rate >= TARGET_CORRECT_RATE and hallucination_rate <= TARGET_HALLUCINATION_RATE,
        "details": details,
    }
    if logger:
        logger.step_end("4", metrics={
            "n_evaluated": n, "correct_rate": round(correct_rate, 3),
            "hallucination_rate": round(hallucination_rate, 3),
            "grounded_rate": round(grounded_rate, 3),
        }, elapsed=0, passed=results["pass"])
    save_results(results, output)
    return results


def _load_from_step3(path: str) -> list[dict]:
    """Load sessions from step3 results JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])
    sessions = []
    for sc in scenarios:
        # Extract last bot message as recommendation
        conv = sc.get("conversation", [])
        last_bot = ""
        for msg in reversed(conv):
            if msg.get("role") == "bot":
                last_bot = msg.get("content", "")
                break
        sessions.append({
            "id":           sc.get("id"),
            "domain":       sc.get("domain"),
            "collected_info": sc.get("collected_info"),
            "recommendation": last_bot,
        })
    return sessions


def _collect_recommendations(scenarios_path: str, api_url: str) -> list[dict]:
    """Re-run advisory scenarios and collect final recommendations."""
    from evaluation.step3_advisory_eval import _run_scenario

    scenarios = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    sessions = []

    for scenario in scenarios:
        sid     = str(uuid.uuid4())
        uid     = f"EVAL_REC_{uuid.uuid4().hex[:6]}"
        print(f"  Running scenario {scenario['id']}...")

        result = _run_scenario(
            opening_message=scenario["opening_message"],
            user_profile=scenario["user_profile"],
            session_id=sid,
            user_id=uid,
            api_url=api_url,
            max_turns=15,
        )

        conv    = result.get("conversation", [])
        last_bot = ""
        for msg in reversed(conv):
            if msg.get("role") == "bot":
                last_bot = msg.get("content", "")
                break

        sessions.append({
            "id":           scenario["id"],
            "domain":       scenario["domain"],
            "collected_info": result.get("collected_info"),
            "recommendation": last_bot,
        })

    return sessions


def main():
    import time as _time
    parser = argparse.ArgumentParser(description="Step 4: Recommendation Quality Evaluation")
    parser.add_argument("--step3-results", default=None,
                        help="Path to step3 output JSON (reuses conversations)")
    parser.add_argument("--scenarios",     default="evaluation/data/advisory_scenarios.json",
                        help="Fallback if step3 results not provided")
    parser.add_argument("--api-url",       default="http://localhost:8000")
    parser.add_argument("--output",        default=None)
    parser.add_argument("--output-dir",    default="evaluation/results")
    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = args.output or str(out_dir / f"step4_{ts}.json")
    logger = EvalLogger(str(out_dir / f"eval_step4_{ts}.log"))

    t0 = _time.time()
    result = run(args.step3_results, args.scenarios, args.api_url, output, logger=logger)
    elapsed = _time.time() - t0

    from evaluation.utils import save_step_report
    rpt = save_step_report("4", result, elapsed, out_dir, ts)
    print(f"\nReport : {rpt}")
    print(f"Log    : {out_dir / ('eval_step4_' + ts + '.log')}")


if __name__ == "__main__":
    main()
