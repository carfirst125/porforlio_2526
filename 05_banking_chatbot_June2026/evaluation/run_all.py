"""
run_all.py — Run all evaluation steps in sequence.

Usage:
  python evaluation/run_all.py
  python evaluation/run_all.py --steps 1a 1b 2           # chọn steps
  python evaluation/run_all.py --output-dir evaluation/results/run_20250115
  python evaluation/run_all.py --fail-fast               # dừng khi step fail
  python evaluation/run_all.py --with-quality-check      # thêm quality check cho step 1c
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils import (
    require_api, print_header, print_separator,
    status_icon, save_results, EvalLogger, save_step_report,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET
)

VALID_STEPS = ["1a", "1b", "1c", "2", "3", "4"]

STEP_TARGETS = {
    "1a": {"name": "Intent Classification",   "metric": "macro_f1",           "target": 0.85, "higher": True},
    "1b": {"name": "Cache FPR",               "metric": "fpr",                "target": 0.05, "higher": False},
    "1c": {"name": "Feedback Sentiment",      "metric": "intent_accuracy",    "target": 0.85, "higher": True},
    "2":  {"name": "RAG Faithfulness",        "metric": "faithfulness_rate",  "target": 0.80, "higher": True},
    "3":  {"name": "Advisory Completion",     "metric": "avg_completion_rate","target": 0.90, "higher": True},
    "4":  {"name": "Recommendation Quality",  "metric": "correct_rate",       "target": 0.70, "higher": True},
}


def run_all(
    steps: list[str],
    output_dir: str,
    api_url: str,
    fail_fast: bool,
    with_quality_check: bool,
) -> None:
    require_api(api_url)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Init logger ────────────────────────────────────────────────────────────
    log_path = out_dir / f"eval_{ts}.log"
    logger = EvalLogger(str(log_path))
    print(f"{CYAN}Log file : {log_path.resolve()}{RESET}")

    print_header(f"VIB Chatbot V3 — Full Evaluation  [{ts}]")
    print(f"Steps     : {steps}")
    print(f"Output    : {out_dir.resolve()}")
    print(f"Fail-fast : {fail_fast}\n")

    logger.info(f"Run started — steps={steps} api={api_url} fail_fast={fail_fast}")

    step_results: dict[str, dict] = {}
    step_times:   dict[str, float] = {}

    for step in steps:
        if step not in VALID_STEPS:
            print(f"{YELLOW}Unknown step '{step}' — skipping{RESET}")
            logger.warning(f"Unknown step '{step}' — skipping")
            continue

        cfg = STEP_TARGETS[step]
        print(f"\n{'━'*60}")
        print(f"{BOLD}Running Step {step}: {cfg['name']}{RESET}")
        print(f"{'━'*60}")

        t0 = time.time()
        out_path = str(out_dir / f"step{step}_{ts}.json")
        result = {}

        try:
            if step == "1a":
                from evaluation.step1_intent_eval import run
                result = run("evaluation/data/intent_samples.json", api_url, out_path, logger=logger)

            elif step == "1b":
                from evaluation.step1_cache_eval import run
                result = run("evaluation/data/cache_pairs.json", api_url, do_seed=True, output=out_path, logger=logger)

            elif step == "1c":
                from evaluation.step1_feedback_eval import run
                result = run("evaluation/data/feedback_samples.json", api_url, with_quality_check, out_path, logger=logger)

            elif step == "2":
                from evaluation.step2_rag_eval import run
                result = run("evaluation/data/rag_samples.json", api_url, use_ragas=False, output=out_path, logger=logger)

            elif step == "3":
                from evaluation.step3_advisory_eval import run
                result = run("evaluation/data/advisory_scenarios.json", api_url, max_turns=15, output=out_path, logger=logger)

            elif step == "4":
                step3_path = str(out_dir / f"step3_{ts}.json") if (out_dir / f"step3_{ts}.json").exists() else None
                from evaluation.step4_recommend_eval import run
                result = run(step3_path, "evaluation/data/advisory_scenarios.json", api_url, out_path, logger=logger)

        except Exception as e:
            print(f"\n{RED}❌ Step {step} raised exception: {e}{RESET}")
            logger.error(f"Step {step} raised exception: {e}")
            result = {"error": str(e), "pass": False}

        elapsed = time.time() - t0
        # Patch elapsed into logger.step_end (called inside each step with 0)
        step_results[step] = result
        step_times[step]   = elapsed
        print(f"\nStep {step} completed in {elapsed:.1f}s")
        logger.info(f"Step {step} wall-clock elapsed: {elapsed:.1f}s")

        # ── Export per-step markdown report ────────────────────────────────────
        try:
            report_path = save_step_report(step, result, elapsed, out_dir, ts)
            print(f"{CYAN}Report   : {report_path}{RESET}")
            logger.info(f"Step {step} report saved: {report_path}")
        except Exception as re:
            print(f"{YELLOW}⚠️  Could not save step report: {re}{RESET}")
            logger.warning(f"Could not save step {step} report: {re}")

        passed = result.get("pass", False)
        if fail_fast and not passed:
            print(f"\n{RED}❌ Fail-fast: stopping after step {step} failure.{RESET}")
            logger.error(f"Fail-fast triggered at step {step}")
            break

    # ── Final scoreboard ───────────────────────────────────────────────────────
    _print_scoreboard(steps, step_results, step_times)

    # Save summary JSON
    summary = {
        "run_at": ts,
        "api_url": api_url,
        "steps_run": steps,
        "log_file": str(log_path),
        "summary": {
            step: {
                "name":    STEP_TARGETS.get(step, {}).get("name", step),
                "pass":    step_results.get(step, {}).get("pass", False),
                "metric":  _get_metric_value(step, step_results.get(step, {})),
                "elapsed": round(step_times.get(step, 0), 1),
            }
            for step in steps if step in step_results
        },
    }
    summary_path = str(out_dir / f"summary_{ts}.json")
    Path(summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{CYAN}Summary saved : {summary_path}{RESET}")
    print(f"{CYAN}Full log      : {log_path.resolve()}{RESET}")
    print(f"{CYAN}Step reports  : {out_dir.resolve()}/report_step*_{ts}.md{RESET}")

    logger.info("=" * 70)
    logger.info(f"Run complete — summary: {summary_path}")
    for step, info in summary["summary"].items():
        status = "PASS" if info["pass"] else "FAIL"
        logger.info(f"  [{status}] Step {step}: {info['name']} — metric={info['metric']} elapsed={info['elapsed']}s")


def _get_metric_value(step: str, result: dict) -> float | None:
    cfg    = STEP_TARGETS.get(step, {})
    metric = cfg.get("metric")
    if not metric or not result:
        return None
    return result.get(metric)


def _print_scoreboard(steps: list, results: dict, times: dict) -> None:
    print(f"\n\n{'╔' + '═'*58 + '╗'}")
    print(f"{'║'}{' VIB Chatbot V3 — Evaluation Scoreboard ':^58}{'║'}")
    print(f"{'╠' + '═'*58 + '╣'}")
    print(f"{'║'} {'Step':<6} {'Name':<28} {'Metric':<12} {'Result':<8} {'Status':<12}{'║'}")
    print(f"{'╠' + '═'*58 + '╣'}")

    passes = 0
    warns  = 0
    fails  = 0

    for step in steps:
        if step not in results:
            continue
        cfg    = STEP_TARGETS.get(step, {"name": step, "metric": "pass", "target": 1.0, "higher": True})
        result = results[step]
        val    = _get_metric_value(step, result)
        passed = result.get("pass", False)

        if val is None:
            status_str = f"{RED}ERROR {RESET}"
            fails += 1
        elif passed:
            status_str = f"{GREEN}✅ PASS {RESET}"
            passes += 1
        else:
            target = cfg["target"]
            higher = cfg["higher"]
            if higher and val >= target * 0.9:
                status_str = f"{YELLOW}⚠️  WARN {RESET}"
                warns += 1
            elif not higher and val <= target * 1.5:
                status_str = f"{YELLOW}⚠️  WARN {RESET}"
                warns += 1
            else:
                status_str = f"{RED}❌ FAIL {RESET}"
                fails += 1

        val_str = f"{val:.3f}" if val is not None else "N/A"
        elapsed = times.get(step, 0)
        print(f"{'║'} {step:<6} {cfg['name']:<28} {cfg.get('metric','')[:11]:<12} {val_str:<8} {status_str:<12}{'║'}")

    print(f"{'╠' + '═'*58 + '╣'}")
    total = passes + warns + fails
    overall = f"{GREEN}{passes} PASS{RESET}  {YELLOW}{warns} WARN{RESET}  {RED}{fails} FAIL{RESET}"
    print(f"{'║'} Overall: {overall}{' '*(47 - len(str(passes)) - len(str(warns)) - len(str(fails)))}{'║'}")
    print(f"{'╚' + '═'*58 + '╝'}")


def main():
    parser = argparse.ArgumentParser(description="Run all evaluation steps")
    parser.add_argument(
        "--steps", nargs="+", default=VALID_STEPS, choices=VALID_STEPS,
        help="Steps to run (default: all)"
    )
    parser.add_argument("--output-dir",         default=None,
                        help="Output directory (default: evaluation/results/run_<timestamp>)")
    parser.add_argument("--api-url",             default="http://localhost:8000")
    parser.add_argument("--fail-fast",           action="store_true")
    parser.add_argument("--with-quality-check",  action="store_true",
                        help="Enable LLM quality check in step 1c")
    args = parser.parse_args()

    from datetime import datetime as _dt
    ts_main = _dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or f"evaluation/results/run_{ts_main}"
    run_all(
        steps=args.steps,
        output_dir=out_dir,
        api_url=args.api_url,
        fail_fast=args.fail_fast,
        with_quality_check=args.with_quality_check,
    )


if __name__ == "__main__":
    main()
