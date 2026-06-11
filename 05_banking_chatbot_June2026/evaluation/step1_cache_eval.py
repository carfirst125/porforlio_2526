"""
Step 1b — Cache Evaluation

Test cache precision/recall:
1. Seed: Gửi câu hỏi seed để populate cache
2. Paraphrase test: Gửi câu paraphrase → phải cache hit
3. Near-miss test: Gửi câu near-miss → KHÔNG được cache hit

Usage:
  python evaluation/step1_cache_eval.py
  python evaluation/step1_cache_eval.py --pairs evaluation/data/cache_pairs.json
  python evaluation/step1_cache_eval.py --no-seed
  python evaluation/step1_cache_eval.py --output-dir evaluation/results/myrun
"""

import argparse
import json
import time
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils import (
    call_chat, require_api, print_header, print_separator,
    status_icon, save_results, EvalLogger, save_step_report,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET
)

TARGET_FPR = 0.05
SEED_WAIT_SECONDS = 2


def run(
    pairs_path: str,
    api_url: str,
    do_seed: bool,
    output: str | None,
    logger: EvalLogger | None = None,
) -> dict:
    require_api(api_url)

    data = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    seed_qa          = data.get("seed_qa", [])
    paraphrase_pairs = data.get("paraphrase_pairs", [])
    near_miss_pairs  = data.get("near_miss_pairs", [])

    print_header("Step 1b — Cache Evaluation")
    n_total = len(paraphrase_pairs) + len(near_miss_pairs)
    if logger:
        logger.step_start("1b", "Cache Evaluation", n_samples=n_total)

    # ── Seed cache ────────────────────────────────────────────────────────────
    seed_user = f"EVAL_CACHE_SEED_{uuid.uuid4().hex[:6]}"

    if do_seed and seed_qa:
        print(f"Seeding cache with {len(seed_qa)} questions...")
        if logger:
            logger.info(f"[1b] Seeding {len(seed_qa)} questions")
        for item in seed_qa:
            q = item["question"]
            try:
                call_chat(q, user_id=seed_user, api_url=api_url)
                print(f"  ✅ Seeded: {q[:60]}")
            except Exception as e:
                print(f"  ❌ Seed failed: {e}")
                if logger:
                    logger.error(f"[1b] Seed failed: {e}")
        print(f"Waiting {SEED_WAIT_SECONDS}s for index to settle...")
        time.sleep(SEED_WAIT_SECONDS)
    else:
        print("Seed step skipped.\n")

    # ── Test paraphrase pairs (should hit) ────────────────────────────────────
    print(f"\n{BOLD}Paraphrase Pairs (should_hit=True){RESET}")
    print_separator()
    if logger:
        logger.info(f"[1b] Testing {len(paraphrase_pairs)} paraphrase pairs")

    test_user = f"EVAL_CACHE_TEST_{uuid.uuid4().hex[:6]}"
    para_results = []

    for i, pair in enumerate(paraphrase_pairs, 1):
        pid = pair["id"]
        q   = pair["paraphrase"]
        expected_hit = pair["should_hit"]
        sid = str(uuid.uuid4())
        try:
            resp       = call_chat(q, session_id=sid, user_id=test_user, api_url=api_url)
            from_cache = resp.get("from_cache", False)
            similarity = resp.get("cache_similarity") or 0.0
        except Exception as e:
            from_cache = False; similarity = 0.0
            print(f"  [{pid}] ERROR: {e}")
            if logger:
                logger.sample("1b-para", i, pid, "ERROR", error=str(e)[:60])

        correct  = from_cache == expected_hit
        hit_icon = "✅" if from_cache else "❌"
        print(f"  [{pid}] {hit_icon}  cache={from_cache}  sim={similarity:.3f}  q={q[:55]}")
        if logger:
            logger.sample("1b-para", i, pid, "PASS" if correct else "FAIL",
                          from_cache=from_cache, similarity=round(similarity, 3),
                          expected_hit=expected_hit)
        para_results.append({
            "id": pid, "query": q, "expected_hit": expected_hit,
            "from_cache": from_cache, "similarity": round(similarity, 3),
            "correct": correct,
        })

    # ── Test near-miss pairs (should NOT hit) ─────────────────────────────────
    print(f"\n{BOLD}Near-miss Pairs (should_hit=False){RESET}")
    print_separator()
    if logger:
        logger.info(f"[1b] Testing {len(near_miss_pairs)} near-miss pairs")

    near_results = []

    for i, pair in enumerate(near_miss_pairs, 1):
        pid = pair["id"]
        q   = pair["near_miss"]
        expected_hit = pair["should_hit"]
        sid = str(uuid.uuid4())
        try:
            resp       = call_chat(q, session_id=sid, user_id=test_user, api_url=api_url)
            from_cache = resp.get("from_cache", False)
            similarity = resp.get("cache_similarity") or 0.0
        except Exception as e:
            from_cache = False; similarity = 0.0
            print(f"  [{pid}] ERROR: {e}")
            if logger:
                logger.sample("1b-near", i, pid, "ERROR", error=str(e)[:60])

        is_fp   = from_cache and not expected_hit
        correct = from_cache == expected_hit
        fp_icon = f"{RED}❌ FP{RESET}" if from_cache else f"{GREEN}✅{RESET}"
        print(f"  [{pid}] {fp_icon}  cache={from_cache}  sim={similarity:.3f}  q={q[:55]}")
        if logger:
            status = "FAIL(FP)" if is_fp else ("PASS" if correct else "FAIL")
            logger.sample("1b-near", i, pid, status,
                          from_cache=from_cache, similarity=round(similarity, 3),
                          false_positive=is_fp)
        near_results.append({
            "id": pid, "query": q, "expected_hit": expected_hit,
            "from_cache": from_cache, "similarity": round(similarity, 3),
            "correct": correct, "false_positive": is_fp,
        })

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_para = len(para_results)
    tpr    = sum(r["from_cache"] for r in para_results) / n_para if n_para else 0.0
    n_near = len(near_results)
    fpr    = sum(r["false_positive"] for r in near_results) / n_near if n_near else 0.0

    print_separator()
    print(f"\n{BOLD}Cache Evaluation Summary{RESET}")
    print_separator()
    tpr_status = status_icon(tpr, 0.75)
    fpr_status = status_icon(fpr, TARGET_FPR, higher_is_better=False)
    print(f"  Paraphrase pairs tested  : {n_para}")
    print(f"  True Positive Rate (TPR) : {tpr:.3f}  {tpr_status}  ({sum(r['from_cache'] for r in para_results)}/{n_para} correctly hit)")
    print(f"  Near-miss pairs tested   : {n_near}")
    print(f"  False Positive Rate (FPR): {fpr:.3f}  {fpr_status}  (target <= {TARGET_FPR})")
    if fpr > TARGET_FPR:
        print(f"\n  {YELLOW}⚠️  FPR above target. Consider increasing CACHE_SIMILARITY_THRESHOLD in .env{RESET}")
    if tpr < 0.60:
        print(f"\n  {YELLOW}⚠️  Low TPR. Consider decreasing CACHE_SIMILARITY_THRESHOLD in .env{RESET}")

    results = {
        "step": "1b_cache",
        "tpr": round(tpr, 3), "fpr": round(fpr, 3),
        "pass": fpr <= TARGET_FPR,
        "n_paraphrase": n_para, "n_near_miss": n_near,
        "paraphrase_details": para_results,
        "near_miss_details": near_results,
    }
    if logger:
        logger.step_end("1b", metrics={"tpr": round(tpr, 3), "fpr": round(fpr, 3)},
                        elapsed=0, passed=results["pass"])
    save_results(results, output)
    return results


def main():
    import time as _time
    parser = argparse.ArgumentParser(description="Step 1b: Cache Evaluation")
    parser.add_argument("--pairs",      default="evaluation/data/cache_pairs.json")
    parser.add_argument("--api-url",    default="http://localhost:8000")
    parser.add_argument("--no-seed",    action="store_true")
    parser.add_argument("--output",     default=None)
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = args.output or str(out_dir / f"step1b_{ts}.json")
    logger = EvalLogger(str(out_dir / f"eval_step1b_{ts}.log"))

    t0 = _time.time()
    result = run(args.pairs, args.api_url, do_seed=not args.no_seed,
                 output=output, logger=logger)
    elapsed = _time.time() - t0

    rpt = save_step_report("1b", result, elapsed, out_dir, ts)
    print(f"\nReport : {rpt}")
    print(f"Log    : {out_dir / ('eval_step1b_' + ts + '.log')}")


if __name__ == "__main__":
    main()
