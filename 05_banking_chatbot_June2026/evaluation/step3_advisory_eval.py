"""
Step 3 — Advisory Pipeline: Field Collection & Turn Efficiency

Simulate multi-turn advisory conversations.
Script đóng vai người dùng, tự động trả lời theo user_profile.

Bot hỏi field nào → script tra trong user_profile → trả lời.
Lặp đến khi missing_fields = [] hoặc turn limit đạt.

Metrics:
- Field Completion Rate = fields collected / fields required
- Turn Efficiency = turns taken / fields required
- Re-ask Rate = số lần bot hỏi lại field đã trả lời / total turns

Usage:
  python evaluation/step3_advisory_eval.py
  python evaluation/step3_advisory_eval.py --scenarios evaluation/data/advisory_scenarios.json
  python evaluation/step3_advisory_eval.py --max-turns 20
  python evaluation/step3_advisory_eval.py --output evaluation/results/step3_advisory.json
"""

import argparse
import json
import uuid
import time
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils import (
    call_chat, get_session_state, require_api, print_header, print_separator,
    status_icon, save_results, EvalLogger,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET
)

TARGET_COMPLETION = 0.90
TARGET_REASK      = 0.10
MAX_TURNS_DEFAULT = 15


def run(
    scenarios_path: str,
    api_url: str,
    max_turns: int,
    output: str | None,
    logger: EvalLogger | None = None,
) -> dict:
    require_api(api_url)

    scenarios = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    print_header("Step 3 — Advisory Field Collection")
    print(f"Scenarios : {len(scenarios)}  |  Max turns per scenario : {max_turns}\n")

    if logger:
        logger.step_start("3", "Advisory Field Collection", n_samples=len(scenarios))

    all_results = []

    for scenario in scenarios:
        sid      = str(uuid.uuid4())
        user_id  = f"EVAL_ADV_{uuid.uuid4().hex[:6]}"
        domain   = scenario["domain"]
        opening  = scenario["opening_message"]
        profile  = scenario["user_profile"]

        print(f"\n{BOLD}Scenario: {scenario['id']}  [{domain}]{RESET}")
        print(f"  Opening: '{opening}'")
        print_separator()

        session_result = _run_scenario(
            opening_message=opening,
            user_profile=profile,
            session_id=sid,
            user_id=user_id,
            api_url=api_url,
            max_turns=max_turns,
        )

        session_result["id"]     = scenario["id"]
        session_result["domain"] = domain
        session_result["note"]   = scenario.get("note", "")
        all_results.append(session_result)

        # Print scenario summary
        cr = session_result["completion_rate"]
        te = session_result["turn_efficiency"]
        rr = session_result["reask_rate"]
        cr_color = GREEN if cr >= TARGET_COMPLETION else (YELLOW if cr >= 0.75 else RED)
        print(f"\n  Completion  : {cr_color}{cr:.2f}{RESET}  "
              f"({session_result['fields_collected']}/{session_result['fields_required']} fields)")
        print(f"  Turns taken : {session_result['turns_taken']}")
        print(f"  Efficiency  : {te:.2f} turns/field")
        rr_color = GREEN if rr <= TARGET_REASK else RED
        print(f"  Re-ask rate : {rr_color}{rr:.2f}{RESET}")

        if logger:
            status = "PASS" if cr >= TARGET_COMPLETION else ("WARN" if cr >= 0.75 else "FAIL")
            logger.sample("3", len(all_results), scenario["id"], status,
                          domain=domain,
                          completion=f"{session_result['fields_collected']}/{session_result['fields_required']}",
                          completion_rate=cr, turns=session_result["turns_taken"],
                          reask_rate=rr)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    n = len(all_results)
    avg_cr = sum(r["completion_rate"] for r in all_results) / n if n else 0.0
    avg_te = sum(r["turn_efficiency"] for r in all_results) / n if n else 0.0
    avg_rr = sum(r["reask_rate"] for r in all_results) / n if n else 0.0

    print(f"\n{'═'*60}")
    print(f"{BOLD}Advisory Evaluation Summary{RESET}")
    print(f"{'═'*60}")
    print(f"  {'Scenario':<12} {'Domain':<14} {'Fields':<10} {'Completion':>10} {'Turns':>6} {'Re-ask':>7}")
    print_separator()
    for r in all_results:
        cr_c = GREEN if r["completion_rate"] >= TARGET_COMPLETION else RED
        print(f"  {r['id']:<12} {r['domain']:<14} "
              f"{r['fields_collected']}/{r['fields_required']:<7} "
              f"{cr_c}{r['completion_rate']:>10.2f}{RESET} "
              f"{r['turns_taken']:>6}  "
              f"{r['reask_rate']:>6.2f}")
    print_separator()

    cr_status = status_icon(avg_cr, TARGET_COMPLETION)
    rr_status = status_icon(avg_rr, TARGET_REASK, higher_is_better=False)
    print(f"\n  Avg Completion Rate : {BOLD}{avg_cr:.3f}{RESET}  {cr_status}  (target ≥ {TARGET_COMPLETION})")
    print(f"  Avg Turn Efficiency : {avg_te:.2f} turns/field")
    print(f"  Avg Re-ask Rate     : {BOLD}{avg_rr:.3f}{RESET}  {rr_status}  (target ≤ {TARGET_REASK})")

    results = {
        "step": "3_advisory",
        "n_scenarios": n,
        "avg_completion_rate": round(avg_cr, 3),
        "avg_turn_efficiency": round(avg_te, 3),
        "avg_reask_rate": round(avg_rr, 3),
        "pass": avg_cr >= TARGET_COMPLETION and avg_rr <= TARGET_REASK,
        "scenarios": all_results,
    }
    if logger:
        logger.step_end("3", metrics={
            "n_scenarios": n, "avg_completion_rate": round(avg_cr, 3),
            "avg_turn_efficiency": round(avg_te, 3), "avg_reask_rate": round(avg_rr, 3),
        }, elapsed=0, passed=results["pass"])
    save_results(results, output)
    return results


def _run_scenario(
    opening_message: str,
    user_profile: dict,
    session_id: str,
    user_id: str,
    api_url: str,
    max_turns: int,
) -> dict:
    """
    Simulate one advisory conversation.
    Returns dict with completion_rate, turns_taken, fields_collected, etc.
    """
    profile_answered: dict[str, str] = {}  # field → answer already given
    turns = []
    advisory_complete = False
    final_collected_info = {}

    # Turn 1: opening message
    print(f"  → USER: {opening_message}")
    try:
        resp   = call_chat(opening_message, session_id=session_id, user_id=user_id, api_url=api_url)
        answer = resp.get("answer", "")
        intent = resp.get("intent", "")
        collected = resp.get("collected_info") or {}
        final_collected_info = collected
        print(f"  ← BOT [{intent}]: {answer[:100]}")
    except Exception as e:
        print(f"  ❌ Turn 1 error: {e}")
        return _empty_result(user_profile)

    turns.append({"role": "user", "content": opening_message})
    turns.append({"role": "bot",  "content": answer, "intent": intent})

    # Extract any info provided in opening message
    _update_collected_from_opening(opening_message, user_profile, profile_answered)

    # Get session state to see missing_fields
    state = get_session_state(session_id, api_url)
    missing = state.get("missing_fields") or []

    # Subsequent turns: answer bot's questions
    reask_count = 0
    turn_n = 1

    while missing and turn_n < max_turns:
        turn_n += 1

        # Find an answer for the first missing field
        next_answer = _find_answer_for_question(answer, user_profile, profile_answered)

        if next_answer is None:
            # Can't map question to profile — send generic
            next_answer = "Tôi không chắc"

        print(f"  → USER: {next_answer}")
        try:
            resp     = call_chat(next_answer, session_id=session_id, user_id=user_id, api_url=api_url)
            answer   = resp.get("answer", "")
            intent   = resp.get("intent", "")
            collected = resp.get("collected_info") or {}
            if collected:
                final_collected_info = collected
        except Exception as e:
            print(f"  ❌ Turn {turn_n} error: {e}")
            break

        print(f"  ← BOT [{intent}]: {answer[:100]}")
        turns.append({"role": "user", "content": next_answer})
        turns.append({"role": "bot",  "content": answer, "intent": intent})

        prev_missing = set(missing)
        state   = get_session_state(session_id, api_url)
        missing = state.get("missing_fields") or []

        # Re-ask detection: if missing_fields didn't shrink after we answered
        new_missing = set(missing)
        if new_missing and new_missing >= prev_missing:
            reask_count += 1

        # Check if advisory is complete (intent shifted to recommendation)
        if intent in ("PRODUCT_CONSULT",) and not missing:
            advisory_complete = True
            break
        if "gợi ý" in answer.lower() or "recommend" in answer.lower() or "phù hợp" in answer.lower():
            advisory_complete = True
            break

    # ── Compute metrics ────────────────────────────────────────────────────────
    required_fields = list(user_profile.keys())
    n_required = len(required_fields)

    # Count how many required fields appear in final collected_info
    n_collected = sum(
        1 for f in required_fields
        if f in (final_collected_info or {}) and final_collected_info[f]
    )

    completion_rate = n_collected / n_required if n_required else 1.0
    turn_efficiency = turn_n / n_required if n_required else turn_n
    reask_rate      = reask_count / turn_n if turn_n else 0.0

    return {
        "fields_required": n_required,
        "fields_collected": n_collected,
        "completion_rate": round(completion_rate, 3),
        "turns_taken": turn_n,
        "turn_efficiency": round(turn_efficiency, 3),
        "reask_count": reask_count,
        "reask_rate": round(reask_rate, 3),
        "advisory_complete": advisory_complete,
        "collected_info": final_collected_info,
        "conversation": turns,
    }


def _update_collected_from_opening(
    message: str, profile: dict, answered: dict
) -> None:
    """Try to detect if opening message already contains profile info."""
    for field, value in profile.items():
        if value.lower()[:10] in message.lower():
            answered[field] = value


def _find_answer_for_question(
    bot_question: str, profile: dict, already_answered: dict
) -> str | None:
    """
    Given bot's question text, find the most relevant answer from user_profile.
    Strategy: check keyword overlap between question and field names/values.
    """
    question_lower = bot_question.lower()

    # Keyword hints per field category
    FIELD_HINTS = {
        "thu_nhap": ["thu nhập", "lương", "kiếm được", "tiền lương"],
        "chi_tieu": ["chi tiêu", "mua sắm", "spending", "tiêu dùng", "dùng tiền"],
        "uu_tien":  ["ưu tiên", "muốn", "quan tâm", "thích", "prefer"],
        "the_hien": ["thẻ", "hiện tại", "đang dùng", "ngân hàng nào"],
        "tuoi":     ["tuổi", "bao nhiêu tuổi", "độ tuổi"],
        "gia_dinh": ["gia đình", "kết hôn", "con", "vợ", "chồng"],
        "bao_hiem": ["bảo hiểm", "mục đích", "mục tiêu"],
        "ngan_sach":["ngân sách", "bao nhiêu tiền", "tháng bao nhiêu", "phí"],
        "muc_dich": ["mục đích", "vay để làm gì", "dùng để", "cần vay để"],
        "so_tien":  ["bao nhiêu tiền", "số tiền", "cần vay", "muốn vay"],
        "tai_san":  ["tài sản", "thế chấp", "sổ đỏ", "nhà đất"],
        "so_tien_gui": ["gửi bao nhiêu", "số tiền", "muốn gửi"],
        "thoi_han": ["kỳ hạn", "bao lâu", "thời hạn"],
        "muc_tieu": ["mục tiêu", "lãi suất", "linh hoạt", "rút trước"],
    }

    # Score each unanswered field
    best_field = None
    best_score = 0

    for field, value in profile.items():
        if field in already_answered:
            continue

        score = 0
        field_lower = field.lower()

        # Direct field name substring match
        if any(part in question_lower for part in field_lower.split("_")):
            score += 2

        # Keyword hints
        for hint_key, hints in FIELD_HINTS.items():
            if hint_key in field_lower:
                score += sum(1 for h in hints if h in question_lower)

        if score > best_score:
            best_score = score
            best_field = field

    if best_field:
        already_answered[best_field] = profile[best_field]
        return profile[best_field]

    # Fallback: return first unanswered field
    for field, value in profile.items():
        if field not in already_answered:
            already_answered[field] = value
            return value

    return None


def _empty_result(scenario_id: str) -> dict:
    return {
        "id": scenario_id, "domain": "unknown",
        "passed": False, "completion_rate": 0.0,
        "turn_efficiency": 0.0, "reask_rate": 0.0,
        "fields_required": 0, "fields_collected": 0,
        "reask_count": 0, "turns_taken": 0,
        "advisory_complete": False, "collected_info": {},
        "conversation": [],
    }


def main():
    import time as _time
    parser = argparse.ArgumentParser(description="Step 3: Advisory Pipeline Evaluation")
    parser.add_argument("--scenarios",  default="evaluation/data/advisory_scenarios.json")
    parser.add_argument("--api-url",    default="http://localhost:8000")
    parser.add_argument("--max-turns",  type=int, default=MAX_TURNS_DEFAULT)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = args.output or str(out_dir / f"step3_{ts}.json")
    logger = EvalLogger(str(out_dir / f"eval_step3_{ts}.log"))

    t0 = _time.time()
    result = run(args.scenarios, args.api_url, args.max_turns, output, logger=logger)
    elapsed = _time.time() - t0

    from evaluation.utils import save_step_report
    rpt = save_step_report("3", result, elapsed, out_dir, ts)
    print(f"\nReport : {rpt}")
    print(f"Log    : {out_dir / ('eval_step3_' + ts + '.log')}")


if __name__ == "__main__":
    main()
