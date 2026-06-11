"""
Step 2 — RAG Quality Evaluation

Đánh giá chất lượng retrieval + generation cho PRODUCT_INFO_QA.

Mặc định: LLM-as-judge (không cần deps thêm)
  - Relevance (1-5): Answer có đúng chủ đề không?
  - Faithfulness (0/1): Answer có bịa thêm không?
  - Completeness (1-5): Answer có đầy đủ không?

Tùy chọn --use-ragas: dùng RAGAS framework
  - Yêu cầu: pip install ragas

Usage:
  python evaluation/step2_rag_eval.py
  python evaluation/step2_rag_eval.py --samples evaluation/data/rag_samples.json
  python evaluation/step2_rag_eval.py --use-ragas
  python evaluation/step2_rag_eval.py --output evaluation/results/step2_rag.json
"""

import argparse
import json
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils import (
    call_chat, require_api, print_header, print_separator,
    status_icon, save_results, judge_rag_answer, EvalLogger,
    GREEN, RED, YELLOW, CYAN, BOLD, RESET
)

TARGET_FAITHFULNESS  = 0.80   # Tỷ lệ câu không hallucinate
TARGET_RELEVANCE_AVG = 3.5    # Trên thang 1-5


def run(
    samples_path: str,
    api_url: str,
    use_ragas: bool,
    output: str | None,
    logger: EvalLogger | None = None,
) -> dict:
    require_api(api_url)

    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    print_header("Step 2 — RAG Quality Evaluation")
    print(f"Samples  : {len(samples)}")
    print(f"Method   : {'RAGAS' if use_ragas else 'LLM-as-judge'}\n")

    if logger:
        logger.step_start("2", "RAG Quality", n_samples=len(samples))

    user_id = f"EVAL_RAG_{uuid.uuid4().hex[:6]}"

    if use_ragas:
        return _run_ragas(samples, api_url, user_id, output, logger)
    else:
        return _run_llm_judge(samples, api_url, user_id, output, logger)


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

def _run_llm_judge(
    samples: list,
    api_url: str,
    user_id: str,
    output: str | None,
    logger: EvalLogger | None = None,
) -> dict:
    details = []
    relevance_scores = []
    faithfulness_scores = []
    completeness_scores = []

    for i, sample in enumerate(samples, 1):
        q   = sample["question"]
        gt  = sample.get("ground_truth_answer", "")
        cat = sample.get("category", "general")
        sid = str(uuid.uuid4())

        print(f"  [{i:2d}/{len(samples)}] {q[:65]}")

        try:
            resp   = call_chat(q, session_id=sid, user_id=user_id, api_url=api_url)
            answer = resp.get("answer", "")
            intent = resp.get("intent", "")
        except Exception as e:
            print(f"        ❌ API error: {e}")
            if logger:
                logger.sample("2", i, sample.get("id", f"s{i}"), "ERROR", error=str(e)[:60])
            continue

        if intent != "PRODUCT_INFO_QA":
            print(f"        {YELLOW}⚠️  Intent mismatch: {intent} (expected PRODUCT_INFO_QA){RESET}")

        # Judge
        scores = judge_rag_answer(q, answer, gt)
        relevance_scores.append(scores["relevance"])
        faithfulness_scores.append(scores["faithfulness"])
        completeness_scores.append(scores["completeness"])

        rel_icon = GREEN if scores["relevance"] >= 4 else (YELLOW if scores["relevance"] >= 3 else RED)
        fai_icon = GREEN if scores["faithfulness"] == 0 else RED
        faith_ok = scores["faithfulness"] == 0
        print(f"        relevance={rel_icon}{scores['relevance']}{RESET}/5  "
              f"faithfulness={fai_icon}{'OK' if faith_ok else 'HALLUCINATION'}{RESET}  "
              f"completeness={scores['completeness']}/5")
        if scores.get("reasoning"):
            print(f"        → {scores['reasoning'][:80]}")

        if logger:
            status = "PASS" if (faith_ok and scores["relevance"] >= 3) else "FAIL"
            logger.sample("2", i, sample.get("id", f"s{i}"), status,
                          relevance=scores["relevance"], faithfulness="OK" if faith_ok else "HALLUC",
                          completeness=scores["completeness"], intent=intent)

        details.append({
            "id":          sample.get("id"),
            "question":    q,
            "category":    cat,
            "bot_answer":  answer[:300],
            "ground_truth": gt[:300],
            "intent":      intent,
            "scores":      scores,
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n = len(details)
    avg_rel  = sum(relevance_scores) / n if n else 0.0
    # Faithfulness: 0=no hallucination (good), 1=hallucination (bad)
    # faithfulness_rate = fraction WITHOUT hallucination
    faith_rate = sum(1 for s in faithfulness_scores if s == 0) / n if n else 0.0
    avg_comp = sum(completeness_scores) / n if n else 0.0

    print_separator()
    print(f"\n{BOLD}RAG Evaluation Summary{RESET}")
    print_separator()
    print(f"  Samples evaluated         : {n}")

    rel_status   = status_icon(avg_rel, TARGET_RELEVANCE_AVG)
    faith_status = status_icon(faith_rate, TARGET_FAITHFULNESS)

    print(f"  Avg Relevance             : {BOLD}{avg_rel:.2f}{RESET}/5  {rel_status}  (target ≥ {TARGET_RELEVANCE_AVG})")
    print(f"  Faithfulness Rate         : {BOLD}{faith_rate:.3f}{RESET}    {faith_status}  (target ≥ {TARGET_FAITHFULNESS})")
    print(f"  Avg Completeness          : {avg_comp:.2f}/5")

    # Per-category breakdown
    cats = list(set(d["category"] for d in details))
    if len(cats) > 1:
        print(f"\n  Per-category:")
        for cat in sorted(cats):
            cat_details = [d for d in details if d["category"] == cat]
            cat_rel   = sum(d["scores"]["relevance"] for d in cat_details) / len(cat_details)
            cat_faith = sum(1 for d in cat_details if d["scores"]["faithfulness"] == 0) / len(cat_details)
            print(f"    {cat:<14}: relevance={cat_rel:.2f}  faithfulness={cat_faith:.2f}  (n={len(cat_details)})")

    results = {
        "step": "2_rag",
        "method": "llm_judge",
        "n_samples": n,
        "avg_relevance": round(avg_rel, 3),
        "faithfulness_rate": round(faith_rate, 3),
        "avg_completeness": round(avg_comp, 3),
        "pass": faith_rate >= TARGET_FAITHFULNESS and avg_rel >= TARGET_RELEVANCE_AVG,
        "details": details,
    }
    if logger:
        logger.step_end("2", metrics={
            "n_samples": n, "avg_relevance": round(avg_rel, 3),
            "faithfulness_rate": round(faith_rate, 3), "avg_completeness": round(avg_comp, 3),
        }, elapsed=0, passed=results["pass"])
    save_results(results, output)
    return results


# ── RAGAS ─────────────────────────────────────────────────────────────────────

def _run_ragas(
    samples: list,
    api_url: str,
    user_id: str,
    output: str | None,
    logger: EvalLogger | None = None,
) -> dict:
    """
    Run RAGAS evaluation. Requires `pip install ragas`.
    Uses the retriever directly to get contexts (imports from src).
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        print(f"{RED}❌ RAGAS not installed. Run: pip install ragas datasets --break-system-packages{RESET}")
        sys.exit(1)

    from src.retrieval.retriever import get_retriever

    rows = []
    for i, sample in enumerate(samples, 1):
        q   = sample["question"]
        gt  = sample.get("ground_truth_answer", "")
        sid = str(uuid.uuid4())

        print(f"  [{i:2d}/{len(samples)}] {q[:65]}")

        try:
            # Get bot answer
            resp   = call_chat(q, session_id=sid, user_id=user_id, api_url=api_url)
            answer = resp.get("answer", "")

            # Get contexts directly from retriever
            retriever = get_retriever()
            docs      = retriever.retrieve(q)
            contexts  = [d.get("content", d.get("text", "")) for d in docs] if docs else [""]

        except Exception as e:
            print(f"        ❌ Error: {e}")
            continue

        rows.append({
            "question":  q,
            "answer":    answer,
            "contexts":  contexts,
            "ground_truth": gt,
        })

    if not rows:
        print(f"{RED}No rows collected for RAGAS evaluation.{RESET}")
        sys.exit(1)

    print(f"\nRunning RAGAS on {len(rows)} samples...")
    dataset = Dataset.from_list(rows)

    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from config.settings import settings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(ChatOllama(
        model=settings.llm_model, base_url=settings.ollama_base_url
    ))
    ragas_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model=settings.embedding_model, base_url=settings.ollama_base_url
    ))

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    print_separator()
    print(f"\n{BOLD}RAGAS Results{RESET}")
    print_separator()
    faith_val = result["faithfulness"]
    relev_val = result["answer_relevancy"]
    print(f"  Faithfulness: {faith_val:.3f}")
    print(f"  Answer Relevancy: {relev_val:.3f}")

    return {
        "faithfulness_rate": round(float(faith_val), 3),
        "answer_relevancy": round(float(relev_val), 3),
    }


def main():
    import time as _time
    parser = argparse.ArgumentParser(description="Step 2: RAG Evaluation")
    parser.add_argument("--samples",    default="evaluation/data/rag_samples.json")
    parser.add_argument("--api-url",    default="http://localhost:8000")
    parser.add_argument("--use-ragas",  action="store_true")
    parser.add_argument("--output",     default=None)
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = args.output or str(out_dir / f"step2_{ts}.json")
    logger = EvalLogger(str(out_dir / f"eval_step2_{ts}.log"))

    t0 = _time.time()
    result = run(args.samples, args.api_url, use_ragas=args.use_ragas,
                 output=output, logger=logger)
    elapsed = _time.time() - t0

    from evaluation.utils import save_step_report
    rpt = save_step_report("2", result, elapsed, out_dir, ts)
    print(f"\nReport : {rpt}")
    print(f"Log    : {out_dir / ('eval_step2_' + ts + '.log')}")


if __name__ == "__main__":
    main()
