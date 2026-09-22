"""
eval/run_eval.py -- retrieval + answer quality evaluation over
eval/dataset.json. Always builds an isolated, disposable vector store
(never the app's persisted chroma_db/) so an eval run can safely try
different chunk sizes / embedding models without touching production
data.

Metrics:
  Retrieval (computed from what retrieve() actually returns, i.e. after
  rerank + parent-dedup -- the same set the generator would see):
    - Recall@K   (a relevant chunk appears anywhere in the top-K)
    - MRR        (reciprocal rank of the first relevant chunk)
    - nDCG@K     (rank-discounted; binary relevance from the dataset labels)
  Answer (LLM-judge, same Cohere client -- no extra framework/dependency):
    - answer_correctness  (does the answer convey the expected facts)
    - faithfulness        (is every claim in the answer supported by context)
  Citations (deterministic, not judged):
    - citation_correctness (every cited document actually among retrieved chunks)
  Refusal (the one out-of-corpus question in the dataset):
    - did the pipeline correctly decline to answer instead of guessing

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --chunk-size 400 --chunk-overlap 80 --tag small-chunks
    python eval/run_eval.py --mode bm25 --no-rerank --tag bm25-only-no-rerank
    python eval/run_eval.py --top-k 3 --skip-answer-metrics   # fast retrieval-only sweep
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import core.ingestion as ingestion
from core.embeddings import EMBED_MODEL
from core.generator import CHAT_MODEL, get_cohere_client
from core.llm_utils import call_with_retry, extract_text
from core.pipeline import answer_query
from core.retriever import CANDIDATE_POOL, FINAL_TOP_K, RagIndex, build_index, retrieve
from core.vectorstore import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_JUDGE_SYSTEM = (
    "You are a strict grading assistant. Respond with ONLY a single number "
    "between 0.0 and 1.0 -- no words, no explanation."
)


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _is_relevant(chunk: dict, relevant_document: str | None, relevant_pages: list[int]) -> bool:
    if not relevant_document or chunk["document_name"] != relevant_document:
        return False
    lo, hi = chunk.get("parent_page_start", chunk["page_number"]), chunk.get("parent_page_end", chunk["page_number"])
    return any(lo <= p <= hi for p in relevant_pages)


def _retrieval_metrics(retrieved: list[dict], relevant_document: str | None, relevant_pages: list[int]) -> dict:
    relevance = [1 if _is_relevant(c, relevant_document, relevant_pages) else 0 for c in retrieved]
    recall = 1.0 if any(relevance) else 0.0
    rr = next((1.0 / (i + 1) for i, r in enumerate(relevance) if r), 0.0)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return {"recall_at_k": recall, "mrr": rr, "ndcg_at_k": ndcg}


def _judge_score(client, prompt: str, judge_model: str) -> float | None:
    try:
        response = call_with_retry(lambda: client.chat(
            model=judge_model,
            messages=[{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
        ))
    except Exception:
        return None
    text = extract_text(response.message)
    m = re.search(r"(\d*\.?\d+)", text)
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(1))))


def _answer_correctness(client, question: str, expected: str, actual: str, judge_model: str) -> float | None:
    prompt = (
        f"Question: {question}\n\nReference answer: {expected}\n\nCandidate answer: {actual}\n\n"
        "Score how well the candidate answer conveys the same key facts as the reference answer "
        "(1.0 = fully correct and complete, 0.0 = missing or contradicts the key facts)."
    )
    return _judge_score(client, prompt, judge_model)


def _faithfulness(client, context: str, actual: str, judge_model: str) -> float | None:
    prompt = (
        f"Context:\n{context}\n\nAnswer:\n{actual}\n\n"
        "Score whether every factual claim in the answer is directly supported by the context "
        "(1.0 = fully supported, no invented facts; 0.0 = contains claims not in the context)."
    )
    return _judge_score(client, prompt, judge_model)


def _citation_correctness(citations: list[str], retrieved: list[dict]) -> float:
    if not citations:
        return 1.0  # nothing fabricated if nothing was cited
    retrieved_docs = {c["document_name"] for c in retrieved}
    hits = sum(1 for cite in citations if any(doc in cite for doc in retrieved_docs))
    return hits / len(citations)


def _build_eval_index(args) -> RagIndex:
    """Always an isolated store -- never the app's persisted chroma_db/ --
    so an eval sweep can freely vary chunk size / embedding model without
    any risk to production data."""
    ingestion.CHUNK_SIZE = args.chunk_size
    ingestion.CHUNK_OVERLAP = args.chunk_overlap

    safe_model = re.sub(r"[^a-zA-Z0-9]+", "-", args.embed_model)
    collection = f"eval_cs{args.chunk_size}_co{args.chunk_overlap}_{safe_model}"
    store = VectorStore(persist_dir=str(RESULTS_DIR / ".tmp_chroma"), collection_name=collection)

    client = get_cohere_client()
    return build_index(DATA_DIR, client, store=store, force_reingest=True, embed_model=args.embed_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-size", type=int, default=ingestion.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=ingestion.CHUNK_OVERLAP)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--judge-model", default=CHAT_MODEL)
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    parser.add_argument("--candidate-pool", type=int, default=CANDIDATE_POOL)
    parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid",
                         help="Which retrieval channel(s) to use -- compares hybrid fusion against vector-only or BM25-only.")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the Cohere Rerank stage.")
    parser.add_argument("--skip-answer-metrics", action="store_true",
                         help="Skip generation + LLM-judge scoring; report retrieval metrics only (fast).")
    parser.add_argument("--tag", default=None, help="Label for the results file; defaults to a timestamp.")
    parser.add_argument("--delay-seconds", type=float, default=0.0,
                         help="Pause between dataset items -- helps a trial API key (10 calls/min) get "
                              "through a full run with --skip-answer-metrics unset without 429s piling up.")
    args = parser.parse_args()

    dataset = _load_dataset()
    print(f"Config: chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap} "
          f"embed_model={args.embed_model} top_k={args.top_k} candidate_pool={args.candidate_pool} "
          f"mode={args.mode} rerank={not args.no_rerank}")

    index = _build_eval_index(args)
    print(f"Indexed {len(index)} chunks into an isolated eval store.\n")

    rows = []
    for item in dataset:
        if args.delay_seconds:
            time.sleep(args.delay_seconds)
        t0 = time.perf_counter()
        retrieved = retrieve(
            index, item["question"], top_k=args.top_k, candidate_pool=args.candidate_pool,
            mode=args.mode, use_rerank=not args.no_rerank,
        )
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        row = {"id": item["id"], "question": item["question"], "retrieval_ms": retrieval_ms,
               "n_retrieved": len(retrieved)}

        if item["relevant_document"] is None:
            row["is_refusal_case"] = True
            row["correctly_refused"] = len(retrieved) == 0
        else:
            row["is_refusal_case"] = False
            row.update(_retrieval_metrics(retrieved, item["relevant_document"], item["relevant_pages"]))

        if not args.skip_answer_metrics:
            result = answer_query(index, item["question"], [], top_k=args.top_k)
            row["grounded"] = result["grounded"]
            row["citation_correctness"] = _citation_correctness(result["citations"], retrieved)
            if item["relevant_document"] is None:
                row["correctly_refused"] = row["correctly_refused"] and not result["grounded"]
            else:
                context = "\n\n".join(c.get("context_text", c["text"]) for c in retrieved)
                row["answer_correctness"] = _answer_correctness(
                    index.client, item["question"], item["expected_answer"], result["answer"], args.judge_model,
                )
                row["faithfulness"] = _faithfulness(index.client, context, result["answer"], args.judge_model)

        rows.append(row)
        print(f"  {item['id']:<35} {row}")

    scored = [r for r in rows if not r["is_refusal_case"]]
    refusal = [r for r in rows if r["is_refusal_case"]]

    def _avg(key):
        vals = [r[key] for r in scored if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "config": vars(args),
        "n_questions": len(dataset),
        "recall_at_k": _avg("recall_at_k"),
        "mrr": _avg("mrr"),
        "ndcg_at_k": _avg("ndcg_at_k"),
        "answer_correctness": _avg("answer_correctness"),
        "faithfulness": _avg("faithfulness"),
        "citation_correctness": _avg("citation_correctness"),
        "refusal_accuracy": (
            round(sum(1 for r in refusal if r["correctly_refused"]) / len(refusal), 4) if refusal else None
        ),
    }

    print("\n=== Summary ===")
    for k, v in summary.items():
        if k != "config":
            print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{tag}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
