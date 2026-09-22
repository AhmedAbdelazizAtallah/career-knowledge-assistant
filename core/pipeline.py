"""
core/pipeline.py -- the single entry point that wires the whole query
flow together: conversational rewrite -> hybrid retrieval -> grounded
generation, with per-stage latency and a full observability trace on
every call. app.py and eval/run_eval.py both call `answer_query` rather
than assembling the stages themselves, so tracing can't be skipped by a
caller that forgets to add it.
"""
from core.generator import generate_answer
from core.observability import QueryTrace, log_trace, stage_timer
from core.query_processing import rewrite_query
from core.retriever import retrieve


def _trace_row(c: dict) -> dict:
    return {
        "chunk_id": c["chunk_id"], "document_name": c["document_name"],
        "section": c["section"], "page_number": c["page_number"],
        "vector_sim": c.get("vector_sim"), "bm25_score": c.get("bm25_score"),
        "rrf_score": round(c["rrf_score"], 5) if c.get("rrf_score") is not None else None,
        "rerank_score": c.get("rerank_score"),
    }


def answer_query(index, query: str, history: list[dict], top_k: int | None = None) -> dict:
    trace = QueryTrace(original_query=query)
    try:
        with stage_timer(trace, "rewrite_ms"):
            rewritten = rewrite_query(index.client, history, query)
        trace.rewritten_query = rewritten

        with stage_timer(trace, "retrieve_ms"):
            retrieved = retrieve(index, rewritten, top_k=top_k)
        trace.retrieved = [_trace_row(c) for c in retrieved]
        trace.final_context_chunk_ids = [c["chunk_id"] for c in retrieved]

        with stage_timer(trace, "generate_ms"):
            result = generate_answer(index, rewritten, history, top_k=top_k, retrieved=retrieved)

        trace.answer = result["answer"]
        trace.citations = result["citations"]
        trace.grounded = result["grounded"]
        result["rewritten_query"] = rewritten
        return result
    except Exception as exc:
        trace.error = str(exc)
        raise
    finally:
        log_trace(trace)
