"""
core/observability.py -- structured per-query tracing, so a poor answer
can be debugged after the fact: what the query was rewritten to, what
was retrieved and how each candidate scored at every stage, what the
final context and answer were, and how long each stage took.

Writes one JSON line per query to a local file. No external tracing
service -- a single-process Streamlit app over a handful of PDFs doesn't
need one, and adding one would be exactly the "unnecessary framework"
this project deliberately avoids.
"""
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

LOG_PATH = Path(os.getenv("RAG_TRACE_LOG", "logs/traces.jsonl"))


@dataclass
class QueryTrace:
    original_query: str
    rewritten_query: str = ""
    retrieved: list[dict] = field(default_factory=list)
    final_context_chunk_ids: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    grounded: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": round(time.time(), 3),
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "retrieved": self.retrieved,
            "final_context_chunk_ids": self.final_context_chunk_ids,
            "answer": self.answer,
            "citations": self.citations,
            "grounded": self.grounded,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@contextmanager
def stage_timer(trace: QueryTrace, stage: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        trace.latency_ms[stage] = round((time.perf_counter() - start) * 1000, 1)


def log_trace(trace: QueryTrace) -> None:
    """Best-effort: a logging failure (disk full, a read-only filesystem
    on some hosting platforms) must never break answering a query."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
    except Exception:
        pass
