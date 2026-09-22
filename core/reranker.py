"""
core/reranker.py -- Cohere Rerank, isolated so the model is swappable
(env var) independently of the rest of the retrieval pipeline.
"""
import os

import cohere

from core.llm_utils import call_with_retry

# Cohere periodically retires undated aliases -- if this starts 404ing
# with a "model was removed" message, check live models with:
#   cohere.ClientV2(api_key=...).models.list(endpoint="rerank")
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")


def rerank(
    client: cohere.ClientV2, query: str, documents: list[str],
    top_n: int | None = None, model: str = RERANK_MODEL,
) -> list[tuple[int, float]]:
    """Returns (original_index, relevance_score) pairs, sorted by
    relevance_score descending. `relevance_score` is a calibrated 0..1
    probability (unlike raw cosine similarity), which is what makes it a
    reliable final gate against forcing an answer out of weak context."""
    if not documents:
        return []
    resp = call_with_retry(lambda: client.rerank(
        model=model, query=query, documents=documents,
        top_n=min(top_n, len(documents)) if top_n else len(documents),
    ))
    return [(r.index, r.relevance_score) for r in resp.results]
