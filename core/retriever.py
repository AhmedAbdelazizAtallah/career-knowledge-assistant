"""
core/retriever.py -- hybrid search (Chroma dense vectors + BM25 sparse
keywords, fused with Reciprocal Rank Fusion) followed by a Cohere Rerank
pass and parent-child context expansion.

Reranking is what keeps only the chunks actually relevant to the query,
instead of handing the generator a noisy top-N and risking "lost in the
middle" degradation. Parent expansion is what keeps a section's
content whole once any fragment of it has been judged relevant, instead
of handing the generator an arbitrary page-sized slice of it.
"""
import logging
import os
import re
from pathlib import Path

import cohere
from rank_bm25 import BM25Okapi

from core.embeddings import EMBED_MODEL, embed_texts
from core.ingestion import build_chunks
from core.reranker import rerank
from core.vectorstore import EmbedModelMismatchError, VectorStore

logger = logging.getLogger(__name__)

# "Top 20-50 candidates -> reranker -> top 5-10" per the standard
# two-stage retrieval pattern: the hybrid stage casts a wide net, the
# reranker (a cross-encoder, far more accurate than embedding similarity
# alone) narrows it down.
CANDIDATE_POOL = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "20"))
FINAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RRF_K = 60  # standard Reciprocal Rank Fusion constant

# Hybrid-stage gate: a chunk only enters the shortlist if at least one
# method is confident. Tuned empirically (see README) against real Cohere
# embedding scores for this dataset.
VECTOR_SIM_THRESHOLD = float(os.getenv("VECTOR_SIM_THRESHOLD", "30.0"))  # percent
BM25_SCORE_THRESHOLD = float(os.getenv("BM25_SCORE_THRESHOLD", "1.0"))

# Rerank-stage gate: Cohere's relevance_score is a calibrated 0..1
# probability (unlike raw cosine similarity), so this threshold is a much
# more reliable final backstop against forcing an answer out of weak
# context. Calibrated against real rerank-v3.5 scores on this dataset: a
# clearly-irrelevant tail sat at 0.066-0.13 while genuinely relevant
# results started at 0.28+ -- 0.15 sits cleanly in that gap.
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.15"))

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "at", "by", "from", "into",
    "about", "what", "which", "who", "whom", "how", "when", "where", "why",
    "do", "does", "did", "you", "your", "i", "we", "they", "he", "she", "them",
    "his", "her", "their", "our", "my", "me", "us", "not", "no", "so", "such",
    "than", "too", "very", "can", "will", "would", "should", "could", "may",
    "might", "must", "have", "has", "had",
}


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


class RagIndex:
    """Wraps a persistent VectorStore (dense search) plus an in-memory
    BM25 index (sparse search) rebuilt from the store's contents at
    startup -- cheap, since it's pure tokenization with no API calls, and
    it means Streamlit restarts don't require re-embedding (and
    re-billing) the whole corpus."""

    def __init__(self, client: cohere.ClientV2, store: VectorStore, embed_model: str = EMBED_MODEL):
        self.client = client
        self.store = store
        self.embed_model = embed_model  # must match whatever model embedded the store's vectors
        self.rows = store.get_all()
        self.token_sets = [set(tokenize(r["text"])) for r in self.rows]
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in self.rows]) if self.rows else None

    def __len__(self) -> int:
        return len(self.rows)


def build_index(
    data_dir: Path, client: cohere.ClientV2,
    store: VectorStore | None = None, force_reingest: bool = False,
    embed_model: str = EMBED_MODEL,
) -> RagIndex:
    """Connects to the persistent vector store and, only if it's empty
    (first run) or a rebuild is explicitly requested, runs ingestion and
    embeds+upserts the corpus. Normal app startup is just a fast local
    read -- no PDF parsing or embedding API calls on every restart.

    `embed_model` only takes effect on an actual (re)ingestion. Raises
    EmbedModelMismatchError rather than silently continuing if a
    non-empty store was built with a *different* embed model than
    requested -- comparing query and document vectors from two different
    embedding models produces meaningless similarity scores with no
    obvious symptom other than "retrieval seems broken."
    """
    # `store or VectorStore()` would be wrong here: VectorStore defines
    # __len__, so a caller-supplied store that's simply empty (the normal
    # state on a fresh deploy before first ingestion) is falsy and would
    # be silently discarded in favor of a brand-new default-path store.
    store = store if store is not None else VectorStore()
    stored_model = store.get_embed_model()

    if force_reingest:
        store.reset()
        stored_model = None
    elif len(store) > 0:
        if stored_model and stored_model != embed_model:
            raise EmbedModelMismatchError(
                f"The store at '{store.persist_dir}' was built with embed model "
                f"'{stored_model}', but '{embed_model}' was requested. Run "
                f"`python scripts/ingest.py --rebuild` to rebuild it with the new "
                f"model before querying, or set COHERE_EMBED_MODEL back to "
                f"'{stored_model}'."
            )
        if not stored_model:
            logger.warning(
                "Store at '%s' has %d chunk(s) but no recorded embed model (built "
                "before this check existed) -- cannot verify it matches '%s'. If "
                "retrieval looks broken, run `python scripts/ingest.py --rebuild`.",
                store.persist_dir, len(store), embed_model,
            )

    if force_reingest or len(store) == 0:
        chunks = build_chunks(data_dir)
        if not chunks:
            raise RuntimeError(f"No text could be extracted from any PDF in {data_dir}")
        embeddings = embed_texts(client, [c.text for c in chunks], input_type="search_document", model=embed_model)
        store.upsert(chunks, embeddings)
        store.set_embed_model(embed_model)
    return RagIndex(client, store, embed_model=embed_model)


def _hybrid_shortlist(
    index: RagIndex, query: str, candidate_pool: int, mode: str = "hybrid",
) -> list[dict]:
    """Stage 1: fuse dense (Chroma) + BM25 via RRF, drop anything neither
    method is confident about. Returns up to `candidate_pool` candidates,
    NOT the final answer set -- that's what reranking (stage 2) narrows
    down. Resilient to a dense-search outage: an empty vector result just
    falls through to BM25-only fusion instead of raising.

    `mode` restricts which channel(s) run -- "hybrid" (default), "vector"
    (dense only), or "bm25" (sparse only) -- so eval/run_eval.py can
    compare retrieval quality across channels on the same corpus."""
    n_chunks = len(index)
    if n_chunks == 0:
        return []

    candidates: dict[str, dict] = {}

    if mode in ("hybrid", "vector"):
        query_vec = embed_texts(index.client, [query], input_type="search_query", model=index.embed_model)
        dense_results = index.store.query(query_vec[0], n_results=candidate_pool) if len(query_vec) else []
        for rank, r in enumerate(dense_results):
            sim = round((1.0 - r["distance"]) * 100, 2)
            candidates[r["chunk_id"]] = dict(r)
            candidates[r["chunk_id"]].update(
                vector_sim=sim, vector_rank=rank, bm25_score=0.0, bm25_rank=None,
            )

    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    # A single incidental shared word (common on a small corpus, where a
    # generic word can look "rare" to BM25's IDF weighting) shouldn't count
    # as a real lexical match -- require at least 2 shared meaningful terms.
    min_overlap = min(2, len(query_token_set)) if query_token_set else 0

    if mode in ("hybrid", "bm25") and index.bm25 is not None:
        bm25_scores = index.bm25.get_scores(query_tokens)
        bm25_top = sorted(range(n_chunks), key=lambda i: bm25_scores[i], reverse=True)[:candidate_pool]
        for rank, idx in enumerate(bm25_top):
            if len(query_token_set & index.token_sets[idx]) < min_overlap:
                continue
            row = index.rows[idx]
            entry = candidates.setdefault(row["chunk_id"], dict(row) | {
                "vector_sim": 0.0, "vector_rank": None, "bm25_score": 0.0, "bm25_rank": None,
            })
            entry["bm25_score"] = round(float(bm25_scores[idx]), 2)
            entry["bm25_rank"] = rank

    def rrf(rank):
        return 0.0 if rank is None else 1.0 / (RRF_K + rank + 1)

    shortlist = []
    for c in candidates.values():
        c["rrf_score"] = rrf(c["vector_rank"]) + rrf(c["bm25_rank"])
        if c["vector_sim"] >= VECTOR_SIM_THRESHOLD or c["bm25_score"] >= BM25_SCORE_THRESHOLD:
            shortlist.append(c)

    shortlist.sort(key=lambda c: c["rrf_score"], reverse=True)
    return shortlist[:candidate_pool]


def retrieve(
    index: RagIndex, query: str, top_k: int | None = None,
    candidate_pool: int = CANDIDATE_POOL, expand_to_parent: bool = True,
    mode: str = "hybrid", use_rerank: bool = True,
) -> list[dict]:
    """Full retrieval pipeline: hybrid shortlist -> Cohere Rerank -> final
    relevance gate -> parent-child expansion. Returns [] when nothing
    survives -- callers must treat that as "truthfully say nothing
    relevant was found", not "try anyway".

    Each result carries `context_text`: the full parent-section text when
    `expand_to_parent` is set (so the generator sees the whole section a
    relevant fragment came from, not just that fragment), else the raw
    chunk text. Results are deduplicated by parent so two child chunks
    from the same section don't send duplicate context.

    `mode` and `use_rerank` exist for eval/run_eval.py to compare
    retrieval-channel and reranking choices on the same corpus; app.py
    always uses the defaults (full hybrid + rerank).
    """
    top_k = top_k or FINAL_TOP_K
    shortlist = _hybrid_shortlist(index, query, candidate_pool, mode=mode)
    if not shortlist:
        return []

    if use_rerank:
        ranked = rerank(index.client, query, [c["text"] for c in shortlist])
        score_threshold = RERANK_SCORE_THRESHOLD
    else:
        # Already sorted by rrf_score in _hybrid_shortlist and gated by
        # the hybrid-stage thresholds -- rrf_score isn't on the same 0..1
        # scale as a rerank relevance_score, so no further score gate here.
        ranked = list(enumerate(c["rrf_score"] for c in shortlist))
        score_threshold = float("-inf")

    results = []
    seen_parents: set[str] = set()
    for idx, score in ranked:
        if score < score_threshold:
            continue
        c = dict(shortlist[idx])
        parent_id = c.get("parent_id") or c["chunk_id"]
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        c["rerank_score"] = round(score, 4)
        c["context_text"] = c.get("parent_text") or c["text"] if expand_to_parent else c["text"]
        results.append(c)
        if len(results) >= top_k:
            break

    return results
