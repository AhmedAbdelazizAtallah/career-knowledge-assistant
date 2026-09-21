"""
core/retriever.py -- hybrid search (FAISS dense vectors + BM25 sparse
keywords, fused with Reciprocal Rank Fusion) followed by a Cohere Rerank
pass. The rerank step is what keeps only the 3-5 chunks that are actually
relevant, instead of handing the generator a noisy top-10 and risking
"lost in the middle" degradation.
"""
import os
import re
from pathlib import Path

import cohere
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from core.ingestion import Chunk, build_chunks

EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
EMBED_BATCH_SIZE = 96  # Cohere's per-call limit for texts

# Cohere periodically retires undated aliases -- if this starts 404ing with
# a "model was removed" message, check live models with:
#   cohere.ClientV2(api_key=...).models.list(endpoint="rerank")
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

CANDIDATE_POOL = 12   # how many chunks the hybrid stage shortlists
FINAL_TOP_K = 4        # how many chunks survive reranking, into the prompt
RRF_K = 60              # standard Reciprocal Rank Fusion constant

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


def embed_texts(client: cohere.ClientV2, texts: list[str], input_type: str) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        resp = client.embed(texts=batch, model=EMBED_MODEL, input_type=input_type,
                             embedding_types=["float"])
        vectors.extend(resp.embeddings.float_)
    return np.array(vectors, dtype="float32")


class RagIndex:
    """In-memory hybrid index: FAISS for dense vector search, BM25 for
    sparse keyword search. Built fresh at process startup from the PDFs in
    data/ -- no vector database file to persist or go stale."""

    def __init__(self, client: cohere.ClientV2, chunks: list[Chunk], embeddings: np.ndarray):
        self.client = client
        self.chunks = chunks
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        self.token_sets = [set(tokenize(c.text)) for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def __len__(self) -> int:
        return len(self.chunks)


def build_index(data_dir: Path, client: cohere.ClientV2) -> RagIndex:
    chunks = build_chunks(data_dir)
    if not chunks:
        raise RuntimeError(f"No text could be extracted from any PDF in {data_dir}")
    embeddings = embed_texts(client, [c.text for c in chunks], input_type="search_document")
    return RagIndex(client, chunks, embeddings)


def _hybrid_shortlist(index: RagIndex, query: str, candidate_pool: int) -> list[dict]:
    """Stage 1: fuse FAISS + BM25 via RRF, drop anything neither method is
    confident about. Returns up to `candidate_pool` candidates, NOT the
    final answer set -- that's what reranking (stage 2) narrows down."""
    candidates: dict[str, dict] = {}
    n_chunks = len(index.chunks)

    query_vec = embed_texts(index.client, [query], input_type="search_query")
    faiss.normalize_L2(query_vec)
    sims, idxs = index.faiss_index.search(query_vec, min(candidate_pool, n_chunks))
    for rank, (idx, sim) in enumerate(zip(idxs[0], sims[0])):
        if idx == -1:
            continue
        c = index.chunks[idx]
        candidates[c.chunk_id] = _chunk_to_dict(c)
        candidates[c.chunk_id].update(vector_sim=round(float(sim) * 100, 2), vector_rank=rank,
                                       bm25_score=0.0, bm25_rank=None)

    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    # A single incidental shared word (common on a small corpus, where a
    # generic word can look "rare" to BM25's IDF weighting) shouldn't count
    # as a real lexical match -- require at least 2 shared meaningful terms.
    min_overlap = min(2, len(query_token_set)) if query_token_set else 0

    bm25_scores = index.bm25.get_scores(query_tokens)
    bm25_top = sorted(range(n_chunks), key=lambda i: bm25_scores[i], reverse=True)[:candidate_pool]
    for rank, idx in enumerate(bm25_top):
        if len(query_token_set & index.token_sets[idx]) < min_overlap:
            continue
        c = index.chunks[idx]
        entry = candidates.setdefault(c.chunk_id, _chunk_to_dict(c) | {
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


def _chunk_to_dict(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id, "text": c.text, "file_name": c.file_name,
        "document_title": c.document_title, "section": c.section, "page_number": c.page_number,
    }


def retrieve(index: RagIndex, query: str, top_k: int = FINAL_TOP_K,
             candidate_pool: int = CANDIDATE_POOL) -> list[dict]:
    """Full retrieval pipeline: hybrid shortlist -> Cohere Rerank -> final
    relevance gate. Returns [] when nothing survives -- callers must treat
    that as "truthfully say nothing relevant was found", not "try anyway"."""
    shortlist = _hybrid_shortlist(index, query, candidate_pool)
    if not shortlist:
        return []

    rerank_resp = index.client.rerank(
        model=RERANK_MODEL, query=query,
        documents=[c["text"] for c in shortlist],
        top_n=min(top_k, len(shortlist)),
    )

    results = []
    for r in rerank_resp.results:
        if r.relevance_score < RERANK_SCORE_THRESHOLD:
            continue
        c = dict(shortlist[r.index])
        c["rerank_score"] = round(r.relevance_score, 4)
        results.append(c)

    return results
