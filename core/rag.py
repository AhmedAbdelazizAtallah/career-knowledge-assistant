"""
core/rag.py -- the whole RAG engine: PDF ingestion, chunking, hybrid
retrieval (FAISS + BM25), and Cohere-powered generation.

SECURITY: the Cohere API key is read exclusively from the server-side
environment via `resolve_secret()` (env var, or Streamlit secrets when run
under Streamlit). It is never hardcoded, never logged, and never returned
to a caller -- only an authenticated `cohere.Client` instance is exposed.
This module has no knowledge of HTTP requests or browsers, so there is no
code path that could leak the key to a client.
"""
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cohere
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------- config --

EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 40
MIN_PAGE_CHARS = 30

TOP_K = 4
CANDIDATE_POOL = 10
MAX_HISTORY_TURNS = 3
EMBED_BATCH_SIZE = 96  # Cohere's per-call limit for texts

# A vector match below this is unreliable; an unrelated query still scores
# some nonzero cosine similarity against any collection. A BM25 score of 0
# means literally no shared vocabulary. A chunk is kept only if at least
# one method is confident -- this is what lets the system say "I don't
# know" instead of forcing an answer out of irrelevant context. These are
# tuned for MiniLM-scale embeddings as a starting point; validate against
# real Cohere embedding scores for your documents and adjust via env vars.
VECTOR_SIM_THRESHOLD = float(os.getenv("VECTOR_SIM_THRESHOLD", "30.0"))  # percent
BM25_SCORE_THRESHOLD = float(os.getenv("BM25_SCORE_THRESHOLD", "1.0"))
RRF_K = 60  # standard Reciprocal Rank Fusion constant, no tuning needed

SYSTEM_PREAMBLE = (
    "You are an expert career consultant. Answer the user's question directly and "
    "comprehensively using ONLY the provided documents, and take the conversation "
    "history into account for follow-up questions.\n"
    "Rules:\n"
    "1. NEVER invent, extrapolate, or fabricate examples, numbers, or facts that are "
    "not present in the provided documents.\n"
    "2. If a formula or framework is given in the documents, present it completely, "
    "including its accompanying rules.\n"
    "3. If the documents do not contain enough information to answer, say so "
    "explicitly instead of guessing."
)

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


# ------------------------------------------------------------- secrets --

def resolve_secret(name: str) -> str | None:
    """Read a secret from Streamlit's secrets manager if available, else
    from the OS environment. Never accepts a value from a request, a
    client, or a URL -- only server-side sources."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def get_cohere_client() -> cohere.Client:
    api_key = resolve_secret("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. For local runs, copy .env.example to .env and "
            "fill it in (or set the env var directly). For a deployed app, add it in "
            "your platform's secrets manager -- see README.md. It must never be "
            "hardcoded in source or committed to git."
        )
    return cohere.Client(api_key=api_key)


# ----------------------------------------------------------- ingestion --

@dataclass(frozen=True)
class PageRecord:
    file_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_name: str
    page_number: int
    text: str


def extract_text_from_pdfs(folder_path: Path) -> list[PageRecord]:
    records: list[PageRecord] = []
    pdf_files = sorted(
        f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"
    )
    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                records.append(PageRecord(pdf_path.name, page_num, text))
    return records


_DISCLAIMER_PATTERN = re.compile(
    r"This guide reflects widely-observed hiring practices.*?industry, and location\.",
    re.DOTALL | re.IGNORECASE,
)


def clean_document_text(text: str) -> str:
    text = _DISCLAIMER_PATTERN.sub("", text)
    text = text.replace("\x7f", "")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # rejoin hyphenated line breaks
    text = re.sub(r"(?:[\r\n]+\s*)+[■•▪►*\-]\s*", r"\n- ", text)  # normalize bullets
    text = re.sub(r"\n[-*]\s*\n", "\n- ", text)
    text = re.sub(r"\n(\d+\.\s)", r"\n\n\1", text)  # break before numbered sections
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


def build_chunks(data_dir: Path) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: list[Chunk] = []
    for page in extract_text_from_pdfs(data_dir):
        cleaned = clean_document_text(page.text)
        if len(cleaned) <= MIN_PAGE_CHARS:
            continue
        for split in splitter.split_text(cleaned):
            stripped = split.strip()
            if len(stripped) <= MIN_CHUNK_CHARS:
                continue
            digest = hashlib.sha256(
                f"{page.file_name}|{page.page_number}|{stripped}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=f"{page.file_name}_p{page.page_number}_{digest}",
                file_name=page.file_name, page_number=page.page_number, text=stripped,
            ))
    return chunks


# --------------------------------------------------------- vector index --

def embed_texts(client: cohere.Client, texts: list[str], input_type: str) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        resp = client.embed(texts=batch, model=EMBED_MODEL, input_type=input_type)
        vectors.extend(resp.embeddings)
    return np.array(vectors, dtype="float32")


class RagIndex:
    """In-memory hybrid index: FAISS for dense vector search, BM25 for
    sparse keyword search. Built fresh at process startup from the PDFs in
    data/ -- no vector database file to persist or go stale."""

    def __init__(self, client: cohere.Client, chunks: list[Chunk], embeddings: np.ndarray):
        self.client = client
        self.chunks = chunks
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        self.token_sets = [set(tokenize(c.text)) for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def __len__(self) -> int:
        return len(self.chunks)


def build_index(data_dir: Path, client: cohere.Client) -> RagIndex:
    chunks = build_chunks(data_dir)
    if not chunks:
        raise RuntimeError(f"No text could be extracted from any PDF in {data_dir}")
    embeddings = embed_texts(client, [c.text for c in chunks], input_type="search_document")
    return RagIndex(client, chunks, embeddings)


# ------------------------------------------------------------ retrieval --

def retrieve(index: RagIndex, query: str, top_k: int = TOP_K,
             candidate_pool: int = CANDIDATE_POOL) -> list[dict]:
    """Hybrid retrieval: fuse FAISS vector search with BM25 keyword search
    via Reciprocal Rank Fusion, then drop anything neither method is
    confident about. Returns [] when nothing clears the bar -- callers
    must treat that as "answer truthfully that nothing relevant was
    found", not as "try anyway"."""
    candidates: dict[str, dict] = {}
    n_chunks = len(index.chunks)

    query_vec = embed_texts(index.client, [query], input_type="search_query")
    faiss.normalize_L2(query_vec)
    sims, idxs = index.faiss_index.search(query_vec, min(candidate_pool, n_chunks))
    for rank, (idx, sim) in enumerate(zip(idxs[0], sims[0])):
        if idx == -1:
            continue
        c = index.chunks[idx]
        candidates[c.chunk_id] = {
            "chunk_id": c.chunk_id, "text": c.text,
            "file_name": c.file_name, "page_number": c.page_number,
            "vector_sim": round(float(sim) * 100, 2), "vector_rank": rank,
            "bm25_score": 0.0, "bm25_rank": None,
        }

    # A single incidental shared word (common on a small corpus, where a
    # generic word can look "rare" to BM25's IDF weighting) shouldn't count
    # as a real lexical match -- require at least 2 shared meaningful terms.
    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    min_overlap = min(2, len(query_token_set)) if query_token_set else 0

    bm25_scores = index.bm25.get_scores(query_tokens)
    bm25_top = sorted(range(n_chunks), key=lambda i: bm25_scores[i], reverse=True)[:candidate_pool]
    for rank, idx in enumerate(bm25_top):
        if len(query_token_set & index.token_sets[idx]) < min_overlap:
            continue
        c = index.chunks[idx]
        entry = candidates.setdefault(c.chunk_id, {
            "chunk_id": c.chunk_id, "text": c.text,
            "file_name": c.file_name, "page_number": c.page_number,
            "vector_sim": 0.0, "vector_rank": None,
            "bm25_score": 0.0, "bm25_rank": None,
        })
        entry["bm25_score"] = round(float(bm25_scores[idx]), 2)
        entry["bm25_rank"] = rank

    def rrf(rank):
        return 0.0 if rank is None else 1.0 / (RRF_K + rank + 1)

    relevant = []
    for c in candidates.values():
        c["score"] = rrf(c["vector_rank"]) + rrf(c["bm25_rank"])
        if c["vector_sim"] >= VECTOR_SIM_THRESHOLD or c["bm25_score"] >= BM25_SCORE_THRESHOLD:
            relevant.append(c)

    relevant.sort(key=lambda c: c["score"], reverse=True)
    return relevant[:top_k]


# ----------------------------------------------------------- generation --

def _as_documents(chunks: list[dict]) -> list[dict]:
    """Cohere's native RAG format. `id` is echoed back in citations but
    never shown to the model -- the model can only cite ids we handed it,
    so a hallucinated file name is structurally impossible here (unlike
    asking the model to type "(file.pdf, Page X)" as free text)."""
    return [
        {"id": c["chunk_id"], "text": c["text"],
         "file_name": c["file_name"], "page_number": str(c["page_number"])}
        for c in chunks
    ]


def generate_answer(index: RagIndex, query: str, history: list[dict], top_k: int = TOP_K) -> dict:
    retrieved = retrieve(index, query, top_k=top_k)
    if not retrieved:
        return {
            "answer": "No relevant information was found in the knowledge base for this question.",
            "sources": [], "cited_sources": [], "grounded": False,
        }

    chat_history = [
        {"role": "USER" if turn["role"] == "user" else "CHATBOT", "message": turn["content"]}
        for turn in history[-2 * MAX_HISTORY_TURNS:]
    ]

    try:
        response = index.client.chat(
            model=CHAT_MODEL,
            message=query,
            preamble=SYSTEM_PREAMBLE,
            chat_history=chat_history,
            documents=_as_documents(retrieved),
            temperature=0.1,
        )
    except Exception as exc:
        return {
            "answer": (
                f"Could not reach Cohere ({CHAT_MODEL}). Check that COHERE_API_KEY is "
                f"valid and has quota remaining. Details: {exc}"
            ),
            "sources": retrieved, "cited_sources": [], "grounded": False,
        }

    by_id = {c["chunk_id"]: c for c in retrieved}
    cited_keys: set[tuple[str, int]] = set()
    for citation in (response.citations or []):
        for doc_id in citation.document_ids:
            c = by_id.get(doc_id)
            if c:
                cited_keys.add((c["file_name"], c["page_number"]))

    cited_sources = [{"file_name": f, "page_number": p} for f, p in sorted(cited_keys)]
    return {
        "answer": response.text,
        "sources": retrieved,
        "cited_sources": cited_sources,
        "grounded": bool(cited_sources),
    }
