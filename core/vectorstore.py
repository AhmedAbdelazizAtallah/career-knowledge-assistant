"""
core/vectorstore.py -- persistent vector store on top of ChromaDB.

Chroma's local PersistentClient writes its HNSW index and metadata to
disk under CHROMA_PERSIST_DIR, so embeddings survive process restarts
instead of being recomputed -- and re-billed against the Cohere API --
on every app start, unlike the previous in-memory-only FAISS index.
Callers only see upsert/query/get_all/reset, so this can be swapped for
a remote Qdrant/pgvector-backed store later without touching retrieval
logic.
"""
import os

import chromadb
import numpy as np
from chromadb.config import Settings

from core.ingestion import Chunk

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "career_knowledge_base")

_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _chunk_metadata(c: Chunk) -> dict:
    return {
        "document_id": c.document_id,
        "document_name": c.document_name,
        "document_title": c.document_title,
        "section": c.section,
        "page_number": c.page_number,
        "parent_id": c.parent_id,
        "parent_text": c.parent_text,
        "parent_page_start": c.parent_page_start,
        "parent_page_end": c.parent_page_end,
    }


class VectorStore:
    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self._client.get_or_create_collection(
            name=collection_name, metadata=_COLLECTION_METADATA,
        )

    def __len__(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        name = self.collection.name
        self._client.delete_collection(name)
        self.collection = self._client.get_or_create_collection(
            name=name, metadata=_COLLECTION_METADATA,
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[_chunk_metadata(c) for c in chunks],
        )

    def query(self, query_embedding: np.ndarray, n_results: int) -> list[dict]:
        """Dense nearest-neighbor search. Returns [] rather than raising
        when the collection is empty or the store is unreachable --
        callers treat an empty dense result as "fall back to BM25 alone",
        not as a fatal error (see core/retriever.py)."""
        n_results = min(n_results, len(self))
        if n_results <= 0:
            return []
        try:
            resp = self.collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        results = []
        for chunk_id, text, meta, distance in zip(
            resp["ids"][0], resp["documents"][0], resp["metadatas"][0], resp["distances"][0]
        ):
            results.append({"chunk_id": chunk_id, "text": text, "distance": distance, **meta})
        return results

    def get_all(self) -> list[dict]:
        """Every chunk's text + metadata, for building the in-memory BM25
        index. BM25 has no meaningful persistent store of its own at this
        corpus size, so it's rebuilt from the persisted vector store at
        startup instead -- cheap (pure tokenization, no API calls)."""
        try:
            resp = self.collection.get(include=["documents", "metadatas"])
        except Exception:
            return []
        results = []
        for chunk_id, text, meta in zip(resp["ids"], resp["documents"], resp["metadatas"]):
            results.append({"chunk_id": chunk_id, "text": text, **meta})
        return results
