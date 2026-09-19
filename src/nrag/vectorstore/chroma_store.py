import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from nrag.ingestion.chunk import Chunk

logger = logging.getLogger(__name__)


class ChromaStore:
    """Thin wrapper around a persistent Chroma collection.

    Ingestion is incremental: chunk IDs are content hashes (see
    `nrag.ingestion.chunk`), so re-running ingest on unchanged source PDFs
    is a cheap no-op instead of re-embedding everything from scratch.
    """

    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Add only chunks whose ID isn't already indexed. Returns count added."""
        if not chunks:
            return 0

        existing_ids: set[str] = set()
        all_ids = [c.chunk_id for c in chunks]
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            existing = self._collection.get(ids=batch_ids, include=[])
            existing_ids.update(existing["ids"])

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            logger.info("No new chunks to index; collection already up to date.")
            return 0

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {"file_name": c.file_name, "page_number": c.page_number} for c in batch
                ],
            )

        logger.info("Indexed %d new chunk(s).", len(new_chunks))
        return len(new_chunks)

    def query(self, query_text: str, n_results: int) -> dict:
        return self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self._collection.count()
