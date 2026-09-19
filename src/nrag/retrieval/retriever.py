from dataclasses import dataclass

from nrag.vectorstore.chroma_store import ChromaStore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    file_name: str
    page_number: int
    similarity: float


class Retriever:
    def __init__(self, store: ChromaStore) -> None:
        self._store = store

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        results = self._store.query(query, n_results=top_k)

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        return [
            RetrievedChunk(
                text=doc,
                file_name=meta["file_name"],
                page_number=meta["page_number"],
                similarity=round((1 - dist) * 100, 2),
            )
            for doc, meta, dist in zip(docs, metas, distances)
        ]
