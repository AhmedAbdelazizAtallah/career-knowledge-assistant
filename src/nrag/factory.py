"""Wires up the concrete objects the API and CLI scripts both need.

Single place that knows how to construct a ChromaStore/Retriever/OllamaClient
from `settings`, so the API layer and CLI scripts don't duplicate wiring.
"""
from nrag.config import Settings, settings
from nrag.generation.ollama_client import OllamaClient
from nrag.rag_pipeline import RagPipeline
from nrag.retrieval.retriever import Retriever
from nrag.vectorstore.chroma_store import ChromaStore


def build_store(cfg: Settings = settings) -> ChromaStore:
    return ChromaStore(
        chroma_dir=cfg.chroma_dir,
        collection_name=cfg.collection_name,
        embedding_model=cfg.embedding_model,
    )


def build_pipeline(store: ChromaStore, cfg: Settings = settings) -> RagPipeline:
    retriever = Retriever(store)
    generator = OllamaClient(
        host=cfg.ollama_host,
        model=cfg.ollama_model,
        temperature=cfg.ollama_temperature,
        timeout_seconds=cfg.ollama_timeout_seconds,
    )
    return RagPipeline(retriever, generator)
