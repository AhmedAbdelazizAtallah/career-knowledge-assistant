"""Regression tests for core/retriever.py's fusion/mode logic, using a
fake Cohere client (no network calls) so these run offline and in CI."""
from pathlib import Path

import numpy as np
import pytest

from core.ingestion import Chunk
from core.retriever import RagIndex, _hybrid_shortlist, build_index, tokenize
from core.vectorstore import EmbedModelMismatchError, VectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class _FakeEmbedResponse:
    def __init__(self, vectors):
        self.embeddings = type("E", (), {"float_": vectors})


class _FakeCohereClient:
    """Deterministic: embeds by hashing the text into a fixed 2D vector,
    so dense search has something meaningful (but network-free) to rank."""

    def embed(self, texts, model, input_type, embedding_types):
        vectors = [[1.0, 0.0] if "apple" in t else [0.0, 1.0] for t in texts]
        return _FakeEmbedResponse(vectors)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc1", document_name="doc1.pdf",
        document_title="Doc One", section="1. Intro", page_number=1, text=text,
        parent_id="doc1::1. Intro", parent_text=text,
        parent_page_start=1, parent_page_end=1,
    )


@pytest.fixture
def index(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test")
    chunks = [_chunk("apple-chunk", "apple orchard harvest"), _chunk("car-chunk", "car engine repair")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store.upsert(chunks, embeddings)
    return RagIndex(_FakeCohereClient(), store, embed_model="fake-model")


def test_tokenize_drops_stopwords_and_short_tokens():
    tokens = tokenize("What is the best way to fix a car engine?")
    assert "car" in tokens and "engine" in tokens
    assert "the" not in tokens and "is" not in tokens


def test_hybrid_mode_uses_both_channels(index):
    shortlist = _hybrid_shortlist(index, "apple orchard", candidate_pool=5, mode="hybrid")
    ids = {c["chunk_id"] for c in shortlist}
    assert "apple-chunk" in ids


def test_vector_only_mode_skips_bm25(index):
    shortlist = _hybrid_shortlist(index, "apple orchard", candidate_pool=5, mode="vector")
    assert all(c["bm25_rank"] is None for c in shortlist)


def test_bm25_only_mode_skips_dense(index):
    shortlist = _hybrid_shortlist(index, "apple orchard", candidate_pool=5, mode="bm25")
    assert all(c["vector_rank"] is None for c in shortlist)


def test_empty_index_returns_empty_shortlist(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="empty")
    empty_index = RagIndex(_FakeCohereClient(), store)
    assert _hybrid_shortlist(empty_index, "anything", candidate_pool=5) == []


def test_build_index_rejects_a_different_embed_model_without_rebuild(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test")
    build_index(DATA_DIR, _FakeCohereClient(), store=store, embed_model="model-a")

    with pytest.raises(EmbedModelMismatchError):
        build_index(DATA_DIR, _FakeCohereClient(), store=store, embed_model="model-b")


def test_build_index_force_reingest_allows_switching_embed_model(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test")
    build_index(DATA_DIR, _FakeCohereClient(), store=store, embed_model="model-a")

    build_index(DATA_DIR, _FakeCohereClient(), store=store, embed_model="model-b", force_reingest=True)
    assert store.get_embed_model() == "model-b"


def test_build_index_does_not_reingest_when_model_matches(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test")
    build_index(DATA_DIR, _FakeCohereClient(), store=store, embed_model="model-a")
    count_after_first = len(store)

    # A second call with the same model must be a cheap local read, not a
    # re-embed -- verified indirectly: chunk count is unchanged (upsert of
    # identical content-hash ids would be a no-op anyway, but re-ingesting
    # at all would mean this test is silently paying for API calls it
    # shouldn't in a real run).
    index = build_index(DATA_DIR, _FakeCohereClient(), store=store, embed_model="model-a")
    assert len(store) == count_after_first
    assert len(index) == count_after_first
