"""Regression tests for core/vectorstore.py -- pure local Chroma
round-trip, no Cohere API calls (fake embedding vectors)."""
import numpy as np
import pytest

from core.ingestion import Chunk
from core.vectorstore import VectorStore


def _chunk(chunk_id: str, text: str, section: str = "1. Intro") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc1", document_name="doc1.pdf",
        document_title="Doc One", section=section, page_number=1, text=text,
        parent_id="doc1::1. Intro", parent_text=text,
        parent_page_start=1, parent_page_end=1,
    )


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test_collection")


def test_upsert_and_query_round_trip(store):
    chunks = [_chunk("a", "apples and oranges"), _chunk("b", "cars and trucks")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store.upsert(chunks, embeddings)

    assert len(store) == 2
    results = store.query(np.array([1.0, 0.0], dtype="float32"), n_results=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"
    assert results[0]["document_name"] == "doc1.pdf"
    assert results[0]["parent_page_start"] == 1


def test_get_all_returns_every_chunk_with_metadata(store):
    chunks = [_chunk("a", "text a"), _chunk("b", "text b")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store.upsert(chunks, embeddings)

    rows = store.get_all()
    assert {r["chunk_id"] for r in rows} == {"a", "b"}
    assert all(r["section"] == "1. Intro" for r in rows)


def test_reset_clears_the_collection(store):
    store.upsert([_chunk("a", "text a")], np.array([[1.0, 0.0]], dtype="float32"))
    assert len(store) == 1
    store.reset()
    assert len(store) == 0


def test_query_on_empty_store_returns_empty_list(store):
    assert store.query(np.array([1.0, 0.0], dtype="float32"), n_results=5) == []


def test_upsert_empty_list_is_a_noop(store):
    store.upsert([], np.zeros((0, 2), dtype="float32"))
    assert len(store) == 0
