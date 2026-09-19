from nrag.ingestion.chunk import chunk_records
from nrag.ingestion.extract import PageRecord


def _record(text: str, page: int = 1) -> PageRecord:
    return PageRecord(
        file_name="sample.pdf",
        file_path="/tmp/sample.pdf",
        page_number=page,
        total_pages=1,
        text=text,
    )


def test_short_text_produces_no_chunks_below_min_length():
    chunks = chunk_records([_record("short")], min_chunk_chars=40)
    assert chunks == []


def test_long_text_is_split_into_multiple_chunks():
    long_text = "Sentence number " + " ".join(f"{i}." for i in range(400))
    chunks = chunk_records([_record(long_text)], chunk_size=200, chunk_overlap=50, min_chunk_chars=10)
    assert len(chunks) > 1
    assert all(c.file_name == "sample.pdf" and c.page_number == 1 for c in chunks)


def test_chunk_id_is_stable_for_identical_content():
    text = "A" * 100
    chunks_a = chunk_records([_record(text)], min_chunk_chars=10)
    chunks_b = chunk_records([_record(text)], min_chunk_chars=10)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_chunk_id_changes_when_content_changes():
    chunks_a = chunk_records([_record("A" * 100)], min_chunk_chars=10)
    chunks_b = chunk_records([_record("B" * 100)], min_chunk_chars=10)
    assert chunks_a[0].chunk_id != chunks_b[0].chunk_id
