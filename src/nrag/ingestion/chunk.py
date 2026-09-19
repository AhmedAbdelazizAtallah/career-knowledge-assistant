import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from nrag.ingestion.extract import PageRecord


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_name: str
    page_number: int
    text: str
    content_hash: str


def _content_hash(file_name: str, page_number: int, text: str) -> str:
    payload = f"{file_name}|{page_number}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def chunk_records(
    records: list[PageRecord],
    chunk_size: int = 700,
    chunk_overlap: int = 180,
    min_chunk_chars: int = 40,
) -> list[Chunk]:
    """Split cleaned page records into overlapping chunks.

    chunk_id is a content hash (not a running counter) so re-ingesting
    unchanged source files produces identical IDs -- that's what lets the
    vector store skip work instead of re-embedding everything every run.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[Chunk] = []
    for record in records:
        for split in splitter.split_text(record.text):
            stripped = split.strip()
            if len(stripped) <= min_chunk_chars:
                continue
            content_hash = _content_hash(record.file_name, record.page_number, stripped)
            chunks.append(
                Chunk(
                    chunk_id=f"{record.file_name}_p{record.page_number}_{content_hash}",
                    file_name=record.file_name,
                    page_number=record.page_number,
                    text=stripped,
                    content_hash=content_hash,
                )
            )
    return chunks
