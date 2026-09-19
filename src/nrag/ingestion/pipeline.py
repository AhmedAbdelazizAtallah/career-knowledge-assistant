import logging
from pathlib import Path

from nrag.ingestion.chunk import chunk_records
from nrag.ingestion.clean import preprocess_records
from nrag.ingestion.extract import extract_text_from_pdfs
from nrag.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def run_ingestion(
    data_dir: Path,
    store: ChromaStore,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    min_page_chars: int,
) -> int:
    """Extract -> clean -> chunk -> incrementally index every PDF in `data_dir`.

    Returns the number of newly indexed chunks (0 means everything was
    already up to date).
    """
    raw_records = extract_text_from_pdfs(data_dir)
    logger.info("Extracted %d raw page(s) from %s", len(raw_records), data_dir)

    cleaned_records = preprocess_records(raw_records, min_page_chars=min_page_chars)
    logger.info("%d page(s) survived cleaning", len(cleaned_records))

    chunks = chunk_records(
        cleaned_records,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
    logger.info("Produced %d chunk(s)", len(chunks))

    added = store.upsert_chunks(chunks)
    logger.info("Ingestion complete. %d new chunk(s) indexed, %d total in collection.", added, store.count())
    return added
