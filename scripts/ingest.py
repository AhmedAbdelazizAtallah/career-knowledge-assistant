"""Run ingestion from the command line, without starting the API.

Usage:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrag.config import settings  # noqa: E402
from nrag.factory import build_store  # noqa: E402
from nrag.ingestion.pipeline import run_ingestion  # noqa: E402
from nrag.logging_config import configure_logging  # noqa: E402


def main() -> None:
    configure_logging()
    store = build_store(settings)
    run_ingestion(
        data_dir=settings.data_dir,
        store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
        min_page_chars=settings.min_page_chars,
    )


if __name__ == "__main__":
    main()
