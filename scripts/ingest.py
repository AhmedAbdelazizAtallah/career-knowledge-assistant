"""
scripts/ingest.py -- one-time (or on-demand) ingestion: parse the PDFs in
data/, embed every chunk via Cohere, and persist them into the Chroma
vector store. Run this whenever the PDFs in data/ change; the Streamlit
app itself only reads the persisted store, it doesn't re-ingest on every
restart (see core/retriever.py:build_index).

Usage:
    python scripts/ingest.py            # ingest only if the store is empty
    python scripts/ingest.py --rebuild  # wipe and re-ingest from scratch
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from core.generator import get_cohere_client
from core.retriever import build_index
from core.vectorstore import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe the existing collection and re-embed everything, even if it's already populated.",
    )
    args = parser.parse_args()

    client = get_cohere_client()
    store = VectorStore()
    before = len(store)

    start = time.perf_counter()
    index = build_index(DATA_DIR, client, store=store, force_reingest=args.rebuild)
    elapsed = time.perf_counter() - start

    if before and not args.rebuild:
        print(f"Store already had {before} chunks -- skipped re-ingestion (pass --rebuild to force).")
    else:
        print(f"Ingested {len(index)} chunks from {DATA_DIR} into '{store.collection.name}' "
              f"at {store.persist_dir} in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
