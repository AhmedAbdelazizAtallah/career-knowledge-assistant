# NRAG — Career Knowledge Assistant

A local RAG (Retrieval-Augmented Generation) service that answers career questions
(resumes, interviews, negotiation, career growth) grounded strictly in a set of
PDF knowledge-base documents. Runs fully locally: Chroma for vector storage,
`sentence-transformers` for embeddings, and Ollama for generation — no external
API keys required.

## Architecture

```
data/                    Source PDFs (the knowledge base)
src/nrag/
  config.py              Env-driven settings (nrag.config.settings)
  ingestion/              extract -> clean -> chunk -> pipeline (incremental indexing)
  vectorstore/            Chroma wrapper (content-hash IDs, upsert-only re-ingestion)
  retrieval/               Top-k semantic retriever
  generation/              Ollama client, prompt building, citation verification
  rag_pipeline.py          Orchestrates retrieve -> generate -> verify citations
  factory.py                Builds the wired-up pipeline for API/CLI use
api/                      FastAPI app (/health, /ingest, /query)
ui/                       Streamlit chat UI, talks to the API over HTTP
scripts/ingest.py         CLI ingestion, no API server needed
tests/                    Unit tests for cleaning, chunking, citation checking
notebooks/Ollama.ipynb    Original exploratory prototype (kept for reference)
```

Ingestion is **incremental**: each chunk's ID is a content hash, so re-running
ingestion after adding one new PDF only embeds the new chunks instead of
wiping and rebuilding the whole collection.

Every generated answer is checked against the actually-retrieved sources — if
the model cites a file/page that wasn't retrieved (or falls back to a
"Document [1]"-style placeholder), that's surfaced back to the caller instead
of silently trusting the model's citation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e . --no-deps    # makes `nrag` importable everywhere (api/, scripts/, tests/)
```

Install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull llama3.2:3b
```

Copy `.env.example` to `.env` if you want to override any default (paths,
model name, chunk size, etc.) — everything in `src/nrag/config.py` is
overridable via `NRAG_*` environment variables.

## Running

1. Index the PDFs in `data/`:

   ```bash
   python scripts/ingest.py
   ```

2. Start the API:

   ```bash
   uvicorn api.main:app --reload
   ```

3. Start the chat UI (separate terminal):

   ```bash
   streamlit run ui/app.py
   ```

Or run everything (Ollama + API + UI) with Docker:

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2:3b
```

## API

- `GET /health` — status + number of indexed chunks
- `POST /ingest` — re-scan `data/` and index any new/changed PDFs
- `POST /query` — `{"query": "...", "top_k": 4}` → answer + sources + citation warnings

## Tests

```bash
pytest
```

Covers text cleaning, chunking (including content-hash stability for
incremental ingestion), and citation verification. It does not cover
retrieval/generation end-to-end since those require the embedding model and
a running Ollama server — see "Next steps" below.

## Known limitations / next steps

- No retrieval-quality evaluation harness (e.g. a golden Q&A set with
  precision/recall or RAGAS scoring) — currently unverified beyond unit tests.
- No conversation memory — every query is stateless/single-turn.
- No auth/rate-limiting on the API — fine for local/internal use, not for a
  public deployment.
- No CI pipeline wired up yet (tests exist, but nothing runs them
  automatically on push).
