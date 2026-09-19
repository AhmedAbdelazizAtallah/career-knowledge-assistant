import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from nrag.config import settings
from nrag.factory import build_pipeline, build_store
from nrag.generation.ollama_client import OllamaGenerationError
from nrag.ingestion.pipeline import run_ingestion
from nrag.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="NRAG Career Knowledge Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once at process startup and reused across requests -- loading the
# embedding model and opening the Chroma collection per-request would be
# far too slow.
_store = build_store(settings)
_pipeline = build_pipeline(_store, settings)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", chunks_indexed=_store.count())


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    added = run_ingestion(
        data_dir=settings.data_dir,
        store=_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
        min_page_chars=settings.min_page_chars,
    )
    return IngestResponse(new_chunks_indexed=added, total_chunks_in_collection=_store.count())


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = _pipeline.answer(request.query, top_k=request.top_k)
    except OllamaGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(
                file_name=s.file_name,
                page_number=s.page_number,
                similarity=s.similarity,
                text=s.text,
            )
            for s in result.sources
        ],
        unverifiable_citations=result.unverifiable_citations,
    )
