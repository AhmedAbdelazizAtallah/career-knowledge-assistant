from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceChunk(BaseModel):
    file_name: str
    page_number: int
    similarity: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    unverifiable_citations: list[str]


class IngestResponse(BaseModel):
    new_chunks_indexed: int
    total_chunks_in_collection: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
