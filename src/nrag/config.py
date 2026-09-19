"""Centralized, environment-driven configuration for the whole app.

Nothing outside this module should hardcode a path, model name, or tuning
constant -- import `settings` instead.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NRAG_", extra="ignore")

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"

    # Chunking
    chunk_size: int = 700
    chunk_overlap: int = 180
    min_chunk_chars: int = 40
    min_page_chars: int = 30

    # Embeddings / vector store
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "career_knowledge_base"

    # Retrieval
    default_top_k: int = 4

    # Generation (local Ollama)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_temperature: float = 0.1
    ollama_timeout_seconds: float = 60.0

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:8501"]


settings = Settings()
