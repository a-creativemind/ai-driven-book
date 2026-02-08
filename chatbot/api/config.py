"""Configuration settings using Pydantic settings management."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Provider Configuration
    llm_provider: str = "gemini"  # Options: "openai", "gemini"
    
    # Google Gemini Configuration
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"

    # OpenAI Configuration (legacy/fallback)
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 768  # Gemini text-embedding-004 uses 768

    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "textbook_chunks"

    # PostgreSQL (Neon) Configuration
    database_url: str

    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # RAG Configuration
    chunk_size_min: int = 300
    chunk_size_max: int = 800
    top_k_results: int = 5
    similarity_threshold: float = 0.7

    # Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
