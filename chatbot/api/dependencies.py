"""FastAPI dependency injection for services."""

from functools import lru_cache
from typing import AsyncGenerator

from .config import Settings, get_settings
from .services.database import DatabaseService
from .services.embedding import EmbeddingService
from .services.rag import RAGService
from .services.vector_store import VectorStoreService


# Singleton instances for services
_database_service: DatabaseService | None = None
_embedding_service: EmbeddingService | None = None
_vector_store_service: VectorStoreService | None = None
_rag_service: RAGService | None = None


async def get_database_service() -> DatabaseService:
    """Get the database service instance."""
    global _database_service
    if _database_service is None:
        _database_service = DatabaseService(get_settings())
        await _database_service.connect()
    return _database_service


async def get_embedding_service() -> EmbeddingService:
    """Get the embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(get_settings())
    return _embedding_service


async def get_vector_store_service() -> VectorStoreService:
    """Get the vector store service instance."""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService(get_settings())
    return _vector_store_service


async def get_rag_service() -> RAGService:
    """Get the RAG service instance."""
    global _rag_service
    if _rag_service is None:
        database = await get_database_service()
        embedding = await get_embedding_service()
        vector_store = await get_vector_store_service()
        _rag_service = RAGService(
            embedding_service=embedding,
            vector_store=vector_store,
            database=database,
            settings=get_settings(),
        )
    return _rag_service


async def cleanup_services() -> None:
    """Clean up all service connections."""
    global _database_service, _rag_service

    if _database_service is not None:
        await _database_service.disconnect()
        _database_service = None

    _rag_service = None
