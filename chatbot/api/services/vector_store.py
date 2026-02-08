"""Qdrant vector store service for similarity search."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from ..config import Settings, get_settings


class VectorStoreService:
    """Service for managing vectors in Qdrant Cloud."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the vector store service.

        Args:
            settings: Application settings. If None, loads from environment.
        """
        self.settings = settings or get_settings()
        self.client = AsyncQdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
        self.collection_name = self.settings.qdrant_collection_name
        self.vector_size = self.settings.embedding_dimensions

    async def initialize_collection(self) -> bool:
        """Create the collection if it doesn't exist.

        Returns:
            True if collection was created, False if it already existed.
        """
        collections = await self.client.get_collections()
        existing_names = [c.name for c in collections.collections]

        if self.collection_name in existing_names:
            return False

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self.vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
            # Optimized indexing settings for cloud tier
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=1000,
            ),
        )
        return True

    async def upsert_chunks(
        self,
        chunk_ids: List[UUID],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> int:
        """Insert or update chunks in the vector store.

        Args:
            chunk_ids: List of unique chunk identifiers
            embeddings: List of embedding vectors
            payloads: List of metadata payloads

        Returns:
            Number of points upserted
        """
        points = [
            qdrant_models.PointStruct(
                id=str(chunk_id),
                vector=embedding,
                payload=payload,
            )
            for chunk_id, embedding, payload in zip(chunk_ids, embeddings, payloads)
        ]

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    async def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter for metadata fields

        Returns:
            List of search results with scores and payloads
        """
        # Build filter if conditions provided
        search_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=value),
                        )
                    )
            search_filter = qdrant_models.Filter(must=must_conditions)

        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter,
        )

        return [
            {
                "chunk_id": UUID(result.id),
                "score": result.score,
                **result.payload,
            }
            for result in results
        ]

    async def delete_by_chapter(self, chapter_id: str) -> int:
        """Delete all chunks for a specific chapter.

        Args:
            chapter_id: Chapter identifier

        Returns:
            Number of points deleted (estimated)
        """
        # First count existing points
        count_result = await self.client.count(
            collection_name=self.collection_name,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="chapter_id",
                        match=qdrant_models.MatchValue(value=chapter_id),
                    )
                ]
            ),
        )

        # Delete the points
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="chapter_id",
                            match=qdrant_models.MatchValue(value=chapter_id),
                        )
                    ]
                )
            ),
        )

        return count_result.count

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.

        Returns:
            Collection info including point count and configuration
        """
        try:
            info = await self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception:
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "points_count": 0,
                "status": "not_found",
            }

    async def health_check(self) -> bool:
        """Check if the vector store is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False
