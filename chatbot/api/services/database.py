"""PostgreSQL database service for metadata storage."""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg

from ..config import Settings, get_settings

# SQL for creating tables
CREATE_TABLES_SQL = """
-- Chapters table
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id VARCHAR(100) PRIMARY KEY,
    part_id VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    difficulty VARCHAR(50) DEFAULT 'intermediate',
    keywords TEXT[] DEFAULT '{}',
    prerequisites TEXT[] DEFAULT '{}',
    file_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunk metadata table
CREATE TABLE IF NOT EXISTS chunk_metadata (
    chunk_id UUID PRIMARY KEY,
    chapter_id VARCHAR(100) NOT NULL REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    section_title VARCHAR(255),
    section_level INTEGER DEFAULT 2,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    content_preview VARCHAR(200),
    difficulty VARCHAR(50) DEFAULT 'intermediate',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster chapter lookups
CREATE INDEX IF NOT EXISTS idx_chunks_chapter_id ON chunk_metadata(chapter_id);
CREATE INDEX IF NOT EXISTS idx_chunks_difficulty ON chunk_metadata(difficulty);
"""


class DatabaseService:
    """Service for managing PostgreSQL database operations."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the database service.

        Args:
            settings: Application settings. If None, loads from environment.
        """
        self.settings = settings or get_settings()
        self.database_url = self.settings.database_url
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if self._pool is None:
            await self.connect()
        async with self._pool.acquire() as conn:
            yield conn

    async def initialize_tables(self) -> None:
        """Create database tables if they don't exist."""
        async with self.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)

    # =========================================================================
    # Chapter Operations
    # =========================================================================

    async def upsert_chapter(
        self,
        chapter_id: str,
        part_id: str,
        title: str,
        file_path: str,
        description: Optional[str] = None,
        difficulty: str = "intermediate",
        keywords: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
    ) -> None:
        """Insert or update a chapter record."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chapters (
                    chapter_id, part_id, title, description, difficulty,
                    keywords, prerequisites, file_path, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (chapter_id) DO UPDATE SET
                    part_id = EXCLUDED.part_id,
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    difficulty = EXCLUDED.difficulty,
                    keywords = EXCLUDED.keywords,
                    prerequisites = EXCLUDED.prerequisites,
                    file_path = EXCLUDED.file_path,
                    updated_at = EXCLUDED.updated_at
                """,
                chapter_id,
                part_id,
                title,
                description,
                difficulty,
                keywords or [],
                prerequisites or [],
                file_path,
                datetime.utcnow(),
            )

    async def get_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        """Get a chapter by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM chapters WHERE chapter_id = $1",
                chapter_id,
            )
            return dict(row) if row else None

    async def get_all_chapters(self) -> List[Dict[str, Any]]:
        """Get all chapters ordered by part and position."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM chapters
                ORDER BY part_id, chapter_id
                """
            )
            return [dict(row) for row in rows]

    async def delete_chapter(self, chapter_id: str) -> bool:
        """Delete a chapter and its chunks."""
        async with self.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chapters WHERE chapter_id = $1",
                chapter_id,
            )
            return "DELETE 1" in result

    # =========================================================================
    # Chunk Metadata Operations
    # =========================================================================

    async def insert_chunk_metadata(
        self,
        chunk_id: UUID,
        chapter_id: str,
        section_title: Optional[str],
        section_level: int,
        chunk_index: int,
        token_count: int,
        content_preview: str,
        difficulty: str = "intermediate",
    ) -> None:
        """Insert chunk metadata."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chunk_metadata (
                    chunk_id, chapter_id, section_title, section_level,
                    chunk_index, token_count, content_preview, difficulty
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                chunk_id,
                chapter_id,
                section_title,
                section_level,
                chunk_index,
                token_count,
                content_preview[:200] if content_preview else "",
                difficulty,
            )

    async def insert_chunk_metadata_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """Insert multiple chunk metadata records."""
        if not chunks:
            return 0

        async with self.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO chunk_metadata (
                    chunk_id, chapter_id, section_title, section_level,
                    chunk_index, token_count, content_preview, difficulty
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                [
                    (
                        chunk["chunk_id"],
                        chunk["chapter_id"],
                        chunk.get("section_title"),
                        chunk.get("section_level", 2),
                        chunk["chunk_index"],
                        chunk["token_count"],
                        chunk.get("content_preview", "")[:200],
                        chunk.get("difficulty", "intermediate"),
                    )
                    for chunk in chunks
                ],
            )
            return len(chunks)

    async def get_chunks_by_chapter(self, chapter_id: str) -> List[Dict[str, Any]]:
        """Get all chunk metadata for a chapter."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM chunk_metadata
                WHERE chapter_id = $1
                ORDER BY chunk_index
                """,
                chapter_id,
            )
            return [dict(row) for row in rows]

    async def get_chunk_count(self, chapter_id: Optional[str] = None) -> int:
        """Get count of chunks, optionally filtered by chapter."""
        async with self.acquire() as conn:
            if chapter_id:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM chunk_metadata WHERE chapter_id = $1",
                    chapter_id,
                )
            else:
                result = await conn.fetchval("SELECT COUNT(*) FROM chunk_metadata")
            return result or 0

    async def delete_chunks_by_chapter(self, chapter_id: str) -> int:
        """Delete all chunk metadata for a chapter."""
        async with self.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chunk_metadata WHERE chapter_id = $1",
                chapter_id,
            )
            # Extract count from result like "DELETE 5"
            count = int(result.split()[-1]) if result else 0
            return count

    async def get_chapter_sections(self, chapter_id: str) -> List[str]:
        """Get unique section titles for a chapter."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT section_title
                FROM chunk_metadata
                WHERE chapter_id = $1 AND section_title IS NOT NULL
                ORDER BY section_title
                """,
                chapter_id,
            )
            return [row["section_title"] for row in rows]

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if the database is accessible."""
        try:
            async with self.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
