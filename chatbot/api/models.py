"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Chat Models
# ============================================================================


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    history: List[ChatMessage] = Field(default_factory=list, description="Conversation history")


class SelectionChatRequest(BaseModel):
    """Request body for selection-based chat endpoint."""

    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    selected_text: str = Field(..., min_length=1, max_length=5000, description="Selected text from the page")
    chapter_id: Optional[str] = Field(None, description="Chapter ID where text was selected")
    history: List[ChatMessage] = Field(default_factory=list, description="Conversation history")


class SourceReference(BaseModel):
    """A source reference from the RAG retrieval."""

    chunk_id: UUID
    chapter_id: str
    chapter_title: str
    section_title: Optional[str] = None
    content_preview: str = Field(..., max_length=300)
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    difficulty: str = Field(default="intermediate")


class ChatResponse(BaseModel):
    """Response body for chat endpoints."""

    answer: str = Field(..., description="Generated answer")
    sources: List[SourceReference] = Field(default_factory=list, description="Source citations")
    tokens_used: Optional[int] = Field(None, description="Total tokens used")


# ============================================================================
# Chapter Models
# ============================================================================


class ChapterMetadata(BaseModel):
    """Metadata for a single chapter."""

    chapter_id: str
    part_id: str
    title: str
    description: Optional[str] = None
    difficulty: str = Field(default="intermediate")
    keywords: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    file_path: str


class ChapterListResponse(BaseModel):
    """Response body for chapter list endpoint."""

    chapters: List[ChapterMetadata]
    total_count: int


class ChapterDetailResponse(BaseModel):
    """Response body for chapter detail endpoint."""

    chapter: ChapterMetadata
    chunk_count: int
    sections: List[str] = Field(default_factory=list)


# ============================================================================
# Ingestion Models
# ============================================================================


class IngestRequest(BaseModel):
    """Request body for ingestion endpoint."""

    force_reingest: bool = Field(default=False, description="Force re-ingestion of all chapters")
    chapter_ids: Optional[List[str]] = Field(None, description="Specific chapters to ingest (None = all)")


class IngestStatus(BaseModel):
    """Status of a single chapter ingestion."""

    chapter_id: str
    status: str = Field(..., description="'success', 'skipped', or 'error'")
    chunks_created: int = 0
    error_message: Optional[str] = None


class IngestResponse(BaseModel):
    """Response body for ingestion endpoint."""

    total_chapters: int
    successful: int
    failed: int
    skipped: int
    chapters: List[IngestStatus]
    duration_seconds: float


# ============================================================================
# Chunk Models (Internal)
# ============================================================================


class ChunkData(BaseModel):
    """Data structure for a text chunk."""

    chunk_id: UUID
    chapter_id: str
    section_title: Optional[str] = None
    section_level: int = 2
    chunk_index: int
    content: str
    token_count: int
    difficulty: str = "intermediate"
    embedding: Optional[List[float]] = None


class ChunkMetadataDB(BaseModel):
    """Chunk metadata stored in PostgreSQL."""

    chunk_id: UUID
    chapter_id: str
    section_title: Optional[str] = None
    section_level: int = 2
    chunk_index: int
    token_count: int
    content_preview: str = Field(..., max_length=200)
    difficulty: str = "intermediate"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Health Check
# ============================================================================


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = "healthy"
    version: str = "1.0.0"
    qdrant_connected: bool = False
    postgres_connected: bool = False
