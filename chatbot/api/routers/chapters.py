"""Chapter API endpoints for metadata retrieval."""

import asyncio
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..models import (
    ChapterDetailResponse,
    ChapterListResponse,
    ChapterMetadata,
    IngestRequest,
    IngestResponse,
    IngestStatus,
)
from ..services.database import DatabaseService
from ..dependencies import get_database_service

router = APIRouter(prefix="/api/chapters", tags=["chapters"])


@router.get("", response_model=ChapterListResponse)
async def list_chapters(
    part_id: Optional[str] = Query(None, description="Filter by part ID"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
    database: DatabaseService = Depends(get_database_service),
) -> ChapterListResponse:
    """List all chapters with optional filtering.

    Args:
        part_id: Optional filter by part ID
        difficulty: Optional filter by difficulty level
        database: Injected database service

    Returns:
        ChapterListResponse with list of chapters and count
    """
    try:
        chapters = await database.get_all_chapters()

        # Apply filters
        if part_id:
            chapters = [c for c in chapters if c["part_id"] == part_id]
        if difficulty:
            chapters = [c for c in chapters if c["difficulty"] == difficulty]

        # Convert to response model
        chapter_list = [
            ChapterMetadata(
                chapter_id=c["chapter_id"],
                part_id=c["part_id"],
                title=c["title"],
                description=c.get("description"),
                difficulty=c.get("difficulty", "intermediate"),
                keywords=c.get("keywords", []),
                prerequisites=c.get("prerequisites", []),
                file_path=c["file_path"],
            )
            for c in chapters
        ]

        return ChapterListResponse(
            chapters=chapter_list,
            total_count=len(chapter_list),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching chapters: {str(e)}",
        )


@router.get("/{chapter_id}", response_model=ChapterDetailResponse)
async def get_chapter(
    chapter_id: str,
    database: DatabaseService = Depends(get_database_service),
) -> ChapterDetailResponse:
    """Get detailed information about a specific chapter.

    Args:
        chapter_id: Chapter identifier
        database: Injected database service

    Returns:
        ChapterDetailResponse with chapter metadata and chunk info
    """
    try:
        chapter = await database.get_chapter(chapter_id)

        if not chapter:
            raise HTTPException(
                status_code=404,
                detail=f"Chapter not found: {chapter_id}",
            )

        # Get additional info
        chunk_count = await database.get_chunk_count(chapter_id)
        sections = await database.get_chapter_sections(chapter_id)

        return ChapterDetailResponse(
            chapter=ChapterMetadata(
                chapter_id=chapter["chapter_id"],
                part_id=chapter["part_id"],
                title=chapter["title"],
                description=chapter.get("description"),
                difficulty=chapter.get("difficulty", "intermediate"),
                keywords=chapter.get("keywords", []),
                prerequisites=chapter.get("prerequisites", []),
                file_path=chapter["file_path"],
            ),
            chunk_count=chunk_count,
            sections=sections,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching chapter: {str(e)}",
        )


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(
    request: IngestRequest,
) -> IngestResponse:
    """Trigger ingestion of textbook chapters.

    This endpoint initiates the ingestion pipeline to process
    markdown files and store them in the vector database.

    Args:
        request: Ingestion request with options

    Returns:
        IngestResponse with ingestion results
    """
    try:
        # Import here to avoid circular imports
        from ...ingest.ingest import IngestionPipeline

        # Determine docs path (relative to this file)
        docs_path = Path(__file__).parent.parent.parent.parent / "docs"

        if not docs_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Docs path not found: {docs_path}",
            )

        # Run ingestion
        pipeline = IngestionPipeline(docs_path)

        await pipeline.initialize()
        try:
            results = await pipeline.ingest_all(
                force_reingest=request.force_reingest,
                chapter_ids=request.chapter_ids,
            )
        finally:
            await pipeline.cleanup()

        # Convert results to response
        chapter_statuses = [
            IngestStatus(
                chapter_id=r["chapter_id"],
                status=r["status"],
                chunks_created=r["chunks_created"],
                error_message=r.get("error_message"),
            )
            for r in results["chapters"]
        ]

        return IngestResponse(
            total_chapters=results["total_chapters"],
            successful=results["successful"],
            failed=results["failed"],
            skipped=results["skipped"],
            chapters=chapter_statuses,
            duration_seconds=results["duration_seconds"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during ingestion: {str(e)}",
        )
