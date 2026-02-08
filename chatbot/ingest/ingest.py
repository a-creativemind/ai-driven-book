"""Main ingestion script for processing textbook chapters into the RAG system."""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chatbot.api.config import get_settings
from chatbot.api.services.database import DatabaseService
from chatbot.api.services.embedding import EmbeddingService
from chatbot.api.services.vector_store import VectorStoreService
from chatbot.ingest.chunker import Chunk, TextChunker
from chatbot.ingest.markdown_parser import MarkdownParser, ParsedChapter


class IngestionPipeline:
    """Pipeline for ingesting textbook chapters into the RAG system."""

    def __init__(self, docs_path: Path):
        """Initialize the ingestion pipeline.

        Args:
            docs_path: Path to the docs directory containing markdown files
        """
        self.docs_path = docs_path
        self.settings = get_settings()

        # Initialize components
        self.parser = MarkdownParser(docs_path)
        self.chunker = TextChunker(
            min_chunk_tokens=self.settings.chunk_size_min,
            max_chunk_tokens=self.settings.chunk_size_max,
        )
        self.embedding_service = EmbeddingService(self.settings)
        self.vector_store = VectorStoreService(self.settings)
        self.database = DatabaseService(self.settings)

    async def initialize(self) -> None:
        """Initialize database and vector store connections."""
        print("Initializing connections...")

        # Connect to database and create tables
        await self.database.connect()
        await self.database.initialize_tables()
        print("  - PostgreSQL: connected and tables initialized")

        # Initialize vector store collection
        created = await self.vector_store.initialize_collection()
        if created:
            print("  - Qdrant: collection created")
        else:
            print("  - Qdrant: collection already exists")

    async def cleanup(self) -> None:
        """Clean up connections."""
        await self.database.disconnect()

    async def ingest_chapter(
        self,
        chapter: ParsedChapter,
        force_reingest: bool = False,
    ) -> dict:
        """Ingest a single chapter into the RAG system.

        Args:
            chapter: Parsed chapter to ingest
            force_reingest: If True, delete existing data before ingesting

        Returns:
            Status dictionary with chapter_id, status, and chunk count
        """
        print(f"\nProcessing: {chapter.title} ({chapter.chapter_id})")

        try:
            # Check if already ingested
            existing_chunks = await self.database.get_chunk_count(chapter.chapter_id)
            if existing_chunks > 0 and not force_reingest:
                print(f"  - Skipping: already has {existing_chunks} chunks")
                return {
                    "chapter_id": chapter.chapter_id,
                    "status": "skipped",
                    "chunks_created": existing_chunks,
                }

            # Delete existing data if force reingest
            if existing_chunks > 0:
                print(f"  - Deleting {existing_chunks} existing chunks...")
                await self.database.delete_chunks_by_chapter(chapter.chapter_id)
                await self.vector_store.delete_by_chapter(chapter.chapter_id)

            # Upsert chapter metadata
            await self.database.upsert_chapter(
                chapter_id=chapter.chapter_id,
                part_id=chapter.part_id,
                title=chapter.title,
                file_path=chapter.file_path,
                description=chapter.description,
                difficulty=chapter.difficulty,
                keywords=chapter.keywords,
                prerequisites=chapter.prerequisites,
            )
            print(f"  - Chapter metadata saved")

            # Chunk the content
            chunks = self.chunker.chunk_chapter(chapter)
            print(f"  - Created {len(chunks)} chunks")

            if not chunks:
                return {
                    "chapter_id": chapter.chapter_id,
                    "status": "success",
                    "chunks_created": 0,
                }

            # Generate embeddings
            print(f"  - Generating embeddings...")
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(texts)
            print(f"  - Generated {len(embeddings)} embeddings")

            # Store in vector database
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            payloads = [
                {
                    "chapter_id": chunk.chapter_id,
                    "section_title": chunk.section_title,
                    "section_level": chunk.section_level,
                    "difficulty": chunk.difficulty,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                }
                for chunk in chunks
            ]
            await self.vector_store.upsert_chunks(chunk_ids, embeddings, payloads)
            print(f"  - Stored vectors in Qdrant")

            # Store metadata in PostgreSQL
            chunk_metadata = [
                {
                    "chunk_id": chunk.chunk_id,
                    "chapter_id": chunk.chapter_id,
                    "section_title": chunk.section_title,
                    "section_level": chunk.section_level,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "content_preview": chunk.content[:200],
                    "difficulty": chunk.difficulty,
                }
                for chunk in chunks
            ]
            await self.database.insert_chunk_metadata_batch(chunk_metadata)
            print(f"  - Stored metadata in PostgreSQL")

            return {
                "chapter_id": chapter.chapter_id,
                "status": "success",
                "chunks_created": len(chunks),
            }

        except Exception as e:
            print(f"  - Error: {e}")
            return {
                "chapter_id": chapter.chapter_id,
                "status": "error",
                "chunks_created": 0,
                "error_message": str(e),
            }

    async def ingest_all(
        self,
        force_reingest: bool = False,
        chapter_ids: Optional[List[str]] = None,
    ) -> dict:
        """Ingest all chapters (or specific ones) into the RAG system.

        Args:
            force_reingest: If True, delete existing data before ingesting
            chapter_ids: If provided, only ingest these chapters

        Returns:
            Summary of ingestion results
        """
        start_time = time.time()

        # Parse all chapters
        print("Parsing markdown files...")
        chapters = self.parser.parse_all_chapters()
        print(f"Found {len(chapters)} chapters")

        # Filter if specific chapters requested
        if chapter_ids:
            chapters = [c for c in chapters if c.chapter_id in chapter_ids]
            print(f"Filtered to {len(chapters)} chapters")

        # Process each chapter
        results = []
        for chapter in chapters:
            result = await self.ingest_chapter(chapter, force_reingest)
            results.append(result)

        # Calculate summary
        duration = time.time() - start_time
        successful = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "error")
        total_chunks = sum(r["chunks_created"] for r in results)

        summary = {
            "total_chapters": len(chapters),
            "successful": successful,
            "skipped": skipped,
            "failed": failed,
            "total_chunks": total_chunks,
            "duration_seconds": round(duration, 2),
            "chapters": results,
        }

        return summary


async def main():
    """Main entry point for the ingestion script."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest textbook chapters into RAG system")
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=Path(__file__).parent.parent.parent / "docs",
        help="Path to docs directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of all chapters",
    )
    parser.add_argument(
        "--chapters",
        nargs="+",
        help="Specific chapter IDs to ingest",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Physical AI Textbook - RAG Ingestion Pipeline")
    print("=" * 60)

    # Validate docs path
    if not args.docs_path.exists():
        print(f"Error: docs path does not exist: {args.docs_path}")
        sys.exit(1)

    # Run ingestion
    pipeline = IngestionPipeline(args.docs_path)

    try:
        await pipeline.initialize()
        results = await pipeline.ingest_all(
            force_reingest=args.force,
            chapter_ids=args.chapters,
        )
    finally:
        await pipeline.cleanup()

    # Print summary
    print("\n" + "=" * 60)
    print("Ingestion Complete")
    print("=" * 60)
    print(f"Total chapters: {results['total_chapters']}")
    print(f"Successful: {results['successful']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print(f"Total chunks: {results['total_chunks']}")
    print(f"Duration: {results['duration_seconds']} seconds")

    if results["failed"] > 0:
        print("\nFailed chapters:")
        for r in results["chapters"]:
            if r["status"] == "error":
                print(f"  - {r['chapter_id']}: {r.get('error_message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
