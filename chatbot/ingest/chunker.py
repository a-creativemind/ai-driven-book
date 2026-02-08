"""Text chunker for splitting content into optimal-sized chunks for embedding."""

import re
from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID, uuid4

import tiktoken

from .markdown_parser import ParsedChapter, Section


@dataclass
class Chunk:
    """Represents a text chunk ready for embedding."""

    chunk_id: UUID
    chapter_id: str
    section_title: Optional[str]
    section_level: int
    chunk_index: int
    content: str
    token_count: int
    difficulty: str


class TextChunker:
    """Chunker that splits chapter content into embedding-ready chunks."""

    def __init__(
        self,
        min_chunk_tokens: int = 300,
        max_chunk_tokens: int = 800,
        overlap_tokens: int = 50,
        model: str = "text-embedding-3-small",
    ):
        """Initialize the chunker.

        Args:
            min_chunk_tokens: Minimum tokens per chunk
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Token overlap between chunks
            model: Model name for tokenizer selection
        """
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

        # Initialize tokenizer
        try:
            self._tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._tokenizer.encode(text))

    def chunk_chapter(self, chapter: ParsedChapter) -> List[Chunk]:
        """Chunk a parsed chapter into embedding-ready chunks.

        Args:
            chapter: Parsed chapter to chunk

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_index = 0

        for section in chapter.sections:
            section_chunks = self._chunk_section(
                section=section,
                chapter_id=chapter.chapter_id,
                difficulty=chapter.difficulty,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _chunk_section(
        self,
        section: Section,
        chapter_id: str,
        difficulty: str,
        start_index: int,
    ) -> List[Chunk]:
        """Chunk a single section.

        Args:
            section: Section to chunk
            chapter_id: Parent chapter ID
            difficulty: Difficulty level
            start_index: Starting chunk index

        Returns:
            List of Chunk objects for this section
        """
        # Clean the content
        content = self._clean_for_embedding(section.content)

        # Skip empty sections
        if not content.strip():
            return []

        token_count = self.count_tokens(content)

        # If section fits in one chunk, return it
        if token_count <= self.max_chunk_tokens:
            if token_count >= self.min_chunk_tokens:
                return [
                    Chunk(
                        chunk_id=uuid4(),
                        chapter_id=chapter_id,
                        section_title=section.title,
                        section_level=section.level,
                        chunk_index=start_index,
                        content=content,
                        token_count=token_count,
                        difficulty=difficulty,
                    )
                ]
            else:
                # Section is too small, might be combined later
                return [
                    Chunk(
                        chunk_id=uuid4(),
                        chapter_id=chapter_id,
                        section_title=section.title,
                        section_level=section.level,
                        chunk_index=start_index,
                        content=content,
                        token_count=token_count,
                        difficulty=difficulty,
                    )
                ]

        # Section is too large, need to split
        return self._split_large_section(
            content=content,
            section_title=section.title,
            section_level=section.level,
            chapter_id=chapter_id,
            difficulty=difficulty,
            start_index=start_index,
        )

    def _split_large_section(
        self,
        content: str,
        section_title: str,
        section_level: int,
        chapter_id: str,
        difficulty: str,
        start_index: int,
    ) -> List[Chunk]:
        """Split a large section into multiple chunks.

        Uses paragraph boundaries where possible.

        Args:
            content: Section content
            section_title: Section title for metadata
            section_level: Heading level
            chapter_id: Parent chapter ID
            difficulty: Difficulty level
            start_index: Starting chunk index

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_index = start_index

        # Split by paragraphs first
        paragraphs = re.split(r"\n\n+", content)
        current_chunk_parts = []
        current_token_count = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds max, split by sentences
            if para_tokens > self.max_chunk_tokens:
                # First, save current chunk if any
                if current_chunk_parts:
                    chunk_content = "\n\n".join(current_chunk_parts)
                    chunks.append(
                        Chunk(
                            chunk_id=uuid4(),
                            chapter_id=chapter_id,
                            section_title=section_title,
                            section_level=section_level,
                            chunk_index=chunk_index,
                            content=chunk_content,
                            token_count=self.count_tokens(chunk_content),
                            difficulty=difficulty,
                        )
                    )
                    chunk_index += 1
                    current_chunk_parts = []
                    current_token_count = 0

                # Split paragraph by sentences
                sentence_chunks = self._split_by_sentences(para, section_title, section_level, chapter_id, difficulty, chunk_index)
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)
                continue

            # Check if adding this paragraph exceeds max
            if current_token_count + para_tokens > self.max_chunk_tokens:
                # Save current chunk
                if current_chunk_parts:
                    chunk_content = "\n\n".join(current_chunk_parts)
                    chunks.append(
                        Chunk(
                            chunk_id=uuid4(),
                            chapter_id=chapter_id,
                            section_title=section_title,
                            section_level=section_level,
                            chunk_index=chunk_index,
                            content=chunk_content,
                            token_count=self.count_tokens(chunk_content),
                            difficulty=difficulty,
                        )
                    )
                    chunk_index += 1

                # Start new chunk, with overlap if possible
                if self.overlap_tokens > 0 and current_chunk_parts:
                    overlap_text = self._get_overlap_text(current_chunk_parts[-1])
                    current_chunk_parts = [overlap_text] if overlap_text else []
                    current_token_count = self.count_tokens(overlap_text) if overlap_text else 0
                else:
                    current_chunk_parts = []
                    current_token_count = 0

            current_chunk_parts.append(para)
            current_token_count += para_tokens

        # Don't forget the last chunk
        if current_chunk_parts:
            chunk_content = "\n\n".join(current_chunk_parts)
            chunks.append(
                Chunk(
                    chunk_id=uuid4(),
                    chapter_id=chapter_id,
                    section_title=section_title,
                    section_level=section_level,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    token_count=self.count_tokens(chunk_content),
                    difficulty=difficulty,
                )
            )

        return chunks

    def _split_by_sentences(
        self,
        text: str,
        section_title: str,
        section_level: int,
        chapter_id: str,
        difficulty: str,
        start_index: int,
    ) -> List[Chunk]:
        """Split text by sentences for very long paragraphs."""
        # Simple sentence splitting (handles . ! ?)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        chunk_index = start_index
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.max_chunk_tokens and current_sentences:
                # Save current chunk
                chunk_content = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        chunk_id=uuid4(),
                        chapter_id=chapter_id,
                        section_title=section_title,
                        section_level=section_level,
                        chunk_index=chunk_index,
                        content=chunk_content,
                        token_count=self.count_tokens(chunk_content),
                        difficulty=difficulty,
                    )
                )
                chunk_index += 1
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Last chunk
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    chunk_id=uuid4(),
                    chapter_id=chapter_id,
                    section_title=section_title,
                    section_level=section_level,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    token_count=self.count_tokens(chunk_content),
                    difficulty=difficulty,
                )
            )

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get the last N tokens of text for overlap."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.overlap_tokens:
            return text
        overlap_tokens = tokens[-self.overlap_tokens:]
        return self._tokenizer.decode(overlap_tokens)

    def _clean_for_embedding(self, content: str) -> str:
        """Clean content for embedding.

        Args:
            content: Raw markdown content

        Returns:
            Cleaned content
        """
        # Remove JSX components
        cleaned = re.sub(r"<[A-Z][^>]*>.*?</[A-Z][^>]*>", "", content, flags=re.DOTALL)
        cleaned = re.sub(r"<div[^>]*>.*?</div>", "", cleaned, flags=re.DOTALL)

        # Remove HTML comments
        cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)

        # Keep code blocks but remove the fence markers
        cleaned = re.sub(r"```[a-z]*\n", "\n", cleaned)
        cleaned = cleaned.replace("```", "")

        # Remove excessive whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()
