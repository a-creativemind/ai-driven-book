"""Markdown parser for extracting content and metadata from chapter files."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import frontmatter


@dataclass
class Section:
    """Represents a section within a chapter."""

    title: str
    level: int  # Heading level (1-6)
    content: str
    start_line: int
    end_line: int


@dataclass
class ParsedChapter:
    """Represents a fully parsed chapter."""

    # Metadata from frontmatter
    chapter_id: str
    part_id: str
    title: str
    description: Optional[str] = None
    difficulty: str = "intermediate"
    keywords: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    file_path: str = ""

    # Content
    raw_content: str = ""
    sections: List[Section] = field(default_factory=list)

    # Additional metadata
    estimated_time: Optional[str] = None
    author: Optional[str] = None
    last_updated: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class MarkdownParser:
    """Parser for extracting structured content from markdown chapter files."""

    # Regex pattern for markdown headings
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    # Patterns to clean content
    JSX_COMPONENT_PATTERN = re.compile(r"<[A-Z][^>]*>.*?</[A-Z][^>]*>", re.DOTALL)
    HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)

    def __init__(self, docs_path: Path):
        """Initialize the parser.

        Args:
            docs_path: Path to the docs directory
        """
        self.docs_path = docs_path

    def parse_file(self, file_path: Path) -> ParsedChapter:
        """Parse a markdown file into a structured chapter.

        Args:
            file_path: Path to the markdown file

        Returns:
            ParsedChapter with extracted metadata and content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)

        metadata = post.metadata
        content = post.content

        # Extract metadata
        chapter = ParsedChapter(
            chapter_id=metadata.get("chapter_id", file_path.stem),
            part_id=metadata.get("part_id", "unknown"),
            title=metadata.get("title", file_path.stem.replace("-", " ").title()),
            description=metadata.get("description"),
            difficulty=metadata.get("difficulty", "intermediate"),
            keywords=metadata.get("keywords", []),
            prerequisites=metadata.get("prerequisites", []),
            file_path=str(file_path.relative_to(self.docs_path)),
            raw_content=content,
            estimated_time=metadata.get("estimated_time"),
            author=metadata.get("author"),
            last_updated=metadata.get("last_updated"),
            tags=metadata.get("tags", []),
        )

        # Parse sections
        chapter.sections = self._extract_sections(content)

        return chapter

    def _extract_sections(self, content: str) -> List[Section]:
        """Extract sections from markdown content.

        Args:
            content: Raw markdown content

        Returns:
            List of Section objects
        """
        lines = content.split("\n")
        sections = []
        current_section = None
        section_start_line = 0
        section_lines = []

        for i, line in enumerate(lines):
            match = self.HEADING_PATTERN.match(line)

            if match:
                # Save previous section if exists
                if current_section is not None:
                    sections.append(
                        Section(
                            title=current_section["title"],
                            level=current_section["level"],
                            content="\n".join(section_lines).strip(),
                            start_line=section_start_line,
                            end_line=i - 1,
                        )
                    )

                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = {"title": title, "level": level}
                section_start_line = i
                section_lines = []
            else:
                section_lines.append(line)

        # Don't forget the last section
        if current_section is not None:
            sections.append(
                Section(
                    title=current_section["title"],
                    level=current_section["level"],
                    content="\n".join(section_lines).strip(),
                    start_line=section_start_line,
                    end_line=len(lines) - 1,
                )
            )

        return sections

    def clean_content(self, content: str) -> str:
        """Clean markdown content for embedding.

        Removes JSX components, HTML comments, and normalizes whitespace.

        Args:
            content: Raw markdown content

        Returns:
            Cleaned content suitable for embedding
        """
        # Remove JSX components (like <div className="...">...</div>)
        cleaned = self.JSX_COMPONENT_PATTERN.sub("", content)

        # Remove HTML comments
        cleaned = self.HTML_COMMENT_PATTERN.sub("", cleaned)

        # Remove code blocks but keep their content description
        cleaned = re.sub(r"```[a-z]*\n", "", cleaned)
        cleaned = cleaned.replace("```", "")

        # Normalize whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def get_all_chapter_files(self) -> List[Path]:
        """Get all markdown chapter files from the docs directory.

        Returns:
            List of paths to markdown files
        """
        files = []
        for md_file in self.docs_path.rglob("*.md"):
            # Skip intro.md and any hidden files
            if md_file.name.startswith("."):
                continue
            files.append(md_file)

        return sorted(files)

    def parse_all_chapters(self) -> List[ParsedChapter]:
        """Parse all chapter files in the docs directory.

        Returns:
            List of ParsedChapter objects
        """
        chapters = []
        for file_path in self.get_all_chapter_files():
            try:
                chapter = self.parse_file(file_path)
                chapters.append(chapter)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                continue

        return chapters
