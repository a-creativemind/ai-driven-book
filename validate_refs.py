import re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Reference:
    number: int
    authors: str
    year: int
    title: str
    source: str
    doi: str | None
    url: str | None

def extract_doi(text: str) -> str | None:
    """Extract DOI from reference text."""
    match = re.search(r'10\.\d{4,}/[^\s]+', text)
    return match.group(0) if match else None

def extract_url(text: str) -> str | None:
    """Extract URL from reference text."""
    match = re.search(r'https?://[^\s]+', text)
    return match.group(0) if match else None

def parse_references(markdown: str) -> list[Reference]:
    """Extract references from markdown."""
    if "## References" not in markdown:
        return []
    ref_section = markdown.split("## References")[-1].split("## ")[0]
    # Match numbered references like "1. Author (Year). *Title*. Source"
    pattern = r'(\d+)\.\s+([^(]+)\((\d{4})\)\.\s+\*([^*]+)\*\.?\s*(.+)?'
    refs = []
    lines = ref_section.strip().split('\n')
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            source = match.group(5).strip() if match.group(5) else ''
            refs.append(Reference(
                number=int(match.group(1)),
                authors=match.group(2).strip(),
                year=int(match.group(3)),
                title=match.group(4).strip(),
                source=source,
                doi=extract_doi(source),
                url=extract_url(source)
            ))
    return refs

def check_citations_in_body(markdown: str, refs: list[Reference]) -> dict:
    """Check if references are cited in the body."""
    body = markdown.split("## References")[0] if "## References" in markdown else markdown
    cited = {}
    for ref in refs:
        # Look for author name mentions or year citations
        author_last = ref.authors.split(',')[0].split('&')[0].strip()
        year_pattern = rf'\b{ref.year}\b'
        author_pattern = rf'\b{author_last}\b'
        cited[ref.number] = bool(re.search(year_pattern, body)) or bool(re.search(author_pattern, body, re.IGNORECASE))
    return cited

if __name__ == '__main__':
    import sys
    chapter_path = sys.argv[1] if len(sys.argv) > 1 else 'docs/physical-ai/control-systems.md'

    content = Path(chapter_path).read_text(encoding='utf-8')
    refs = parse_references(content)

    print(f'Chapter: {chapter_path}')
    print(f'References Found: {len(refs)}')
    print()
    print(f'| # | Authors | Year | Title | DOI | Status |')
    print(f'|---|---------|------|-------|-----|--------|')

    issues = []
    for ref in refs:
        title_short = ref.title[:30] + '...' if len(ref.title) > 30 else ref.title
        doi_display = ref.doi[:15] + '...' if ref.doi and len(ref.doi) > 15 else (ref.doi or '-')

        if ref.doi:
            status = 'VALID'
        elif ref.url:
            status = 'URL OK'
        else:
            status = 'WARN'
            issues.append(f'Reference {ref.number}: No DOI found')

        author_short = ref.authors[:20] + '...' if len(ref.authors) > 20 else ref.authors
        print(f'| {ref.number} | {author_short} | {ref.year} | {title_short} | {doi_display} | {status} |')

    # Check citations
    citation_check = check_citations_in_body(content, refs)
    orphans = [num for num, cited in citation_check.items() if not cited]

    if issues:
        print()
        print('Issues:')
        for issue in issues:
            print(f'- {issue}')

    print()
    print('Citation Check:')
    print(f'- All references cited in body: {"YES" if not orphans else "NO"}')
    print(f'- Orphan references: {orphans if orphans else "None"}')

    print()
    print('Suggestions:')
    # Check for common control theory references
    known_refs = ['Astrom', 'Franklin', 'Siciliano', 'Ogata', 'Ziegler', 'Nichols']
    found_refs = [a for a in known_refs if any(a.lower() in r.authors.lower() for r in refs)]
    missing_refs = [a for a in known_refs if a not in found_refs]

    if not any('Murray' in r.authors for r in refs):
        print('- Consider adding: Astrom & Murray (2021) "Feedback Systems" - seminal modern control text')
    if 'LQR' not in content and 'Kalman' not in content:
        print('- Consider adding coverage of LQR/LQG control for completeness')

    print()
    valid_count = sum(1 for r in refs if r.doi or r.url)
    print(f'Summary: {valid_count}/{len(refs)} references have DOI/URL')
