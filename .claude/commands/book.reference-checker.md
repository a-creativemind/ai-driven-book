---
description: Validates and enriches academic references in textbook chapters using web search and citation databases
---

## Input

```text
$ARGUMENTS
```

Parse: `<chapter-path>` or `all` for full validation

## Task

Validate academic references and suggest improvements for textbook chapters.

## Steps

1. **Extract References**
   - Find `## References` section
   - Parse each numbered reference
   - Extract: authors, year, title, journal/publisher, DOI/URL

2. **Validate Each Reference**

   | Check | Method |
   |-------|--------|
   | DOI validity | Query doi.org resolver |
   | Author names | Check formatting consistency |
   | Year accuracy | Verify against source |
   | Title match | Cross-reference with search |
   | URL accessibility | HTTP HEAD request |

3. **Citation Format Check**

   Expected format (APA-like):
   ```
   1. Last, F., & Last, F. (Year). *Title*. Journal, Vol(Issue), Pages. DOI
   2. Last, F. (Year). *Book Title*. Publisher.
   ```

4. **Relevance Analysis**
   - Check if reference is cited in chapter body
   - Flag orphan references (listed but not cited)
   - Flag missing references (cited but not listed)

5. **Enrichment Suggestions**
   - Search for newer related papers (post-2023)
   - Suggest seminal works if missing
   - Recommend open-access alternatives

## Reference Validation Script

```python
import re
from dataclasses import dataclass

@dataclass
class Reference:
    number: int
    authors: str
    year: int
    title: str
    source: str
    doi: str | None
    url: str | None

def parse_references(markdown: str) -> list[Reference]:
    """Extract references from markdown."""
    ref_section = markdown.split("## References")[-1]
    pattern = r'(\d+)\.\s+(.+?)\((\d{4})\)\.\s+\*(.+?)\*\.(.+)'
    refs = []
    for match in re.finditer(pattern, ref_section):
        refs.append(Reference(
            number=int(match.group(1)),
            authors=match.group(2).strip(),
            year=int(match.group(3)),
            title=match.group(4).strip(),
            source=match.group(5).strip(),
            doi=extract_doi(match.group(5)),
            url=extract_url(match.group(5))
        ))
    return refs

def extract_doi(text: str) -> str | None:
    """Extract DOI from reference text."""
    match = re.search(r'10\.\d{4,}/[^\s]+', text)
    return match.group(0) if match else None
```

## Output Format

```
Chapter: docs/physical-ai/sim2real.md
References Found: 7

| # | Authors | Year | Title | DOI | Status |
|---|---------|------|-------|-----|--------|
| 1 | Tobin et al. | 2017 | Domain Randomization... | 10.1109/... | VALID |
| 2 | Peng et al. | 2018 | Sim-to-Real... | 10.15607/... | VALID |
| 3 | OpenAI | 2019 | Solving Rubik's... | - | URL OK |
| 4 | Tan et al. | 2018 | Sim-to-Real... | MISSING | WARN |

Issues:
- Reference 4: No DOI found, suggest adding

Citation Check:
- All references cited in body: YES
- Orphan references: None
- Missing citations: None

Suggestions:
- Consider adding: Rusu et al. (2017) "Sim-to-Real Robot Learning"
- Consider adding: Muratore et al. (2022) "Robot Learning from Randomized Simulations"

Summary: 7/7 references valid
```
