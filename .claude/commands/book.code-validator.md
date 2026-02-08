---
description: Validates Python code examples in textbook chapters for syntax, style, and execution correctness
---

## Input

```text
$ARGUMENTS
```

Parse: `<chapter-path>` or `all` for full validation

## Task

Extract and validate all Python code blocks from chapter markdown files.

## Steps

1. **Extract Code Blocks**
   ```python
   # Find all ```python ... ``` blocks in the chapter
   # Track line numbers for error reporting
   ```

2. **Syntax Validation**
   - Parse with `ast.parse()` to check syntax
   - Report line number and error for failures

3. **Style Checks**
   | Check | Requirement |
   |-------|-------------|
   | Docstrings | All classes and functions must have docstrings |
   | Type hints | Function parameters and returns should have hints |
   | Imports | Standard library, then third-party, then local |
   | Line length | Max 88 characters (Black standard) |

4. **Execution Test** (optional, use `--exec` flag)
   - Create isolated environment
   - Run each code block
   - Capture output
   - Compare with documented "Output:" section

5. **Dependency Check**
   - List all imports used
   - Verify against `package.json` or `requirements.txt`
   - Flag missing dependencies

## Validation Script

```python
import ast
import re
from pathlib import Path

def extract_python_blocks(markdown_path: str) -> list[dict]:
    """Extract Python code blocks with metadata."""
    content = Path(markdown_path).read_text(encoding='utf-8')
    pattern = r'```python\n(.*?)```'
    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        blocks.append({
            'code': match.group(1),
            'start': content[:match.start()].count('\n') + 1
        })
    return blocks

def validate_syntax(code: str) -> tuple[bool, str]:
    """Check Python syntax validity."""
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"

def check_docstrings(code: str) -> list[str]:
    """Check for missing docstrings."""
    tree = ast.parse(code)
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not ast.get_docstring(node):
                issues.append(f"{node.name} missing docstring")
    return issues
```

## Output Format

```
Chapter: docs/physical-ai/control-systems.md
Code Blocks: 12

| # | Lines | Syntax | Docstrings | Type Hints | Status |
|---|-------|--------|------------|------------|--------|
| 1 | 15-42 | PASS   | PASS       | WARN       | OK     |
| 2 | 56-89 | PASS   | PASS       | PASS       | OK     |
| 3 | 102-130 | FAIL | -          | -          | ERROR  |

Errors:
- Block 3, Line 105: IndentationError: unexpected indent

Warnings:
- Block 1: Missing type hints on `calculate_response`

Summary: 11/12 blocks valid (91.7%)
```
