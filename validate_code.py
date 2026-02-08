import ast
import re
from pathlib import Path

def extract_python_blocks(markdown_path: str) -> list:
    content = Path(markdown_path).read_text(encoding='utf-8')
    pattern = r'```python\n(.*?)```'
    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        blocks.append({
            'code': match.group(1),
            'start': content[:match.start()].count('\n') + 1,
            'end': content[:match.end()].count('\n')
        })
    return blocks

def validate_syntax(code: str) -> tuple:
    try:
        ast.parse(code)
        return True, 'OK'
    except SyntaxError as e:
        return False, f'Line {e.lineno}: {e.msg}'

def check_docstrings(code: str) -> list:
    try:
        tree = ast.parse(code)
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append(f'{node.name} missing docstring')
        return issues
    except:
        return ['Parse error']

def check_type_hints(code: str) -> list:
    try:
        tree = ast.parse(code)
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                missing_params = []
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        missing_params.append(arg.arg)
                if node.returns is None and node.name != '__init__' and node.name != '__post_init__':
                    issues.append(f'{node.name}: missing return type')
                if missing_params:
                    issues.append(f'{node.name}: missing param types')
        return issues
    except:
        return []

if __name__ == '__main__':
    import sys
    chapter_path = sys.argv[1] if len(sys.argv) > 1 else 'docs/physical-ai/control-systems.md'

    blocks = extract_python_blocks(chapter_path)
    print(f'Chapter: {chapter_path}')
    print(f'Code Blocks: {len(blocks)}')
    print()
    print(f'| #  | Lines     | Syntax | Docstrings | Type Hints | Status |')
    print(f'|----|-----------|--------|------------|------------|--------|')

    errors = []
    warnings = []
    valid_count = 0

    for i, block in enumerate(blocks, 1):
        syntax_ok, syntax_msg = validate_syntax(block['code'])
        if syntax_ok:
            doc_issues = check_docstrings(block['code'])
            type_issues = check_type_hints(block['code'])

            doc_status = 'PASS' if not doc_issues else 'WARN'
            type_status = 'PASS' if not type_issues else 'WARN'
            status = 'OK'

            if doc_issues:
                warnings.extend([f'Block {i}: {issue}' for issue in doc_issues])
            if type_issues:
                warnings.extend([f'Block {i}: {issue}' for issue in type_issues])

            valid_count += 1
        else:
            doc_status = '-'
            type_status = '-'
            status = 'ERROR'
            errors.append(f'Block {i}, Line {block["start"]}: {syntax_msg}')

        print(f'| {i:<2} | {block["start"]}-{block["end"]:<4} | {"PASS" if syntax_ok else "FAIL":<6} | {doc_status:<10} | {type_status:<10} | {status:<6} |')

    if errors:
        print()
        print('Errors:')
        for e in errors:
            print(f'- {e}')

    if warnings[:10]:
        print()
        print('Warnings (first 10):')
        for w in warnings[:10]:
            print(f'- {w}')

    print()
    print(f'Summary: {valid_count}/{len(blocks)} blocks valid ({100*valid_count/len(blocks):.1f}%)')
