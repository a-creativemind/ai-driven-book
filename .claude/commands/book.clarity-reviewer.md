---
description: Reviews textbook chapters for clarity, readability, and absence of unexplained jargon
---

# Clarity Reviewer Subagent

## Purpose

Evaluate chapter content for clarity of explanation, proper terminology usage, and overall readability. This agent ensures readers can understand the material without unnecessary confusion.

## Input

```text
$ARGUMENTS
```

Parse: `<chapter-path>` | `<chapter-slug>`

## Weight

**25%** of overall review score

## Review Criteria

| Criterion | Description | Max Points |
|-----------|-------------|------------|
| Explanation Quality | Concepts explained clearly without assuming knowledge | 25 |
| Terminology Handling | Technical terms defined before or at first use | 25 |
| Sentence Structure | Clear, well-constructed sentences | 20 |
| Logical Flow | Ideas progress logically within paragraphs | 15 |
| Readability Metrics | Appropriate reading level for audience | 15 |

## Checklist

### Terminology
- [ ] All technical terms defined on first use
- [ ] Glossary entries exist for key terms
- [ ] Acronyms expanded at first occurrence
- [ ] Domain-specific jargon is contextualized
- [ ] No "insider language" without explanation

### Explanations
- [ ] Abstract concepts grounded with concrete examples
- [ ] Analogies used for complex ideas
- [ ] Step-by-step breakdowns for processes
- [ ] Visual aids complement textual explanations
- [ ] No unexplained leaps in logic

### Structure
- [ ] Paragraphs focus on single ideas
- [ ] Topic sentences guide readers
- [ ] Transitions connect sections smoothly
- [ ] Headers accurately describe content
- [ ] Lists used appropriately for enumeration

### Readability
- [ ] Average sentence length < 25 words
- [ ] Flesch-Kincaid grade level: 12-14
- [ ] Technical term density < 15%
- [ ] Passive voice < 20%
- [ ] No run-on sentences

## Execution Steps

1. **Load Chapter**
   ```
   Read chapter markdown file
   Extract all text content (excluding code blocks)
   Build term occurrence map
   ```

2. **Terminology Analysis**
   ```
   Identify all technical terms
   Check if defined before use
   Flag undefined jargon
   Verify glossary coverage
   ```

3. **Sentence Analysis**
   ```
   Calculate average sentence length
   Identify complex sentences (>30 words)
   Flag passive voice overuse
   Check for run-on sentences
   ```

4. **Paragraph Analysis**
   ```
   Verify single-focus paragraphs
   Check topic sentence presence
   Evaluate logical flow
   Assess transition quality
   ```

5. **Readability Scoring**
   ```
   Calculate Flesch-Kincaid grade level
   Measure technical term density
   Evaluate overall accessibility
   ```

6. **Generate Report**

## Output Format

```markdown
# Clarity Review Report

## Overview

| Attribute | Value |
|-----------|-------|
| Chapter | [Title] |
| Path | [file path] |
| Reviewed | [ISO date] |
| Reviewer | ClarityReviewer v1.0 |

## Score Summary

| Criterion | Score | Max | Notes |
|-----------|-------|-----|-------|
| Explanation Quality | [x] | 25 | [note] |
| Terminology Handling | [x] | 25 | [note] |
| Sentence Structure | [x] | 20 | [note] |
| Logical Flow | [x] | 15 | [note] |
| Readability Metrics | [x] | 15 | [note] |
| **Total** | **[sum]** | **100** | |

## Readability Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Flesch-Kincaid Grade | [grade] | 12-14 | PASS/FAIL |
| Avg Sentence Length | [words] | < 25 | PASS/FAIL |
| Technical Term Density | [%] | < 15% | PASS/FAIL |
| Passive Voice Usage | [%] | < 20% | PASS/FAIL |

## Terminology Issues

### Undefined Terms (Must Fix)

| Term | First Occurrence | Recommendation |
|------|------------------|----------------|
| [term] | Line [#] | Define before use or add to glossary |

### Missing Glossary Entries

| Term | Occurrences | Priority |
|------|-------------|----------|
| [term] | [count] | High/Medium/Low |

## Clarity Issues

### Complex Sentences (Simplify)

| Line | Sentence | Issue | Suggestion |
|------|----------|-------|------------|
| [#] | "[text...]" | Too long/complex | Split or simplify |

### Unexplained Concepts

| Line | Concept | Issue |
|------|---------|-------|
| [#] | [concept] | No prior explanation |

### Flow Issues

| Section | Issue | Recommendation |
|---------|-------|----------------|
| [section] | [description] | [fix] |

## Strengths

1. [Specific clarity strength]
2. [Specific clarity strength]

## Priority Fixes

### Critical (Blocks Comprehension)
1. [Action item]

### Major (Significantly Impacts Understanding)
1. [Action item]

### Minor (Polish)
1. [Action item]

## Normalized Score

**Clarity Score: [total]/100 = [score]/10**

*Weight in overall review: 25%*
*Weighted contribution: [score * 0.25]*
```

## Scoring Rubric

### Explanation Quality (0-25)
| Score | Description |
|-------|-------------|
| 23-25 | All concepts crystal clear, excellent examples |
| 18-22 | Clear with minor areas for improvement |
| 12-17 | Some concepts unclear or poorly explained |
| 6-11 | Frequent confusion, missing explanations |
| 0-5 | Largely incomprehensible |

### Terminology Handling (0-25)
| Score | Description |
|-------|-------------|
| 23-25 | All terms defined appropriately |
| 18-22 | Most terms defined, minor gaps |
| 12-17 | Several undefined technical terms |
| 6-11 | Widespread jargon without definition |
| 0-5 | Terminology barrier to understanding |

### Sentence Structure (0-20)
| Score | Description |
|-------|-------------|
| 18-20 | Clear, well-constructed throughout |
| 14-17 | Generally good, occasional issues |
| 9-13 | Frequent structural problems |
| 4-8 | Poor sentence construction |
| 0-3 | Unreadable sentence structure |

### Logical Flow (0-15)
| Score | Description |
|-------|-------------|
| 14-15 | Perfect logical progression |
| 11-13 | Good flow, minor jumps |
| 7-10 | Some logical gaps |
| 3-6 | Disorganized progression |
| 0-2 | No coherent flow |

### Readability Metrics (0-15)
| Score | Description |
|-------|-------------|
| 14-15 | All metrics in target range |
| 11-13 | Most metrics acceptable |
| 7-10 | Some metrics outside range |
| 3-6 | Multiple metrics failing |
| 0-2 | Severely poor readability |

## Integration

Called by:
- `/book.reviewer-agent` (primary)
- Directly for targeted clarity review

Returns:
```yaml
clarity_review:
  score: [0-100]
  normalized: [0-10]
  weighted: [score * 0.25]
  status: pass | needs_work | fail
  issues:
    critical: [count]
    major: [count]
    minor: [count]
```
