---
description: Validates clarity and pedagogy of textbook chapters
---

# ReviewerAgent - Orchestrator

## Purpose

Orchestrate specialized review subagents to comprehensively evaluate textbook chapters for pedagogical quality, clarity, accuracy, engagement, and accessibility. Aggregates results into a unified review report with weighted scoring.

## Input

```text
$ARGUMENTS
```

Parse: `<chapter-path>` | `<chapter-slug>` | `all`

## Subagents

| Agent | Command | Weight | Focus |
|-------|---------|--------|-------|
| Clarity Reviewer | `/book.clarity-reviewer` | 25% | Explanations, terminology, readability |
| Accuracy Checker | `/book.accuracy-checker` | 25% | Facts, formulas, algorithms, currency |
| Pedagogy Reviewer | `/book.pedagogy-reviewer` | 20% | Learning objectives, scaffolding, assessment |
| Engagement Reviewer | `/book.engagement-reviewer` | 15% | Examples, visuals, interactivity |
| Accessibility Auditor | `/book.accessibility-auditor` | 15% | Skill levels, modalities, inclusion |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      REVIEWER AGENT                              │
│                       (Orchestrator)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: chapter-path                                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              PARALLEL SUBAGENT EXECUTION                 │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │   Clarity    │  │   Accuracy   │  │   Pedagogy   │  │    │
│  │  │  (25%)       │  │   (25%)      │  │   (20%)      │  │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │    │
│  │         │                 │                 │           │    │
│  │  ┌──────────────┐  ┌──────────────┐                    │    │
│  │  │  Engagement  │  │ Accessibility│                    │    │
│  │  │   (15%)      │  │   (15%)      │                    │    │
│  │  └──────┬───────┘  └──────┬───────┘                    │    │
│  │         │                 │                             │    │
│  └─────────┼─────────────────┼─────────────────────────────┘    │
│            │                 │                                   │
│            └────────┬────────┘                                   │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  SCORE AGGREGATION                       │    │
│  │                                                          │    │
│  │  Overall = (Clarity × 0.25) + (Accuracy × 0.25) +       │    │
│  │            (Pedagogy × 0.20) + (Engagement × 0.15) +    │    │
│  │            (Accessibility × 0.15)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 UNIFIED REPORT                           │    │
│  │                                                          │    │
│  │  • Dimension scores and weighted totals                 │    │
│  │  • Consolidated issues by severity                      │    │
│  │  • Priority fix recommendations                         │    │
│  │  • Verdict: APPROVED / NEEDS REVISION / MAJOR REVISION  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Execution Steps

1. **Parse Input**
   ```
   Resolve chapter path from input
   Verify chapter file exists
   Load chapter metadata
   ```

2. **Run Subagents (Parallel)**
   ```
   Execute all 5 subagents simultaneously:

   /book.clarity-reviewer <chapter-path>
   /book.accuracy-checker <chapter-path>
   /book.pedagogy-reviewer <chapter-path>
   /book.engagement-reviewer <chapter-path>
   /book.accessibility-auditor <chapter-path>
   ```

3. **Collect Results**
   ```
   Wait for all subagents to complete
   Collect individual reports
   Parse scores and issues
   ```

4. **Aggregate Scores**
   ```
   Calculate weighted overall score:

   overall = (clarity.normalized × 0.25) +
             (accuracy.normalized × 0.25) +
             (pedagogy.normalized × 0.20) +
             (engagement.normalized × 0.15) +
             (accessibility.normalized × 0.15)
   ```

5. **Consolidate Issues**
   ```
   Merge all issues from subagents
   Deduplicate overlapping issues
   Sort by severity (critical → major → minor)
   Prioritize fixes
   ```

6. **Determine Verdict**
   ```
   if overall >= 8.0 and critical_issues == 0:
       verdict = "APPROVED"
   elif overall >= 6.0 and critical_issues <= 2:
       verdict = "NEEDS REVISION"
   else:
       verdict = "MAJOR REVISION"
   ```

7. **Generate Report**

## Output Format

```markdown
# Chapter Review Report

## Overview

| Attribute | Value |
|-----------|-------|
| Chapter | [Title] |
| Path | [file path] |
| Reviewed | [ISO date] |
| Reviewer | ReviewerAgent v2.0 (Orchestrated) |

## Executive Summary

**Overall Score: [X.X]/10**

**Verdict: [APPROVED / NEEDS REVISION / MAJOR REVISION]**

**Confidence: [High / Medium / Low]**

| Issue Type | Count |
|------------|-------|
| Critical | [n] |
| Major | [n] |
| Minor | [n] |

## Dimension Scores

| Dimension | Raw | Weight | Weighted | Status |
|-----------|-----|--------|----------|--------|
| Clarity | [x]/10 | 25% | [x×0.25] | [indicator] |
| Accuracy | [x]/10 | 25% | [x×0.25] | [indicator] |
| Pedagogy | [x]/10 | 20% | [x×0.20] | [indicator] |
| Engagement | [x]/10 | 15% | [x×0.15] | [indicator] |
| Accessibility | [x]/10 | 15% | [x×0.15] | [indicator] |
| **Overall** | | **100%** | **[sum]** | |

### Score Visualization

```
Clarity        [████████░░] 8.0
Accuracy       [███████░░░] 7.0
Pedagogy       [█████████░] 9.0
Engagement     [███████░░░] 7.0
Accessibility  [████████░░] 8.0
─────────────────────────────
Overall        [████████░░] 7.8
```

## Strengths

### Clarity
- [Key strength from clarity review]

### Accuracy
- [Key strength from accuracy review]

### Pedagogy
- [Key strength from pedagogy review]

### Engagement
- [Key strength from engagement review]

### Accessibility
- [Key strength from accessibility review]

## Consolidated Issues

### Critical (Must Fix Before Publish)

| # | Dimension | Line | Issue | Recommendation |
|---|-----------|------|-------|----------------|
| 1 | [dim] | [#] | [issue] | [fix] |

### Major (Should Fix)

| # | Dimension | Line | Issue | Recommendation |
|---|-----------|------|-------|----------------|
| 1 | [dim] | [#] | [issue] | [fix] |

### Minor (Consider)

| # | Dimension | Line | Issue | Recommendation |
|---|-----------|------|-------|----------------|
| 1 | [dim] | [#] | [issue] | [fix] |

## Priority Action Items

### Before Publish
1. [ ] [Action from critical issues]
2. [ ] [Action from critical issues]

### Recommended
1. [ ] [Action from major issues]
2. [ ] [Action from major issues]

### Optional Improvements
1. [ ] [Action from minor issues]

## Detailed Subagent Reports

<details>
<summary>📖 Clarity Review (25%)</summary>

[Embedded clarity review report]

</details>

<details>
<summary>✅ Accuracy Check (25%)</summary>

[Embedded accuracy review report]

</details>

<details>
<summary>🎓 Pedagogy Review (20%)</summary>

[Embedded pedagogy review report]

</details>

<details>
<summary>🎯 Engagement Review (15%)</summary>

[Embedded engagement review report]

</details>

<details>
<summary>♿ Accessibility Audit (15%)</summary>

[Embedded accessibility review report]

</details>

## Verdict Rationale

**Score Analysis:**
- [Explanation of score]

**Issue Assessment:**
- [Summary of critical blockers or lack thereof]

**Recommendation:**
- [Final recommendation]

## Next Steps

| If Verdict | Action |
|------------|--------|
| APPROVED | Proceed to publication |
| NEEDS REVISION | Address critical/major issues, re-review |
| MAJOR REVISION | Significant rewrite required |

---

*Review generated by ReviewerAgent v2.0*
*Subagents: ClarityReviewer, AccuracyChecker, PedagogyReviewer, EngagementReviewer, AccessibilityAuditor*
```

## Scoring Thresholds

| Overall Score | Verdict | Description |
|---------------|---------|-------------|
| 9.0 - 10.0 | APPROVED | Excellent quality, ready for publication |
| 8.0 - 8.9 | APPROVED | High quality with minor polish needed |
| 7.0 - 7.9 | NEEDS REVISION | Good foundation, specific issues to address |
| 6.0 - 6.9 | NEEDS REVISION | Multiple areas need improvement |
| 4.0 - 5.9 | MAJOR REVISION | Significant problems throughout |
| 0.0 - 3.9 | MAJOR REVISION | Fundamental rewrite required |

## Critical Issue Override

Regardless of overall score, verdict is NEEDS REVISION or worse if:
- Any factual errors (accuracy critical issues)
- Code examples that don't run
- Missing learning objectives
- Accessibility barriers (no alt text, broken structure)

## Integration

### Invoked By
- `/book.orchestrator` (after generation)
- Manual quality assurance
- CI/CD pipeline for automated checks

### Stores Results

Updates chapter frontmatter:
```yaml
review:
  date: [ISO date]
  version: 2.0
  overall_score: [X.X]
  scores:
    clarity: [X.X]
    accuracy: [X.X]
    pedagogy: [X.X]
    engagement: [X.X]
    accessibility: [X.X]
  verdict: [APPROVED|NEEDS_REVISION|MAJOR_REVISION]
  issues:
    critical: [n]
    major: [n]
    minor: [n]
```

Creates review report:
```
reviews/<chapter-slug>-review-[date].md
```

## Example Usage

**Input:** `embodiment`

**Process:**
1. Resolves to `docs/physical-ai/embodiment.md`
2. Runs all 5 subagents in parallel
3. Aggregates scores:
   - Clarity: 9.0 × 0.25 = 2.25
   - Accuracy: 8.0 × 0.25 = 2.00
   - Pedagogy: 9.0 × 0.20 = 1.80
   - Engagement: 8.0 × 0.15 = 1.20
   - Accessibility: 8.0 × 0.15 = 1.20
   - **Overall: 8.45**
4. Issues: 0 critical, 2 major, 5 minor
5. **Verdict: APPROVED**

**Output:** Comprehensive review report with all details

## Batch Mode

For `all` input:
```
Process each chapter in sequence:
1. Generate individual reports
2. Create summary table
3. Identify chapters needing attention
4. Generate book-wide quality report
```

### Batch Output

```markdown
# Book-Wide Quality Report

## Chapter Status

| Chapter | Score | Verdict | Critical | Major |
|---------|-------|---------|----------|-------|
| embodiment | 8.5 | APPROVED | 0 | 2 |
| sensors | 7.2 | NEEDS REVISION | 1 | 4 |
| ... | ... | ... | ... | ... |

## Overall Book Quality

**Average Score: [X.X]/10**

**Ready for Publication: [Y/N]**

**Chapters Requiring Attention:**
1. [chapter] - [reason]
```
