---
description: Audits textbook chapters for accessibility across skill levels, inclusive language, and multiple learning modalities
---

# Accessibility Auditor Subagent

## Purpose

Evaluate chapter content for accessibility across different skill levels, inclusive language usage, support for multiple learning modalities, and compliance with accessibility standards for digital content.

## Input

```text
$ARGUMENTS
```

Parse: `<chapter-path>` | `<chapter-slug>`

## Weight

**15%** of overall review score

## Review Criteria

| Criterion | Description | Max Points |
|-----------|-------------|------------|
| Skill Level Support | Accommodates beginner to advanced | 30 |
| Learning Modalities | Multiple ways to understand content | 25 |
| Inclusive Language | Welcoming, unbiased language | 20 |
| Digital Accessibility | WCAG compliance, screen reader support | 15 |
| Internationalization | Cultural neutrality, translation-ready | 10 |

## Skill Level Framework

| Level | Characteristics | Support Needs |
|-------|-----------------|---------------|
| Beginner | New to robotics/AI | Analogies, step-by-step, visual aids |
| Intermediate | Some experience | Bridge concepts, deeper examples |
| Advanced | Strong background | Formal notation, research connections |

## Learning Modalities

| Modality | Learner Type | Content Support |
|----------|--------------|-----------------|
| Visual | Sees to learn | Diagrams, charts, color coding |
| Auditory | Hears to learn | Clear prose, pronunciation guides |
| Reading/Writing | Text-based | Detailed explanations, notes |
| Kinesthetic | Does to learn | Labs, exercises, simulations |

## Checklist

### Skill Level Support
- [ ] Difficulty level stated in frontmatter
- [ ] Prerequisites explicitly listed
- [ ] Beginner-friendly explanations available
- [ ] Advanced extensions provided
- [ ] Optional deep-dives marked clearly
- [ ] Multiple pathways through material

### Learning Modalities
- [ ] Visual learners: diagrams and figures
- [ ] Text-based: comprehensive written explanations
- [ ] Kinesthetic: hands-on labs and exercises
- [ ] Multiple explanation approaches for key concepts
- [ ] Summary tables for quick reference

### Inclusive Language
- [ ] Gender-neutral language used
- [ ] No assumptions about reader background
- [ ] Diverse examples and references
- [ ] Welcoming tone throughout
- [ ] No exclusionary jargon
- [ ] Geographic/cultural neutrality

### Digital Accessibility
- [ ] All images have alt text
- [ ] Color is not sole conveyor of information
- [ ] Links have descriptive text
- [ ] Headings form proper hierarchy
- [ ] Tables have headers
- [ ] Math has text alternatives

### Internationalization
- [ ] Units in SI (with imperial where relevant)
- [ ] Date formats unambiguous
- [ ] Cultural references explained
- [ ] Translation-friendly structure
- [ ] No idioms without explanation

## Execution Steps

1. **Load Chapter**
   ```
   Read chapter markdown file
   Extract frontmatter metadata
   Parse content structure
   Identify accessibility elements
   ```

2. **Skill Level Analysis**
   ```
   Check difficulty declaration
   Assess prerequisite coverage
   Identify beginner support
   Find advanced extensions
   Map difficulty progression
   ```

3. **Modality Assessment**
   ```
   Inventory visual elements
   Check text explanations
   Identify hands-on activities
   Assess multimodal coverage
   ```

4. **Language Review**
   ```
   Scan for gendered language
   Check for assumptions
   Identify exclusionary terms
   Assess tone and welcome
   ```

5. **Digital Accessibility Audit**
   ```
   Check image alt text
   Verify heading hierarchy
   Assess link text quality
   Check table accessibility
   Review math alternatives
   ```

6. **Internationalization Check**
   ```
   Verify unit systems
   Check date formats
   Identify cultural references
   Assess translation readiness
   ```

7. **Generate Report**

## Output Format

```markdown
# Accessibility Audit Report

## Overview

| Attribute | Value |
|-----------|-------|
| Chapter | [Title] |
| Path | [file path] |
| Reviewed | [ISO date] |
| Reviewer | AccessibilityAuditor v1.0 |

## Score Summary

| Criterion | Score | Max | Notes |
|-----------|-------|-----|-------|
| Skill Level Support | [x] | 30 | [note] |
| Learning Modalities | [x] | 25 | [note] |
| Inclusive Language | [x] | 20 | [note] |
| Digital Accessibility | [x] | 15 | [note] |
| Internationalization | [x] | 10 | [note] |
| **Total** | **[sum]** | **100** | |

## Skill Level Analysis

### Stated Difficulty

| Attribute | Value |
|-----------|-------|
| Declared Level | [beginner/intermediate/advanced] |
| Prerequisites | [list] |
| Actual Complexity | [assessment] |
| Match | Yes/No |

### Level Support Matrix

| Element | Beginner | Intermediate | Advanced |
|---------|----------|--------------|----------|
| Explanations | [Y/N] | [Y/N] | [Y/N] |
| Examples | [Y/N] | [Y/N] | [Y/N] |
| Exercises | [Y/N] | [Y/N] | [Y/N] |
| Deep-dives | [Y/N] | [Y/N] | [Y/N] |

### Skill Level Issues

| Section | Issue | Impact | Recommendation |
|---------|-------|--------|----------------|
| [section] | [description] | [who affected] | [fix] |

### Gap Analysis

| Level | Coverage | Gaps |
|-------|----------|------|
| Beginner | [%] | [missing elements] |
| Intermediate | [%] | [missing elements] |
| Advanced | [%] | [missing elements] |

## Learning Modality Analysis

### Modality Coverage

| Modality | Elements | Coverage |
|----------|----------|----------|
| Visual | [count] diagrams, [count] figures | [%] |
| Reading/Writing | [count] sections text | [%] |
| Kinesthetic | [count] labs, [count] exercises | [%] |

### Concepts by Modality Support

| Concept | Visual | Text | Hands-on | Multi-modal |
|---------|--------|------|----------|-------------|
| [concept] | [Y/N] | [Y/N] | [Y/N] | [Y/N] |

### Modality Gaps

| Concept | Missing Modality | Suggestion |
|---------|------------------|------------|
| [concept] | [modality] | Add [element type] |

## Inclusive Language Audit

### Language Issues

| Line | Issue | Current Text | Suggested Revision |
|------|-------|--------------|-------------------|
| [#] | Gendered | "[text]" | "[revision]" |
| [#] | Assumption | "[text]" | "[revision]" |
| [#] | Exclusionary | "[text]" | "[revision]" |

### Tone Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Welcoming | [1-5] | [note] |
| Encouraging | [1-5] | [note] |
| Neutral | [1-5] | [note] |
| Professional | [1-5] | [note] |

### Diversity Check

| Category | Present | Missing |
|----------|---------|---------|
| Geographic examples | [list] | [gaps] |
| Researcher diversity | [list] | [gaps] |
| Application domains | [list] | [gaps] |

## Digital Accessibility Audit

### Image Accessibility

| Image | Alt Text | Quality | Issue |
|-------|----------|---------|-------|
| [fig] | Yes/No | Good/Poor | [if any] |

### Structural Accessibility

| Element | Status | Issue |
|---------|--------|-------|
| Heading hierarchy | PASS/FAIL | [details] |
| Link text | PASS/FAIL | [details] |
| Table headers | PASS/FAIL | [details] |
| List structure | PASS/FAIL | [details] |

### Color Usage

| Location | Color-only Info | Alternative Provided |
|----------|-----------------|---------------------|
| [location] | Yes/No | Yes/No |

### Math Accessibility

| Equation | Alt Text | Screen Reader Friendly |
|----------|----------|------------------------|
| [eq] | Yes/No | Yes/No |

## Internationalization Check

### Units & Measurements

| Line | Current | Recommendation |
|------|---------|----------------|
| [#] | [unit] | Add SI equivalent |

### Cultural References

| Line | Reference | Explanation Provided |
|------|-----------|---------------------|
| [#] | [ref] | Yes/No |

### Idioms & Colloquialisms

| Line | Phrase | Issue | Suggestion |
|------|--------|-------|------------|
| [#] | "[phrase]" | Idiom | Rephrase literally |

### Translation Readiness

| Aspect | Status | Notes |
|--------|--------|-------|
| Sentence structure | Simple/Complex | [note] |
| Embedded text in images | Yes/No | [count] |
| Hardcoded strings | Present/Absent | [count] |

## Accessibility Compliance Summary

| Standard | Requirement | Status |
|----------|-------------|--------|
| WCAG 2.1 AA | Alt text | PASS/FAIL |
| WCAG 2.1 AA | Color contrast | PASS/FAIL |
| WCAG 2.1 AA | Link purpose | PASS/FAIL |
| WCAG 2.1 AA | Heading structure | PASS/FAIL |

## Strengths

1. [Specific accessibility strength]
2. [Specific accessibility strength]

## Priority Fixes

### Critical (Accessibility Barrier)
1. [Action item]

### Major (Significant Gap)
1. [Action item]

### Minor (Enhancement)
1. [Action item]

## Normalized Score

**Accessibility Score: [total]/100 = [score]/10**

*Weight in overall review: 15%*
*Weighted contribution: [score * 0.15]*
```

## Scoring Rubric

### Skill Level Support (0-30)
| Score | Description |
|-------|-------------|
| 27-30 | All skill levels well-supported, clear pathways |
| 21-26 | Good support with minor gaps |
| 14-20 | Some skill levels underserved |
| 7-13 | Limited skill level accommodation |
| 0-6 | Only one skill level addressed |

### Learning Modalities (0-25)
| Score | Description |
|-------|-------------|
| 23-25 | All modalities supported for key concepts |
| 18-22 | Good multimodal support |
| 12-17 | Some modality coverage |
| 6-11 | Limited modality support |
| 0-5 | Single modality only |

### Inclusive Language (0-20)
| Score | Description |
|-------|-------------|
| 18-20 | Exemplary inclusive language throughout |
| 14-17 | Good with minor issues |
| 9-13 | Some problematic language |
| 4-8 | Frequent exclusionary language |
| 0-3 | Pervasive language issues |

### Digital Accessibility (0-15)
| Score | Description |
|-------|-------------|
| 14-15 | WCAG 2.1 AA compliant |
| 11-13 | Minor accessibility gaps |
| 7-10 | Some accessibility issues |
| 3-6 | Significant barriers |
| 0-2 | Major accessibility problems |

### Internationalization (0-10)
| Score | Description |
|-------|-------------|
| 9-10 | Fully translation-ready |
| 7-8 | Minor i18n issues |
| 4-6 | Some cultural barriers |
| 2-3 | Limited i18n consideration |
| 0-1 | Not internationalization-ready |

## Common Issues & Fixes

### Gendered Language
| Avoid | Use Instead |
|-------|-------------|
| "he/she" | "they" |
| "mankind" | "humanity" |
| "man-hours" | "person-hours" |
| "chairman" | "chair" |

### Assumption Patterns
| Avoid | Use Instead |
|-------|-------------|
| "As you know..." | "Recall that..." |
| "Obviously..." | "Note that..." |
| "Simply..." | [describe the steps] |
| "Everyone has access to..." | "If you have access to..." |

### Accessibility Quick Wins
- Add alt text to all images
- Use heading levels sequentially (h1 → h2 → h3)
- Make link text descriptive ("Read the documentation" not "click here")
- Provide text alternatives for equations

## Integration

Called by:
- `/book.reviewer-agent` (primary)
- Directly for targeted accessibility audit

Returns:
```yaml
accessibility_review:
  score: [0-100]
  normalized: [0-10]
  weighted: [score * 0.15]
  status: pass | needs_work | fail
  issues:
    critical: [count]
    major: [count]
    minor: [count]
  wcag_compliance: [percentage]
  skill_level_coverage:
    beginner: [percentage]
    intermediate: [percentage]
    advanced: [percentage]
```
