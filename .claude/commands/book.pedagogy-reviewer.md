---
description: Reviews textbook chapters for pedagogical structure, learning objectives alignment, and educational scaffolding
---

# Pedagogy Reviewer Subagent

## Purpose

Evaluate chapter content for sound pedagogical design including learning objectives quality, content scaffolding, prerequisite handling, and assessment alignment. Ensures the chapter follows best practices in instructional design.

## Input

```text
$ARGUMENTS
```

Parse: `<chapter-path>` | `<chapter-slug>`

## Weight

**20%** of overall review score

## Review Criteria

| Criterion | Description | Max Points |
|-----------|-------------|------------|
| Learning Objectives | Measurable, Bloom's taxonomy aligned | 25 |
| Content Scaffolding | Builds from simple to complex | 25 |
| Prerequisite Handling | Clear dependencies, prior knowledge | 20 |
| Assessment Alignment | Exercises match objectives | 20 |
| Summary Quality | Reinforces key takeaways | 10 |

## Bloom's Taxonomy Reference

| Level | Verbs | Cognitive Process |
|-------|-------|-------------------|
| Remember | Define, list, recall, identify | Retrieving |
| Understand | Explain, describe, summarize | Comprehending |
| Apply | Implement, execute, use | Applying |
| Analyze | Differentiate, organize, compare | Breaking down |
| Evaluate | Critique, judge, assess | Making judgments |
| Create | Design, construct, develop | Producing new |

## Checklist

### Learning Objectives
- [ ] 3-7 learning objectives present
- [ ] Each objective starts with action verb
- [ ] Objectives are measurable (can be assessed)
- [ ] Objectives span multiple Bloom's levels
- [ ] Objectives align with chapter content
- [ ] No vague verbs (understand, know, learn)

### Content Scaffolding
- [ ] Introduction motivates the topic
- [ ] Concrete examples precede abstract concepts
- [ ] Complexity increases gradually
- [ ] Each section builds on previous
- [ ] No unexplained forward references
- [ ] Key concepts highlighted/called out

### Prerequisites
- [ ] Prerequisites listed in frontmatter
- [ ] Required prior knowledge explicit
- [ ] Links to prerequisite chapters provided
- [ ] No assumed knowledge beyond prerequisites
- [ ] Review of necessary background included

### Assessment
- [ ] Exercises match learning objectives
- [ ] Multiple difficulty levels (beginner, intermediate, advanced)
- [ ] Practice opportunities before assessment
- [ ] Self-check questions throughout
- [ ] Solution hints or approaches provided

### Summary
- [ ] Key points restated concisely
- [ ] Connections to other chapters noted
- [ ] Common misconceptions addressed
- [ ] Next steps or further reading suggested

## Execution Steps

1. **Load Chapter**
   ```
   Read chapter markdown file
   Extract frontmatter (objectives, prerequisites)
   Parse section structure
   Extract exercises
   ```

2. **Analyze Learning Objectives**
   ```
   For each objective:
     - Identify action verb
     - Map to Bloom's level
     - Check measurability
     - Verify content coverage
   ```

3. **Evaluate Scaffolding**
   ```
   Map content progression:
     - Identify concept introduction order
     - Check example-theory sequence
     - Verify complexity gradient
     - Flag backwards references
   ```

4. **Check Prerequisites**
   ```
   Extract prerequisite claims
   Verify links to prior chapters
   Identify implicit assumptions
   Check background coverage
   ```

5. **Assess Alignment**
   ```
   Map exercises to objectives:
     - Coverage matrix
     - Difficulty distribution
     - Gap analysis
   ```

6. **Review Summary**
   ```
   Check completeness
   Verify key point coverage
   Assess connection quality
   ```

7. **Generate Report**

## Output Format

```markdown
# Pedagogy Review Report

## Overview

| Attribute | Value |
|-----------|-------|
| Chapter | [Title] |
| Path | [file path] |
| Reviewed | [ISO date] |
| Reviewer | PedagogyReviewer v1.0 |

## Score Summary

| Criterion | Score | Max | Notes |
|-----------|-------|-----|-------|
| Learning Objectives | [x] | 25 | [note] |
| Content Scaffolding | [x] | 25 | [note] |
| Prerequisite Handling | [x] | 20 | [note] |
| Assessment Alignment | [x] | 20 | [note] |
| Summary Quality | [x] | 10 | [note] |
| **Total** | **[sum]** | **100** | |

## Learning Objectives Analysis

### Objectives Table

| # | Objective | Action Verb | Bloom's Level | Measurable | Covered |
|---|-----------|-------------|---------------|------------|---------|
| LO1 | "[text]" | [verb] | [level] | Yes/No | Yes/Partial/No |
| LO2 | "[text]" | [verb] | [level] | Yes/No | Yes/Partial/No |

### Bloom's Distribution

```
Create      | [count] ████
Evaluate    | [count] ██
Analyze     | [count] ███████
Apply       | [count] █████████
Understand  | [count] ██████
Remember    | [count] ███
```

### Issues

| Objective | Issue | Recommendation |
|-----------|-------|----------------|
| LO[#] | [issue] | [fix] |

## Scaffolding Analysis

### Content Flow

```
Section 1: [title]
  └── Introduces: [concepts]
  └── Builds on: [prior concepts]

Section 2: [title]
  └── Introduces: [concepts]
  └── Builds on: [concepts from S1]

[...]
```

### Scaffolding Issues

| Section | Issue | Impact | Recommendation |
|---------|-------|--------|----------------|
| [section] | [description] | [severity] | [fix] |

### Concept Introduction Order

| Concept | Introduced | First Used | Gap |
|---------|------------|------------|-----|
| [concept] | Section [#] | Section [#] | OK/Issue |

## Prerequisite Analysis

### Stated Prerequisites
- [ ] [Prerequisite 1] - Link: [Y/N]
- [ ] [Prerequisite 2] - Link: [Y/N]

### Implicit Prerequisites (Not Stated)

| Concept | Assumed In | Should Be Listed |
|---------|------------|------------------|
| [concept] | Section [#] | Yes/Background |

### Background Coverage

| Required Knowledge | Coverage |
|--------------------|----------|
| [topic] | Adequate/Missing/Brief |

## Assessment Alignment

### Objective-Exercise Matrix

|  | Ex 1 | Ex 2 | Ex 3 | Coverage |
|--|------|------|------|----------|
| LO1 | X | | | Partial |
| LO2 | | X | X | Full |
| LO3 | | | | **None** |

### Difficulty Distribution

| Difficulty | Count | Target |
|------------|-------|--------|
| Beginner | [n] | 1-2 |
| Intermediate | [n] | 1-2 |
| Advanced | [n] | 1 |

### Exercise Issues

| Exercise | Issue | Recommendation |
|----------|-------|----------------|
| Ex [#] | [description] | [fix] |

## Summary Assessment

| Element | Present | Quality |
|---------|---------|---------|
| Key points | Yes/No | Good/Adequate/Poor |
| Connections | Yes/No | Good/Adequate/Poor |
| Misconceptions | Yes/No | Good/Adequate/Poor |
| Next steps | Yes/No | Good/Adequate/Poor |

## Strengths

1. [Specific pedagogical strength]
2. [Specific pedagogical strength]

## Priority Fixes

### Critical (Learning Path Blocked)
1. [Action item]

### Major (Significant Pedagogical Gap)
1. [Action item]

### Minor (Enhancement)
1. [Action item]

## Normalized Score

**Pedagogy Score: [total]/100 = [score]/10**

*Weight in overall review: 20%*
*Weighted contribution: [score * 0.20]*
```

## Scoring Rubric

### Learning Objectives (0-25)
| Score | Description |
|-------|-------------|
| 23-25 | All objectives measurable, diverse Bloom's levels, fully covered |
| 18-22 | Good objectives with minor issues |
| 12-17 | Some objectives unclear or uncovered |
| 6-11 | Poor objective quality or coverage |
| 0-5 | Objectives missing or unusable |

### Content Scaffolding (0-25)
| Score | Description |
|-------|-------------|
| 23-25 | Perfect progression, excellent scaffolding |
| 18-22 | Good flow with minor jumps |
| 12-17 | Some scaffolding issues |
| 6-11 | Poor concept ordering |
| 0-5 | No coherent scaffolding |

### Prerequisite Handling (0-20)
| Score | Description |
|-------|-------------|
| 18-20 | Prerequisites clear, all linked, well-covered |
| 14-17 | Good prerequisite handling |
| 9-13 | Some gaps in prerequisite coverage |
| 4-8 | Significant prerequisite issues |
| 0-3 | Prerequisites ignored |

### Assessment Alignment (0-20)
| Score | Description |
|-------|-------------|
| 18-20 | All objectives assessed, good difficulty spread |
| 14-17 | Good alignment with minor gaps |
| 9-13 | Some objectives unassessed |
| 4-8 | Poor exercise-objective alignment |
| 0-3 | Exercises don't match objectives |

### Summary Quality (0-10)
| Score | Description |
|-------|-------------|
| 9-10 | Comprehensive, reinforcing summary |
| 7-8 | Good summary coverage |
| 4-6 | Adequate but incomplete |
| 2-3 | Minimal summary |
| 0-1 | No meaningful summary |

## Instructional Design Principles

### Gagné's Nine Events of Instruction
1. Gain attention
2. Inform learners of objectives
3. Stimulate recall of prior learning
4. Present content
5. Provide guidance
6. Elicit performance
7. Provide feedback
8. Assess performance
9. Enhance retention and transfer

### Constructive Alignment
```
Learning Objectives
        ↓
Teaching Activities ←→ Assessment Tasks
```

## Integration

Called by:
- `/book.reviewer-agent` (primary)
- Directly for targeted pedagogy review

Returns:
```yaml
pedagogy_review:
  score: [0-100]
  normalized: [0-10]
  weighted: [score * 0.20]
  status: pass | needs_work | fail
  issues:
    critical: [count]
    major: [count]
    minor: [count]
  objectives:
    count: [n]
    coverage: [percentage]
    bloom_distribution: {...}
```
