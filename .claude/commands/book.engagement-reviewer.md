---
description: Reviews textbook chapters for reader engagement including examples, visuals, interactivity, and real-world applications
---

# Engagement Reviewer Subagent

## Purpose

Evaluate chapter content for reader engagement factors including quality of examples, visual elements, interactive components, real-world applications, and elements that maintain reader interest throughout the material.

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
| Example Quality | Relevant, varied, illuminating examples | 30 |
| Visual Elements | Diagrams, figures, illustrations | 25 |
| Real-World Applications | Practical relevance demonstrated | 20 |
| Interactive Elements | Labs, exercises, thought questions | 15 |
| Callouts & Highlights | Key insights, warnings, tips | 10 |

## Engagement Elements

### Types of Examples
- **Motivating examples**: Hook readers at section start
- **Worked examples**: Step-by-step problem solving
- **Counter-examples**: Show what doesn't work
- **Edge cases**: Explore boundary conditions
- **Industry examples**: Real applications in use

### Visual Types
- **Conceptual diagrams**: Abstract relationships
- **Technical schematics**: System architecture
- **Flowcharts**: Process flows
- **Data visualizations**: Graphs, plots
- **Photos/Renders**: Real robots, simulations

### Interactive Elements
- **Inline questions**: "What do you think happens if...?"
- **Mini-exercises**: Quick practice opportunities
- **Lab sections**: Hands-on activities
- **Reflection prompts**: Deeper thinking
- **Code experiments**: Modify and observe

## Checklist

### Examples
- [ ] Every major concept has at least one example
- [ ] Examples are relevant to robotics domain
- [ ] Variety of example types used
- [ ] Examples progress from simple to complex
- [ ] Real-world context provided for examples
- [ ] Code examples produce interesting outputs

### Visual Elements
- [ ] Key concepts illustrated visually
- [ ] All figures have descriptive captions
- [ ] All images have alt text
- [ ] Diagrams use consistent styling
- [ ] Equations complemented with visual intuition
- [ ] System architectures shown graphically

### Real-World Applications
- [ ] Chapter opens with motivating application
- [ ] Industry use cases mentioned
- [ ] Current research referenced
- [ ] Practical limitations discussed
- [ ] Career relevance noted where appropriate

### Interactivity
- [ ] Questions embedded in content
- [ ] Lab exercises included
- [ ] Opportunities for experimentation
- [ ] Self-assessment checkpoints
- [ ] Links to interactive resources

### Callouts
- [ ] "Key Insight" boxes for important concepts
- [ ] "Warning" boxes for common pitfalls
- [ ] "Tip" boxes for best practices
- [ ] "Note" boxes for additional context
- [ ] "Try It" boxes for quick experiments

## Execution Steps

1. **Load Chapter**
   ```
   Read chapter markdown file
   Extract all examples
   Catalog visual elements
   Identify interactive sections
   Map callout usage
   ```

2. **Example Analysis**
   ```
   For each example:
     - Classify type
     - Assess relevance
     - Check completeness
     - Evaluate variety
   ```

3. **Visual Inventory**
   ```
   Count and classify figures
   Check caption presence
   Verify alt text
   Assess concept coverage
   ```

4. **Application Assessment**
   ```
   Identify real-world references
   Evaluate industry relevance
   Check research currency
   Assess practical context
   ```

5. **Interactivity Check**
   ```
   Count inline questions
   Identify lab sections
   Map reflection prompts
   Assess engagement opportunities
   ```

6. **Callout Audit**
   ```
   Inventory callout types
   Check appropriate usage
   Verify key concepts highlighted
   Assess distribution
   ```

7. **Generate Report**

## Output Format

```markdown
# Engagement Review Report

## Overview

| Attribute | Value |
|-----------|-------|
| Chapter | [Title] |
| Path | [file path] |
| Reviewed | [ISO date] |
| Reviewer | EngagementReviewer v1.0 |

## Score Summary

| Criterion | Score | Max | Notes |
|-----------|-------|-----|-------|
| Example Quality | [x] | 30 | [note] |
| Visual Elements | [x] | 25 | [note] |
| Real-World Applications | [x] | 20 | [note] |
| Interactive Elements | [x] | 15 | [note] |
| Callouts & Highlights | [x] | 10 | [note] |
| **Total** | **[sum]** | **100** | |

## Example Analysis

### Example Inventory

| # | Type | Topic | Relevance | Quality |
|---|------|-------|-----------|---------|
| 1 | [type] | [topic] | High/Med/Low | Good/Adequate/Poor |
| 2 | [type] | [topic] | High/Med/Low | Good/Adequate/Poor |

### Example Distribution

```
Motivating   | [count] ████
Worked       | [count] ██████████
Counter      | [count] ██
Edge Case    | [count] ███
Industry     | [count] █████
```

### Example Issues

| Section | Issue | Recommendation |
|---------|-------|----------------|
| [section] | No example for [concept] | Add worked example |

## Visual Elements Analysis

### Figure Inventory

| # | Type | Caption | Alt Text | Quality |
|---|------|---------|----------|---------|
| Fig 1 | [type] | Yes/No | Yes/No | Good/Adequate/Poor |
| Fig 2 | [type] | Yes/No | Yes/No | Good/Adequate/Poor |

### Visual Coverage

| Concept | Visual Support |
|---------|----------------|
| [concept] | Yes/No/Partial |

### Visual Issues

| Figure | Issue | Recommendation |
|--------|-------|----------------|
| Fig [#] | [description] | [fix] |

### Missing Visuals

| Section | Concept | Suggested Visual |
|---------|---------|------------------|
| [section] | [concept] | [diagram type] |

## Real-World Applications

### Application References

| Section | Application | Industry | Current |
|---------|-------------|----------|---------|
| [section] | [app] | [industry] | Yes/No |

### Application Coverage

- Opening hook: Present/Missing
- Industry examples: [count]
- Research references: [count]
- Practical considerations: [count]

### Application Gaps

| Section | Missing Context | Suggestion |
|---------|-----------------|------------|
| [section] | [what's missing] | [application to add] |

## Interactive Elements

### Interactivity Inventory

| Type | Count | Sections |
|------|-------|----------|
| Inline questions | [n] | [list] |
| Mini-exercises | [n] | [list] |
| Lab activities | [n] | [list] |
| Reflection prompts | [n] | [list] |
| Code experiments | [n] | [list] |

### Interactivity Distribution

| Section | Elements | Engagement Level |
|---------|----------|------------------|
| [section] | [count] | High/Medium/Low |

### Interactivity Gaps

| Section | Issue | Suggestion |
|---------|-------|------------|
| [section] | No interactive elements | Add [type] |

## Callouts Analysis

### Callout Inventory

| Type | Count | Usage Appropriateness |
|------|-------|----------------------|
| Key Insight | [n] | Good/Overused/Underused |
| Warning | [n] | Good/Overused/Underused |
| Tip | [n] | Good/Overused/Underused |
| Note | [n] | Good/Overused/Underused |
| Try It | [n] | Good/Overused/Underused |

### Key Concepts Without Callouts

| Section | Concept | Suggested Callout |
|---------|---------|-------------------|
| [section] | [concept] | Key Insight |

## Engagement Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Examples per section | [avg] | 2+ | PASS/FAIL |
| Figures per chapter | [n] | 5+ | PASS/FAIL |
| Code-to-text ratio | [%] | 20-40% | PASS/FAIL |
| Interactive elements | [n] | 5+ | PASS/FAIL |
| Callouts | [n] | 3+ | PASS/FAIL |

## Strengths

1. [Specific engagement strength]
2. [Specific engagement strength]

## Priority Fixes

### Critical (Chapter Feels Dry)
1. [Action item]

### Major (Engagement Gap)
1. [Action item]

### Minor (Polish)
1. [Action item]

## Normalized Score

**Engagement Score: [total]/100 = [score]/10**

*Weight in overall review: 15%*
*Weighted contribution: [score * 0.15]*
```

## Scoring Rubric

### Example Quality (0-30)
| Score | Description |
|-------|-------------|
| 27-30 | Excellent, varied, illuminating examples throughout |
| 21-26 | Good examples with minor gaps |
| 14-20 | Adequate examples but lacking variety or depth |
| 7-13 | Few or poor quality examples |
| 0-6 | Examples missing or unhelpful |

### Visual Elements (0-25)
| Score | Description |
|-------|-------------|
| 23-25 | Rich visual support, all concepts illustrated |
| 18-22 | Good visuals with minor gaps |
| 12-17 | Some visual support |
| 6-11 | Minimal visuals |
| 0-5 | No visual elements |

### Real-World Applications (0-20)
| Score | Description |
|-------|-------------|
| 18-20 | Excellent practical context throughout |
| 14-17 | Good real-world connections |
| 9-13 | Some practical relevance |
| 4-8 | Limited real-world context |
| 0-3 | No practical applications |

### Interactive Elements (0-15)
| Score | Description |
|-------|-------------|
| 14-15 | Highly interactive, frequent engagement opportunities |
| 11-13 | Good interactivity |
| 7-10 | Some interactive elements |
| 3-6 | Minimal interactivity |
| 0-2 | No interactive elements |

### Callouts & Highlights (0-10)
| Score | Description |
|-------|-------------|
| 9-10 | Effective use of callouts for key concepts |
| 7-8 | Good callout usage |
| 4-6 | Some callouts present |
| 2-3 | Minimal highlighting |
| 0-1 | No callouts used |

## Best Practices

### Opening Hook Patterns
- Start with a problem the reader can relate to
- Show a surprising result
- Reference a famous robot or breakthrough
- Ask a compelling question

### Example Progression
```
Simple example (intuition)
    ↓
Worked example (mechanics)
    ↓
Variation (deeper understanding)
    ↓
Edge case (boundaries)
    ↓
Industry application (relevance)
```

## Integration

Called by:
- `/book.reviewer-agent` (primary)
- Directly for targeted engagement review

Returns:
```yaml
engagement_review:
  score: [0-100]
  normalized: [0-10]
  weighted: [score * 0.15]
  status: pass | needs_work | fail
  issues:
    critical: [count]
    major: [count]
    minor: [count]
  metrics:
    examples: [count]
    figures: [count]
    interactive: [count]
    callouts: [count]
```
