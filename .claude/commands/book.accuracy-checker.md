---
description: Validates technical accuracy of robotics and AI content, formulas, and factual claims
---

# Accuracy Checker Subagent

## Purpose

Verify the technical accuracy of chapter content including factual claims, mathematical formulas, algorithm descriptions, and domain-specific concepts in Physical AI and Humanoid Robotics.

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
| Factual Correctness | Claims align with established knowledge | 30 |
| Mathematical Accuracy | Formulas and equations are correct | 25 |
| Algorithm Correctness | Algorithms described accurately | 20 |
| Currency | Information is up-to-date | 15 |
| Consistency | No internal contradictions | 10 |

## Domain Knowledge Areas

### Physical AI Foundations
- Embodied cognition principles
- Sensor specifications and limitations
- Control theory fundamentals
- Sim-to-real transfer methods

### Humanoid Robotics
- Kinematics (forward/inverse)
- Dynamics equations (Euler-Lagrange, Newton-Euler)
- Gait patterns and stability criteria
- Manipulation frameworks (task space, joint space)

### Learning Systems
- RL algorithms (PPO, SAC, TD3)
- Imitation learning methods
- Foundation model architectures
- Training procedures and hyperparameters

### Tooling
- ROS 2 concepts and APIs
- Simulation physics (MuJoCo, Isaac Sim)
- Hardware interfaces

## Checklist

### Factual Claims
- [ ] Historical facts verified (dates, names, events)
- [ ] Technical specifications accurate (sensor ranges, motor capabilities)
- [ ] Algorithm attributions correct (original authors, papers)
- [ ] Platform details current (software versions, APIs)
- [ ] No exaggerated or misleading claims

### Mathematical Content
- [ ] Equations dimensionally consistent
- [ ] Variable definitions provided
- [ ] Matrix operations valid (dimensions match)
- [ ] Derivatives and integrals correct
- [ ] Numerical values reasonable

### Algorithms
- [ ] Pseudocode matches description
- [ ] Complexity analysis correct (Big-O)
- [ ] Edge cases acknowledged
- [ ] Assumptions stated explicitly
- [ ] Standard algorithms named correctly

### Currency
- [ ] No deprecated APIs or methods
- [ ] Hardware references current
- [ ] Software versions specified
- [ ] Recent developments acknowledged
- [ ] Outdated techniques labeled as such

## Execution Steps

1. **Load Chapter**
   ```
   Read chapter markdown file
   Extract all technical claims
   Extract mathematical content
   Extract algorithm descriptions
   ```

2. **Claim Extraction**
   ```
   Identify factual statements
   Identify numerical values
   Identify attributions
   Flag uncertain claims
   ```

3. **Mathematical Verification**
   ```
   Parse LaTeX formulas
   Check dimensional consistency
   Verify standard equations
   Validate numerical examples
   ```

4. **Algorithm Review**
   ```
   Extract algorithm descriptions
   Compare to authoritative sources
   Verify complexity claims
   Check implementation correctness
   ```

5. **Currency Check**
   ```
   Identify technology references
   Check version currency
   Flag deprecated methods
   Note recent alternatives
   ```

6. **Cross-Reference**
   ```
   Check internal consistency
   Verify chapter cross-references
   Compare to spec definitions
   ```

7. **Generate Report**

## Output Format

```markdown
# Accuracy Review Report

## Overview

| Attribute | Value |
|-----------|-------|
| Chapter | [Title] |
| Path | [file path] |
| Reviewed | [ISO date] |
| Reviewer | AccuracyChecker v1.0 |

## Score Summary

| Criterion | Score | Max | Notes |
|-----------|-------|-----|-------|
| Factual Correctness | [x] | 30 | [note] |
| Mathematical Accuracy | [x] | 25 | [note] |
| Algorithm Correctness | [x] | 20 | [note] |
| Currency | [x] | 15 | [note] |
| Consistency | [x] | 10 | [note] |
| **Total** | **[sum]** | **100** | |

## Factual Issues

### Errors (Must Fix)

| Line | Claim | Issue | Correct Information |
|------|-------|-------|---------------------|
| [#] | "[claim]" | [what's wrong] | [correct fact] |

### Unverifiable Claims

| Line | Claim | Recommendation |
|------|-------|----------------|
| [#] | "[claim]" | Add citation or rephrase |

## Mathematical Issues

### Formula Errors

| Line | Formula | Issue | Correction |
|------|---------|-------|------------|
| [#] | $formula$ | [error type] | $corrected$ |

### Dimensional Inconsistencies

| Line | Expression | Left Units | Right Units |
|------|------------|------------|-------------|
| [#] | [expr] | [units] | [units] |

### Missing Definitions

| Line | Variable | Context |
|------|----------|---------|
| [#] | [var] | Used without definition |

## Algorithm Issues

### Description Errors

| Line | Algorithm | Issue | Correct Description |
|------|-----------|-------|---------------------|
| [#] | [name] | [error] | [correction] |

### Complexity Errors

| Line | Claim | Actual Complexity |
|------|-------|-------------------|
| [#] | [claimed] | [correct] |

## Currency Issues

### Deprecated Content

| Line | Reference | Issue | Modern Alternative |
|------|-----------|-------|-------------------|
| [#] | [ref] | Deprecated in [year] | [alternative] |

### Outdated Information

| Line | Topic | Current State |
|------|-------|---------------|
| [#] | [topic] | [update needed] |

## Consistency Issues

| Location 1 | Location 2 | Contradiction |
|------------|------------|---------------|
| Line [#] | Line [#] | [description] |

## Verification Summary

| Category | Verified | Unverifiable | Errors |
|----------|----------|--------------|--------|
| Facts | [n] | [n] | [n] |
| Formulas | [n] | - | [n] |
| Algorithms | [n] | - | [n] |

## Strengths

1. [Specific accuracy strength]
2. [Specific accuracy strength]

## Priority Fixes

### Critical (Factually Wrong)
1. [Action item with line reference]

### Major (Misleading or Outdated)
1. [Action item with line reference]

### Minor (Imprecise)
1. [Action item with line reference]

## Normalized Score

**Accuracy Score: [total]/100 = [score]/10**

*Weight in overall review: 25%*
*Weighted contribution: [score * 0.25]*
```

## Scoring Rubric

### Factual Correctness (0-30)
| Score | Description |
|-------|-------------|
| 27-30 | All claims verified, no errors |
| 21-26 | Minor inaccuracies only |
| 14-20 | Some significant errors |
| 7-13 | Multiple factual errors |
| 0-6 | Fundamentally incorrect content |

### Mathematical Accuracy (0-25)
| Score | Description |
|-------|-------------|
| 23-25 | All formulas correct and complete |
| 18-22 | Minor notation issues only |
| 12-17 | Some formula errors |
| 6-11 | Multiple mathematical errors |
| 0-5 | Pervasive mathematical problems |

### Algorithm Correctness (0-20)
| Score | Description |
|-------|-------------|
| 18-20 | Algorithms perfectly described |
| 14-17 | Minor algorithmic issues |
| 9-13 | Some algorithm errors |
| 4-8 | Significant algorithm problems |
| 0-3 | Algorithms incorrectly presented |

### Currency (0-15)
| Score | Description |
|-------|-------------|
| 14-15 | All information current |
| 11-13 | Minor outdated references |
| 7-10 | Some deprecated content |
| 3-6 | Significantly outdated |
| 0-2 | Obsolete information throughout |

### Consistency (0-10)
| Score | Description |
|-------|-------------|
| 9-10 | Perfectly consistent |
| 7-8 | Minor inconsistencies |
| 4-6 | Some contradictions |
| 2-3 | Frequent contradictions |
| 0-1 | Internally contradictory |

## Common Issues Database

### Robotics Fact Checks
- DOF counts for standard platforms
- Standard sensor specifications
- Common algorithm attributions
- Historical milestones

### Formula Library
- Forward kinematics (DH parameters)
- Inverse dynamics (recursive Newton-Euler)
- Control laws (PID, LQR)
- RL objectives (policy gradient, value functions)

## Integration

Called by:
- `/book.reviewer-agent` (primary)
- Directly for targeted accuracy review

Returns:
```yaml
accuracy_review:
  score: [0-100]
  normalized: [0-10]
  weighted: [score * 0.25]
  status: pass | needs_work | fail
  issues:
    critical: [count]
    major: [count]
    minor: [count]
  unverifiable_claims: [count]
```
