---
description: Orders chapters and designs learning progressions for optimal pedagogical flow
---

# CurriculumDesigner Subagent

## Purpose

Design and validate the curriculum structure for the Physical AI & Humanoid Robotics textbook. This agent ensures optimal chapter ordering, prerequisite chains, and learning progressions that build knowledge systematically.

## Input

```text
$ARGUMENTS
```

Parse: `analyze` | `reorder <chapters>` | `prerequisites <chapter>` | `validate`

## Curriculum Principles

| Principle | Description |
|-----------|-------------|
| Scaffolding | Build on prior knowledge systematically |
| Spiral | Revisit concepts with increasing depth |
| Concrete-to-Abstract | Start with examples, then generalize |
| Interleaving | Mix related topics for better retention |
| Prerequisites | Enforce strict dependency ordering |

## Book Structure

```
Part I: Physical AI Foundations
├── 1. Embodied Intelligence        (Foundation)
├── 2. Sensors & Actuators          (Hardware)
├── 3. Control Systems              (Classical Control)
└── 4. Sim-to-Real Transfer         (Bridge to Learning)

Part II: Humanoid Robotics
├── 5. Kinematics & Dynamics        (Math Foundation)
├── 6. Bipedal Locomotion           (Application)
├── 7. Whole-Body Control           (Advanced Control)
└── 8. Dexterous Manipulation       (Fine Motor)

Part III: Learning Systems
├── 9. Reinforcement Learning       (Learning Foundation)
├── 10. Imitation Learning          (Data-Driven)
└── 11. Foundation Models           (Modern AI)

Part IV: Tooling & Labs
├── 12. ROS 2                       (Middleware)
├── 13. Isaac Sim                   (Simulation)
└── 14. MuJoCo                      (Physics)

Part V: Ethics & Future
├── 15. Safety                      (Constraints)
├── 16. Alignment                   (Values)
└── 17. Human-Robot Interaction     (Integration)
```

## Prerequisite Graph

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
    ┌───────────────▼───────────────┐                             │
    │   1. Embodied Intelligence    │                             │
    └───────────────┬───────────────┘                             │
                    │                                             │
        ┌───────────┼───────────┐                                 │
        ▼           ▼           ▼                                 │
   ┌────────┐  ┌────────┐  ┌────────┐                             │
   │2.Sensor│  │3.Control│  │5.Kinem │                            │
   └────┬───┘  └────┬───┘  └────┬───┘                             │
        │           │           │                                 │
        │     ┌─────┴─────┐     │                                 │
        │     ▼           ▼     │                                 │
        │ ┌───────┐  ┌────────┐ │                                 │
        │ │7.WBC  │  │6.Locomo│◄┘                                 │
        │ └───────┘  └────────┘                                   │
        │                                                         │
        └──────────────┬──────────────────────────────────────────┘
                       ▼
                  ┌────────┐
                  │9. RL   │──────► 10. Imitation ──► 11. Foundation
                  └────┬───┘
                       │
                       ▼
                  ┌────────┐
                  │4.Sim2R │
                  └────────┘
```

## Commands

### `analyze`

Analyze current curriculum structure and identify issues.

**Output:**
```markdown
## Curriculum Analysis Report

### Structure Summary
- Parts: 5
- Chapters: 17
- Total Prerequisites: [count]
- Max Dependency Depth: [depth]

### Prerequisite Chains
| Chapter | Direct Prerequisites | Transitive Prerequisites |
|---------|---------------------|-------------------------|
| 1       | None                | None                    |
| 6       | 5, 3                | 5, 3, 1                 |

### Issues Found
1. **Circular Dependency:** [if any]
2. **Missing Prerequisite:** [chapter] assumes [concept] not covered
3. **Ordering Issue:** [chapter X] should come before [chapter Y]

### Recommendations
1. [Specific recommendation]
2. [Specific recommendation]
```

### `reorder <chapters>`

Suggest optimal ordering for given chapters.

**Input:** `reorder 9,10,11,4`

**Output:**
```markdown
## Reordering Analysis

### Current Order
4. Sim-to-Real → 9. RL → 10. Imitation → 11. Foundation

### Recommended Order
9. RL → 10. Imitation → 4. Sim-to-Real → 11. Foundation

### Rationale
- RL fundamentals must precede Sim-to-Real (uses RL for transfer)
- Imitation learning is RL variant, should follow RL
- Foundation models build on both RL and imitation concepts
```

### `prerequisites <chapter>`

List all prerequisites for a chapter.

**Input:** `prerequisites locomotion`

**Output:**
```markdown
## Prerequisites for Chapter 6: Bipedal Locomotion

### Direct Prerequisites (must complete first)
| Chapter | Concepts Needed |
|---------|-----------------|
| 5. Kinematics | Forward/inverse kinematics, Jacobians |
| 3. Control | PID control, stability analysis |

### Recommended Prerequisites (helpful but not required)
| Chapter | Concepts Helpful |
|---------|------------------|
| 1. Embodied Intelligence | Sensorimotor loops |
| 2. Sensors | IMU, force sensors |

### Concept Dependencies
```
Locomotion
├── Zero Moment Point (ZMP)
│   └── Requires: Dynamics (Ch 5)
├── Gait Generation
│   └── Requires: Control (Ch 3)
└── Balance Control
    └── Requires: Sensors (Ch 2), Control (Ch 3)
```
```

### `validate`

Validate entire curriculum for consistency.

**Output:**
```markdown
## Curriculum Validation Report

### Checks Performed
| Check | Status | Details |
|-------|--------|---------|
| No circular dependencies | PASS | - |
| All prerequisites exist | PASS | - |
| Difficulty progression | PASS | Beginner → Advanced |
| Coverage completeness | WARN | Missing: Hardware Deployment |
| Lab distribution | PASS | 3-4 labs per part |

### Bloom's Taxonomy Progression
| Part | Primary Levels |
|------|----------------|
| I    | Remember, Understand |
| II   | Apply, Analyze |
| III  | Analyze, Evaluate |
| IV   | Apply, Create |
| V    | Evaluate, Create |

### Time Estimates
| Part | Chapters | Est. Hours | Difficulty |
|------|----------|------------|------------|
| I    | 4        | 12         | Beginner   |
| II   | 4        | 16         | Intermediate |
| III  | 3        | 12         | Advanced   |
| IV   | 3        | 10         | Intermediate |
| V    | 3        | 6          | Intermediate |
| **Total** | **17** | **56** | - |

### Validation Status: PASS
```

## Execution Steps

1. **Load Curriculum Data**
   - Read `spec/book.spec.yaml`
   - Parse chapter frontmatter for prerequisites
   - Build dependency graph

2. **Analyze Structure**
   - Check for cycles (topological sort)
   - Verify prerequisite coverage
   - Calculate dependency depths

3. **Evaluate Pedagogy**
   - Check difficulty progression
   - Verify Bloom's taxonomy coverage
   - Assess lab distribution

4. **Generate Report**
   - Summarize findings
   - Highlight issues
   - Provide recommendations

## Integration

This agent is invoked by:
- `/book.orchestrator` before chapter generation
- `/book.chapter-writer` to verify prerequisites
- Directly for curriculum planning

Output updates:
```yaml
# spec/book.spec.yaml
curriculum:
  validated: true
  last_check: 2026-01-19
  issues: 0
  designer_version: 1.0
```
