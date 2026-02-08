---
description: Orchestrates book subagents to generate, validate, and refine complete textbook chapters
---

## Input

```text
$ARGUMENTS
```

Parse: `generate <chapter>` | `validate <chapter>` | `full <chapter>`

## Subagents

| Agent | Command | Purpose |
|-------|---------|---------|
| Chapter Writer | `/book.chapter-writer` | Generate chapter content |
| Code Validator | `/book.code-validator` | Validate Python examples |
| Reference Checker | `/book.reference-checker` | Validate citations |

## Workflows

### `generate <chapter>`
1. Run `/book.chapter-writer <chapter>`
2. Report generation status

### `validate <chapter>`
1. Run `/book.code-validator <chapter-path>`
2. Run `/book.reference-checker <chapter-path>`
3. Aggregate results

### `full <chapter>` (Complete Pipeline)

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. GENERATE                                            │
│     └─> /book.chapter-writer <chapter>                  │
│         └─> docs/<part>/<chapter>.md                    │
│                                                          │
│  2. VALIDATE CODE                                       │
│     └─> /book.code-validator <chapter-path>             │
│         ├─> Syntax check                                │
│         ├─> Style check                                 │
│         └─> Output verification                         │
│                                                          │
│  3. VALIDATE REFERENCES                                 │
│     └─> /book.reference-checker <chapter-path>          │
│         ├─> DOI validation                              │
│         ├─> Citation check                              │
│         └─> Enrichment suggestions                      │
│                                                          │
│  4. FIX ISSUES (if any)                                 │
│     └─> Auto-fix or report for manual review            │
│                                                          │
│  5. FINAL REPORT                                        │
│     └─> Quality score and status                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Execution Steps

1. **Parse Command**
   ```
   Input: "full kinematics"
   Action: generate + validate
   Chapter: Chapter 5 - Kinematics & Dynamics
   Path: docs/humanoid-robotics/kinematics.md
   ```

2. **Run Chapter Writer**
   - Invoke `/book.chapter-writer 5 Kinematics and Dynamics`
   - Wait for completion
   - Capture output path and stats

3. **Run Code Validator**
   - Invoke `/book.code-validator docs/humanoid-robotics/kinematics.md`
   - Collect issues

4. **Run Reference Checker**
   - Invoke `/book.reference-checker docs/humanoid-robotics/kinematics.md`
   - Collect issues

5. **Issue Resolution**
   - If code syntax errors: auto-fix and re-validate
   - If missing docstrings: add them
   - If reference issues: suggest corrections

6. **Generate Report**

## Output Format

```
═══════════════════════════════════════════════════════════
                 CHAPTER GENERATION REPORT
═══════════════════════════════════════════════════════════

Chapter: 5 - Kinematics and Dynamics
Path: docs/humanoid-robotics/kinematics.md
Generated: 2026-01-19

───────────────────────────────────────────────────────────
                      GENERATION
───────────────────────────────────────────────────────────
Status: SUCCESS
Lines: 1,245
Sections: 7
Code Examples: 12
Learning Objectives: 5

───────────────────────────────────────────────────────────
                    CODE VALIDATION
───────────────────────────────────────────────────────────
Syntax:     12/12 PASS
Docstrings: 12/12 PASS
Type Hints: 10/12 WARN (2 missing)
Score: 95%

───────────────────────────────────────────────────────────
                  REFERENCE VALIDATION
───────────────────────────────────────────────────────────
References: 7
Valid DOIs: 6/7
Citations:  All matched
Score: 95%

───────────────────────────────────────────────────────────
                      OVERALL
───────────────────────────────────────────────────────────
Quality Score: 95/100
Status: READY FOR REVIEW

Warnings:
- 2 functions missing type hints (non-blocking)
- 1 reference missing DOI (non-blocking)

═══════════════════════════════════════════════════════════
```

## Chapter Mapping

| Input | Chapter | Part | Output Path |
|-------|---------|------|-------------|
| 1, embodiment | Embodied Intelligence | I | physical-ai/embodiment.md |
| 2, sensors | Sensors & Actuators | I | physical-ai/sensors-actuators.md |
| 3, control | Control Systems | I | physical-ai/control-systems.md |
| 4, sim2real | Sim-to-Real Transfer | I | physical-ai/sim2real.md |
| 5, kinematics | Kinematics & Dynamics | II | humanoid-robotics/kinematics.md |
| 6, locomotion | Locomotion | II | humanoid-robotics/locomotion.md |
| 7, manipulation | Manipulation | II | humanoid-robotics/manipulation.md |
| 8, hri | Human-Robot Interaction | II | humanoid-robotics/hri.md |
| 9, imitation | Imitation Learning | III | learning-systems/imitation.md |
| 10, rl | Reinforcement Learning | III | learning-systems/rl.md |
| 11, foundation | Foundation Models | III | learning-systems/foundation-models.md |
| 12, perception | Perception Pipeline | IV | integration/perception.md |
| 13, planning | Planning & Navigation | IV | integration/planning.md |
| 14, safety | Safety & Ethics | IV | integration/safety.md |
| 15, platforms | Robot Platforms | V | deployment/platforms.md |
| 16, edge | Edge Deployment | V | deployment/edge.md |
| 17, future | Future Directions | V | deployment/future.md |
