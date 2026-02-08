---
description: Specialized agent for writing spec-compliant textbook chapters with proper structure, learning objectives, and code examples
---

## Input

```text
$ARGUMENTS
```

Parse: `<chapter-number> <chapter-title>` or `<chapter-slug>`

## Task

Write a complete textbook chapter for "Physical AI & Humanoid Robotics".

## Steps

1. **Load Schema**
   ```
   Read: spec/chapter.schema.yaml
   Read: spec/learning-objectives.schema.yaml
   ```

2. **Determine Chapter Details**
   - Map chapter number to part folder and prerequisites
   - Part I (1-4): `docs/physical-ai/`
   - Part II (5-8): `docs/humanoid-robotics/`
   - Part III (9-11): `docs/learning-systems/`
   - Part IV (12-14): `docs/integration/`
   - Part V (15-17): `docs/deployment/`

3. **Generate Chapter Content**

   Required elements:
   | Element | Requirement |
   |---------|-------------|
   | Frontmatter | All YAML fields from schema |
   | Learning Objectives | 5 objectives, Bloom's taxonomy table |
   | Sections | 5-8 main sections |
   | Code Examples | 10+ Python with outputs |
   | Math | LaTeX formulas where needed |
   | Summary | Key takeaways + connections |
   | Exercises | 3 with varying difficulty |
   | References | 7+ academic citations |

4. **Write File**
   ```
   Output: docs/<part-folder>/<chapter-slug>.md
   Target: 800-1500 lines
   ```

5. **Validate**
   - No placeholder text
   - All code has output blocks
   - Learning objectives use action verbs
   - References are properly formatted

## Output

Report:
- File path
- Line count
- Code example count
- Validation status
