---
id: 001
title: Write Chapter 1 - Embodied Intelligence
stage: spec
date: 2026-01-19
surface: agent
model: claude-opus-4-5
feature: physical-ai
branch: master
user: Amena
command: none
labels: ["chapter", "content", "embodiment", "physical-ai"]
links:
  spec: spec/chapter.schema.yaml
  ticket: null
  adr: null
  pr: null
files:
  - docs/physical-ai/embodiment.md
tests:
  - Validates against chapter.schema.yaml
  - Contains 5 learning objectives with Bloom's levels
  - Contains 4 Python code examples
  - Contains 7 academic references
  - Contains 3 exercises
---

## Prompt

User selected option "1" to write Chapter 1 (Embodied Intelligence) with full spec-compliant content after initializing the Docusaurus project structure.

## Response snapshot

Created comprehensive Chapter 1 (~580 lines) covering Embodied Intelligence with:

### Structure (chapter.schema.yaml compliant)
- **Frontmatter**: chapter_id, part_id, difficulty, author, last_updated, prerequisites, tags
- **Learning Objectives**: 5 objectives with Bloom's Taxonomy levels (Understand, Remember, Analyze, Evaluate, Apply)
- **7 Main Sections**: Definition, History, Sensorimotor Loop, Morphological Computation, Nature Examples, Robotics Implications, Advantages/Limitations

### Content Highlights
1. **What is Embodied Intelligence?**
   - Core definition and key insight (gaze heuristic example)
   - Comparison table: Embodied vs Disembodied AI

2. **Historical Context**
   - Symbolic AI era and its limitations
   - Brooks' behavioral turn
   - Embodied cognition movement

3. **Sensorimotor Loop**
   - ASCII diagram of loop structure
   - Python example: PassiveWalkerLeg simulation

4. **Morphological Computation**
   - Python example: RigidGripper vs SoftGripper comparison
   - Trade-off table

5. **Nature Examples**
   - Ant navigation, Octopus arm control, Human locomotion

6. **Robotics Design Principles**
   - Design for niche (Python example)
   - Exploit environment, Match body to task
   - Robust controller example

7. **Summary & Exercises**
   - 6 key takeaways
   - 3 exercises mapped to learning objectives

### References
- 7 academic citations (Pfeifer & Bongard, Brooks, Clark, etc.)
- 3 further reading links

## Outcome

- ✅ Impact: First complete chapter ready for Docusaurus build and review
- 🧪 Tests: Content validates against chapter.schema.yaml requirements
- 📁 Files: 1 file modified (docs/physical-ai/embodiment.md)
- 🔁 Next prompts: Write Chapter 2 (Sensors & Actuators), Create subagents, Build RAG backend
- 🧠 Reflection: Chapter provides solid theoretical foundation with practical code examples demonstrating each concept

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Schema compliance - PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Add interactive diagrams or visualizations for sensorimotor loop
