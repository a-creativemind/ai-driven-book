---
id: 002
title: Generate Spec-Kit YAML Schemas
stage: general
date: 2026-01-19
surface: agent
model: claude-opus-4-5
feature: none
branch: master
user: Amena
command: none
labels: ["schemas", "spec-kit", "yaml", "validation"]
links:
  spec: PROJECT_SPEC.md
  ticket: null
  adr: null
  pr: null
files:
  - spec/book.spec.yaml
  - spec/chapter.schema.yaml
  - spec/learning-objectives.schema.yaml
tests:
  - none
---

## Prompt

User selected option "1" to generate the Spec-Kit YAML schemas from the available next steps after creating PROJECT_SPEC.md.

## Response snapshot

Created three comprehensive YAML schema files:

### 1. `spec/book.spec.yaml`
- Book metadata (title, authors, license, keywords)
- Target audience and prerequisites
- Complete structure definition with 5 parts and 17 chapters
- Validation rules configuration
- Code validation settings
- Build configuration for Docusaurus
- RAG chatbot integration settings
- Internationalization (i18n) configuration

### 2. `spec/chapter.schema.yaml`
- JSON Schema 2020-12 compliant
- Required fields: title, chapter_id, part_id, difficulty, learning_objectives, core_concepts, examples, summary, references
- Reusable definitions ($defs) for:
  - learning_objective (with Bloom's levels)
  - concept (with formulas support)
  - figure (with accessibility requirements)
  - example (with code support)
  - exercise (with hints and solutions)
  - reference (academic citations)
  - resource (further reading)
- Validation examples included

### 3. `spec/learning-objectives.schema.yaml`
- Complete Bloom's Taxonomy integration
- ABCD method for writing objectives
- Action verbs reference for each cognitive level
- Writing guidelines with good/poor examples
- Validation rules for objectives
- Example objectives for embodiment and kinematics chapters

## Outcome

- ✅ Impact: Established validation foundation for all textbook content
- 🧪 Tests: Schemas include validation examples
- 📁 Files: 3 files created in spec/ directory
- 🔁 Next prompts: Initialize Docusaurus project, Write Chapter 1, Create subagents, Build RAG backend
- 🧠 Reflection: Schemas are comprehensive with instructional design best practices (Bloom's Taxonomy, ABCD method) built in

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A - schemas define validation rules
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Test schema validation against sample chapter content
