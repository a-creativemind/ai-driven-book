# Physical AI & Humanoid Robotics Textbook

## Spec-Driven Unified Book + RAG System Project

---

## 1. Project Overview

This project requires building a spec-driven, AI-authored textbook titled:

**Physical AI & Humanoid Robotics: From Foundations to Embodied Intelligence**

The book must be:

- Written using **Claude Code**
- Structured and validated using **Spec-Kit Plus**
- Built with **Docusaurus v3**
- Deployed on **GitHub Pages**

In addition, the project must embed a **Retrieval-Augmented Generation (RAG) chatbot** capable of answering questions about the book's content, including user-selected text only.

---

## 2. Mandatory Core Requirements (100 Points)

### 2.1 AI / Spec-Driven Book Creation

#### Required Tools

| Tool | Purpose |
|------|---------|
| Claude Code | Content authoring, refactoring, agents |
| Spec-Kit Plus | Content specs, validation, automation |
| Docusaurus v3 | Static site generation |
| GitHub Pages | Deployment |

#### Constraints

- All book content must be generated or refactored using Claude Code
- All chapters must conform to Spec-Kit Plus schemas
- No ad-hoc markdown structure is allowed outside the defined specs

---

### 2.2 Repository Structure (Required)

```
physical-ai-textbook/
├── docs/
│   ├── intro.md
│   ├── physical-ai/
│   │   ├── embodiment.md
│   │   ├── sensors-actuators.md
│   │   ├── control-systems.md
│   │   └── sim2real.md
│   ├── humanoid-robotics/
│   │   ├── kinematics.md
│   │   ├── locomotion.md
│   │   ├── manipulation.md
│   │   └── perception.md
│   ├── ai-systems/
│   │   ├── rl.md
│   │   ├── imitation-learning.md
│   │   └── foundation-models.md
│   ├── labs/
│   │   ├── isaac-sim.md
│   │   ├── mujoco.md
│   │   └── ros2.md
│   └── ethics-future.md
│
├── spec/
│   ├── book.spec.yaml
│   ├── chapter.schema.yaml
│   └── learning-objectives.schema.yaml
│
├── chatbot/
│   ├── api/
│   │   └── main.py
│   ├── ingest/
│   └── embeddings/
│
├── website/
└── docusaurus.config.ts
```

---

### 2.3 Spec-Kit Plus Usage (Required)

Spec-Kit Plus must enforce:

- Chapter structure
- Learning objectives
- Code block validation
- Terminology consistency
- Auto-generated table of contents

All chapters must comply with:

```yaml
required:
  - title
  - learning_objectives
  - core_concepts
  - diagrams_or_figures
  - examples
  - summary
  - references
```

---

## 3. Textbook Curriculum (Required)

### Part I – Physical AI Foundations

| Chapter | Topic |
|---------|-------|
| 1.1 | Embodied Intelligence |
| 1.2 | Sensors & Actuators |
| 1.3 | Control Systems |
| 1.4 | Real-Time & Feedback Systems |

### Part II – Humanoid Robotics

| Chapter | Topic |
|---------|-------|
| 2.1 | Kinematics & Dynamics |
| 2.2 | Bipedal Locomotion |
| 2.3 | Whole-Body Control |
| 2.4 | Dexterous Manipulation |

### Part III – Learning Systems

| Chapter | Topic |
|---------|-------|
| 3.1 | Reinforcement Learning |
| 3.2 | Imitation Learning |
| 3.3 | Sim-to-Real Transfer |
| 3.4 | Foundation Models for Robotics |

### Part IV – Tooling & Labs

| Chapter | Topic |
|---------|-------|
| 4.1 | ROS 2 |
| 4.2 | Isaac Sim |
| 4.3 | MuJoCo |
| 4.4 | Hardware Deployment |

### Part V – Ethics & Future

| Chapter | Topic |
|---------|-------|
| 5.1 | Safety |
| 5.2 | Alignment |
| 5.3 | Human-Robot Interaction |

---

## 4. Integrated RAG Chatbot (Required)

### 4.1 Architecture

```
Docusaurus Frontend
        ↓
Chat UI (OpenAI ChatKit)
        ↓
FastAPI Backend
        ↓
OpenAI Agents SDK
        ↓
Qdrant Cloud (Vector Store)
        ↓
Neon Serverless Postgres (Metadata)
```

### 4.2 Mandatory Chatbot Capabilities

- [ ] Answer questions about the **entire book**
- [ ] Answer questions using only **user-selected text**
- [ ] Restrict context strictly to retrieved passages
- [ ] Be aware of **chapter and section metadata**

### 4.3 RAG Ingestion Rules

| Step | Description |
|------|-------------|
| 1 | Markdown → chunked embeddings |
| 2 | Store embeddings in Qdrant Cloud Free Tier |
| 3 | Store metadata in Neon Serverless Postgres |

Each chunk must include:

```json
{
  "chapter": "string",
  "section": "string",
  "difficulty_level": "beginner | intermediate | advanced"
}
```

---

## 5. Bonus Features

### 5.1 Bonus #1: Reusable Intelligence (+50 Points)

Create reusable **Claude Code Subagents**, documented in the project.

#### Required Subagents

| Subagent | Role |
|----------|------|
| `RoboticsProfessor` | Writes theory sections |
| `LabInstructor` | Generates hands-on labs |
| `MathExplainer` | Simplifies equations |
| `CurriculumDesigner` | Orders chapters |
| `ReviewerAgent` | Validates clarity and pedagogy |

**Constraint:** Subagents must be reused across chapters.

---

### 5.2 Bonus #2: Signup & Signin (+50 Points)

#### Auth Stack

| Component | Technology |
|-----------|------------|
| Auth Library | Better Auth |
| Sessions | JWT-based |
| Database | Neon Postgres |
| Providers | OAuth + Email |

#### Required Signup Questions

1. Programming experience (Beginner → Advanced)
2. Robotics experience
3. Math background
4. Hardware access (GPU, simulator, robot)

#### User Profile Schema

```json
{
  "programming": "beginner | intermediate | advanced",
  "robotics": "beginner | intermediate | advanced",
  "math": "beginner | intermediate | advanced",
  "hardware": ["gpu", "simulator", "robot"]
}
```

---

### 5.3 Bonus #3: Personalized Chapters (+50 Points)

Each chapter must include a button:

```
🎯 Personalize This Chapter
```

#### Behavior

- Content rewritten dynamically using an agent
- Depth and examples adjusted to user profile

| Profile Level | Output Style |
|---------------|--------------|
| Beginner | Intuitive explanations, analogies, visuals |
| Advanced | Equations, code examples, technical depth |

---

### 5.4 Bonus #4: Urdu Translation (+50 Points)

Each chapter must include a button:

```
🌐 Translate to Urdu
```

#### Requirements

- [ ] Uses OpenAI translation agent
- [ ] RTL UI support
- [ ] Cached translations in database
- [ ] Per-chapter translation

---

## 6. Deployment Requirements

| Component | Platform |
|-----------|----------|
| Book | GitHub Pages |
| Backend | Vercel / Fly.io |
| Database | Neon Serverless Postgres |
| Vector DB | Qdrant Cloud Free Tier |

---

## 7. Acceptance Criteria

The project is considered complete when:

- [ ] The book is publicly accessible on GitHub Pages
- [ ] All chapters pass Spec-Kit validation
- [ ] RAG chatbot answers book-based questions correctly
- [ ] User-selected text Q&A works correctly
- [ ] Bonus features function as described (if implemented)

---

## 8. Scoring Summary

| Feature | Points |
|---------|--------|
| AI-authored textbook | 100 |
| RAG chatbot | Required |
| Claude subagents | +50 |
| Auth & profiling | +50 |
| Personalized chapters | +50 |
| Urdu translation | +50 |
| **Maximum Score** | **300** |

---

## 9. Instructions for Claude Code

```
Follow this specification strictly.
Do not generate content that violates the schemas.
Prefer reusable agents, modular specs, and pedagogically sound explanations.
Treat this document as the authoritative source of truth.
```

---

## 10. Implementation Phases

### Phase 1: Project Setup
1. Initialize Docusaurus v3 project
2. Configure Spec-Kit Plus schemas
3. Set up repository structure
4. Configure GitHub Pages deployment

### Phase 2: Content Creation
1. Create chapter schemas
2. Generate all chapter content using Claude Code
3. Validate content against schemas
4. Build table of contents

### Phase 3: RAG Chatbot
1. Set up FastAPI backend
2. Configure Qdrant Cloud
3. Set up Neon Postgres
4. Implement ingestion pipeline
5. Build chat UI component

### Phase 4: Bonus Features (Optional)
1. Create Claude Code subagents
2. Implement authentication system
3. Build personalization feature
4. Add Urdu translation support

### Phase 5: Deployment & Testing
1. Deploy to GitHub Pages
2. Deploy backend services
3. End-to-end testing
4. Performance optimization

---

## 11. Technical Dependencies

### Frontend
```json
{
  "docusaurus": "^3.0.0",
  "react": "^18.0.0",
  "typescript": "^5.0.0"
}
```

### Backend
```python
# requirements.txt
fastapi>=0.100.0
uvicorn>=0.23.0
openai>=1.0.0
qdrant-client>=1.6.0
psycopg2-binary>=2.9.0
python-jose>=3.3.0
```

### Infrastructure
- Node.js >= 18
- Python >= 3.11
- PostgreSQL (Neon)
- Qdrant Cloud

---

## 12. Environment Variables

```env
# OpenAI
OPENAI_API_KEY=

# Qdrant
QDRANT_URL=
QDRANT_API_KEY=

# Neon Postgres
DATABASE_URL=

# Auth (if implementing bonus)
JWT_SECRET=
OAUTH_CLIENT_ID=
OAUTH_CLIENT_SECRET=
```

---

## 13. API Endpoints (Chatbot)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send message to chatbot |
| POST | `/api/chat/selection` | Query with selected text |
| GET | `/api/chapters` | List all chapters |
| GET | `/api/chapters/{id}` | Get chapter metadata |
| POST | `/api/ingest` | Trigger content ingestion |

---

## 14. Quality Gates

### Content Quality
- [ ] All chapters have learning objectives
- [ ] All code examples are runnable
- [ ] All diagrams have alt text
- [ ] References are properly cited

### Technical Quality
- [ ] Lighthouse score >= 90
- [ ] All API endpoints tested
- [ ] Error handling implemented
- [ ] Security best practices followed

### Documentation
- [ ] README.md complete
- [ ] API documentation available
- [ ] Deployment guide included
- [ ] Contributing guidelines present

---

## 15. Glossary

| Term | Definition |
|------|------------|
| Physical AI | AI systems that interact with the physical world |
| RAG | Retrieval-Augmented Generation |
| Sim2Real | Simulation to Reality transfer |
| Spec-Kit Plus | Specification-driven development toolkit |
| Claude Code | AI coding assistant by Anthropic |

---

## Appendix A: Chapter Schema

```yaml
# chapter.schema.yaml
type: object
required:
  - title
  - learning_objectives
  - core_concepts
  - examples
  - summary
  - references
properties:
  title:
    type: string
    minLength: 5
    maxLength: 100
  learning_objectives:
    type: array
    minItems: 3
    items:
      type: string
  core_concepts:
    type: array
    minItems: 1
    items:
      type: object
      required:
        - name
        - description
      properties:
        name:
          type: string
        description:
          type: string
  diagrams_or_figures:
    type: array
    items:
      type: object
      required:
        - caption
        - alt_text
      properties:
        caption:
          type: string
        alt_text:
          type: string
        path:
          type: string
  examples:
    type: array
    minItems: 1
    items:
      type: object
      required:
        - title
        - content
      properties:
        title:
          type: string
        content:
          type: string
        code:
          type: string
        language:
          type: string
  summary:
    type: string
    minLength: 100
  references:
    type: array
    items:
      type: string
  difficulty:
    type: string
    enum: [beginner, intermediate, advanced]
  prerequisites:
    type: array
    items:
      type: string
```

---

## Appendix B: Book Spec

```yaml
# book.spec.yaml
title: "Physical AI & Humanoid Robotics: From Foundations to Embodied Intelligence"
version: "1.0.0"
authors:
  - name: "AI-Generated"
    tool: "Claude Code"
structure:
  parts:
    - id: physical-ai
      title: "Physical AI Foundations"
      chapters:
        - embodiment
        - sensors-actuators
        - control-systems
        - sim2real
    - id: humanoid-robotics
      title: "Humanoid Robotics"
      chapters:
        - kinematics
        - locomotion
        - manipulation
        - perception
    - id: ai-systems
      title: "Learning Systems"
      chapters:
        - rl
        - imitation-learning
        - foundation-models
    - id: labs
      title: "Tooling & Labs"
      chapters:
        - isaac-sim
        - mujoco
        - ros2
    - id: ethics
      title: "Ethics & Future"
      chapters:
        - ethics-future
validation:
  schema: chapter.schema.yaml
  strict: true
```

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-19
**Status:** Ready for Implementation
