# Physical AI & Humanoid Robotics Textbook

> From Foundations to Embodied Intelligence

An AI-authored textbook built with Docusaurus v3, Claude Code, and Spec-Kit Plus.

## Overview

This comprehensive textbook covers:

- **Part I**: Physical AI Foundations (Embodiment, Sensors, Control, Sim2Real)
- **Part II**: Humanoid Robotics (Kinematics, Locomotion, Manipulation, Perception)
- **Part III**: Learning Systems (RL, Imitation Learning, Foundation Models)
- **Part IV**: Tooling & Labs (ROS 2, Isaac Sim, MuJoCo)
- **Part V**: Ethics & Future (Safety, Alignment, HRI)

## Features

- AI-generated content validated against Spec-Kit Plus schemas
- Integrated RAG chatbot for Q&A
- Personalized chapter content based on user profile
- Urdu translation support
- Hands-on lab exercises

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## Project Structure

```
physical-ai-textbook/
├── docs/                 # Textbook content
├── spec/                 # Spec-Kit Plus schemas
├── chatbot/              # RAG chatbot backend
├── src/                  # React components & CSS
├── static/               # Static assets
└── docusaurus.config.ts  # Docusaurus configuration
```

## Tech Stack

- **Frontend**: Docusaurus v3, React 18, TypeScript
- **Backend**: FastAPI, Python
- **Database**: Neon Postgres
- **Vector Store**: Qdrant Cloud
- **AI**: Claude Code, OpenAI Agents SDK

## Development

See [PROJECT_SPEC.md](./PROJECT_SPEC.md) for full requirements.

## License

CC BY-NC-SA 4.0
