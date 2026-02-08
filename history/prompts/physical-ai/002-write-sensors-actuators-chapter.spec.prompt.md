---
id: 002
title: Write Chapter 2 - Sensors and Actuators
stage: spec
date: 2026-01-19
surface: agent
model: claude-opus-4-5
feature: physical-ai
branch: master
user: Amena
command: none
labels: ["chapter", "content", "sensors", "actuators", "hardware"]
links:
  spec: spec/chapter.schema.yaml
  ticket: null
  adr: null
  pr: null
files:
  - docs/physical-ai/sensors-actuators.md
tests:
  - Validates against chapter.schema.yaml
  - Contains 5 learning objectives with Bloom's levels
  - Contains 10 Python code examples
  - Contains 7 academic references
  - Contains 3 exercises
---

## Prompt

User selected option "1" to write Chapter 2 (Sensors & Actuators) with full spec-compliant content after completing Chapter 1 (Embodied Intelligence).

## Response snapshot

Created comprehensive Chapter 2 (~1240 lines) covering Sensors & Actuators with:

### Structure (chapter.schema.yaml compliant)
- **Frontmatter**: chapter_id, part_id, difficulty, prerequisites, tags
- **Learning Objectives**: 5 objectives (Remember, Understand, Apply x2, Analyze)
- **6 Main Sections**: Introduction, Proprioceptive Sensors, Exteroceptive Sensors, Actuators, Sensor Fusion, Practical Considerations

### Content Highlights

**Section 2: Proprioceptive Sensors**
- Incremental & absolute encoders with quadrature decoding code
- IMU with complementary filter orientation estimator
- Force/torque sensors with gripper controller example
- Summary comparison table

**Section 3: Exteroceptive Sensors**
- Camera pinhole model with 3D-to-2D projection code
- Depth sensor types comparison (Stereo, Structured Light, ToF, LiDAR)
- LiDAR scan processing with obstacle detection code
- Summary comparison table

**Section 4: Actuators**
- DC motor physics simulation with back-EMF
- Servo motor with PID position control
- Stepper motor with microstepping modes
- Actuator comparison table

**Section 5: Sensor Fusion**
- Weighted average fusion with uncertainty reduction
- Kalman filter preview with tracking example

**Section 6: Practical Considerations**
- Sensor selection guidelines
- Depth sensor recommendation helper function
- Common pitfalls table

### Code Examples (10 total)
1. IncrementalEncoder - quadrature decoding
2. SimpleOrientationEstimator - IMU complementary filter
3. ForceSensor + GripperController - force feedback
4. CameraIntrinsics - 3D projection
5. DepthCamera - sensor comparison
6. LidarScan - obstacle detection
7. DCMotor - physics simulation
8. ServoMotor - PID control
9. StepperMotor - microstepping
10. KalmanFilter1D - state estimation

## Outcome

- ✅ Impact: Complete hardware foundations chapter for Physical AI section
- 🧪 Tests: Content validates against chapter.schema.yaml requirements
- 📁 Files: 1 file modified (docs/physical-ai/sensors-actuators.md)
- 🔁 Next prompts: Write Chapter 3 (Control Systems), Create subagents, Build RAG backend
- 🧠 Reflection: Comprehensive coverage of robot I/O with practical code examples for each sensor/actuator type

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Schema compliance - PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Add real sensor datasheets as references
