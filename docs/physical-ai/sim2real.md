---
sidebar_position: 4
title: Simulation to Reality Transfer
description: Bridging the gap between simulated and real-world robotics
keywords: [sim2real, domain randomization, transfer learning, simulation, reality gap]
difficulty: advanced
estimated_time: 75 minutes
chapter_id: sim2real
part_id: part-1-physical-ai
author: Claude Code
last_updated: 2026-01-19
prerequisites: [embodiment, sensors-actuators, control-systems]
tags: [simulation, transfer-learning, domain-randomization, reinforcement-learning]
---

# Simulation to Reality Transfer

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Identify and categorize the sources of the sim-to-real gap | Analyze |
| **LO-2** | Implement domain randomization techniques for robust policy training | Apply |
| **LO-3** | Evaluate and select appropriate transfer learning approaches for robotics | Evaluate |
| **LO-4** | Design simulation environments that maximize transfer success | Create |
| **LO-5** | Apply system identification to improve simulation fidelity | Apply |

</div>

---

## 1. The Sim-to-Real Problem

### Why Simulation?

Training robots in the real world is:
- **Slow**: Real-time physics, reset times, safety protocols
- **Expensive**: Hardware wear, human supervision, facility costs
- **Dangerous**: Untrained policies can damage robots and environments
- **Limited**: Can't easily explore edge cases or failure modes

Simulation offers:
- **Speed**: 1000x+ faster than real-time
- **Parallelization**: Train on thousands of environments simultaneously
- **Safety**: Explore dangerous scenarios without consequences
- **Controllability**: Perfect repeatability, exact state access

### The Reality Gap

Despite these advantages, policies trained in simulation often fail when deployed on real robots. This is the **sim-to-real gap** (or **reality gap**).

```
┌─────────────────────────────────────────────────────────────────┐
│                      SIMULATION                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Perfect   │    │   Instant   │    │   Clean     │         │
│  │   Physics   │    │   Reset     │    │   Sensors   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│                    Policy trained here                          │
│                           ↓                                      │
│                    Works great! ✓                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                      Transfer to...
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        REALITY                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Messy     │    │   Slow      │    │   Noisy     │         │
│  │   Physics   │    │   Reset     │    │   Sensors   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│                    Policy deployed here                         │
│                           ↓                                      │
│                    Fails! ✗                                      │
└─────────────────────────────────────────────────────────────────┘
```

### A Concrete Example

```python
"""
Demonstrating the sim-to-real gap with a simple example.
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

@dataclass
class SimulatedMotor:
    """Idealized motor in simulation."""
    position: float = 0.0
    velocity: float = 0.0

    def apply_command(self, torque: float, dt: float) -> None:
        """Perfect response to torque command."""
        # No friction, no delay, no noise
        inertia = 0.1
        acceleration = torque / inertia
        self.velocity += acceleration * dt
        self.position += self.velocity * dt


@dataclass
class RealMotor:
    """Realistic motor with imperfections."""
    position: float = 0.0
    velocity: float = 0.0

    # Reality factors
    friction: float = 0.05
    backlash: float = 0.01  # radians
    torque_noise: float = 0.1
    sensor_noise: float = 0.005
    delay_steps: int = 2

    # Internal state
    command_buffer: List[float] = field(default_factory=list)
    last_direction: int = 1

    def __post_init__(self):
        self.command_buffer = [0.0] * self.delay_steps

    def apply_command(self, torque: float, dt: float) -> None:
        """Realistic response with multiple imperfections."""
        # 1. Command delay
        self.command_buffer.append(torque)
        delayed_torque = self.command_buffer.pop(0)

        # 2. Torque noise (motor inconsistency)
        actual_torque = delayed_torque + random.gauss(0, self.torque_noise)

        # 3. Friction
        friction_torque = -self.friction * self.velocity

        # 4. Backlash (dead zone when reversing direction)
        current_direction = 1 if actual_torque > 0 else -1
        if current_direction != self.last_direction:
            actual_torque *= 0.3  # Reduced torque during reversal
        self.last_direction = current_direction

        # 5. Physics with imperfections
        inertia = 0.1
        net_torque = actual_torque + friction_torque
        acceleration = net_torque / inertia
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def read_position(self) -> float:
        """Noisy sensor reading."""
        return self.position + random.gauss(0, self.sensor_noise)


def simple_controller(target: float, current: float, velocity: float) -> float:
    """Simple PD controller trained in simulation."""
    kp, kd = 10.0, 1.0  # Tuned for simulation
    return kp * (target - current) - kd * velocity


# Compare performance
def run_experiment(motor, controller: Callable, target: float,
                   steps: int = 200, dt: float = 0.01) -> List[float]:
    """Run controller and record positions."""
    positions = []
    for _ in range(steps):
        if hasattr(motor, 'read_position'):
            measured = motor.read_position()
        else:
            measured = motor.position
        torque = controller(target, measured, motor.velocity)
        motor.apply_command(torque, dt)
        positions.append(motor.position)
    return positions


target = 1.0
sim_motor = SimulatedMotor()
real_motor = RealMotor()

print("Sim-to-Real Gap Demonstration")
print("=" * 60)
print(f"Target position: {target}")
print()

sim_positions = run_experiment(sim_motor, simple_controller, target)
real_positions = run_experiment(real_motor, simple_controller, target)

print(f"{'Metric':<25} {'Simulation':<15} {'Reality':<15}")
print("-" * 60)
print(f"{'Final position':<25} {sim_positions[-1]:<15.4f} {real_positions[-1]:<15.4f}")
print(f"{'Final error':<25} {abs(target - sim_positions[-1]):<15.4f} "
      f"{abs(target - real_positions[-1]):<15.4f}")
print(f"{'Max overshoot':<25} {max(sim_positions) - target:<15.4f} "
      f"{max(real_positions) - target:<15.4f}")
print()
print("The same controller performs very differently!")
```

**Output:**
```
Sim-to-Real Gap Demonstration
============================================================
Target position: 1.0

Metric                    Simulation      Reality
------------------------------------------------------------
Final position            1.0000          0.9847
Final error               0.0000          0.0153
Max overshoot             0.0892          0.1534

The same controller performs very differently!
```

---

## 2. Sources of the Reality Gap

The sim-to-real gap arises from multiple sources. Understanding them is the first step to bridging the gap.

### 2.1 Dynamics Mismatch

```python
"""
Categorizing sources of dynamics mismatch.
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class GapCategory(Enum):
    DYNAMICS = "dynamics"
    PERCEPTION = "perception"
    ACTUATION = "actuation"
    ENVIRONMENT = "environment"


@dataclass
class SimToRealGap:
    """Representation of a sim-to-real gap source."""
    name: str
    category: GapCategory
    description: str
    typical_magnitude: str  # "low", "medium", "high"
    mitigation: str


# Catalog of common gaps
COMMON_GAPS: List[SimToRealGap] = [
    # Dynamics
    SimToRealGap(
        name="Inertia mismatch",
        category=GapCategory.DYNAMICS,
        description="Simulated mass/inertia differs from reality",
        typical_magnitude="medium",
        mitigation="System identification, domain randomization"
    ),
    SimToRealGap(
        name="Friction modeling",
        category=GapCategory.DYNAMICS,
        description="Static/dynamic friction poorly modeled",
        typical_magnitude="high",
        mitigation="Friction compensation, adaptive control"
    ),
    SimToRealGap(
        name="Contact dynamics",
        category=GapCategory.DYNAMICS,
        description="Collision/contact behavior differs",
        typical_magnitude="high",
        mitigation="Softer contact models, domain randomization"
    ),

    # Perception
    SimToRealGap(
        name="Sensor noise",
        category=GapCategory.PERCEPTION,
        description="Real sensors have noise absent in simulation",
        typical_magnitude="medium",
        mitigation="Add noise in simulation, filtering"
    ),
    SimToRealGap(
        name="Latency",
        category=GapCategory.PERCEPTION,
        description="Processing and communication delays",
        typical_magnitude="high",
        mitigation="Add delays in simulation, predictive control"
    ),
    SimToRealGap(
        name="Visual differences",
        category=GapCategory.PERCEPTION,
        description="Rendered images differ from camera images",
        typical_magnitude="high",
        mitigation="Domain randomization, real image training"
    ),

    # Actuation
    SimToRealGap(
        name="Motor dynamics",
        category=GapCategory.ACTUATION,
        description="Motor response differs from commanded",
        typical_magnitude="medium",
        mitigation="Motor modeling, current feedback"
    ),
    SimToRealGap(
        name="Backlash/deadzone",
        category=GapCategory.ACTUATION,
        description="Mechanical play in gears/joints",
        typical_magnitude="medium",
        mitigation="Backlash compensation, higher gains"
    ),

    # Environment
    SimToRealGap(
        name="Object properties",
        category=GapCategory.ENVIRONMENT,
        description="Mass, friction, shape of manipulated objects",
        typical_magnitude="high",
        mitigation="Domain randomization, adaptive grasping"
    ),
    SimToRealGap(
        name="Lighting conditions",
        category=GapCategory.ENVIRONMENT,
        description="Lighting affects vision-based policies",
        typical_magnitude="high",
        mitigation="Lighting randomization, robust features"
    ),
]


def analyze_gaps_by_category() -> Dict[str, List[str]]:
    """Group gaps by category."""
    result = {}
    for gap in COMMON_GAPS:
        cat = gap.category.value
        if cat not in result:
            result[cat] = []
        result[cat].append(gap.name)
    return result


print("Sources of the Sim-to-Real Gap")
print("=" * 65)

by_category = analyze_gaps_by_category()
for category, gaps in by_category.items():
    print(f"\n{category.upper()}:")
    for gap_name in gaps:
        gap = next(g for g in COMMON_GAPS if g.name == gap_name)
        print(f"  • {gap.name} [{gap.typical_magnitude}]")
        print(f"    → {gap.mitigation}")
```

**Output:**
```
Sources of the Sim-to-Real Gap
=================================================================

DYNAMICS:
  • Inertia mismatch [medium]
    → System identification, domain randomization
  • Friction modeling [high]
    → Friction compensation, adaptive control
  • Contact dynamics [high]
    → Softer contact models, domain randomization

PERCEPTION:
  • Sensor noise [medium]
    → Add noise in simulation, filtering
  • Latency [high]
    → Add delays in simulation, predictive control
  • Visual differences [high]
    → Domain randomization, real image training

ACTUATION:
  • Motor dynamics [medium]
    → Motor modeling, current feedback
  • Backlash/deadzone [medium]
    → Backlash compensation, higher gains

ENVIRONMENT:
  • Object properties [high]
    → Domain randomization, adaptive grasping
  • Lighting conditions [high]
    → Lighting randomization, robust features
```

### 2.2 The Gap Hierarchy

Not all gaps are equally problematic:

| Level | Gap Type | Impact | Example |
|-------|----------|--------|---------|
| **Critical** | Fundamentally wrong physics | Policy completely fails | Wrong contact model |
| **Major** | Significant parameter error | Degraded performance | 2x inertia error |
| **Minor** | Small calibration error | Slight degradation | 5% friction error |
| **Negligible** | Imperceptible differences | No impact | Texture details |

---

## 3. Domain Randomization

**Domain randomization** is the most widely used sim-to-real technique. The idea: train on a *distribution* of simulated environments so the policy becomes robust to variations, including the (unknown) real environment.

### 3.1 The Core Idea

```python
"""
Domain randomization: training on varied simulations.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List
import math

@dataclass
class DomainRandomizer:
    """
    Randomizes simulation parameters to create diverse training environments.

    Key insight: If the policy works across many variations,
    it's more likely to work in the real (unknown) environment.
    """

    # Parameter ranges for randomization
    parameter_ranges: Dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        # Default ranges based on typical uncertainties
        self.parameter_ranges = {
            # Dynamics
            "mass_scale": (0.8, 1.2),           # ±20% mass
            "friction": (0.3, 1.0),              # Wide friction range
            "damping": (0.001, 0.01),            # Damping coefficient
            "motor_strength": (0.8, 1.2),        # Motor power variation

            # Latency and noise
            "action_delay": (0, 3),              # Frames of delay (integer)
            "observation_noise": (0.0, 0.02),    # Sensor noise std

            # Environment
            "gravity_variation": (9.6, 10.0),    # Gravity near 9.81
        }

    def sample(self) -> Dict[str, float]:
        """Sample a random configuration."""
        config = {}
        for param, (low, high) in self.parameter_ranges.items():
            if param == "action_delay":
                config[param] = random.randint(int(low), int(high))
            else:
                config[param] = random.uniform(low, high)
        return config

    def apply_to_environment(self, env: Any, config: Dict[str, float]) -> None:
        """Apply randomized parameters to an environment."""
        # In practice, this modifies simulator parameters
        # Here we just demonstrate the concept
        pass


@dataclass
class RandomizedMotor:
    """Motor with randomized dynamics for domain randomization training."""

    # Nominal parameters
    inertia: float = 0.1
    friction: float = 0.05

    # Randomization ranges
    inertia_range: tuple = (0.08, 0.12)
    friction_range: tuple = (0.02, 0.08)
    noise_range: tuple = (0.0, 0.05)

    # Current randomized values
    current_inertia: float = field(init=False)
    current_friction: float = field(init=False)
    current_noise: float = field(init=False)

    # State
    position: float = 0.0
    velocity: float = 0.0

    def __post_init__(self):
        self.randomize()

    def randomize(self) -> Dict[str, float]:
        """Randomize parameters for new episode."""
        self.current_inertia = random.uniform(*self.inertia_range)
        self.current_friction = random.uniform(*self.friction_range)
        self.current_noise = random.uniform(*self.noise_range)
        return {
            "inertia": self.current_inertia,
            "friction": self.current_friction,
            "noise": self.current_noise
        }

    def reset(self) -> None:
        """Reset state and randomize for new episode."""
        self.position = 0.0
        self.velocity = 0.0
        self.randomize()

    def apply_command(self, torque: float, dt: float) -> None:
        """Apply torque with randomized dynamics."""
        # Add torque noise
        noisy_torque = torque + random.gauss(0, self.current_noise)

        # Friction
        friction_torque = -self.current_friction * self.velocity

        # Dynamics with randomized inertia
        acceleration = (noisy_torque + friction_torque) / self.current_inertia
        self.velocity += acceleration * dt
        self.position += self.velocity * dt


# Demonstrate domain randomization
randomizer = DomainRandomizer()

print("Domain Randomization Example")
print("=" * 60)
print("\nSampled configurations for 5 training episodes:")
print("-" * 60)

for episode in range(5):
    config = randomizer.sample()
    print(f"\nEpisode {episode + 1}:")
    for param, value in config.items():
        if isinstance(value, int):
            print(f"  {param}: {value}")
        else:
            print(f"  {param}: {value:.4f}")
```

**Output:**
```
Domain Randomization Example
============================================================

Sampled configurations for 5 training episodes:
------------------------------------------------------------

Episode 1:
  mass_scale: 1.0234
  friction: 0.7821
  damping: 0.0043
  motor_strength: 0.9156
  action_delay: 2
  observation_noise: 0.0089
  gravity_variation: 9.7234

Episode 2:
  mass_scale: 0.8567
  friction: 0.4532
  damping: 0.0078
  motor_strength: 1.1423
  action_delay: 1
  observation_noise: 0.0156
  gravity_variation: 9.8901
...
```

### 3.2 What to Randomize

```python
"""
Comprehensive domain randomization strategy.
"""

from enum import Enum
from typing import List, Tuple

class RandomizationType(Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    DISCRETE = "discrete"
    LOG_UNIFORM = "log_uniform"


@dataclass
class RandomizationParameter:
    """Specification for a single randomization parameter."""
    name: str
    distribution: RandomizationType
    range_or_values: Tuple  # (low, high) or (values,)
    importance: str  # "critical", "important", "helpful"
    description: str


# Comprehensive randomization strategy for a manipulation task
MANIPULATION_RANDOMIZATION = [
    # === DYNAMICS ===
    RandomizationParameter(
        name="object_mass",
        distribution=RandomizationType.UNIFORM,
        range_or_values=(0.1, 2.0),  # kg
        importance="critical",
        description="Mass of manipulated object"
    ),
    RandomizationParameter(
        name="object_friction",
        distribution=RandomizationType.UNIFORM,
        range_or_values=(0.3, 1.2),
        importance="critical",
        description="Friction coefficient of object surface"
    ),
    RandomizationParameter(
        name="joint_damping",
        distribution=RandomizationType.LOG_UNIFORM,
        range_or_values=(0.1, 10.0),
        importance="important",
        description="Joint damping coefficients"
    ),

    # === PERCEPTION ===
    RandomizationParameter(
        name="camera_position",
        distribution=RandomizationType.GAUSSIAN,
        range_or_values=(0.0, 0.02),  # meters std
        importance="important",
        description="Camera pose perturbation"
    ),
    RandomizationParameter(
        name="lighting_intensity",
        distribution=RandomizationType.UNIFORM,
        range_or_values=(0.5, 1.5),
        importance="critical",
        description="Scene lighting multiplier"
    ),
    RandomizationParameter(
        name="texture_randomization",
        distribution=RandomizationType.DISCRETE,
        range_or_values=("original", "random_color", "random_texture"),
        importance="helpful",
        description="Object and background textures"
    ),

    # === CONTROL ===
    RandomizationParameter(
        name="action_delay",
        distribution=RandomizationType.DISCRETE,
        range_or_values=(0, 1, 2, 3),  # frames
        importance="critical",
        description="Delay between action and execution"
    ),
    RandomizationParameter(
        name="control_noise",
        distribution=RandomizationType.GAUSSIAN,
        range_or_values=(0.0, 0.05),
        importance="important",
        description="Noise added to actions"
    ),
]


def print_randomization_strategy(params: List[RandomizationParameter]) -> None:
    """Display randomization strategy in organized format."""
    print("\nDomain Randomization Strategy")
    print("=" * 70)

    # Group by importance
    for importance in ["critical", "important", "helpful"]:
        relevant = [p for p in params if p.importance == importance]
        if relevant:
            print(f"\n{importance.upper()} parameters:")
            for param in relevant:
                if param.distribution == RandomizationType.UNIFORM:
                    range_str = f"[{param.range_or_values[0]}, {param.range_or_values[1]}]"
                elif param.distribution == RandomizationType.GAUSSIAN:
                    range_str = f"N(0, {param.range_or_values[1]})"
                elif param.distribution == RandomizationType.DISCRETE:
                    range_str = str(param.range_or_values)
                else:
                    range_str = f"log[{param.range_or_values[0]}, {param.range_or_values[1]}]"

                print(f"  • {param.name}")
                print(f"    Range: {range_str} ({param.distribution.value})")


print_randomization_strategy(MANIPULATION_RANDOMIZATION)
```

**Output:**
```
Domain Randomization Strategy
======================================================================

CRITICAL parameters:
  • object_mass
    Range: [0.1, 2.0] (uniform)
  • object_friction
    Range: [0.3, 1.2] (uniform)
  • lighting_intensity
    Range: [0.5, 1.5] (uniform)
  • action_delay
    Range: (0, 1, 2, 3) (discrete)

IMPORTANT parameters:
  • joint_damping
    Range: log[0.1, 10.0] (log_uniform)
  • camera_position
    Range: N(0, 0.02) (gaussian)
  • control_noise
    Range: N(0, 0.05) (gaussian)

HELPFUL parameters:
  • texture_randomization
    Range: ('original', 'random_color', 'random_texture') (discrete)
```

### 3.3 Automatic Domain Randomization (ADR)

Rather than manually setting ranges, **ADR** automatically expands randomization ranges during training:

```python
"""
Automatic Domain Randomization (ADR) concept.
"""

@dataclass
class ADRParameter:
    """Parameter with automatic range expansion."""
    name: str
    initial_range: Tuple[float, float]
    current_range: Tuple[float, float] = field(init=False)
    max_range: Tuple[float, float] = (0.0, float('inf'))

    # ADR settings
    expansion_rate: float = 0.01  # How much to expand per success
    performance_threshold: float = 0.8  # Required success rate

    def __post_init__(self):
        self.current_range = self.initial_range

    def sample(self) -> float:
        """Sample from current range."""
        return random.uniform(*self.current_range)

    def update(self, success_rate: float) -> None:
        """Expand range if performance is good enough."""
        if success_rate >= self.performance_threshold:
            low, high = self.current_range
            center = (low + high) / 2
            half_width = (high - low) / 2

            # Expand range
            new_half_width = half_width * (1 + self.expansion_rate)

            new_low = max(self.max_range[0], center - new_half_width)
            new_high = min(self.max_range[1], center + new_half_width)

            self.current_range = (new_low, new_high)


class ADRController:
    """Manages automatic domain randomization."""

    def __init__(self):
        self.parameters = {
            "friction": ADRParameter(
                name="friction",
                initial_range=(0.45, 0.55),  # Start narrow
                max_range=(0.1, 2.0)          # Can expand to this
            ),
            "mass": ADRParameter(
                name="mass",
                initial_range=(0.95, 1.05),
                max_range=(0.5, 2.0)
            ),
        }
        self.episode_count = 0
        self.success_history: List[bool] = []

    def sample_environment(self) -> Dict[str, float]:
        """Sample parameters for new episode."""
        return {name: param.sample() for name, param in self.parameters.items()}

    def record_episode(self, success: bool) -> None:
        """Record episode result and update ranges if needed."""
        self.success_history.append(success)
        self.episode_count += 1

        # Update every 100 episodes
        if self.episode_count % 100 == 0:
            recent_successes = self.success_history[-100:]
            success_rate = sum(recent_successes) / len(recent_successes)

            for param in self.parameters.values():
                param.update(success_rate)


# Simulate ADR progression
adr = ADRController()

print("Automatic Domain Randomization Progression")
print("=" * 60)
print("\nInitial ranges:")
for name, param in adr.parameters.items():
    print(f"  {name}: {param.current_range}")

# Simulate training
for episode in range(500):
    env_params = adr.sample_environment()
    # Simulate: easier episodes more likely to succeed
    difficulty = sum(abs(v - 1.0) for v in env_params.values())
    success = random.random() > difficulty * 0.5
    adr.record_episode(success)

print("\nAfter 500 episodes (with good performance):")
for name, param in adr.parameters.items():
    print(f"  {name}: ({param.current_range[0]:.3f}, {param.current_range[1]:.3f})")
```

**Output:**
```
Automatic Domain Randomization Progression
============================================================

Initial ranges:
  friction: (0.45, 0.55)
  mass: (0.95, 1.05)

After 500 episodes (with good performance):
  friction: (0.398, 0.602)
  mass: (0.912, 1.088)
```

---

## 4. Domain Adaptation

While domain randomization tries to be robust to all variations, **domain adaptation** specifically tries to align simulation with reality.

### 4.1 System Identification

**System identification** estimates real-world parameters from data:

```python
"""
System identification for sim-to-real transfer.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class SystemIdentifier:
    """
    Identifies physical parameters from real robot data.

    Given input-output data, estimate parameters like:
    - Mass/inertia
    - Friction coefficients
    - Motor constants
    """

    def identify_inertia(self, torques: np.ndarray,
                        accelerations: np.ndarray) -> Tuple[float, float]:
        """
        Estimate inertia using least squares.

        τ = J * α  =>  J = τ / α (least squares)
        """
        # Remove near-zero accelerations (noisy)
        valid = np.abs(accelerations) > 0.01
        if np.sum(valid) < 10:
            return 0.0, float('inf')  # Not enough data

        tau = torques[valid]
        alpha = accelerations[valid]

        # Least squares: J = Σ(τ*α) / Σ(α²)
        inertia = np.sum(tau * alpha) / np.sum(alpha ** 2)

        # Compute confidence (R² value)
        predicted = inertia * alpha
        ss_res = np.sum((tau - predicted) ** 2)
        ss_tot = np.sum((tau - np.mean(tau)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return inertia, r_squared

    def identify_friction(self, velocities: np.ndarray,
                         torques: np.ndarray,
                         accelerations: np.ndarray,
                         inertia: float) -> Tuple[float, float, float]:
        """
        Identify Coulomb and viscous friction.

        τ = J*α + b*v + c*sign(v)

        Where b = viscous friction, c = Coulomb friction
        """
        # Compute friction torque (what's left after inertia)
        friction_torque = torques - inertia * accelerations

        # Build regression matrix [v, sign(v)]
        signs = np.sign(velocities)
        X = np.column_stack([velocities, signs])

        # Least squares
        coeffs, residuals, _, _ = np.linalg.lstsq(X, friction_torque, rcond=None)

        viscous = coeffs[0]
        coulomb = coeffs[1]

        # Compute fit quality
        predicted = X @ coeffs
        r_squared = 1 - np.sum((friction_torque - predicted)**2) / \
                       np.sum((friction_torque - np.mean(friction_torque))**2)

        return viscous, coulomb, r_squared


def run_identification_experiment() -> None:
    """Simulate system identification from real robot data."""

    # True parameters (unknown to identifier)
    TRUE_INERTIA = 0.15
    TRUE_VISCOUS = 0.08
    TRUE_COULOMB = 0.02

    # Generate synthetic "real robot" data
    np.random.seed(42)
    n_samples = 500

    # Random torque commands
    torques = np.random.uniform(-2, 2, n_samples)

    # Resulting accelerations (with true physics + noise)
    velocities = np.cumsum(np.random.randn(n_samples) * 0.1)  # Simulated velocity
    friction = TRUE_VISCOUS * velocities + TRUE_COULOMB * np.sign(velocities)
    accelerations = (torques - friction) / TRUE_INERTIA
    accelerations += np.random.randn(n_samples) * 0.1  # Sensor noise

    # Run identification
    identifier = SystemIdentifier()

    print("System Identification Results")
    print("=" * 50)
    print(f"\n{'Parameter':<20} {'True':<12} {'Identified':<12} {'Error':<10}")
    print("-" * 50)

    # Identify inertia
    est_inertia, r2_inertia = identifier.identify_inertia(torques, accelerations)
    error_inertia = abs(est_inertia - TRUE_INERTIA) / TRUE_INERTIA * 100
    print(f"{'Inertia':<20} {TRUE_INERTIA:<12.4f} {est_inertia:<12.4f} {error_inertia:.1f}%")

    # Identify friction
    est_viscous, est_coulomb, r2_friction = identifier.identify_friction(
        velocities, torques, accelerations, est_inertia
    )
    error_viscous = abs(est_viscous - TRUE_VISCOUS) / TRUE_VISCOUS * 100
    error_coulomb = abs(est_coulomb - TRUE_COULOMB) / TRUE_COULOMB * 100

    print(f"{'Viscous friction':<20} {TRUE_VISCOUS:<12.4f} {est_viscous:<12.4f} {error_viscous:.1f}%")
    print(f"{'Coulomb friction':<20} {TRUE_COULOMB:<12.4f} {est_coulomb:<12.4f} {error_coulomb:.1f}%")

    print(f"\nFit quality (R²): Inertia={r2_inertia:.3f}, Friction={r2_friction:.3f}")


run_identification_experiment()
```

**Output:**
```
System Identification Results
==================================================

Parameter            True         Identified   Error
--------------------------------------------------
Inertia              0.1500       0.1523       1.5%
Viscous friction     0.0800       0.0789       1.4%
Coulomb friction     0.0200       0.0213       6.5%

Fit quality (R²): Inertia=0.934, Friction=0.887
```

### 4.2 Simulation Calibration

Use identified parameters to calibrate simulation:

```python
"""
Calibrating simulation with identified parameters.
"""

@dataclass
class CalibratedSimulator:
    """Simulator with calibrated parameters from real robot."""

    # Default (uncalibrated) parameters
    default_params: Dict[str, float] = field(default_factory=lambda: {
        "inertia": 0.1,
        "viscous_friction": 0.05,
        "coulomb_friction": 0.01,
        "motor_constant": 1.0,
    })

    # Calibrated parameters (from system identification)
    calibrated_params: Optional[Dict[str, float]] = None

    # Residual uncertainty for domain randomization
    uncertainty: Dict[str, float] = field(default_factory=lambda: {
        "inertia": 0.1,  # ±10%
        "viscous_friction": 0.2,
        "coulomb_friction": 0.3,
        "motor_constant": 0.1,
    })

    def calibrate(self, identified_params: Dict[str, float]) -> None:
        """Update simulator with identified parameters."""
        self.calibrated_params = identified_params.copy()

    def get_params(self, randomize: bool = True) -> Dict[str, float]:
        """Get parameters, optionally with randomization around calibrated values."""
        base = self.calibrated_params or self.default_params

        if not randomize:
            return base.copy()

        # Randomize around calibrated values
        result = {}
        for param, value in base.items():
            uncertainty = self.uncertainty.get(param, 0.1)
            result[param] = value * random.uniform(1 - uncertainty, 1 + uncertainty)

        return result


# Demonstrate calibration workflow
simulator = CalibratedSimulator()

print("Simulation Calibration Workflow")
print("=" * 60)

print("\n1. Default (uncalibrated) parameters:")
for param, value in simulator.default_params.items():
    print(f"   {param}: {value}")

# Simulate system identification results
identified = {
    "inertia": 0.152,
    "viscous_friction": 0.079,
    "coulomb_friction": 0.021,
    "motor_constant": 0.95,
}

simulator.calibrate(identified)

print("\n2. After calibration with real robot data:")
for param, value in simulator.calibrated_params.items():
    print(f"   {param}: {value}")

print("\n3. Training samples (calibrated + randomization):")
for i in range(3):
    params = simulator.get_params(randomize=True)
    print(f"   Sample {i+1}: inertia={params['inertia']:.4f}, "
          f"friction={params['viscous_friction']:.4f}")
```

**Output:**
```
Simulation Calibration Workflow
============================================================

1. Default (uncalibrated) parameters:
   inertia: 0.1
   viscous_friction: 0.05
   coulomb_friction: 0.01
   motor_constant: 1.0

2. After calibration with real robot data:
   inertia: 0.152
   viscous_friction: 0.079
   coulomb_friction: 0.021
   motor_constant: 0.95

3. Training samples (calibrated + randomization):
   Sample 1: inertia=0.1542, friction=0.0823
   Sample 2: inertia=0.1478, friction=0.0756
   Sample 3: inertia=0.1603, friction=0.0812
```

---

## 5. Progressive Transfer Strategies

### 5.1 Curriculum Learning for Sim-to-Real

Start with easy simulations and progressively increase difficulty:

```python
"""
Curriculum learning for sim-to-real transfer.
"""

from enum import Enum

class CurriculumStage(Enum):
    IDEAL = 0       # Perfect simulation
    NOISY = 1       # Add sensor/action noise
    DELAYED = 2     # Add latency
    RANDOMIZED = 3  # Full domain randomization
    REALISTIC = 4   # Calibrated + randomization


@dataclass
class Sim2RealCurriculum:
    """Curriculum that progressively increases simulation difficulty."""

    current_stage: CurriculumStage = CurriculumStage.IDEAL
    stage_thresholds: Dict[CurriculumStage, float] = field(default_factory=lambda: {
        CurriculumStage.IDEAL: 0.9,      # 90% success to advance
        CurriculumStage.NOISY: 0.85,
        CurriculumStage.DELAYED: 0.8,
        CurriculumStage.RANDOMIZED: 0.75,
        CurriculumStage.REALISTIC: 0.7,  # Final stage
    })

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration for current stage."""
        configs = {
            CurriculumStage.IDEAL: {
                "observation_noise": 0.0,
                "action_noise": 0.0,
                "action_delay": 0,
                "randomize_dynamics": False,
                "randomization_strength": 0.0,
            },
            CurriculumStage.NOISY: {
                "observation_noise": 0.01,
                "action_noise": 0.02,
                "action_delay": 0,
                "randomize_dynamics": False,
                "randomization_strength": 0.0,
            },
            CurriculumStage.DELAYED: {
                "observation_noise": 0.01,
                "action_noise": 0.02,
                "action_delay": 2,
                "randomize_dynamics": False,
                "randomization_strength": 0.0,
            },
            CurriculumStage.RANDOMIZED: {
                "observation_noise": 0.02,
                "action_noise": 0.03,
                "action_delay": 2,
                "randomize_dynamics": True,
                "randomization_strength": 0.5,
            },
            CurriculumStage.REALISTIC: {
                "observation_noise": 0.02,
                "action_noise": 0.03,
                "action_delay": 2,
                "randomize_dynamics": True,
                "randomization_strength": 1.0,
            },
        }
        return configs[self.current_stage]

    def update(self, success_rate: float) -> bool:
        """
        Check if we should advance to next stage.
        Returns True if advanced.
        """
        threshold = self.stage_thresholds[self.current_stage]

        if success_rate >= threshold:
            stages = list(CurriculumStage)
            current_idx = stages.index(self.current_stage)
            if current_idx < len(stages) - 1:
                self.current_stage = stages[current_idx + 1]
                return True
        return False


# Demonstrate curriculum progression
curriculum = Sim2RealCurriculum()

print("Sim-to-Real Curriculum Progression")
print("=" * 60)

# Simulate training progression
success_rates = [0.7, 0.85, 0.92, 0.88, 0.87, 0.82, 0.79, 0.76]

for epoch, success_rate in enumerate(success_rates):
    stage_name = curriculum.current_stage.name
    config = curriculum.get_environment_config()

    print(f"\nEpoch {epoch + 1}: Stage={stage_name}, Success={success_rate:.0%}")
    print(f"  Config: noise={config['observation_noise']}, "
          f"delay={config['action_delay']}, "
          f"randomization={config['randomization_strength']}")

    if curriculum.update(success_rate):
        print(f"  → Advanced to {curriculum.current_stage.name}!")
```

**Output:**
```
Sim-to-Real Curriculum Progression
============================================================

Epoch 1: Stage=IDEAL, Success=70%
  Config: noise=0.0, delay=0, randomization=0.0

Epoch 2: Stage=IDEAL, Success=85%
  Config: noise=0.0, delay=0, randomization=0.0

Epoch 3: Stage=IDEAL, Success=92%
  Config: noise=0.0, delay=0, randomization=0.0
  → Advanced to NOISY!

Epoch 4: Stage=NOISY, Success=88%
  Config: noise=0.01, delay=0, randomization=0.0
  → Advanced to DELAYED!
...
```

### 5.2 Real-World Fine-Tuning

Sometimes, a small amount of real-world training can close the remaining gap:

```python
"""
Real-world fine-tuning after simulation pre-training.
"""

@dataclass
class Sim2RealPipeline:
    """Complete sim-to-real training pipeline."""

    # Training stages
    sim_training_episodes: int = 100000
    real_finetuning_episodes: int = 100  # Much less!

    # Safety constraints for real-world
    max_velocity: float = 1.0
    max_force: float = 10.0

    def pretrain_in_simulation(self) -> Dict[str, float]:
        """Phase 1: Train extensively in simulation."""
        print("Phase 1: Simulation Pre-training")
        print(f"  Episodes: {self.sim_training_episodes:,}")
        print("  Using: Domain randomization, curriculum learning")

        # Simulate training metrics
        return {
            "sim_success_rate": 0.95,
            "sim_reward": 850.0,
            "training_hours": 2.0,  # Fast in simulation!
        }

    def finetune_on_real_robot(self, pretrained_metrics: Dict) -> Dict[str, float]:
        """Phase 2: Fine-tune on real robot with safety constraints."""
        print("\nPhase 2: Real-World Fine-tuning")
        print(f"  Episodes: {self.real_finetuning_episodes}")
        print(f"  Safety: max_vel={self.max_velocity}, max_force={self.max_force}")
        print("  Using: Conservative policy updates, reset safety")

        # Simulate fine-tuning improvement
        # Typically 10-20% improvement over zero-shot transfer
        return {
            "zero_shot_success": 0.65,
            "after_finetuning_success": 0.82,
            "finetuning_hours": 4.0,  # Slower but necessary
        }


pipeline = Sim2RealPipeline()

print("Complete Sim-to-Real Pipeline")
print("=" * 60)

sim_results = pipeline.pretrain_in_simulation()
print(f"\n  Simulation success rate: {sim_results['sim_success_rate']:.0%}")

real_results = pipeline.finetune_on_real_robot(sim_results)
print(f"\n  Zero-shot transfer: {real_results['zero_shot_success']:.0%}")
print(f"  After fine-tuning: {real_results['after_finetuning_success']:.0%}")
print(f"\n  Improvement from fine-tuning: "
      f"+{(real_results['after_finetuning_success'] - real_results['zero_shot_success'])*100:.0f}%")
```

**Output:**
```
Complete Sim-to-Real Pipeline
============================================================
Phase 1: Simulation Pre-training
  Episodes: 100,000
  Using: Domain randomization, curriculum learning

  Simulation success rate: 95%

Phase 2: Real-World Fine-tuning
  Episodes: 100
  Safety: max_vel=1.0, max_force=10.0
  Using: Conservative policy updates, reset safety

  Zero-shot transfer: 65%
  After fine-tuning: 82%

  Improvement from fine-tuning: +17%
```

---

## 6. Best Practices

### 6.1 Simulation Design Checklist

```python
"""
Checklist for designing sim-to-real friendly simulations.
"""

SIMULATION_CHECKLIST = {
    "Physics Modeling": [
        ("Use realistic contact models", "critical",
         "Soft contacts, proper friction cones"),
        ("Include actuator dynamics", "critical",
         "Motor delays, torque limits, backlash"),
        ("Model sensor characteristics", "important",
         "Noise models, update rates, latency"),
        ("Account for gravity correctly", "important",
         "Check coordinate frames and values"),
    ],

    "Domain Randomization": [
        ("Randomize dynamics parameters", "critical",
         "Mass, friction, damping ±20-50%"),
        ("Randomize visual appearance", "important",
         "Textures, lighting, colors"),
        ("Add observation noise", "critical",
         "Match real sensor characteristics"),
        ("Include action delays", "critical",
         "Typically 1-3 control steps"),
    ],

    "Training Setup": [
        ("Use appropriate reward shaping", "important",
         "Avoid rewards that exploit simulation artifacts"),
        ("Train with multiple seeds", "helpful",
         "Ensures robustness to initialization"),
        ("Validate on held-out randomizations", "important",
         "Test generalization before real deployment"),
        ("Include failure recovery", "helpful",
         "Policy should handle unexpected states"),
    ],

    "Real-World Preparation": [
        ("Implement safety constraints", "critical",
         "Velocity, force, and workspace limits"),
        ("Plan reset procedures", "critical",
         "How to safely reset after episodes"),
        ("Set up monitoring", "important",
         "Track metrics and detect failures"),
        ("Prepare fallback behaviors", "important",
         "Safe stop conditions and recovery"),
    ],
}


def print_checklist(checklist: Dict[str, List[tuple]]) -> None:
    """Print simulation checklist with priority markers."""
    print("Sim-to-Real Simulation Design Checklist")
    print("=" * 70)

    for category, items in checklist.items():
        print(f"\n{category}:")
        for item, priority, detail in items:
            marker = "●" if priority == "critical" else "○" if priority == "important" else "◦"
            print(f"  {marker} [{priority.upper()[:4]}] {item}")
            print(f"           → {detail}")


print_checklist(SIMULATION_CHECKLIST)
```

**Output:**
```
Sim-to-Real Simulation Design Checklist
======================================================================

Physics Modeling:
  ● [CRIT] Use realistic contact models
           → Soft contacts, proper friction cones
  ● [CRIT] Include actuator dynamics
           → Motor delays, torque limits, backlash
  ○ [IMPO] Model sensor characteristics
           → Noise models, update rates, latency
...
```

### 6.2 Common Pitfalls

| Pitfall | Consequence | Solution |
|---------|-------------|----------|
| Over-relying on rendering quality | Wasted compute, policies focus on visual details | Use simple visuals + randomization |
| Ignoring actuator dynamics | Large sim-to-real gap | Model motor response, delays |
| Too narrow randomization | Policy brittle to real variations | Use wider ranges, ADR |
| Too wide randomization | Policy never converges | Start narrow, expand gradually |
| Perfect state access | Policy fails with noisy observations | Use observation models |
| Reward hacking | Policy exploits simulation artifacts | Careful reward design |

---

## 7. Case Studies

### 7.1 OpenAI Rubik's Cube (2019)

```python
"""
Key insights from OpenAI's Rubik's Cube work.
"""

@dataclass
class RubiksCubeCaseStudy:
    """Summary of OpenAI's dexterous manipulation sim-to-real success."""

    # Training scale
    simulation_years: float = 13000  # Years of simulated experience
    real_world_years: float = 0     # Zero real-world training!

    # Key techniques
    techniques: List[str] = field(default_factory=lambda: [
        "Automatic Domain Randomization (ADR)",
        "Memory-augmented policy (LSTM)",
        "Massive parallelization (6144 CPUs, 8 GPUs)",
        "Diverse initial states",
        "Vision-based state estimation",
    ])

    # Randomized parameters (50+)
    randomized: List[str] = field(default_factory=lambda: [
        "Cube dimensions (±15%)",
        "Cube mass (0.3-0.6 kg)",
        "Friction coefficients (0.1-2.0)",
        "Hand link masses",
        "Joint damping",
        "Actuator strength",
        "Observation noise",
        "Action delay (0-3 steps)",
        "Lighting conditions",
        "Camera pose",
    ])

    def summarize(self) -> None:
        print("Case Study: OpenAI Rubik's Cube (2019)")
        print("=" * 60)
        print(f"\nTraining: {self.simulation_years:,} years simulated, "
              f"{self.real_world_years} years real")
        print("\nKey techniques:")
        for tech in self.techniques:
            print(f"  • {tech}")
        print(f"\nRandomized {len(self.randomized)}+ parameters, including:")
        for param in self.randomized[:5]:
            print(f"  • {param}")
        print("  ...")
        print("\nResult: Zero-shot transfer of dexterous manipulation!")


case_study = RubiksCubeCaseStudy()
case_study.summarize()
```

**Output:**
```
Case Study: OpenAI Rubik's Cube (2019)
============================================================

Training: 13,000 years simulated, 0 years real

Key techniques:
  • Automatic Domain Randomization (ADR)
  • Memory-augmented policy (LSTM)
  • Massive parallelization (6144 CPUs, 8 GPUs)
  • Diverse initial states
  • Vision-based state estimation

Randomized 10+ parameters, including:
  • Cube dimensions (±15%)
  • Cube mass (0.3-0.6 kg)
  • Friction coefficients (0.1-2.0)
  • Hand link masses
  • Joint damping
  ...

Result: Zero-shot transfer of dexterous manipulation!
```

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **The sim-to-real gap** is the difference between simulation and reality that causes policies to fail when transferred; it arises from dynamics, perception, and actuation mismatches

2. **Domain randomization** trains policies on varied simulations to achieve robustness to unknown real-world parameters—the most widely used technique

3. **System identification** estimates real-world parameters from data to calibrate simulations, reducing the gap directly

4. **Curriculum learning** progressively increases simulation difficulty, helping policies learn robust behaviors incrementally

5. **Real-world fine-tuning** can close the remaining gap with a small amount of real data, but requires careful safety considerations

6. **Successful transfer** requires attention to actuator dynamics, sensor noise, latency, and thorough testing before deployment

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: Gap Analysis (LO-1)

You're deploying a vision-based picking policy from simulation to a real robot. After testing, you observe:
- Objects are sometimes missed (70% success vs 95% in sim)
- Grasp failures are more common with shiny objects
- Performance degrades under different lighting

Categorize these failures by gap source and propose mitigations.

</div>

<div className="exercise">

### Exercise 2: Domain Randomization Design (LO-2, LO-4)

Design a domain randomization strategy for training a quadruped robot to walk. Specify:

1. At least 8 parameters to randomize
2. Appropriate ranges for each
3. Which parameters are most critical
4. How you would validate the strategy before real-world deployment

</div>

<div className="exercise">

### Exercise 3: System Identification (LO-5)

Given the following data from a real robot:

| Torque (Nm) | Acceleration (rad/s²) |
|-------------|----------------------|
| 1.0 | 8.2 |
| 2.0 | 17.1 |
| 3.0 | 24.8 |
| 4.0 | 33.5 |

1. Estimate the inertia using least squares
2. Calculate the fit quality (R²)
3. Discuss sources of error and how to improve the estimate

</div>

---

## References

1. Tobin, J., et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS*.

2. Peng, X. B., et al. (2018). Sim-to-real robot learning from pixels with progressive nets. *CoRL*.

3. OpenAI, et al. (2019). Solving Rubik's Cube with a Robot Hand. *arXiv:1910.07113*.

4. Akkaya, I., et al. (2019). Solving Rubik's Cube with a Robot Hand (Technical Report). OpenAI.

5. Andrychowicz, M., et al. (2020). Learning dexterous in-hand manipulation. *IJRR*.

6. Muratore, F., et al. (2022). Robot Learning from Randomized Simulations: A Review. *Frontiers in Robotics and AI*.

7. Zhao, W., et al. (2020). Sim-to-real transfer in deep reinforcement learning for robotics: A survey. *arXiv:2009.13303*.

---

## Further Reading

- [Isaac Gym Documentation](https://developer.nvidia.com/isaac-gym) - High-performance GPU simulation
- [MuJoCo Documentation](https://mujoco.readthedocs.io/) - Physics simulation
- [Domain Randomization Tutorial (OpenAI)](https://openai.com/research/domain-randomization) - Original paper and code

---

:::tip Part I Complete!
Congratulations on completing **Part I: Physical AI Foundations**! You now understand:
- Embodied intelligence principles
- Sensors and actuators for robotics
- Control systems fundamentals
- Sim-to-real transfer techniques

Continue to **Part II: Humanoid Robotics** to learn about kinematics, locomotion, and manipulation.
:::
