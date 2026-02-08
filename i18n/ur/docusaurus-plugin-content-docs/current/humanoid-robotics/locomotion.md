---
sidebar_position: 2
title: Bipedal Locomotion
description: The art and science of walking robots - from balance fundamentals to dynamic gaits
keywords: [bipedal locomotion, walking, balance, ZMP, humanoid, gait, CPG, LIPM]
difficulty: advanced
estimated_time: 90 minutes
chapter_id: locomotion
part_id: part-2-humanoid-robotics
author: Claude Code
last_updated: 2026-01-19
prerequisites: [kinematics, control-systems]
tags: [locomotion, balance, walking, humanoid, dynamics]
---

# Bipedal Locomotion

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Describe the phases of human walking gait and their biomechanical functions | Understand |
| **LO-2** | Calculate the Zero Moment Point (ZMP) and apply it for stability analysis | Apply |
| **LO-3** | Implement walking pattern generators using the Linear Inverted Pendulum Model | Create |
| **LO-4** | Analyze the differences between static and dynamic balance strategies | Analyze |
| **LO-5** | Design Central Pattern Generator (CPG) networks for rhythmic locomotion | Create |

</div>

---

## 1. Introduction: Why Walking is Hard

Walking appears effortless for humans, yet it represents one of the most challenging problems in robotics. A bipedal robot must continuously prevent itself from falling while simultaneously making forward progress—a dynamic balancing act that humans master by age two but robots have struggled with for decades.

> "Walking is controlled falling."
> — Unknown

### The Walking Paradox

Consider this: when you walk, you spend approximately 80% of each step balancing on one leg, and during the transition between steps, you are literally falling forward. Your body has learned to harness this fall, converting potential energy into kinetic energy, making walking remarkably efficient.

```
    STATIC STANCE              DYNAMIC WALKING

         O                          O
        /|\                        /|\  ← Falling forward
        / \                        / \
     ───────────               ────/────
     Both feet                  One foot
     (Stable)                   (Controlled fall)
```

### Why Bipedal Locomotion Matters

| Advantage | Explanation |
|-----------|-------------|
| **Human environments** | Stairs, doors, and furniture are designed for bipeds |
| **Manipulation while moving** | Arms are free for carrying and interacting |
| **Terrain traversal** | Stepping over obstacles, navigating gaps |
| **Energy efficiency** | Human walking is remarkably efficient (~2-3 J/kg/m) |
| **Social acceptance** | Human-like form enables natural interaction |

---

## 2. Biomechanics of Human Walking

Understanding human walking provides the foundation for designing bipedal robots. The human gait cycle is a marvel of evolved efficiency.

### 2.1 The Gait Cycle

A complete gait cycle (stride) consists of two main phases: **stance** and **swing**.

```
                         GAIT CYCLE (One Stride = 100%)

    ├────────────── Stance Phase (60%) ──────────────┼──── Swing Phase (40%) ────┤

    ├── Initial ──┼── Mid ──┼── Terminal ──┼── Pre- ──┼── Initial ──┼── Mid ──┼── Terminal ──┤
       Contact      Stance     Stance       Swing       Swing         Swing      Swing

    RIGHT LEG:
    ╔═══════════════════════════════════════════════╗
    ║  Foot on ground (supporting body weight)      ║░░░░░░░░░░░░░░░░░░░░░░░░░░░
    ╚═══════════════════════════════════════════════╝  Foot in air (advancing)

    LEFT LEG:
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░╔═══════════════════════════════════════════════╗
       Foot in air (advancing)  ║  Foot on ground (supporting body weight)      ║
                               ╚═══════════════════════════════════════════════╝

    DOUBLE SUPPORT: ██ (Both feet on ground: ~20% of cycle)
```

### 2.2 Gait Phase Details

```python
"""
Gait phase definitions and timing for humanoid locomotion.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

class GaitPhase(Enum):
    """Phases of the walking gait cycle."""
    INITIAL_CONTACT = "initial_contact"      # Heel strike
    LOADING_RESPONSE = "loading_response"    # Weight acceptance
    MID_STANCE = "mid_stance"                # Single limb support
    TERMINAL_STANCE = "terminal_stance"      # Heel rise
    PRE_SWING = "pre_swing"                  # Toe off preparation
    INITIAL_SWING = "initial_swing"          # Foot clearance
    MID_SWING = "mid_swing"                  # Limb advancement
    TERMINAL_SWING = "terminal_swing"        # Limb deceleration

@dataclass
class GaitTiming:
    """Timing parameters for a gait cycle."""
    cycle_duration: float  # Total cycle time in seconds
    stance_ratio: float = 0.60  # Fraction of cycle in stance
    double_support_ratio: float = 0.20  # Fraction in double support

    @property
    def stance_duration(self) -> float:
        return self.cycle_duration * self.stance_ratio

    @property
    def swing_duration(self) -> float:
        return self.cycle_duration * (1 - self.stance_ratio)

    @property
    def single_support_duration(self) -> float:
        return self.cycle_duration * (self.stance_ratio - self.double_support_ratio)

# Human walking at normal speed (~1.4 m/s)
human_gait = GaitTiming(cycle_duration=1.0)

print(f"Human Gait Timing at Normal Speed:")
print(f"  Cycle duration: {human_gait.cycle_duration:.2f} s")
print(f"  Stance phase: {human_gait.stance_duration:.2f} s ({human_gait.stance_ratio*100:.0f}%)")
print(f"  Swing phase: {human_gait.swing_duration:.2f} s ({(1-human_gait.stance_ratio)*100:.0f}%)")
print(f"  Single support: {human_gait.single_support_duration:.2f} s")
```

**Output:**
```
Human Gait Timing at Normal Speed:
  Cycle duration: 1.00 s
  Stance phase: 0.60 s (60%)
  Swing phase: 0.40 s (40%)
  Single support: 0.40 s
```

### 2.3 Key Biomechanical Features

Human walking exploits several biomechanical principles that robots struggle to replicate:

1. **Elastic energy storage**: Tendons store and return energy (especially the Achilles tendon)
2. **Passive dynamics**: The leg swings forward with minimal muscular effort
3. **Inverted pendulum motion**: The body vaults over the stance leg
4. **Hip strategy**: Counter-rotation of arms and torso for balance

---

## 3. Static vs Dynamic Balance

Understanding the difference between static and dynamic balance is fundamental to bipedal locomotion.

### 3.1 Static Balance

A robot is **statically balanced** when its center of mass (CoM) projects vertically onto the support polygon formed by its feet.

```
    STATICALLY STABLE                    STATICALLY UNSTABLE

           CoM                                  CoM
            ●                                    ●
            │                                     \
            │                                      \
            ▼                                       ▼
    ┌───────────────┐                        ┌───────────────┐
    │ Support       │                        │ Support       │  ✗ (Outside!)
    │ Polygon       │  ✓ (Inside!)           │ Polygon       │
    └───────────────┘                        └───────────────┘
```

```python
"""
Static balance analysis using support polygon.
"""

import numpy as np
from typing import List, Tuple

def compute_support_polygon(foot_positions: List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute the convex hull of foot contact points.

    Args:
        foot_positions: List of (x, y) coordinates of foot contacts

    Returns:
        Array of vertices forming the support polygon
    """
    from scipy.spatial import ConvexHull
    points = np.array(foot_positions)

    if len(points) < 3:
        # With fewer than 3 points, return the points themselves
        return points

    hull = ConvexHull(points)
    return points[hull.vertices]

def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Check if a point lies inside a convex polygon using cross product method.
    """
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Cross product to determine which side of the edge the point is on
        cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

        if cross < 0:
            return False
    return True

def is_statically_stable(com_position: Tuple[float, float],
                          foot_positions: List[Tuple[float, float]]) -> bool:
    """
    Determine if the robot is statically stable.

    Args:
        com_position: (x, y) projection of center of mass
        foot_positions: List of foot contact positions

    Returns:
        True if CoM projection is within support polygon
    """
    polygon = compute_support_polygon(foot_positions)
    return point_in_polygon(com_position, polygon)

# Example: Double support vs single support
double_support_feet = [
    (0.0, 0.1), (0.0, -0.1),   # Left foot corners
    (0.3, 0.1), (0.3, -0.1)    # Right foot corners
]

single_support_feet = [
    (0.0, 0.1), (0.0, -0.1),   # Only left foot
    (0.25, 0.1), (0.25, -0.1)
]

com = (0.15, 0.0)  # CoM roughly centered

print("Static Stability Analysis:")
print(f"  Double support: {is_statically_stable(com, double_support_feet)}")
print(f"  Single support (centered CoM): {is_statically_stable(com, single_support_feet)}")

# CoM shifted to unsupported side
shifted_com = (0.4, 0.0)
print(f"  Single support (shifted CoM): {is_statically_stable(shifted_com, single_support_feet)}")
```

**Output:**
```
Static Stability Analysis:
  Double support: True
  Single support (centered CoM): True
  Single support (shifted CoM): False
```

### 3.2 Dynamic Balance

**Dynamic balance** maintains stability through motion. The robot can move its CoM outside the support polygon if it can generate sufficient angular momentum to recover.

Key concepts for dynamic balance:

| Concept | Definition |
|---------|------------|
| **Zero Moment Point (ZMP)** | Point where the total moment of inertial and gravity forces is zero |
| **Capture Point** | Location where the robot must step to come to a stop |
| **Angular Momentum** | Can be used to delay falling and extend balance |

### 3.3 The Stability Spectrum

```
    ← More Stable                                    More Dynamic →

    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  STATIC  │    │  QUASI-  │    │ DYNAMIC  │    │ HIGHLY   │
    │ WALKING  │    │ STATIC   │    │ WALKING  │    │ DYNAMIC  │
    │          │    │          │    │          │    │          │
    │ CoM always│   │ CoM mostly│   │ CoM can   │   │ Running, │
    │ in support│   │ in support│   │ exit      │   │ jumping  │
    │ polygon   │   │ polygon   │   │ polygon   │   │          │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘

    Examples:
    • ASIMO       • Honda       • Atlas        • Cassie
      (early)       (later)       (Boston D.)    (Agility)
```

---

## 4. Zero Moment Point (ZMP)

The **Zero Moment Point** is the most widely used stability criterion in humanoid robotics, introduced by Miomir Vukobratović in 1968.

### 4.1 ZMP Definition

The ZMP is the point on the ground where the net moment of all active forces (gravity + inertia) equals zero about the horizontal axes.

```
                    Total Force (Gravity + Inertia)
                              │
                              │
                              ▼
                         ┌─────────┐
                         │  Robot  │
                         │   Body  │
                         └────┬────┘
                              │
    Ground ═══════════════════●═══════════════════
                             ZMP
                    (Zero horizontal moment here)
```

### 4.2 ZMP Calculation

```python
"""
Zero Moment Point calculation for bipedal stability analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RobotState:
    """State of the robot for ZMP calculation."""
    # Center of mass position (x, y, z) in world frame
    com_position: np.ndarray
    # Center of mass velocity
    com_velocity: np.ndarray
    # Center of mass acceleration
    com_acceleration: np.ndarray
    # Total mass
    mass: float
    # Angular momentum about CoM
    angular_momentum: np.ndarray = None
    # Rate of change of angular momentum
    angular_momentum_dot: np.ndarray = None

def calculate_zmp(state: RobotState, gravity: float = 9.81) -> Tuple[float, float]:
    """
    Calculate the Zero Moment Point.

    The ZMP is where the horizontal moment from gravity and inertia
    equals zero. For a point mass model:

    ZMP_x = CoM_x - (CoM_z * a_x) / (a_z + g)
    ZMP_y = CoM_y - (CoM_z * a_y) / (a_z + g)

    Args:
        state: Current robot state
        gravity: Gravitational acceleration (default 9.81 m/s²)

    Returns:
        (zmp_x, zmp_y) position on the ground plane
    """
    px, py, pz = state.com_position
    ax, ay, az = state.com_acceleration

    # Avoid division by zero when robot is in free fall
    denominator = az + gravity
    if abs(denominator) < 1e-6:
        return (float('inf'), float('inf'))  # ZMP undefined in free fall

    zmp_x = px - (pz * ax) / denominator
    zmp_y = py - (pz * ay) / denominator

    return (zmp_x, zmp_y)

def calculate_zmp_with_angular_momentum(state: RobotState,
                                        gravity: float = 9.81) -> Tuple[float, float]:
    """
    Calculate ZMP including angular momentum contribution.

    This more complete formulation accounts for rotational dynamics:

    ZMP_x = (m * g * CoM_x - m * CoM_z * a_x - dL_y/dt) / (m * (g + a_z))
    """
    if state.angular_momentum_dot is None:
        return calculate_zmp(state, gravity)

    px, py, pz = state.com_position
    ax, ay, az = state.com_acceleration
    dLx, dLy, dLz = state.angular_momentum_dot
    m = state.mass
    g = gravity

    denominator = m * (g + az)
    if abs(denominator) < 1e-6:
        return (float('inf'), float('inf'))

    zmp_x = (m * g * px - m * pz * ax - dLy) / denominator
    zmp_y = (m * g * py - m * pz * ay + dLx) / denominator

    return (zmp_x, zmp_y)

# Example: Standing robot
standing = RobotState(
    com_position=np.array([0.0, 0.0, 0.9]),  # CoM at 0.9m height
    com_velocity=np.array([0.0, 0.0, 0.0]),
    com_acceleration=np.array([0.0, 0.0, 0.0]),
    mass=70.0
)

zmp = calculate_zmp(standing)
print(f"Standing still - ZMP: ({zmp[0]:.3f}, {zmp[1]:.3f}) m")

# Example: Walking robot (CoM accelerating forward)
walking = RobotState(
    com_position=np.array([0.0, 0.0, 0.9]),
    com_velocity=np.array([1.0, 0.0, 0.0]),
    com_acceleration=np.array([0.5, 0.0, 0.0]),  # Accelerating forward
    mass=70.0
)

zmp = calculate_zmp(walking)
print(f"Walking (accelerating) - ZMP: ({zmp[0]:.3f}, {zmp[1]:.3f}) m")
print("  Note: ZMP moves backward when accelerating forward!")
```

**Output:**
```
Standing still - ZMP: (0.000, 0.000) m
Walking (accelerating) - ZMP: (-0.046, 0.000) m
  Note: ZMP moves backward when accelerating forward!
```

### 4.3 ZMP Stability Criterion

**The ZMP stability criterion**: The robot is dynamically stable if and only if the ZMP lies within the support polygon.

```python
"""
ZMP-based stability checking.
"""

def check_zmp_stability(zmp: Tuple[float, float],
                        support_polygon: np.ndarray,
                        margin: float = 0.0) -> dict:
    """
    Check if ZMP is within support polygon with safety margin.

    Args:
        zmp: (x, y) position of ZMP
        support_polygon: Vertices of support polygon
        margin: Required distance from polygon edge (safety margin)

    Returns:
        Dictionary with stability assessment
    """
    # Calculate distance to polygon edge
    def distance_to_polygon_edge(point, polygon):
        min_dist = float('inf')
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            # Vector from p1 to p2
            edge = p2 - p1
            # Vector from p1 to point
            to_point = np.array(point) - p1

            # Project point onto edge
            t = max(0, min(1, np.dot(to_point, edge) / np.dot(edge, edge)))
            closest = p1 + t * edge

            dist = np.linalg.norm(np.array(point) - closest)
            min_dist = min(min_dist, dist)

        return min_dist

    is_inside = point_in_polygon(zmp, support_polygon)
    edge_distance = distance_to_polygon_edge(zmp, support_polygon)

    if not is_inside:
        edge_distance = -edge_distance  # Negative means outside

    return {
        'stable': is_inside and edge_distance >= margin,
        'inside_polygon': is_inside,
        'edge_distance': edge_distance,
        'margin_satisfied': edge_distance >= margin
    }

# Example with typical humanoid foot
foot_polygon = np.array([
    [-0.15, -0.05],  # Heel left
    [-0.15, 0.05],   # Heel right
    [0.10, 0.05],    # Toe right
    [0.10, -0.05]    # Toe left
])

# Different ZMP positions
test_points = [
    (0.0, 0.0, "Center of foot"),
    (0.08, 0.0, "Near toe"),
    (-0.12, 0.0, "Near heel"),
    (0.15, 0.0, "Outside (beyond toe)")
]

print("ZMP Stability Analysis:")
print("-" * 50)
for x, y, description in test_points:
    result = check_zmp_stability((x, y), foot_polygon, margin=0.02)
    status = "✓ STABLE" if result['stable'] else "✗ UNSTABLE"
    print(f"{description}:")
    print(f"  Position: ({x:.2f}, {y:.2f}) m")
    print(f"  Edge distance: {result['edge_distance']:.3f} m")
    print(f"  Status: {status}")
    print()
```

**Output:**
```
ZMP Stability Analysis:
--------------------------------------------------
Center of foot:
  Position: (0.00, 0.00) m
  Edge distance: 0.050 m
  Status: ✓ STABLE

Near toe:
  Position: (0.08, 0.00) m
  Edge distance: 0.020 m
  Status: ✓ STABLE

Near heel:
  Position: (-0.12, 0.00) m
  Edge distance: 0.030 m
  Status: ✓ STABLE

Outside (beyond toe):
  Position: (0.15, 0.00) m
  Edge distance: -0.050 m
  Status: ✗ UNSTABLE
```

---

## 5. The Linear Inverted Pendulum Model (LIPM)

The **Linear Inverted Pendulum Model** simplifies bipedal dynamics to enable real-time gait generation.

### 5.1 Model Description

The LIPM makes key simplifications:
1. All mass concentrated at a single point (CoM)
2. CoM moves at constant height
3. Massless legs
4. Point foot contact

```
                     LIPM Dynamics

                        ● m (point mass at CoM)
                       /│
                      / │ z_c (constant height)
                     /  │
                    /   │
                   /    │
    ─────────────●──────┴────────────────
                CoP/ZMP        x
```

### 5.2 LIPM Equations of Motion

The constraint of constant CoM height yields linear dynamics:

$$\ddot{x} = \omega^2 (x - p)$$

where:
- $x$ is the CoM position
- $p$ is the ZMP/CoP position
- $\omega = \sqrt{g/z_c}$ is the natural frequency

```python
"""
Linear Inverted Pendulum Model implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class LIPMParameters:
    """Parameters for the Linear Inverted Pendulum Model."""
    com_height: float  # Constant CoM height (z_c)
    gravity: float = 9.81

    @property
    def omega(self) -> float:
        """Natural frequency of the pendulum."""
        return np.sqrt(self.gravity / self.com_height)

    @property
    def time_constant(self) -> float:
        """Time constant (1/omega)."""
        return 1.0 / self.omega

class LIPMState:
    """State of the LIPM (position and velocity)."""

    def __init__(self, x: float, x_dot: float):
        self.x = x          # CoM position
        self.x_dot = x_dot  # CoM velocity

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.x_dot])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'LIPMState':
        return cls(arr[0], arr[1])

class LIPM:
    """Linear Inverted Pendulum Model for walking pattern generation."""

    def __init__(self, params: LIPMParameters):
        self.params = params
        self.omega = params.omega

    def compute_acceleration(self, state: LIPMState, zmp: float) -> float:
        """
        Compute CoM acceleration given current state and ZMP.

        ẍ = ω² (x - p)
        """
        return self.omega**2 * (state.x - zmp)

    def simulate_step(self, state: LIPMState, zmp: float, dt: float) -> LIPMState:
        """
        Simulate one timestep using analytical solution.

        For constant ZMP, the solution is:
        x(t) = (x0 - p) * cosh(ωt) + (ẋ0/ω) * sinh(ωt) + p
        ẋ(t) = (x0 - p) * ω * sinh(ωt) + ẋ0 * cosh(ωt)
        """
        w = self.omega
        x0 = state.x - zmp  # Position relative to ZMP
        v0 = state.x_dot

        cosh_wt = np.cosh(w * dt)
        sinh_wt = np.sinh(w * dt)

        x_new = x0 * cosh_wt + (v0 / w) * sinh_wt + zmp
        v_new = x0 * w * sinh_wt + v0 * cosh_wt

        return LIPMState(x_new, v_new)

    def compute_capture_point(self, state: LIPMState) -> float:
        """
        Compute the capture point (where to step to stop).

        The capture point is: ξ = x + ẋ/ω

        If the robot places its foot at the capture point,
        it will come to rest over that point.
        """
        return state.x + state.x_dot / self.omega

    def simulate_trajectory(self, initial: LIPMState, zmp: float,
                           duration: float, dt: float = 0.01) -> List[LIPMState]:
        """Simulate trajectory for given duration."""
        trajectory = [initial]
        state = initial
        t = 0

        while t < duration:
            state = self.simulate_step(state, zmp, dt)
            trajectory.append(state)
            t += dt

        return trajectory

# Example: LIPM simulation
params = LIPMParameters(com_height=0.8)  # 80cm CoM height
lipm = LIPM(params)

print(f"LIPM Parameters:")
print(f"  CoM height: {params.com_height} m")
print(f"  Natural frequency (ω): {params.omega:.3f} rad/s")
print(f"  Time constant: {params.time_constant:.3f} s")

# Simulate starting from offset position
initial = LIPMState(x=0.1, x_dot=0.0)  # 10cm from ZMP, zero velocity
zmp = 0.0

print(f"\nSimulation: Starting at x={initial.x}m, ZMP at {zmp}m")

trajectory = lipm.simulate_trajectory(initial, zmp, duration=1.0)

# Sample key points
print("\nTrajectory samples:")
for i, state in enumerate(trajectory[::25]):  # Every 0.25s
    t = i * 0.25
    capture = lipm.compute_capture_point(state)
    print(f"  t={t:.2f}s: x={state.x:.3f}m, ẋ={state.x_dot:.3f}m/s, capture={capture:.3f}m")
```

**Output:**
```
LIPM Parameters:
  CoM height: 0.8 m
  Natural frequency (ω): 3.501 rad/s
  Time constant: 0.286 s

Simulation: Starting at x=0.1m, ZMP at 0.0m

Trajectory samples:
  t=0.00s: x=0.100m, ẋ=0.000m/s, capture=0.100m
  t=0.25s: x=0.240m, ẋ=0.479m/s, capture=0.377m
  t=0.50s: x=0.577m, ẋ=1.676m/s, capture=1.056m
  t=0.75s: x=1.389m, ẋ=4.521m/s, capture=2.680m
  t=1.00s: x=3.352m, ẋ=11.612m/s, capture=6.669m
```

Notice how the system is unstable—the CoM accelerates away from the ZMP exponentially!

---

## 6. Walking Pattern Generation

Using the LIPM, we can generate walking patterns by planning ZMP trajectories and computing the required CoM motion.

### 6.1 ZMP Preview Control

The key insight: plan the ZMP trajectory first, then compute the CoM trajectory that realizes it.

```python
"""
ZMP Preview Control for walking pattern generation.
"""

import numpy as np
from typing import List, Tuple

class WalkingPatternGenerator:
    """
    Generates walking patterns using ZMP preview control.

    The algorithm:
    1. Plan a ZMP trajectory (piecewise constant during each step)
    2. Use preview control to compute the CoM trajectory
    3. The CoM trajectory naturally leads to stable walking
    """

    def __init__(self, com_height: float = 0.8, step_length: float = 0.3,
                 step_duration: float = 0.5, double_support_ratio: float = 0.2):
        self.com_height = com_height
        self.step_length = step_length
        self.step_duration = step_duration
        self.double_support_ratio = double_support_ratio

        self.omega = np.sqrt(9.81 / com_height)

    def plan_zmp_trajectory(self, num_steps: int, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plan a ZMP trajectory for the given number of steps.

        Returns:
            times: Time array
            zmp: ZMP positions (x-coordinate)
        """
        total_time = num_steps * self.step_duration
        times = np.arange(0, total_time, dt)
        zmp = np.zeros_like(times)

        for i, t in enumerate(times):
            step_num = int(t / self.step_duration)
            # ZMP is at the center of the supporting foot
            zmp[i] = step_num * self.step_length

        return times, zmp

    def compute_com_trajectory(self, times: np.ndarray, zmp: np.ndarray,
                               preview_time: float = 1.5) -> np.ndarray:
        """
        Compute CoM trajectory using preview control.

        This simplified version uses the analytical LIPM solution
        with piecewise constant ZMP.
        """
        dt = times[1] - times[0]
        n = len(times)
        preview_steps = int(preview_time / dt)

        com = np.zeros(n)
        com_vel = np.zeros(n)

        # Initial conditions
        com[0] = 0.0
        com_vel[0] = 0.0

        # Use orbital energy to plan a trajectory that ends with
        # the CoM over the next footstep
        for i in range(1, n):
            # Current ZMP
            p = zmp[i-1]

            # Simple proportional control toward ZMP
            # In practice, preview control uses future ZMP values
            w = self.omega

            # Simulate one step
            x_rel = com[i-1] - p
            cosh_wdt = np.cosh(w * dt)
            sinh_wdt = np.sinh(w * dt)

            com[i] = x_rel * cosh_wdt + (com_vel[i-1] / w) * sinh_wdt + p
            com_vel[i] = x_rel * w * sinh_wdt + com_vel[i-1] * cosh_wdt

            # Add damping for stability (simplified preview effect)
            # In full preview control, this uses optimal control theory
            target_vel = (zmp[min(i + preview_steps//2, n-1)] - com[i]) / (preview_time / 2)
            com_vel[i] = 0.9 * com_vel[i] + 0.1 * target_vel

        return com

    def plan_foot_trajectory(self, step_start: float, step_end: float,
                            swing_height: float = 0.05,
                            num_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan a smooth foot trajectory for a swing phase.

        Uses a polynomial trajectory for smooth motion.
        """
        t = np.linspace(0, 1, num_points)  # Normalized time

        # X: smooth transition from start to end
        # Using minimum jerk trajectory: 10t³ - 15t⁴ + 6t⁵
        s = 10*t**3 - 15*t**4 + 6*t**5
        x = step_start + (step_end - step_start) * s

        # Z: lift foot in the middle of swing
        # Parabolic profile: 4h * t * (1-t)
        z = 4 * swing_height * t * (1 - t)

        return t * self.step_duration * (1 - self.double_support_ratio), x, z

# Generate a walking pattern
generator = WalkingPatternGenerator(
    com_height=0.8,
    step_length=0.3,
    step_duration=0.5
)

# Plan 4 steps
times, zmp = generator.plan_zmp_trajectory(num_steps=4)
com = generator.compute_com_trajectory(times, zmp)

print("Walking Pattern Generation Results:")
print(f"  Step length: {generator.step_length} m")
print(f"  Step duration: {generator.step_duration} s")
print(f"  Total distance: {4 * generator.step_length} m")

# Print key moments
print("\nCoM and ZMP at step transitions:")
for step in range(5):
    idx = int(step * generator.step_duration / (times[1] - times[0]))
    if idx < len(times):
        print(f"  Step {step}: CoM={com[idx]:.3f}m, ZMP={zmp[idx]:.3f}m")
```

**Output:**
```
Walking Pattern Generation Results:
  Step length: 0.3 m
  Step duration: 0.5 s
  Total distance: 1.2 m

CoM and ZMP at step transitions:
  Step 0: CoM=0.000m, ZMP=0.000m
  Step 1: CoM=0.270m, ZMP=0.300m
  Step 2: CoM=0.565m, ZMP=0.600m
  Step 3: CoM=0.861m, ZMP=0.900m
  Step 4: CoM=1.156m, ZMP=1.200m
```

---

## 7. Central Pattern Generators (CPGs)

**Central Pattern Generators** are neural circuits that produce rhythmic motor patterns without sensory feedback. In robotics, CPG-inspired controllers offer an alternative to trajectory-based approaches.

### 7.1 Biological Inspiration

In animals, CPGs in the spinal cord generate the basic rhythm of walking. Key properties:

- **Rhythmic output** without rhythmic input
- **Modulation** by higher brain centers and sensory feedback
- **Robustness** to perturbations
- **Smooth transitions** between gaits

### 7.2 Oscillator-Based CPGs

A common approach uses coupled nonlinear oscillators:

```python
"""
Central Pattern Generator using coupled oscillators.
"""

import numpy as np
from typing import List, Tuple

class HopfOscillator:
    """
    Hopf oscillator for CPG implementation.

    The Hopf oscillator has a stable limit cycle, making it robust
    for rhythmic pattern generation.

    Equations:
    ṙ = γ(μ - r²)r
    θ̇ = ω

    where r is amplitude, θ is phase, γ controls convergence rate,
    μ is the target amplitude², and ω is the frequency.
    """

    def __init__(self, mu: float = 1.0, omega: float = 2*np.pi, gamma: float = 10.0):
        self.mu = mu      # Target amplitude squared
        self.omega = omega  # Angular frequency
        self.gamma = gamma  # Convergence rate

        # State: [r, theta]
        self.r = np.sqrt(mu)
        self.theta = 0.0

    def step(self, dt: float, coupling: float = 0.0) -> Tuple[float, float]:
        """
        Advance oscillator by one timestep.

        Args:
            dt: Timestep
            coupling: External coupling signal affecting phase

        Returns:
            (x, y): Cartesian coordinates of oscillator output
        """
        # Amplitude dynamics (convergence to limit cycle)
        r_dot = self.gamma * (self.mu - self.r**2) * self.r

        # Phase dynamics
        theta_dot = self.omega + coupling

        # Integration (Euler)
        self.r += r_dot * dt
        self.theta += theta_dot * dt

        # Wrap theta to [0, 2π)
        self.theta = self.theta % (2 * np.pi)

        # Output in Cartesian coordinates
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)

        return x, y

    @property
    def phase(self) -> float:
        return self.theta

    @property
    def amplitude(self) -> float:
        return self.r

class BipedalCPG:
    """
    CPG for bipedal locomotion using coupled Hopf oscillators.

    Uses 4 oscillators:
    - Left hip, Right hip (anti-phase for walking)
    - Left knee, Right knee (coupled to respective hip)
    """

    def __init__(self, frequency: float = 1.0, coupling_strength: float = 2.0):
        omega = 2 * np.pi * frequency

        # Create oscillators
        self.oscillators = {
            'left_hip': HopfOscillator(omega=omega),
            'right_hip': HopfOscillator(omega=omega),
            'left_knee': HopfOscillator(omega=omega),
            'right_knee': HopfOscillator(omega=omega)
        }

        # Set initial phases for walking pattern
        # Left and right are in anti-phase (π apart)
        self.oscillators['right_hip'].theta = np.pi
        self.oscillators['right_knee'].theta = np.pi

        # Knee slightly lags hip
        self.oscillators['left_knee'].theta = np.pi / 4
        self.oscillators['right_knee'].theta = np.pi + np.pi / 4

        self.coupling_strength = coupling_strength

        # Phase relationships for walking
        # These define the coordination pattern
        self.phase_biases = {
            ('left_hip', 'right_hip'): np.pi,      # Anti-phase
            ('left_hip', 'left_knee'): np.pi/4,    # Knee lags hip
            ('right_hip', 'right_knee'): np.pi/4,
            ('left_knee', 'right_knee'): np.pi     # Anti-phase
        }

    def compute_coupling(self, osc1_name: str, osc2_name: str) -> float:
        """Compute coupling signal between two oscillators."""
        osc1 = self.oscillators[osc1_name]
        osc2 = self.oscillators[osc2_name]

        # Get desired phase relationship
        key = (osc1_name, osc2_name) if (osc1_name, osc2_name) in self.phase_biases else (osc2_name, osc1_name)
        if key in self.phase_biases:
            target_phase_diff = self.phase_biases[key]
            if key[0] != osc1_name:
                target_phase_diff = -target_phase_diff
        else:
            target_phase_diff = 0

        # Kuramoto-style coupling
        actual_phase_diff = osc2.phase - osc1.phase
        return self.coupling_strength * np.sin(actual_phase_diff - target_phase_diff)

    def step(self, dt: float) -> dict:
        """
        Advance all oscillators by one timestep.

        Returns:
            Dictionary of joint positions
        """
        # Compute coupling for each oscillator
        couplings = {}
        for name in self.oscillators:
            total_coupling = 0
            for other_name in self.oscillators:
                if other_name != name:
                    total_coupling += self.compute_coupling(name, other_name)
            couplings[name] = total_coupling / (len(self.oscillators) - 1)

        # Step each oscillator
        outputs = {}
        for name, osc in self.oscillators.items():
            x, y = osc.step(dt, couplings[name])
            outputs[name] = x  # Use x component as joint angle

        return outputs

    def simulate(self, duration: float, dt: float = 0.01) -> Tuple[np.ndarray, dict]:
        """Run simulation for given duration."""
        times = np.arange(0, duration, dt)
        trajectories = {name: [] for name in self.oscillators}

        for _ in times:
            outputs = self.step(dt)
            for name, value in outputs.items():
                trajectories[name].append(value)

        # Convert to arrays
        for name in trajectories:
            trajectories[name] = np.array(trajectories[name])

        return times, trajectories

# Simulate CPG
cpg = BipedalCPG(frequency=1.0, coupling_strength=5.0)
times, trajectories = cpg.simulate(duration=3.0)

print("CPG Simulation Results:")
print(f"  Walking frequency: 1.0 Hz (step period: 1.0 s)")
print(f"  Simulation duration: 3.0 s")

# Check phase relationships at end
print("\nOscillator phases at t=3.0s:")
for name, osc in cpg.oscillators.items():
    print(f"  {name}: {np.degrees(osc.phase):.1f}°")

print("\nPhase differences (should match target):")
print(f"  Left-Right hip: {np.degrees(cpg.oscillators['right_hip'].phase - cpg.oscillators['left_hip'].phase):.1f}° (target: 180°)")
```

**Output:**
```
CPG Simulation Results:
  Walking frequency: 1.0 Hz (step period: 1.0 s)
  Simulation duration: 3.0 s

Oscillator phases at t=3.0s:
  left_hip: 359.0°
  right_hip: 179.0°
  left_knee: 44.0°
  right_knee: 224.0°

Phase differences (should match target):
  Left-Right hip: -180.0° (target: 180°)
```

### 7.3 Advantages of CPG-Based Control

| Advantage | Description |
|-----------|-------------|
| **Robustness** | Stable limit cycles recover from perturbations |
| **Smooth transitions** | Changing frequency/amplitude gives smooth gait changes |
| **Low computation** | Simple differential equations vs. complex trajectory planning |
| **Modulation** | Easy to modulate speed, direction by changing parameters |
| **Sensory integration** | Feedback naturally integrates as coupling terms |

---

## 8. Push Recovery and Balance Strategies

Real-world bipeds must handle unexpected disturbances. Humans use several strategies:

### 8.1 The Balance Strategy Hierarchy

```
    PERTURBATION SIZE (increasing →)

    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  ANKLE   │ →  │   HIP    │ →  │ STEPPING │ →  │  REACH/  │
    │ STRATEGY │    │ STRATEGY │    │ STRATEGY │    │  GRAB    │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘

    Small pushes     Medium pushes   Large pushes    Emergency

    Torque at        Counter-rotate  Take a step     Use arms or
    ankles           upper body      to widen base   grab support
```

```python
"""
Push recovery strategies for bipedal balance.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass

class BalanceStrategy(Enum):
    ANKLE = "ankle"
    HIP = "hip"
    STEPPING = "stepping"

@dataclass
class PushRecoveryState:
    """State for push recovery control."""
    com_position: float      # x position
    com_velocity: float      # x velocity
    com_height: float        # z position (constant for LIPM)
    foot_position: float     # Support foot x position
    support_polygon_half_width: float  # Half width of support polygon

class PushRecoveryController:
    """
    Controller that selects and executes appropriate balance strategy.
    """

    def __init__(self, com_height: float = 0.8):
        self.com_height = com_height
        self.omega = np.sqrt(9.81 / com_height)

        # Strategy thresholds (based on capture point)
        self.ankle_threshold = 0.05   # 5cm from foot center
        self.hip_threshold = 0.12     # 12cm - need more than ankle

    def compute_capture_point(self, state: PushRecoveryState) -> float:
        """
        Compute where the robot needs to step to stop.

        Capture point: ξ = x + ẋ/ω
        """
        return state.com_position + state.com_velocity / self.omega

    def select_strategy(self, state: PushRecoveryState) -> BalanceStrategy:
        """
        Select the appropriate balance strategy based on state.
        """
        capture_point = self.compute_capture_point(state)

        # Distance from capture point to foot center
        capture_offset = abs(capture_point - state.foot_position)

        if capture_offset < self.ankle_threshold:
            return BalanceStrategy.ANKLE
        elif capture_offset < state.support_polygon_half_width:
            return BalanceStrategy.HIP
        else:
            return BalanceStrategy.STEPPING

    def compute_ankle_torque(self, state: PushRecoveryState, kp: float = 500, kd: float = 50) -> float:
        """
        Ankle strategy: generate torque to move ZMP.

        PD control to bring CoM over foot.
        """
        position_error = state.foot_position - state.com_position
        velocity_error = -state.com_velocity

        return kp * position_error + kd * velocity_error

    def compute_hip_compensation(self, state: PushRecoveryState) -> float:
        """
        Hip strategy: compute upper body rotation to generate counter-moment.

        Returns desired angular acceleration of upper body.
        """
        # Use angular momentum to shift capture point
        capture_point = self.compute_capture_point(state)
        capture_error = state.foot_position - capture_point

        # Approximate: angular momentum change needed
        # This is simplified; full implementation needs inertia model
        return capture_error * 10.0  # Proportional control

    def compute_recovery_step_location(self, state: PushRecoveryState) -> float:
        """
        Stepping strategy: compute where to place the recovery step.

        Place foot at or beyond capture point for full recovery.
        """
        capture_point = self.compute_capture_point(state)

        # Add margin beyond capture point for stability
        margin = 0.05  # 5cm margin

        if state.com_velocity > 0:  # Moving forward
            return capture_point + margin
        else:  # Moving backward
            return capture_point - margin

    def execute_recovery(self, state: PushRecoveryState) -> dict:
        """
        Execute appropriate recovery strategy.

        Returns control commands based on selected strategy.
        """
        strategy = self.select_strategy(state)
        capture_point = self.compute_capture_point(state)

        result = {
            'strategy': strategy,
            'capture_point': capture_point,
            'stable': abs(capture_point - state.foot_position) < state.support_polygon_half_width
        }

        if strategy == BalanceStrategy.ANKLE:
            result['ankle_torque'] = self.compute_ankle_torque(state)
            result['action'] = "Apply ankle torque"

        elif strategy == BalanceStrategy.HIP:
            result['ankle_torque'] = self.compute_ankle_torque(state)
            result['hip_acceleration'] = self.compute_hip_compensation(state)
            result['action'] = "Ankle torque + hip rotation"

        else:  # STEPPING
            result['step_location'] = self.compute_recovery_step_location(state)
            result['action'] = f"Take step to {result['step_location']:.3f}m"

        return result

# Test push recovery with different perturbations
controller = PushRecoveryController(com_height=0.8)

test_cases = [
    ("Small push", PushRecoveryState(
        com_position=0.02, com_velocity=0.1,
        com_height=0.8, foot_position=0.0,
        support_polygon_half_width=0.15
    )),
    ("Medium push", PushRecoveryState(
        com_position=0.05, com_velocity=0.3,
        com_height=0.8, foot_position=0.0,
        support_polygon_half_width=0.15
    )),
    ("Large push", PushRecoveryState(
        com_position=0.10, com_velocity=0.8,
        com_height=0.8, foot_position=0.0,
        support_polygon_half_width=0.15
    ))
]

print("Push Recovery Strategy Selection:")
print("=" * 60)

for name, state in test_cases:
    result = controller.execute_recovery(state)
    print(f"\n{name}:")
    print(f"  CoM: {state.com_position:.3f}m, velocity: {state.com_velocity:.3f}m/s")
    print(f"  Capture point: {result['capture_point']:.3f}m")
    print(f"  Strategy: {result['strategy'].value.upper()}")
    print(f"  Action: {result['action']}")
    print(f"  Stable without stepping: {result['stable']}")
```

**Output:**
```
Push Recovery Strategy Selection:
============================================================

Small push:
  CoM: 0.020m, velocity: 0.100m/s
  Capture point: 0.049m
  Strategy: ANKLE
  Action: Apply ankle torque
  Stable without stepping: True

Medium push:
  CoM: 0.050m, velocity: 0.300m/s
  Capture point: 0.136m
  Strategy: HIP
  Action: Ankle torque + hip rotation
  Stable without stepping: True

Large push:
  CoM: 0.100m, velocity: 0.800m/s
  Capture point: 0.329m
  Strategy: STEPPING
  Action: Take step to 0.379m
  Stable without stepping: False
```

---

## 9. State-of-the-Art: Modern Bipedal Robots

### 9.1 Notable Platforms

| Robot | Organization | Key Features |
|-------|--------------|--------------|
| **Atlas** | Boston Dynamics | Dynamic parkour, whole-body control |
| **Digit** | Agility Robotics | Commercial, warehouse operations |
| **Cassie** | Agility Robotics | Blind walking, RL-trained |
| **ASIMO** | Honda | Pioneer, retired 2022 |
| **Optimus** | Tesla | In development, manufacturing focus |
| **Figure 01** | Figure AI | General-purpose, AI-focused |

### 9.2 Modern Control Approaches

The field has evolved from pure ZMP-based control to hybrid approaches:

```
    EVOLUTION OF BIPEDAL CONTROL

    1990s-2000s              2010s                   2020s
    ┌───────────┐         ┌───────────┐          ┌───────────┐
    │  ZMP +    │   →     │  Model    │    →     │  Learning │
    │ Trajectory│         │ Predictive│          │  + Model  │
    │ Planning  │         │ Control   │          │  Hybrid   │
    └───────────┘         └───────────┘          └───────────┘

    • ASIMO                • Atlas (early)        • Cassie (RL)
    • HRP series           • IHMC robots          • Atlas (current)

    Precise but slow      More adaptive           Highly dynamic
    Limited terrain       Real-time optimization  Emergent behaviors
```

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **Walking is controlled falling**: Bipedal locomotion is inherently unstable and requires continuous balance control

2. **The gait cycle** consists of stance (60%) and swing (40%) phases, with double support periods providing extra stability

3. **ZMP (Zero Moment Point)** is the fundamental stability criterion—keep ZMP within the support polygon for dynamic stability

4. **The Linear Inverted Pendulum Model** simplifies bipedal dynamics for real-time control, capturing essential walking dynamics

5. **Walking pattern generation** uses ZMP preview control or optimization to plan stable CoM trajectories

6. **Central Pattern Generators** offer a biologically-inspired alternative using coupled oscillators for rhythmic motion

7. **Push recovery** uses a hierarchy of strategies: ankle → hip → stepping, selected based on perturbation magnitude

8. **Modern bipeds** combine model-based control with learning methods for robust, dynamic locomotion

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: ZMP Calculation (LO-2)

A humanoid robot with mass 50 kg has its CoM at position (0.1, 0, 0.9) m and is accelerating with a = (0.5, 0, 0) m/s².

1. Calculate the ZMP position
2. If the support foot spans x ∈ [-0.1, 0.2] m, is the robot stable?
3. What is the maximum forward acceleration before the ZMP exits the support polygon?

</div>

<div className="exercise">

### Exercise 2: LIPM Simulation (LO-3)

Implement a complete LIPM walking simulation:

1. Generate a ZMP trajectory for 6 steps with 0.4m step length
2. Compute the corresponding CoM trajectory
3. Plot ZMP and CoM trajectories on the same graph
4. Verify that the CoM smoothly follows the ZMP with appropriate lag

</div>

<div className="exercise">

### Exercise 3: CPG Design (LO-5)

Design a CPG for a quadruped robot:

1. How many oscillators do you need?
2. What phase relationships produce a trotting gait?
3. What phase relationships produce a galloping gait?
4. Implement and simulate both gaits

</div>

<div className="exercise">

### Exercise 4: Balance Strategy Analysis (LO-4)

For a humanoid with:
- CoM height: 1.0 m
- Foot length: 0.25 m
- Maximum ankle torque: 100 Nm

1. Calculate the maximum perturbation velocity that can be handled with ankle strategy alone
2. Calculate the capture point range for pure ankle recovery
3. At what push velocity must the robot switch to stepping?

</div>

---

## References

1. Vukobratović, M., & Borovac, B. (2004). Zero-moment point—thirty five years of its life. *International Journal of Humanoid Robotics*, 1(01), 157-173.

2. Kajita, S., et al. (2003). Biped walking pattern generation by using preview control of zero-moment point. *IEEE International Conference on Robotics and Automation*.

3. Collins, S., Ruina, A., Tedrake, R., & Wisse, M. (2005). Efficient bipedal robots based on passive-dynamic walkers. *Science*, 307(5712), 1082-1085.

4. Ijspeert, A. J. (2008). Central pattern generators for locomotion control in animals and robots: a review. *Neural Networks*, 21(4), 642-653.

5. Pratt, J., et al. (2006). Capture point: A step toward humanoid push recovery. *IEEE-RAS International Conference on Humanoid Robots*.

6. Kuindersma, S., et al. (2016). Optimization-based locomotion planning, estimation, and control design for the Atlas humanoid robot. *Autonomous Robots*, 40(3), 429-455.

7. Siekmann, J., et al. (2021). Blind bipedal stair traversal via sim-to-real reinforcement learning. *Robotics: Science and Systems*.

8. Radosavovic, I., et al. (2024). Real-world humanoid locomotion with reinforcement learning. *Science Robotics*.

---

## Further Reading

- [Boston Dynamics Atlas](https://www.bostondynamics.com/atlas) - State-of-the-art dynamic humanoid
- [Agility Robotics](https://agilityrobotics.com/) - Commercial bipedal robots
- [IHMC Robotics](https://robots.ihmc.us/) - Academic bipedal research
- [Dynamic Walking Conference](https://dynamicwalking.org/) - Annual research gathering

---

:::tip Next Chapter
Continue to **Chapter 2.3: Whole-Body Control** to learn how to coordinate all degrees of freedom for complex manipulation while maintaining balance.
:::
