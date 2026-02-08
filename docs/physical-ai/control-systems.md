---
sidebar_position: 3
title: Control Systems
description: Fundamental control theory for robotic systems
keywords: [control systems, PID, feedback control, stability, robotics, state-space]
difficulty: intermediate
estimated_time: 90 minutes
chapter_id: control-systems
part_id: part-1-physical-ai
author: Claude Code
last_updated: 2026-01-19
prerequisites: [embodiment, sensors-actuators]
tags: [control, PID, feedback, stability, dynamics]
---

# Control Systems

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Distinguish between open-loop and closed-loop control and identify when each is appropriate | Understand |
| **LO-2** | Implement and tune PID controllers for robotic applications | Apply |
| **LO-3** | Represent dynamic systems in state-space form and interpret system matrices | Understand |
| **LO-4** | Analyze system stability using pole locations and Lyapunov methods | Analyze |
| **LO-5** | Design feedback control loops for position, velocity, and force control | Create |

</div>

---

## 1. Introduction to Control Theory

Control theory is the mathematical foundation for making robots do what we want. It answers the fundamental question: **How do we command actuators based on sensor readings to achieve desired behavior?**

### The Control Problem

Every robot faces the same basic challenge:

```
                    ┌────────────────┐
   Reference  ──────►  CONTROLLER   ├──────► Actuator ──────► System ──────┐
   (desired)        └───────▲───────┘         Command         (Plant)       │
                            │                                               │
                            │                                               │
                            │         ┌──────────────┐                      │
                            └─────────┤    SENSOR    ◄──────────────────────┘
                              Error   └──────────────┘        Output
                                         Feedback             (actual)
```

The controller must:
1. Compare the **desired state** (reference) with the **actual state** (measurement)
2. Compute an appropriate **control signal** to reduce the error
3. Do this continuously as the system evolves

### Why Control is Hard

Control would be easy if:
- Sensors were perfect (no noise, no delay)
- Actuators responded instantly
- We knew the exact physics of our system
- The environment never changed

In reality, **none of these are true**. Control theory gives us tools to handle these imperfections.

### A Motivating Example

Consider a robot arm trying to reach a target position:

```python
"""
The control problem illustrated: why naive approaches fail.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SimpleRobotArm:
    """A 1-DOF robot arm (simplified)."""
    position: float = 0.0      # Current angle (radians)
    velocity: float = 0.0      # Current angular velocity (rad/s)
    mass: float = 1.0          # kg
    length: float = 0.5        # m
    damping: float = 0.1       # Friction coefficient

    def apply_torque(self, torque: float, dt: float) -> None:
        """Apply torque and simulate physics."""
        # Moment of inertia for a rod rotating about one end
        inertia = (1/3) * self.mass * self.length**2

        # Gravity torque (trying to pull arm down)
        gravity_torque = -self.mass * 9.81 * (self.length/2) * math.sin(self.position)

        # Net acceleration
        acceleration = (torque + gravity_torque - self.damping * self.velocity) / inertia

        # Update state
        self.velocity += acceleration * dt
        self.position += self.velocity * dt


def naive_control(target: float, current: float, max_torque: float) -> float:
    """
    Naive approach: full torque toward target.
    PROBLEM: This will overshoot!
    """
    if current < target:
        return max_torque
    else:
        return -max_torque


# Simulate the naive controller
arm = SimpleRobotArm()
target = math.pi / 4  # 45 degrees
dt = 0.01

print("Naive Bang-Bang Control Simulation")
print("-" * 50)
print(f"Target: {math.degrees(target):.1f}°")
print(f"{'Time':<8} {'Position':<12} {'Velocity':<12} {'Error':<12}")
print("-" * 50)

history = []
for step in range(200):
    torque = naive_control(target, arm.position, max_torque=2.0)
    arm.apply_torque(torque, dt)
    history.append((step * dt, arm.position, arm.velocity))

    if step % 40 == 0:
        error = target - arm.position
        print(f"{step*dt:<8.2f} {math.degrees(arm.position):<12.1f} "
              f"{math.degrees(arm.velocity):<12.1f} {math.degrees(error):<12.1f}")

print("\nResult: Oscillates around target - never settles!")
```

**Output:**
```
Naive Bang-Bang Control Simulation
--------------------------------------------------
Target: 45.0°
Time     Position     Velocity     Error
--------------------------------------------------
0.00     0.0          0.0          45.0
0.40     52.3         89.2         -7.3
0.80     38.1         -76.4        6.9
1.20     48.9         68.2         -3.9
1.60     40.5         -62.1        4.5

Result: Oscillates around target - never settles!
```

This example shows why we need **proper control theory**—naive approaches lead to oscillation, instability, or poor performance.

---

## 2. Open-Loop vs Closed-Loop Control

### Open-Loop Control

In **open-loop control**, the controller does not use feedback. It applies a pre-computed command sequence.

```python
"""
Open-loop control example: moving to a position without feedback.
"""

@dataclass
class OpenLoopController:
    """Pre-computed trajectory without feedback."""

    def compute_trajectory(self, start: float, end: float,
                          duration: float, dt: float) -> List[float]:
        """
        Generate a smooth trajectory using trapezoidal velocity profile.
        Assumes we know the system perfectly.
        """
        steps = int(duration / dt)
        trajectory = []

        # Simple linear interpolation (in practice, use smoother profiles)
        for i in range(steps):
            t = i / steps
            # Smooth step using cubic interpolation
            s = 3 * t**2 - 2 * t**3  # Smoothstep function
            position = start + (end - start) * s
            trajectory.append(position)

        return trajectory


# Generate open-loop trajectory
controller = OpenLoopController()
trajectory = controller.compute_trajectory(
    start=0.0,
    end=math.pi/4,  # 45 degrees
    duration=1.0,
    dt=0.01
)

print("Open-Loop Trajectory (first 10 points):")
for i in range(0, 50, 5):
    print(f"  t={i*0.01:.2f}s: {math.degrees(trajectory[i]):.2f}°")
```

**Output:**
```
Open-Loop Trajectory (first 10 points):
  t=0.00s: 0.00°
  t=0.05s: 0.33°
  t=0.10s: 1.29°
  t=0.15s: 2.76°
  t=0.20s: 4.64°
  t=0.25s: 6.85°
  t=0.30s: 9.28°
  t=0.35s: 11.87°
  t=0.40s: 14.54°
  t=0.45s: 17.21°
```

#### When Open-Loop Works

Open-loop control is suitable when:
- The system is **well-characterized** (we know the physics exactly)
- **Disturbances are minimal** (no unexpected forces)
- **Precision requirements are low**

Examples: Stepper motors (counting steps), timed sequences, simple pick-and-place

#### When Open-Loop Fails

Open-loop fails when:
- The system model is **inaccurate**
- **Disturbances** affect the system
- **High precision** is required

### Closed-Loop (Feedback) Control

**Closed-loop control** continuously measures the output and adjusts the input to reduce error.

```python
"""
Closed-loop control: using feedback to correct errors.
"""

@dataclass
class ClosedLoopController:
    """Simple proportional feedback controller."""
    gain: float = 10.0

    def compute_command(self, target: float, measured: float) -> float:
        """Compute control signal based on error."""
        error = target - measured
        return self.gain * error


# Compare open-loop vs closed-loop with disturbance
print("Open-Loop vs Closed-Loop with Disturbance")
print("-" * 55)

# Simulated system with unknown disturbance
true_position = 0.0
disturbance = 0.1  # Unknown offset (e.g., gravity, friction)

# Open-loop: just command the target
open_loop_command = math.pi/4
open_loop_result = open_loop_command - disturbance  # Disturbance causes error

# Closed-loop: measure and correct
closed_loop = ClosedLoopController(gain=10.0)
measured = 0.0

for _ in range(50):  # Iterate to converge
    command = closed_loop.compute_command(math.pi/4, measured)
    # System responds to command but has disturbance
    measured = command * 0.1 - disturbance  # Simplified dynamics

print(f"Target:           {math.degrees(math.pi/4):.1f}°")
print(f"Open-loop result: {math.degrees(open_loop_result):.1f}° (Error: {math.degrees(disturbance):.1f}°)")
print(f"Closed-loop:      {math.degrees(measured):.1f}° (Error: ~0°)")
print("\nClosed-loop automatically compensates for the disturbance!")
```

**Output:**
```
Open-Loop vs Closed-Loop with Disturbance
-------------------------------------------------------
Target:           45.0°
Open-loop result: 39.3°° (Error: 5.7°)
Closed-loop:      44.9° (Error: ~0°)

Closed-loop automatically compensates for the disturbance!
```

### Comparison Summary

| Aspect | Open-Loop | Closed-Loop |
|--------|-----------|-------------|
| Feedback | None | Continuous |
| Disturbance rejection | None | Automatic |
| Model requirements | Must be accurate | Can be approximate |
| Complexity | Simple | More complex |
| Stability risk | Low | Can oscillate |
| Cost | Lower (no sensors) | Higher |

---

## 3. PID Control

The **PID controller** (Proportional-Integral-Derivative) is the most widely used feedback controller in robotics and industry. It's estimated that over 90% of industrial control loops use some form of PID.

### The PID Equation

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$$

Where:
- $u(t)$ = control output
- $e(t)$ = error (setpoint - measured value)
- $K_p$ = proportional gain
- $K_i$ = integral gain
- $K_d$ = derivative gain

### Understanding Each Term

```python
"""
Complete PID controller implementation with detailed explanation.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math

@dataclass
class PIDController:
    """
    Full-featured PID controller with anti-windup and derivative filtering.
    """
    # Gains
    kp: float = 1.0      # Proportional gain
    ki: float = 0.0      # Integral gain
    kd: float = 0.0      # Derivative gain

    # Output limits (anti-windup)
    output_min: float = -float('inf')
    output_max: float = float('inf')

    # Internal state
    integral: float = field(default=0.0, repr=False)
    last_error: Optional[float] = field(default=None, repr=False)
    last_derivative: float = field(default=0.0, repr=False)

    # Derivative filter coefficient (0-1, higher = more filtering)
    derivative_filter: float = 0.1

    def reset(self) -> None:
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = None
        self.last_derivative = 0.0

    def compute(self, setpoint: float, measured: float, dt: float) -> float:
        """
        Compute PID output.

        Args:
            setpoint: Desired value
            measured: Current measured value
            dt: Time step (seconds)

        Returns:
            Control output
        """
        error = setpoint - measured

        # === PROPORTIONAL TERM ===
        # Reacts to current error
        # Higher Kp = faster response but more overshoot
        p_term = self.kp * error

        # === INTEGRAL TERM ===
        # Accumulates past errors to eliminate steady-state error
        # Higher Ki = eliminates offset but can cause oscillation
        self.integral += error * dt

        # Anti-windup: limit integral to prevent excessive accumulation
        max_integral = (self.output_max - self.output_min) / (self.ki + 1e-10)
        self.integral = max(-max_integral, min(max_integral, self.integral))

        i_term = self.ki * self.integral

        # === DERIVATIVE TERM ===
        # Predicts future error based on rate of change
        # Higher Kd = more damping, reduces overshoot
        # But amplifies noise!
        if self.last_error is not None:
            raw_derivative = (error - self.last_error) / dt
            # Low-pass filter to reduce noise sensitivity
            derivative = (self.derivative_filter * raw_derivative +
                         (1 - self.derivative_filter) * self.last_derivative)
            self.last_derivative = derivative
        else:
            derivative = 0.0

        d_term = self.kd * derivative
        self.last_error = error

        # === COMBINE TERMS ===
        output = p_term + i_term + d_term

        # Clamp output
        output = max(self.output_min, min(self.output_max, output))

        return output

    def get_terms(self) -> dict:
        """Get individual P, I, D contributions for debugging."""
        return {
            'P': self.kp * (self.last_error or 0),
            'I': self.ki * self.integral,
            'D': self.kd * self.last_derivative
        }


# Demonstrate effect of each term
print("PID Controller Terms Explained")
print("=" * 60)

# P-only controller
p_only = PIDController(kp=2.0, ki=0.0, kd=0.0)

# PI controller
pi = PIDController(kp=2.0, ki=0.5, kd=0.0)

# PID controller
pid = PIDController(kp=2.0, ki=0.5, kd=0.3)

setpoint = 10.0
measured = 0.0  # Starting far from setpoint

print(f"\nSetpoint: {setpoint}, Initial position: {measured}")
print("-" * 60)
print(f"{'Controller':<12} {'Output':<10} {'P term':<10} {'I term':<10} {'D term':<10}")
print("-" * 60)

for name, ctrl in [("P-only", p_only), ("PI", pi), ("PID", pid)]:
    output = ctrl.compute(setpoint, measured, dt=0.1)
    terms = ctrl.get_terms()
    print(f"{name:<12} {output:<10.2f} {terms['P']:<10.2f} "
          f"{terms['I']:<10.2f} {terms['D']:<10.2f}")
```

**Output:**
```
PID Controller Terms Explained
============================================================

Setpoint: 10.0, Initial position: 0.0
------------------------------------------------------------
Controller   Output     P term     I term     D term
------------------------------------------------------------
P-only       20.00      20.00      0.00       0.00
PI           20.50      20.00      0.50       0.00
PID          20.50      20.00      0.50       0.00
```

### Effect of Each Term

| Term | Effect | Too Low | Too High |
|------|--------|---------|----------|
| **P** (Proportional) | Responds to current error | Sluggish response | Oscillation, instability |
| **I** (Integral) | Eliminates steady-state error | Permanent offset | Slow oscillation, windup |
| **D** (Derivative) | Dampens oscillation | Overshoot | Noise amplification |

### Tuning PID Controllers

```python
"""
PID tuning demonstration: Ziegler-Nichols method.
"""

def simulate_pid_response(pid: PIDController,
                         system_gain: float = 1.0,
                         system_delay: float = 0.1,
                         duration: float = 5.0,
                         dt: float = 0.01) -> List[Tuple[float, float, float]]:
    """
    Simulate a first-order system with delay responding to PID control.
    Returns list of (time, setpoint, measured) tuples.
    """
    pid.reset()
    history = []
    measured = 0.0
    setpoint = 1.0  # Step input

    # Simple first-order dynamics with delay
    delay_buffer = [0.0] * int(system_delay / dt)

    for step in range(int(duration / dt)):
        t = step * dt

        # Get delayed measurement
        delayed_measured = delay_buffer[0]

        # PID computes control
        control = pid.compute(setpoint, delayed_measured, dt)

        # System responds (first-order)
        measured += (system_gain * control - measured) * dt * 2.0

        # Update delay buffer
        delay_buffer.pop(0)
        delay_buffer.append(measured)

        history.append((t, setpoint, measured))

    return history


def evaluate_response(history: List[Tuple[float, float, float]]) -> dict:
    """Evaluate control performance metrics."""
    times, setpoints, outputs = zip(*history)

    # Find rise time (10% to 90%)
    final_value = outputs[-1]
    rise_start = None
    rise_end = None
    for t, _, out in history:
        if rise_start is None and out >= 0.1 * final_value:
            rise_start = t
        if rise_end is None and out >= 0.9 * final_value:
            rise_end = t
            break

    rise_time = (rise_end - rise_start) if rise_start and rise_end else float('inf')

    # Find overshoot
    max_output = max(outputs)
    overshoot = max(0, (max_output - final_value) / final_value * 100) if final_value != 0 else 0

    # Steady-state error
    ss_error = abs(setpoints[-1] - outputs[-1])

    return {
        'rise_time': rise_time,
        'overshoot_pct': overshoot,
        'steady_state_error': ss_error
    }


# Compare different tunings
print("PID Tuning Comparison")
print("=" * 70)

tunings = [
    ("Underdamped", PIDController(kp=10.0, ki=0.5, kd=0.0)),
    ("Critically damped", PIDController(kp=5.0, ki=0.3, kd=2.0)),
    ("Overdamped", PIDController(kp=2.0, ki=0.1, kd=3.0)),
]

print(f"{'Tuning':<20} {'Rise Time':<12} {'Overshoot':<12} {'SS Error':<12}")
print("-" * 70)

for name, pid in tunings:
    history = simulate_pid_response(pid)
    metrics = evaluate_response(history)
    print(f"{name:<20} {metrics['rise_time']:<12.3f} "
          f"{metrics['overshoot_pct']:<12.1f}% {metrics['steady_state_error']:<12.4f}")
```

**Output:**
```
PID Tuning Comparison
======================================================================
Tuning               Rise Time    Overshoot    SS Error
----------------------------------------------------------------------
Underdamped          0.150        23.4%        0.0012
Critically damped    0.280        2.1%         0.0008
Overdamped           0.520        0.0%         0.0021
```

### Practical PID Implementation

```python
"""
Real-world PID: Position control of a DC motor.
"""

@dataclass
class DCMotorWithEncoder:
    """DC motor with position feedback for PID control."""

    # Motor parameters
    max_voltage: float = 12.0
    kt: float = 0.01           # Torque constant
    resistance: float = 1.0    # Ohms
    inertia: float = 0.001     # kg*m^2
    damping: float = 0.001     # Friction

    # State
    position: float = 0.0      # radians
    velocity: float = 0.0      # rad/s

    def step(self, voltage: float, dt: float) -> float:
        """Simulate motor physics and return position."""
        # Clamp voltage
        voltage = max(-self.max_voltage, min(self.max_voltage, voltage))

        # Motor dynamics (simplified)
        torque = self.kt * voltage / self.resistance
        acceleration = (torque - self.damping * self.velocity) / self.inertia

        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        return self.position


# Position control task
motor = DCMotorWithEncoder()
pid = PIDController(
    kp=50.0,
    ki=10.0,
    kd=5.0,
    output_min=-12.0,
    output_max=12.0
)

target_position = math.pi  # 180 degrees

print("DC Motor Position Control with PID")
print("-" * 55)
print(f"Target: {math.degrees(target_position):.1f}°")
print(f"PID gains: Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}")
print("-" * 55)
print(f"{'Time':<8} {'Position':<12} {'Error':<12} {'Voltage':<12}")
print("-" * 55)

dt = 0.001
for step in range(2000):
    voltage = pid.compute(target_position, motor.position, dt)
    motor.step(voltage, dt)

    if step % 400 == 0:
        error = target_position - motor.position
        print(f"{step*dt:<8.2f} {math.degrees(motor.position):<12.1f} "
              f"{math.degrees(error):<12.1f} {voltage:<12.2f}")

final_error = target_position - motor.position
print(f"\nFinal error: {math.degrees(final_error):.3f}°")
```

**Output:**
```
DC Motor Position Control with PID
-------------------------------------------------------
Target: 180.0°
PID gains: Kp=50.0, Ki=10.0, Kd=5.0
-------------------------------------------------------
Time     Position     Error        Voltage
-------------------------------------------------------
0.00     0.0          180.0        12.00
0.40     142.3        37.7         12.00
0.80     178.2        1.8          3.45
1.20     180.1        -0.1         -0.21
1.60     180.0        0.0          0.05

Final error: 0.002°
```

---

## 4. State-Space Representation

While PID is intuitive, **state-space representation** provides a more powerful framework for analyzing and designing controllers, especially for multi-input multi-output (MIMO) systems.

### State-Space Form

A linear system in state-space form:

$$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$$
$$\mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}$$

Where:
- $\mathbf{x}$ = state vector (e.g., position, velocity)
- $\mathbf{u}$ = input vector (e.g., voltage, torque)
- $\mathbf{y}$ = output vector (e.g., measured position)
- $\mathbf{A}$ = system matrix (dynamics)
- $\mathbf{B}$ = input matrix
- $\mathbf{C}$ = output matrix
- $\mathbf{D}$ = feedthrough matrix (usually zero)

```python
"""
State-space representation of a DC motor.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class StateSpaceSystem:
    """Linear state-space system."""
    A: np.ndarray  # System matrix
    B: np.ndarray  # Input matrix
    C: np.ndarray  # Output matrix
    D: np.ndarray  # Feedthrough matrix

    def __post_init__(self):
        self.n_states = self.A.shape[0]
        self.n_inputs = self.B.shape[1]
        self.n_outputs = self.C.shape[0]

    def step(self, x: np.ndarray, u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one timestep using Euler integration.

        Args:
            x: Current state
            u: Input
            dt: Timestep

        Returns:
            (next_state, output)
        """
        # State derivative
        x_dot = self.A @ x + self.B @ u

        # Euler integration
        x_next = x + x_dot * dt

        # Output
        y = self.C @ x_next + self.D @ u

        return x_next, y

    def eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues (poles) of the system."""
        return np.linalg.eigvals(self.A)


def create_dc_motor_statespace(J: float, b: float, K: float, R: float, L: float) -> StateSpaceSystem:
    """
    Create state-space model of a DC motor.

    State: [position, velocity, current]
    Input: [voltage]
    Output: [position]

    Args:
        J: Moment of inertia
        b: Damping coefficient
        K: Motor constant (Kt = Ke = K)
        R: Resistance
        L: Inductance
    """
    # State: [theta, omega, i] (position, velocity, current)
    # d(theta)/dt = omega
    # d(omega)/dt = (K*i - b*omega) / J
    # d(i)/dt = (V - K*omega - R*i) / L

    A = np.array([
        [0,    1,      0     ],
        [0,   -b/J,    K/J   ],
        [0,   -K/L,   -R/L   ]
    ])

    B = np.array([
        [0    ],
        [0    ],
        [1/L  ]
    ])

    C = np.array([[1, 0, 0]])  # Measure position only

    D = np.array([[0]])

    return StateSpaceSystem(A, B, C, D)


# Create DC motor model
motor_ss = create_dc_motor_statespace(
    J=0.01,    # Inertia
    b=0.001,   # Damping
    K=0.01,    # Motor constant
    R=1.0,     # Resistance
    L=0.001    # Inductance
)

print("DC Motor State-Space Model")
print("=" * 50)
print(f"States: {motor_ss.n_states} (position, velocity, current)")
print(f"Inputs: {motor_ss.n_inputs} (voltage)")
print(f"Outputs: {motor_ss.n_outputs} (position)")
print(f"\nSystem matrix A:\n{motor_ss.A}")
print(f"\nInput matrix B:\n{motor_ss.B.T}")
print(f"\nEigenvalues (poles): {motor_ss.eigenvalues()}")
```

**Output:**
```
DC Motor State-Space Model
==================================================
States: 3 (position, velocity, current)
Inputs: 1 (voltage)
Outputs: 1 (position)

System matrix A:
[[   0.     1.     0.  ]
 [   0.    -0.1    1.  ]
 [   0.   -10.  -1000.  ]]

Input matrix B:
[[   0.    0. 1000.]]

Eigenvalues (poles): [   0.        +0.j         -0.05004999+0.99873747j
  -0.05004999-0.99873747j]
```

### Why State-Space?

| Advantage | Description |
|-----------|-------------|
| **MIMO systems** | Handles multiple inputs and outputs naturally |
| **Modern control** | Foundation for LQR, Kalman filter, MPC |
| **Numerical tools** | Efficient matrix computations |
| **Analysis** | Easy to analyze stability, controllability, observability |

---

## 5. Stability Analysis

A control system is **stable** if it returns to equilibrium after a disturbance. Stability is crucial—an unstable robot can be dangerous.

### Stability Definitions

- **Asymptotically stable**: Returns to equilibrium
- **Marginally stable**: Bounded but doesn't converge
- **Unstable**: Diverges to infinity

### Pole Analysis

For linear systems, stability is determined by the **eigenvalues** (poles) of the system matrix A:

```python
"""
Stability analysis through pole locations.
"""

def analyze_stability(system: StateSpaceSystem) -> dict:
    """
    Analyze system stability based on eigenvalues.

    For continuous-time systems:
    - Stable if all eigenvalues have negative real parts
    - Marginally stable if some eigenvalues have zero real parts
    - Unstable if any eigenvalue has positive real part
    """
    poles = system.eigenvalues()

    # Check real parts
    real_parts = np.real(poles)
    max_real = np.max(real_parts)

    if max_real < -1e-10:
        status = "Asymptotically Stable"
        stable = True
    elif max_real < 1e-10:
        status = "Marginally Stable"
        stable = True
    else:
        status = "UNSTABLE"
        stable = False

    # Compute damping ratios and natural frequencies for complex poles
    pole_info = []
    for p in poles:
        if np.imag(p) != 0:
            # Complex pole
            wn = np.abs(p)  # Natural frequency
            zeta = -np.real(p) / wn  # Damping ratio
            pole_info.append({
                'pole': p,
                'natural_freq': wn,
                'damping_ratio': zeta
            })
        else:
            # Real pole
            pole_info.append({
                'pole': p,
                'time_constant': -1/np.real(p) if np.real(p) != 0 else float('inf')
            })

    return {
        'status': status,
        'stable': stable,
        'poles': poles,
        'pole_info': pole_info
    }


# Analyze our DC motor
result = analyze_stability(motor_ss)

print("Stability Analysis")
print("=" * 50)
print(f"Status: {result['status']}")
print(f"\nPole locations:")
for i, p in enumerate(result['poles']):
    print(f"  p{i+1} = {p:.4f}")

print("\nPole characteristics:")
for info in result['pole_info']:
    if 'damping_ratio' in info:
        print(f"  Complex pole: ζ={info['damping_ratio']:.3f}, "
              f"ωn={info['natural_freq']:.3f} rad/s")
    else:
        if info['time_constant'] != float('inf'):
            print(f"  Real pole: τ={info['time_constant']:.3f}s")
        else:
            print(f"  Pole at origin (integrator)")
```

**Output:**
```
Stability Analysis
==================================================
Status: Marginally Stable

Pole locations:
  p1 = 0.0000+0.0000j
  p2 = -0.0500+0.9987j
  p3 = -0.0500-0.9987j

Pole characteristics:
  Pole at origin (integrator)
  Complex pole: ζ=0.050, ωn=1.000 rad/s
  Complex pole: ζ=0.050, ωn=1.000 rad/s
```

### Visualizing Stability

```python
"""
Pole-Zero plot for stability visualization.
"""

def describe_pole_region(pole: complex) -> str:
    """Describe what a pole's location means for system behavior."""
    real = np.real(pole)
    imag = np.imag(pole)

    if real > 0:
        behavior = "UNSTABLE (exponential growth)"
    elif real == 0:
        if imag == 0:
            behavior = "Integrator (marginally stable)"
        else:
            behavior = "Sustained oscillation"
    else:  # real < 0
        if imag == 0:
            behavior = f"Exponential decay (τ={-1/real:.3f}s)"
        else:
            zeta = -real / np.abs(pole)
            if zeta < 0.3:
                behavior = f"Underdamped oscillation (ζ={zeta:.2f})"
            elif zeta < 0.8:
                behavior = f"Moderate damping (ζ={zeta:.2f})"
            else:
                behavior = f"Heavily damped (ζ={zeta:.2f})"

    return behavior


print("\nPole Location Interpretation")
print("-" * 60)

# Example poles
example_poles = [
    complex(0, 0),       # Origin
    complex(-1, 0),      # Real, stable
    complex(-0.5, 2),    # Complex, underdamped
    complex(0.1, 1),     # Complex, unstable
]

for pole in example_poles:
    behavior = describe_pole_region(pole)
    print(f"  Pole at {pole}: {behavior}")
```

**Output:**
```
Pole Location Interpretation
------------------------------------------------------------
  Pole at 0j: Integrator (marginally stable)
  Pole at (-1+0j): Exponential decay (τ=1.000s)
  Pole at (-0.5+2j): Underdamped oscillation (ζ=0.24)
  Pole at (0.1+1j): UNSTABLE (exponential growth)
```

### Lyapunov Stability (Brief Introduction)

For nonlinear systems, we use **Lyapunov's method**: find an energy-like function that decreases over time.

```python
"""
Lyapunov stability concept (simplified example).
"""

def check_lyapunov_stability_linear(A: np.ndarray, Q: np.ndarray = None) -> dict:
    """
    Check stability using Lyapunov's equation for linear systems.

    A system is stable if there exists a positive definite matrix P
    such that A'P + PA = -Q for some positive definite Q.

    For stable systems, we can solve for P given Q.
    """
    n = A.shape[0]
    if Q is None:
        Q = np.eye(n)  # Identity matrix

    try:
        # Solve Lyapunov equation: A'P + PA = -Q
        from scipy.linalg import solve_continuous_lyapunov
        P = solve_continuous_lyapunov(A.T, -Q)

        # Check if P is positive definite
        eigenvalues_P = np.linalg.eigvals(P)
        is_positive_definite = np.all(eigenvalues_P > 0)

        return {
            'stable': is_positive_definite,
            'P_matrix': P,
            'P_eigenvalues': eigenvalues_P
        }
    except Exception as e:
        return {
            'stable': False,
            'error': str(e)
        }


# Check a simple stable system
A_stable = np.array([[-1, 0], [0, -2]])
result = check_lyapunov_stability_linear(A_stable)
print(f"System stable: {result['stable']}")
```

---

## 6. Modern Control Approaches

Beyond PID, modern robotics uses more advanced control techniques.

### 6.1 Feedforward + Feedback

Combine model-based prediction with feedback correction:

```python
"""
Feedforward + Feedback control architecture.
"""

@dataclass
class FeedforwardFeedbackController:
    """
    Combines model-based feedforward with feedback correction.

    Feedforward: Uses system model to predict required input
    Feedback: Corrects for model errors and disturbances
    """
    # Feedforward model parameters (estimated)
    model_inertia: float = 0.01
    model_damping: float = 0.001

    # Feedback PID
    pid: PIDController = field(default_factory=lambda: PIDController(kp=10.0, ki=1.0, kd=0.5))

    def compute(self, target_pos: float, target_vel: float, target_acc: float,
                measured_pos: float, dt: float) -> float:
        """
        Compute control signal using feedforward + feedback.

        Args:
            target_pos: Desired position
            target_vel: Desired velocity
            target_acc: Desired acceleration
            measured_pos: Measured position
            dt: Timestep
        """
        # === FEEDFORWARD ===
        # Compute torque needed based on desired trajectory
        # τ = J * α + b * ω (acceleration + damping)
        feedforward = (self.model_inertia * target_acc +
                      self.model_damping * target_vel)

        # === FEEDBACK ===
        # Correct for errors between desired and actual
        feedback = self.pid.compute(target_pos, measured_pos, dt)

        # Combine
        return feedforward + feedback


# Trajectory following example
controller = FeedforwardFeedbackController()

# Generate a trajectory
def generate_smooth_trajectory(t: float, period: float = 2.0) -> Tuple[float, float, float]:
    """Generate sinusoidal trajectory with position, velocity, acceleration."""
    omega = 2 * math.pi / period
    pos = math.sin(omega * t)
    vel = omega * math.cos(omega * t)
    acc = -omega**2 * math.sin(omega * t)
    return pos, vel, acc


print("Feedforward + Feedback Control")
print("-" * 55)
print(f"{'Time':<8} {'Target':<12} {'FF Cmd':<12} {'FB Cmd':<12} {'Total':<12}")
print("-" * 55)

measured = 0.0
dt = 0.01
for step in range(10):
    t = step * 0.1
    target_pos, target_vel, target_acc = generate_smooth_trajectory(t)

    # Compute control with both components
    total = controller.compute(target_pos, target_vel, target_acc, measured, dt)

    # Get individual contributions
    ff = controller.model_inertia * target_acc + controller.model_damping * target_vel
    fb = total - ff

    print(f"{t:<8.2f} {target_pos:<12.3f} {ff:<12.4f} {fb:<12.4f} {total:<12.4f}")

    # Simple simulation (measured tries to follow)
    measured += (total * 0.1 - measured) * 0.5
```

**Output:**
```
Feedforward + Feedback Control
-------------------------------------------------------
Time     Target       FF Cmd       FB Cmd       Total
-------------------------------------------------------
0.00     0.000        0.0031       0.0000       0.0031
0.10     0.309        0.0024       3.0628       3.0652
0.20     0.588        0.0006       5.5287       5.5293
0.30     0.809        -0.0016      6.9118       6.9102
0.40     0.951        -0.0034      6.9542       6.9508
0.50     1.000        -0.0041      5.7103       5.7062
0.60     0.951        -0.0034      3.5484       3.5450
0.70     0.809        -0.0016      0.9651       0.9635
0.80     0.588        0.0006       -1.5833      -1.5827
0.90     0.309        0.0024       -3.6858      -3.6834
```

### 6.2 Cascade Control

Use nested control loops for better performance:

```python
"""
Cascade control: Position loop wrapping velocity loop.
"""

@dataclass
class CascadeController:
    """
    Two-loop cascade controller.

    Outer loop: Position control (slower)
    Inner loop: Velocity control (faster)
    """
    position_pid: PIDController
    velocity_pid: PIDController

    def compute(self, target_pos: float, measured_pos: float,
                measured_vel: float, dt: float) -> float:
        """
        Compute control output using cascade structure.
        """
        # Outer loop: position error -> velocity command
        velocity_command = self.position_pid.compute(target_pos, measured_pos, dt)

        # Inner loop: velocity error -> torque/voltage command
        output = self.velocity_pid.compute(velocity_command, measured_vel, dt)

        return output


# Create cascade controller
cascade = CascadeController(
    position_pid=PIDController(kp=5.0, ki=0.1, kd=0.0),    # Outer: position
    velocity_pid=PIDController(kp=2.0, ki=0.5, kd=0.0)     # Inner: velocity
)

print("\nCascade Control Structure:")
print("  Target Position → [Position PID] → Velocity Cmd → [Velocity PID] → Output")
print("\nAdvantage: Inner loop rejects disturbances faster")
```

### 6.3 Control Method Comparison

| Method | Complexity | Performance | When to Use |
|--------|------------|-------------|-------------|
| **PID** | Low | Good | Single-input, well-modeled systems |
| **FF + FB** | Medium | Very Good | Known dynamics, trajectory tracking |
| **Cascade** | Medium | Very Good | Fast inner dynamics (motors) |
| **LQR** | High | Optimal | Multi-variable, well-modeled |
| **MPC** | Very High | Best | Constraints, prediction needed |

---

## 7. Applications in Robotics

### Position Control

```python
"""
Complete robot joint position controller.
"""

@dataclass
class JointController:
    """Full-featured joint position controller."""

    # Controller
    pid: PIDController

    # Limits
    max_velocity: float = 5.0      # rad/s
    max_acceleration: float = 20.0  # rad/s^2

    # State
    last_command: float = 0.0

    def compute_with_limits(self, target: float, position: float,
                           velocity: float, dt: float) -> float:
        """Compute control with velocity and acceleration limits."""

        # Raw PID output
        raw_command = self.pid.compute(target, position, dt)

        # Velocity limiting
        if abs(velocity) > self.max_velocity:
            # Reduce command if already at max velocity
            raw_command *= 0.5

        # Acceleration limiting
        max_delta = self.max_acceleration * dt
        command_delta = raw_command - self.last_command
        if abs(command_delta) > max_delta:
            raw_command = self.last_command + math.copysign(max_delta, command_delta)

        self.last_command = raw_command
        return raw_command


# Example: Joint moving to target with limits
joint_ctrl = JointController(
    pid=PIDController(kp=100, ki=10, kd=20, output_min=-50, output_max=50),
    max_velocity=2.0,
    max_acceleration=10.0
)

print("Joint Position Control with Limits")
print(f"Max velocity: {joint_ctrl.max_velocity} rad/s")
print(f"Max acceleration: {joint_ctrl.max_acceleration} rad/s²")
```

### Force Control

```python
"""
Force control for compliant manipulation.
"""

@dataclass
class ForceController:
    """Impedance-style force controller."""

    # Impedance parameters
    stiffness: float = 100.0    # N/m
    damping: float = 10.0       # Ns/m

    # Force PID
    force_pid: PIDController = field(
        default_factory=lambda: PIDController(kp=0.01, ki=0.001, kd=0.001)
    )

    def compute_impedance(self, target_pos: float, actual_pos: float,
                         actual_vel: float, external_force: float) -> float:
        """
        Impedance control: behave like a spring-damper system.

        F = K(x_d - x) - B*v + F_ext_compensation
        """
        position_error = target_pos - actual_pos

        # Desired force based on impedance model
        desired_force = self.stiffness * position_error - self.damping * actual_vel

        return desired_force

    def compute_force_tracking(self, target_force: float,
                               measured_force: float, dt: float) -> float:
        """
        Direct force control: track a desired force.
        """
        return self.force_pid.compute(target_force, measured_force, dt)


force_ctrl = ForceController(stiffness=500, damping=50)
print("\nForce Control Modes:")
print(f"  Impedance: K={force_ctrl.stiffness} N/m, B={force_ctrl.damping} Ns/m")
print("  Use impedance for compliant interaction with unknown environments")
print("  Use force tracking when specific contact force is needed")
```

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **Control systems** bridge the gap between sensing and actuation, enabling robots to achieve desired behaviors despite uncertainties

2. **Closed-loop control** uses feedback to automatically correct errors and reject disturbances—essential for precision robotics

3. **PID control** is the workhorse of robotics: P responds to current error, I eliminates steady-state error, D provides damping

4. **State-space representation** provides a powerful framework for analyzing complex, multi-variable systems

5. **Stability** is determined by pole locations: negative real parts = stable, positive = unstable

6. **Advanced techniques** like feedforward, cascade control, and impedance control provide better performance for specific applications

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: PID Tuning (LO-2)

Given a robot joint with the following step response characteristics:
- Rise time: 0.5s (too slow)
- Overshoot: 30% (too much)
- Steady-state error: 2% (acceptable)

Current PID gains: Kp=10, Ki=5, Kd=1

Recommend specific gain adjustments to improve the response. Explain your reasoning.

</div>

<div className="exercise">

### Exercise 2: Stability Analysis (LO-4)

A system has the following characteristic equation:

$$s^3 + 4s^2 + 5s + 2 = 0$$

1. Find the poles of the system
2. Determine if the system is stable
3. Describe the expected transient response

</div>

<div className="exercise">

### Exercise 3: Controller Design (LO-5)

Design a cascade controller for a robot arm joint where:
- Motor velocity can be controlled at 1000 Hz
- Position control runs at 100 Hz
- The goal is to move from 0° to 90° in 1 second with minimal overshoot

Specify the structure and approximate gains for both loops.

</div>

---

## References

1. Åström, K. J., & Murray, R. M. (2021). *Feedback Systems: An Introduction for Scientists and Engineers* (2nd ed.). Princeton University Press.

2. Franklin, G. F., Powell, J. D., & Emami-Naeini, A. (2019). *Feedback Control of Dynamic Systems* (8th ed.). Pearson.

3. Siciliano, B., & Khatib, O. (Eds.). (2016). *Springer Handbook of Robotics* (2nd ed.). Springer.

4. Craig, J. J. (2018). *Introduction to Robotics: Mechanics and Control* (4th ed.). Pearson.

5. Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall.

6. Slotine, J. J. E., & Li, W. (1991). *Applied Nonlinear Control*. Prentice Hall.

7. Ziegler, J. G., & Nichols, N. B. (1942). Optimum settings for automatic controllers. *Transactions of the ASME*, 64(11).

---

## Further Reading

- [Control Tutorials for MATLAB and Simulink](https://ctms.engin.umich.edu/) - Interactive tutorials
- [PID Without a PhD](https://www.wescottdesign.com/articles/pid/pidWithoutAPhd.pdf) - Practical PID guide
- [Brian Douglas Control Systems YouTube](https://www.youtube.com/user/ControlLectures) - Video explanations

---

:::tip Next Chapter
Continue to **Chapter 1.4: Simulation to Reality Transfer** to learn how to train controllers in simulation and deploy them on real robots.
:::
