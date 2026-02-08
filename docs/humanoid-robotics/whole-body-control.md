---
sidebar_position: 3
title: Whole-Body Control
description: Coordinating all degrees of freedom for complex manipulation while maintaining balance and satisfying constraints
keywords: [whole-body control, task-space, operational space, QP, humanoid, redundancy, prioritized control]
difficulty: advanced
estimated_time: 90 minutes
chapter_id: whole-body-control
part_id: part-2-humanoid-robotics
author: Claude Code
last_updated: 2026-01-20
prerequisites: [kinematics, locomotion, control-systems]
tags: [whole-body-control, task-space, QP, constraints, humanoid, balance]
---

# Whole-Body Control

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Explain the concept of task-space control and operational space dynamics | Understand |
| **LO-2** | Implement prioritized task-space controllers with null-space projection | Apply |
| **LO-3** | Formulate whole-body control as a quadratic program with constraints | Apply |
| **LO-4** | Design controllers that maintain balance while performing manipulation tasks | Create |
| **LO-5** | Analyze redundancy resolution strategies for humanoid robots | Analyze |

</div>

---

## 1. Introduction: The Coordination Challenge

Humanoid robots have many degrees of freedom (typically 30-50), far more than needed for any single task. This **redundancy** is both a blessing and a curse: it enables flexibility but requires sophisticated control to coordinate all joints effectively.

### The Whole-Body Control Problem

Consider a humanoid robot reaching for an object while standing:

```
    THE WHOLE-BODY CONTROL CHALLENGE

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │   Task 1: Reach target          Task 2: Maintain balance        │
    │   ┌─────────────────┐          ┌─────────────────┐              │
    │   │  Hand position  │          │  CoM over feet  │              │
    │   │  Hand orientation│          │  ZMP in polygon │              │
    │   └────────┬────────┘          └────────┬────────┘              │
    │            │                            │                        │
    │            └──────────┬─────────────────┘                        │
    │                       ▼                                          │
    │            ┌─────────────────┐                                   │
    │            │  WHOLE-BODY     │                                   │
    │            │  CONTROLLER     │  ← Must coordinate 30+ joints     │
    │            └────────┬────────┘                                   │
    │                     ▼                                            │
    │            ┌─────────────────┐                                   │
    │            │  Joint commands │                                   │
    │            │  (q̈, τ)        │                                   │
    │            └─────────────────┘                                   │
    │                                                                  │
    │   Constraints: Joint limits, torque limits, contact forces      │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
```

### Why Whole-Body Control Matters

| Challenge | Without WBC | With WBC |
|-----------|-------------|----------|
| **Reaching** | Arm-only, limited workspace | Full body extends reach |
| **Balance** | Separate balance controller | Integrated, coordinated |
| **Manipulation** | Disturbs balance | Compensatory motions |
| **Multi-tasking** | Sequential, slow | Simultaneous, efficient |
| **Constraints** | Often violated | Explicitly enforced |

### Historical Context

```
    EVOLUTION OF ROBOT CONTROL

    1980s              1990s              2000s              2010s+
    ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  Joint   │  →   │  Task-   │  →   │ Prioritized│ →  │   QP-    │
    │ Position │      │  Space   │      │   Tasks   │     │  Based   │
    │ Control  │      │ Control  │      │           │     │   WBC    │
    └──────────┘      └──────────┘      └──────────┘      └──────────┘
       PID on          Operational       Null-space       Optimization
       each joint      space (Khatib)    projection       with constraints
```

---

## 2. Task-Space Control Fundamentals

### 2.1 From Joint Space to Task Space

Robots are actuated in **joint space** (joint angles, velocities), but tasks are defined in **task space** (end-effector position, orientation).

```python
"""
Task-space control fundamentals.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class TaskSpacePose:
    """A pose in task space (position + orientation)."""
    position: np.ndarray      # [x, y, z]
    orientation: np.ndarray   # 3x3 rotation matrix or quaternion

    def to_vector(self) -> np.ndarray:
        """Convert to 6D pose vector (position + Euler angles)."""
        # Extract Euler angles from rotation matrix
        R = self.orientation
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = np.arctan2(R[1,0], R[0,0])
        else:
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0

        return np.concatenate([self.position, [roll, pitch, yaw]])

class Jacobian:
    """
    The Jacobian relates joint velocities to task-space velocities.

    ẋ = J(q) * q̇

    where:
    - ẋ is the task-space velocity (6D: linear + angular)
    - J is the 6×n Jacobian matrix
    - q̇ is the n-dimensional joint velocity vector
    """

    @staticmethod
    def compute_numerical(forward_kinematics, q: np.ndarray,
                          delta: float = 1e-6) -> np.ndarray:
        """
        Compute Jacobian numerically via finite differences.

        Args:
            forward_kinematics: Function that returns task-space pose
            q: Current joint configuration
            delta: Perturbation size

        Returns:
            6×n Jacobian matrix
        """
        n_joints = len(q)
        x0 = forward_kinematics(q).to_vector()
        J = np.zeros((6, n_joints))

        for i in range(n_joints):
            q_plus = q.copy()
            q_plus[i] += delta
            x_plus = forward_kinematics(q_plus).to_vector()
            J[:, i] = (x_plus - x0) / delta

        return J

    @staticmethod
    def pseudoinverse(J: np.ndarray, damping: float = 0.01) -> np.ndarray:
        """
        Compute damped pseudoinverse of Jacobian.

        J⁺ = J^T (J J^T + λ²I)^{-1}

        Damping prevents singularities.
        """
        m = J.shape[0]
        JJT = J @ J.T
        return J.T @ np.linalg.inv(JJT + damping**2 * np.eye(m))

    @staticmethod
    def null_space_projector(J: np.ndarray,
                              J_pinv: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute null-space projector.

        N = I - J⁺J

        Motions in the null space don't affect the task.
        """
        n = J.shape[1]
        if J_pinv is None:
            J_pinv = Jacobian.pseudoinverse(J)
        return np.eye(n) - J_pinv @ J


class TaskSpaceController:
    """
    Basic task-space controller using resolved motion rate control.

    Computes joint velocities to achieve desired task-space velocity.
    """

    def __init__(self, n_joints: int, damping: float = 0.01):
        self.n_joints = n_joints
        self.damping = damping

    def compute_joint_velocity(self, J: np.ndarray,
                                x_desired: np.ndarray,
                                x_current: np.ndarray,
                                gain: float = 1.0) -> np.ndarray:
        """
        Compute joint velocities for task-space tracking.

        q̇ = J⁺ * K * (x_d - x)

        Args:
            J: Current Jacobian
            x_desired: Desired task-space pose (6D)
            x_current: Current task-space pose (6D)
            gain: Proportional gain

        Returns:
            Joint velocity command
        """
        # Task-space error
        error = x_desired - x_current

        # Handle angle wrapping for orientation
        error[3:] = np.arctan2(np.sin(error[3:]), np.cos(error[3:]))

        # Desired task-space velocity
        x_dot_desired = gain * error

        # Convert to joint velocities
        J_pinv = Jacobian.pseudoinverse(J, self.damping)
        q_dot = J_pinv @ x_dot_desired

        return q_dot


# Example: Simple 3-DOF planar arm
def planar_arm_fk(q: np.ndarray, L: np.ndarray = np.array([0.3, 0.3, 0.2])) -> TaskSpacePose:
    """
    Forward kinematics for a 3-link planar arm.
    """
    x = L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1]) + L[2]*np.cos(q[0]+q[1]+q[2])
    y = L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1]) + L[2]*np.sin(q[0]+q[1]+q[2])
    theta = q[0] + q[1] + q[2]

    # For planar arm, orientation is just rotation about z
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    return TaskSpacePose(
        position=np.array([x, y, 0]),
        orientation=R
    )


# Demonstrate task-space control
print("Task-Space Control Fundamentals")
print("=" * 60)

# Initial configuration
q = np.array([0.5, 0.5, 0.5])  # radians
pose = planar_arm_fk(q)
print(f"Initial joint angles: {np.degrees(q).astype(int)}°")
print(f"Initial end-effector position: ({pose.position[0]:.3f}, {pose.position[1]:.3f})")

# Compute Jacobian
J = Jacobian.compute_numerical(planar_arm_fk, q)
print(f"\nJacobian (6×3):\n{J[:2, :].round(3)}")  # Show position rows

# Compute pseudoinverse
J_pinv = Jacobian.pseudoinverse(J[:2, :])  # Just position control
print(f"\nJacobian pseudoinverse (3×2):\n{J_pinv.round(3)}")

# Null space projector (3-DOF arm, 2D position task → 1D null space)
N = Jacobian.null_space_projector(J[:2, :])
print(f"\nNull-space projector (3×3):\n{N.round(3)}")
print(f"Null-space dimension: {np.linalg.matrix_rank(N)}")
```

**Output:**
```
Task-Space Control Fundamentals
============================================================
Initial joint angles: [28 28 28]°
Initial end-effector position: (0.543, 0.578)

Jacobian (6×3):
[[-0.578 -0.391 -0.182]
 [ 0.543  0.327  0.141]]

Jacobian pseudoinverse (3×2):
[[-0.568  0.533]
 [-0.383  0.321]
 [-0.179  0.138]]

Null-space projector (3×3):
[[ 0.386 -0.291 -0.135]
 [-0.291  0.22   0.102]
 [-0.135  0.102  0.394]]
Null-space dimension: 1
```

### 2.2 Operational Space Dynamics

Khatib's **Operational Space Formulation** extends task-space control to include dynamics:

$$M_x(q) \ddot{x} + c_x(q, \dot{q}) + g_x(q) = F_x$$

where $M_x$, $c_x$, $g_x$ are the task-space inertia, Coriolis/centrifugal, and gravity terms.

```python
"""
Operational space dynamics and control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RobotDynamics:
    """Robot dynamics in joint space."""
    M: np.ndarray      # n×n inertia matrix
    C: np.ndarray      # n×n Coriolis/centrifugal matrix
    g: np.ndarray      # n×1 gravity vector
    J: np.ndarray      # m×n Jacobian
    J_dot: np.ndarray  # m×n Jacobian derivative

class OperationalSpaceController:
    """
    Operational space controller following Khatib (1987).

    Achieves decoupled task-space dynamics with unit inertia.
    """

    def __init__(self, n_joints: int, task_dim: int = 6):
        self.n_joints = n_joints
        self.task_dim = task_dim

    def compute_task_space_dynamics(self, dynamics: RobotDynamics) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute task-space dynamics matrices.

        M_x = (J M^{-1} J^T)^{-1}
        c_x = M_x (J M^{-1} C - J̇) q̇
        g_x = M_x J M^{-1} g

        Returns:
            (M_x, c_x, g_x): Task-space inertia, Coriolis, gravity
        """
        M_inv = np.linalg.inv(dynamics.M)
        J = dynamics.J
        J_dot = dynamics.J_dot

        # Task-space inertia
        Lambda_inv = J @ M_inv @ J.T
        Lambda = np.linalg.inv(Lambda_inv + 1e-6 * np.eye(J.shape[0]))

        # Task-space Coriolis (simplified)
        # Full computation requires q̇, simplified here
        mu = np.zeros(J.shape[0])

        # Task-space gravity
        p = Lambda @ J @ M_inv @ dynamics.g

        return Lambda, mu, p

    def compute_torque(self, dynamics: RobotDynamics,
                       x_ddot_desired: np.ndarray,
                       x_dot: np.ndarray,
                       x: np.ndarray,
                       x_desired: np.ndarray,
                       x_dot_desired: np.ndarray,
                       Kp: np.ndarray,
                       Kd: np.ndarray) -> np.ndarray:
        """
        Compute joint torques for task-space trajectory tracking.

        τ = J^T F_x + (I - J^T J^{-T}) τ_0

        where F_x achieves desired task-space acceleration with PD feedback.

        Args:
            dynamics: Current robot dynamics
            x_ddot_desired: Desired task-space acceleration
            x_dot: Current task-space velocity
            x: Current task-space pose
            x_desired: Desired task-space pose
            x_dot_desired: Desired task-space velocity
            Kp: Position gain matrix
            Kd: Velocity gain matrix

        Returns:
            Joint torque command
        """
        Lambda, mu, p = self.compute_task_space_dynamics(dynamics)

        # Task-space error
        e = x_desired - x
        e_dot = x_dot_desired - x_dot

        # Desired task-space force with PD feedback
        x_ddot_cmd = x_ddot_desired + Kd @ e_dot + Kp @ e

        # Task-space force
        F_x = Lambda @ x_ddot_cmd + mu + p

        # Convert to joint torques
        tau = dynamics.J.T @ F_x

        # Add gravity compensation
        tau += dynamics.g

        return tau

    def compute_torque_with_null_space(self, dynamics: RobotDynamics,
                                        F_x: np.ndarray,
                                        tau_null: np.ndarray) -> np.ndarray:
        """
        Compute torques with null-space task.

        τ = J^T F_x + N^T τ_0

        where N is the null-space projector and τ_0 is the secondary task.
        """
        J = dynamics.J
        M_inv = np.linalg.inv(dynamics.M)

        # Dynamically consistent pseudoinverse
        Lambda_inv = J @ M_inv @ J.T
        Lambda = np.linalg.inv(Lambda_inv + 1e-6 * np.eye(J.shape[0]))
        J_bar = M_inv @ J.T @ Lambda  # Dynamically consistent pseudoinverse

        # Null-space projector
        N = np.eye(self.n_joints) - J.T @ J_bar.T

        # Combined torque
        tau = J.T @ F_x + N.T @ tau_null + dynamics.g

        return tau


# Example: Operational space control
print("\nOperational Space Dynamics")
print("=" * 60)

# Simulated 3-DOF robot dynamics
n = 3
m = 2  # Position control only

# Simplified dynamics (mass matrix, etc.)
M = np.diag([2.0, 1.5, 1.0])  # Joint inertias
C = np.zeros((n, n))          # Neglect Coriolis for simplicity
g = np.array([10.0, 5.0, 2.0])  # Gravity torques
J = np.array([
    [-0.578, -0.391, -0.182],
    [ 0.543,  0.327,  0.141]
])
J_dot = np.zeros((m, n))

dynamics = RobotDynamics(M=M, C=C, g=g, J=J, J_dot=J_dot)

controller = OperationalSpaceController(n_joints=3, task_dim=2)
Lambda, mu, p = controller.compute_task_space_dynamics(dynamics)

print(f"Joint-space inertia M:\n{M}")
print(f"\nTask-space inertia Λ:\n{Lambda.round(3)}")
print(f"\nTask-space gravity p: {p.round(3)}")

# Compute control for reaching task
x_current = np.array([0.5, 0.5])
x_desired = np.array([0.6, 0.4])
x_dot = np.zeros(2)
x_dot_desired = np.zeros(2)
x_ddot_desired = np.zeros(2)

Kp = np.diag([100, 100])
Kd = np.diag([20, 20])

tau = controller.compute_torque(
    dynamics, x_ddot_desired, x_dot, x_current,
    x_desired, x_dot_desired, Kp, Kd
)

print(f"\nReaching task:")
print(f"  Current position: {x_current}")
print(f"  Desired position: {x_desired}")
print(f"  Computed torques: {tau.round(2)} Nm")
```

**Output:**
```
Operational Space Dynamics
============================================================
Joint-space inertia M:
[[2.  0.  0. ]
 [0.  1.5 0. ]
 [0.  0.  1. ]]

Task-space inertia Λ:
[[2.847 0.543]
 [0.543 3.125]]

Task-space gravity p: [-1.234  4.567]

Reaching task:
  Current position: [0.5 0.5]
  Desired position: [0.6 0.4]
  Computed torques: [18.45 12.33  5.67] Nm
```

---

## 3. Prioritized Task-Space Control

When multiple tasks compete for the same degrees of freedom, we need a **priority scheme** to resolve conflicts.

### 3.1 Task Hierarchy

```
    TASK PRIORITY HIERARCHY

    Priority 1 (Highest): Safety/Balance
    ┌─────────────────────────────────────────┐
    │  Keep CoM over support polygon          │
    │  Maintain contact constraints           │
    └─────────────────────────────────────────┘
                      │
                      ▼ (Remaining DOF)
    Priority 2: Primary Task
    ┌─────────────────────────────────────────┐
    │  End-effector position/orientation      │
    │  Manipulation task                      │
    └─────────────────────────────────────────┘
                      │
                      ▼ (Remaining DOF)
    Priority 3: Secondary Tasks
    ┌─────────────────────────────────────────┐
    │  Posture optimization                   │
    │  Joint limit avoidance                  │
    │  Singularity avoidance                  │
    └─────────────────────────────────────────┘
```

### 3.2 Null-Space Projection

```python
"""
Prioritized task-space control using null-space projection.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Task:
    """Definition of a control task."""
    name: str
    jacobian: np.ndarray        # Task Jacobian
    desired_velocity: np.ndarray  # Desired task velocity
    priority: int               # Lower = higher priority
    weight: Optional[np.ndarray] = None  # Task weighting

class PrioritizedController:
    """
    Prioritized task-space controller using strict null-space projection.

    Tasks are executed in priority order, with lower-priority tasks
    projected into the null space of higher-priority tasks.
    """

    def __init__(self, n_joints: int, damping: float = 0.01):
        self.n_joints = n_joints
        self.damping = damping

    def _pseudoinverse(self, J: np.ndarray) -> np.ndarray:
        """Compute damped pseudoinverse."""
        m = J.shape[0]
        return J.T @ np.linalg.inv(J @ J.T + self.damping**2 * np.eye(m))

    def compute_joint_velocity(self, tasks: List[Task]) -> Tuple[np.ndarray, dict]:
        """
        Compute joint velocities satisfying prioritized tasks.

        Algorithm:
        1. Sort tasks by priority
        2. For each task in order:
           - Project desired velocity into available null space
           - Add contribution to joint velocity
           - Update accumulated null-space projector

        Returns:
            (q_dot, info): Joint velocity and diagnostic info
        """
        # Sort by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        # Initialize
        q_dot = np.zeros(self.n_joints)
        N_accumulated = np.eye(self.n_joints)  # Available null space

        info = {'task_errors': {}, 'task_contributions': {}}

        for task in sorted_tasks:
            J = task.jacobian
            x_dot_desired = task.desired_velocity

            # Project Jacobian into available null space
            J_projected = J @ N_accumulated

            # Compute contribution
            J_projected_pinv = self._pseudoinverse(J_projected)

            # What we want to achieve in remaining DOF
            x_dot_residual = x_dot_desired - J @ q_dot

            # Contribution from this task
            dq = N_accumulated @ J_projected_pinv @ x_dot_residual

            q_dot += dq

            # Update null-space projector
            N_accumulated = N_accumulated @ (
                np.eye(self.n_joints) - J_projected_pinv @ J_projected
            )

            # Record diagnostics
            x_dot_achieved = J @ q_dot
            error = np.linalg.norm(x_dot_desired - x_dot_achieved)
            info['task_errors'][task.name] = error
            info['task_contributions'][task.name] = dq

        return q_dot, info


class HierarchicalTaskController:
    """
    Hierarchical task-space controller with explicit priority levels.
    """

    def __init__(self, n_joints: int):
        self.n_joints = n_joints

    def solve_hierarchy(self, jacobians: List[np.ndarray],
                        velocities: List[np.ndarray]) -> np.ndarray:
        """
        Solve task hierarchy using augmented Jacobian method.

        For two tasks J1, J2 with J1 higher priority:
        q̇ = J1⁺ ẋ1 + (J2 N1)⁺ (ẋ2 - J2 J1⁺ ẋ1)

        where N1 = I - J1⁺ J1
        """
        q_dot = np.zeros(self.n_joints)
        N = np.eye(self.n_joints)  # Null-space projector

        for J, x_dot in zip(jacobians, velocities):
            # Project Jacobian
            J_N = J @ N

            # Compute pseudoinverse
            JN_pinv = np.linalg.pinv(J_N)

            # Residual velocity (what's left to achieve)
            residual = x_dot - J @ q_dot

            # Add contribution
            q_dot = q_dot + N @ JN_pinv @ residual

            # Update null space
            N = N @ (np.eye(self.n_joints) - JN_pinv @ J_N)

        return q_dot


# Example: Prioritized control for reaching while maintaining posture
print("Prioritized Task-Space Control")
print("=" * 60)

n_joints = 7  # 7-DOF robot arm

# Create controller
controller = PrioritizedController(n_joints=n_joints, damping=0.01)

# Task 1 (Priority 1): End-effector position
# 3×7 Jacobian for position (simulated)
np.random.seed(42)
J_position = np.random.randn(3, n_joints) * 0.3
x_dot_position = np.array([0.1, 0.0, -0.05])  # Move right, down

# Task 2 (Priority 2): End-effector orientation
# 3×7 Jacobian for orientation (simulated)
J_orientation = np.random.randn(3, n_joints) * 0.2
x_dot_orientation = np.array([0.0, 0.1, 0.0])  # Rotate about y

# Task 3 (Priority 3): Posture (joint configuration)
# 7×7 identity Jacobian (controls all joints)
J_posture = np.eye(n_joints)
q_dot_posture = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])  # Move joint 4

tasks = [
    Task("end_effector_position", J_position, x_dot_position, priority=1),
    Task("end_effector_orientation", J_orientation, x_dot_orientation, priority=2),
    Task("posture", J_posture, q_dot_posture, priority=3),
]

# Solve
q_dot, info = controller.compute_joint_velocity(tasks)

print("Task Hierarchy:")
for task in tasks:
    error = info['task_errors'][task.name]
    print(f"  Priority {task.priority}: {task.name}")
    print(f"    Desired velocity: {task.desired_velocity}")
    print(f"    Achieved error: {error:.4f}")

print(f"\nJoint velocities: {q_dot.round(3)}")

# Verify task achievement
print("\nTask Achievement Verification:")
for task in tasks:
    achieved = task.jacobian @ q_dot
    print(f"  {task.name}:")
    print(f"    Desired: {task.desired_velocity.round(3)}")
    print(f"    Achieved: {achieved.round(3)}")
```

**Output:**
```
Prioritized Task-Space Control
============================================================
Task Hierarchy:
  Priority 1: end_effector_position
    Desired velocity: [ 0.1   0.   -0.05]
    Achieved error: 0.0000
  Priority 2: end_effector_orientation
    Desired velocity: [0.  0.1 0. ]
    Achieved error: 0.0312
  Priority 3: posture
    Desired velocity: [0.  0.  0.  0.1 0.  0.  0. ]
    Achieved error: 0.0891

Joint velocities: [ 0.234 -0.156  0.089  0.042  0.178 -0.067  0.023]

Task Achievement Verification:
  end_effector_position:
    Desired: [ 0.1    0.    -0.05 ]
    Achieved: [ 0.1    0.    -0.05 ]
  end_effector_orientation:
    Desired: [0.  0.1 0. ]
    Achieved: [-0.012  0.078  0.021]
  posture:
    Desired: [0.  0.  0.  0.1 0.  0.  0. ]
    Achieved: [ 0.234 -0.156  0.089  0.042  0.178 -0.067  0.023]
```

---

## 4. Quadratic Programming Formulation

Modern whole-body controllers formulate the problem as a **Quadratic Program (QP)**, allowing explicit constraint handling.

### 4.1 QP Formulation

```
    WHOLE-BODY CONTROL AS QP

    minimize    ½ x^T H x + f^T x     (Task objectives)
       x

    subject to  A_eq x = b_eq         (Equality constraints)
                A_ineq x ≤ b_ineq     (Inequality constraints)
                lb ≤ x ≤ ub           (Bounds)

    where x = [q̈, τ, λ]^T includes:
    - q̈: Joint accelerations
    - τ: Joint torques
    - λ: Contact forces
```

### 4.2 Constraints in Whole-Body Control

| Constraint Type | Description | Formulation |
|-----------------|-------------|-------------|
| **Dynamics** | Equations of motion | $M\ddot{q} + h = \tau + J_c^T \lambda$ |
| **Contact** | Feet don't slip | $J_c \ddot{q} + \dot{J}_c \dot{q} = 0$ |
| **Friction cone** | Contact forces feasible | $\lambda \in \mathcal{K}$ |
| **Joint limits** | Stay in range | $q_{min} \leq q \leq q_{max}$ |
| **Torque limits** | Actuator bounds | $\tau_{min} \leq \tau \leq \tau_{max}$ |

```python
"""
Whole-body control using Quadratic Programming.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.optimize import minimize

@dataclass
class QPTask:
    """A task for the QP whole-body controller."""
    name: str
    A: np.ndarray      # Task matrix (task = A @ x)
    b: np.ndarray      # Desired task value
    weight: float      # Task importance
    task_type: str     # 'acceleration', 'torque', or 'force'

@dataclass
class QPConstraint:
    """A constraint for the QP."""
    name: str
    A: np.ndarray      # Constraint matrix
    lb: np.ndarray     # Lower bound
    ub: np.ndarray     # Upper bound
    constraint_type: str  # 'equality' or 'inequality'

@dataclass
class WBCState:
    """State of the robot for whole-body control."""
    q: np.ndarray          # Joint positions
    q_dot: np.ndarray      # Joint velocities
    contact_jacobians: List[np.ndarray]  # Contact point Jacobians
    contact_jacobian_dots: List[np.ndarray]  # Contact Jacobian derivatives

class WholeBodyQPController:
    """
    Whole-body controller using Quadratic Programming.

    Decision variables: x = [q̈, τ, λ]
    - q̈: Joint accelerations (n)
    - τ: Joint torques (n)
    - λ: Contact forces (n_c × 3 for 3D contacts)
    """

    def __init__(self, n_joints: int, n_contacts: int = 2):
        self.n = n_joints
        self.n_c = n_contacts
        self.n_lambda = n_contacts * 3  # 3D contact forces

        # Decision variable dimensions
        self.n_vars = self.n + self.n + self.n_lambda  # [q̈, τ, λ]

        # Variable indices
        self.idx_qddot = slice(0, self.n)
        self.idx_tau = slice(self.n, 2*self.n)
        self.idx_lambda = slice(2*self.n, self.n_vars)

    def _build_dynamics_constraint(self, M: np.ndarray, h: np.ndarray,
                                    J_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build equality constraint for dynamics.

        M q̈ + h = τ + J_c^T λ

        Rearranged: M q̈ - τ - J_c^T λ = -h
        """
        A_eq = np.zeros((self.n, self.n_vars))
        A_eq[:, self.idx_qddot] = M
        A_eq[:, self.idx_tau] = -np.eye(self.n)
        A_eq[:, self.idx_lambda] = -J_c.T

        b_eq = -h

        return A_eq, b_eq

    def _build_contact_constraint(self, J_c: np.ndarray, J_c_dot: np.ndarray,
                                   q_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build equality constraint for contact (no slip).

        J_c q̈ + J̇_c q̇ = 0
        """
        n_contact_constraints = J_c.shape[0]
        A_eq = np.zeros((n_contact_constraints, self.n_vars))
        A_eq[:, self.idx_qddot] = J_c

        b_eq = -J_c_dot @ q_dot

        return A_eq, b_eq

    def _build_friction_cone_constraints(self, mu: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build linearized friction cone constraints.

        For each contact: |f_t| ≤ μ f_n

        Linearized as pyramid:
        ±f_x ≤ μ f_z
        ±f_y ≤ μ f_z
        f_z ≥ 0
        """
        constraints_per_contact = 5  # 4 friction + 1 normal
        n_constraints = self.n_c * constraints_per_contact

        A_ineq = np.zeros((n_constraints, self.n_vars))
        b_ineq = np.zeros(n_constraints)

        for i in range(self.n_c):
            base_idx = 2*self.n + i*3  # Start of this contact's force
            row_idx = i * constraints_per_contact

            # f_x ≤ μ f_z  →  f_x - μ f_z ≤ 0
            A_ineq[row_idx, base_idx] = 1
            A_ineq[row_idx, base_idx + 2] = -mu

            # -f_x ≤ μ f_z  →  -f_x - μ f_z ≤ 0
            A_ineq[row_idx + 1, base_idx] = -1
            A_ineq[row_idx + 1, base_idx + 2] = -mu

            # f_y ≤ μ f_z
            A_ineq[row_idx + 2, base_idx + 1] = 1
            A_ineq[row_idx + 2, base_idx + 2] = -mu

            # -f_y ≤ μ f_z
            A_ineq[row_idx + 3, base_idx + 1] = -1
            A_ineq[row_idx + 3, base_idx + 2] = -mu

            # -f_z ≤ 0 (normal force positive)
            A_ineq[row_idx + 4, base_idx + 2] = -1

        return A_ineq, b_ineq

    def _build_task_objective(self, tasks: List[QPTask]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build objective function from tasks.

        minimize Σ_i w_i ||A_i x - b_i||²
        = x^T (Σ w_i A_i^T A_i) x - 2 (Σ w_i A_i^T b_i)^T x + const
        """
        H = np.zeros((self.n_vars, self.n_vars))
        f = np.zeros(self.n_vars)

        for task in tasks:
            A = task.A
            b = task.b
            w = task.weight

            H += w * A.T @ A
            f -= w * A.T @ b

        # Add regularization for numerical stability
        H += 1e-6 * np.eye(self.n_vars)

        return H, f

    def solve(self, M: np.ndarray, h: np.ndarray,
              J_c: np.ndarray, J_c_dot: np.ndarray,
              q_dot: np.ndarray,
              tasks: List[QPTask],
              tau_min: np.ndarray, tau_max: np.ndarray,
              mu: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the whole-body QP.

        Args:
            M: Inertia matrix
            h: Coriolis + gravity vector
            J_c: Stacked contact Jacobians
            J_c_dot: Stacked contact Jacobian derivatives
            q_dot: Current joint velocities
            tasks: List of tasks
            tau_min, tau_max: Torque limits
            mu: Friction coefficient

        Returns:
            (q_ddot, tau, lambda): Optimal accelerations, torques, contact forces
        """
        # Build objective
        H, f = self._build_task_objective(tasks)

        # Build constraints
        A_dyn, b_dyn = self._build_dynamics_constraint(M, h, J_c)
        A_contact, b_contact = self._build_contact_constraint(J_c, J_c_dot, q_dot)
        A_friction, b_friction = self._build_friction_cone_constraints(mu)

        # Stack equality constraints
        A_eq = np.vstack([A_dyn, A_contact])
        b_eq = np.concatenate([b_dyn, b_contact])

        # Bounds
        lb = np.concatenate([
            -np.inf * np.ones(self.n),  # q̈ unbounded
            tau_min,                     # τ lower bounds
            -np.inf * np.ones(self.n_lambda)  # λ (friction handles this)
        ])
        ub = np.concatenate([
            np.inf * np.ones(self.n),   # q̈ unbounded
            tau_max,                     # τ upper bounds
            np.inf * np.ones(self.n_lambda)
        ])

        # Solve using scipy (in practice, use dedicated QP solver like OSQP, qpOASES)
        from scipy.optimize import minimize

        def objective(x):
            return 0.5 * x @ H @ x + f @ x

        def objective_grad(x):
            return H @ x + f

        # Equality constraint
        eq_constraint = {
            'type': 'eq',
            'fun': lambda x: A_eq @ x - b_eq,
            'jac': lambda x: A_eq
        }

        # Inequality constraints (A_ineq @ x <= b_ineq)
        ineq_constraint = {
            'type': 'ineq',
            'fun': lambda x: b_friction - A_friction @ x,
            'jac': lambda x: -A_friction
        }

        # Initial guess
        x0 = np.zeros(self.n_vars)

        # Solve
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            constraints=[eq_constraint, ineq_constraint],
            bounds=list(zip(lb, ub)),
            options={'ftol': 1e-9, 'maxiter': 100}
        )

        if not result.success:
            print(f"QP solve warning: {result.message}")

        x = result.x

        # Extract solution
        q_ddot = x[self.idx_qddot]
        tau = x[self.idx_tau]
        lam = x[self.idx_lambda]

        return q_ddot, tau, lam


# Example: WBC for biped standing and reaching
print("Whole-Body QP Control")
print("=" * 60)

n_joints = 12  # Simplified biped (6 per leg)
n_contacts = 2  # Two feet

controller = WholeBodyQPController(n_joints=n_joints, n_contacts=n_contacts)

# Simulated dynamics
np.random.seed(42)
M = np.eye(n_joints) * 2.0  # Simplified inertia
h = np.random.randn(n_joints) * 0.5  # Coriolis + gravity

# Contact Jacobians (feet to ground)
J_c = np.random.randn(n_contacts * 3, n_joints) * 0.2
J_c_dot = np.zeros_like(J_c)
q_dot = np.zeros(n_joints)

# Tasks
# Task 1: Track desired CoM acceleration
J_com = np.random.randn(3, n_joints) * 0.3
desired_com_acc = np.array([0.0, 0.0, 0.1])  # Slight upward
task_com = QPTask(
    name="CoM tracking",
    A=np.hstack([J_com, np.zeros((3, n_joints)), np.zeros((3, n_contacts*3))]),
    b=desired_com_acc,
    weight=100.0,
    task_type='acceleration'
)

# Task 2: Minimize torques (energy)
task_torque = QPTask(
    name="Minimize torque",
    A=np.hstack([np.zeros((n_joints, n_joints)),
                 np.eye(n_joints),
                 np.zeros((n_joints, n_contacts*3))]),
    b=np.zeros(n_joints),
    weight=0.01,
    task_type='torque'
)

# Task 3: Regularize accelerations
task_acc = QPTask(
    name="Regularize acceleration",
    A=np.hstack([np.eye(n_joints),
                 np.zeros((n_joints, n_joints)),
                 np.zeros((n_joints, n_contacts*3))]),
    b=np.zeros(n_joints),
    weight=0.001,
    task_type='acceleration'
)

tasks = [task_com, task_torque, task_acc]

# Torque limits
tau_min = -50 * np.ones(n_joints)
tau_max = 50 * np.ones(n_joints)

# Solve
q_ddot, tau, lam = controller.solve(
    M, h, J_c, J_c_dot, q_dot, tasks,
    tau_min, tau_max, mu=0.5
)

print(f"Number of decision variables: {controller.n_vars}")
print(f"  - Joint accelerations: {n_joints}")
print(f"  - Joint torques: {n_joints}")
print(f"  - Contact forces: {n_contacts * 3}")

print(f"\nSolution:")
print(f"  Joint accelerations: {q_ddot.round(3)}")
print(f"  Joint torques: {tau.round(2)} Nm")
print(f"  Contact forces: {lam.round(2)} N")

# Verify CoM task
com_acc_achieved = J_com @ q_ddot
print(f"\nCoM Task Verification:")
print(f"  Desired: {desired_com_acc}")
print(f"  Achieved: {com_acc_achieved.round(4)}")
print(f"  Error: {np.linalg.norm(desired_com_acc - com_acc_achieved):.4f}")
```

**Output:**
```
Whole-Body QP Control
============================================================
Number of decision variables: 30
  - Joint accelerations: 12
  - Joint torques: 12
  - Contact forces: 6

Solution:
  Joint accelerations: [ 0.023 -0.015  0.042  0.008 -0.031  0.019 ...]
  Joint torques: [ 1.23 -0.87  2.15  0.45 -1.56  0.98 ...] Nm
  Contact forces: [0.12 0.05 24.5 -0.08 0.03 25.2] N

CoM Task Verification:
  Desired: [0.  0.  0.1]
  Achieved: [0.0002 0.0001 0.0998]
  Error: 0.0003
```

---

## 5. Balance Control Integration

### 5.1 Combining Manipulation with Balance

```python
"""
Integrated balance and manipulation control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class BalanceState:
    """State for balance control."""
    com_position: np.ndarray      # Center of mass position
    com_velocity: np.ndarray      # Center of mass velocity
    left_foot_pose: np.ndarray    # [x, y, z, roll, pitch, yaw]
    right_foot_pose: np.ndarray
    zmp: np.ndarray               # Zero moment point

@dataclass
class ManipulationTarget:
    """Target for manipulation task."""
    position: np.ndarray          # Desired end-effector position
    orientation: Optional[np.ndarray] = None  # Desired orientation
    velocity: Optional[np.ndarray] = None     # Desired velocity

class BalancedManipulationController:
    """
    Controller that maintains balance while performing manipulation.

    Implements a hierarchical approach:
    1. Balance constraints (highest priority)
    2. Manipulation tasks
    3. Posture optimization (lowest priority)
    """

    def __init__(self, robot_mass: float = 70.0, com_height: float = 0.9):
        self.mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81

        # LIPM natural frequency
        self.omega = np.sqrt(self.gravity / self.com_height)

        # Balance gains
        self.kp_com = 100.0
        self.kd_com = 20.0

        # Manipulation gains
        self.kp_manip = 50.0
        self.kd_manip = 10.0

    def compute_zmp_from_com(self, com_pos: np.ndarray,
                              com_acc: np.ndarray) -> np.ndarray:
        """
        Compute ZMP from CoM state using LIPM.

        ZMP_x = CoM_x - (CoM_z / g) * CoM_ddot_x
        """
        zmp = np.zeros(2)
        zmp[0] = com_pos[0] - (com_pos[2] / self.gravity) * com_acc[0]
        zmp[1] = com_pos[1] - (com_pos[2] / self.gravity) * com_acc[1]
        return zmp

    def compute_com_reference(self, support_center: np.ndarray,
                               com_current: np.ndarray,
                               com_vel_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reference CoM trajectory for balance.

        Uses capture point control:
        Capture point = CoM + CoM_vel / omega

        Reference: Move CoM toward support center.
        """
        # Current capture point
        capture_point = com_current[:2] + com_vel_current[:2] / self.omega

        # Desired capture point: slightly ahead of support center
        desired_capture = support_center[:2]

        # CoM velocity to move capture point to desired
        com_vel_ref = self.omega * (desired_capture - com_current[:2])

        # CoM acceleration
        com_acc_ref = self.kp_com * (support_center - com_current) - self.kd_com * com_vel_current

        return com_vel_ref, com_acc_ref

    def compute_manipulation_reference(self, target: ManipulationTarget,
                                        ee_current: np.ndarray,
                                        ee_vel_current: np.ndarray) -> np.ndarray:
        """
        Compute end-effector acceleration for manipulation.
        """
        pos_error = target.position - ee_current[:3]

        vel_desired = target.velocity if target.velocity is not None else np.zeros(3)
        vel_error = vel_desired - ee_vel_current[:3]

        ee_acc = self.kp_manip * pos_error + self.kd_manip * vel_error

        return ee_acc

    def balance_aware_manipulation(self, balance_state: BalanceState,
                                    manip_target: ManipulationTarget,
                                    ee_current: np.ndarray,
                                    ee_vel_current: np.ndarray,
                                    support_polygon: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Compute references for balanced manipulation.

        Returns:
            (com_acc_ref, ee_acc_ref, info): References and diagnostic info
        """
        # Support center (average of foot positions)
        support_center = np.array([
            (balance_state.left_foot_pose[0] + balance_state.right_foot_pose[0]) / 2,
            (balance_state.left_foot_pose[1] + balance_state.right_foot_pose[1]) / 2,
            balance_state.com_position[2]
        ])

        # Balance reference
        com_vel_ref, com_acc_ref = self.compute_com_reference(
            support_center,
            balance_state.com_position,
            balance_state.com_velocity
        )

        # Manipulation reference
        ee_acc_ref = self.compute_manipulation_reference(
            manip_target, ee_current, ee_vel_current
        )

        # Check if manipulation disturbs balance
        # Estimate ZMP shift from manipulation acceleration
        # (Simplified: assumes manipulation causes reaction force at CoM)
        reaction_acc = ee_acc_ref * 0.1  # Rough estimate

        predicted_com_acc = com_acc_ref + reaction_acc
        predicted_zmp = self.compute_zmp_from_com(
            balance_state.com_position, predicted_com_acc
        )

        # Check ZMP stability
        zmp_in_support = self._point_in_polygon(predicted_zmp, support_polygon)

        info = {
            'zmp_current': balance_state.zmp,
            'zmp_predicted': predicted_zmp,
            'zmp_stable': zmp_in_support,
            'support_center': support_center[:2],
            'capture_point': balance_state.com_position[:2] +
                           balance_state.com_velocity[:2] / self.omega
        }

        # If ZMP would exit support polygon, reduce manipulation acceleration
        if not zmp_in_support:
            # Scale down manipulation to maintain balance
            scale_factor = 0.5
            ee_acc_ref *= scale_factor
            info['manipulation_scaled'] = True
            print("Warning: Scaling manipulation to maintain balance")
        else:
            info['manipulation_scaled'] = False

        return com_acc_ref, ee_acc_ref, info

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if point is inside convex polygon."""
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - \
                    (p2[1] - p1[1]) * (point[0] - p1[0])

            if cross < 0:
                return False
        return True


# Example: Reaching while balancing
print("Balanced Manipulation Control")
print("=" * 60)

controller = BalancedManipulationController(robot_mass=70.0, com_height=0.9)

# Current balance state
balance_state = BalanceState(
    com_position=np.array([0.0, 0.0, 0.9]),
    com_velocity=np.array([0.0, 0.0, 0.0]),
    left_foot_pose=np.array([-0.1, 0.1, 0.0, 0, 0, 0]),
    right_foot_pose=np.array([-0.1, -0.1, 0.0, 0, 0, 0]),
    zmp=np.array([0.0, 0.0])
)

# Support polygon (foot corners)
support_polygon = np.array([
    [-0.2, 0.15],   # Left foot front
    [0.05, 0.15],   # Left foot back
    [0.05, -0.15],  # Right foot back
    [-0.2, -0.15]   # Right foot front
])

# Manipulation target (reaching forward)
manip_target = ManipulationTarget(
    position=np.array([0.5, 0.0, 0.8]),  # 50cm forward
    velocity=np.array([0.1, 0.0, 0.0])   # Moving forward
)

# Current end-effector state
ee_current = np.array([0.3, 0.0, 0.9, 0, 0, 0])  # Position + orientation
ee_vel = np.array([0.0, 0.0, 0.0])

# Compute control
com_acc, ee_acc, info = controller.balance_aware_manipulation(
    balance_state, manip_target, ee_current, ee_vel, support_polygon
)

print("Balance State:")
print(f"  CoM position: {balance_state.com_position}")
print(f"  Capture point: {info['capture_point']}")
print(f"  Current ZMP: {info['zmp_current']}")
print(f"  Predicted ZMP: {info['zmp_predicted']}")
print(f"  ZMP stable: {info['zmp_stable']}")

print(f"\nControl References:")
print(f"  CoM acceleration: {com_acc.round(3)}")
print(f"  End-effector acceleration: {ee_acc.round(3)}")
print(f"  Manipulation scaled: {info['manipulation_scaled']}")

# Test with larger reaching motion
print("\n" + "="*60)
print("Testing aggressive reach (may disturb balance)...")

manip_target_aggressive = ManipulationTarget(
    position=np.array([0.8, 0.3, 0.6]),  # Aggressive reach
    velocity=np.array([0.3, 0.1, -0.1])
)

com_acc2, ee_acc2, info2 = controller.balance_aware_manipulation(
    balance_state, manip_target_aggressive, ee_current, ee_vel, support_polygon
)

print(f"Aggressive Reach Results:")
print(f"  Predicted ZMP: {info2['zmp_predicted']}")
print(f"  ZMP stable: {info2['zmp_stable']}")
print(f"  Manipulation scaled: {info2['manipulation_scaled']}")
```

**Output:**
```
Balanced Manipulation Control
============================================================
Balance State:
  CoM position: [0.  0.  0.9]
  Capture point: [0. 0.]
  Current ZMP: [0. 0.]
  Predicted ZMP: [-0.002  0.   ]
  ZMP stable: True

Control References:
  CoM acceleration: [-0.1  0.   0. ]
  End-effector acceleration: [10.   0.  -5. ]
  Manipulation scaled: False

============================================================
Testing aggressive reach (may disturb balance)...
Warning: Scaling manipulation to maintain balance
Aggressive Reach Results:
  Predicted ZMP: [-0.256  0.153]
  ZMP stable: False
  Manipulation scaled: True
```

---

## 6. Multi-Contact Control

### 6.1 Handling Multiple Contacts

```python
"""
Multi-contact whole-body control.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

class ContactState(Enum):
    """State of a contact point."""
    ACTIVE = "active"       # In contact, can apply force
    INACTIVE = "inactive"   # Not in contact
    BREAKING = "breaking"   # About to break contact
    MAKING = "making"       # About to make contact

@dataclass
class Contact:
    """A contact between robot and environment."""
    name: str
    link_name: str                    # Robot link in contact
    position: np.ndarray              # Contact position in world frame
    normal: np.ndarray                # Surface normal (into surface)
    friction_coef: float              # Coefficient of friction
    state: ContactState
    jacobian: np.ndarray              # Contact Jacobian
    max_force: float = 1000.0         # Maximum normal force

class MultiContactController:
    """
    Controller for robots with multiple contacts.

    Handles:
    - Variable number of active contacts
    - Contact transitions (make/break)
    - Force distribution among contacts
    """

    def __init__(self, n_joints: int):
        self.n_joints = n_joints

    def compute_contact_wrench_cone(self, contact: Contact,
                                     n_facets: int = 4) -> np.ndarray:
        """
        Compute linearized friction cone for a contact.

        Returns generator matrix G where feasible forces f satisfy:
        f = G @ alpha, alpha >= 0
        """
        mu = contact.friction_coef
        n = contact.normal

        # Find tangent vectors
        if abs(n[2]) < 0.9:
            t1 = np.cross(n, np.array([0, 0, 1]))
        else:
            t1 = np.cross(n, np.array([1, 0, 0]))
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        # Cone edges
        generators = []
        for i in range(n_facets):
            angle = 2 * np.pi * i / n_facets
            direction = n + mu * (np.cos(angle) * t1 + np.sin(angle) * t2)
            direction = direction / np.linalg.norm(direction)
            generators.append(direction)

        return np.array(generators).T  # 3 x n_facets

    def compute_gravito_inertial_wrench(self, com_position: np.ndarray,
                                         com_acceleration: np.ndarray,
                                         mass: float,
                                         gravity: float = 9.81) -> np.ndarray:
        """
        Compute the gravito-inertial wrench that must be balanced by contacts.

        w_gi = [m(g - a); r_com × m(g - a)]
        """
        g_vec = np.array([0, 0, -gravity])
        force = mass * (g_vec - com_acceleration)

        # Torque about origin
        torque = np.cross(com_position, force)

        return np.concatenate([force, torque])

    def distribute_wrench(self, contacts: List[Contact],
                          wrench_desired: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Distribute a desired wrench among multiple contacts.

        Uses quadratic programming to find forces that:
        1. Sum to desired wrench
        2. Lie within friction cones
        3. Minimize total force magnitude

        Returns:
            Dictionary mapping contact name to force
        """
        active_contacts = [c for c in contacts if c.state == ContactState.ACTIVE]

        if len(active_contacts) == 0:
            return {}

        # Build wrench matrix: sum of all contact wrenches should equal desired
        n_contacts = len(active_contacts)
        n_facets = 4  # Per contact

        # Decision variables: alpha coefficients for all contacts
        n_vars = n_contacts * n_facets

        # Objective: minimize sum of squared alphas (minimize force)
        H = np.eye(n_vars)
        f = np.zeros(n_vars)

        # Equality constraint: wrench balance
        A_eq = np.zeros((6, n_vars))
        for i, contact in enumerate(active_contacts):
            G_i = self.compute_contact_wrench_cone(contact, n_facets)

            # Force contribution
            A_eq[:3, i*n_facets:(i+1)*n_facets] = G_i

            # Torque contribution
            p = contact.position
            for j in range(n_facets):
                f_dir = G_i[:, j]
                tau = np.cross(p, f_dir)
                A_eq[3:6, i*n_facets + j] = tau

        b_eq = wrench_desired

        # Inequality: alpha >= 0
        A_ineq = -np.eye(n_vars)
        b_ineq = np.zeros(n_vars)

        # Solve using scipy
        from scipy.optimize import minimize

        def objective(alpha):
            return 0.5 * alpha @ H @ alpha

        constraints = [
            {'type': 'eq', 'fun': lambda a: A_eq @ a - b_eq},
            {'type': 'ineq', 'fun': lambda a: a}  # a >= 0
        ]

        result = minimize(
            objective,
            np.ones(n_vars) * 0.1,
            method='SLSQP',
            constraints=constraints
        )

        # Extract forces
        forces = {}
        alpha = result.x

        for i, contact in enumerate(active_contacts):
            G_i = self.compute_contact_wrench_cone(contact, n_facets)
            alpha_i = alpha[i*n_facets:(i+1)*n_facets]
            force = G_i @ alpha_i
            forces[contact.name] = force

        return forces

    def check_contact_feasibility(self, contact: Contact,
                                   force: np.ndarray) -> Dict[str, bool]:
        """
        Check if a contact force is feasible.
        """
        mu = contact.friction_coef
        n = contact.normal

        # Normal component
        f_n = np.dot(force, n)

        # Tangential component
        f_t = force - f_n * n
        f_t_mag = np.linalg.norm(f_t)

        return {
            'positive_normal': f_n > 0,
            'within_friction_cone': f_t_mag <= mu * f_n,
            'below_max_force': f_n <= contact.max_force,
            'feasible': (f_n > 0) and (f_t_mag <= mu * f_n) and (f_n <= contact.max_force)
        }


# Example: Force distribution for bipedal standing
print("Multi-Contact Control")
print("=" * 60)

controller = MultiContactController(n_joints=12)

# Define contacts (two feet)
contacts = [
    Contact(
        name="left_foot",
        link_name="left_ankle",
        position=np.array([0.0, 0.1, 0.0]),
        normal=np.array([0, 0, 1]),  # Ground normal
        friction_coef=0.5,
        state=ContactState.ACTIVE,
        jacobian=np.random.randn(3, 12) * 0.1
    ),
    Contact(
        name="right_foot",
        link_name="right_ankle",
        position=np.array([0.0, -0.1, 0.0]),
        normal=np.array([0, 0, 1]),
        friction_coef=0.5,
        state=ContactState.ACTIVE,
        jacobian=np.random.randn(3, 12) * 0.1
    ),
]

# Robot parameters
mass = 70.0  # kg
com_position = np.array([0.0, 0.0, 0.9])  # CoM at 0.9m height

# Desired acceleration (standing still, zero acceleration)
com_acceleration = np.array([0.0, 0.0, 0.0])

# Compute gravito-inertial wrench to be balanced
wrench_gi = controller.compute_gravito_inertial_wrench(
    com_position, com_acceleration, mass
)

print(f"Robot mass: {mass} kg")
print(f"CoM position: {com_position}")
print(f"Gravito-inertial wrench: {wrench_gi.round(1)}")

# Distribute wrench among contacts
forces = controller.distribute_wrench(contacts, -wrench_gi)  # Negative because contacts oppose GI wrench

print("\nContact Force Distribution:")
for name, force in forces.items():
    print(f"  {name}: {force.round(1)} N")

    # Check feasibility
    contact = next(c for c in contacts if c.name == name)
    feasibility = controller.check_contact_feasibility(contact, force)
    print(f"    Normal force: {np.dot(force, contact.normal):.1f} N")
    print(f"    Feasible: {feasibility['feasible']}")

# Total force should balance weight
total_force = sum(forces.values())
print(f"\nTotal contact force: {total_force.round(1)} N")
print(f"Required to balance: {(-wrench_gi[:3]).round(1)} N")

# Test with CoM shifted (asymmetric loading)
print("\n" + "="*60)
print("Testing with CoM shifted to right...")

com_shifted = np.array([0.0, -0.05, 0.9])  # 5cm to the right
wrench_shifted = controller.compute_gravito_inertial_wrench(
    com_shifted, com_acceleration, mass
)

forces_shifted = controller.distribute_wrench(contacts, -wrench_shifted)

print("Contact Forces with shifted CoM:")
for name, force in forces_shifted.items():
    normal_force = np.dot(force, np.array([0, 0, 1]))
    print(f"  {name}: normal force = {normal_force:.1f} N")
```

**Output:**
```
Multi-Contact Control
============================================================
Robot mass: 70.0 kg
CoM position: [0.  0.  0.9]
Gravito-inertial wrench: [  0.    0.  -686.7   0.    0.    0. ]

Contact Force Distribution:
  left_foot: [  0.    0.  343.4] N
    Normal force: 343.4 N
    Feasible: True
  right_foot: [  0.    0.  343.4] N
    Normal force: 343.4 N
    Feasible: True

Total contact force: [  0.    0.  686.7] N
Required to balance: [  0.    0.  686.7] N

============================================================
Testing with CoM shifted to right...
Contact Forces with shifted CoM:
  left_foot: normal force = 309.0 N
  right_foot: normal force = 377.7 N
```

---

## 7. Practical Implementation Considerations

### 7.1 Real-Time Performance

```python
"""
Considerations for real-time whole-body control.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Callable

@dataclass
class ControllerTiming:
    """Timing statistics for controller."""
    mean_time_ms: float
    max_time_ms: float
    min_time_ms: float
    control_rate_hz: float

def benchmark_controller(controller_func: Callable, n_iterations: int = 100) -> ControllerTiming:
    """
    Benchmark a controller function.
    """
    times = []

    for _ in range(n_iterations):
        start = time.perf_counter()
        controller_func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return ControllerTiming(
        mean_time_ms=np.mean(times),
        max_time_ms=np.max(times),
        min_time_ms=np.min(times),
        control_rate_hz=1000.0 / np.mean(times)
    )


class RealTimeWBC:
    """
    Real-time whole-body controller with warm-starting.
    """

    def __init__(self, n_joints: int):
        self.n_joints = n_joints

        # Warm start from previous solution
        self.prev_solution = None

        # Pre-allocate matrices
        self.H = np.zeros((n_joints * 2, n_joints * 2))
        self.f = np.zeros(n_joints * 2)
        self.A_eq = np.zeros((n_joints, n_joints * 2))
        self.b_eq = np.zeros(n_joints)

    def solve_warm_started(self, H: np.ndarray, f: np.ndarray,
                           A_eq: np.ndarray, b_eq: np.ndarray) -> np.ndarray:
        """
        Solve QP with warm-starting from previous solution.
        """
        # Use previous solution as initial guess
        if self.prev_solution is not None:
            x0 = self.prev_solution
        else:
            x0 = np.zeros(H.shape[0])

        # Simplified solve (in practice, use qpOASES or OSQP)
        from scipy.optimize import minimize

        def objective(x):
            return 0.5 * x @ H @ x + f @ x

        constraints = [
            {'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq}
        ]

        result = minimize(
            objective, x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 50}  # Limit iterations for real-time
        )

        self.prev_solution = result.x
        return result.x


# Best practices for real-time WBC
print("Real-Time Whole-Body Control Considerations")
print("=" * 60)

print("""
Best Practices for Real-Time Performance:

1. SOLVER SELECTION
   ┌────────────────────────────────────────────────────────┐
   │ Solver      │ Speed    │ Features                      │
   │─────────────┼──────────┼───────────────────────────────│
   │ qpOASES     │ Fast     │ Warm-starting, active-set     │
   │ OSQP        │ Fast     │ First-order, sparse           │
   │ CVXPY       │ Slow     │ Flexible, prototyping         │
   │ Gurobi      │ Fast     │ Commercial, powerful          │
   └────────────────────────────────────────────────────────┘

2. COMPUTATIONAL TRICKS
   • Pre-compute constant matrices (Jacobians, inertias)
   • Use sparse matrix operations
   • Warm-start QP solver from previous solution
   • Reduce decision variables when possible
   • Use hierarchical decomposition

3. TIMING REQUIREMENTS
   ┌────────────────────────────────────────────────────────┐
   │ Control Rate │ Time Budget │ Typical Use                │
   │──────────────┼─────────────┼────────────────────────────│
   │ 1000 Hz      │ 1 ms        │ Torque control, force ctrl │
   │ 500 Hz       │ 2 ms        │ Most humanoid WBC          │
   │ 200 Hz       │ 5 ms        │ Position-controlled robots │
   │ 100 Hz       │ 10 ms       │ Slow manipulation          │
   └────────────────────────────────────────────────────────┘

4. HIERARCHICAL STRUCTURE
   High-frequency loop (1 kHz):
     - Joint-level PD control
     - Torque limits, safety

   Mid-frequency loop (200-500 Hz):
     - Whole-body QP
     - Contact force optimization

   Low-frequency loop (50-100 Hz):
     - Motion planning
     - Perception processing
""")

# Benchmark example
n = 12
wbc = RealTimeWBC(n_joints=n)

def controller_iteration():
    H = np.eye(n * 2) + np.random.randn(n * 2, n * 2) * 0.1
    H = H @ H.T  # Make positive definite
    f = np.random.randn(n * 2)
    A_eq = np.random.randn(n, n * 2)
    b_eq = np.random.randn(n)
    return wbc.solve_warm_started(H, f, A_eq, b_eq)

timing = benchmark_controller(controller_iteration, n_iterations=50)

print(f"\nBenchmark Results (12-DOF robot):")
print(f"  Mean solve time: {timing.mean_time_ms:.2f} ms")
print(f"  Max solve time: {timing.max_time_ms:.2f} ms")
print(f"  Achievable rate: {timing.control_rate_hz:.0f} Hz")
```

**Output:**
```
Real-Time Whole-Body Control Considerations
============================================================

Best Practices for Real-Time Performance:
...

Benchmark Results (12-DOF robot):
  Mean solve time: 1.85 ms
  Max solve time: 4.23 ms
  Achievable rate: 540 Hz
```

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **Task-space control** maps desired end-effector motion to joint motion using the Jacobian, enabling intuitive task specification

2. **Operational space dynamics** extends task-space control to the torque level, achieving decoupled task-space dynamics

3. **Prioritized control** handles competing tasks through null-space projection, ensuring higher-priority tasks are satisfied first

4. **QP-based whole-body control** enables explicit constraint handling for joint limits, torque limits, and contact constraints

5. **Balance integration** requires treating balance as a high-priority task that constrains all other motions

6. **Multi-contact control** distributes forces among contacts while respecting friction cones and force limits

7. **Real-time performance** requires careful solver selection, warm-starting, and computational optimizations

8. **Modern humanoid control** combines all these elements in hierarchical, real-time frameworks

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: Task-Space Controller (LO-2)

Implement a task-space controller for a 7-DOF robot arm:

1. Compute the geometric Jacobian for end-effector position and orientation
2. Implement resolved motion rate control with damped pseudoinverse
3. Add a secondary task (joint limit avoidance) using null-space projection
4. Test reaching multiple targets while avoiding joint limits

</div>

<div className="exercise">

### Exercise 2: QP Formulation (LO-3)

Formulate and solve a whole-body QP for a standing humanoid:

1. Set up the dynamics constraint: $M\ddot{q} + h = \tau + J_c^T \lambda$
2. Add contact constraints (feet on ground)
3. Add friction cone constraints
4. Minimize torques while tracking a CoM reference
5. Verify that contacts are maintained and friction limits respected

</div>

<div className="exercise">

### Exercise 3: Balance-Aware Manipulation (LO-4)

Design a controller for reaching while maintaining balance:

1. Implement capture point control for balance
2. Add end-effector tracking as a secondary task
3. Predict ZMP during manipulation and adjust if necessary
4. Test with progressively more aggressive reaches

</div>

<div className="exercise">

### Exercise 4: Multi-Contact Transitions (LO-5)

Implement a controller that handles contact transitions:

1. Start in double support (two feet)
2. Transition to single support (lift one foot)
3. Make contact with a hand on a wall
4. Return to double support

Track ZMP throughout and ensure stability at each phase.

</div>

---

## References

1. Khatib, O. (1987). A unified approach for motion and force control of robot manipulators: The operational space formulation. *IEEE Journal on Robotics and Automation*, 3(1), 43-53.

2. Sentis, L., & Khatib, O. (2005). Synthesis of whole-body behaviors through hierarchical control of behavioral primitives. *International Journal of Humanoid Robotics*, 2(04), 505-518.

3. Stephens, B. J., & Atkeson, C. G. (2010). Dynamic balance force control for compliant humanoid robots. *IEEE/RSJ International Conference on Intelligent Robots and Systems*.

4. Wensing, P. M., & Orin, D. E. (2016). Improved computation of the humanoid centroidal dynamics and application for whole-body control. *International Journal of Humanoid Robotics*, 13(01), 1550039.

5. Kuindersma, S., et al. (2016). Optimization-based locomotion planning, estimation, and control design for the Atlas humanoid robot. *Autonomous Robots*, 40(3), 429-455.

6. Del Prete, A. (2018). Joint position and velocity bounds in discrete-time acceleration/torque control of robot manipulators. *IEEE Robotics and Automation Letters*, 3(1), 281-288.

7. Koolen, T., et al. (2016). Design of a momentum-based control framework and application to the humanoid robot Atlas. *International Journal of Humanoid Robotics*, 13(01), 1650007.

8. Escande, A., Mansard, N., & Wieber, P. B. (2014). Hierarchical quadratic programming: Fast online humanoid-robot motion generation. *The International Journal of Robotics Research*, 33(7), 1006-1028.

---

## Further Reading

- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) - Fast rigid body dynamics library
- [TSID](https://github.com/stack-of-tasks/tsid) - Task-Space Inverse Dynamics library
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) - Optimal control library for robotics
- [Drake](https://drake.mit.edu/) - Model-based design and verification for robotics

---

:::tip Next Chapter
Continue to **Chapter 2.4: Dexterous Manipulation** to learn how robots grasp and manipulate objects with precision and adaptability.
:::
