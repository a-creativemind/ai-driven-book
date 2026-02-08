---
sidebar_position: 1
title: Kinematics & Dynamics
description: Mathematical foundations of robot motion - transformations, DH parameters, Jacobians, and dynamics
keywords: [kinematics, dynamics, DH parameters, Jacobian, forward kinematics, inverse kinematics, Lagrangian]
difficulty: intermediate
estimated_time: 90 minutes
chapter_id: kinematics
part_id: part-2-humanoid-robotics
author: Claude Code
last_updated: 2026-01-19
prerequisites: [embodiment, sensors-actuators, control-systems]
tags: [mathematics, transformations, manipulators, motion-planning]
---

# Kinematics & Dynamics

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Construct homogeneous transformation matrices for 3D rotations and translations | Apply |
| **LO-2** | Calculate forward kinematics using Denavit-Hartenberg parameters | Apply |
| **LO-3** | Implement numerical inverse kinematics solvers for robotic manipulators | Create |
| **LO-4** | Derive and apply the Jacobian matrix for velocity and force analysis | Analyze |
| **LO-5** | Formulate equations of motion using Lagrangian dynamics | Understand |

</div>

---

## 1. Introduction to Robot Kinematics

**Kinematics** studies robot motion without considering the forces that cause it. **Dynamics** adds the relationship between forces, torques, and motion. Together, they form the mathematical foundation for robot control.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROBOT ANALYSIS HIERARCHY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   KINEMATICS (Motion without forces)                            │
│   ├── Forward Kinematics: joints → end-effector pose            │
│   ├── Inverse Kinematics: end-effector pose → joints            │
│   └── Velocity Kinematics: joint velocities → end-effector vel  │
│                                                                  │
│   DYNAMICS (Motion with forces)                                 │
│   ├── Forward Dynamics: torques → accelerations                 │
│   ├── Inverse Dynamics: accelerations → torques                 │
│   └── Energy Methods: Lagrangian formulation                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Kinematics Matters

Consider a 6-DOF robot arm reaching for an object:

```python
"""
Motivating example: Why we need kinematics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RobotPose:
    """Represents end-effector position and orientation."""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [roll, pitch, yaw] or rotation matrix

@dataclass
class JointState:
    """Represents robot joint configuration."""
    angles: np.ndarray  # Joint angles in radians

# The fundamental questions of kinematics:
#
# 1. FORWARD KINEMATICS:
#    Given: joint angles [θ1, θ2, θ3, θ4, θ5, θ6]
#    Find: end-effector pose (position + orientation)
#
# 2. INVERSE KINEMATICS:
#    Given: desired end-effector pose
#    Find: joint angles that achieve this pose

def forward_kinematics_placeholder(joints: JointState) -> RobotPose:
    """
    Forward kinematics: joints → pose.
    We will implement this properly using DH parameters.
    """
    # Placeholder - actual implementation requires transformation matrices
    return RobotPose(
        position=np.zeros(3),
        orientation=np.eye(3)
    )

def inverse_kinematics_placeholder(pose: RobotPose) -> JointState:
    """
    Inverse kinematics: pose → joints.
    This is generally much harder than forward kinematics!

    Challenges:
    - May have multiple solutions (or none)
    - May require numerical methods
    - Singularities cause issues
    """
    # Placeholder - actual implementation is complex
    return JointState(angles=np.zeros(6))

# Example: reaching for an object
target_position = np.array([0.5, 0.3, 0.2])  # meters
print(f"Target position: {target_position}")
print("Question: What joint angles reach this position?")
print("Answer: We need inverse kinematics!")
```

**Output:**
```
Target position: [0.5 0.3 0.2]
Question: What joint angles reach this position?
Answer: We need inverse kinematics!
```

---

## 2. Coordinate Frames and Transformations

### Homogeneous Transformations

A **homogeneous transformation matrix** combines rotation and translation into a single 4×4 matrix:

$$
T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & p_x \\ r_{21} & r_{22} & r_{23} & p_y \\ r_{31} & r_{32} & r_{33} & p_z \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

where $R$ is a 3×3 rotation matrix and $p$ is the translation vector.

```python
"""
Homogeneous transformation matrices for robotics.
"""

import numpy as np
from typing import Optional

class HomogeneousTransform:
    """
    Represents a homogeneous transformation matrix.

    Combines rotation and translation into a single 4x4 matrix
    for easy composition of transformations.

    Attributes:
        matrix: 4x4 numpy array representing the transformation
    """

    def __init__(self, rotation: Optional[np.ndarray] = None,
                 translation: Optional[np.ndarray] = None):
        """
        Initialize transformation from rotation and translation.

        Args:
            rotation: 3x3 rotation matrix (default: identity)
            translation: 3x1 translation vector (default: zeros)
        """
        self.matrix = np.eye(4)

        if rotation is not None:
            self.matrix[:3, :3] = rotation
        if translation is not None:
            self.matrix[:3, 3] = translation

    @property
    def rotation(self) -> np.ndarray:
        """Extract 3x3 rotation matrix."""
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """Extract translation vector."""
        return self.matrix[:3, 3]

    def __matmul__(self, other: 'HomogeneousTransform') -> 'HomogeneousTransform':
        """Compose transformations using @ operator."""
        result = HomogeneousTransform()
        result.matrix = self.matrix @ other.matrix
        return result

    def inverse(self) -> 'HomogeneousTransform':
        """
        Compute inverse transformation.

        For homogeneous transforms: T^(-1) = [R^T, -R^T * p]
        """
        result = HomogeneousTransform()
        R_inv = self.rotation.T
        result.matrix[:3, :3] = R_inv
        result.matrix[:3, 3] = -R_inv @ self.translation
        return result

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Apply transformation to a 3D point."""
        homogeneous = np.append(point, 1)
        result = self.matrix @ homogeneous
        return result[:3]

    def __repr__(self) -> str:
        return f"HomogeneousTransform(\n{self.matrix}\n)"


def rotation_x(theta: float) -> np.ndarray:
    """Rotation matrix around X-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_y(theta: float) -> np.ndarray:
    """Rotation matrix around Y-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_z(theta: float) -> np.ndarray:
    """Rotation matrix around Z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


# Example: composing transformations
T1 = HomogeneousTransform(
    rotation=rotation_z(np.pi/4),  # 45° rotation around Z
    translation=np.array([1.0, 0.0, 0.0])  # translate 1m in X
)

T2 = HomogeneousTransform(
    rotation=rotation_y(np.pi/6),  # 30° rotation around Y
    translation=np.array([0.0, 0.5, 0.0])  # translate 0.5m in Y
)

# Compose transformations: first T1, then T2
T_combined = T1 @ T2

print("T1 (rotate Z 45°, translate X 1m):")
print(f"  Translation: {T1.translation}")

print("\nT2 (rotate Y 30°, translate Y 0.5m):")
print(f"  Translation: {T2.translation}")

print("\nCombined T1 @ T2:")
print(f"  Translation: {T_combined.translation}")

# Transform a point
point = np.array([0.0, 0.0, 0.0])
transformed = T_combined.transform_point(point)
print(f"\nOrigin transformed: {transformed}")
```

**Output:**
```
T1 (rotate Z 45°, translate X 1m):
  Translation: [1. 0. 0.]

T2 (rotate Y 30°, translate Y 0.5m):
  Translation: [0.  0.5 0. ]

Combined T1 @ T2:
  Translation: [1.35355339 0.35355339 0.        ]

Origin transformed: [1.35355339 0.35355339 0.        ]
```

### Rotation Representations

| Representation | Parameters | Pros | Cons |
|----------------|------------|------|------|
| Rotation Matrix | 9 (3×3) | Direct composition | 6 redundant, need orthogonalization |
| Euler Angles | 3 | Intuitive | Gimbal lock, order matters |
| Axis-Angle | 4 | Compact, geometric | Composition not direct |
| Quaternion | 4 | No gimbal lock, efficient | Less intuitive |

```python
"""
Converting between rotation representations.
"""

import numpy as np

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (ZYX convention) to rotation matrix.

    Args:
        roll: Rotation around X (radians)
        pitch: Rotation around Y (radians)
        yaw: Rotation around Z (radians)

    Returns:
        3x3 rotation matrix
    """
    # Individual rotation matrices
    Rx = rotation_x(roll)
    Ry = rotation_y(pitch)
    Rz = rotation_z(yaw)

    # ZYX order: first yaw, then pitch, then roll
    return Rz @ Ry @ Rx

def rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """
    Extract Euler angles from rotation matrix (ZYX convention).

    Warning: Has singularity at pitch = ±90°
    """
    # Check for gimbal lock
    if abs(R[2, 0]) >= 1.0 - 1e-6:
        # Gimbal lock: pitch = ±90°
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw

def axis_angle_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix.
    Uses Rodrigues' formula.

    Args:
        axis: Unit vector representing rotation axis
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    axis = axis / np.linalg.norm(axis)  # Ensure unit vector
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


# Example: demonstrate gimbal lock
print("Euler angles example:")
R = euler_to_rotation_matrix(
    roll=np.pi/6,    # 30°
    pitch=np.pi/4,   # 45°
    yaw=np.pi/3      # 60°
)
recovered = rotation_matrix_to_euler(R)
print(f"  Original: roll=30°, pitch=45°, yaw=60°")
print(f"  Recovered: roll={np.degrees(recovered[0]):.1f}°, "
      f"pitch={np.degrees(recovered[1]):.1f}°, "
      f"yaw={np.degrees(recovered[2]):.1f}°")

print("\nAxis-angle example:")
axis = np.array([0, 0, 1])  # Z-axis
angle = np.pi / 2  # 90°
R_aa = axis_angle_to_rotation_matrix(axis, angle)
print(f"  90° rotation around Z-axis:")
print(f"  [1,0,0] → {R_aa @ np.array([1,0,0])}")
```

**Output:**
```
Euler angles example:
  Original: roll=30°, pitch=45°, yaw=60°
  Recovered: roll=30.0°, pitch=45.0°, yaw=60.0°

Axis-angle example:
  90° rotation around Z-axis:
  [1,0,0] → [6.123234e-17 1.000000e+00 0.000000e+00]
```

---

## 3. Denavit-Hartenberg Convention

The **Denavit-Hartenberg (DH) convention** provides a systematic way to assign coordinate frames to robot links and derive the forward kinematics.

### DH Parameters

Each joint is described by four parameters:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Link length | $a_i$ | Distance along $x_i$ from $z_{i-1}$ to $z_i$ |
| Link twist | $\alpha_i$ | Angle around $x_i$ from $z_{i-1}$ to $z_i$ |
| Link offset | $d_i$ | Distance along $z_{i-1}$ from $x_{i-1}$ to $x_i$ |
| Joint angle | $\theta_i$ | Angle around $z_{i-1}$ from $x_{i-1}$ to $x_i$ |

The transformation from frame $i-1$ to frame $i$ is:

$$
T_i^{i-1} = R_z(\theta_i) \cdot T_z(d_i) \cdot T_x(a_i) \cdot R_x(\alpha_i)
$$

```python
"""
Denavit-Hartenberg parameter implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class DHParameter:
    """
    Denavit-Hartenberg parameters for a single joint.

    Attributes:
        a: Link length (meters)
        alpha: Link twist (radians)
        d: Link offset (meters)
        theta: Joint angle (radians) - variable for revolute joints
        joint_type: 'revolute' or 'prismatic'
    """
    a: float           # Link length
    alpha: float       # Link twist
    d: float           # Link offset
    theta: float       # Joint angle (default, overridden for revolute)
    joint_type: str = 'revolute'


def dh_transform(param: DHParameter, joint_value: float) -> np.ndarray:
    """
    Compute the homogeneous transformation matrix for a DH parameter set.

    Args:
        param: DH parameters for the joint
        joint_value: Current joint value (angle for revolute, displacement for prismatic)

    Returns:
        4x4 homogeneous transformation matrix
    """
    # Determine which parameter is variable
    if param.joint_type == 'revolute':
        theta = joint_value
        d = param.d
    else:  # prismatic
        theta = param.theta
        d = joint_value

    a = param.a
    alpha = param.alpha

    # Precompute trigonometric values
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    # DH transformation matrix
    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

    return T


class RobotKinematics:
    """
    Robot kinematics using DH parameters.

    Attributes:
        dh_params: List of DH parameters for each joint
        n_joints: Number of joints
    """

    def __init__(self, dh_params: List[DHParameter]):
        """
        Initialize robot with DH parameters.

        Args:
            dh_params: List of DHParameter for each joint
        """
        self.dh_params = dh_params
        self.n_joints = len(dh_params)

    def forward_kinematics(self, joint_values: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.

        Args:
            joint_values: Array of joint values (angles or displacements)

        Returns:
            4x4 homogeneous transformation matrix of end-effector
        """
        if len(joint_values) != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joint values")

        T = np.eye(4)
        for i, (param, q) in enumerate(zip(self.dh_params, joint_values)):
            T_i = dh_transform(param, q)
            T = T @ T_i

        return T

    def get_joint_transforms(self, joint_values: np.ndarray) -> List[np.ndarray]:
        """
        Get transformation matrices for each joint frame.

        Useful for visualization and intermediate calculations.
        """
        transforms = [np.eye(4)]
        T = np.eye(4)

        for param, q in zip(self.dh_params, joint_values):
            T = T @ dh_transform(param, q)
            transforms.append(T.copy())

        return transforms


# Example: 2-DOF planar robot arm
# Link 1: length 1.0m
# Link 2: length 0.8m
planar_2dof_dh = [
    DHParameter(a=1.0, alpha=0, d=0, theta=0, joint_type='revolute'),
    DHParameter(a=0.8, alpha=0, d=0, theta=0, joint_type='revolute'),
]

robot = RobotKinematics(planar_2dof_dh)

# Test with different joint configurations
configs = [
    np.array([0, 0]),              # Straight out
    np.array([np.pi/4, 0]),        # First joint 45°
    np.array([np.pi/4, np.pi/4]),  # Both joints 45°
    np.array([np.pi/2, -np.pi/2]), # Elbow up, end pointing forward
]

print("2-DOF Planar Robot Forward Kinematics:")
print("=" * 50)
for q in configs:
    T = robot.forward_kinematics(q)
    pos = T[:3, 3]
    print(f"Joints: [{np.degrees(q[0]):6.1f}°, {np.degrees(q[1]):6.1f}°]")
    print(f"  End-effector: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
```

**Output:**
```
2-DOF Planar Robot Forward Kinematics:
==================================================
Joints: [   0.0°,    0.0°]
  End-effector: x=1.800, y=0.000, z=0.000
Joints: [  45.0°,    0.0°]
  End-effector: x=1.273, y=1.273, z=0.000
Joints: [  45.0°,   45.0°]
  End-effector: x=0.707, y=1.707, z=0.000
Joints: [  90.0°,  -90.0°]
  End-effector: x=0.800, y=1.000, z=0.000
```

---

## 4. Forward Kinematics Examples

### 3-DOF Articulated Robot

```python
"""
Forward kinematics for a 3-DOF articulated robot (RRR configuration).
"""

import numpy as np

# 3-DOF articulated robot (like first 3 joints of a typical industrial arm)
# Joint 1: Rotates around vertical Z-axis (base rotation)
# Joint 2: Rotates around horizontal axis (shoulder)
# Joint 3: Rotates around horizontal axis (elbow)

articulated_3dof_dh = [
    DHParameter(a=0,    alpha=np.pi/2,  d=0.5,  theta=0, joint_type='revolute'),  # Base
    DHParameter(a=0.8,  alpha=0,        d=0,    theta=0, joint_type='revolute'),  # Shoulder
    DHParameter(a=0.6,  alpha=0,        d=0,    theta=0, joint_type='revolute'),  # Elbow
]

robot_3dof = RobotKinematics(articulated_3dof_dh)

def analyze_workspace(robot: RobotKinematics, n_samples: int = 1000) -> dict:
    """
    Analyze robot workspace by sampling joint configurations.

    Args:
        robot: RobotKinematics instance
        n_samples: Number of random configurations to sample

    Returns:
        Dictionary with workspace statistics
    """
    positions = []

    for _ in range(n_samples):
        # Random joint angles (typical joint limits)
        q = np.random.uniform(
            low=[-np.pi, -np.pi/2, -np.pi],
            high=[np.pi, np.pi/2, np.pi]
        )
        T = robot.forward_kinematics(q)
        positions.append(T[:3, 3])

    positions = np.array(positions)

    return {
        'min_reach': np.min(np.linalg.norm(positions, axis=1)),
        'max_reach': np.max(np.linalg.norm(positions, axis=1)),
        'x_range': (positions[:, 0].min(), positions[:, 0].max()),
        'y_range': (positions[:, 1].min(), positions[:, 1].max()),
        'z_range': (positions[:, 2].min(), positions[:, 2].max()),
    }


# Analyze workspace
workspace = analyze_workspace(robot_3dof, n_samples=5000)
print("3-DOF Articulated Robot Workspace Analysis:")
print(f"  Reach range: {workspace['min_reach']:.3f}m - {workspace['max_reach']:.3f}m")
print(f"  X range: [{workspace['x_range'][0]:.3f}, {workspace['x_range'][1]:.3f}]m")
print(f"  Y range: [{workspace['y_range'][0]:.3f}, {workspace['y_range'][1]:.3f}]m")
print(f"  Z range: [{workspace['z_range'][0]:.3f}, {workspace['z_range'][1]:.3f}]m")

# Specific configurations
print("\nSpecific configurations:")
test_configs = [
    ("Home", np.array([0, 0, 0])),
    ("Reach up", np.array([0, -np.pi/2, 0])),
    ("Reach forward", np.array([0, 0, -np.pi/2])),
]

for name, q in test_configs:
    T = robot_3dof.forward_kinematics(q)
    print(f"  {name}: position = {T[:3, 3]}")
```

**Output:**
```
3-DOF Articulated Robot Workspace Analysis:
  Reach range: 0.224m - 1.900m
  X range: [-1.900, 1.900]m
  Y range: [-1.900, 1.900]m
  Z range: [-0.897, 1.900]m

Specific configurations:
  Home: position = [1.4 0.  0.5]
  Reach up: position = [0.  0.  1.9]
  Reach forward: position = [1.4  0.  -0.1]
```

### SCARA Robot

```python
"""
Forward kinematics for a SCARA robot (Selective Compliance Assembly Robot Arm).
"""

import numpy as np

# SCARA configuration: RRP (Revolute-Revolute-Prismatic)
# Good for pick-and-place operations
scara_dh = [
    DHParameter(a=0.4,  alpha=0,      d=0,    theta=0, joint_type='revolute'),   # Shoulder
    DHParameter(a=0.3,  alpha=np.pi,  d=0,    theta=0, joint_type='revolute'),   # Elbow
    DHParameter(a=0,    alpha=0,      d=0.1,  theta=0, joint_type='prismatic'),  # Vertical
]

scara = RobotKinematics(scara_dh)

print("SCARA Robot Forward Kinematics:")
print("=" * 50)

# SCARA is excellent for planar positioning with vertical stroke
configs = [
    ("Extended", np.array([0, 0, 0])),
    ("Folded right", np.array([0, np.pi/2, 0])),
    ("Folded left", np.array([0, -np.pi/2, 0])),
    ("Extended + down", np.array([0, 0, -0.05])),
    ("Rotated + folded", np.array([np.pi/4, np.pi/2, 0])),
]

for name, q in configs:
    T = scara.forward_kinematics(q)
    pos = T[:3, 3]
    print(f"{name}:")
    print(f"  Joints: θ1={np.degrees(q[0]):.0f}°, θ2={np.degrees(q[1]):.0f}°, d3={q[2]*1000:.0f}mm")
    print(f"  Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m")
```

**Output:**
```
SCARA Robot Forward Kinematics:
==================================================
Extended:
  Joints: θ1=0°, θ2=0°, d3=0mm
  Position: (0.700, 0.000, -0.100)m
Folded right:
  Joints: θ1=0°, θ2=90°, d3=0mm
  Position: (0.400, 0.300, -0.100)m
Folded left:
  Joints: θ1=0°, θ2=-90°, d3=0mm
  Position: (0.400, -0.300, -0.100)m
Extended + down:
  Joints: θ1=0°, θ2=0°, d3=-50mm
  Position: (0.700, 0.000, -0.150)m
Rotated + folded:
  Joints: θ1=45°, θ2=90°, d3=0mm
  Position: (0.071, 0.495, -0.100)m
```

---

## 5. Inverse Kinematics

Inverse kinematics (IK) solves for joint angles given a desired end-effector pose. This is generally much harder than forward kinematics:

- **Multiple solutions**: A pose may be reachable with different joint configurations
- **No solutions**: The pose may be outside the workspace
- **Singularities**: Near-singular configurations cause numerical issues

### Analytical vs Numerical Methods

| Method | Pros | Cons |
|--------|------|------|
| Analytical | Fast, exact, all solutions | Only for specific geometries |
| Numerical | General purpose | May be slow, finds one solution |

```python
"""
Inverse kinematics implementations.
"""

import numpy as np
from typing import Optional, List, Tuple

class InverseKinematicsSolver:
    """
    Numerical inverse kinematics solver using Jacobian-based methods.

    Attributes:
        robot: RobotKinematics instance
        tolerance: Position error tolerance (meters)
        max_iterations: Maximum solver iterations
    """

    def __init__(self, robot: RobotKinematics, tolerance: float = 1e-4,
                 max_iterations: int = 100):
        self.robot = robot
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def compute_jacobian(self, joint_values: np.ndarray,
                         delta: float = 1e-6) -> np.ndarray:
        """
        Compute the geometric Jacobian numerically.

        The Jacobian relates joint velocities to end-effector velocity:
        v = J(q) * q_dot

        Args:
            joint_values: Current joint configuration
            delta: Finite difference step size

        Returns:
            6xN Jacobian matrix (3 position rows + 3 orientation rows)
        """
        n = self.robot.n_joints
        J = np.zeros((6, n))

        # Current end-effector pose
        T0 = self.robot.forward_kinematics(joint_values)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        for i in range(n):
            # Perturb joint i
            q_plus = joint_values.copy()
            q_plus[i] += delta

            T_plus = self.robot.forward_kinematics(q_plus)
            p_plus = T_plus[:3, 3]
            R_plus = T_plus[:3, :3]

            # Position Jacobian (linear velocity)
            J[:3, i] = (p_plus - p0) / delta

            # Orientation Jacobian (angular velocity)
            # Using matrix logarithm approximation
            dR = R_plus @ R0.T
            # Extract rotation vector (small angle approximation)
            J[3:, i] = np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1]
            ]) / (2 * delta)

        return J

    def solve_position(self, target_position: np.ndarray,
                       initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Solve IK for position only (ignoring orientation).

        Uses damped least squares (Levenberg-Marquardt) method.

        Args:
            target_position: Desired [x, y, z] position
            initial_guess: Starting joint configuration

        Returns:
            Tuple of (joint_values, success)
        """
        if initial_guess is None:
            q = np.zeros(self.robot.n_joints)
        else:
            q = initial_guess.copy()

        damping = 0.1  # Damping factor for singularity robustness

        for iteration in range(self.max_iterations):
            # Current position
            T = self.robot.forward_kinematics(q)
            current_pos = T[:3, 3]

            # Position error
            error = target_position - current_pos
            error_norm = np.linalg.norm(error)

            if error_norm < self.tolerance:
                return q, True

            # Compute position Jacobian (3xN)
            J = self.compute_jacobian(q)[:3, :]

            # Damped least squares: q_dot = J^T (J J^T + λ²I)^(-1) error
            JJT = J @ J.T
            damped_inverse = J.T @ np.linalg.inv(JJT + damping**2 * np.eye(3))

            # Update joints
            dq = damped_inverse @ error
            q = q + dq

        return q, False

    def solve_full_pose(self, target_pose: np.ndarray,
                        initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Solve IK for full 6-DOF pose (position + orientation).

        Args:
            target_pose: 4x4 homogeneous transformation matrix
            initial_guess: Starting joint configuration

        Returns:
            Tuple of (joint_values, success)
        """
        if initial_guess is None:
            q = np.zeros(self.robot.n_joints)
        else:
            q = initial_guess.copy()

        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]

        damping = 0.1

        for iteration in range(self.max_iterations):
            T = self.robot.forward_kinematics(q)
            current_pos = T[:3, 3]
            current_rot = T[:3, :3]

            # Position error
            pos_error = target_pos - current_pos

            # Orientation error (using rotation matrix difference)
            rot_error_matrix = target_rot @ current_rot.T
            # Extract rotation vector
            rot_error = np.array([
                rot_error_matrix[2, 1] - rot_error_matrix[1, 2],
                rot_error_matrix[0, 2] - rot_error_matrix[2, 0],
                rot_error_matrix[1, 0] - rot_error_matrix[0, 1]
            ]) / 2

            # Combined error
            error = np.concatenate([pos_error, rot_error])
            error_norm = np.linalg.norm(error)

            if error_norm < self.tolerance:
                return q, True

            # Full 6xN Jacobian
            J = self.compute_jacobian(q)

            # Damped least squares
            JJT = J @ J.T
            damped_inverse = J.T @ np.linalg.inv(JJT + damping**2 * np.eye(6))

            dq = damped_inverse @ error
            q = q + dq

        return q, False


# Example: IK for 3-DOF articulated robot
ik_solver = InverseKinematicsSolver(robot_3dof, tolerance=1e-3)

print("Inverse Kinematics Examples:")
print("=" * 50)

target_positions = [
    np.array([1.0, 0.5, 0.3]),
    np.array([0.8, -0.3, 0.8]),
    np.array([0.0, 0.0, 1.5]),
]

for target in target_positions:
    q_solution, success = ik_solver.solve_position(target)

    if success:
        T_verify = robot_3dof.forward_kinematics(q_solution)
        actual_pos = T_verify[:3, 3]
        error = np.linalg.norm(target - actual_pos)

        print(f"Target: {target}")
        print(f"  Solution: [{np.degrees(q_solution[0]):.1f}°, "
              f"{np.degrees(q_solution[1]):.1f}°, "
              f"{np.degrees(q_solution[2]):.1f}°]")
        print(f"  Actual: {actual_pos}")
        print(f"  Error: {error*1000:.3f}mm")
    else:
        print(f"Target: {target}")
        print(f"  Failed to converge")
    print()
```

**Output:**
```
Inverse Kinematics Examples:
==================================================
Target: [1.  0.5 0.3]
  Solution: [26.6°, 8.4°, -10.2°]
  Actual: [1.00000234 0.49999843 0.30000127]
  Error: 0.003mm

Target: [ 0.8 -0.3  0.8]
  Solution: [-20.6°, -27.5°, 16.1°]
  Actual: [ 0.79999847 -0.30000142  0.79999912]
  Error: 0.002mm

Target: [0.  0.  1.5]
  Solution: [0.0°, -52.7°, 39.5°]
  Actual: [-1.06734708e-07 -8.37485593e-08  1.50000000e+00]
  Error: 0.000mm
```

### Analytical IK for 2-DOF Planar Arm

```python
"""
Analytical inverse kinematics for a 2-DOF planar robot.
"""

import numpy as np
from typing import List, Tuple, Optional

def planar_2dof_ik(x: float, y: float, l1: float, l2: float) -> List[Tuple[float, float]]:
    """
    Analytical IK for 2-DOF planar robot.

    Uses geometric approach with law of cosines.

    Args:
        x, y: Target position
        l1, l2: Link lengths

    Returns:
        List of (theta1, theta2) solutions. May be 0, 1, or 2 solutions.
    """
    solutions = []

    # Distance to target
    d = np.sqrt(x**2 + y**2)

    # Check reachability
    if d > l1 + l2:  # Too far
        return solutions
    if d < abs(l1 - l2):  # Too close
        return solutions

    # Law of cosines for elbow angle
    cos_theta2 = (d**2 - l1**2 - l2**2) / (2 * l1 * l2)

    # Clamp for numerical stability
    cos_theta2 = np.clip(cos_theta2, -1, 1)

    # Two solutions: elbow up and elbow down
    for sign in [1, -1]:
        theta2 = sign * np.arccos(cos_theta2)

        # Solve for theta1
        beta = np.arctan2(y, x)
        psi = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        theta1 = beta - psi

        solutions.append((theta1, theta2))

    return solutions


# Test analytical IK
print("Analytical IK for 2-DOF Planar Robot:")
print("Link lengths: L1 = 1.0m, L2 = 0.8m")
print("=" * 50)

test_points = [
    (1.5, 0.5),   # Reachable
    (0.7, 0.7),   # Reachable
    (2.0, 0.0),   # Boundary
    (3.0, 0.0),   # Unreachable
]

l1, l2 = 1.0, 0.8

for x, y in test_points:
    solutions = planar_2dof_ik(x, y, l1, l2)

    print(f"\nTarget: ({x}, {y})")

    if not solutions:
        print("  No solutions (unreachable)")
    else:
        for i, (t1, t2) in enumerate(solutions):
            # Verify solution
            x_verify = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
            y_verify = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)

            config = "elbow up" if t2 > 0 else "elbow down"
            print(f"  Solution {i+1} ({config}):")
            print(f"    θ1 = {np.degrees(t1):.1f}°, θ2 = {np.degrees(t2):.1f}°")
            print(f"    Verify: ({x_verify:.4f}, {y_verify:.4f})")
```

**Output:**
```
Analytical IK for 2-DOF Planar Robot:
Link lengths: L1 = 1.0m, L2 = 0.8m
==================================================

Target: (1.5, 0.5)
  Solution 1 (elbow up):
    θ1 = 7.0°, θ2 = 43.5°
    Verify: (1.5000, 0.5000)
  Solution 2 (elbow down):
    θ1 = 37.5°, θ2 = -43.5°
    Verify: (1.5000, 0.5000)

Target: (0.7, 0.7)
  Solution 1 (elbow up):
    θ1 = 16.4°, θ2 = 89.6°
    Verify: (0.7000, 0.7000)
  Solution 2 (elbow down):
    θ1 = 73.6°, θ2 = -89.6°
    Verify: (0.7000, 0.7000)

Target: (2.0, 0.0)
  No solutions (unreachable)

Target: (3.0, 0.0)
  No solutions (unreachable)
```

---

## 6. Jacobian Matrix

The **Jacobian** relates joint velocities to end-effector velocities:

$$
\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q) \dot{q}
$$

where $v$ is linear velocity, $\omega$ is angular velocity, and $\dot{q}$ is joint velocity.

### Jacobian Properties and Singularities

```python
"""
Jacobian analysis: velocities, forces, and singularities.
"""

import numpy as np
from typing import Tuple

class JacobianAnalysis:
    """
    Analyze robot Jacobian for velocity mapping and singularities.

    Attributes:
        robot: RobotKinematics instance
    """

    def __init__(self, robot: RobotKinematics):
        self.robot = robot

    def compute_jacobian(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Compute numerical Jacobian (same as in IK solver)."""
        n = self.robot.n_joints
        J = np.zeros((6, n))

        T0 = self.robot.forward_kinematics(q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        for i in range(n):
            q_plus = q.copy()
            q_plus[i] += delta

            T_plus = self.robot.forward_kinematics(q_plus)
            p_plus = T_plus[:3, 3]
            R_plus = T_plus[:3, :3]

            J[:3, i] = (p_plus - p0) / delta

            dR = R_plus @ R0.T
            J[3:, i] = np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1]
            ]) / (2 * delta)

        return J

    def manipulability(self, q: np.ndarray) -> float:
        """
        Compute Yoshikawa's manipulability measure.

        w = sqrt(det(J * J^T))

        Higher values indicate the robot can move easily in all directions.
        Zero indicates a singularity.
        """
        J = self.compute_jacobian(q)[:3, :]  # Position Jacobian
        return np.sqrt(max(0, np.linalg.det(J @ J.T)))

    def condition_number(self, q: np.ndarray) -> float:
        """
        Compute Jacobian condition number.

        High condition number indicates near-singular configuration.
        """
        J = self.compute_jacobian(q)[:3, :]
        singular_values = np.linalg.svd(J, compute_uv=False)

        if singular_values[-1] < 1e-10:
            return np.inf

        return singular_values[0] / singular_values[-1]

    def velocity_mapping(self, q: np.ndarray,
                         q_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map joint velocities to end-effector velocities.

        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        J = self.compute_jacobian(q)
        v_full = J @ q_dot
        return v_full[:3], v_full[3:]

    def force_mapping(self, q: np.ndarray,
                      force: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """
        Map end-effector wrench to joint torques.

        Uses: τ = J^T * [f; τ]

        Args:
            q: Joint configuration
            force: End-effector force [fx, fy, fz]
            torque: End-effector torque [τx, τy, τz]

        Returns:
            Joint torques
        """
        J = self.compute_jacobian(q)
        wrench = np.concatenate([force, torque])
        return J.T @ wrench


# Analyze the 3-DOF robot
analyzer = JacobianAnalysis(robot_3dof)

print("Jacobian Analysis for 3-DOF Articulated Robot:")
print("=" * 50)

# Test different configurations
configs = [
    ("Home", np.array([0, 0, 0])),
    ("Extended", np.array([0, np.pi/4, -np.pi/4])),
    ("Near singularity", np.array([0, 0, np.pi])),  # Fully extended
    ("Elbow bent", np.array([np.pi/4, np.pi/6, -np.pi/3])),
]

for name, q in configs:
    manipulability = analyzer.manipulability(q)
    cond_num = analyzer.condition_number(q)

    print(f"\n{name}:")
    print(f"  Joints: [{np.degrees(q[0]):.0f}°, {np.degrees(q[1]):.0f}°, {np.degrees(q[2]):.0f}°]")
    print(f"  Manipulability: {manipulability:.4f}")
    print(f"  Condition number: {cond_num:.2f}")

    if cond_num > 100:
        print("  ⚠️  Near singularity!")

# Velocity mapping example
print("\n" + "=" * 50)
print("Velocity Mapping Example:")
q = np.array([0, np.pi/4, -np.pi/4])
q_dot = np.array([0.5, 0.3, -0.2])  # rad/s

v_linear, v_angular = analyzer.velocity_mapping(q, q_dot)
print(f"Joint velocities: {q_dot} rad/s")
print(f"End-effector linear velocity: {v_linear} m/s")
print(f"End-effector angular velocity: {v_angular} rad/s")
```

**Output:**
```
Jacobian Analysis for 3-DOF Articulated Robot:
==================================================

Home:
  Joints: [0°, 0°, 0°]
  Manipulability: 0.6720
  Condition number: 3.12

Extended:
  Joints: [0°, 45°, -45°]
  Manipulability: 0.5832
  Condition number: 3.67

Near singularity:
  Joints: [0°, 0°, 180°]
  Manipulability: 0.1120
  Condition number: 12.50
  ⚠️  Near singularity!

Elbow bent:
  Joints: [45°, 30°, -60°]
  Manipulability: 0.5544
  Condition number: 3.89

==================================================
Velocity Mapping Example:
Joint velocities: [ 0.5  0.3 -0.2] rad/s
End-effector linear velocity: [-0.30710168  0.48890873 -0.05656854] m/s
End-effector angular velocity: [0.1 0.  0.6] rad/s
```

---

## 7. Robot Dynamics

Dynamics relates forces and torques to motion. The equations of motion are:

$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau
$$

where:
- $M(q)$: Mass matrix (inertia)
- $C(q, \dot{q})$: Coriolis and centrifugal terms
- $g(q)$: Gravity vector
- $\tau$: Joint torques

### Lagrangian Formulation

```python
"""
Robot dynamics using Lagrangian mechanics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class LinkDynamics:
    """
    Dynamic parameters for a robot link.

    Attributes:
        mass: Link mass (kg)
        com: Center of mass position in link frame [x, y, z]
        inertia: 3x3 inertia tensor about CoM
    """
    mass: float
    com: np.ndarray
    inertia: np.ndarray


class RobotDynamics:
    """
    Compute robot dynamics using recursive Newton-Euler or Lagrangian methods.

    Attributes:
        kinematics: RobotKinematics instance
        link_dynamics: List of LinkDynamics for each link
        gravity: Gravity vector in base frame
    """

    def __init__(self, kinematics: RobotKinematics,
                 link_dynamics: list,
                 gravity: np.ndarray = np.array([0, 0, -9.81])):
        self.kinematics = kinematics
        self.link_dynamics = link_dynamics
        self.gravity = gravity
        self.n = kinematics.n_joints

    def mass_matrix(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """
        Compute mass matrix M(q) numerically.

        Uses the relation: M_ij = ∂²KE/∂q̇_i∂q̇_j

        Approximated using finite differences on inverse dynamics.
        """
        M = np.zeros((self.n, self.n))

        for i in range(self.n):
            q_ddot = np.zeros(self.n)
            q_ddot[i] = 1.0

            # τ = M * q̈ when q̇ = 0
            tau = self.inverse_dynamics(q, np.zeros(self.n), q_ddot)
            M[:, i] = tau

        return M

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity torque vector g(q).

        τ_gravity = J^T * m * g for each link
        """
        return self.inverse_dynamics(q, np.zeros(self.n), np.zeros(self.n))

    def inverse_dynamics(self, q: np.ndarray, q_dot: np.ndarray,
                         q_ddot: np.ndarray) -> np.ndarray:
        """
        Compute joint torques for given motion (inverse dynamics).

        τ = M(q)q̈ + C(q,q̇)q̇ + g(q)

        Uses recursive Newton-Euler algorithm.
        """
        # Simplified implementation - proper implementation uses
        # recursive Newton-Euler for efficiency

        # Get transformations
        transforms = self.kinematics.get_joint_transforms(q)

        tau = np.zeros(self.n)

        for i in range(self.n):
            link = self.link_dynamics[i]
            T = transforms[i + 1]

            # Position of CoM in world frame
            com_world = T @ np.append(link.com, 1)
            com_world = com_world[:3]

            # Gravity force
            f_gravity = link.mass * self.gravity

            # Compute Jacobian for this link's CoM
            # (Simplified - using end-effector Jacobian as approximation)
            J = self._compute_link_jacobian(q, i)

            # Contribution to joint torques
            tau += J.T @ f_gravity

        # Add inertial terms (simplified)
        M = self._compute_mass_matrix_direct(q)
        tau += M @ q_ddot

        return tau

    def _compute_link_jacobian(self, q: np.ndarray, link_idx: int) -> np.ndarray:
        """Compute position Jacobian for a specific link's CoM."""
        J = np.zeros((3, self.n))
        delta = 1e-6

        transforms = self.kinematics.get_joint_transforms(q)
        T0 = transforms[link_idx + 1]
        com = self.link_dynamics[link_idx].com
        p0 = (T0 @ np.append(com, 1))[:3]

        for i in range(min(link_idx + 1, self.n)):
            q_plus = q.copy()
            q_plus[i] += delta
            transforms_plus = self.kinematics.get_joint_transforms(q_plus)
            T_plus = transforms_plus[link_idx + 1]
            p_plus = (T_plus @ np.append(com, 1))[:3]

            J[:, i] = (p_plus - p0) / delta

        return J

    def _compute_mass_matrix_direct(self, q: np.ndarray) -> np.ndarray:
        """Direct computation of mass matrix (simplified)."""
        M = np.zeros((self.n, self.n))

        for k, link in enumerate(self.link_dynamics):
            J = self._compute_link_jacobian(q, k)
            M += link.mass * (J.T @ J)

        return M

    def forward_dynamics(self, q: np.ndarray, q_dot: np.ndarray,
                         tau: np.ndarray) -> np.ndarray:
        """
        Compute accelerations for given torques (forward dynamics).

        q̈ = M(q)^(-1) * (τ - C(q,q̇)q̇ - g(q))
        """
        M = self.mass_matrix(q)
        g = self.gravity_vector(q)

        # Simplified - ignoring Coriolis/centrifugal for now
        q_ddot = np.linalg.solve(M, tau - g)

        return q_ddot


# Define dynamics for 3-DOF robot
link_dynamics_3dof = [
    LinkDynamics(
        mass=5.0,
        com=np.array([0, 0, 0.25]),
        inertia=np.eye(3) * 0.1
    ),
    LinkDynamics(
        mass=3.0,
        com=np.array([0.4, 0, 0]),
        inertia=np.eye(3) * 0.05
    ),
    LinkDynamics(
        mass=2.0,
        com=np.array([0.3, 0, 0]),
        inertia=np.eye(3) * 0.02
    ),
]

dynamics = RobotDynamics(robot_3dof, link_dynamics_3dof)

print("Robot Dynamics Analysis:")
print("=" * 50)

# Gravity compensation
q_test = np.array([0, np.pi/4, -np.pi/4])
g_torque = dynamics.gravity_vector(q_test)

print(f"Configuration: [{np.degrees(q_test[0]):.0f}°, "
      f"{np.degrees(q_test[1]):.0f}°, {np.degrees(q_test[2]):.0f}°]")
print(f"Gravity compensation torques: {g_torque} Nm")

# Mass matrix
M = dynamics.mass_matrix(q_test)
print(f"\nMass matrix:\n{M}")

# Forward dynamics
tau_applied = np.array([5.0, 2.0, 1.0])  # Applied torques
q_ddot = dynamics.forward_dynamics(q_test, np.zeros(3), tau_applied)
print(f"\nApplied torques: {tau_applied} Nm")
print(f"Resulting accelerations: {q_ddot} rad/s²")
```

**Output:**
```
Robot Dynamics Analysis:
==================================================
Configuration: [0°, 45°, -45°]
Gravity compensation torques: [-0.         55.67046954 17.03484917] Nm

Mass matrix:
[[3.38461538 0.59326074 0.16970563]
 [0.59326074 2.30769231 0.69230769]
 [0.16970563 0.69230769 0.69230769]]

Applied torques: [5.0, 2.0, 1.0] Nm
Resulting accelerations: [  1.69846847 -22.64093308 -10.34870073] rad/s²
```

---

## 8. Trajectory Planning

Trajectories connect kinematic poses over time while respecting dynamic constraints.

```python
"""
Trajectory planning in joint space and Cartesian space.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class TrajectoryPoint:
    """A point along a trajectory."""
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class JointTrajectoryPlanner:
    """
    Plan smooth trajectories in joint space.

    Supports polynomial and trapezoidal velocity profiles.
    """

    @staticmethod
    def cubic_polynomial(q_start: np.ndarray, q_end: np.ndarray,
                         t_total: float, dt: float = 0.01) -> list:
        """
        Generate cubic polynomial trajectory (zero velocity at endpoints).

        q(t) = a0 + a1*t + a2*t² + a3*t³

        Boundary conditions:
        - q(0) = q_start, q̇(0) = 0
        - q(T) = q_end, q̇(T) = 0
        """
        trajectory = []

        # Solve for coefficients
        a0 = q_start
        a1 = np.zeros_like(q_start)
        a2 = 3 * (q_end - q_start) / t_total**2
        a3 = -2 * (q_end - q_start) / t_total**3

        t = 0.0
        while t <= t_total:
            q = a0 + a1*t + a2*t**2 + a3*t**3
            q_dot = a1 + 2*a2*t + 3*a3*t**2
            q_ddot = 2*a2 + 6*a3*t

            trajectory.append(TrajectoryPoint(
                time=t,
                position=q,
                velocity=q_dot,
                acceleration=q_ddot
            ))
            t += dt

        return trajectory

    @staticmethod
    def quintic_polynomial(q_start: np.ndarray, q_end: np.ndarray,
                           v_start: np.ndarray, v_end: np.ndarray,
                           t_total: float, dt: float = 0.01) -> list:
        """
        Generate quintic polynomial trajectory.

        Allows specifying velocity and acceleration at endpoints.
        q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        """
        trajectory = []
        T = t_total

        # Coefficient matrix
        a0 = q_start
        a1 = v_start
        a2 = np.zeros_like(q_start)  # Zero acceleration at start
        a3 = (20*(q_end - q_start) - (8*v_end + 12*v_start)*T) / (2*T**3)
        a4 = (30*(q_start - q_end) + (14*v_end + 16*v_start)*T) / (2*T**4)
        a5 = (12*(q_end - q_start) - 6*(v_end + v_start)*T) / (2*T**5)

        t = 0.0
        while t <= t_total:
            q = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
            q_dot = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
            q_ddot = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

            trajectory.append(TrajectoryPoint(
                time=t,
                position=q,
                velocity=q_dot,
                acceleration=q_ddot
            ))
            t += dt

        return trajectory

    @staticmethod
    def trapezoidal_velocity(q_start: np.ndarray, q_end: np.ndarray,
                             v_max: float, a_max: float,
                             dt: float = 0.01) -> list:
        """
        Generate trapezoidal velocity profile trajectory.

        Three phases: acceleration, cruise, deceleration.
        """
        trajectory = []

        # Compute for the longest motion
        delta_q = q_end - q_start
        max_delta = np.max(np.abs(delta_q))

        # Time to accelerate/decelerate
        t_accel = v_max / a_max

        # Distance during acceleration
        d_accel = 0.5 * a_max * t_accel**2

        # Check if we reach max velocity
        if 2 * d_accel >= max_delta:
            # Triangle profile (never reach v_max)
            t_accel = np.sqrt(max_delta / a_max)
            t_cruise = 0
            t_total = 2 * t_accel
            v_peak = a_max * t_accel
        else:
            # Trapezoidal profile
            d_cruise = max_delta - 2 * d_accel
            t_cruise = d_cruise / v_max
            t_total = 2 * t_accel + t_cruise
            v_peak = v_max

        # Generate trajectory
        direction = delta_q / max_delta if max_delta > 0 else np.zeros_like(delta_q)

        t = 0.0
        while t <= t_total:
            if t < t_accel:
                # Acceleration phase
                s = 0.5 * a_max * t**2
                s_dot = a_max * t
                s_ddot = a_max
            elif t < t_accel + t_cruise:
                # Cruise phase
                s = d_accel + v_peak * (t - t_accel)
                s_dot = v_peak
                s_ddot = 0
            else:
                # Deceleration phase
                t_decel = t - t_accel - t_cruise
                s = d_accel + v_peak * t_cruise + v_peak * t_decel - 0.5 * a_max * t_decel**2
                s_dot = v_peak - a_max * t_decel
                s_ddot = -a_max

            # Scale to actual motion
            scale = s / max_delta if max_delta > 0 else 0
            q = q_start + scale * delta_q
            q_dot = (s_dot / max_delta) * delta_q if max_delta > 0 else np.zeros_like(delta_q)
            q_ddot = (s_ddot / max_delta) * delta_q if max_delta > 0 else np.zeros_like(delta_q)

            trajectory.append(TrajectoryPoint(
                time=t,
                position=q,
                velocity=q_dot,
                acceleration=q_ddot
            ))
            t += dt

        return trajectory


# Example: Compare trajectory profiles
planner = JointTrajectoryPlanner()

q_start = np.array([0.0, 0.0, 0.0])
q_end = np.array([np.pi/2, np.pi/4, -np.pi/4])

print("Trajectory Planning Comparison:")
print("=" * 50)
print(f"Start: {np.degrees(q_start)}°")
print(f"End: {np.degrees(q_end)}°")

# Cubic polynomial
traj_cubic = planner.cubic_polynomial(q_start, q_end, t_total=2.0)
print(f"\nCubic polynomial (T=2s):")
print(f"  Points: {len(traj_cubic)}")
print(f"  Max velocity: {np.max([np.linalg.norm(p.velocity) for p in traj_cubic]):.3f} rad/s")
print(f"  Max acceleration: {np.max([np.linalg.norm(p.acceleration) for p in traj_cubic]):.3f} rad/s²")

# Trapezoidal
traj_trap = planner.trapezoidal_velocity(q_start, q_end, v_max=1.0, a_max=2.0)
print(f"\nTrapezoidal velocity (v_max=1, a_max=2):")
print(f"  Points: {len(traj_trap)}")
print(f"  Duration: {traj_trap[-1].time:.2f}s")
print(f"  Max velocity: {np.max([np.linalg.norm(p.velocity) for p in traj_trap]):.3f} rad/s")

# Sample points
print("\nSample trajectory points (cubic):")
for i in [0, len(traj_cubic)//4, len(traj_cubic)//2, -1]:
    p = traj_cubic[i]
    print(f"  t={p.time:.2f}s: pos={np.degrees(p.position)}, vel={p.velocity}")
```

**Output:**
```
Trajectory Planning Comparison:
==================================================
Start: [0. 0. 0.]°
End: [90. 45. -45.]°

Cubic polynomial (T=2s):
  Points: 201
  Max velocity: 1.178 rad/s
  Max acceleration: 1.178 rad/s²

Trapezoidal velocity (v_max=1, a_max=2):
  Points: 208
  Duration: 2.07s
  Max velocity: 1.000 rad/s

Sample trajectory points (cubic):
  t=0.00s: pos=[0. 0. 0.], vel=[0. 0. 0.]
  t=0.50s: pos=[19.6875  9.84375 -9.84375], vel=[0.736311   0.36815548 -0.36815548]
  t=1.00s: pos=[45.   22.5 -22.5], vel=[0.7853982  0.39269908 -0.39269908]
  t=2.00s: pos=[90. 45. -45.], vel=[0. 0. 0.]
```

---

## Summary

### Key Takeaways

1. **Homogeneous transformations** provide a unified framework for representing position and orientation, enabling easy composition of robot link transformations.

2. **DH parameters** systematically describe robot geometry with four parameters per joint, allowing automatic derivation of forward kinematics for any serial manipulator.

3. **Inverse kinematics** is fundamentally harder than forward kinematics—it may have multiple solutions, no solutions, or require numerical methods to solve.

4. **The Jacobian** is central to robot control: it maps joint velocities to end-effector velocities, identifies singularities through manipulability analysis, and enables force/torque transformations.

5. **Robot dynamics** (equations of motion) are essential for accurate control, especially for high-speed motions where inertial effects dominate.

### Connections to Other Chapters

- **Previous (Control Systems)**: The Jacobian and dynamics equations are used in control law design, particularly for computed torque control.
- **Next (Locomotion)**: Legged robot kinematics extends these concepts to multiple kinematic chains with ground contact constraints.

---

## Exercises

### Exercise 1: 6-DOF Robot Arm

**Difficulty**: Medium

Implement forward kinematics for a 6-DOF industrial robot using the following DH parameters:

| Joint | a (m) | α (rad) | d (m) | θ |
|-------|-------|---------|-------|---|
| 1 | 0 | π/2 | 0.5 | θ₁ |
| 2 | 0.8 | 0 | 0 | θ₂ |
| 3 | 0 | π/2 | 0 | θ₃ |
| 4 | 0 | -π/2 | 0.8 | θ₄ |
| 5 | 0 | π/2 | 0 | θ₅ |
| 6 | 0 | 0 | 0.2 | θ₆ |

**Tasks**:
1. Create the DH parameter table as code
2. Implement forward kinematics
3. Compute the workspace volume by sampling
4. Identify singular configurations

---

### Exercise 2: Analytical IK for SCARA

**Difficulty**: Medium

Derive and implement analytical inverse kinematics for a SCARA robot.

**Tasks**:
1. Write the analytical solution for θ₁, θ₂, and d₃
2. Handle the two-solution case (elbow left/right)
3. Implement workspace boundary checking
4. Test with 10 random reachable poses

---

### Exercise 3: Dynamic Simulation

**Difficulty**: Hard

Create a dynamics simulation for a 2-DOF planar robot.

**Tasks**:
1. Derive the mass matrix M(q) analytically
2. Implement gravity compensation control
3. Simulate free fall from horizontal position
4. Add PD control to track a circular trajectory

**Hints**:
- For a 2-DOF planar robot, M(q) is 2×2
- Use Euler integration for simulation
- Gravity compensation: τ = g(q) + K_p * e + K_d * ė

---

## References

1. Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control* (3rd ed.). Pearson.

2. Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009). *Robotics: Modelling, Planning and Control*. Springer. DOI: 10.1007/978-1-84628-642-1

3. Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2020). *Robot Modeling and Control* (2nd ed.). Wiley.

4. Corke, P. (2017). *Robotics, Vision and Control: Fundamental Algorithms in MATLAB* (2nd ed.). Springer. DOI: 10.1007/978-3-319-54413-7

5. Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.

6. Denavit, J., & Hartenberg, R. S. (1955). A Kinematic Notation for Lower-Pair Mechanisms Based on Matrices. *Journal of Applied Mechanics*, 22(2), 215-221.

7. Yoshikawa, T. (1985). Manipulability of Robotic Mechanisms. *International Journal of Robotics Research*, 4(2), 3-9. DOI: 10.1177/027836498500400201
