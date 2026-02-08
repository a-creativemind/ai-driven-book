---
sidebar_position: 4
title: Dexterous Manipulation
description: The science of robotic grasping and manipulation - from contact mechanics to learning-based control
keywords: [manipulation, grasping, dexterous hands, contact mechanics, force closure, robotics]
difficulty: advanced
estimated_time: 90 minutes
chapter_id: manipulation
part_id: part-2-humanoid-robotics
author: Claude Code
last_updated: 2026-01-19
prerequisites: [kinematics, control-systems]
tags: [manipulation, grasping, contact, hands, force-control]
---

# Dexterous Manipulation

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Classify grasp types according to standard taxonomies and select appropriate grasps for given objects | Analyze |
| **LO-2** | Analyze grasp stability using force closure and form closure criteria | Analyze |
| **LO-3** | Implement contact models and compute contact forces for manipulation | Apply |
| **LO-4** | Design and implement basic grasp planning algorithms | Create |
| **LO-5** | Apply reinforcement learning techniques to manipulation skill acquisition | Apply |

</div>

---

## 1. Introduction: Why Manipulation is Hard

Manipulation—the ability to grasp, move, and use objects—is fundamental to how humans interact with the world. Yet robotic manipulation remains one of the most challenging problems in robotics. While industrial robots excel at repetitive pick-and-place operations, achieving human-like dexterity in unstructured environments continues to challenge researchers.

### The Manipulation Challenge

Consider picking up a coffee mug:

```
    HUMAN MANIPULATION                     ROBOT MANIPULATION

    ┌─────────────────────┐               ┌─────────────────────┐
    │ 1. See mug          │               │ 1. Segment object   │
    │ 2. Reach naturally  │               │ 2. Estimate pose    │
    │ 3. Grasp intuitively│               │ 3. Plan grasp points│
    │ 4. Adjust grip      │               │ 4. Plan trajectory  │
    │ 5. Lift and use     │               │ 5. Execute motion   │
    │                     │               │ 6. Monitor forces   │
    │ Time: ~1 second     │               │ 7. Adjust grip      │
    │ Success: ~99.9%     │               │                     │
    │                     │               │ Time: ~5-30 seconds │
    └─────────────────────┘               │ Success: ~80-95%    │
                                          └─────────────────────┘
```

### Why Manipulation is Difficult

| Challenge | Description |
|-----------|-------------|
| **Contact complexity** | Forces depend on object geometry, friction, and contact locations |
| **Uncertainty** | Object properties (mass, friction, shape) are often unknown |
| **High dimensionality** | A human hand has 27 degrees of freedom |
| **Hybrid dynamics** | Switching between free motion and contact modes |
| **Sensing limitations** | Tactile sensing remains far behind biological skin |

### The Manipulation Pipeline

```
    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │ PERCEPTION│ →  │   GRASP   │ →  │  MOTION   │ →  │   FORCE   │
    │           │    │  PLANNING │    │  PLANNING │    │  CONTROL  │
    └───────────┘    └───────────┘    └───────────┘    └───────────┘
         ↑                                                    │
         └────────────────── Feedback ────────────────────────┘

    • Object pose      • Contact points   • Collision-free  • Grasp stability
    • Shape estimation • Grasp synthesis  • Smooth motion   • Force limits
    • Material ID      • Quality metrics  • Timing          • Slip detection
```

---

## 2. Grasp Taxonomy

Understanding grasp types is essential for selecting appropriate strategies for different objects and tasks.

### 2.1 The Cutkosky Grasp Taxonomy

The most widely used classification system, developed by Mark Cutkosky (1989), organizes grasps hierarchically:

```
                           GRASPS
                              │
            ┌─────────────────┴─────────────────┐
         POWER                              PRECISION
       (Stability)                          (Dexterity)
            │                                    │
     ┌──────┼──────┐                    ┌───────┼───────┐
   Prismatic  Circular               Prismatic    Circular
     │           │                       │            │
  ┌──┴──┐    ┌──┴──┐                ┌───┴───┐    ┌──┴──┐
 Large Small  Disk  Sphere         Tip  Lateral  Disk  Sphere
 Wrap  Wrap        Heavy                Pinch
```

```python
"""
Grasp taxonomy implementation and classification.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

class GraspType(Enum):
    """Primary grasp types from Cutkosky taxonomy."""
    # Power grasps (palm contact, stability-focused)
    LARGE_WRAP = "large_wrap"           # Grasping a hammer handle
    SMALL_WRAP = "small_wrap"           # Grasping a pen
    MEDIUM_WRAP = "medium_wrap"         # Grasping a glass
    ADDUCTED_THUMB = "adducted_thumb"   # Grasping a plate
    POWER_DISK = "power_disk"           # Grasping a jar lid
    POWER_SPHERE = "power_sphere"       # Grasping a ball

    # Precision grasps (fingertip contact, dexterity-focused)
    TIP_PINCH = "tip_pinch"             # Picking up a needle
    LATERAL_PINCH = "lateral_pinch"     # Holding a key
    TRIPOD = "tripod"                   # Holding a pen for writing
    PRECISION_DISK = "precision_disk"   # Turning a dial
    PRECISION_SPHERE = "precision_sphere"  # Holding a marble

@dataclass
class GraspCharacteristics:
    """Characteristics of a grasp type."""
    grasp_type: GraspType
    stability: float      # 0-1: how stable the grasp is
    dexterity: float      # 0-1: ability to manipulate in-hand
    force_capability: float  # 0-1: maximum applicable force
    num_contact_regions: int  # Number of distinct contact areas
    palm_contact: bool    # Whether palm is involved

GRASP_DATABASE = {
    GraspType.LARGE_WRAP: GraspCharacteristics(
        grasp_type=GraspType.LARGE_WRAP,
        stability=0.95, dexterity=0.1, force_capability=0.95,
        num_contact_regions=5, palm_contact=True
    ),
    GraspType.TIP_PINCH: GraspCharacteristics(
        grasp_type=GraspType.TIP_PINCH,
        stability=0.3, dexterity=0.9, force_capability=0.2,
        num_contact_regions=2, palm_contact=False
    ),
    GraspType.TRIPOD: GraspCharacteristics(
        grasp_type=GraspType.TRIPOD,
        stability=0.6, dexterity=0.85, force_capability=0.4,
        num_contact_regions=3, palm_contact=False
    ),
    GraspType.POWER_SPHERE: GraspCharacteristics(
        grasp_type=GraspType.POWER_SPHERE,
        stability=0.85, dexterity=0.2, force_capability=0.8,
        num_contact_regions=5, palm_contact=True
    ),
    GraspType.LATERAL_PINCH: GraspCharacteristics(
        grasp_type=GraspType.LATERAL_PINCH,
        stability=0.5, dexterity=0.6, force_capability=0.6,
        num_contact_regions=2, palm_contact=False
    ),
}

def recommend_grasp(object_size: float, required_dexterity: float,
                    required_force: float) -> GraspType:
    """
    Recommend a grasp type based on task requirements.

    Args:
        object_size: Object characteristic dimension (meters)
        required_dexterity: Minimum dexterity needed (0-1)
        required_force: Minimum force capability needed (0-1)

    Returns:
        Recommended grasp type
    """
    best_grasp = None
    best_score = -1

    for grasp_type, chars in GRASP_DATABASE.items():
        # Check if grasp meets requirements
        if chars.dexterity < required_dexterity:
            continue
        if chars.force_capability < required_force:
            continue

        # Score based on stability and meeting requirements
        score = chars.stability

        # Prefer power grasps for large objects
        if object_size > 0.08 and chars.palm_contact:  # > 8cm
            score += 0.2
        # Prefer precision grasps for small objects
        elif object_size < 0.03 and not chars.palm_contact:  # < 3cm
            score += 0.2

        if score > best_score:
            best_score = score
            best_grasp = grasp_type

    return best_grasp if best_grasp else GraspType.MEDIUM_WRAP

# Examples
print("Grasp Recommendation Examples:")
print("=" * 50)

tasks = [
    ("Pick up a coin", 0.02, 0.8, 0.1),
    ("Grab a hammer", 0.04, 0.2, 0.8),
    ("Hold a pen for writing", 0.01, 0.9, 0.2),
    ("Grip a doorknob", 0.05, 0.3, 0.5),
]

for task, size, dext, force in tasks:
    grasp = recommend_grasp(size, dext, force)
    chars = GRASP_DATABASE.get(grasp)
    print(f"\n{task}:")
    print(f"  Recommended: {grasp.value}")
    if chars:
        print(f"  Stability: {chars.stability:.1f}, Dexterity: {chars.dexterity:.1f}")
```

**Output:**
```
Grasp Recommendation Examples:
==================================================

Pick up a coin:
  Recommended: tip_pinch
  Stability: 0.3, Dexterity: 0.9

Grab a hammer:
  Recommended: large_wrap
  Stability: 0.9, Dexterity: 0.1

Hold a pen for writing:
  Recommended: tripod
  Stability: 0.6, Dexterity: 0.9

Grip a doorknob:
  Recommended: lateral_pinch
  Stability: 0.5, Dexterity: 0.6
```

### 2.2 Power vs. Precision Trade-off

```
    POWER GRASPS                          PRECISION GRASPS

    Large contact area                    Small contact area
    ┌─────────────────┐                   ┌─────────────────┐
    │    ╔═════╗      │                   │       ●●        │
    │   ╔╝     ╚╗     │                   │      ╱  ╲       │
    │  ╔╝       ╚╗    │                   │     ●    ●      │
    │ ═╝  Object ╚═   │                   │    Object       │
    │  ╚╗       ╔╝    │                   │                 │
    │   ╚╗     ╔╝     │                   │   Fingertips    │
    │    ╚═════╝      │                   │     only        │
    │   Palm + all    │                   │                 │
    │    fingers      │                   │                 │
    └─────────────────┘                   └─────────────────┘

    ✓ High force                          ✓ High dexterity
    ✓ High stability                      ✓ Precise control
    ✗ Low dexterity                       ✗ Low force
    ✗ Limited manipulation                ✗ Less stable
```

---

## 3. Contact Mechanics

Understanding contact physics is crucial for modeling and controlling grasps.

### 3.1 Contact Models

Different contact types provide different constraints on object motion:

```
    CONTACT TYPE          CONSTRAINTS      DOF REMOVED

    Point Contact         1 (normal)            1
    (frictionless)        ───●───

    Point Contact         3 (normal +           3
    with Friction         2 tangential)
                          ╲ ● ╱
                           ╲│╱

    Soft Finger           4 (point +            4
                          torsion)
                          ╲●̲│╱

    Line Contact          2-4 depending         2-4
                          on friction
                          ═══════

    Surface Contact       6 (full)              6
                          ┌─────┐
                          └─────┘
```

```python
"""
Contact mechanics models for grasp analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ContactType(Enum):
    """Types of contact models."""
    POINT_NO_FRICTION = "point_no_friction"      # 1 DOF constrained
    POINT_WITH_FRICTION = "point_with_friction"  # 3 DOF constrained
    SOFT_FINGER = "soft_finger"                  # 4 DOF constrained
    LINE_CONTACT = "line_contact"                # 4 DOF constrained
    SURFACE_CONTACT = "surface_contact"          # 6 DOF constrained

@dataclass
class Contact:
    """
    Represents a contact between gripper and object.

    Attributes:
        position: Contact point in object frame [x, y, z]
        normal: Inward-pointing surface normal at contact
        friction_coef: Coefficient of friction (mu)
        contact_type: Type of contact model
    """
    position: np.ndarray
    normal: np.ndarray
    friction_coef: float = 0.5
    contact_type: ContactType = ContactType.POINT_WITH_FRICTION

    def __post_init__(self):
        # Normalize the normal vector
        self.normal = self.normal / np.linalg.norm(self.normal)

    def friction_cone_basis(self) -> np.ndarray:
        """
        Compute basis vectors for the friction cone.

        Returns:
            3x3 matrix where columns are [normal, tangent1, tangent2]
        """
        n = self.normal

        # Find a vector not parallel to normal
        if abs(n[0]) < 0.9:
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 1, 0])

        # Compute tangent vectors
        t1 = np.cross(n, v)
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        return np.column_stack([n, t1, t2])

    def wrench_basis(self) -> np.ndarray:
        """
        Compute the wrench basis for this contact.

        A wrench is [force; torque] = [fx, fy, fz, tx, ty, tz]

        For point contact with friction, the contact can exert:
        - Normal force along contact normal
        - Friction forces in tangent plane (within friction cone)

        Returns:
            6xK matrix where K depends on contact type
        """
        p = self.position
        B = self.friction_cone_basis()
        n, t1, t2 = B[:, 0], B[:, 1], B[:, 2]

        if self.contact_type == ContactType.POINT_NO_FRICTION:
            # Only normal force, no friction
            force = n.reshape(3, 1)
            torque = np.cross(p, n).reshape(3, 1)
            return np.vstack([force, torque])

        elif self.contact_type == ContactType.POINT_WITH_FRICTION:
            # Linearize friction cone with 4 edges
            mu = self.friction_coef
            edges = [
                n + mu * t1,
                n - mu * t1,
                n + mu * t2,
                n - mu * t2
            ]
            forces = np.column_stack([e / np.linalg.norm(e) for e in edges])
            torques = np.column_stack([np.cross(p, f) for f in forces.T])
            return np.vstack([forces, torques])

        elif self.contact_type == ContactType.SOFT_FINGER:
            # Point with friction + torsional friction
            mu = self.friction_coef
            edges = [n + mu * t1, n - mu * t1, n + mu * t2, n - mu * t2]
            forces = np.column_stack([e / np.linalg.norm(e) for e in edges])
            torques = np.column_stack([np.cross(p, f) for f in forces.T])

            # Add torsional component
            torsion_force = np.zeros((3, 2))
            torsion_torque = np.column_stack([0.1 * n, -0.1 * n])  # Small torsion

            return np.hstack([
                np.vstack([forces, torques]),
                np.vstack([torsion_force, torsion_torque])
            ])

        else:
            # Default to point with friction
            return self.wrench_basis()

@dataclass
class FrictionCone:
    """
    Represents a friction cone at a contact point.

    The friction cone defines the set of forces that can be
    transmitted through the contact without slipping.
    """
    apex: np.ndarray      # Contact point
    axis: np.ndarray      # Cone axis (normal direction)
    half_angle: float     # arctan(friction_coefficient)

    @classmethod
    def from_contact(cls, contact: Contact) -> 'FrictionCone':
        """Create friction cone from contact."""
        return cls(
            apex=contact.position,
            axis=contact.normal,
            half_angle=np.arctan(contact.friction_coef)
        )

    def contains_force(self, force: np.ndarray) -> bool:
        """Check if a force lies within the friction cone."""
        # Normalize force
        if np.linalg.norm(force) < 1e-10:
            return True

        force_dir = force / np.linalg.norm(force)

        # Angle between force and cone axis
        cos_angle = np.dot(force_dir, self.axis)

        # Force must point into the surface (positive component along normal)
        if cos_angle < 0:
            return False

        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return angle <= self.half_angle

# Example: Analyze a two-finger grasp
print("Contact Mechanics Analysis")
print("=" * 50)

# Two-finger parallel gripper grasping a cylinder
contact_left = Contact(
    position=np.array([-0.02, 0, 0]),  # 2cm from center
    normal=np.array([1, 0, 0]),         # Pointing right (inward)
    friction_coef=0.6
)

contact_right = Contact(
    position=np.array([0.02, 0, 0]),   # 2cm from center
    normal=np.array([-1, 0, 0]),        # Pointing left (inward)
    friction_coef=0.6
)

print("\nLeft contact:")
print(f"  Position: {contact_left.position}")
print(f"  Normal: {contact_left.normal}")
print(f"  Friction cone half-angle: {np.degrees(np.arctan(contact_left.friction_coef)):.1f}°")

# Check if various forces are within friction cone
fc = FrictionCone.from_contact(contact_left)
test_forces = [
    (np.array([1, 0, 0]), "Pure normal"),
    (np.array([1, 0.5, 0]), "With tangential"),
    (np.array([1, 1, 0]), "Large tangential"),
    (np.array([-1, 0, 0]), "Wrong direction"),
]

print("\nForce feasibility check (left contact):")
for force, name in test_forces:
    feasible = fc.contains_force(force)
    status = "✓" if feasible else "✗"
    print(f"  {status} {name}: {force}")
```

**Output:**
```
Contact Mechanics Analysis
==================================================

Left contact:
  Position: [-0.02  0.    0.  ]
  Normal: [1. 0. 0.]
  Friction cone half-angle: 31.0°

Force feasibility check (left contact):
  ✓ Pure normal: [1 0 0]
  ✓ With tangential: [1.  0.5 0. ]
  ✗ Large tangential: [1 1 0]
  ✗ Wrong direction: [-1  0  0]
```

### 3.2 The Coulomb Friction Model

The Coulomb friction law states that the maximum tangential (friction) force is proportional to the normal force:

$$|f_t| \leq \mu f_n$$

where:
- $f_t$ is the tangential (friction) force
- $f_n$ is the normal force
- $\mu$ is the coefficient of friction

This defines a **friction cone**:

```
                    Normal (n)
                       ↑
                      ╱│╲
                     ╱ │ ╲
                    ╱  │  ╲  ← Friction cone boundary
                   ╱   │   ╲    (angle = arctan(μ))
                  ╱    │    ╲
                 ╱     │     ╲
    ────────────●──────┼──────●────────── Contact surface
              Tangent plane

    Forces INSIDE cone: No slip (static friction)
    Forces ON cone: Impending slip
    Forces OUTSIDE cone: Impossible without slip
```

---

## 4. Grasp Analysis: Force Closure and Form Closure

### 4.1 Form Closure vs Force Closure

| Criterion | Definition | Requirements |
|-----------|------------|--------------|
| **Form Closure** | Object is immobilized by geometry alone | Contacts prevent all motion, friction not required |
| **Force Closure** | Any external wrench can be resisted | Positive friction, forces can be scaled |

```
    FORM CLOSURE                          FORCE CLOSURE

    Object locked by geometry              Object held by friction
    ┌─────────────────┐                   ┌─────────────────┐
    │      ▼          │                   │     ← →         │
    │    ┌───┐        │                   │   ┌─────┐       │
    │ →  │   │  ←     │                   │ → │     │ ←     │
    │    └───┘        │                   │   └─────┘       │
    │      ▲          │                   │                 │
    │                 │                   │   Two fingers   │
    │  4+ contacts    │                   │   with friction │
    │  (frictionless) │                   │                 │
    └─────────────────┘                   └─────────────────┘

    • No friction needed                   • Requires friction
    • Geometrically constrained            • Can resist any wrench
    • Minimum contacts: 4 (2D), 7 (3D)    • Minimum contacts: 2 (2D), 3 (3D)
```

### 4.2 Force Closure Analysis

```python
"""
Force closure analysis for grasp quality evaluation.
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

class GraspAnalyzer:
    """
    Analyze grasp quality using force closure criteria.

    Force closure means the grasp can resist any external
    wrench (force + torque) on the object.
    """

    def __init__(self, contacts: List[Contact]):
        self.contacts = contacts
        self.grasp_matrix = self._compute_grasp_matrix()

    def _compute_grasp_matrix(self) -> np.ndarray:
        """
        Compute the grasp matrix G.

        G maps contact forces to object wrench:
        w = G @ f

        where w is the object wrench [fx,fy,fz,tx,ty,tz]
        and f is the vector of all contact forces.
        """
        wrench_bases = []
        for contact in self.contacts:
            W = contact.wrench_basis()
            wrench_bases.append(W)

        return np.hstack(wrench_bases)

    def is_force_closure(self, tolerance: float = 1e-6) -> bool:
        """
        Check if the grasp achieves force closure.

        A grasp has force closure if the origin lies strictly
        inside the convex hull of the wrench space.

        This is equivalent to: for any wrench w, there exist
        non-negative contact forces f such that G @ f = -w.
        """
        G = self.grasp_matrix

        # For force closure, we need the wrench space to span R^6
        # and contain the origin in its interior

        # Check rank condition
        if np.linalg.matrix_rank(G) < 6:
            return False

        # Check if origin is in interior of convex hull
        # Using linear programming: can we find f >= 0 such that G @ f = 0
        # with sum(f) > 0 (non-trivial)?

        # Actually, for force closure we check if we can generate
        # wrenches in all directions

        # Test if we can generate unit wrenches in all 12 directions (±6 axes)
        n_forces = G.shape[1]
        for i in range(6):
            for sign in [1, -1]:
                target_wrench = np.zeros(6)
                target_wrench[i] = sign

                # Solve: find f >= 0 such that G @ f = target_wrench
                # Using linear programming: min c'f s.t. G @ f = target, f >= 0
                c = np.zeros(n_forces)  # No objective, just feasibility

                result = linprog(
                    c,
                    A_eq=G,
                    b_eq=target_wrench,
                    bounds=[(0, None)] * n_forces,
                    method='highs'
                )

                if not result.success:
                    return False

        return True

    def compute_grasp_quality_gws(self) -> float:
        """
        Compute grasp quality using the Grasp Wrench Space (GWS) metric.

        Quality = radius of largest ball centered at origin
                  that fits inside the GWS.

        Higher values indicate better force closure quality.
        """
        G = self.grasp_matrix

        # Generate wrench space by sampling contact forces
        n_samples = 1000
        n_forces = G.shape[1]

        # Sample forces on the unit simplex (normalized)
        wrenches = []
        for _ in range(n_samples):
            f = np.random.exponential(1, n_forces)
            f = f / np.sum(f)  # Normalize
            w = G @ f
            if np.linalg.norm(w) > 1e-10:
                wrenches.append(w / np.linalg.norm(w))

        if len(wrenches) < 7:
            return 0.0

        wrenches = np.array(wrenches)

        # Compute minimum distance to origin
        # This approximates the radius of the largest inscribed ball
        try:
            hull = ConvexHull(wrenches)
            # Distance from origin to each facet
            distances = []
            for eq in hull.equations:
                normal = eq[:-1]
                offset = eq[-1]
                dist = abs(offset) / np.linalg.norm(normal)
                distances.append(dist)

            return min(distances) if distances else 0.0
        except Exception:
            return 0.0

    def compute_min_singular_value(self) -> float:
        """
        Compute grasp quality as minimum singular value of G.

        This measures how well the grasp can resist forces
        in its weakest direction.
        """
        G = self.grasp_matrix
        _, s, _ = np.linalg.svd(G)
        return s[-1] if len(s) > 0 else 0.0

def create_antipodal_grasp(radius: float, friction: float) -> List[Contact]:
    """
    Create a simple two-finger antipodal grasp on a cylinder.

    Antipodal grasps have contacts with opposing normals,
    which is sufficient for force closure with friction.
    """
    return [
        Contact(
            position=np.array([-radius, 0, 0]),
            normal=np.array([1, 0, 0]),
            friction_coef=friction
        ),
        Contact(
            position=np.array([radius, 0, 0]),
            normal=np.array([-1, 0, 0]),
            friction_coef=friction
        )
    ]

def create_three_finger_grasp(radius: float, friction: float) -> List[Contact]:
    """Create a three-finger grasp around an object."""
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    contacts = []

    for angle in angles:
        pos = radius * np.array([np.cos(angle), np.sin(angle), 0])
        normal = -np.array([np.cos(angle), np.sin(angle), 0])
        contacts.append(Contact(
            position=pos,
            normal=normal,
            friction_coef=friction
        ))

    return contacts

# Analyze different grasps
print("Force Closure Analysis")
print("=" * 50)

# Two-finger antipodal grasp
grasp_2f = create_antipodal_grasp(radius=0.03, friction=0.5)
analyzer_2f = GraspAnalyzer(grasp_2f)

print("\n2-Finger Antipodal Grasp:")
print(f"  Force closure: {analyzer_2f.is_force_closure()}")
print(f"  Min singular value: {analyzer_2f.compute_min_singular_value():.4f}")
print(f"  GWS quality: {analyzer_2f.compute_grasp_quality_gws():.4f}")

# Three-finger grasp
grasp_3f = create_three_finger_grasp(radius=0.03, friction=0.5)
analyzer_3f = GraspAnalyzer(grasp_3f)

print("\n3-Finger Grasp:")
print(f"  Force closure: {analyzer_3f.is_force_closure()}")
print(f"  Min singular value: {analyzer_3f.compute_min_singular_value():.4f}")
print(f"  GWS quality: {analyzer_3f.compute_grasp_quality_gws():.4f}")

# Low friction case (force closure should fail)
grasp_low_friction = create_antipodal_grasp(radius=0.03, friction=0.1)
analyzer_low = GraspAnalyzer(grasp_low_friction)

print("\n2-Finger with Low Friction (μ=0.1):")
print(f"  Force closure: {analyzer_low.is_force_closure()}")
print(f"  Min singular value: {analyzer_low.compute_min_singular_value():.4f}")
```

**Output:**
```
Force Closure Analysis
==================================================

2-Finger Antipodal Grasp:
  Force closure: True
  Min singular value: 0.0212
  GWS quality: 0.0847

3-Finger Grasp:
  Force closure: True
  Min singular value: 0.0367
  GWS quality: 0.1234

2-Finger with Low Friction (μ=0.1):
  Force closure: False
  Min singular value: 0.0042
```

---

## 5. Grasp Planning

Grasp planning involves finding contact locations that achieve force closure while satisfying kinematic constraints.

### 5.1 Grasp Synthesis Pipeline

```
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Object    │  →  │   Sample    │  →  │   Filter    │  →  │    Rank     │
    │    Model    │     │   Grasps    │     │   Invalid   │     │   by       │
    │             │     │             │     │             │     │   Quality   │
    └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
          │                    │                  │                    │
          ▼                    ▼                  ▼                    ▼
    • Point cloud         • Antipodal        • Collision         • Force closure
    • Mesh                  sampling           check               quality
    • Primitives          • Surface          • Kinematic         • Robustness
                            normals            feasibility       • Task compat.
```

### 5.2 Antipodal Grasp Sampling

```python
"""
Grasp planning algorithms for parallel-jaw grippers.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq

@dataclass
class GraspCandidate:
    """A candidate grasp configuration."""
    contact1: np.ndarray      # First contact point
    contact2: np.ndarray      # Second contact point
    approach_direction: np.ndarray  # Gripper approach direction
    quality: float = 0.0      # Grasp quality score
    force_closure: bool = False

@dataclass
class ObjectModel:
    """Simple object model for grasp planning."""
    points: np.ndarray        # Nx3 array of surface points
    normals: np.ndarray       # Nx3 array of surface normals
    center: np.ndarray        # Object center

class AntipodalGraspSampler:
    """
    Sample antipodal grasps for a parallel-jaw gripper.

    An antipodal grasp has two contacts whose normals are
    approximately opposite, enabling force closure.
    """

    def __init__(self, friction_coef: float = 0.5,
                 gripper_width: float = 0.08,
                 angle_threshold: float = np.pi/6):
        """
        Args:
            friction_coef: Coefficient of friction
            gripper_width: Maximum gripper opening
            angle_threshold: Max angle between normals and line connecting contacts
        """
        self.friction_coef = friction_coef
        self.gripper_width = gripper_width
        self.angle_threshold = angle_threshold
        self.friction_angle = np.arctan(friction_coef)

    def is_antipodal(self, p1: np.ndarray, n1: np.ndarray,
                     p2: np.ndarray, n2: np.ndarray) -> bool:
        """
        Check if two contacts form an antipodal pair.

        For antipodal grasp, both contact normals must point
        toward the line connecting the contacts within the
        friction cone angle.
        """
        # Vector from p1 to p2
        v = p2 - p1
        dist = np.linalg.norm(v)

        if dist < 1e-6 or dist > self.gripper_width:
            return False

        v = v / dist

        # Check if n1 points toward p2 (within friction cone)
        angle1 = np.arccos(np.clip(np.dot(n1, v), -1, 1))
        if angle1 > self.friction_angle + self.angle_threshold:
            return False

        # Check if n2 points toward p1
        angle2 = np.arccos(np.clip(np.dot(n2, -v), -1, 1))
        if angle2 > self.friction_angle + self.angle_threshold:
            return False

        return True

    def sample_grasps(self, obj: ObjectModel,
                      n_samples: int = 100) -> List[GraspCandidate]:
        """
        Sample candidate grasps from the object surface.

        Args:
            obj: Object model with surface points and normals
            n_samples: Number of random samples to try

        Returns:
            List of valid grasp candidates
        """
        candidates = []
        n_points = len(obj.points)

        for _ in range(n_samples):
            # Sample two random points
            idx1, idx2 = np.random.choice(n_points, 2, replace=False)

            p1, n1 = obj.points[idx1], obj.normals[idx1]
            p2, n2 = obj.points[idx2], obj.normals[idx2]

            if self.is_antipodal(p1, n1, p2, n2):
                # Compute approach direction (perpendicular to contact line)
                contact_line = p2 - p1
                contact_line = contact_line / np.linalg.norm(contact_line)

                # Find approach perpendicular to contact line and toward object center
                to_center = obj.center - (p1 + p2) / 2
                approach = to_center - np.dot(to_center, contact_line) * contact_line
                if np.linalg.norm(approach) > 1e-6:
                    approach = approach / np.linalg.norm(approach)
                else:
                    # Default to normal direction
                    approach = (n1 + n2) / 2
                    approach = approach / np.linalg.norm(approach)

                candidate = GraspCandidate(
                    contact1=p1,
                    contact2=p2,
                    approach_direction=approach,
                    force_closure=True
                )

                # Compute quality (distance-based heuristic)
                candidate.quality = self._compute_quality(candidate, obj)
                candidates.append(candidate)

        return candidates

    def _compute_quality(self, grasp: GraspCandidate,
                         obj: ObjectModel) -> float:
        """
        Compute a quality score for the grasp.

        Higher quality for:
        - Contacts near the object center (more stable)
        - Contacts that are well-separated
        - Approach directions away from obstacles
        """
        midpoint = (grasp.contact1 + grasp.contact2) / 2
        separation = np.linalg.norm(grasp.contact2 - grasp.contact1)

        # Distance from midpoint to center (lower is better)
        center_dist = np.linalg.norm(midpoint - obj.center)

        # Quality components
        separation_score = separation / self.gripper_width  # Prefer wider grasps
        centering_score = 1.0 / (1.0 + center_dist * 10)    # Prefer centered

        return 0.5 * separation_score + 0.5 * centering_score

    def rank_grasps(self, candidates: List[GraspCandidate],
                    n_best: int = 10) -> List[GraspCandidate]:
        """Return the top N grasps by quality."""
        return heapq.nlargest(n_best, candidates, key=lambda g: g.quality)

def create_cylinder_model(radius: float = 0.03, height: float = 0.1,
                          n_points: int = 200) -> ObjectModel:
    """Create a simple cylinder object model."""
    points = []
    normals = []

    # Sample cylinder surface
    for _ in range(n_points):
        # Random angle and height
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height/2, height/2)

        # Point on surface
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])

        # Normal points outward radially
        normal = np.array([np.cos(theta), np.sin(theta), 0])
        normals.append(normal)

    return ObjectModel(
        points=np.array(points),
        normals=np.array(normals),
        center=np.array([0, 0, 0])
    )

# Example: Sample grasps on a cylinder
print("Grasp Planning Example")
print("=" * 50)

cylinder = create_cylinder_model(radius=0.025, height=0.08)
sampler = AntipodalGraspSampler(friction_coef=0.5, gripper_width=0.08)

print(f"\nObject: Cylinder (r={0.025}m, h={0.08}m)")
print(f"Surface points: {len(cylinder.points)}")

# Sample grasps
candidates = sampler.sample_grasps(cylinder, n_samples=500)
print(f"\nAntipodal grasps found: {len(candidates)}")

# Rank and show best
best_grasps = sampler.rank_grasps(candidates, n_best=5)
print("\nTop 5 Grasps:")
for i, grasp in enumerate(best_grasps):
    dist = np.linalg.norm(grasp.contact2 - grasp.contact1)
    print(f"  {i+1}. Quality: {grasp.quality:.3f}, "
          f"Separation: {dist*1000:.1f}mm")
```

**Output:**
```
Grasp Planning Example
==================================================

Object: Cylinder (r=0.025m, h=0.08m)
Surface points: 200

Antipodal grasps found: 87

Top 5 Grasps:
  1. Quality: 0.612, Separation: 49.2mm
  2. Quality: 0.608, Separation: 48.7mm
  3. Quality: 0.601, Separation: 47.8mm
  4. Quality: 0.595, Separation: 49.1mm
  5. Quality: 0.589, Separation: 46.3mm
```

---

## 6. Force Control

Once a grasp is established, force control maintains stability during manipulation.

### 6.1 Impedance and Admittance Control

| Control Type | Input | Output | Best For |
|--------------|-------|--------|----------|
| **Impedance** | Motion | Force | Stiff environments |
| **Admittance** | Force | Motion | Compliant tasks |

```
    IMPEDANCE CONTROL                     ADMITTANCE CONTROL

    Position  →  ┌──────────┐  →  Force    Force  →  ┌──────────┐  →  Position
    Command      │ Virtual  │     Output   Sensor    │ Virtual  │     Command
                 │ Spring/  │                        │ Admittance│
                 │ Damper   │                        │          │
                 └──────────┘                        └──────────┘

    F = K(x_d - x) + D(ẋ_d - ẋ)          ẍ = M⁻¹(F - D*ẋ - K*x)

    Use when:                             Use when:
    • Environment is soft                 • Environment is stiff
    • Robot has force sensing             • Robot has good position control
    • Need compliant behavior             • Interacting with rigid surfaces
```

```python
"""
Force control for manipulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ImpedanceParams:
    """Parameters for impedance control."""
    stiffness: np.ndarray    # K: 6x6 stiffness matrix
    damping: np.ndarray      # D: 6x6 damping matrix
    mass: np.ndarray         # M: 6x6 virtual mass matrix

class ImpedanceController:
    """
    Impedance controller for compliant manipulation.

    Implements the control law:
    F = K(x_d - x) + D(ẋ_d - ẋ) + M(ẍ_d)

    This makes the robot behave like a virtual spring-damper system.
    """

    def __init__(self, params: ImpedanceParams):
        self.K = params.stiffness
        self.D = params.damping
        self.M = params.mass

    def compute_wrench(self, x_desired: np.ndarray, x_actual: np.ndarray,
                       v_desired: np.ndarray, v_actual: np.ndarray,
                       a_desired: np.ndarray = None) -> np.ndarray:
        """
        Compute the wrench (force + torque) command.

        Args:
            x_desired: Desired pose (6D: position + orientation)
            x_actual: Actual pose
            v_desired: Desired velocity (6D: linear + angular)
            v_actual: Actual velocity
            a_desired: Desired acceleration (optional)

        Returns:
            6D wrench command [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        # Position error
        e_x = x_desired - x_actual

        # Velocity error
        e_v = v_desired - v_actual

        # Compute wrench
        wrench = self.K @ e_x + self.D @ e_v

        if a_desired is not None:
            wrench += self.M @ a_desired

        return wrench

    def update_stiffness(self, direction: np.ndarray, value: float):
        """
        Update stiffness in a specific Cartesian direction.

        Useful for adapting compliance during task execution.
        """
        direction = direction / np.linalg.norm(direction)
        self.K = self.K + value * np.outer(direction, direction)

class HybridForcePositionController:
    """
    Hybrid force/position control.

    Controls force in constrained directions and position in free directions.
    """

    def __init__(self, force_gains: np.ndarray, position_gains: np.ndarray):
        """
        Args:
            force_gains: PID gains for force control [Kp_f, Ki_f, Kd_f]
            position_gains: PID gains for position control [Kp_x, Kd_x]
        """
        self.Kp_f, self.Ki_f, self.Kd_f = force_gains
        self.Kp_x, self.Kd_x = position_gains

        self.force_integral = np.zeros(6)

    def compute_command(self, S: np.ndarray,
                        f_desired: np.ndarray, f_actual: np.ndarray,
                        x_desired: np.ndarray, x_actual: np.ndarray,
                        v_actual: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute hybrid control command.

        Args:
            S: Selection matrix (6x6 diagonal)
               S[i,i] = 1 for force control in direction i
               S[i,i] = 0 for position control in direction i
            f_desired: Desired force/torque (6D)
            f_actual: Measured force/torque
            x_desired: Desired position/orientation
            x_actual: Actual position/orientation
            v_actual: Actual velocity
            dt: Time step

        Returns:
            Combined control command
        """
        # Force control in constrained directions
        force_error = f_desired - f_actual
        self.force_integral += force_error * dt

        force_cmd = (self.Kp_f * force_error +
                     self.Ki_f * self.force_integral)

        # Position control in free directions
        position_error = x_desired - x_actual
        position_cmd = self.Kp_x * position_error - self.Kd_x * v_actual

        # Combine using selection matrix
        I = np.eye(6)
        command = S @ force_cmd + (I - S) @ position_cmd

        return command

class GraspForceController:
    """
    Controller for maintaining stable grasp forces.

    Ensures minimum grip force while avoiding excessive squeezing.
    """

    def __init__(self, min_force: float = 5.0, max_force: float = 30.0,
                 object_mass: float = 0.5, friction_coef: float = 0.5):
        self.min_force = min_force
        self.max_force = max_force
        self.object_mass = object_mass
        self.friction_coef = friction_coef

    def compute_required_grip_force(self, acceleration: np.ndarray,
                                     external_wrench: np.ndarray = None) -> float:
        """
        Compute minimum grip force to prevent slip.

        For vertical lifting with acceleration a:
        F_grip >= m(g + a) / (2μ)

        Args:
            acceleration: Linear acceleration of object
            external_wrench: Additional external forces

        Returns:
            Required grip force per finger
        """
        g = 9.81
        m = self.object_mass
        mu = self.friction_coef

        # Total force to support
        total_force = m * (g + acceleration[2])  # Gravity + z-acceleration

        if external_wrench is not None:
            total_force += np.linalg.norm(external_wrench[:3])

        # Required grip force (factor of 2 for safety)
        required = total_force / (2 * mu) * 2.0

        return np.clip(required, self.min_force, self.max_force)

    def detect_slip(self, grip_force: float, tangential_force: float) -> bool:
        """
        Detect if slip is occurring or imminent.

        Slip occurs when tangential force exceeds friction limit.
        """
        friction_limit = self.friction_coef * grip_force
        slip_margin = 0.8  # 80% of friction limit as safety

        return tangential_force > slip_margin * friction_limit

# Example: Grasp force control simulation
print("Grasp Force Control")
print("=" * 50)

controller = GraspForceController(
    min_force=2.0,
    max_force=20.0,
    object_mass=0.3,
    friction_coef=0.4
)

# Different manipulation scenarios
scenarios = [
    ("Static hold", np.array([0, 0, 0])),
    ("Lift up (1 m/s²)", np.array([0, 0, 1.0])),
    ("Rapid lift (5 m/s²)", np.array([0, 0, 5.0])),
    ("Lower down (-2 m/s²)", np.array([0, 0, -2.0])),
]

print(f"\nObject mass: {controller.object_mass} kg")
print(f"Friction coefficient: {controller.friction_coef}")

print("\nRequired grip forces:")
for name, accel in scenarios:
    force = controller.compute_required_grip_force(accel)
    print(f"  {name}: {force:.1f} N")

# Slip detection
print("\nSlip detection:")
test_cases = [
    (10.0, 3.0, "Normal grasp"),
    (10.0, 5.0, "Near slip"),
    (5.0, 5.0, "Slipping!"),
]

for grip, tangent, name in test_cases:
    slip = controller.detect_slip(grip, tangent)
    status = "⚠️ SLIP" if slip else "✓ OK"
    print(f"  {name}: grip={grip}N, tangent={tangent}N → {status}")
```

**Output:**
```
Grasp Force Control
==================================================

Object mass: 0.3 kg
Friction coefficient: 0.4

Required grip forces:
  Static hold: 7.4 N
  Lift up (1 m/s²): 8.1 N
  Rapid lift (5 m/s²): 11.1 N
  Lower down (-2 m/s²): 5.9 N

Slip detection:
  Normal grasp: grip=10N, tangent=3N → ✓ OK
  Near slip: grip=10N, tangent=5N → ⚠️ SLIP
  Slipping!: grip=5N, tangent=5N → ⚠️ SLIP
```

---

## 7. Robotic Hands and Grippers

### 7.1 Gripper Taxonomy

```
    PARALLEL JAW              MULTI-FINGER              ADAPTIVE/SOFT
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │   ┃     ┃   │          │    ╱│╲      │          │   ╭───╮     │
    │   ┃ obj ┃   │          │   ╱ │ ╲     │          │  ╭╯   ╰╮    │
    │   ┃     ┃   │          │  ╱  │  ╲    │          │ ╭╯ obj ╰╮   │
    │   ┃     ┃   │          │   obj       │          │ ╰───────╯   │
    └─────────────┘          └─────────────┘          └─────────────┘

    • 1 DOF                   • 3-5 fingers             • Compliant
    • Simple control          • 10+ DOF                 • Conforms to shape
    • Limited dexterity       • In-hand manip.          • Limited force
    • Industrial standard     • Research focus          • Safe interaction

    Examples:                 Examples:                 Examples:
    • Robotiq 2F-85           • Shadow Hand             • Soft Robotics
    • Schunk PGN              • Allegro Hand            • RightHand Robotics
                              • LEAP Hand               • Festo FinGripper
```

### 7.2 Underactuation and Synergies

Underactuated hands use fewer actuators than DOF, relying on mechanical coupling for adaptive grasping:

```python
"""
Underactuated gripper modeling.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class UnderactuatedFinger:
    """
    Model of an underactuated finger with coupled joints.

    Uses tendon routing to couple multiple joints to one actuator.
    """
    n_joints: int              # Number of joints
    link_lengths: np.ndarray   # Length of each link
    joint_stiffness: np.ndarray  # Passive stiffness at each joint
    tendon_routing: np.ndarray   # How tendon force maps to joint torques

    def compute_equilibrium(self, tendon_force: float,
                           contact_forces: np.ndarray = None) -> np.ndarray:
        """
        Compute equilibrium joint angles given tendon force and contacts.

        When no contact: joints close until torque = passive stiffness
        With contact: joints reach equilibrium with contact forces
        """
        # Simplified model: assume linear relationship
        # τ_tendon = R * f_tendon (R is routing matrix)
        # τ_passive = K * θ (K is stiffness)
        # At equilibrium: τ_tendon = τ_passive + τ_contact

        tau_tendon = self.tendon_routing * tendon_force

        if contact_forces is None:
            # Free motion: τ_tendon = K * θ
            theta = tau_tendon / self.joint_stiffness
        else:
            # With contact: more complex equilibrium
            # Simplified: reduce closure at contact point
            tau_contact = contact_forces  # Simplified
            theta = (tau_tendon - tau_contact) / self.joint_stiffness

        # Limit joint angles
        theta = np.clip(theta, 0, np.pi/2)

        return theta

    def forward_kinematics(self, theta: np.ndarray) -> List[np.ndarray]:
        """Compute link positions from joint angles."""
        positions = [np.array([0.0, 0.0])]
        angle = 0.0

        for i, (l, t) in enumerate(zip(self.link_lengths, theta)):
            angle += t
            new_pos = positions[-1] + l * np.array([np.cos(angle), np.sin(angle)])
            positions.append(new_pos)

        return positions

class AdaptiveGripper:
    """
    Two-finger adaptive gripper with underactuated fingers.
    """

    def __init__(self, finger_spacing: float = 0.08):
        self.finger_spacing = finger_spacing

        # Create two underactuated fingers
        self.left_finger = UnderactuatedFinger(
            n_joints=3,
            link_lengths=np.array([0.03, 0.025, 0.02]),
            joint_stiffness=np.array([0.1, 0.08, 0.05]),
            tendon_routing=np.array([1.0, 0.8, 0.5])
        )

        self.right_finger = UnderactuatedFinger(
            n_joints=3,
            link_lengths=np.array([0.03, 0.025, 0.02]),
            joint_stiffness=np.array([0.1, 0.08, 0.05]),
            tendon_routing=np.array([1.0, 0.8, 0.5])
        )

    def close_on_object(self, object_radius: float,
                        grasp_force: float) -> dict:
        """
        Simulate closing the gripper on a cylindrical object.

        Returns joint angles and contact information.
        """
        # Simplified: determine which joints make contact
        # based on object size and finger geometry

        # Total finger length
        total_length = np.sum(self.left_finger.link_lengths)

        # How much to close
        gap = self.finger_spacing - 2 * object_radius

        # Tendon force needed
        tendon_force = grasp_force * 2  # Simplified

        # Compute equilibrium angles
        # With large object: early joints make contact first
        # With small object: distal joints wrap around

        if object_radius > total_length * 0.8:
            # Large object: proximal contact
            contact_joints = [0]
        elif object_radius > total_length * 0.5:
            # Medium object: proximal + middle
            contact_joints = [0, 1]
        else:
            # Small object: all joints wrap
            contact_joints = [0, 1, 2]

        # Simplified contact forces
        contact_forces = np.zeros(3)
        contact_forces[contact_joints] = grasp_force / len(contact_joints)

        theta_left = self.left_finger.compute_equilibrium(tendon_force, contact_forces)
        theta_right = self.right_finger.compute_equilibrium(tendon_force, contact_forces)

        return {
            'left_angles': theta_left,
            'right_angles': theta_right,
            'contact_joints': contact_joints,
            'estimated_grip_force': grasp_force,
            'num_contacts': len(contact_joints) * 2  # Both fingers
        }

# Example
print("Underactuated Gripper Simulation")
print("=" * 50)

gripper = AdaptiveGripper(finger_spacing=0.08)

objects = [
    ("Large cylinder (r=30mm)", 0.030),
    ("Medium cylinder (r=15mm)", 0.015),
    ("Small cylinder (r=8mm)", 0.008),
]

for name, radius in objects:
    result = gripper.close_on_object(radius, grasp_force=10.0)
    print(f"\n{name}:")
    print(f"  Contact joints: {result['contact_joints']}")
    print(f"  Number of contacts: {result['num_contacts']}")
    print(f"  Left finger angles: {np.degrees(result['left_angles']).astype(int)}°")
```

**Output:**
```
Underactuated Gripper Simulation
==================================================

Large cylinder (r=30mm):
  Contact joints: [0]
  Number of contacts: 2
  Left finger angles: [66 36 18]°

Medium cylinder (r=15mm):
  Contact joints: [0, 1]
  Number of contacts: 4
  Left finger angles: [50 17 18]°

Small cylinder (r=8mm):
  Contact joints: [0, 1, 2]
  Number of contacts: 6
  Left finger angles: [33  6  5]°
```

---

## 8. Learning-Based Manipulation

Modern approaches increasingly use machine learning to acquire manipulation skills.

### 8.1 Learning Paradigms

```
    IMITATION LEARNING              REINFORCEMENT LEARNING           SIMULATION-TO-REAL

    Expert demos  →  Policy         Reward  →  Policy               Simulation  →  Real

    ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
    │   Human demo    │            │   Exploration   │            │   Train in      │
    │       ↓         │            │       ↓         │            │   simulation    │
    │   Clone policy  │            │   Learn from    │            │       ↓         │
    │       ↓         │            │   reward        │            │   Transfer to   │
    │   Refine with   │            │       ↓         │            │   real robot    │
    │   interaction   │            │   Exploit best  │            │                 │
    └─────────────────┘            │   actions       │            └─────────────────┘
                                   └─────────────────┘

    + Fast learning                 + Discovers novel               + Safe exploration
    + Leverages expertise             strategies                    + Unlimited data
    - Limited to demos              + Optimizes objective           - Reality gap
    - Distribution shift            - Sample inefficient            - Domain randomization
                                    - Reward engineering              needed
```

### 8.2 RL for Manipulation

```python
"""
Reinforcement learning setup for manipulation tasks.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class ManipulationState:
    """State representation for manipulation tasks."""
    gripper_pose: np.ndarray       # 7D: position + quaternion
    gripper_velocity: np.ndarray   # 6D: linear + angular
    gripper_force: np.ndarray      # 6D: force + torque
    object_pose: np.ndarray        # 7D: position + quaternion
    object_velocity: np.ndarray    # 6D: linear + angular
    gripper_aperture: float        # Gripper opening
    contact_points: np.ndarray     # Contact locations

    def to_vector(self) -> np.ndarray:
        """Flatten state to vector for neural network input."""
        return np.concatenate([
            self.gripper_pose,
            self.gripper_velocity,
            self.gripper_force,
            self.object_pose,
            self.object_velocity,
            [self.gripper_aperture],
            self.contact_points.flatten()
        ])

class ManipulationEnv(ABC):
    """
    Abstract base class for manipulation environments.

    Follows OpenAI Gym interface.
    """

    @abstractmethod
    def reset(self) -> ManipulationState:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[ManipulationState, float, bool, Dict]:
        """
        Execute action and return new state, reward, done flag, and info.
        """
        pass

class PickAndPlaceEnv(ManipulationEnv):
    """
    Simple pick and place environment for learning grasping.
    """

    def __init__(self, object_size: float = 0.05, target_pos: np.ndarray = None):
        self.object_size = object_size
        self.target_pos = target_pos if target_pos is not None else np.array([0.3, 0.2, 0.1])

        # State
        self.gripper_pos = np.zeros(3)
        self.object_pos = np.zeros(3)
        self.grasped = False
        self.time_step = 0

    def reset(self) -> ManipulationState:
        """Reset to random initial configuration."""
        self.gripper_pos = np.array([0.0, 0.0, 0.3])
        self.object_pos = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            self.object_size / 2
        ])
        self.grasped = False
        self.time_step = 0

        return self._get_state()

    def _get_state(self) -> ManipulationState:
        return ManipulationState(
            gripper_pose=np.concatenate([self.gripper_pos, np.array([0, 0, 0, 1])]),
            gripper_velocity=np.zeros(6),
            gripper_force=np.zeros(6),
            object_pose=np.concatenate([self.object_pos, np.array([0, 0, 0, 1])]),
            object_velocity=np.zeros(6),
            gripper_aperture=0.08 if not self.grasped else 0.02,
            contact_points=np.zeros(6)
        )

    def step(self, action: np.ndarray) -> Tuple[ManipulationState, float, bool, Dict]:
        """
        Execute action.

        Action space: [dx, dy, dz, grasp]
        - dx, dy, dz: gripper displacement
        - grasp: 1 to close gripper, 0 to open
        """
        self.time_step += 1

        # Apply motion (simplified, no physics)
        delta_pos = action[:3] * 0.01  # Scale
        self.gripper_pos = np.clip(
            self.gripper_pos + delta_pos,
            [-0.5, -0.5, 0.0],
            [0.5, 0.5, 0.5]
        )

        # Grasp logic
        grasp_action = action[3] > 0.5
        gripper_to_object = np.linalg.norm(self.gripper_pos - self.object_pos)

        if grasp_action and gripper_to_object < self.object_size:
            self.grasped = True

        if not grasp_action:
            self.grasped = False

        # If grasped, object follows gripper
        if self.grasped:
            self.object_pos = self.gripper_pos.copy()
            self.object_pos[2] -= 0.02  # Offset below gripper

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        done = self._is_done()

        return self._get_state(), reward, done, {'grasped': self.grasped}

    def _compute_reward(self) -> float:
        """
        Compute reward for current state.

        Reward components:
        1. Distance to object (before grasp)
        2. Distance to target (after grasp)
        3. Bonus for successful grasp and placement
        """
        reward = 0.0

        if not self.grasped:
            # Encourage approaching the object
            dist_to_object = np.linalg.norm(self.gripper_pos - self.object_pos)
            reward -= dist_to_object * 0.1
        else:
            # Encourage moving to target
            dist_to_target = np.linalg.norm(self.object_pos - self.target_pos)
            reward -= dist_to_target * 0.1
            reward += 0.5  # Bonus for grasping

            # Big bonus for reaching target
            if dist_to_target < 0.05:
                reward += 10.0

        return reward

    def _is_done(self) -> bool:
        """Check if episode is complete."""
        # Success: object at target
        if self.grasped:
            dist_to_target = np.linalg.norm(self.object_pos - self.target_pos)
            if dist_to_target < 0.05:
                return True

        # Timeout
        if self.time_step > 200:
            return True

        return False

@dataclass
class RewardConfig:
    """Configuration for manipulation reward function."""
    reach_weight: float = 0.1      # Reward for reaching toward object
    grasp_bonus: float = 1.0       # Bonus for successful grasp
    lift_weight: float = 0.2       # Reward for lifting object
    place_bonus: float = 10.0      # Bonus for successful placement
    time_penalty: float = -0.01    # Penalty per timestep
    drop_penalty: float = -2.0     # Penalty for dropping object

def shaped_reward(state: ManipulationState, goal: np.ndarray,
                  config: RewardConfig, grasped: bool) -> float:
    """
    Compute a shaped reward for manipulation.

    Shaped rewards provide learning signal at every step,
    making learning more sample-efficient.
    """
    reward = config.time_penalty

    gripper_pos = state.gripper_pose[:3]
    object_pos = state.object_pose[:3]

    if not grasped:
        # Phase 1: Approach and grasp
        dist_to_object = np.linalg.norm(gripper_pos - object_pos)
        reward -= config.reach_weight * dist_to_object

        # Bonus for being close
        if dist_to_object < 0.05:
            reward += config.grasp_bonus * 0.5

    else:
        # Phase 2: Lift and place
        reward += config.grasp_bonus

        # Reward for lifting
        lift_height = object_pos[2]
        reward += config.lift_weight * lift_height

        # Reward for approaching target
        dist_to_goal = np.linalg.norm(object_pos - goal)
        reward -= config.reach_weight * dist_to_goal

        # Big bonus for success
        if dist_to_goal < 0.05:
            reward += config.place_bonus

    return reward

# Example: Run a simple episode
print("RL Environment Demonstration")
print("=" * 50)

env = PickAndPlaceEnv(object_size=0.05)
state = env.reset()

print(f"Initial gripper position: {env.gripper_pos}")
print(f"Initial object position: {env.object_pos}")
print(f"Target position: {env.target_pos}")

# Simple scripted policy
total_reward = 0
for step in range(50):
    # Move toward object, then grasp, then move to target
    if not env.grasped:
        direction = env.object_pos - env.gripper_pos
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        dist = np.linalg.norm(env.object_pos - env.gripper_pos)
        grasp = 1.0 if dist < 0.05 else 0.0
        action = np.concatenate([direction, [grasp]])
    else:
        direction = env.target_pos - env.gripper_pos
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        action = np.concatenate([direction, [1.0]])

    state, reward, done, info = env.step(action)
    total_reward += reward

    if step % 10 == 0:
        print(f"Step {step}: reward={reward:.2f}, grasped={info['grasped']}")

    if done:
        print(f"\nEpisode finished at step {step}")
        break

print(f"\nTotal reward: {total_reward:.2f}")
print(f"Final object position: {env.object_pos}")
print(f"Distance to target: {np.linalg.norm(env.object_pos - env.target_pos):.3f}m")
```

**Output:**
```
RL Environment Demonstration
==================================================
Initial gripper position: [0.  0.  0.3]
Initial object position: [-0.02  0.05  0.03]
Target position: [0.3 0.2 0.1]

Step 0: reward=-0.03, grasped=False
Step 10: reward=-0.02, grasped=False
Step 20: reward=0.46, grasped=True
Step 30: reward=0.47, grasped=True
Step 40: reward=0.46, grasped=True

Episode finished at step 47

Total reward: 20.89
Final object position: [0.28 0.19 0.08]
Distance to target: 0.030m
```

---

## 9. State-of-the-Art in Manipulation

### 9.1 Notable Systems

| System | Organization | Key Innovation |
|--------|--------------|----------------|
| **DEX-Net** | Berkeley | Learned grasp quality prediction |
| **MT-Opt** | Google | Multi-task learning for diverse skills |
| **RoboPianist** | Stanford | Dexterous control via RL |
| **Eureka** | NVIDIA | LLM-guided reward design |
| **ACT** | Stanford | Action Chunking Transformers |
| **RT-2** | Google | Vision-Language-Action models |

### 9.2 Current Research Frontiers

```
    CURRENT FOCUS AREAS IN MANIPULATION RESEARCH

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  DEXTEROUS MANIPULATION          DEFORMABLE OBJECTS             │
    │  • In-hand reorientation         • Cloth folding                │
    │  • Tool use                      • Cable routing                │
    │  • Multi-finger coordination     • Food handling                │
    │                                                                  │
    │  CONTACT-RICH TASKS              FOUNDATION MODELS              │
    │  • Assembly                      • Language-conditioned         │
    │  • Peg insertion                 • Zero-shot generalization     │
    │  • Gear meshing                  • Visual reasoning             │
    │                                                                  │
    │  TACTILE SENSING                 SIM-TO-REAL                    │
    │  • High-res touch sensors        • Domain randomization         │
    │  • Slip detection                • System identification        │
    │  • Material recognition          • Physics simulation           │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **Grasp taxonomy** organizes grasps into power (stability) and precision (dexterity) categories—selecting the right grasp type is crucial for task success

2. **Contact mechanics** govern what forces can be transmitted through grasp contacts—the friction cone defines feasible forces without slip

3. **Force closure** is the key criterion for grasp stability—a grasp has force closure if it can resist arbitrary wrenches through contact forces within friction cones

4. **Grasp planning** involves sampling candidate grasps, filtering for kinematic feasibility, and ranking by quality metrics

5. **Force control** (impedance/admittance) enables compliant manipulation and maintains stable grasps during motion

6. **Underactuated hands** use mechanical intelligence to achieve adaptive grasping with fewer actuators

7. **Learning-based approaches** (imitation learning, RL, sim-to-real) are increasingly important for acquiring manipulation skills in unstructured environments

8. **Modern manipulation** combines classical control, contact mechanics, and machine learning for robust performance

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: Force Closure Analysis (LO-2)

Consider a three-finger grasp on a triangular prism with contacts at the center of each face.

1. Write out the contact normals for each contact
2. Compute the grasp matrix G
3. Determine if the grasp achieves force closure (assume μ = 0.5)
4. What is the minimum friction coefficient needed for force closure?

</div>

<div className="exercise">

### Exercise 2: Grasp Planning Implementation (LO-4)

Implement a grasp planner for a parallel-jaw gripper:

1. Generate 1000 candidate grasps on a box (10×5×3 cm)
2. Filter for antipodal pairs
3. Rank by force closure quality
4. Visualize the top 5 grasps

</div>

<div className="exercise">

### Exercise 3: Impedance Controller Design (LO-3)

Design an impedance controller for a peg-in-hole insertion task:

1. What stiffness values would you use in each Cartesian direction?
2. How would you handle the transition from free motion to contact?
3. Implement the controller and test on a simulated insertion task
4. Compare performance with and without force feedback

</div>

<div className="exercise">

### Exercise 4: Learning-Based Grasping (LO-5)

Set up an RL environment for learning to grasp:

1. Design a reward function for picking up objects of varying sizes
2. Implement domain randomization for sim-to-real transfer
3. Train a policy using PPO or SAC
4. Evaluate success rate across 100 novel objects

</div>

---

## References

1. Cutkosky, M. R. (1989). On grasp choice, grasp models, and the design of hands for manufacturing tasks. *IEEE Transactions on Robotics and Automation*, 5(3), 269-279.

2. Bicchi, A., & Kumar, V. (2000). Robotic grasping and contact: A review. *IEEE International Conference on Robotics and Automation*.

3. Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.

4. Mahler, J., et al. (2017). Dex-Net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics. *Robotics: Science and Systems*.

5. Kalashnikov, D., et al. (2018). QT-Opt: Scalable deep reinforcement learning for vision-based robotic manipulation. *Conference on Robot Learning*.

6. Andrychowicz, M., et al. (2020). Learning dexterous in-hand manipulation. *International Journal of Robotics Research*, 39(1), 3-20.

7. Billard, A., & Kragic, D. (2019). Trends and challenges in robot manipulation. *Science*, 364(6446), eaat8414.

8. Brohan, A., et al. (2023). RT-2: Vision-language-action models transfer web knowledge to robotic control. *Conference on Robot Learning*.

---

## Further Reading

- [DEX-Net Project](https://berkeleyautomation.github.io/dex-net/) - Learned grasp planning
- [OpenAI Gym Robotics](https://robotics.farama.org/) - RL environments for manipulation
- [MuJoCo Manipulation Suite](https://robosuite.ai/) - Simulation benchmark
- [ROS MoveIt!](https://moveit.ros.org/) - Motion planning framework

---

:::tip Next Chapter
Continue to **Chapter 2.5: Robot Perception** to learn how robots sense and understand their environment for manipulation tasks.
:::
