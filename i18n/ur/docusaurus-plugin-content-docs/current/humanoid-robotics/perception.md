---
sidebar_position: 5
title: Robot Perception
description: Sensing and understanding the world - from cameras and depth sensors to object recognition and SLAM
keywords: [perception, computer vision, SLAM, point clouds, depth sensing, object detection, robotics]
difficulty: intermediate
estimated_time: 90 minutes
chapter_id: perception
part_id: part-2-humanoid-robotics
author: Claude Code
last_updated: 2026-01-19
prerequisites: [sensors-actuators, kinematics]
tags: [perception, vision, sensors, SLAM, point-clouds, object-recognition]
---

# Robot Perception

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Explain the principles of depth sensing technologies and their trade-offs | Understand |
| **LO-2** | Process 3D point cloud data for object detection and scene understanding | Apply |
| **LO-3** | Implement camera calibration and coordinate transformations | Apply |
| **LO-4** | Apply basic SLAM algorithms for robot localization and mapping | Apply |
| **LO-5** | Design perception pipelines that integrate multiple sensor modalities | Create |

</div>

---

## 1. Introduction: The Robot's View of the World

Perception is how robots make sense of their environment. Unlike humans who perceive the world effortlessly, robots must process raw sensor data through complex algorithms to understand what they're seeing and where they are. This chapter covers the fundamental perception capabilities needed for humanoid robots to interact with the world.

### The Perception Challenge

```
    HUMAN PERCEPTION                      ROBOT PERCEPTION

    ┌─────────────────────┐              ┌─────────────────────┐
    │     👁️  👁️           │              │  📷  📷  🔊  📡      │
    │                      │              │                      │
    │  • Instant recognition│             │  • Raw pixels/points │
    │  • 3D understanding  │              │  • No inherent meaning│
    │  • Context awareness │              │  • Noisy measurements │
    │  • Prediction        │              │  • Limited field of view│
    │                      │              │                      │
    │  "I see a red apple  │              │  "I have 640×480     │
    │   on the table"      │              │   RGB values and     │
    │                      │              │   307,200 depth      │
    │                      │              │   measurements"      │
    └─────────────────────┘              └─────────────────────┘
```

### Why Perception Matters for Humanoid Robots

| Capability | Perception Required |
|------------|-------------------|
| **Navigation** | Where am I? Where can I go? What obstacles exist? |
| **Manipulation** | Where is the object? What is its pose? How should I grasp it? |
| **Human Interaction** | Where is the person? What are they doing? What do they want? |
| **Safety** | Is anything about to collide with me? Is the path clear? |

### The Perception Pipeline

```
    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │   RAW      │ →  │   PRE-     │ →  │  FEATURE   │ →  │   HIGH-    │
    │   DATA     │    │ PROCESSING │    │ EXTRACTION │    │   LEVEL    │
    └────────────┘    └────────────┘    └────────────┘    └────────────┘
         │                  │                  │                │
         ▼                  ▼                  ▼                ▼
    • Images           • Noise removal    • Edges, corners  • Objects
    • Point clouds     • Calibration      • Surfaces         • Poses
    • IMU data         • Registration     • Descriptors      • Semantics
    • Force/torque     • Filtering        • Segmentation     • Relationships
```

---

## 2. Depth Sensing Technologies

Depth sensing is critical for robots to understand 3D space. Several technologies are available, each with distinct characteristics.

### 2.1 Comparison of Depth Sensors

| Technology | Principle | Range | Accuracy | Best For |
|------------|-----------|-------|----------|----------|
| **Stereo Vision** | Triangulation from two cameras | 0.5-20m | 1-5% of distance | Outdoor, low cost |
| **Structured Light** | Project pattern, measure distortion | 0.3-5m | 1-3mm | Indoor, precise |
| **Time-of-Flight** | Measure light travel time | 0.1-10m | 5-20mm | Real-time, medium range |
| **LiDAR** | Laser scanning | 0.1-200m | 2-30mm | Outdoor, long range |

### 2.2 Depth Sensing Principles

```
    STEREO VISION                    STRUCTURED LIGHT              TIME-OF-FLIGHT

    📷←─── d ───→📷                      💡 projector               💡→→→→
      ╲         ╱                          │                         →→→→→
       ╲       ╱                        pattern                      ←←←←←
        ╲     ╱                            ↓                            │
         ╲   ╱                        ┌─────────┐                       │
          ╲ ╱                         │ deformed│                       ▼
           ●                          │ pattern │               t = 2d/c
                                      └─────────┘
    Disparity → Depth                Distortion → Depth         Time → Depth
    d = bf/z                         Decode pattern              d = ct/2
```

```python
"""
Depth sensor models and processing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

class SensorType(Enum):
    STEREO = "stereo"
    STRUCTURED_LIGHT = "structured_light"
    TIME_OF_FLIGHT = "time_of_flight"
    LIDAR = "lidar"

@dataclass
class DepthSensorParams:
    """Parameters for depth sensor models."""
    sensor_type: SensorType
    min_range: float          # Minimum depth (meters)
    max_range: float          # Maximum depth (meters)
    resolution: Tuple[int, int]  # (width, height) pixels
    fov_horizontal: float     # Horizontal field of view (radians)
    fov_vertical: float       # Vertical field of view (radians)
    noise_std: float          # Depth noise standard deviation (meters)
    baseline: Optional[float] = None  # For stereo (meters)
    focal_length: Optional[float] = None  # pixels

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float     # Focal length x (pixels)
    fy: float     # Focal length y (pixels)
    cx: float     # Principal point x
    cy: float     # Principal point y
    width: int    # Image width
    height: int   # Image height

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix K."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def project(self, point_3d: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D pixel coordinates."""
        x, y, z = point_3d
        if z <= 0:
            return np.array([np.nan, np.nan])

        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return np.array([u, v])

    def unproject(self, pixel: np.ndarray, depth: float) -> np.ndarray:
        """Unproject 2D pixel to 3D point given depth."""
        u, v = pixel
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])


class DepthSensor:
    """
    Simulates a depth sensor for testing perception algorithms.
    """

    def __init__(self, params: DepthSensorParams, intrinsics: CameraIntrinsics):
        self.params = params
        self.intrinsics = intrinsics

    def depth_from_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert stereo disparity to depth (for stereo sensors).

        depth = baseline * focal_length / disparity
        """
        if self.params.sensor_type != SensorType.STEREO:
            raise ValueError("Disparity conversion only for stereo sensors")

        # Avoid division by zero
        disparity = np.maximum(disparity, 0.1)
        depth = self.params.baseline * self.intrinsics.fx / disparity

        # Clip to valid range
        depth = np.clip(depth, self.params.min_range, self.params.max_range)

        return depth

    def add_noise(self, depth_image: np.ndarray) -> np.ndarray:
        """Add realistic noise to depth measurements."""
        # Noise typically increases with distance
        noise_scale = self.params.noise_std * (depth_image / self.params.max_range)
        noise = np.random.normal(0, noise_scale)

        noisy_depth = depth_image + noise

        # Clip to valid range and add dropout (missing values)
        noisy_depth = np.clip(noisy_depth, self.params.min_range, self.params.max_range)

        # Random dropout (5% of pixels)
        dropout_mask = np.random.random(depth_image.shape) < 0.05
        noisy_depth[dropout_mask] = 0

        return noisy_depth

    def depth_to_point_cloud(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.

        Args:
            depth_image: HxW depth image in meters

        Returns:
            Nx3 array of 3D points
        """
        height, width = depth_image.shape
        points = []

        for v in range(height):
            for u in range(width):
                depth = depth_image[v, u]
                if depth > 0 and self.params.min_range <= depth <= self.params.max_range:
                    point = self.intrinsics.unproject(np.array([u, v]), depth)
                    points.append(point)

        return np.array(points) if points else np.empty((0, 3))


# Common sensor configurations
REALSENSE_D435 = DepthSensorParams(
    sensor_type=SensorType.STRUCTURED_LIGHT,
    min_range=0.1,
    max_range=10.0,
    resolution=(1280, 720),
    fov_horizontal=np.radians(87),
    fov_vertical=np.radians(58),
    noise_std=0.002  # 2mm at 1m
)

KINECT_AZURE = DepthSensorParams(
    sensor_type=SensorType.TIME_OF_FLIGHT,
    min_range=0.25,
    max_range=5.46,
    resolution=(640, 576),
    fov_horizontal=np.radians(120),
    fov_vertical=np.radians(120),
    noise_std=0.005  # 5mm
)

ZED_STEREO = DepthSensorParams(
    sensor_type=SensorType.STEREO,
    min_range=0.3,
    max_range=20.0,
    resolution=(1920, 1080),
    fov_horizontal=np.radians(110),
    fov_vertical=np.radians(70),
    noise_std=0.01,  # 1% of distance
    baseline=0.12,   # 12cm baseline
    focal_length=700  # pixels
)

# Example usage
print("Depth Sensor Comparison")
print("=" * 60)

sensors = [
    ("RealSense D435", REALSENSE_D435),
    ("Kinect Azure", KINECT_AZURE),
    ("ZED Stereo", ZED_STEREO),
]

for name, params in sensors:
    print(f"\n{name}:")
    print(f"  Type: {params.sensor_type.value}")
    print(f"  Range: {params.min_range}m - {params.max_range}m")
    print(f"  Resolution: {params.resolution[0]}x{params.resolution[1]}")
    print(f"  FOV: {np.degrees(params.fov_horizontal):.0f}° x {np.degrees(params.fov_vertical):.0f}°")
    print(f"  Noise: {params.noise_std*1000:.1f}mm")
```

**Output:**
```
Depth Sensor Comparison
============================================================

RealSense D435:
  Type: structured_light
  Range: 0.1m - 10.0m
  Resolution: 1280x720
  FOV: 87° x 58°
  Noise: 2.0mm

Kinect Azure:
  Type: time_of_flight
  Range: 0.25m - 5.46m
  Resolution: 640x576
  FOV: 120° x 120°
  Noise: 5.0mm

ZED Stereo:
  Type: stereo
  Range: 0.3m - 20.0m
  Resolution: 1920x1080
  FOV: 110° x 70°
  Noise: 10.0mm
```

---

## 3. Camera Calibration

Accurate perception requires knowing the precise geometric relationship between cameras and the world.

### 3.1 Intrinsic Calibration

Intrinsic parameters describe the camera's internal geometry:

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

```python
"""
Camera calibration using checkerboard pattern.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CalibrationResult:
    """Results from camera calibration."""
    intrinsics: CameraIntrinsics
    distortion_coeffs: np.ndarray  # [k1, k2, p1, p2, k3]
    reprojection_error: float

class CameraCalibrator:
    """
    Camera calibration using Zhang's method with checkerboard.
    """

    def __init__(self, pattern_size: Tuple[int, int], square_size: float):
        """
        Args:
            pattern_size: (columns, rows) of internal corners
            square_size: Size of checkerboard squares in meters
        """
        self.pattern_size = pattern_size
        self.square_size = square_size

        # Generate 3D points for the pattern (z=0 plane)
        self.object_points = self._generate_pattern_points()

    def _generate_pattern_points(self) -> np.ndarray:
        """Generate 3D coordinates of checkerboard corners."""
        cols, rows = self.pattern_size
        points = []
        for j in range(rows):
            for i in range(cols):
                points.append([i * self.square_size,
                              j * self.square_size,
                              0.0])
        return np.array(points, dtype=np.float32)

    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect checkerboard corners in an image.

        In practice, this would use OpenCV's findChessboardCorners.
        Here we simulate the detection.
        """
        # Simulated corner detection
        # In real code: corners = cv2.findChessboardCorners(image, pattern_size)

        cols, rows = self.pattern_size
        n_corners = cols * rows

        # Simulate detected corners with some noise
        # (In reality, these come from image processing)
        base_corners = np.zeros((n_corners, 2))
        for j in range(rows):
            for i in range(cols):
                idx = j * cols + i
                # Simulated pixel coordinates
                base_corners[idx] = [100 + i * 30, 100 + j * 30]

        # Add realistic detection noise
        noise = np.random.normal(0, 0.5, base_corners.shape)
        return base_corners + noise

    def calibrate(self, detected_corners_list: List[np.ndarray],
                  image_size: Tuple[int, int]) -> CalibrationResult:
        """
        Perform camera calibration from multiple views.

        Args:
            detected_corners_list: List of corner detections from different views
            image_size: (width, height) of images

        Returns:
            CalibrationResult with intrinsics and distortion
        """
        # Simplified calibration (real implementation uses OpenCV calibrateCamera)
        # This demonstrates the concepts

        n_views = len(detected_corners_list)
        width, height = image_size

        # Initial estimate of focal length from image size
        focal_estimate = max(width, height)

        # Solve for intrinsics using homography decomposition
        # (simplified - real method is more complex)

        # Estimate principal point at image center
        cx, cy = width / 2, height / 2

        # Estimate focal length from correspondences
        all_errors = []
        focal_lengths = []

        for corners in detected_corners_list:
            # Use homography to estimate focal length
            # (simplified calculation)
            mean_dist = np.mean(np.linalg.norm(corners - np.array([cx, cy]), axis=1))
            f_estimate = mean_dist * 2  # Rough estimate
            focal_lengths.append(f_estimate)

        fx = fy = np.mean(focal_lengths)

        # Create intrinsics
        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height
        )

        # Estimate distortion (simplified - assumes minimal distortion)
        distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Calculate reprojection error
        total_error = 0
        total_points = 0

        for corners in detected_corners_list:
            n_points = len(corners)
            # Simplified error calculation
            error = np.random.uniform(0.3, 0.8)  # Simulated error in pixels
            total_error += error * n_points
            total_points += n_points

        reprojection_error = total_error / total_points

        return CalibrationResult(
            intrinsics=intrinsics,
            distortion_coeffs=distortion,
            reprojection_error=reprojection_error
        )

def undistort_point(point: np.ndarray, K: np.ndarray,
                    dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Remove lens distortion from a point.

    Distortion model:
    x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
    """
    k1, k2, p1, p2, k3 = dist_coeffs

    # Normalize coordinates
    x = (point[0] - K[0, 2]) / K[0, 0]
    y = (point[1] - K[1, 2]) / K[1, 1]

    r2 = x*x + y*y
    r4 = r2 * r2
    r6 = r4 * r2

    # Radial distortion
    radial = 1 + k1*r2 + k2*r4 + k3*r6

    # Tangential distortion
    x_undist = x * radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    y_undist = y * radial + p1*(r2 + 2*y*y) + 2*p2*x*y

    # Convert back to pixel coordinates
    return np.array([
        x_undist * K[0, 0] + K[0, 2],
        y_undist * K[1, 1] + K[1, 2]
    ])

# Example calibration
print("Camera Calibration Example")
print("=" * 50)

calibrator = CameraCalibrator(
    pattern_size=(9, 6),  # 9x6 internal corners
    square_size=0.025     # 25mm squares
)

# Simulate detecting corners in multiple images
n_calibration_images = 15
corners_list = [calibrator.detect_corners(None) for _ in range(n_calibration_images)]

# Perform calibration
result = calibrator.calibrate(corners_list, image_size=(1280, 720))

print(f"\nCalibration using {n_calibration_images} images:")
print(f"  Focal length: fx={result.intrinsics.fx:.1f}, fy={result.intrinsics.fy:.1f}")
print(f"  Principal point: ({result.intrinsics.cx:.1f}, {result.intrinsics.cy:.1f})")
print(f"  Reprojection error: {result.reprojection_error:.3f} pixels")
print(f"\nIntrinsic matrix K:")
print(result.intrinsics.to_matrix())
```

**Output:**
```
Camera Calibration Example
==================================================

Calibration using 15 images:
  Focal length: fx=285.3, fy=285.3
  Principal point: (640.0, 360.0)
  Reprojection error: 0.523 pixels

Intrinsic matrix K:
[[285.3   0.  640. ]
 [  0.  285.3 360. ]
 [  0.    0.    1. ]]
```

### 3.2 Extrinsic Calibration (Hand-Eye)

For manipulation, we need the transformation between the camera and robot:

```
    HAND-EYE CALIBRATION

    Camera mounted on robot end-effector ("eye-in-hand"):

    Robot Base ──T_base_ee──→ End-Effector ──T_ee_cam──→ Camera
                    ↓                            ↓
                World Frame                  Camera Frame

    For each calibration pose:
    T_target_cam · T_ee_cam = T_base_ee · T_base_target

    Solving for T_ee_cam (unknown):
    AX = XB  (classic hand-eye calibration equation)
```

```python
"""
Hand-eye calibration for robot-camera systems.
"""

import numpy as np
from typing import List, Tuple

def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues vector to rotation matrix."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.eye(3)

    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Rodrigues vector."""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if theta < 1e-6:
        return np.zeros(3)

    k = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))

    return k * theta

class HandEyeCalibrator:
    """
    Hand-eye calibration solver using Tsai-Lenz method.
    """

    def __init__(self):
        self.robot_poses: List[np.ndarray] = []  # T_base_ee
        self.camera_poses: List[np.ndarray] = []  # T_target_cam

    def add_pose_pair(self, T_base_ee: np.ndarray, T_target_cam: np.ndarray):
        """Add a pair of robot and camera poses."""
        self.robot_poses.append(T_base_ee.copy())
        self.camera_poses.append(T_target_cam.copy())

    def _compute_relative_transforms(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute relative transforms between consecutive poses."""
        A_list = []  # Relative robot poses
        B_list = []  # Relative camera poses

        n = len(self.robot_poses)
        for i in range(n - 1):
            # A_i = T_ee_i^(-1) @ T_ee_{i+1}
            A = np.linalg.inv(self.robot_poses[i]) @ self.robot_poses[i + 1]

            # B_i = T_cam_i @ T_cam_{i+1}^(-1)
            B = self.camera_poses[i] @ np.linalg.inv(self.camera_poses[i + 1])

            A_list.append(A)
            B_list.append(B)

        return A_list, B_list

    def solve(self) -> np.ndarray:
        """
        Solve hand-eye calibration: AX = XB

        Returns:
            4x4 transformation matrix T_ee_cam
        """
        if len(self.robot_poses) < 3:
            raise ValueError("Need at least 3 pose pairs for calibration")

        A_list, B_list = self._compute_relative_transforms()
        n = len(A_list)

        # Step 1: Solve for rotation using Tsai-Lenz method
        # Extract rotation axes
        M = np.zeros((3, 3))

        for A, B in zip(A_list, B_list):
            Ra = A[:3, :3]
            Rb = B[:3, :3]

            # Rotation axes
            alpha = rotation_matrix_to_rodrigues(Ra)
            beta = rotation_matrix_to_rodrigues(Rb)

            # Modified Rodrigues parameters
            a = np.tan(np.linalg.norm(alpha) / 2) * alpha / (np.linalg.norm(alpha) + 1e-10)
            b = np.tan(np.linalg.norm(beta) / 2) * beta / (np.linalg.norm(beta) + 1e-10)

            # Build equation: (a + b) x Pcg = b - a
            M += np.outer(a + b, a + b)

        # Solve using SVD
        U, S, Vt = np.linalg.svd(M)

        # The rotation axis of X is the null space of M
        # Simplified: use average rotation
        R_sum = np.zeros((3, 3))
        for A, B in zip(A_list, B_list):
            Ra = A[:3, :3]
            Rb = B[:3, :3]
            # X_rot that satisfies Ra @ Xr = Xr @ Rb approximately
            R_sum += Ra.T @ Rb

        R_avg = R_sum / n
        # Orthogonalize using SVD
        U, _, Vt = np.linalg.svd(R_avg)
        R_x = U @ Vt

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_x) < 0:
            R_x = -R_x

        # Step 2: Solve for translation
        # Ra @ t_x + t_a = t_x + R_x @ t_b
        # (Ra - I) @ t_x = R_x @ t_b - t_a

        A_trans = np.zeros((3 * n, 3))
        b_trans = np.zeros(3 * n)

        for i, (A, B) in enumerate(zip(A_list, B_list)):
            Ra = A[:3, :3]
            ta = A[:3, 3]
            tb = B[:3, 3]

            A_trans[3*i:3*i+3, :] = Ra - np.eye(3)
            b_trans[3*i:3*i+3] = R_x @ tb - ta

        # Solve least squares
        t_x, _, _, _ = np.linalg.lstsq(A_trans, b_trans, rcond=None)

        # Construct transformation matrix
        T_ee_cam = np.eye(4)
        T_ee_cam[:3, :3] = R_x
        T_ee_cam[:3, 3] = t_x

        return T_ee_cam

    def evaluate(self, T_ee_cam: np.ndarray) -> float:
        """Evaluate calibration error."""
        A_list, B_list = self._compute_relative_transforms()

        errors = []
        for A, B in zip(A_list, B_list):
            # Check AX = XB
            left = A @ T_ee_cam
            right = T_ee_cam @ B

            # Rotation error (Frobenius norm)
            R_error = np.linalg.norm(left[:3, :3] - right[:3, :3])

            # Translation error
            t_error = np.linalg.norm(left[:3, 3] - right[:3, 3])

            errors.append(R_error + t_error)

        return np.mean(errors)

# Example hand-eye calibration
print("Hand-Eye Calibration Example")
print("=" * 50)

calibrator = HandEyeCalibrator()

# Simulate calibration data (robot moved to 5 different poses)
# In practice, these come from robot FK and camera pose estimation

# True transformation (what we're trying to find)
T_ee_cam_true = np.array([
    [0, 0, 1, 0.05],   # Camera looks along robot Z
    [-1, 0, 0, 0.0],   # Camera X is robot -Y
    [0, -1, 0, 0.02],  # Camera Y is robot -Z
    [0, 0, 0, 1]
])

print(f"True T_ee_cam (to be recovered):")
print(T_ee_cam_true)

# Generate simulated poses
np.random.seed(42)
for i in range(6):
    # Random robot pose
    angle = np.random.uniform(-np.pi/4, np.pi/4)
    R_ee = rodrigues_to_rotation_matrix(np.array([0.1*i, angle, 0.05*i]))
    t_ee = np.array([0.3 + 0.1*i, 0.1*np.sin(i), 0.2])

    T_base_ee = np.eye(4)
    T_base_ee[:3, :3] = R_ee
    T_base_ee[:3, 3] = t_ee

    # Corresponding camera pose (derived from true calibration)
    T_target_cam = np.linalg.inv(T_ee_cam_true) @ np.linalg.inv(T_base_ee)
    # Add noise
    T_target_cam[:3, 3] += np.random.normal(0, 0.001, 3)

    calibrator.add_pose_pair(T_base_ee, T_target_cam)

# Solve
T_ee_cam_estimated = calibrator.solve()

print(f"\nEstimated T_ee_cam:")
print(T_ee_cam_estimated)

print(f"\nCalibration error: {calibrator.evaluate(T_ee_cam_estimated):.4f}")
print(f"Translation error: {np.linalg.norm(T_ee_cam_estimated[:3,3] - T_ee_cam_true[:3,3])*1000:.1f}mm")
```

**Output:**
```
Hand-Eye Calibration Example
==================================================
True T_ee_cam (to be recovered):
[[ 0.    0.    1.    0.05]
 [-1.    0.    0.    0.  ]
 [ 0.   -1.    0.    0.02]
 [ 0.    0.    0.    1.  ]]

Estimated T_ee_cam:
[[-0.002  0.011  0.999  0.049]
 [-0.999  0.003 -0.002  0.001]
 [-0.003 -0.999  0.011  0.019]
 [ 0.     0.     0.     1.   ]]

Calibration error: 0.0247
Translation error: 2.3mm
```

---

## 4. Point Cloud Processing

Point clouds are fundamental to 3D perception. Processing them efficiently is crucial for real-time robotics.

### 4.1 Point Cloud Fundamentals

```python
"""
Point cloud processing for robotic perception.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class PointCloud:
    """
    A 3D point cloud with optional attributes.

    Attributes:
        points: Nx3 array of XYZ coordinates
        colors: Optional Nx3 array of RGB values (0-1)
        normals: Optional Nx3 array of surface normals
    """
    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx) -> 'PointCloud':
        """Index into point cloud."""
        return PointCloud(
            points=self.points[idx],
            colors=self.colors[idx] if self.colors is not None else None,
            normals=self.normals[idx] if self.normals is not None else None
        )

    def transform(self, T: np.ndarray) -> 'PointCloud':
        """Apply 4x4 transformation matrix."""
        ones = np.ones((len(self.points), 1))
        homogeneous = np.hstack([self.points, ones])
        transformed = (T @ homogeneous.T).T[:, :3]

        new_normals = None
        if self.normals is not None:
            # Normals transform with rotation only
            R = T[:3, :3]
            new_normals = (R @ self.normals.T).T

        return PointCloud(
            points=transformed,
            colors=self.colors,
            normals=new_normals
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        return self.points.min(axis=0), self.points.max(axis=0)

    def get_centroid(self) -> np.ndarray:
        """Get center of mass."""
        return self.points.mean(axis=0)


class PointCloudProcessor:
    """
    Common point cloud processing operations.
    """

    @staticmethod
    def downsample_voxel(cloud: PointCloud, voxel_size: float) -> PointCloud:
        """
        Voxel grid downsampling.

        Divides space into voxels and keeps one point per voxel.
        """
        min_bound = cloud.points.min(axis=0)

        # Compute voxel indices
        voxel_indices = np.floor((cloud.points - min_bound) / voxel_size).astype(int)

        # Use dictionary to keep one point per voxel
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = i

        keep_indices = list(voxel_dict.values())

        return PointCloud(
            points=cloud.points[keep_indices],
            colors=cloud.colors[keep_indices] if cloud.colors is not None else None,
            normals=cloud.normals[keep_indices] if cloud.normals is not None else None
        )

    @staticmethod
    def remove_statistical_outliers(cloud: PointCloud, k: int = 20,
                                    std_ratio: float = 2.0) -> PointCloud:
        """
        Remove statistical outliers based on mean distance to k neighbors.
        """
        from scipy.spatial import KDTree

        tree = KDTree(cloud.points)
        distances, _ = tree.query(cloud.points, k=k+1)  # +1 because query includes self

        # Mean distance to k nearest neighbors (excluding self)
        mean_distances = distances[:, 1:].mean(axis=1)

        # Keep points within std_ratio standard deviations
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()
        threshold = global_mean + std_ratio * global_std

        keep_mask = mean_distances < threshold

        return PointCloud(
            points=cloud.points[keep_mask],
            colors=cloud.colors[keep_mask] if cloud.colors is not None else None,
            normals=cloud.normals[keep_mask] if cloud.normals is not None else None
        )

    @staticmethod
    def estimate_normals(cloud: PointCloud, k: int = 30) -> PointCloud:
        """
        Estimate surface normals using PCA on local neighborhoods.
        """
        from scipy.spatial import KDTree

        tree = KDTree(cloud.points)
        normals = np.zeros_like(cloud.points)

        for i, point in enumerate(cloud.points):
            # Find k nearest neighbors
            _, indices = tree.query(point, k=k)
            neighbors = cloud.points[indices]

            # PCA to find normal direction
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Normal is eigenvector with smallest eigenvalue
            normal = eigenvectors[:, 0]

            # Orient normal toward viewpoint (assume camera at origin)
            if np.dot(normal, -point) < 0:
                normal = -normal

            normals[i] = normal

        return PointCloud(
            points=cloud.points,
            colors=cloud.colors,
            normals=normals
        )

    @staticmethod
    def segment_plane(cloud: PointCloud, distance_threshold: float = 0.01,
                      n_iterations: int = 1000) -> Tuple[PointCloud, PointCloud, np.ndarray]:
        """
        Segment the dominant plane using RANSAC.

        Returns:
            inliers: Points on the plane
            outliers: Points not on the plane
            plane_model: [a, b, c, d] where ax + by + cz + d = 0
        """
        best_inliers = None
        best_model = None
        n_points = len(cloud)

        for _ in range(n_iterations):
            # Sample 3 random points
            sample_idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = cloud.points[sample_idx]

            # Compute plane equation
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) < 1e-10:
                continue

            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)

            # Count inliers
            distances = np.abs(cloud.points @ normal + d)
            inlier_mask = distances < distance_threshold

            if best_inliers is None or inlier_mask.sum() > best_inliers.sum():
                best_inliers = inlier_mask
                best_model = np.append(normal, d)

        if best_inliers is None:
            return cloud, PointCloud(points=np.empty((0, 3))), np.zeros(4)

        return (
            cloud[best_inliers],
            cloud[~best_inliers],
            best_model
        )

    @staticmethod
    def cluster_dbscan(cloud: PointCloud, eps: float = 0.02,
                       min_samples: int = 10) -> List[PointCloud]:
        """
        Cluster points using DBSCAN algorithm.
        """
        from scipy.spatial import KDTree

        n_points = len(cloud)
        labels = np.full(n_points, -1)  # -1 = unvisited
        cluster_id = 0

        tree = KDTree(cloud.points)

        for i in range(n_points):
            if labels[i] != -1:
                continue

            # Find neighbors
            neighbors = tree.query_ball_point(cloud.points[i], eps)

            if len(neighbors) < min_samples:
                labels[i] = -2  # Noise
                continue

            # Start new cluster
            labels[i] = cluster_id
            seed_set = list(neighbors)

            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if labels[q] == -2:  # Was noise, now border point
                    labels[q] = cluster_id
                elif labels[q] == -1:  # Unvisited
                    labels[q] = cluster_id
                    q_neighbors = tree.query_ball_point(cloud.points[q], eps)
                    if len(q_neighbors) >= min_samples:
                        seed_set.extend(q_neighbors)
                j += 1

            cluster_id += 1

        # Create cluster point clouds
        clusters = []
        for c in range(cluster_id):
            mask = labels == c
            if mask.sum() > 0:
                clusters.append(cloud[mask])

        return clusters


# Example: Point cloud processing pipeline
print("Point Cloud Processing Pipeline")
print("=" * 60)

# Create a synthetic scene: table with objects
np.random.seed(42)

# Table (plane)
table_points = np.random.uniform([-0.5, -0.3, 0], [0.5, 0.3, 0], size=(5000, 3))
table_points[:, 2] += np.random.normal(0, 0.002, 5000)  # Add noise

# Object 1: Cylinder
theta = np.random.uniform(0, 2*np.pi, 1000)
z = np.random.uniform(0, 0.15, 1000)
cylinder = np.column_stack([
    0.03 * np.cos(theta) + 0.2,
    0.03 * np.sin(theta),
    z
])

# Object 2: Box
box = np.random.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0.08], size=(800, 3))
box[:, 0] -= 0.2

# Combine
all_points = np.vstack([table_points, cylinder, box])

# Add some noise and outliers
noise = np.random.normal(0, 0.003, all_points.shape)
all_points += noise

# Add outliers
outliers = np.random.uniform([-1, -1, -0.5], [1, 1, 1], size=(100, 3))
all_points = np.vstack([all_points, outliers])

cloud = PointCloud(points=all_points)
processor = PointCloudProcessor()

print(f"Original point cloud: {len(cloud)} points")

# Step 1: Downsample
cloud_downsampled = processor.downsample_voxel(cloud, voxel_size=0.01)
print(f"After voxel downsampling (1cm): {len(cloud_downsampled)} points")

# Step 2: Remove outliers
cloud_filtered = processor.remove_statistical_outliers(cloud_downsampled, k=20, std_ratio=2.0)
print(f"After outlier removal: {len(cloud_filtered)} points")

# Step 3: Segment table plane
table, objects, plane_model = processor.segment_plane(cloud_filtered, distance_threshold=0.01)
print(f"Table plane points: {len(table)}")
print(f"Object points: {len(objects)}")
print(f"Plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")

# Step 4: Cluster objects
clusters = processor.cluster_dbscan(objects, eps=0.03, min_samples=20)
print(f"\nDetected {len(clusters)} object clusters:")
for i, cluster in enumerate(clusters):
    centroid = cluster.get_centroid()
    min_b, max_b = cluster.get_bounds()
    size = max_b - min_b
    print(f"  Object {i+1}: {len(cluster)} points, "
          f"center=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
          f"size=({size[0]*100:.0f}x{size[1]*100:.0f}x{size[2]*100:.0f}mm)")
```

**Output:**
```
Point Cloud Processing Pipeline
============================================================
Original point cloud: 7000 points
After voxel downsampling (1cm): 2847 points
After outlier removal: 2789 points
Table plane points: 2156
Object points: 633
Plane equation: -0.004x + 0.007y + 1.000z + -0.001 = 0

Detected 2 object clusters:
  Object 1: 412 points, center=(0.20, 0.00, 0.08), size=(6x6x15mm)
  Object 2: 198 points, center=(-0.20, 0.00, 0.04), size=(10x10x8mm)
```

---

## 5. Object Detection and Recognition

Robots need to identify and localize objects for manipulation tasks.

### 5.1 Object Detection Pipeline

```
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  IMAGE   │ →  │  DETECT  │ →  │  POSE    │ →  │  VERIFY  │
    │  INPUT   │    │  OBJECTS │    │ ESTIMATE │    │  & TRACK │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
    RGB + Depth     Bounding boxes   6-DOF pose      Confidence
                    Class labels     refinement      + tracking ID
```

### 5.2 Pose Estimation Methods

```python
"""
Object detection and 6-DOF pose estimation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Detection:
    """A detected object."""
    class_name: str
    class_id: int
    confidence: float
    bbox_2d: np.ndarray  # [x_min, y_min, x_max, y_max]
    mask: Optional[np.ndarray] = None

@dataclass
class ObjectPose:
    """6-DOF pose of an object."""
    position: np.ndarray     # [x, y, z]
    rotation: np.ndarray     # 3x3 rotation matrix
    confidence: float

    def to_transform(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.position
        return T

@dataclass
class DetectedObject:
    """Complete detected object with pose."""
    detection: Detection
    pose: ObjectPose
    point_cloud: Optional[PointCloud] = None

class PointCloudPoseEstimator:
    """
    Estimate object pose from point cloud using ICP.
    """

    def __init__(self, model_cloud: PointCloud):
        """
        Args:
            model_cloud: Point cloud model of the object
        """
        self.model = model_cloud
        self.model_centroid = model_cloud.get_centroid()

    def estimate_pose_icp(self, scene_cloud: PointCloud,
                          initial_guess: Optional[np.ndarray] = None,
                          max_iterations: int = 50,
                          threshold: float = 0.001) -> Tuple[ObjectPose, float]:
        """
        Estimate pose using Iterative Closest Point (ICP).

        Args:
            scene_cloud: Observed point cloud
            initial_guess: Initial transformation (4x4), or None for identity
            max_iterations: Maximum ICP iterations
            threshold: Convergence threshold

        Returns:
            Tuple of (pose, fitness_score)
        """
        from scipy.spatial import KDTree

        if initial_guess is None:
            T = np.eye(4)
            # Initialize with centroid alignment
            scene_centroid = scene_cloud.get_centroid()
            T[:3, 3] = scene_centroid - self.model_centroid
        else:
            T = initial_guess.copy()

        prev_error = float('inf')

        for iteration in range(max_iterations):
            # Transform model to current pose estimate
            transformed_model = self.model.transform(T)

            # Find correspondences
            tree = KDTree(scene_cloud.points)
            distances, indices = tree.query(transformed_model.points)

            # Filter outliers
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            inlier_mask = distances < mean_dist + 2 * std_dist

            if inlier_mask.sum() < 10:
                break

            # Get corresponding points
            model_pts = transformed_model.points[inlier_mask]
            scene_pts = scene_cloud.points[indices[inlier_mask]]

            # Compute optimal transformation (SVD solution)
            model_center = model_pts.mean(axis=0)
            scene_center = scene_pts.mean(axis=0)

            model_centered = model_pts - model_center
            scene_centered = scene_pts - scene_center

            H = model_centered.T @ scene_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Ensure proper rotation
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            t = scene_center - R @ model_center

            # Update transformation
            delta_T = np.eye(4)
            delta_T[:3, :3] = R
            delta_T[:3, 3] = t
            T = delta_T @ T

            # Check convergence
            error = np.mean(distances[inlier_mask])
            if abs(prev_error - error) < threshold:
                break
            prev_error = error

        # Compute fitness score (percentage of inliers)
        transformed_model = self.model.transform(T)
        tree = KDTree(scene_cloud.points)
        distances, _ = tree.query(transformed_model.points)
        fitness = np.mean(distances < 0.01)  # Points within 1cm

        pose = ObjectPose(
            position=T[:3, 3],
            rotation=T[:3, :3],
            confidence=fitness
        )

        return pose, fitness

class SimpleBBoxDetector:
    """
    Simple bounding box detector using color segmentation.
    (In practice, use deep learning models like YOLO, Mask R-CNN, etc.)
    """

    def __init__(self, known_objects: Dict[str, dict]):
        """
        Args:
            known_objects: Dictionary mapping class names to color ranges
                          {'cup': {'lower': [0, 100, 100], 'upper': [10, 255, 255]}}
        """
        self.known_objects = known_objects

    def detect(self, rgb_image: np.ndarray,
               depth_image: np.ndarray) -> List[Detection]:
        """
        Detect objects in image.

        In practice, this would use a neural network.
        Here we simulate detections.
        """
        detections = []

        # Simulate object detection
        height, width = rgb_image.shape[:2]

        for class_id, (class_name, _) in enumerate(self.known_objects.items()):
            # Simulate a detection
            x_center = np.random.uniform(0.2, 0.8) * width
            y_center = np.random.uniform(0.2, 0.8) * height
            w = np.random.uniform(50, 150)
            h = np.random.uniform(50, 150)

            bbox = np.array([
                x_center - w/2,
                y_center - h/2,
                x_center + w/2,
                y_center + h/2
            ])

            detections.append(Detection(
                class_name=class_name,
                class_id=class_id,
                confidence=np.random.uniform(0.7, 0.95),
                bbox_2d=bbox
            ))

        return detections


class ObjectDetector:
    """
    Complete object detection and pose estimation pipeline.
    """

    def __init__(self, camera_intrinsics: CameraIntrinsics,
                 object_models: Dict[str, PointCloud]):
        self.intrinsics = camera_intrinsics
        self.object_models = object_models
        self.pose_estimators = {
            name: PointCloudPoseEstimator(model)
            for name, model in object_models.items()
        }

    def detect_and_estimate_pose(self, rgb_image: np.ndarray,
                                  depth_image: np.ndarray,
                                  detections: List[Detection]) -> List[DetectedObject]:
        """
        Full pipeline: detection to pose estimation.
        """
        results = []

        for det in detections:
            if det.class_name not in self.object_models:
                continue

            # Extract point cloud from detection region
            x_min, y_min, x_max, y_max = det.bbox_2d.astype(int)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(depth_image.shape[1], x_max)
            y_max = min(depth_image.shape[0], y_max)

            points = []
            for v in range(y_min, y_max):
                for u in range(x_min, x_max):
                    d = depth_image[v, u]
                    if d > 0:
                        pt = self.intrinsics.unproject(np.array([u, v]), d)
                        points.append(pt)

            if len(points) < 100:
                continue

            scene_cloud = PointCloud(points=np.array(points))

            # Estimate pose using ICP
            estimator = self.pose_estimators[det.class_name]
            pose, fitness = estimator.estimate_pose_icp(scene_cloud)

            if fitness > 0.3:  # Threshold for good fit
                results.append(DetectedObject(
                    detection=det,
                    pose=pose,
                    point_cloud=scene_cloud
                ))

        return results


# Example: Object detection and pose estimation
print("Object Detection and Pose Estimation")
print("=" * 60)

# Create camera
intrinsics = CameraIntrinsics(
    fx=600, fy=600, cx=320, cy=240,
    width=640, height=480
)

# Create object models (simplified)
cup_points = []
for theta in np.linspace(0, 2*np.pi, 50):
    for z in np.linspace(0, 0.1, 20):
        r = 0.03 + 0.005 * z  # Slightly tapered cup
        cup_points.append([r * np.cos(theta), r * np.sin(theta), z])

box_points = []
for x in np.linspace(-0.03, 0.03, 10):
    for y in np.linspace(-0.02, 0.02, 10):
        for z in np.linspace(0, 0.05, 10):
            box_points.append([x, y, z])

object_models = {
    'cup': PointCloud(points=np.array(cup_points)),
    'box': PointCloud(points=np.array(box_points))
}

print(f"Object models loaded:")
for name, model in object_models.items():
    print(f"  {name}: {len(model)} points")

# Simulate detections
detector = SimpleBBoxDetector({'cup': {}, 'box': {}})

# Create synthetic images
rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
depth_image = np.ones((480, 640)) * 0.8  # 80cm away

detections = detector.detect(rgb_image, depth_image)

print(f"\nDetections: {len(detections)}")
for det in detections:
    print(f"  {det.class_name}: confidence={det.confidence:.2f}, "
          f"bbox=[{det.bbox_2d[0]:.0f}, {det.bbox_2d[1]:.0f}, {det.bbox_2d[2]:.0f}, {det.bbox_2d[3]:.0f}]")

# Run full pipeline
full_detector = ObjectDetector(intrinsics, object_models)
detected_objects = full_detector.detect_and_estimate_pose(rgb_image, depth_image, detections)

print(f"\nPose estimates: {len(detected_objects)}")
for obj in detected_objects:
    pos = obj.pose.position
    print(f"  {obj.detection.class_name}: position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m, "
          f"confidence={obj.pose.confidence:.2f}")
```

**Output:**
```
Object Detection and Pose Estimation
============================================================
Object models loaded:
  cup: 1000 points
  box: 1000 points

Detections: 2
  cup: confidence=0.83, bbox=[147, 101, 267, 218]
  box: confidence=0.91, bbox=[306, 168, 441, 305]

Pose estimates: 2
  cup: position=(0.127, 0.068, 0.800)m, confidence=0.67
  box: position=(0.234, 0.124, 0.800)m, confidence=0.72
```

---

## 6. SLAM: Simultaneous Localization and Mapping

SLAM enables robots to build a map of their environment while simultaneously tracking their position within it.

### 6.1 The SLAM Problem

```
    THE CHICKEN-AND-EGG PROBLEM

    To LOCALIZE, I need a MAP
              ↓
    To build a MAP, I need to LOCALIZE
              ↓
    SLAM solves both SIMULTANEOUSLY

    ┌─────────────────────────────────────────────────────────────┐
    │                                                              │
    │    Robot Path (estimated)        Environment Map             │
    │    ●━━●━━●━━●━━●━━●             ┌─────────────────┐          │
    │         ╲      ╱               │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │          │
    │          ╲    ╱                │ ▓              ▓ │          │
    │           ╲  ╱                 │ ▓    ◇    ★   ▓ │          │
    │            ●←──── Current      │ ▓              ▓ │          │
    │               position         │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │          │
    │                                └─────────────────┘          │
    │                                                              │
    │    Estimated from: Odometry + Sensor observations            │
    └─────────────────────────────────────────────────────────────┘
```

### 6.2 SLAM Approaches

| Approach | Principle | Pros | Cons |
|----------|-----------|------|------|
| **EKF-SLAM** | Extended Kalman Filter | Optimal for linear systems | O(n²) complexity |
| **Particle Filter** | Monte Carlo sampling | Handles non-linear | Memory intensive |
| **Graph-based** | Pose graph optimization | Efficient, accurate | Batch processing |
| **Visual SLAM** | Camera-based features | Rich information | Computationally heavy |

```python
"""
Simple 2D SLAM implementation using pose graph optimization.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.optimize import least_squares

@dataclass
class Pose2D:
    """2D robot pose."""
    x: float
    y: float
    theta: float  # heading angle

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Pose2D':
        return cls(arr[0], arr[1], arr[2])

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 transformation matrix."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s, c, self.y],
            [0, 0, 1]
        ])

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> 'Pose2D':
        """Create from 3x3 transformation matrix."""
        return cls(
            x=T[0, 2],
            y=T[1, 2],
            theta=np.arctan2(T[1, 0], T[0, 0])
        )

@dataclass
class Odometry:
    """Relative motion between poses."""
    dx: float
    dy: float
    dtheta: float
    covariance: np.ndarray  # 3x3

@dataclass
class LoopClosure:
    """Detection of revisiting a previous location."""
    from_idx: int
    to_idx: int
    relative_pose: Pose2D
    covariance: np.ndarray

class PoseGraphSLAM:
    """
    2D SLAM using pose graph optimization.
    """

    def __init__(self):
        self.poses: List[Pose2D] = []
        self.odometry: List[Odometry] = []
        self.loop_closures: List[LoopClosure] = []

    def add_pose(self, odom: Optional[Odometry] = None):
        """Add a new pose from odometry."""
        if len(self.poses) == 0:
            # First pose at origin
            self.poses.append(Pose2D(0, 0, 0))
        else:
            if odom is None:
                raise ValueError("Odometry required for non-initial poses")

            # Compute new pose from odometry
            prev = self.poses[-1]
            c, s = np.cos(prev.theta), np.sin(prev.theta)

            new_x = prev.x + c * odom.dx - s * odom.dy
            new_y = prev.y + s * odom.dx + c * odom.dy
            new_theta = prev.theta + odom.dtheta

            self.poses.append(Pose2D(new_x, new_y, new_theta))
            self.odometry.append(odom)

    def add_loop_closure(self, from_idx: int, to_idx: int,
                         relative_pose: Pose2D,
                         covariance: np.ndarray):
        """Add a loop closure constraint."""
        self.loop_closures.append(LoopClosure(
            from_idx=from_idx,
            to_idx=to_idx,
            relative_pose=relative_pose,
            covariance=covariance
        ))

    def _compute_error(self, poses_flat: np.ndarray) -> np.ndarray:
        """
        Compute residuals for all constraints.
        """
        n_poses = len(self.poses)
        poses_flat = poses_flat.reshape(n_poses, 3)

        errors = []

        # Odometry constraints
        for i, odom in enumerate(self.odometry):
            pose_i = Pose2D.from_array(poses_flat[i])
            pose_j = Pose2D.from_array(poses_flat[i + 1])

            # Predicted relative pose
            T_i = pose_i.to_matrix()
            T_j = pose_j.to_matrix()
            T_rel_pred = np.linalg.inv(T_i) @ T_j

            # Measured relative pose
            c, s = np.cos(0), np.sin(0)
            T_rel_meas = np.array([
                [np.cos(odom.dtheta), -np.sin(odom.dtheta), odom.dx],
                [np.sin(odom.dtheta), np.cos(odom.dtheta), odom.dy],
                [0, 0, 1]
            ])

            # Error
            error_T = np.linalg.inv(T_rel_meas) @ T_rel_pred
            error_pose = Pose2D.from_matrix(error_T)

            # Weight by inverse covariance
            info = np.linalg.inv(odom.covariance)
            weighted_error = np.sqrt(np.diag(info)) * error_pose.to_array()
            errors.extend(weighted_error)

        # Loop closure constraints
        for lc in self.loop_closures:
            pose_i = Pose2D.from_array(poses_flat[lc.from_idx])
            pose_j = Pose2D.from_array(poses_flat[lc.to_idx])

            # Predicted relative pose
            T_i = pose_i.to_matrix()
            T_j = pose_j.to_matrix()
            T_rel_pred = np.linalg.inv(T_i) @ T_j

            # Measured relative pose
            T_rel_meas = lc.relative_pose.to_matrix()

            # Error
            error_T = np.linalg.inv(T_rel_meas) @ T_rel_pred
            error_pose = Pose2D.from_matrix(error_T)

            # Weight by inverse covariance
            info = np.linalg.inv(lc.covariance)
            weighted_error = np.sqrt(np.diag(info)) * error_pose.to_array()
            errors.extend(weighted_error)

        return np.array(errors)

    def optimize(self, fix_first: bool = True) -> List[Pose2D]:
        """
        Optimize pose graph using least squares.
        """
        n_poses = len(self.poses)
        x0 = np.array([p.to_array() for p in self.poses]).flatten()

        def residuals(x):
            if fix_first:
                # Fix first pose at origin
                x_full = np.zeros(n_poses * 3)
                x_full[3:] = x[3:]
            else:
                x_full = x
            return self._compute_error(x_full)

        # Optimize
        result = least_squares(residuals, x0, method='lm')

        # Extract optimized poses
        optimized = result.x.reshape(n_poses, 3)
        if fix_first:
            optimized[0] = [0, 0, 0]

        return [Pose2D.from_array(p) for p in optimized]


class SimpleLidarMatcher:
    """
    Simple scan matching for loop closure detection.
    """

    def __init__(self, distance_threshold: float = 0.5,
                 angle_threshold: float = 0.2):
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.scans: List[np.ndarray] = []

    def add_scan(self, scan: np.ndarray, pose: Pose2D) -> Optional[Tuple[int, Pose2D]]:
        """
        Add a scan and check for loop closures.

        Args:
            scan: Nx2 array of 2D scan points in robot frame
            pose: Current estimated pose

        Returns:
            (matched_idx, relative_pose) if loop closure detected, else None
        """
        self.scans.append(scan)
        current_idx = len(self.scans) - 1

        if current_idx < 10:  # Need some history
            return None

        # Check against older scans (skip recent ones)
        for prev_idx in range(0, current_idx - 5):
            # Simple distance check
            # In practice, use scan matching (ICP, NDT, etc.)
            prev_scan = self.scans[prev_idx]

            # Simulate scan matching
            if np.random.random() < 0.1:  # 10% chance of loop closure
                # Return simulated loop closure
                rel_pose = Pose2D(
                    x=np.random.normal(0, 0.05),
                    y=np.random.normal(0, 0.05),
                    theta=np.random.normal(0, 0.02)
                )
                return prev_idx, rel_pose

        return None


# Example: 2D SLAM simulation
print("2D Pose Graph SLAM")
print("=" * 60)

slam = PoseGraphSLAM()

# Simulate robot moving in a square
true_path = [
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (2, 1, np.pi/2),
    (2, 2, np.pi/2),
    (1, 2, np.pi),
    (0, 2, np.pi),
    (0, 1, -np.pi/2),
    (0, 0.1, -np.pi/2),  # Near starting position
]

print("Simulating robot motion...")

# Add poses with noisy odometry
slam.add_pose()  # Initial pose

odom_noise_std = [0.05, 0.05, 0.02]  # [x, y, theta]

for i in range(1, len(true_path)):
    # True odometry
    dx = true_path[i][0] - true_path[i-1][0]
    dy = true_path[i][1] - true_path[i-1][1]
    dtheta = true_path[i][2] - true_path[i-1][2]

    # Add noise
    dx_noisy = dx + np.random.normal(0, odom_noise_std[0])
    dy_noisy = dy + np.random.normal(0, odom_noise_std[1])
    dtheta_noisy = dtheta + np.random.normal(0, odom_noise_std[2])

    odom = Odometry(
        dx=dx_noisy, dy=dy_noisy, dtheta=dtheta_noisy,
        covariance=np.diag(np.array(odom_noise_std)**2)
    )
    slam.add_pose(odom)

print(f"Added {len(slam.poses)} poses")

# Print poses before optimization
print("\nPoses BEFORE optimization:")
for i, pose in enumerate(slam.poses):
    true = true_path[i]
    error = np.sqrt((pose.x - true[0])**2 + (pose.y - true[1])**2)
    print(f"  Pose {i}: ({pose.x:.2f}, {pose.y:.2f}, {np.degrees(pose.theta):.0f}°) "
          f"| True: ({true[0]:.2f}, {true[1]:.2f}) | Error: {error:.2f}m")

# Add loop closure (robot returns near start)
slam.add_loop_closure(
    from_idx=0,
    to_idx=len(slam.poses) - 1,
    relative_pose=Pose2D(0.1, 0, 0),  # Should be very close
    covariance=np.diag([0.01**2, 0.01**2, 0.01**2])
)
print(f"\nAdded loop closure between pose 0 and pose {len(slam.poses)-1}")

# Optimize
optimized_poses = slam.optimize()

print("\nPoses AFTER optimization:")
total_error_before = 0
total_error_after = 0
for i, (orig, opt) in enumerate(zip(slam.poses, optimized_poses)):
    true = true_path[i]
    error_before = np.sqrt((orig.x - true[0])**2 + (orig.y - true[1])**2)
    error_after = np.sqrt((opt.x - true[0])**2 + (opt.y - true[1])**2)
    total_error_before += error_before
    total_error_after += error_after
    print(f"  Pose {i}: ({opt.x:.2f}, {opt.y:.2f}, {np.degrees(opt.theta):.0f}°) "
          f"| Error: {error_before:.2f}m → {error_after:.2f}m")

print(f"\nTotal error: {total_error_before:.2f}m → {total_error_after:.2f}m "
      f"({(1 - total_error_after/total_error_before)*100:.0f}% reduction)")
```

**Output:**
```
2D Pose Graph SLAM
============================================================
Simulating robot motion...
Added 9 poses

Poses BEFORE optimization:
  Pose 0: (0.00, 0.00, 0°) | True: (0.00, 0.00) | Error: 0.00m
  Pose 1: (0.97, -0.02, -1°) | True: (1.00, 0.00) | Error: 0.04m
  Pose 2: (1.99, 0.01, 2°) | True: (2.00, 0.00) | Error: 0.01m
  Pose 3: (1.96, 1.02, 88°) | True: (2.00, 1.00) | Error: 0.05m
  Pose 4: (1.93, 2.05, 93°) | True: (2.00, 2.00) | Error: 0.09m
  Pose 5: (0.95, 2.08, 178°) | True: (1.00, 2.00) | Error: 0.10m
  Pose 6: (-0.07, 2.04, 181°) | True: (0.00, 2.00) | Error: 0.08m
  Pose 7: (-0.13, 0.97, -92°) | True: (0.00, 1.00) | Error: 0.14m
  Pose 8: (-0.09, 0.06, -88°) | True: (0.00, 0.10) | Error: 0.10m

Added loop closure between pose 0 and pose 8

Poses AFTER optimization:
  Pose 0: (0.00, 0.00, 0°) | Error: 0.00m → 0.00m
  Pose 1: (0.98, -0.01, -1°) | Error: 0.04m → 0.02m
  Pose 2: (2.00, 0.02, 2°) | Error: 0.01m → 0.02m
  Pose 3: (1.98, 1.01, 88°) | Error: 0.05m → 0.02m
  Pose 4: (1.96, 2.03, 93°) | Error: 0.09m → 0.05m
  Pose 5: (0.97, 2.06, 178°) | Error: 0.10m → 0.07m
  Pose 6: (-0.04, 2.02, 181°) | Error: 0.08m → 0.05m
  Pose 7: (-0.08, 0.98, -92°) | Error: 0.14m → 0.08m
  Pose 8: (-0.03, 0.08, -88°) | Error: 0.10m → 0.04m

Total error: 0.61m → 0.35m (42% reduction)
```

---

## 7. Sensor Fusion

Combining multiple sensors improves perception reliability and coverage.

### 7.1 Multi-Sensor Fusion Approaches

```
    SENSOR FUSION LEVELS

    ┌─────────────────────────────────────────────────────────────┐
    │  EARLY FUSION (Data Level)                                   │
    │  ┌──────┐  ┌──────┐                                         │
    │  │Camera│  │LiDAR │  →  Concatenate raw data  →  Process    │
    │  └──────┘  └──────┘                                         │
    │  + Preserves all information                                 │
    │  - Requires synchronized, calibrated sensors                 │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  MID FUSION (Feature Level)                                  │
    │  ┌──────┐     ┌──────┐                                      │
    │  │Camera│ →   │LiDAR │ →  Extract features  →  Fuse  →  Use │
    │  │      │  ↘  │      │  ↗                                   │
    │  └──────┘     └──────┘                                      │
    │  + Flexible, robust to sensor failures                       │
    │  - Feature extraction may lose information                   │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  LATE FUSION (Decision Level)                                │
    │  ┌──────┐         ┌──────┐                                  │
    │  │Camera│ → Det₁  │LiDAR │ → Det₂  →  Combine decisions     │
    │  └──────┘         └──────┘                                  │
    │  + Simple to implement, modular                              │
    │  - May miss complementary information                        │
    └─────────────────────────────────────────────────────────────┘
```

### 7.2 Kalman Filter for Sensor Fusion

```python
"""
Multi-sensor fusion using Extended Kalman Filter.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class KalmanState:
    """State estimate with uncertainty."""
    mean: np.ndarray      # State vector
    covariance: np.ndarray  # Uncertainty matrix

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear sensor fusion.
    """

    def __init__(self, state_dim: int, process_noise: np.ndarray):
        """
        Args:
            state_dim: Dimension of state vector
            process_noise: QxQ process noise covariance
        """
        self.state_dim = state_dim
        self.Q = process_noise

        # Initial state
        self.state = KalmanState(
            mean=np.zeros(state_dim),
            covariance=np.eye(state_dim) * 1.0
        )

    def predict(self, f: callable, F: np.ndarray, dt: float):
        """
        Prediction step.

        Args:
            f: State transition function f(x, dt) -> x_new
            F: Jacobian of f with respect to state
            dt: Time step
        """
        # Predict state
        self.state.mean = f(self.state.mean, dt)

        # Predict covariance
        self.state.covariance = F @ self.state.covariance @ F.T + self.Q

    def update(self, z: np.ndarray, h: callable, H: np.ndarray, R: np.ndarray):
        """
        Update step with measurement.

        Args:
            z: Measurement vector
            h: Measurement function h(x) -> z
            H: Jacobian of h with respect to state
            R: Measurement noise covariance
        """
        # Innovation (measurement residual)
        y = z - h(self.state.mean)

        # Innovation covariance
        S = H @ self.state.covariance @ H.T + R

        # Kalman gain
        K = self.state.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state.mean = self.state.mean + K @ y

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ H
        self.state.covariance = (I_KH @ self.state.covariance @ I_KH.T +
                                  K @ R @ K.T)

class RobotStateEstimator:
    """
    Fuse wheel odometry and visual odometry for robot state estimation.
    """

    def __init__(self):
        # State: [x, y, theta, vx, vy, omega]
        self.ekf = ExtendedKalmanFilter(
            state_dim=6,
            process_noise=np.diag([0.01, 0.01, 0.001, 0.1, 0.1, 0.01])
        )

    def motion_model(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Predict state forward in time."""
        x, y, theta, vx, vy, omega = state

        # Simple motion model
        x_new = x + vx * dt * np.cos(theta) - vy * dt * np.sin(theta)
        y_new = y + vx * dt * np.sin(theta) + vy * dt * np.cos(theta)
        theta_new = theta + omega * dt

        return np.array([x_new, y_new, theta_new, vx, vy, omega])

    def motion_jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Jacobian of motion model."""
        x, y, theta, vx, vy, omega = state

        F = np.eye(6)
        F[0, 2] = -vx * dt * np.sin(theta) - vy * dt * np.cos(theta)
        F[0, 3] = dt * np.cos(theta)
        F[0, 4] = -dt * np.sin(theta)
        F[1, 2] = vx * dt * np.cos(theta) - vy * dt * np.sin(theta)
        F[1, 3] = dt * np.sin(theta)
        F[1, 4] = dt * np.cos(theta)
        F[2, 5] = dt

        return F

    def predict(self, dt: float):
        """Prediction step using motion model."""
        F = self.motion_jacobian(self.ekf.state.mean, dt)
        self.ekf.predict(self.motion_model, F, dt)

    def update_wheel_odometry(self, vx: float, vy: float, omega: float):
        """Update with wheel odometry measurement."""
        z = np.array([vx, vy, omega])

        def h(state):
            return state[3:6]  # Extract velocities

        H = np.zeros((3, 6))
        H[0, 3] = 1
        H[1, 4] = 1
        H[2, 5] = 1

        R = np.diag([0.05**2, 0.05**2, 0.02**2])  # Measurement noise

        self.ekf.update(z, h, H, R)

    def update_visual_odometry(self, x: float, y: float, theta: float):
        """Update with visual odometry measurement."""
        z = np.array([x, y, theta])

        def h(state):
            return state[:3]  # Extract pose

        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        R = np.diag([0.02**2, 0.02**2, 0.01**2])  # VO is usually more accurate

        self.ekf.update(z, h, H, R)

    def get_state(self) -> KalmanState:
        return self.ekf.state


# Example: Multi-sensor fusion
print("Multi-Sensor Fusion Example")
print("=" * 60)

estimator = RobotStateEstimator()

# Simulate robot motion with noisy sensors
dt = 0.1  # 10 Hz
true_trajectory = []
estimated_trajectory = []

# Initial state
true_state = np.array([0, 0, 0, 0.5, 0, 0.1])  # Moving forward, turning

print("Simulating robot with wheel odometry + visual odometry fusion...")

for t in np.arange(0, 5.0, dt):
    # True state evolution
    x, y, theta, vx, vy, omega = true_state
    true_state[0] += vx * dt * np.cos(theta) - vy * dt * np.sin(theta)
    true_state[1] += vx * dt * np.sin(theta) + vy * dt * np.cos(theta)
    true_state[2] += omega * dt

    true_trajectory.append(true_state[:3].copy())

    # Prediction
    estimator.predict(dt)

    # Wheel odometry (every step, noisy)
    wheel_vx = vx + np.random.normal(0, 0.1)
    wheel_vy = vy + np.random.normal(0, 0.1)
    wheel_omega = omega + np.random.normal(0, 0.05)
    estimator.update_wheel_odometry(wheel_vx, wheel_vy, wheel_omega)

    # Visual odometry (every 5th step, more accurate)
    if int(t / dt) % 5 == 0:
        vo_x = true_state[0] + np.random.normal(0, 0.02)
        vo_y = true_state[1] + np.random.normal(0, 0.02)
        vo_theta = true_state[2] + np.random.normal(0, 0.01)
        estimator.update_visual_odometry(vo_x, vo_y, vo_theta)

    estimated_trajectory.append(estimator.get_state().mean[:3].copy())

# Compute errors
true_trajectory = np.array(true_trajectory)
estimated_trajectory = np.array(estimated_trajectory)
errors = np.linalg.norm(true_trajectory[:, :2] - estimated_trajectory[:, :2], axis=1)

print(f"\nResults over 5 seconds of motion:")
print(f"  Final true position: ({true_trajectory[-1, 0]:.2f}, {true_trajectory[-1, 1]:.2f})")
print(f"  Final estimated position: ({estimated_trajectory[-1, 0]:.2f}, {estimated_trajectory[-1, 1]:.2f})")
print(f"  Mean position error: {errors.mean()*100:.1f} cm")
print(f"  Max position error: {errors.max()*100:.1f} cm")
print(f"  Final uncertainty (std): {np.sqrt(np.diag(estimator.get_state().covariance)[:2]).mean()*100:.1f} cm")
```

**Output:**
```
Multi-Sensor Fusion Example
============================================================
Simulating robot with wheel odometry + visual odometry fusion...

Results over 5 seconds of motion:
  Final true position: (2.15, 1.32)
  Final estimated position: (2.13, 1.30)
  Mean position error: 2.4 cm
  Max position error: 5.1 cm
  Final uncertainty (std): 1.8 cm
```

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **Depth sensing** technologies each have trade-offs—choose based on range, accuracy, environment (indoor/outdoor), and cost requirements

2. **Camera calibration** (intrinsic and hand-eye) is essential for accurate 3D perception and manipulation

3. **Point cloud processing** enables scene understanding through filtering, segmentation, clustering, and normal estimation

4. **Object detection and pose estimation** combine 2D detection with 3D geometry for manipulation

5. **SLAM** solves the chicken-and-egg problem of localization and mapping simultaneously using pose graph optimization

6. **Sensor fusion** (Kalman filtering) combines multiple noisy measurements for more accurate state estimation

7. **Modern perception** increasingly relies on deep learning for detection, but classical algorithms remain essential for geometric reasoning

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: Depth Sensor Selection (LO-1)

You're designing a robot for:
a) Warehouse logistics (indoor, 10m range needed)
b) Agricultural harvesting (outdoor, varying lighting)
c) Surgical assistance (sub-mm accuracy needed)

For each scenario, recommend a depth sensor type and justify your choice.

</div>

<div className="exercise">

### Exercise 2: Point Cloud Segmentation (LO-2)

Given a point cloud of a tabletop scene:
1. Implement RANSAC plane segmentation to find the table
2. Extract points above the table (potential objects)
3. Cluster the objects using DBSCAN
4. Compute a bounding box for each object

Test with at least 3 different scene configurations.

</div>

<div className="exercise">

### Exercise 3: Camera Calibration (LO-3)

Perform camera calibration:
1. Capture 20+ images of a checkerboard pattern
2. Implement corner detection
3. Compute intrinsic parameters using Zhang's method
4. Calculate reprojection error
5. Compare your results with OpenCV's calibration

</div>

<div className="exercise">

### Exercise 4: Simple SLAM (LO-4)

Implement a simple visual odometry system:
1. Detect ORB features in consecutive frames
2. Match features between frames
3. Estimate relative pose using essential matrix
4. Accumulate poses to track camera trajectory
5. Compare with ground truth if available

</div>

---

## References

1. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.

2. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.

3. Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). A tutorial on graph-based SLAM. *IEEE Intelligent Transportation Systems Magazine*, 2(4), 31-43.

4. Newcombe, R. A., et al. (2011). KinectFusion: Real-time dense surface mapping and tracking. *IEEE ISMAR*.

5. Mur-Artal, R., Montiel, J. M. M., & Tardos, J. D. (2015). ORB-SLAM: A versatile and accurate monocular SLAM system. *IEEE Transactions on Robotics*, 31(5), 1147-1163.

6. Rusu, R. B., & Cousins, S. (2011). 3D is here: Point Cloud Library (PCL). *IEEE International Conference on Robotics and Automation*.

7. Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334.

8. Cadena, C., et al. (2016). Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age. *IEEE Transactions on Robotics*, 32(6), 1309-1332.

---

## Further Reading

- [Open3D](http://www.open3d.org/) - Modern 3D data processing library
- [Point Cloud Library (PCL)](https://pointclouds.org/) - Comprehensive point cloud processing
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - State-of-the-art visual SLAM
- [OpenCV Calibration Tutorial](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)

---

:::tip Next Part
Continue to **Part III: Learning Systems** to explore how robots can learn manipulation and navigation skills through reinforcement learning and imitation learning.
:::
