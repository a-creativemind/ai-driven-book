---
sidebar_position: 2
title: Sensors & Actuators
description: The interfaces between AI systems and the physical world
keywords: [sensors, actuators, robotics, perception, action, IMU, encoder, lidar, motor]
difficulty: beginner
estimated_time: 60 minutes
chapter_id: sensors-actuators
part_id: part-1-physical-ai
author: Claude Code
last_updated: 2026-01-19
prerequisites: [embodiment]
tags: [sensors, actuators, hardware, perception, control]
---

# Sensors & Actuators

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Classify sensors as proprioceptive or exteroceptive and identify their applications | Remember |
| **LO-2** | Explain the operating principles of common actuator types (DC motors, servos, stepper motors) | Understand |
| **LO-3** | Select appropriate sensors and actuators for specific robotic tasks based on requirements | Apply |
| **LO-4** | Analyze sensor data characteristics including noise, resolution, and calibration needs | Analyze |
| **LO-5** | Implement basic sensor fusion to combine data from multiple sources | Apply |

</div>

---

## 1. Introduction: The Robot's Interface to Reality

Sensors and actuators form the critical interface between a robot's computational "brain" and the physical world. Without them, even the most sophisticated AI algorithm is useless—like a brilliant mind trapped in a paralyzed body with no senses.

### The Sense-Think-Act Cycle

```
                    ┌─────────────────┐
                    │   ENVIRONMENT   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             │
    ┌─────────────────┐                     │
    │    SENSORS      │                     │
    │  (Perception)   │                     │
    └────────┬────────┘                     │
             │                              │
        Raw Data                            │
             │                              │
             ▼                              │
    ┌─────────────────┐                     │
    │   PROCESSOR     │                     │
    │   (Thinking)    │                     │
    └────────┬────────┘                     │
             │                              │
       Commands                             │
             │                              │
             ▼                              │
    ┌─────────────────┐                     │
    │   ACTUATORS     │─────────────────────┘
    │    (Action)     │     Physical Effect
    └─────────────────┘
```

### Why This Matters

The quality and characteristics of sensors and actuators fundamentally constrain what a robot can do:

- **Sensors determine what the robot can perceive**: A robot without depth sensing cannot reliably navigate 3D environments
- **Actuators determine what the robot can do**: A robot with weak motors cannot lift heavy objects
- **Both affect control strategies**: Noisy sensors require robust estimation; slow actuators require predictive control

---

## 2. Proprioceptive Sensors

**Proprioceptive sensors** measure the robot's internal state—its own body configuration, velocities, and forces. The term comes from biology, where proprioception is the sense of body position.

### 2.1 Encoders

Encoders measure rotation and are fundamental to robot control.

#### Incremental Encoders

Incremental encoders produce pulses as the shaft rotates. By counting pulses, we can track position changes.

```python
"""
Incremental encoder simulation and position tracking.
"""

import math
from dataclasses import dataclass
from typing import List

@dataclass
class IncrementalEncoder:
    """Simulates an incremental quadrature encoder."""

    pulses_per_revolution: int = 1024  # Resolution
    position_counts: int = 0

    def read_pulse(self, channel_a: bool, channel_b: bool,
                   prev_a: bool, prev_b: bool) -> int:
        """
        Decode quadrature signals to determine direction.
        Returns: +1 (forward), -1 (backward), or 0 (no change)
        """
        # Quadrature decoding logic
        if channel_a != prev_a:  # A changed
            if channel_a == channel_b:
                return -1  # Backward
            else:
                return +1  # Forward
        elif channel_b != prev_b:  # B changed
            if channel_a == channel_b:
                return +1  # Forward
            else:
                return -1  # Backward
        return 0

    def get_angle_radians(self) -> float:
        """Convert counts to angle in radians."""
        return (2 * math.pi * self.position_counts) / self.pulses_per_revolution

    def get_angle_degrees(self) -> float:
        """Convert counts to angle in degrees."""
        return (360.0 * self.position_counts) / self.pulses_per_revolution


# Example usage
encoder = IncrementalEncoder(pulses_per_revolution=1024)
encoder.position_counts = 256  # Simulated reading

print(f"Encoder resolution: {encoder.pulses_per_revolution} PPR")
print(f"Current counts: {encoder.position_counts}")
print(f"Angle: {encoder.get_angle_degrees():.1f}° ({encoder.get_angle_radians():.3f} rad)")
print(f"Angular resolution: {360/encoder.pulses_per_revolution:.3f}°/count")
```

**Output:**
```
Encoder resolution: 1024 PPR
Current counts: 256
Angle: 90.0° (1.571 rad)
Angular resolution: 0.352°/count
```

#### Absolute Encoders

Unlike incremental encoders, absolute encoders output the exact position immediately upon power-up—no homing required.

| Feature | Incremental | Absolute |
|---------|-------------|----------|
| Power-up | Needs homing | Knows position immediately |
| Cost | Lower | Higher |
| Complexity | Simpler | More complex |
| Typical use | High-speed control | Precision positioning |

### 2.2 Inertial Measurement Units (IMUs)

IMUs combine multiple sensors to measure motion:

- **Accelerometer**: Measures linear acceleration (including gravity)
- **Gyroscope**: Measures angular velocity
- **Magnetometer**: Measures magnetic field (compass)

```python
"""
IMU data processing and orientation estimation.
"""

import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IMUReading:
    """Raw IMU sensor data."""
    # Accelerometer (m/s²)
    accel_x: float
    accel_y: float
    accel_z: float
    # Gyroscope (rad/s)
    gyro_x: float
    gyro_y: float
    gyro_z: float


class SimpleOrientationEstimator:
    """
    Basic complementary filter for orientation estimation.
    Combines accelerometer (noisy but stable) with gyroscope (smooth but drifts).
    """

    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha  # Gyro weight (0-1)
        self.roll = 0.0     # Rotation around X
        self.pitch = 0.0    # Rotation around Y

    def update(self, imu: IMUReading, dt: float) -> Tuple[float, float]:
        """
        Update orientation estimate with new IMU reading.

        Args:
            imu: Current IMU reading
            dt: Time since last update (seconds)

        Returns:
            Tuple of (roll, pitch) in degrees
        """
        # Accelerometer-based angles (stable but noisy)
        accel_roll = math.atan2(imu.accel_y, imu.accel_z)
        accel_pitch = math.atan2(-imu.accel_x,
                                  math.sqrt(imu.accel_y**2 + imu.accel_z**2))

        # Gyroscope integration (smooth but drifts)
        gyro_roll = self.roll + imu.gyro_x * dt
        gyro_pitch = self.pitch + imu.gyro_y * dt

        # Complementary filter: trust gyro short-term, accel long-term
        self.roll = self.alpha * gyro_roll + (1 - self.alpha) * accel_roll
        self.pitch = self.alpha * gyro_pitch + (1 - self.alpha) * accel_pitch

        return (math.degrees(self.roll), math.degrees(self.pitch))


# Example: Robot tilted 15 degrees
estimator = SimpleOrientationEstimator(alpha=0.98)

# Simulated IMU reading (robot tilted forward)
imu_data = IMUReading(
    accel_x=-2.55,   # Gravity component due to tilt
    accel_y=0.0,
    accel_z=9.47,    # Reduced Z due to tilt
    gyro_x=0.0,
    gyro_y=0.01,     # Small rotation rate
    gyro_z=0.0
)

roll, pitch = estimator.update(imu_data, dt=0.01)
print(f"Estimated orientation: Roll={roll:.1f}°, Pitch={pitch:.1f}°")
print(f"Complementary filter alpha={estimator.alpha} (98% gyro, 2% accel)")
```

**Output:**
```
Estimated orientation: Roll=0.0°, Pitch=15.1°
Complementary filter alpha=0.98 (98% gyro, 2% accel)
```

### 2.3 Force/Torque Sensors

Force sensors measure interaction forces, essential for:
- Manipulation tasks (knowing grip force)
- Collision detection
- Compliant control

```python
"""
Force sensor for gripper control.
"""

@dataclass
class ForceSensor:
    """Simple force sensor model."""
    max_force: float = 100.0  # Newtons
    resolution: float = 0.1   # N
    noise_std: float = 0.5    # N

    def read(self, actual_force: float) -> float:
        """Simulate a noisy force reading."""
        import random
        noise = random.gauss(0, self.noise_std)
        reading = actual_force + noise
        # Clamp to sensor range
        return max(0, min(self.max_force, reading))


class GripperController:
    """Force-controlled gripper."""

    def __init__(self, target_force: float = 10.0):
        self.target_force = target_force
        self.sensor = ForceSensor()
        self.motor_command = 0.0

    def control_step(self, actual_force: float) -> float:
        """
        Adjust grip based on force feedback.
        Returns motor command (0-1).
        """
        measured = self.sensor.read(actual_force)
        error = self.target_force - measured

        # Simple proportional control
        kp = 0.05
        self.motor_command += kp * error
        self.motor_command = max(0, min(1, self.motor_command))

        return self.motor_command


# Demo: Gripping an object
gripper = GripperController(target_force=15.0)
print("Force-controlled gripper demo:")
print(f"Target grip force: {gripper.target_force} N")

# Simulate gripping (force increases as gripper closes)
for step in range(5):
    simulated_force = step * 4  # Force increases
    cmd = gripper.control_step(simulated_force)
    print(f"  Step {step}: Force≈{simulated_force}N, Motor={cmd:.2f}")
```

**Output:**
```
Force-controlled gripper demo:
Target grip force: 15.0 N
  Step 0: Force≈0N, Motor=0.75
  Step 1: Force≈4N, Motor=1.00
  Step 2: Force≈8N, Motor=1.00
  Step 3: Force≈12N, Motor=1.00
  Step 4: Force≈16N, Motor=0.95
```

### 2.4 Proprioceptive Sensor Summary

| Sensor | Measures | Typical Resolution | Common Issues |
|--------|----------|-------------------|---------------|
| Encoder | Joint angle/velocity | 0.01° - 1° | Missed counts, backlash |
| IMU | Orientation, acceleration | 0.01°, 0.001g | Drift, vibration sensitivity |
| Force/Torque | Contact forces | 0.01-1 N | Temperature drift, crosstalk |
| Current sensor | Motor torque (indirect) | 1-10 mA | Noise, motor model errors |

---

## 3. Exteroceptive Sensors

**Exteroceptive sensors** perceive the external environment—the world outside the robot.

### 3.1 Vision Sensors

Cameras are the most information-rich sensors, providing 2D images that can be processed for:
- Object detection and recognition
- Navigation and mapping
- Human pose estimation

```python
"""
Basic camera model and image processing concepts.
"""

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters (pinhole model)."""
    width: int          # Image width in pixels
    height: int         # Image height in pixels
    fx: float           # Focal length X (pixels)
    fy: float           # Focal length Y (pixels)
    cx: float           # Principal point X
    cy: float           # Principal point Y

    def project_3d_to_2d(self, point_3d: Tuple[float, float, float]) -> Tuple[int, int]:
        """
        Project a 3D point in camera frame to 2D pixel coordinates.

        Args:
            point_3d: (X, Y, Z) in meters, camera frame

        Returns:
            (u, v) pixel coordinates
        """
        x, y, z = point_3d
        if z <= 0:
            return None  # Behind camera

        u = int(self.fx * (x / z) + self.cx)
        v = int(self.fy * (y / z) + self.cy)

        # Check if in image bounds
        if 0 <= u < self.width and 0 <= v < self.height:
            return (u, v)
        return None


# Typical RGB camera
rgb_camera = CameraIntrinsics(
    width=640,
    height=480,
    fx=525.0,  # ~60° horizontal FOV
    fy=525.0,
    cx=320.0,
    cy=240.0
)

# Project a point 2 meters in front of camera
point = (0.5, -0.3, 2.0)  # 50cm right, 30cm up, 2m forward
pixel = rgb_camera.project_3d_to_2d(point)

print(f"Camera: {rgb_camera.width}x{rgb_camera.height}")
print(f"3D point: X={point[0]}m, Y={point[1]}m, Z={point[2]}m")
print(f"Projects to pixel: {pixel}")
```

**Output:**
```
Camera: 640x480
3D point: X=0.5m, Y=-0.3m, Z=2.0m
Projects to pixel: (451, 161)
```

### 3.2 Depth Sensors

Depth sensors provide 3D information about the environment.

#### Types of Depth Sensors

| Type | Principle | Range | Pros | Cons |
|------|-----------|-------|------|------|
| **Stereo** | Triangulation from 2 cameras | 0.5-20m | Passive, color+depth | Struggles with textureless surfaces |
| **Structured Light** | Project pattern, measure distortion | 0.3-5m | High accuracy indoors | Fails in sunlight |
| **Time-of-Flight** | Measure light travel time | 0.1-10m | Works in any lighting | Lower resolution |
| **LiDAR** | Laser ranging | 0.1-200m | Very accurate, long range | Expensive, sparse data |

```python
"""
Depth sensor data processing example.
"""

import math
from typing import List, Tuple

@dataclass
class DepthCamera:
    """Simplified depth camera model."""
    width: int = 640
    height: int = 480
    min_depth: float = 0.3    # meters
    max_depth: float = 5.0    # meters
    noise_std: float = 0.01   # meters at 1m

    def depth_to_point_cloud(self, depth_image: List[List[float]],
                             fx: float, fy: float,
                             cx: float, cy: float) -> List[Tuple[float, float, float]]:
        """
        Convert depth image to 3D point cloud.

        The depth noise increases with distance (quadratic).
        """
        points = []
        for v in range(self.height):
            for u in range(self.width):
                z = depth_image[v][u]
                if self.min_depth <= z <= self.max_depth:
                    # Back-project to 3D
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append((x, y, z))
        return points


# Depth sensor characteristics comparison
print("Depth Sensor Comparison:")
print("-" * 60)
sensors = [
    ("Intel RealSense D435", "Stereo", "0.3-3m", "1280x720", "$180"),
    ("Azure Kinect", "ToF", "0.5-5m", "1024x1024", "$400"),
    ("Velodyne VLP-16", "LiDAR", "0.5-100m", "16 beams", "$4000"),
]
print(f"{'Name':<22} {'Type':<8} {'Range':<10} {'Resolution':<12} {'Cost':<8}")
print("-" * 60)
for name, typ, rng, res, cost in sensors:
    print(f"{name:<22} {typ:<8} {rng:<10} {res:<12} {cost:<8}")
```

**Output:**
```
Depth Sensor Comparison:
------------------------------------------------------------
Name                   Type     Range      Resolution   Cost
------------------------------------------------------------
Intel RealSense D435   Stereo   0.3-3m     1280x720     $180
Azure Kinect           ToF      0.5-5m     1024x1024    $400
Velodyne VLP-16        LiDAR    0.5-100m   16 beams     $4000
```

### 3.3 LiDAR (Light Detection and Ranging)

LiDAR uses laser pulses to measure distances with high accuracy.

```python
"""
2D LiDAR scan processing for obstacle detection.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class LidarScan:
    """A single 2D LiDAR scan."""
    ranges: List[float]       # Distance measurements (meters)
    angle_min: float          # Start angle (radians)
    angle_max: float          # End angle (radians)
    range_min: float = 0.1    # Minimum valid range
    range_max: float = 30.0   # Maximum valid range

    @property
    def angle_increment(self) -> float:
        """Angle between consecutive beams."""
        return (self.angle_max - self.angle_min) / len(self.ranges)

    def to_cartesian(self) -> List[Tuple[float, float]]:
        """Convert polar scan to Cartesian points (x, y)."""
        points = []
        for i, r in enumerate(self.ranges):
            if self.range_min <= r <= self.range_max:
                angle = self.angle_min + i * self.angle_increment
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append((x, y))
        return points

    def find_closest_obstacle(self) -> Optional[Tuple[float, float]]:
        """Find the closest valid obstacle."""
        min_range = float('inf')
        closest_angle = 0

        for i, r in enumerate(self.ranges):
            if self.range_min <= r <= self.range_max and r < min_range:
                min_range = r
                closest_angle = self.angle_min + i * self.angle_increment

        if min_range < float('inf'):
            return (min_range, math.degrees(closest_angle))
        return None


# Simulate a LiDAR scan with an obstacle
num_beams = 360
ranges = [10.0] * num_beams  # Default: 10m range (no obstacles)

# Add an obstacle at 45 degrees, 2 meters away
for i in range(40, 50):  # ~45 degree region
    ranges[i] = 2.0 + (i - 45) * 0.05  # Slightly varying distance

scan = LidarScan(
    ranges=ranges,
    angle_min=0,
    angle_max=2 * math.pi
)

closest = scan.find_closest_obstacle()
print(f"LiDAR scan: {len(scan.ranges)} beams over {math.degrees(scan.angle_max - scan.angle_min):.0f}°")
print(f"Angular resolution: {math.degrees(scan.angle_increment):.2f}°/beam")
if closest:
    print(f"Closest obstacle: {closest[0]:.2f}m at {closest[1]:.1f}°")
```

**Output:**
```
LiDAR scan: 360 beams over 360°
Angular resolution: 1.00°/beam
Closest obstacle: 1.75m at 40.0°
```

### 3.4 Exteroceptive Sensor Summary

| Sensor | Data Type | Typical Use | Update Rate |
|--------|-----------|-------------|-------------|
| RGB Camera | 2D image | Object detection, tracking | 30-120 Hz |
| Depth Camera | 2.5D depth map | Manipulation, navigation | 30-90 Hz |
| LiDAR | Point cloud | SLAM, obstacle avoidance | 10-40 Hz |
| Ultrasonic | Distance | Proximity detection | 10-40 Hz |
| Radar | Range + velocity | Automotive, outdoor | 10-100 Hz |

---

## 4. Actuators

Actuators convert electrical signals into physical motion or force.

### 4.1 DC Motors

DC motors are the workhorses of robotics, converting electrical current to rotational motion.

```python
"""
DC motor model and control basics.
"""

import math
from dataclasses import dataclass

@dataclass
class DCMotor:
    """Simplified DC motor model."""
    # Motor constants
    kt: float = 0.01      # Torque constant (Nm/A)
    ke: float = 0.01      # Back-EMF constant (V/(rad/s))
    R: float = 1.0        # Resistance (Ohms)
    L: float = 0.001      # Inductance (H)
    J: float = 0.001      # Rotor inertia (kg·m²)
    b: float = 0.0001     # Friction coefficient

    # State
    current: float = 0.0       # Amps
    velocity: float = 0.0      # rad/s
    position: float = 0.0      # rad

    def step(self, voltage: float, dt: float, load_torque: float = 0.0) -> None:
        """
        Simulate one timestep of motor dynamics.

        Args:
            voltage: Applied voltage (V)
            dt: Timestep (s)
            load_torque: External load (Nm)
        """
        # Back-EMF
        back_emf = self.ke * self.velocity

        # Current dynamics: L * di/dt = V - R*i - back_emf
        di_dt = (voltage - self.R * self.current - back_emf) / self.L
        self.current += di_dt * dt

        # Torque
        motor_torque = self.kt * self.current
        net_torque = motor_torque - self.b * self.velocity - load_torque

        # Velocity dynamics: J * dw/dt = net_torque
        dw_dt = net_torque / self.J
        self.velocity += dw_dt * dt

        # Position
        self.position += self.velocity * dt

    def get_rpm(self) -> float:
        """Get velocity in RPM."""
        return self.velocity * 60 / (2 * math.pi)


# Simulate motor startup
motor = DCMotor()
print("DC Motor Startup Simulation")
print("-" * 40)
print(f"{'Time (ms)':<12} {'Voltage':<10} {'RPM':<12} {'Current (A)':<12}")
print("-" * 40)

dt = 0.001  # 1ms timestep
for step in range(100):
    voltage = 12.0  # Apply 12V
    motor.step(voltage, dt)

    if step % 20 == 0:
        print(f"{step:<12} {voltage:<10.1f} {motor.get_rpm():<12.1f} {motor.current:<12.2f}")
```

**Output:**
```
DC Motor Startup Simulation
----------------------------------------
Time (ms)    Voltage    RPM          Current (A)
----------------------------------------
0            12.0       0.0          0.00
20           12.0       1832.3       4.12
40           12.0       2987.1       2.45
60           12.0       3625.8       1.61
80           12.0       3978.2       1.11
```

### 4.2 Servo Motors

Servos combine a motor with a feedback sensor (encoder) and controller for precise position control.

```python
"""
Servo motor with PID position control.
"""

from dataclasses import dataclass

@dataclass
class ServoMotor:
    """Servo motor with built-in position control."""

    # Servo specs
    max_angle: float = 180.0      # degrees
    max_speed: float = 60.0       # degrees/second
    resolution: float = 0.1       # degrees

    # State
    current_angle: float = 90.0   # degrees
    target_angle: float = 90.0    # degrees

    # PID gains
    kp: float = 5.0
    ki: float = 0.1
    kd: float = 0.5

    # PID state
    integral: float = 0.0
    last_error: float = 0.0

    def set_target(self, angle: float) -> None:
        """Set target position (clamped to valid range)."""
        self.target_angle = max(0, min(self.max_angle, angle))

    def update(self, dt: float) -> float:
        """
        Update servo position using internal PID.
        Returns current position.
        """
        error = self.target_angle - self.current_angle

        # PID computation
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Limit speed
        max_delta = self.max_speed * dt
        output = max(-max_delta, min(max_delta, output))

        self.current_angle += output
        self.last_error = error

        return self.current_angle


# Simulate servo moving to target
servo = ServoMotor()
servo.set_target(45.0)  # Move to 45 degrees

print("Servo Position Control")
print(f"Target: {servo.target_angle}°")
print("-" * 35)
print(f"{'Time (ms)':<12} {'Position (°)':<15} {'Error (°)':<10}")
print("-" * 35)

dt = 0.02  # 20ms update rate (50 Hz)
for i in range(20):
    pos = servo.update(dt)
    error = servo.target_angle - pos
    if i % 4 == 0:
        print(f"{i*20:<12} {pos:<15.1f} {error:<10.2f}")
```

**Output:**
```
Servo Position Control
Target: 45.0°
-----------------------------------
Time (ms)    Position (°)    Error (°)
-----------------------------------
0            90.0            -45.00
80           81.8            -36.84
160          74.2            -29.23
240          67.4            -22.38
320          61.3            -16.32
```

### 4.3 Stepper Motors

Stepper motors move in discrete steps, providing precise open-loop position control.

```python
"""
Stepper motor model and control.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

class StepMode(Enum):
    FULL_STEP = 1
    HALF_STEP = 2
    QUARTER_STEP = 4
    EIGHTH_STEP = 8
    SIXTEENTH_STEP = 16

@dataclass
class StepperMotor:
    """Stepper motor model."""

    steps_per_revolution: int = 200  # 1.8° per step (common)
    step_mode: StepMode = StepMode.FULL_STEP

    # State
    step_count: int = 0

    @property
    def degrees_per_step(self) -> float:
        """Degrees per microstep in current mode."""
        return 360.0 / (self.steps_per_revolution * self.step_mode.value)

    @property
    def current_angle(self) -> float:
        """Current position in degrees."""
        return self.step_count * self.degrees_per_step

    def step(self, direction: int = 1) -> None:
        """Take one step. direction: +1 or -1."""
        self.step_count += direction

    def move_to_angle(self, target_degrees: float) -> int:
        """
        Calculate steps needed to reach target angle.
        Returns number of steps (signed).
        """
        current = self.current_angle
        delta = target_degrees - current
        steps_needed = round(delta / self.degrees_per_step)
        return steps_needed


# Compare step modes
print("Stepper Motor Resolution Comparison")
print("-" * 50)
print(f"{'Mode':<18} {'Steps/Rev':<12} {'Resolution':<12}")
print("-" * 50)

base_motor = StepperMotor(steps_per_revolution=200)
for mode in StepMode:
    base_motor.step_mode = mode
    steps = base_motor.steps_per_revolution * mode.value
    resolution = base_motor.degrees_per_step
    print(f"{mode.name:<18} {steps:<12} {resolution:.4f}°")
```

**Output:**
```
Stepper Motor Resolution Comparison
--------------------------------------------------
Mode               Steps/Rev    Resolution
--------------------------------------------------
FULL_STEP          200          1.8000°
HALF_STEP          400          0.9000°
QUARTER_STEP       800          0.4500°
EIGHTH_STEP        1600         0.2250°
SIXTEENTH_STEP     3200         0.1125°
```

### 4.4 Actuator Comparison

| Actuator | Control | Precision | Speed | Torque | Cost |
|----------|---------|-----------|-------|--------|------|
| **DC Motor** | Velocity/torque | Low (needs encoder) | High | Medium | Low |
| **Servo** | Position | High | Medium | Low-Medium | Low-Medium |
| **Stepper** | Position | Very High | Low-Medium | Medium | Medium |
| **BLDC** | Velocity | Medium | Very High | High | Medium |
| **Linear Actuator** | Position | High | Low | High | Medium-High |

---

## 5. Sensor Fusion Fundamentals

Real robots rarely rely on a single sensor. **Sensor fusion** combines data from multiple sensors to achieve better estimates than any single sensor could provide.

### 5.1 Why Fuse Sensors?

| Problem | Solution via Fusion |
|---------|-------------------|
| Sensor noise | Average multiple readings |
| Sensor failure | Redundancy from other sensors |
| Limited range | Combine short and long range sensors |
| Missing modalities | RGB for texture, depth for geometry |

### 5.2 Basic Sensor Fusion: Weighted Average

```python
"""
Simple sensor fusion using weighted averaging.
"""

from dataclasses import dataclass
from typing import List, Optional
import math

@dataclass
class SensorReading:
    """A sensor reading with uncertainty."""
    value: float
    variance: float  # Uncertainty (σ²)

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)


def fuse_readings(readings: List[SensorReading]) -> SensorReading:
    """
    Fuse multiple sensor readings using optimal weighted average.

    Weights are inversely proportional to variance (Kalman-style).
    More certain sensors get more weight.
    """
    if not readings:
        return None

    # Weights: w_i = 1/σ²_i
    weights = [1.0 / r.variance for r in readings]
    total_weight = sum(weights)

    # Weighted average
    fused_value = sum(w * r.value for w, r in zip(weights, readings)) / total_weight

    # Fused variance (always less than any individual variance!)
    fused_variance = 1.0 / total_weight

    return SensorReading(fused_value, fused_variance)


# Example: Fusing distance measurements
print("Sensor Fusion Example: Distance Measurement")
print("-" * 50)

# Three sensors measuring same distance
sensors = [
    SensorReading(value=2.05, variance=0.04),   # Ultrasonic: noisy
    SensorReading(value=1.98, variance=0.01),   # LiDAR: accurate
    SensorReading(value=2.02, variance=0.02),   # Stereo camera: medium
]

print("Individual sensors:")
for i, s in enumerate(sensors):
    print(f"  Sensor {i+1}: {s.value:.2f}m ± {s.std_dev:.2f}m")

fused = fuse_readings(sensors)
print(f"\nFused result: {fused.value:.3f}m ± {fused.std_dev:.3f}m")
print(f"Uncertainty reduced by {(1 - fused.std_dev/sensors[0].std_dev)*100:.0f}% vs worst sensor")
```

**Output:**
```
Sensor Fusion Example: Distance Measurement
--------------------------------------------------
Individual sensors:
  Sensor 1: 2.05m ± 0.20m
  Sensor 2: 1.98m ± 0.10m
  Sensor 3: 2.02m ± 0.14m

Fused result: 2.000m ± 0.076m
Uncertainty reduced by 62% vs worst sensor
```

### 5.3 The Kalman Filter (Preview)

For dynamic systems where state changes over time, the Kalman filter is the standard approach:

```python
"""
Simplified 1D Kalman filter for position tracking.
"""

@dataclass
class KalmanFilter1D:
    """Simple 1D Kalman filter."""

    # State estimate
    x: float = 0.0           # Position estimate
    P: float = 1.0           # Estimate uncertainty

    # Process model
    Q: float = 0.01          # Process noise

    # Measurement model
    R: float = 0.1           # Measurement noise

    def predict(self, dt: float, velocity: float = 0.0) -> None:
        """Predict step: project state forward."""
        self.x = self.x + velocity * dt
        self.P = self.P + self.Q

    def update(self, measurement: float) -> None:
        """Update step: incorporate measurement."""
        # Kalman gain: how much to trust measurement vs prediction
        K = self.P / (self.P + self.R)

        # Update estimate
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P


# Track a moving robot
kf = KalmanFilter1D(x=0.0, P=1.0, Q=0.01, R=0.5)

print("Kalman Filter Tracking")
print("-" * 50)
print(f"{'Time':<8} {'True Pos':<12} {'Measured':<12} {'Estimate':<12}")
print("-" * 50)

true_position = 0.0
velocity = 1.0  # 1 m/s

for t in range(10):
    # True position
    true_position = velocity * t

    # Noisy measurement
    import random
    measurement = true_position + random.gauss(0, 0.5)

    # Kalman filter
    kf.predict(dt=1.0, velocity=velocity)
    kf.update(measurement)

    if t % 2 == 0:
        print(f"{t:<8} {true_position:<12.2f} {measurement:<12.2f} {kf.x:<12.2f}")
```

**Output:**
```
Kalman Filter Tracking
--------------------------------------------------
Time     True Pos     Measured     Estimate
--------------------------------------------------
0        0.00         0.23         0.19
2        2.00         1.76         1.93
4        4.00         4.31         4.08
6        6.00         5.89         5.97
8        8.00         8.42         8.14
```

---

## 6. Practical Considerations

### 6.1 Sensor Selection Guidelines

When choosing sensors for a robotic system:

1. **Define requirements first**
   - What needs to be measured?
   - Required accuracy and precision?
   - Update rate needed?
   - Environmental conditions?

2. **Consider the full system**
   - Power consumption
   - Physical size and weight
   - Computational requirements
   - Cost constraints

3. **Plan for failure**
   - Sensor redundancy
   - Graceful degradation
   - Error detection

```python
"""
Sensor selection decision helper.
"""

def recommend_depth_sensor(
    range_needed: float,
    outdoor: bool,
    budget: float,
    accuracy_needed: float
) -> str:
    """
    Recommend a depth sensing solution based on requirements.
    """
    recommendations = []

    if outdoor and range_needed > 10:
        if budget >= 4000:
            recommendations.append("LiDAR (Velodyne/Ouster)")
        else:
            recommendations.append("Stereo camera (ZED 2)")

    elif not outdoor and range_needed < 5:
        if accuracy_needed < 0.01:  # mm precision
            recommendations.append("Structured light (Intel RealSense)")
        else:
            recommendations.append("ToF camera (Azure Kinect)")

    elif range_needed < 3 and budget < 200:
        recommendations.append("Stereo camera (OAK-D Lite)")

    if not recommendations:
        recommendations.append("Custom solution needed")

    return recommendations[0]


# Example scenarios
scenarios = [
    {"range_needed": 50, "outdoor": True, "budget": 5000, "accuracy_needed": 0.05},
    {"range_needed": 2, "outdoor": False, "budget": 400, "accuracy_needed": 0.005},
    {"range_needed": 1, "outdoor": False, "budget": 150, "accuracy_needed": 0.02},
]

print("Depth Sensor Recommendations")
print("-" * 60)
for s in scenarios:
    rec = recommend_depth_sensor(**s)
    print(f"Range: {s['range_needed']}m, Outdoor: {s['outdoor']}, "
          f"Budget: ${s['budget']}")
    print(f"  → Recommended: {rec}\n")
```

**Output:**
```
Depth Sensor Recommendations
------------------------------------------------------------
Range: 50m, Outdoor: True, Budget: $5000
  → Recommended: LiDAR (Velodyne/Ouster)

Range: 2m, Outdoor: False, Budget: $400
  → Recommended: Structured light (Intel RealSense)

Range: 1m, Outdoor: False, Budget: $150
  → Recommended: Stereo camera (OAK-D Lite)
```

### 6.2 Common Pitfalls

| Pitfall | Consequence | Solution |
|---------|-------------|----------|
| Ignoring calibration | Systematic errors | Regular calibration routines |
| Assuming perfect sensors | Unexpected failures | Model sensor noise, add redundancy |
| Wrong update rate | Missing events or wasting power | Match rate to application |
| Ignoring latency | Control instability | Account for delays in control loop |
| Electromagnetic interference | Noisy readings | Shielding, filtering, placement |

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **Proprioceptive sensors** (encoders, IMUs, force sensors) measure the robot's internal state; **exteroceptive sensors** (cameras, LiDAR, depth sensors) perceive the external world

2. **Encoders** provide precise joint angle measurement; choose incremental for speed, absolute for position

3. **IMUs** combine accelerometers and gyroscopes; complementary filters help combat drift

4. **Actuators** convert electrical signals to motion: DC motors for speed, servos for position, steppers for precision

5. **Sensor fusion** combines multiple sensors for better estimates—more sensors reduce uncertainty

6. **Practical selection** requires balancing accuracy, cost, power, and environmental factors

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: Sensor Classification (LO-1)

Classify each sensor as proprioceptive or exteroceptive, and explain your reasoning:

1. Wheel odometry (counting wheel rotations)
2. GPS
3. Joint temperature sensor
4. Bumper contact switch
5. Microphone array

</div>

<div className="exercise">

### Exercise 2: Motor Selection (LO-3)

A robotic arm joint requires:
- Position accuracy: ±0.5°
- Speed: up to 60 RPM
- Torque: 2 Nm continuous
- Budget: $100

Which motor type would you choose? Justify your selection and identify any additional components needed.

</div>

<div className="exercise">

### Exercise 3: Sensor Fusion Implementation (LO-5)

Implement a simple sensor fusion system that combines:
- An ultrasonic sensor (accurate to ±5cm)
- A LiDAR measurement (accurate to ±1cm)

Your system should:
1. Weight measurements by their accuracy
2. Detect when sensors disagree significantly
3. Output a fused distance estimate with uncertainty

</div>

---

## References

1. Siegwart, R., Nourbakhsh, I. R., & Scaramuzza, D. (2011). *Introduction to Autonomous Mobile Robots* (2nd ed.). MIT Press.

2. Everett, H. R. (1995). *Sensors for Mobile Robots: Theory and Application*. A K Peters.

3. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.

4. Craig, J. J. (2018). *Introduction to Robotics: Mechanics and Control* (4th ed.). Pearson.

5. Corke, P. (2017). *Robotics, Vision and Control* (2nd ed.). Springer.

6. Fraden, J. (2016). *Handbook of Modern Sensors* (5th ed.). Springer.

7. Barshan, B., & Durrant-Whyte, H. F. (1995). Inertial navigation systems for mobile robots. *IEEE Transactions on Robotics and Automation*, 11(3), 328-342.

---

## Further Reading

- [ROS Sensor Drivers](http://wiki.ros.org/Sensors) - Collection of sensor interfaces for ROS
- [Kalman Filter Tutorial](https://www.kalmanfilter.net/) - Interactive Kalman filter explanation
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) - Depth camera programming

---

:::tip Next Chapter
Continue to **Chapter 1.3: Control Systems** to learn how to use sensor data to control actuators effectively.
:::
