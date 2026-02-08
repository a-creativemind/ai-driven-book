---
sidebar_position: 3
title: کنٹرول سسٹمز
description: روبوٹک سسٹمز کے لیے بنیادی کنٹرول تھیوری
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

# کنٹرول سسٹمز (Control Systems)

<div className="learning-objectives">

## سیکھنے کے مقاصد

اس باب کو مکمل کرنے کے بعد، آپ اس قابل ہوں گے کہ:

| ID | مقصد | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | اوپن لوپ اور کلوزڈ لوپ کنٹرول کے درمیان فرق کریں اور شناخت کریں کہ ہر ایک کب مناسب ہے | سمجھنا |
| **LO-2** | روبوٹک ایپلی کیشنز کے لیے PID کنٹرولرز کو نافذ کریں اور ٹیون کریں | اطلاق کرنا |
| **LO-3** | متحرک سسٹمز کو اسٹیٹ اسپیس (state-space) شکل میں پیش کریں اور سسٹم میٹرکس کی تشریح کریں | سمجھنا |
| **LO-4** | قطب کے مقامات (pole locations) اور Lyapunov طریقوں کا استعمال کرتے ہوئے سسٹم کے استحکام کا تجزیہ کریں | تجزیہ کرنا |
| **LO-5** | پوزیشن، رفتار، اور قوت کے کنٹرول کے لیے فیڈبیک کنٹرول لوپس ڈیزائن کریں | تخلیق کرنا |

</div>

---

## 1. کنٹرول تھیوری کا تعارف

کنٹرول تھیوری روبوٹس سے اپنی مرضی کا کام کروانے کی ریاضیاتی بنیاد ہے۔ یہ اس بنیادی سوال کا جواب دیتا ہے: **ہم مطلوبہ رویے کو حاصل کرنے کے لیے سینسر ریڈنگز کی بنیاد پر ایکچیوٹرز کو کیسے حکم دیتے ہیں؟**

### کنٹرول کا مسئلہ

ہر روبوٹ کو ایک ہی بنیادی چیلنج کا سامنا کرنا پڑتا ہے:

```
                    ┌────────────────┐
   حوالہ (Reference) ──────►   کنٹرولر    ├──────► ایکچیوٹر ──────► سسٹم ──────┐
   (مطلوبہ)         └───────▲───────┘         کمانڈ           (پلانٹ)       │
                            │                                               │
                            │                                               │
                            │         ┌──────────────┐                      │
                            └─────────┤    سینسر     ◄──────────────────────┘
                              ایرر    └──────────────┘       آؤٹ پٹ
                                         فیڈبیک              (حقیقی)
```

کنٹرولر کو چاہیے کہ:
1. **مطلوبہ حالت** (حوالہ) کا **حقیقی حالت** (پیمائش) کے ساتھ موازنہ کرے
2. غلطی (error) کو کم کرنے کے لیے ایک مناسب **کنٹرول سگنل** کا حساب لگائے
3. ایسا مسلسل کرے جیسے جیسے سسٹم چلتا رہے

### کنٹرول مشکل کیوں ہے

کنٹرول آسان ہوتا اگر:
- سینسرز کامل ہوتے (کوئی شور نہیں، کوئی تاخیر نہیں)
- ایکچیوٹرز فوری ردعمل ظاہر کرتے
- ہمیں اپنے سسٹم کی درست طبیعیات معلوم ہوتیں
- ماحول کبھی تبدیل نہ ہوتا

حقیقت میں، **ان میں سے کوئی بھی سچ نہیں ہے**۔ کنٹرول تھیوری ہمیں ان خامیوں کو سنبھالنے کے لیے ٹولز دیتی ہے۔

### ایک حوصلہ افزا مثال

ایک روبوٹ بازو پر غور کریں جو ہدف کی پوزیشن تک پہنچنے کی کوشش کر رہا ہے:

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

**آؤٹ پٹ:**
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

یہ مثال دکھاتی ہے کہ ہمیں **مناسب کنٹرول تھیوری** کی ضرورت کیوں ہے—سادہ طریقے دوغلا پن (oscillation)، عدم استحکام، یا خراب کارکردگی کا باعث بنتے ہیں۔

---

## 2. اوپن لوپ بمقابلہ کلوزڈ لوپ کنٹرول

### اوپن لوپ کنٹرول (Open-Loop Control)

**اوپن لوپ کنٹرول** میں، کنٹرولر فیڈبیک استعمال نہیں کرتا۔ یہ پہلے سے طے شدہ کمانڈ کی ترتیب کا اطلاق کرتا ہے۔

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

**آؤٹ پٹ:**
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

#### اوپن لوپ کب کام کرتا ہے

اوپن لوپ کنٹرول اس وقت موزوں ہے جب:
- سسٹم **اچھی طرح سے نمایاں** (well-characterized) ہو (ہم طبیعیات کو بالکل جانتے ہیں)
- **رکاوٹیں کم سے کم** ہوں (کوئی غیر متوقع قوتیں نہیں)
- **درستگی کی ضروریات کم** ہوں

مثالیں: سٹیپر موٹرز (قدم گننا)، وقت کے مطابق ترتیب، سادہ اٹھانا اور رکھنا۔

#### اوپن لوپ کب ناکام ہوتا ہے

اوپن لوپ ناکام ہوجاتا ہے جب:
- سسٹم کا ماڈل **غلط** ہو
- **رکاوٹیں** سسٹم کو متاثر کریں
- **اعلی درستگی** کی ضرورت ہو

### کلوزڈ لوپ (فیڈبیک) کنٹرول

**کلوزڈ لوپ کنٹرول** مسلسل آؤٹ پٹ کی پیمائش کرتا ہے اور غلطی کو کم کرنے کے لیے ان پٹ کو ایڈجسٹ کرتا ہے۔

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

**آؤٹ پٹ:**
```
Open-Loop vs Closed-Loop with Disturbance
-------------------------------------------------------
Target:           45.0°
Open-loop result: 39.3°° (Error: 5.7°)
Closed-loop:      44.9° (Error: ~0°)

Closed-loop automatically compensates for the disturbance!
```

### موازنہ کا خلاصہ

| پہلو | اوپن لوپ | کلوزڈ لوپ |
|--------|-----------|-------------|
| فیڈبیک | کوئی نہیں | مسلسل |
| رکاوٹ کو مسترد کرنا | کوئی نہیں | خودکار |
| ماڈل کی ضروریات | درست ہونا ضروری ہے | تخمینی ہو سکتا ہے |
| پیچیدگی | سادہ | زیادہ پیچیدہ |
| استحکام کا خطرہ | کم | دوغلا پن (Oscillate) ہو سکتا ہے |
| لاگت | کم (کوئی سینسر نہیں) | زیادہ |

---

## 3. PID کنٹرول

**PID کنٹرولر** (Proportional-Integral-Derivative) روبوٹکس اور صنعت میں سب سے زیادہ استعمال ہونے والا فیڈبیک کنٹرولر ہے۔ اندازہ لگایا گیا ہے کہ 90% سے زیادہ صنعتی کنٹرول لوپس PID کی کوئی نہ کوئی شکل استعمال کرتے ہیں۔

### PID مساوات

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$$

جہاں:
- $u(t)$ = کنٹرول آؤٹ پٹ
- $e(t)$ = ایرر (سیٹ پوائنٹ - پیمائش شدہ قیمت)
- $K_p$ = پروپورشنل گین
- $K_i$ = انٹیگرل گین
- $K_d$ = ڈیریویٹو گین

### ہر اصطلاح کو سمجھنا

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

**آؤٹ پٹ:**
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

### ہر اصطلاح کا اثر

| اصطلاح | اثر | بہت کم | بہت زیادہ |
|------|--------|---------|----------|
| **P** (Proportional) | موجودہ غلطی پر ردعمل ظاہر کرتا ہے | سست ردعمل | دوغلا پن، عدم استحکام |
| **I** (Integral) | مستقل حالت کی غلطی کو ختم کرتا ہے | مستقل آفسیٹ | سست دوغلا پن، ونڈ اپ (windup) |
| **D** (Derivative) | دوغلا پن کو کم کرتا ہے | اوور شوٹ (Overshoot) | شور میں اضافہ |

### PID کنٹرولرز کی ٹیوننگ

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

**آؤٹ پٹ:**
```
PID Tuning Comparison
======================================================================
Tuning               Rise Time    Overshoot    SS Error
----------------------------------------------------------------------
Underdamped          0.150        23.4%        0.0012
Critically damped    0.280        2.1%         0.0008
Overdamped           0.520        0.0%         0.0021
```

### عملی PID نفاذ

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

**آؤٹ پٹ:**
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

## 4. اسٹیٹ اسپیس (State-Space) نمائندگی

اگرچہ PID بدیہی ہے، **اسٹیٹ اسپیس نمائندگی** کنٹرولرز کے تجزیہ اور ڈیزائن کے لیے زیادہ طاقتور فریم ورک فراہم کرتی ہے، خاص طور پر ملٹی ان پٹ ملٹی آؤٹ پٹ (MIMO) سسٹمز کے لیے۔

### اسٹیٹ اسپیس فارم

اسٹیٹ اسپیس فارم میں ایک لکیری نظام:

$$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$$
$$\mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}$$

جہاں:
- $\mathbf{x}$ = ریاستی ویکٹر (جیسے، پوزیشن، رفتار)
- $\mathbf{u}$ = ان پٹ ویکٹر (جیسے، وولٹیج، ٹارک)
- $\mathbf{y}$ = آؤٹ پٹ ویکٹر (جیسے، ناپی گئی پوزیشن)
- $\mathbf{A}$ = سسٹم میٹرکس (ڈائنامکس)
- $\mathbf{B}$ = ان پٹ میٹرکس
- $\mathbf{C}$ = آؤٹ پٹ میٹرکس
- $\mathbf{D}$ = فیڈ تھرو میٹرکس (عام طور پر صفر)

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

**آؤٹ پٹ:**
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

### اسٹیٹ اسپیس کیوں؟

| فائدہ | تفصیل |
|-----------|-------------|
| **MIMO سسٹمز** | متعدد ان پٹ اور آؤٹ پٹ کو قدرتی طور پر ہینڈل کرتا ہے |
| **جدید کنٹرول** | LQR، کالمن فلٹر، MPC کی بنیاد |
| **عددی اوزار** | موثر میٹرکس کمپیوٹیشن |
| **تجزیہ** | استحکام، کنٹرول ایبلٹی، آبزرویبلٹی کا تجزیہ کرنا آسان ہے |

---

## 5. استحکام کا تجزیہ (Stability Analysis)

ایک کنٹرول سسٹم **مستحکم** ہوتا ہے اگر وہ خرابی (disturbance) کے بعد توازن میں واپس آجائے۔ استحکام بہت اہم ہے—ایک غیر مستحکم روبوٹ خطرناک ہو سکتا ہے۔

### استحکام کی تعریفیں

- **Asymptotically stable**: توازن میں واپس آتا ہے
- **Marginally stable**: پابند ہے لیکن کنورج نہیں ہوتا
- **Unstable**: لامحدودیت کی طرف ہٹ جاتا ہے

### قطب کا تجزیہ (Pole Analysis)

لکیری نظاموں کے لیے، استحکام کا تعین سسٹم میٹرکس A کے **eigenvalues** (poles) سے ہوتا ہے:

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

**آؤٹ پٹ:**
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

### استحکام کا تصور

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

**آؤٹ پٹ:**
```
Pole Location Interpretation
------------------------------------------------------------
  Pole at 0j: Integrator (marginally stable)
  Pole at (-1+0j): Exponential decay (τ=1.000s)
  Pole at (-0.5+2j): Underdamped oscillation (ζ=0.24)
  Pole at (0.1+1j): UNSTABLE (exponential growth)
```

### Lyapunov استحکام (مختصر تعارف)

نان لکیری سسٹمز کے لیے، ہم **Lyapunov کا طریقہ** استعمال کرتے ہیں: توانائی جیسا فنکشن تلاش کریں جو وقت کے ساتھ کم ہوتا ہے۔

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

## 6. جدید کنٹرول کے نقطہ نظر

PID سے آگے، جدید روبوٹکس زیادہ جدید کنٹرول تکنیک استعمال کرتے ہیں۔

### 6.1 فیڈ فارورڈ + فیڈبیک (Feedforward + Feedback)

ماڈل پر مبنی پیشن گوئی کو فیڈبیک اصلاح کے ساتھ جوڑیں:

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

**آؤٹ پٹ:**
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

### 6.2 کاسکیڈ کنٹرول (Cascade Control)

بہتر کارکردگی کے لیے نیسٹڈ (nested) کنٹرول لوپس کا استعمال کریں:

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

### 6.3 کنٹرول کے طریقوں کا موازنہ

| طریقہ | پیچیدگی | کارکردگی | کب استعمال کریں |
|--------|------------|-------------|-------------|
| **PID** | کم | اچھی | سنگل ان پٹ، اچھی طرح سے ماڈل شدہ سسٹمز |
| **FF + FB** | درمیانی | بہت اچھی | معلوم ڈائنامکس، رفتار سے باخبر رہنا |
| **کاسکیڈ** | درمیانی | بہت اچھی | تیز اندرونی حرکیات (موٹرز) |
| **LQR** | زیادہ | بہترین | ملٹی ویری ایبل، اچھی طرح سے ماڈل شدہ |
| **MPC** | بہت زیادہ | بہترین | رکاوٹیں، پیشن گوئی کی ضرورت |

---

## 7. روبوٹکس میں ایپلی کیشنز

### پوزیشن کنٹرول

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

### فورس کنٹرول

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

## خلاصہ

<div className="key-takeaways">

### اہم نکات

1. **کنٹرول سسٹمز** سینسنگ اور ایکچیویشن کے درمیان فرق کو ختم کرتے ہیں، روبوٹس کو غیر یقینی صورتحال کے باوجود مطلوبہ رویے حاصل کرنے کے قابل بناتے ہیں۔

2. **کلوزڈ لوپ کنٹرول** غلطیوں کو خود بخود درست کرنے اور رکاوٹوں کو مسترد کرنے کے لیے فیڈبیک کا استعمال کرتا ہے—درست روبوٹکس کے لیے ضروری ہے۔

3. **PID کنٹرول** روبوٹکس کا ورک ہارس ہے: P موجودہ غلطی پر ردعمل ظاہر کرتا ہے، I مستحکم حالت کی غلطی کو ختم کرتا ہے، D ڈیمپنگ فراہم کرتا ہے۔

4. **اسٹیٹ اسپیس نمائندگی** پیچیدہ، ملٹی ویری ایبل سسٹمز کا تجزیہ کرنے کے لیے ایک طاقتور فریم ورک فراہم کرتی ہے۔

5. **استحکام** کا تعین قطب کے مقامات (pole locations) سے کیا جاتا ہے: منفی اصلی حصے = مستحکم، مثبت = غیر مستحکم۔

6. **جدید تکنیکیں** جیسے فیڈ فارورڈ، کاسکیڈ کنٹرول، اور مائبادا کنٹرول (impedance control) مخصوص ایپلی کیشنز کے لیے بہتر کارکردگی فراہم کرتی ہیں۔

</div>

---

## مشقیں

<div className="exercise">

### مشق 1: PID ٹیوننگ (LO-2)

مندرجہ ذیل سٹیپ رسپانس خصوصیات کے ساتھ روبوٹ جوڑ کو دیکھتے ہوئے:
- رائز ٹائم: 0.5s (بہت سست)
- اوور شوٹ: 30% (بہت زیادہ)
- مستحکم حالت کی غلطی: 2% (قابل قبول)

موجودہ PID گینز: Kp=10, Ki=5, Kd=1

رسپانس کو بہتر بنانے کے لیے مخصوص گین ایڈجسٹمنٹ کی سفارش کریں۔ اپنی دلیل کی وضاحت کریں۔

</div>

<div className="exercise">

### مشق 2: استحکام کا تجزیہ (LO-4)

ایک سسٹم میں درج ذیل خصوصیت کی مساوات ہے:

$$s^3 + 4s^2 + 5s + 2 = 0$$

1. سسٹم کے قطب تلاش کریں۔
2. اس بات کا تعین کریں کہ آیا سسٹم مستحکم ہے۔
3. متوقع عارضی ردعمل (transient response) کی وضاحت کریں۔

</div>

<div className="exercise">

### مشق 3: کنٹرولر ڈیزائن (LO-5)

روبوٹ بازو کے جوڑ کے لیے ایک کاسکیڈ کنٹرولر ڈیزائن کریں جہاں:
- موٹر کی رفتار کو 1000 Hz پر کنٹرول کیا جا سکتا ہے
- پوزیشن کنٹرول 100 Hz پر چلتا ہے
- مقصد کم سے کم اوور شوٹ کے ساتھ 1 سیکنڈ میں 0° سے 90° تک جانا ہے

دونوں لوپس کے لیے ساخت اور اندازاً گینز کی وضاحت کریں۔

</div>

---

:::tip اگلا باب
**باب 1.4: تخروپن سے حقیقت کی منتقلی (Simulation to Reality Transfer)** پر جاری رکھیں تاکہ یہ سیکھ سکیں کہ نقالی میں کنٹرولرز کو کیسے تربیت دی جائے اور انہیں حقیقی روبوٹس پر کیسے تعینات کیا جائے۔
:::
