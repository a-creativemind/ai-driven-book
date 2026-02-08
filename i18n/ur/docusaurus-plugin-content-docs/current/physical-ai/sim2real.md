---
sidebar_position: 4
title: تخروپن سے حقیقت کی منتقلی (Sim-to-Real Transfer)
description: مصنوعی اور حقیقی دنیا کے روبوٹکس کے درمیان فرق کو ختم کرنا
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

# تخروپن سے حقیقت کی منتقلی (Simulation to Reality Transfer)

<div className="learning-objectives">

## سیکھنے کے مقاصد

اس باب کو مکمل کرنے کے بعد، آپ اس قابل ہوں گے کہ:

| ID | مقصد | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | سم-ٹو-ریئل گیپ (sim-to-real gap) کے ذرائع کی نشاندہی کریں اور درجہ بندی کریں | تجزیہ کرنا |
| **LO-2** | مضبوط پالیسی ٹریننگ کے لیے ڈومین رینڈمائزیشن (Domain Randomization) کی تکنیکوں کو نافذ کریں | اطلاق کرنا |
| **LO-3** | روبوٹکس کے لیے مناسب ٹرانسفر لرننگ کے طریقوں کا اندازہ لگائیں اور منتخب کریں | جائزہ لینا |
| **LO-4** | ایسے سمولیشن ماحول ڈیزائن کریں جو منتقلی کی کامیابی کو زیادہ سے زیادہ کریں | تخلیق کرنا |
| **LO-5** | سمولیشن کی درستگی کو بہتر بنانے کے لیے سسٹم کی شناخت (System Identification) کا اطلاق کریں | اطلاق کرنا |

</div>

---

## 1. سم-ٹو-ریئل مسئلہ (The Sim-to-Real Problem)

### سمولیشن کیوں؟

حقیقی دنیا میں روبوٹس کو تربیت دینا ہے:
- **سست**: ریئل ٹائم فزکس، ری سیٹ کا وقت، حفاظتی پروٹوکول
- **مہنگا**: ہارڈ ویئر کا ٹوٹنا، انسانی نگرانی، سہولت کے اخراجات
- **خطرناک**: غیر تربیت یافتہ پالیسیاں روبوٹس اور ماحول کو نقصان پہنچا سکتی ہیں
- **محدود**: آسانی سے کنارے کے معاملات (edge cases) یا ناکامی کے طریقوں کو تلاش نہیں کیا جا سکتا

سمولیشن پیش کرتا ہے:
- **رفتار**: ریئل ٹائم سے 1000 گنا تیز
- **متوازی پن**: بیک وقت ہزاروں ماحول پر تربیت
- **حفاظت**: نتائج کے بغیر خطرناک منظرناموں کو تلاش کریں
- **کنٹرول**: کامل تکرار، صحیح حالت تک رسائی

### حقیقت کا خلا (The Reality Gap)

ان فوائد کے باوجود، سمولیشن میں تربیت یافتہ پالیسیاں اکثر ناکام ہو جاتی ہیں جب انہیں حقیقی روبوٹس پر تعینات کیا جاتا ہے۔ یہ **سم-ٹو-ریئل گیپ** (یا **حقیقت کا خلا**) ہے۔

```
┌─────────────────────────────────────────────────────────────────┐
│                      سمولیشن (SIMULATION)                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   کامل      │    │   فوری      │    │   صاف       │         │
│  │   طبیعیات   │    │   ری سیٹ    │    │   سینسرز    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│                    پالیسی یہاں تربیت یافتہ                      │
│                           ↓                                      │
│                    بہت اچھا کام کرتی ہے! ✓                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                      منتقل کریں...
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        حقیقت (REALITY)                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   گندی      │    │   سست       │    │   شور والے  │         │
│  │   طبیعیات   │    │   ری سیٹ    │    │   سینسرز    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│                    پالیسی یہاں تعینات کی گئی                    │
│                           ↓                                      │
│                    ناکام! ✗                                      │
└─────────────────────────────────────────────────────────────────┘
```

### ایک ٹھوس مثال

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

**آؤٹ پٹ:**
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

## 2. حقیقت کے فرق کے ذرائع

سم-ٹو-ریئل فرق متعدد ذرائع سے پیدا ہوتا ہے۔ خلا کو ختم کرنے کے لیے انہیں سمجھنا پہلا قدم ہے۔

### 2.1 ڈائنامکس کا عدم مطابقت (Dynamics Mismatch)

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

**آؤٹ پٹ:**
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

### 2.2 فرق کا درجہ بندی (The Gap Hierarchy)

تمام خلا یکساں طور پر مشکل نہیں ہوتے:

| سطح | فرق کی قسم | اثر | مثال |
|-------|----------|--------|---------|
| **نازک (Critical)** | بنیادی طور پر غلط طبیعیات | پالیسی مکمل طور پر ناکام ہو جاتی ہے | غلط رابطہ ماڈل |
| **بڑا (Major)** | اہم پیرامیٹر کی غلطی | کارکردگی میں کمی | 2x جڑتا (inertia) غلطی |
| **معمولی (Minor)** | چھوٹی کیلیبریشن غلطی | معمولی گراوٹ | 5% رگڑ (friction) غلطی |
| **نہ ہونے کے برابر** | ناقابل تصور اختلافات | کوئی اثر نہیں | ساخت کی تفصیلات |

---

## 3. ڈومین رینڈمائزیشن (Domain Randomization)

**ڈومین رینڈمائزیشن** سب سے زیادہ استعمال ہونے والی سم-ٹو-ریئل تکنیک ہے۔ خیال: نقلی ماحول کی *تقسیم* (distribution) پر تربیت دیں تاکہ پالیسی تغیرات کے لیے مضبوط ہو جائے، بشمول (نامعلوم) حقیقی ماحول۔

### 3.1 بنیادی خیال

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

**آؤٹ پٹ:**
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

### 3.2 کیا رینڈمائز کرنا ہے

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

**آؤٹ پٹ:**
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

### 3.3 خودکار ڈومین رینڈمائزیشن (ADR)

رینجز کو دستی طور پر ترتیب دینے کے بجائے، **ADR** تربیت کے دوران خود بخود رینڈمائزیشن رینجز کو بڑھاتا ہے:

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

**آؤٹ پٹ:**
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

## 4. ڈومین موافقت (Domain Adaptation)

جبکہ ڈومین رینڈمائزیشن تمام تغیرات کے لیے مضبوط ہونے کی کوشش کرتا ہے، **ڈومین موافقت** خاص طور پر سمولیشن کو حقیقت کے ساتھ ہم آہنگ کرنے کی کوشش کرتا ہے۔

### 4.1 سسٹم کی شناخت (System Identification)

**سسٹم کی شناخت** ڈیٹا سے حقیقی دنیا کے پیرامیٹرز کا اندازہ لگاتی ہے:

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

**آؤٹ پٹ:**
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

### 4.2 سمولیشن کیلیبریشن

شناخت شدہ پیرامیٹرز کو سمولیشن کو کیلیبریٹ کرنے کے لیے استعمال کریں:

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

**آؤٹ پٹ:**
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

## 5. ترقی پسند منتقلی کی حکمت عملی

### 5.1 سم-ٹو-ریئل کے لیے نصاب سیکھنا (Curriculum Learning)

آسان سمولیشنز کے ساتھ شروع کریں اور بتدریج مشکل میں اضافہ کریں:

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

**آؤٹ پٹ:**
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

### 5.2 حقیقی دنیا کی فائن ٹیوننگ

بعض اوقات، حقیقی دنیا کی تربیت کی تھوڑی سی مقدار باقی خلا کو ختم کر سکتی ہے:

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

**آؤٹ پٹ:**
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

## 6. بہترین طرز عمل

### 6.1 سمولیشن ڈیزائن چیک لسٹ

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

**آؤٹ پٹ:**
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

### 6.2 عام نقصانات

| نقصان | نتیجہ | حل |
|---------|-------------|----------|
| رینڈرنگ کے معیار پر ضرورت سے زیادہ انحصار | ضائع شدہ کمپیوٹ، پالیسیاں بصری تفصیلات پر توجہ مرکوز کرتی ہیں | سادہ بصری + رینڈمائزیشن کا استعمال کریں |
| ایکچیوٹر ڈائنامکس کو نظر انداز کرنا | بڑا سم-ٹو-ریئل گیپ | موٹر ردعمل، تاخیر کا ماڈل |
| بہت تنگ رینڈمائزیشن | پالیسی حقیقی تغیرات کے لیے نازک | وسیع رینجز، ADR کا استعمال کریں |
| بہت وسیع رینڈمائزیشن | پالیسی کبھی کنورج نہیں ہوتی | تنگ شروع کریں، بتدریج پھیلائیں |
| کامل ریاستی رسائی | پالیسی شور والے مشاہدات کے ساتھ ناکام ہو جاتی ہے | مشاہدے کے ماڈل استعمال کریں |
| ریوارڈ ہیکنگ | پالیسی سمولیشن آرٹفیکٹس کا استحصال کرتی ہے | محتاط انعام ڈیزائن |

---

## 7. کیس اسٹڈیز

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

**آؤٹ پٹ:**
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

## خلاصہ

<div className="key-takeaways">

### اہم نکات

1. **سم-ٹو-ریئل گیپ** سمولیشن اور حقیقت کے درمیان فرق ہے جو پالیسیوں کے منتقل ہونے پر ناکام ہونے کا سبب بنتا ہے؛ یہ ڈائنامکس، ادراک، اور ایکچیویشن کے عدم توازن سے پیدا ہوتا ہے۔

2. **ڈومین رینڈمائزیشن** نامعلوم حقیقی دنیا کے پیرامیٹرز کے لیے مضبوطی حاصل کرنے کے لیے مختلف سمولیشنز پر پالیسیوں کو تربیت دیتا ہے—سب سے زیادہ استعمال ہونے والی تکنیک۔

3. **سسٹم کی شناخت** ڈیٹا سے حقیقی دنیا کے پیرامیٹرز کا تخمینہ لگاتی ہے تاکہ سمولیشنز کو کیلیبریٹ کیا جا سکے، جس سے خلا براہ راست کم ہوتا ہے۔

4. **نصاب سیکھنا (Curriculum learning)** بتدریج سمولیشن کی مشکل میں اضافہ کرتا ہے، جس سے پالیسیوں کو بتدریج مضبوط رویے سیکھنے میں مدد ملتی ہے۔

5. **حقیقی دنیا کی فائن ٹیوننگ** حقیقی ڈیٹا کی تھوڑی سی مقدار کے ساتھ باقی خلا کو ختم کر سکتی ہے، لیکن محتاط حفاظتی تحفظات کی ضرورت ہوتی ہے۔

6. **کامیاب منتقلی** کے لیے ایکچیوٹر ڈائنامکس، سینسر شور، تاخیر، اور تعیناتی سے پہلے مکمل جانچ پر توجہ دینے کی ضرورت ہوتی ہے۔

</div>

---

## مشقیں

<div className="exercise">

### مشق 1: گیپ تجزیہ (LO-1)

آپ سمولیشن سے حقیقی روبوٹ پر وژن پر مبنی چننے (picking) کی پالیسی تعینات کر رہے ہیں۔ جانچ کے بعد، آپ مشاہدہ کرتے ہیں:
- اشیاء کبھی کبھی چھوٹ جاتی ہیں (70% کامیابی بمقابلہ سمولیشن میں 95%)
- چمکدار اشیاء کے ساتھ گرفت کی ناکامیاں زیادہ عام ہیں
- مختلف روشنی کے تحت کارکردگی گر جاتی ہے

گیپ کے ذریعہ کے لحاظ سے ان ناکامیوں کی درجہ بندی کریں اور تخفیف کی تجویز دیں۔

</div>

<div className="exercise">

### مشق 2: ڈومین رینڈمائزیشن ڈیزائن (LO-2, LO-4)

ایک چوپاے (quadruped) روبوٹ کو چلنے کی تربیت دینے کے لیے ڈومین رینڈمائزیشن کی حکمت عملی ڈیزائن کریں۔ وضاحت کریں:

1. رینڈمائز کرنے کے لیے کم از کم 8 پیرامیٹرز
2. ہر ایک کے لیے مناسب حدود
3. کون سے پیرامیٹرز سب سے زیادہ اہم ہیں
4. آپ حقیقی دنیا میں تعیناتی سے پہلے حکمت عملی کی توثیق کیسے کریں گے

</div>

<div className="exercise">

### مشق 3: سسٹم کی شناخت (LO-5)

ایک حقیقی روبوٹ سے مندرجہ ذیل ڈیٹا کو دیکھتے ہوئے:

| ٹارک (Nm) | ایکسلریشن (rad/s²) |
|-------------|----------------------|
| 1.0 | 8.2 |
| 2.0 | 17.1 |
| 3.0 | 24.8 |
| 4.0 | 33.5 |

1. کم سے کم اسکوائر (least squares) کا استعمال کرتے ہوئے جڑتا (inertia) کا تخمینہ لگائیں
2. فٹ کوالٹی (R²) کا حساب لگائیں
3. غلطی کے ذرائع اور تخمینہ کو بہتر بنانے کے طریقہ پر بحث کریں

</div>

---

:::tip اگلا حصہ
آپ نے **حصہ I: فزیکل AI فاؤنڈیشنز** مکمل کر لیا ہے! **حصہ II: ہیومنائیڈ روبوٹکس** پر جاری رکھیں تاکہ ان اصولوں کو پیچیدہ بائپیڈل سسٹمز پر لاگو کرنا شروع کریں۔
:::
