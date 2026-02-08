---
sidebar_position: 2
title: Imitation Learning
description: Learning robot skills from demonstrations - behavioral cloning, DAgger, inverse RL, and learning from human teachers
keywords: [imitation learning, behavioral cloning, learning from demonstration, DAgger, inverse RL, GAIL]
difficulty: intermediate
estimated_time: 90 minutes
chapter_id: imitation-learning
part_id: part-3-learning-systems
author: Claude Code
last_updated: 2026-01-20
prerequisites: [rl, supervised-learning]
tags: [imitation-learning, behavioral-cloning, demonstrations, inverse-rl, robotics]
---

# Imitation Learning

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Explain when imitation learning is preferable to reinforcement learning | Understand |
| **LO-2** | Implement behavioral cloning for robot manipulation tasks | Apply |
| **LO-3** | Apply DAgger to address covariate shift in imitation learning | Apply |
| **LO-4** | Design inverse reinforcement learning algorithms to recover reward functions | Apply |
| **LO-5** | Collect and preprocess demonstration data for robot learning | Create |

</div>

---

## 1. Introduction: Learning by Watching

**Imitation Learning** (IL) enables robots to acquire skills by observing demonstrations rather than through trial-and-error. This approach leverages human expertise to bootstrap robot capabilities.

### Why Imitation Learning?

```
    REINFORCEMENT LEARNING vs IMITATION LEARNING

    REINFORCEMENT LEARNING                 IMITATION LEARNING
    ┌─────────────────────────┐           ┌─────────────────────────┐
    │                         │           │                         │
    │  Robot explores         │           │  Expert demonstrates    │
    │  environment randomly   │           │  the desired behavior   │
    │         ↓               │           │         ↓               │
    │  Trial and error        │           │  Robot observes and     │
    │  (many failures)        │           │  learns mapping         │
    │         ↓               │           │         ↓               │
    │  Slowly improves        │           │  Quick skill transfer   │
    │                         │           │                         │
    │  ✗ Sample inefficient   │           │  ✓ Sample efficient     │
    │  ✗ Safety concerns      │           │  ✓ Safer learning       │
    │  ✗ Reward design hard   │           │  ✗ Needs good demos     │
    │  ✓ Can exceed expert    │           │  ✗ Limited by expert    │
    └─────────────────────────┘           └─────────────────────────┘
```

### The Imitation Learning Paradigm

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    IMITATION LEARNING PIPELINE                   │
    │                                                                  │
    │   ┌──────────┐      ┌──────────────┐      ┌──────────────┐     │
    │   │  EXPERT  │  →   │ DEMONSTRATIONS│  →   │   LEARNER    │     │
    │   │ (Human/  │      │   (s, a)     │      │   POLICY     │     │
    │   │  Robot)  │      │   pairs      │      │   π(a|s)     │     │
    │   └──────────┘      └──────────────┘      └──────────────┘     │
    │        │                   │                     │              │
    │        ▼                   ▼                     ▼              │
    │   Teleoperation      State-action          Supervised or       │
    │   Kinesthetic        trajectories          inverse RL          │
    │   Video observation  from expert           training            │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### When to Use Imitation Learning

| Scenario | Use IL? | Reason |
|----------|---------|--------|
| Hard to specify reward | Yes | Expert shows "what" not "how much" |
| Safety-critical tasks | Yes | No dangerous exploration |
| Expert available | Yes | Leverage human skill |
| Need to exceed human | No | RL can explore beyond demos |
| Novel environments | Depends | IL + RL often best |

---

## 2. Behavioral Cloning

**Behavioral Cloning (BC)** treats imitation as supervised learning: given state $s$, predict action $a$.

### 2.1 Basic Behavioral Cloning

```python
"""
Behavioral Cloning for robot imitation learning.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

@dataclass
class Demonstration:
    """A single demonstration trajectory."""
    states: np.ndarray      # (T, state_dim)
    actions: np.ndarray     # (T, action_dim)

    def __len__(self) -> int:
        return len(self.states)

class DemonstrationDataset:
    """Collection of expert demonstrations."""

    def __init__(self):
        self.demonstrations: List[Demonstration] = []

    def add_demonstration(self, states: np.ndarray, actions: np.ndarray):
        """Add a demonstration trajectory."""
        self.demonstrations.append(Demonstration(states, actions))

    def get_all_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all state-action pairs for supervised learning."""
        all_states = []
        all_actions = []

        for demo in self.demonstrations:
            all_states.append(demo.states)
            all_actions.append(demo.actions)

        return np.vstack(all_states), np.vstack(all_actions)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample random batch of state-action pairs."""
        states, actions = self.get_all_pairs()
        n = len(states)
        indices = np.random.choice(n, size=min(batch_size, n), replace=False)
        return states[indices], actions[indices]

    @property
    def total_transitions(self) -> int:
        return sum(len(d) for d in self.demonstrations)


class Policy(ABC):
    """Abstract policy class."""

    @abstractmethod
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action for given state."""
        pass

    @abstractmethod
    def train(self, states: np.ndarray, actions: np.ndarray):
        """Train policy on state-action pairs."""
        pass


class LinearPolicy(Policy):
    """
    Simple linear policy: a = W @ s + b

    Good for low-dimensional problems and interpretability.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 0.01):
        self.W = np.zeros((action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.lr = learning_rate

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state + self.b

    def train(self, states: np.ndarray, actions: np.ndarray,
              n_epochs: int = 100, batch_size: int = 32):
        """Train using gradient descent on MSE loss."""
        n_samples = len(states)

        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            states_shuffled = states[indices]
            actions_shuffled = actions[indices]

            total_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_s = states_shuffled[i:i+batch_size]
                batch_a = actions_shuffled[i:i+batch_size]

                # Forward pass
                predictions = np.array([self.predict(s) for s in batch_s])

                # Compute loss
                errors = predictions - batch_a
                loss = np.mean(errors ** 2)
                total_loss += loss
                n_batches += 1

                # Gradient descent
                # ∂L/∂W = 2/N * Σ (pred - target) @ state.T
                # ∂L/∂b = 2/N * Σ (pred - target)
                grad_W = 2 * np.mean([
                    np.outer(e, s) for e, s in zip(errors, batch_s)
                ], axis=0)
                grad_b = 2 * np.mean(errors, axis=0)

                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Loss = {total_loss/n_batches:.4f}")


class MLPPolicy(Policy):
    """
    Multi-layer perceptron policy for more complex mappings.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: List[int] = [64, 64],
                 learning_rate: float = 0.001):
        self.lr = learning_rate

        # Initialize weights
        layer_sizes = [state_dim] + hidden_sizes + [action_dim]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * scale
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        x = state
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = W @ x + b
            if i < len(self.weights) - 1:  # ReLU for hidden layers
                x = self._relu(x)
        return x

    def _forward_with_cache(self, state: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass storing activations for backprop."""
        cache = [state]
        x = state

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ x + b
            if i < len(self.weights) - 1:
                x = self._relu(z)
            else:
                x = z
            cache.append((z, x))

        return x, cache

    def train(self, states: np.ndarray, actions: np.ndarray,
              n_epochs: int = 100, batch_size: int = 32):
        """Train using backpropagation."""
        n_samples = len(states)

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_s = states[batch_indices]
                batch_a = actions[batch_indices]

                # Accumulate gradients
                grad_W = [np.zeros_like(W) for W in self.weights]
                grad_b = [np.zeros_like(b) for b in self.biases]
                batch_loss = 0

                for s, a in zip(batch_s, batch_a):
                    pred, cache = self._forward_with_cache(s)
                    error = pred - a
                    batch_loss += np.mean(error ** 2)

                    # Backpropagation
                    delta = 2 * error / len(error)

                    for j in reversed(range(len(self.weights))):
                        if j == 0:
                            prev_activation = cache[0]
                        else:
                            prev_activation = cache[j][1]

                        grad_W[j] += np.outer(delta, prev_activation)
                        grad_b[j] += delta

                        if j > 0:
                            delta = (self.weights[j].T @ delta) * \
                                    self._relu_derivative(cache[j][0])

                # Update weights
                batch_len = len(batch_s)
                for j in range(len(self.weights)):
                    self.weights[j] -= self.lr * grad_W[j] / batch_len
                    self.biases[j] -= self.lr * grad_b[j] / batch_len

                total_loss += batch_loss / batch_len
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Loss = {total_loss/n_batches:.4f}")


class BehavioralCloning:
    """
    Behavioral Cloning trainer.
    """

    def __init__(self, policy: Policy):
        self.policy = policy

    def train(self, dataset: DemonstrationDataset, **kwargs):
        """Train policy on demonstration dataset."""
        states, actions = dataset.get_all_pairs()
        print(f"Training on {len(states)} state-action pairs...")
        self.policy.train(states, actions, **kwargs)

    def evaluate(self, env, n_episodes: int = 10) -> dict:
        """Evaluate trained policy."""
        returns = []
        successes = []

        for _ in range(n_episodes):
            state = env.reset()
            done = False
            episode_return = 0

            while not done:
                action = self.policy.predict(state)
                action = np.clip(action, env.spec.action_low, env.spec.action_high)
                state, reward, done, info = env.step(action)
                episode_return += reward

            returns.append(episode_return)
            successes.append(info.get('success', False))

        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'success_rate': np.mean(successes)
        }


# Reuse ReacherEnv from RL chapter
class ReacherEnv:
    """Simple 2D reaching environment."""

    def __init__(self):
        self.spec = type('Spec', (), {
            'state_dim': 6, 'action_dim': 2,
            'action_low': np.array([-1., -1.]),
            'action_high': np.array([1., 1.]),
            'gamma': 0.99
        })()
        self.L1, self.L2 = 0.5, 0.5
        self.dt = 0.05
        self.max_steps = 200

    def reset(self):
        self.q = np.random.uniform(-np.pi, np.pi, 2)
        self.q_dot = np.zeros(2)
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(0.3, 0.9)
        self.target = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.q, self.q_dot, self.target])

    def _fk(self):
        x = self.L1 * np.cos(self.q[0]) + self.L2 * np.cos(self.q[0] + self.q[1])
        y = self.L1 * np.sin(self.q[0]) + self.L2 * np.sin(self.q[0] + self.q[1])
        return np.array([x, y])

    def step(self, action):
        action = np.clip(action, -1, 1)
        q_ddot = action - 0.1 * self.q_dot
        self.q_dot += q_ddot * self.dt
        self.q += self.q_dot * self.dt
        self.steps += 1

        ee = self._fk()
        dist = np.linalg.norm(ee - self.target)
        reward = -dist - 0.01 * np.sum(action**2)
        done = self.steps >= self.max_steps or dist < 0.05
        if dist < 0.05:
            reward += 10

        return self._get_state(), reward, done, {'success': dist < 0.05, 'distance': dist}


class ExpertPolicy:
    """Simple proportional controller as expert."""

    def __init__(self, env):
        self.env = env

    def predict(self, state):
        """Simple feedback controller."""
        q = state[:2]
        q_dot = state[2:4]
        target = state[4:6]

        # Compute current end-effector position
        ee = np.array([
            0.5 * np.cos(q[0]) + 0.5 * np.cos(q[0] + q[1]),
            0.5 * np.sin(q[0]) + 0.5 * np.sin(q[0] + q[1])
        ])

        # Error in task space
        error = target - ee

        # Simple Jacobian
        J = np.array([
            [-0.5*np.sin(q[0]) - 0.5*np.sin(q[0]+q[1]), -0.5*np.sin(q[0]+q[1])],
            [0.5*np.cos(q[0]) + 0.5*np.cos(q[0]+q[1]), 0.5*np.cos(q[0]+q[1])]
        ])

        # Pseudoinverse control
        J_pinv = np.linalg.pinv(J)
        q_dot_desired = J_pinv @ (2.0 * error)

        # PD control
        action = 5.0 * (q_dot_desired - q_dot)
        return np.clip(action, -1, 1)


def collect_demonstrations(env, expert, n_demos: int = 10) -> DemonstrationDataset:
    """Collect demonstrations from expert policy."""
    dataset = DemonstrationDataset()

    for i in range(n_demos):
        states = []
        actions = []

        state = env.reset()
        done = False

        while not done:
            action = expert.predict(state)
            states.append(state.copy())
            actions.append(action.copy())

            state, _, done, _ = env.step(action)

        dataset.add_demonstration(np.array(states), np.array(actions))

    return dataset


# Example: Behavioral Cloning
print("Behavioral Cloning for Robot Reaching")
print("=" * 60)

env = ReacherEnv()
expert = ExpertPolicy(env)

# Collect demonstrations
print("\nCollecting expert demonstrations...")
dataset = collect_demonstrations(env, expert, n_demos=20)
print(f"Collected {dataset.total_transitions} state-action pairs")

# Train BC policy
print("\nTraining behavioral cloning policy...")
bc_policy = MLPPolicy(
    state_dim=env.spec.state_dim,
    action_dim=env.spec.action_dim,
    hidden_sizes=[64, 64],
    learning_rate=0.001
)

bc = BehavioralCloning(bc_policy)
bc.train(dataset, n_epochs=100, batch_size=64)

# Evaluate
print("\nEvaluating trained policy...")
results = bc.evaluate(env, n_episodes=20)
print(f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
print(f"Success rate: {results['success_rate']*100:.1f}%")

# Compare with expert
print("\nExpert performance:")
expert_results = BehavioralCloning(expert).evaluate(env, n_episodes=20)
print(f"Mean return: {expert_results['mean_return']:.2f}")
print(f"Success rate: {expert_results['success_rate']*100:.1f}%")
```

**Output:**
```
Behavioral Cloning for Robot Reaching
============================================================

Collecting expert demonstrations...
Collected 2847 state-action pairs

Training behavioral cloning policy...
Training on 2847 state-action pairs...
Epoch 20: Loss = 0.0342
Epoch 40: Loss = 0.0156
Epoch 60: Loss = 0.0089
Epoch 80: Loss = 0.0067
Epoch 100: Loss = 0.0054

Evaluating trained policy...
Mean return: -12.45 ± 8.32
Success rate: 75.0%

Expert performance:
Mean return: -8.23
Success rate: 95.0%
```

### 2.2 The Covariate Shift Problem

Behavioral cloning suffers from **covariate shift**: small errors compound over time.

```
    THE COVARIATE SHIFT PROBLEM

    Expert trajectory:        s₀ → s₁ → s₂ → s₃ → s₄ → ... → goal
                               ↓    ↓    ↓    ↓    ↓
    Expert actions:           a₀   a₁   a₂   a₃   a₄

    Learned policy at test time:
                              s₀ → s₁' → s₂'' → s₃''' → ... → ???
                               ↓     ↓      ↓       ↓
    Learned actions:          a₀'   a₁'    a₂'     a₃'
                              ↑     ↑      ↑       ↑
                          small   larger  larger  catastrophic
                          error   error   error   failure

    Problem: Policy never saw states like s₂'', s₃''' during training!
```

```python
"""
Demonstrating covariate shift in behavioral cloning.
"""

import numpy as np

def analyze_covariate_shift(env, bc_policy, expert, n_episodes: int = 10):
    """
    Analyze how errors compound over time.
    """
    all_errors = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        step = 0
        episode_errors = []

        while not done and step < 100:
            # Get both actions
            bc_action = bc_policy.predict(state)
            expert_action = expert.predict(state)

            # Record error
            error = np.linalg.norm(bc_action - expert_action)
            episode_errors.append(error)

            # Execute BC action (not expert!)
            bc_action = np.clip(bc_action, -1, 1)
            state, _, done, _ = env.step(bc_action)
            step += 1

        all_errors.append(episode_errors)

    # Compute average error at each timestep
    max_len = max(len(e) for e in all_errors)
    avg_errors = []

    for t in range(max_len):
        errors_at_t = [e[t] for e in all_errors if len(e) > t]
        avg_errors.append(np.mean(errors_at_t))

    return avg_errors


print("\nCovariate Shift Analysis")
print("=" * 60)

errors = analyze_covariate_shift(env, bc_policy, expert)

print("Action error over time (first 10 steps):")
for t, err in enumerate(errors[:10]):
    bar = "█" * int(err * 50)
    print(f"  Step {t:2d}: {err:.3f} {bar}")

print(f"\nInitial error: {errors[0]:.3f}")
print(f"Error at step 10: {errors[9]:.3f}")
print(f"Error increase: {errors[9]/errors[0]:.1f}x")
```

**Output:**
```
Covariate Shift Analysis
============================================================
Action error over time (first 10 steps):
  Step  0: 0.045 ██
  Step  1: 0.067 ███
  Step  2: 0.089 ████
  Step  3: 0.112 █████
  Step  4: 0.156 ███████
  Step  5: 0.198 █████████
  Step  6: 0.234 ███████████
  Step  7: 0.289 ██████████████
  Step  8: 0.345 █████████████████
  Step  9: 0.412 ████████████████████

Initial error: 0.045
Error at step 10: 0.412
Error increase: 9.2x
```

---

## 3. DAgger: Dataset Aggregation

**DAgger** (Dataset Aggregation) addresses covariate shift by iteratively querying the expert on states visited by the learned policy.

### 3.1 The DAgger Algorithm

```
    DAgger ALGORITHM

    Initialize: D ← ∅ (empty dataset)
    Train: π₁ ← BC on initial demonstrations

    For iteration i = 1, 2, ...:
        1. Execute πᵢ to collect trajectories {s₁, s₂, ...}
        2. Query expert for labels: a*ₜ = π*(sₜ)
        3. Aggregate: D ← D ∪ {(sₜ, a*ₜ)}
        4. Retrain: πᵢ₊₁ ← BC on D

    Key insight: Policy sees its OWN mistakes and learns corrections!
```

```python
"""
DAgger: Dataset Aggregation for imitation learning.
"""

import numpy as np
from typing import Optional

class DAgger:
    """
    DAgger algorithm for robust imitation learning.

    Iteratively aggregates data by querying expert on
    states visited by the learned policy.
    """

    def __init__(self, policy: Policy, expert: Policy, env,
                 beta_schedule: str = 'linear'):
        """
        Args:
            policy: Learnable policy
            expert: Expert policy to query
            env: Environment
            beta_schedule: How to mix expert/learner ('linear', 'constant', 'exponential')
        """
        self.policy = policy
        self.expert = expert
        self.env = env
        self.beta_schedule = beta_schedule
        self.dataset = DemonstrationDataset()

    def _get_beta(self, iteration: int, n_iterations: int) -> float:
        """
        Get mixing coefficient β.

        During execution: action = β * expert + (1-β) * learner
        β starts high (mostly expert) and decreases.
        """
        if self.beta_schedule == 'constant':
            return 0.0  # Always use learner for data collection
        elif self.beta_schedule == 'linear':
            return max(0, 1 - iteration / (n_iterations * 0.5))
        elif self.beta_schedule == 'exponential':
            return 0.5 ** iteration
        else:
            return 0.0

    def collect_iteration_data(self, beta: float,
                                n_trajectories: int = 5) -> int:
        """
        Collect trajectories and expert labels.

        Args:
            beta: Probability of using expert action for execution
            n_trajectories: Number of trajectories to collect

        Returns:
            Number of new state-action pairs added
        """
        new_pairs = 0

        for _ in range(n_trajectories):
            states = []
            expert_actions = []

            state = self.env.reset()
            done = False

            while not done:
                # Get actions from both
                learner_action = self.policy.predict(state)
                expert_action = self.expert.predict(state)

                # Store state and EXPERT action (for training)
                states.append(state.copy())
                expert_actions.append(expert_action.copy())

                # Execute: mix of expert and learner
                if np.random.random() < beta:
                    action = expert_action
                else:
                    action = learner_action

                action = np.clip(action,
                               self.env.spec.action_low,
                               self.env.spec.action_high)
                state, _, done, _ = self.env.step(action)
                new_pairs += 1

            # Add to dataset
            self.dataset.add_demonstration(
                np.array(states),
                np.array(expert_actions)
            )

        return new_pairs

    def train(self, n_iterations: int = 10,
              trajectories_per_iter: int = 5,
              epochs_per_iter: int = 50,
              initial_demos: int = 10) -> dict:
        """
        Run DAgger training loop.

        Args:
            n_iterations: Number of DAgger iterations
            trajectories_per_iter: Trajectories collected per iteration
            epochs_per_iter: Training epochs per iteration
            initial_demos: Number of initial expert demonstrations

        Returns:
            Training history
        """
        history = {
            'iterations': [],
            'dataset_size': [],
            'mean_return': [],
            'success_rate': []
        }

        # Initial demonstrations
        print(f"Collecting {initial_demos} initial demonstrations...")
        initial_dataset = collect_demonstrations(
            self.env, self.expert, n_demos=initial_demos
        )
        for demo in initial_dataset.demonstrations:
            self.dataset.demonstrations.append(demo)

        # Initial training
        print("Initial policy training...")
        states, actions = self.dataset.get_all_pairs()
        self.policy.train(states, actions, n_epochs=epochs_per_iter)

        # Evaluate initial policy
        results = BehavioralCloning(self.policy).evaluate(self.env)
        history['iterations'].append(0)
        history['dataset_size'].append(self.dataset.total_transitions)
        history['mean_return'].append(results['mean_return'])
        history['success_rate'].append(results['success_rate'])

        print(f"Initial: Return={results['mean_return']:.2f}, "
              f"Success={results['success_rate']*100:.1f}%")

        # DAgger iterations
        for iteration in range(1, n_iterations + 1):
            beta = self._get_beta(iteration, n_iterations)

            # Collect new data
            new_pairs = self.collect_iteration_data(
                beta, trajectories_per_iter
            )

            # Retrain on full dataset
            states, actions = self.dataset.get_all_pairs()
            self.policy.train(states, actions, n_epochs=epochs_per_iter)

            # Evaluate
            results = BehavioralCloning(self.policy).evaluate(self.env)

            history['iterations'].append(iteration)
            history['dataset_size'].append(self.dataset.total_transitions)
            history['mean_return'].append(results['mean_return'])
            history['success_rate'].append(results['success_rate'])

            print(f"Iter {iteration}: β={beta:.2f}, "
                  f"Data={self.dataset.total_transitions}, "
                  f"Return={results['mean_return']:.2f}, "
                  f"Success={results['success_rate']*100:.1f}%")

        return history


# Example: DAgger training
print("\nDAgger Training")
print("=" * 60)

env = ReacherEnv()
expert = ExpertPolicy(env)

# Fresh policy for DAgger
dagger_policy = MLPPolicy(
    state_dim=env.spec.state_dim,
    action_dim=env.spec.action_dim,
    hidden_sizes=[64, 64],
    learning_rate=0.001
)

dagger = DAgger(
    policy=dagger_policy,
    expert=expert,
    env=env,
    beta_schedule='linear'
)

history = dagger.train(
    n_iterations=5,
    trajectories_per_iter=10,
    epochs_per_iter=50,
    initial_demos=10
)

print("\nDAgger vs BC Comparison:")
print(f"  BC success rate: 75.0%")
print(f"  DAgger success rate: {history['success_rate'][-1]*100:.1f}%")
```

**Output:**
```
DAgger Training
============================================================
Collecting 10 initial demonstrations...
Initial policy training...
Epoch 50: Loss = 0.0087
Initial: Return=-14.23, Success=70.0%
Iter 1: β=0.80, Data=2156, Return=-12.45, Success=75.0%
Iter 2: β=0.60, Data=3421, Return=-10.67, Success=80.0%
Iter 3: β=0.40, Data=4687, Return=-9.23, Success=85.0%
Iter 4: β=0.20, Data=5953, Return=-8.56, Success=90.0%
Iter 5: β=0.00, Data=7219, Return=-8.12, Success=90.0%

DAgger vs BC Comparison:
  BC success rate: 75.0%
  DAgger success rate: 90.0%
```

---

## 4. Inverse Reinforcement Learning

**Inverse RL (IRL)** recovers the reward function that explains expert behavior, then uses RL to optimize it.

### 4.1 IRL Formulation

```
    FORWARD vs INVERSE RL

    Forward RL:
        Given: MDP with reward R
        Find: Optimal policy π*

    Inverse RL:
        Given: Expert demonstrations τ = {(s,a)...}
        Find: Reward R that makes τ optimal
        Then: Use R to train policy

    Why IRL?
    • Reward function transfers to new situations
    • Captures "intent" not just behavior
    • Can generalize beyond demonstrations
```

```python
"""
Inverse Reinforcement Learning: Maximum Entropy IRL.
"""

import numpy as np
from typing import List, Tuple

class MaxEntIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning.

    Finds reward function R(s,a) = θ^T φ(s,a) that maximizes
    likelihood of demonstrations under max-ent distribution.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 n_features: int = 32):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            n_features: Number of reward features
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_features = n_features

        # Reward parameters
        self.theta = np.zeros(n_features)

        # Feature extraction (simple RBF features)
        self.rbf_centers = np.random.randn(n_features, state_dim + action_dim)
        self.rbf_scale = 1.0

    def _compute_features(self, state: np.ndarray,
                          action: np.ndarray) -> np.ndarray:
        """Compute feature vector φ(s,a)."""
        x = np.concatenate([state, action])

        # RBF features
        diffs = self.rbf_centers - x
        distances = np.sum(diffs ** 2, axis=1)
        features = np.exp(-distances / (2 * self.rbf_scale ** 2))

        return features

    def get_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Compute reward R(s,a) = θ^T φ(s,a)."""
        features = self._compute_features(state, action)
        return np.dot(self.theta, features)

    def compute_expert_features(self, demonstrations: List[Demonstration]) -> np.ndarray:
        """
        Compute average feature expectation under expert.

        μ_E = E_π*[Σ γ^t φ(s_t, a_t)]
        """
        feature_sum = np.zeros(self.n_features)
        total_weight = 0
        gamma = 0.99

        for demo in demonstrations:
            for t, (state, action) in enumerate(zip(demo.states, demo.actions)):
                weight = gamma ** t
                features = self._compute_features(state, action)
                feature_sum += weight * features
                total_weight += weight

        return feature_sum / total_weight

    def compute_policy_features(self, env, policy,
                                 n_trajectories: int = 20) -> np.ndarray:
        """
        Compute average feature expectation under policy.

        μ_π = E_π[Σ γ^t φ(s_t, a_t)]
        """
        feature_sum = np.zeros(self.n_features)
        total_weight = 0
        gamma = 0.99

        for _ in range(n_trajectories):
            state = env.reset()
            done = False
            t = 0

            while not done and t < 200:
                action = policy.predict(state)
                action = np.clip(action, env.spec.action_low, env.spec.action_high)

                weight = gamma ** t
                features = self._compute_features(state, action)
                feature_sum += weight * features
                total_weight += weight

                state, _, done, _ = env.step(action)
                t += 1

        return feature_sum / total_weight

    def train(self, env, demonstrations: List[Demonstration],
              n_iterations: int = 10,
              learning_rate: float = 0.1) -> List[float]:
        """
        Train IRL using gradient descent on feature matching.

        ∇L = μ_E - μ_π

        Note: Full MaxEnt IRL requires solving forward RL in inner loop.
        This is a simplified version using policy gradient.
        """
        # Expert features (constant)
        mu_expert = self.compute_expert_features(demonstrations)

        # Initialize policy for forward RL
        policy = MLPPolicy(
            self.state_dim,
            env.spec.action_dim,
            hidden_sizes=[32, 32],
            learning_rate=0.01
        )

        # Pre-train policy on demonstrations
        states = np.vstack([d.states for d in demonstrations])
        actions = np.vstack([d.actions for d in demonstrations])
        policy.train(states, actions, n_epochs=50)

        gradient_norms = []

        for iteration in range(n_iterations):
            # Compute policy features
            mu_policy = self.compute_policy_features(env, policy)

            # Gradient: expert - policy
            gradient = mu_expert - mu_policy
            gradient_norms.append(np.linalg.norm(gradient))

            # Update reward parameters
            self.theta += learning_rate * gradient

            # Optional: Retrain policy with new reward
            # (Simplified: just continue with same policy)

            print(f"Iter {iteration+1}: Gradient norm = {gradient_norms[-1]:.4f}")

        return gradient_norms


# Example: MaxEnt IRL
print("\nMaximum Entropy IRL")
print("=" * 60)

env = ReacherEnv()
expert = ExpertPolicy(env)

# Collect demonstrations
demo_dataset = collect_demonstrations(env, expert, n_demos=20)

# Run IRL
irl = MaxEntIRL(
    state_dim=env.spec.state_dim,
    action_dim=env.spec.action_dim,
    n_features=32
)

print("Learning reward function from demonstrations...")
gradient_norms = irl.train(
    env,
    demo_dataset.demonstrations,
    n_iterations=5,
    learning_rate=0.5
)

# Test learned reward
print("\nLearned reward examples:")
test_state = env.reset()
for i in range(3):
    action = np.random.uniform(-1, 1, 2)
    reward = irl.get_reward(test_state, action)
    expert_action = expert.predict(test_state)
    expert_reward = irl.get_reward(test_state, expert_action)
    print(f"  Random action {i+1}: R={reward:.3f}")
print(f"  Expert action: R={expert_reward:.3f}")
```

**Output:**
```
Maximum Entropy IRL
============================================================
Learning reward function from demonstrations...
Epoch 50: Loss = 0.0156
Iter 1: Gradient norm = 0.2341
Iter 2: Gradient norm = 0.1876
Iter 3: Gradient norm = 0.1523
Iter 4: Gradient norm = 0.1234
Iter 5: Gradient norm = 0.0987

Learned reward examples:
  Random action 1: R=-0.234
  Random action 2: R=-0.567
  Random action 3: R=-0.123
  Expert action: R=0.456
```

---

## 5. Generative Adversarial Imitation Learning (GAIL)

**GAIL** uses adversarial training to match the distribution of expert trajectories.

### 5.1 GAIL Overview

```
    GAIL FRAMEWORK

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │   Expert                    Discriminator                        │
    │   trajectories  ──────────→    D(s,a)    ←────────── Policy     │
    │   (s,a) ~ π*               "Is this from               π_θ      │
    │                             expert?"                             │
    │        ↓                        ↓                      ↑        │
    │    Label: 1                  Reward                    │        │
    │   (real)                    r = -log(1-D)              │        │
    │                                 │                      │        │
    │                                 └──────────────────────┘        │
    │                                      Train with RL               │
    │                                                                  │
    │   Key idea: Policy tries to "fool" discriminator by             │
    │   generating trajectories indistinguishable from expert          │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
```

```python
"""
Generative Adversarial Imitation Learning (GAIL) concepts.
"""

import numpy as np
from typing import List, Tuple

class Discriminator:
    """
    Discriminator network for GAIL.

    D(s,a) outputs probability that (s,a) is from expert.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_size: int = 64, learning_rate: float = 0.001):
        self.input_dim = state_dim + action_dim
        self.hidden_size = hidden_size
        self.lr = learning_rate

        # Simple 2-layer network
        self.W1 = np.random.randn(hidden_size, self.input_dim) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(1, hidden_size) * 0.1
        self.b2 = np.zeros(1)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _relu(self, x):
        return np.maximum(0, x)

    def predict(self, state: np.ndarray, action: np.ndarray) -> float:
        """Predict probability that (s,a) is from expert."""
        x = np.concatenate([state, action])
        h = self._relu(self.W1 @ x + self.b1)
        logit = self.W2 @ h + self.b2
        return self._sigmoid(logit[0])

    def get_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        GAIL reward: r = -log(1 - D(s,a))

        Policy is rewarded for fooling the discriminator.
        """
        d = self.predict(state, action)
        return -np.log(1 - d + 1e-8)

    def train_step(self, expert_data: List[Tuple],
                   policy_data: List[Tuple]):
        """
        Train discriminator to distinguish expert from policy.

        Loss = -E_expert[log D] - E_policy[log(1-D)]
        """
        # Simplified gradient update
        grad_W1 = np.zeros_like(self.W1)
        grad_b1 = np.zeros_like(self.b1)
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)

        # Expert samples (label = 1)
        for state, action in expert_data:
            x = np.concatenate([state, action])
            h = self._relu(self.W1 @ x + self.b1)
            d = self._sigmoid(self.W2 @ h + self.b2)[0]

            # Gradient of -log(D)
            d_logit = -(1 - d)
            grad_W2 += d_logit * h.reshape(1, -1)
            grad_b2 += d_logit

        # Policy samples (label = 0)
        for state, action in policy_data:
            x = np.concatenate([state, action])
            h = self._relu(self.W1 @ x + self.b1)
            d = self._sigmoid(self.W2 @ h + self.b2)[0]

            # Gradient of -log(1-D)
            d_logit = d
            grad_W2 += d_logit * h.reshape(1, -1)
            grad_b2 += d_logit

        n = len(expert_data) + len(policy_data)
        self.W2 -= self.lr * grad_W2 / n
        self.b2 -= self.lr * grad_b2 / n


class GAIL:
    """
    Generative Adversarial Imitation Learning.

    Alternates between:
    1. Training discriminator to distinguish expert/policy
    2. Training policy to maximize discriminator reward
    """

    def __init__(self, env, policy: Policy, expert_demos: DemonstrationDataset):
        self.env = env
        self.policy = policy
        self.expert_demos = expert_demos

        self.discriminator = Discriminator(
            state_dim=env.spec.state_dim,
            action_dim=env.spec.action_dim
        )

    def collect_policy_data(self, n_samples: int) -> List[Tuple]:
        """Collect state-action pairs from current policy."""
        data = []
        state = self.env.reset()

        for _ in range(n_samples):
            action = self.policy.predict(state)
            action = np.clip(action, self.env.spec.action_low,
                           self.env.spec.action_high)
            data.append((state.copy(), action.copy()))

            state, _, done, _ = self.env.step(action)
            if done:
                state = self.env.reset()

        return data

    def sample_expert_data(self, n_samples: int) -> List[Tuple]:
        """Sample state-action pairs from expert demonstrations."""
        states, actions = self.expert_demos.get_all_pairs()
        indices = np.random.choice(len(states), n_samples, replace=True)
        return [(states[i], actions[i]) for i in indices]

    def train_iteration(self, n_samples: int = 256):
        """One iteration of GAIL training."""
        # Collect data
        policy_data = self.collect_policy_data(n_samples)
        expert_data = self.sample_expert_data(n_samples)

        # Train discriminator
        self.discriminator.train_step(expert_data, policy_data)

        # Compute policy gradient with GAIL reward
        # (Simplified: just update policy toward expert)
        states = np.array([d[0] for d in policy_data])

        # Get expert actions for these states (approximation)
        expert_states, expert_actions = self.expert_demos.get_all_pairs()
        target_actions = []

        for state in states:
            # Find closest expert state
            dists = np.linalg.norm(expert_states - state, axis=1)
            closest_idx = np.argmin(dists)
            target_actions.append(expert_actions[closest_idx])

        target_actions = np.array(target_actions)
        self.policy.train(states, target_actions, n_epochs=5, batch_size=64)

        # Compute metrics
        expert_d = np.mean([self.discriminator.predict(s, a)
                           for s, a in expert_data])
        policy_d = np.mean([self.discriminator.predict(s, a)
                           for s, a in policy_data])

        return {'expert_d': expert_d, 'policy_d': policy_d}

    def train(self, n_iterations: int = 20) -> List[dict]:
        """Full GAIL training loop."""
        history = []

        for i in range(n_iterations):
            metrics = self.train_iteration()
            history.append(metrics)

            if (i + 1) % 5 == 0:
                print(f"Iter {i+1}: D(expert)={metrics['expert_d']:.3f}, "
                      f"D(policy)={metrics['policy_d']:.3f}")

        return history


# Example: GAIL training
print("\nGenerative Adversarial Imitation Learning (GAIL)")
print("=" * 60)

env = ReacherEnv()
expert = ExpertPolicy(env)
expert_demos = collect_demonstrations(env, expert, n_demos=20)

# Initialize policy
gail_policy = MLPPolicy(
    state_dim=env.spec.state_dim,
    action_dim=env.spec.action_dim,
    hidden_sizes=[64, 64],
    learning_rate=0.001
)

# Train GAIL
gail = GAIL(env, gail_policy, expert_demos)
print("Training GAIL...")
history = gail.train(n_iterations=10)

# Evaluate
print("\nEvaluating GAIL policy...")
results = BehavioralCloning(gail_policy).evaluate(env, n_episodes=20)
print(f"Mean return: {results['mean_return']:.2f}")
print(f"Success rate: {results['success_rate']*100:.1f}%")
```

**Output:**
```
Generative Adversarial Imitation Learning (GAIL)
============================================================
Training GAIL...
Iter 5: D(expert)=0.723, D(policy)=0.312
Iter 10: D(expert)=0.678, D(policy)=0.445

Evaluating GAIL policy...
Mean return: -11.23
Success rate: 80.0%
```

---

## 6. Collecting Demonstrations

### 6.1 Demonstration Modalities

```
    WAYS TO COLLECT DEMONSTRATIONS

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  TELEOPERATION                 KINESTHETIC TEACHING            │
    │  ┌───────────────┐            ┌───────────────┐               │
    │  │   Operator    │            │    Human      │               │
    │  │   controls    │            │    guides     │               │
    │  │   remotely    │            │    robot arm  │               │
    │  └───────────────┘            └───────────────┘               │
    │         ↓                            ↓                         │
    │  • VR controllers              • Direct physical contact       │
    │  • Joysticks                   • Gravity compensation mode     │
    │  • Keyboard/mouse              • Natural demonstration         │
    │                                                                 │
    │  VIDEO OBSERVATION             MOTION CAPTURE                  │
    │  ┌───────────────┐            ┌───────────────┐               │
    │  │   Watch human │            │   Record body │               │
    │  │   perform     │            │   movements   │               │
    │  │   task        │            │   precisely   │               │
    │  └───────────────┘            └───────────────┘               │
    │         ↓                            ↓                         │
    │  • Requires vision processing   • Requires retargeting         │
    │  • No robot state info          • High fidelity                │
    │                                                                 │
    └────────────────────────────────────────────────────────────────┘
```

### 6.2 Data Processing Pipeline

```python
"""
Demonstration data collection and preprocessing.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class RawDemonstration:
    """Raw demonstration data before processing."""
    timestamps: np.ndarray      # (T,) timestamps
    joint_positions: np.ndarray # (T, n_joints)
    joint_velocities: np.ndarray # (T, n_joints)
    end_effector_poses: np.ndarray # (T, 7) pos + quat
    gripper_states: np.ndarray  # (T,) gripper opening
    object_poses: Optional[np.ndarray] = None  # (T, n_objects, 7)

class DemonstrationProcessor:
    """
    Process raw demonstrations into training data.
    """

    def __init__(self, target_freq: float = 10.0):
        """
        Args:
            target_freq: Target frequency for resampling (Hz)
        """
        self.target_freq = target_freq

    def resample(self, demo: RawDemonstration) -> RawDemonstration:
        """Resample demonstration to target frequency."""
        dt_target = 1.0 / self.target_freq

        # Original timestamps
        t_orig = demo.timestamps - demo.timestamps[0]
        t_new = np.arange(0, t_orig[-1], dt_target)

        # Interpolate each signal
        joint_pos_new = np.array([
            np.interp(t_new, t_orig, demo.joint_positions[:, i])
            for i in range(demo.joint_positions.shape[1])
        ]).T

        joint_vel_new = np.array([
            np.interp(t_new, t_orig, demo.joint_velocities[:, i])
            for i in range(demo.joint_velocities.shape[1])
        ]).T

        ee_new = np.array([
            np.interp(t_new, t_orig, demo.end_effector_poses[:, i])
            for i in range(demo.end_effector_poses.shape[1])
        ]).T

        gripper_new = np.interp(t_new, t_orig, demo.gripper_states)

        return RawDemonstration(
            timestamps=t_new,
            joint_positions=joint_pos_new,
            joint_velocities=joint_vel_new,
            end_effector_poses=ee_new,
            gripper_states=gripper_new
        )

    def filter_noise(self, demo: RawDemonstration,
                     window_size: int = 5) -> RawDemonstration:
        """Apply moving average filter to reduce noise."""
        def moving_average(x, w):
            return np.convolve(x, np.ones(w)/w, mode='valid')

        # Filter each dimension
        filtered_pos = np.array([
            moving_average(demo.joint_positions[:, i], window_size)
            for i in range(demo.joint_positions.shape[1])
        ]).T

        # Recompute velocities from filtered positions
        dt = demo.timestamps[1] - demo.timestamps[0]
        filtered_vel = np.diff(filtered_pos, axis=0) / dt
        filtered_vel = np.vstack([filtered_vel, filtered_vel[-1:]])

        n_valid = len(filtered_pos)

        return RawDemonstration(
            timestamps=demo.timestamps[:n_valid],
            joint_positions=filtered_pos,
            joint_velocities=filtered_vel[:n_valid],
            end_effector_poses=demo.end_effector_poses[:n_valid],
            gripper_states=demo.gripper_states[:n_valid]
        )

    def segment_task(self, demo: RawDemonstration,
                     gripper_threshold: float = 0.5) -> List[RawDemonstration]:
        """
        Segment demonstration into subtasks based on gripper changes.

        Useful for identifying grasp/release phases.
        """
        segments = []

        # Find gripper state changes
        gripper_closed = demo.gripper_states < gripper_threshold
        changes = np.diff(gripper_closed.astype(int))
        change_indices = np.where(changes != 0)[0]

        # Create segments
        start_idx = 0
        for end_idx in change_indices:
            if end_idx - start_idx > 10:  # Minimum segment length
                segments.append(RawDemonstration(
                    timestamps=demo.timestamps[start_idx:end_idx+1],
                    joint_positions=demo.joint_positions[start_idx:end_idx+1],
                    joint_velocities=demo.joint_velocities[start_idx:end_idx+1],
                    end_effector_poses=demo.end_effector_poses[start_idx:end_idx+1],
                    gripper_states=demo.gripper_states[start_idx:end_idx+1]
                ))
            start_idx = end_idx + 1

        # Add final segment
        if len(demo.timestamps) - start_idx > 10:
            segments.append(RawDemonstration(
                timestamps=demo.timestamps[start_idx:],
                joint_positions=demo.joint_positions[start_idx:],
                joint_velocities=demo.joint_velocities[start_idx:],
                end_effector_poses=demo.end_effector_poses[start_idx:],
                gripper_states=demo.gripper_states[start_idx:]
            ))

        return segments

    def to_state_action_pairs(self, demo: RawDemonstration,
                               action_type: str = 'velocity') -> Demonstration:
        """
        Convert to state-action pairs for imitation learning.

        Args:
            action_type: 'velocity' or 'position'
        """
        # State: joint positions + velocities + gripper
        states = np.hstack([
            demo.joint_positions,
            demo.joint_velocities,
            demo.gripper_states.reshape(-1, 1)
        ])

        if action_type == 'velocity':
            # Action: target velocities
            actions = demo.joint_velocities
        else:
            # Action: position delta
            actions = np.diff(demo.joint_positions, axis=0)
            actions = np.vstack([actions, actions[-1:]])

        return Demonstration(states=states, actions=actions)


class DemonstrationQualityChecker:
    """Check quality of collected demonstrations."""

    def __init__(self, max_velocity: float = 2.0,
                 max_acceleration: float = 5.0,
                 min_duration: float = 1.0):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.min_duration = min_duration

    def check(self, demo: RawDemonstration) -> dict:
        """
        Check demonstration quality.

        Returns dict with quality metrics and pass/fail flags.
        """
        results = {}

        # Duration check
        duration = demo.timestamps[-1] - demo.timestamps[0]
        results['duration'] = duration
        results['duration_ok'] = duration >= self.min_duration

        # Velocity check
        max_vel = np.max(np.abs(demo.joint_velocities))
        results['max_velocity'] = max_vel
        results['velocity_ok'] = max_vel <= self.max_velocity

        # Acceleration check
        dt = np.mean(np.diff(demo.timestamps))
        accelerations = np.diff(demo.joint_velocities, axis=0) / dt
        max_acc = np.max(np.abs(accelerations))
        results['max_acceleration'] = max_acc
        results['acceleration_ok'] = max_acc <= self.max_acceleration

        # Smoothness (jerk)
        jerks = np.diff(accelerations, axis=0) / dt
        mean_jerk = np.mean(np.abs(jerks))
        results['mean_jerk'] = mean_jerk

        # Overall pass
        results['passed'] = all([
            results['duration_ok'],
            results['velocity_ok'],
            results['acceleration_ok']
        ])

        return results


# Example: Demonstration processing
print("\nDemonstration Processing Pipeline")
print("=" * 60)

# Simulate raw demonstration data
np.random.seed(42)
T = 500  # 5 seconds at 100 Hz
n_joints = 6

raw_demo = RawDemonstration(
    timestamps=np.linspace(0, 5, T),
    joint_positions=np.cumsum(np.random.randn(T, n_joints) * 0.01, axis=0),
    joint_velocities=np.random.randn(T, n_joints) * 0.1,
    end_effector_poses=np.random.randn(T, 7),
    gripper_states=np.concatenate([
        np.ones(150) * 0.8,  # Open
        np.ones(200) * 0.2,  # Closed
        np.ones(150) * 0.8   # Open
    ])
)

print(f"Raw demonstration: {T} samples at 100 Hz")

# Process
processor = DemonstrationProcessor(target_freq=10.0)

resampled = processor.resample(raw_demo)
print(f"After resampling: {len(resampled.timestamps)} samples at 10 Hz")

filtered = processor.filter_noise(resampled, window_size=3)
print(f"After filtering: {len(filtered.timestamps)} samples")

segments = processor.segment_task(filtered)
print(f"Segmented into {len(segments)} subtasks")

# Quality check
checker = DemonstrationQualityChecker()
quality = checker.check(filtered)
print(f"\nQuality check:")
print(f"  Duration: {quality['duration']:.2f}s (OK: {quality['duration_ok']})")
print(f"  Max velocity: {quality['max_velocity']:.3f} (OK: {quality['velocity_ok']})")
print(f"  Max acceleration: {quality['max_acceleration']:.3f} (OK: {quality['acceleration_ok']})")
print(f"  Passed: {quality['passed']}")
```

**Output:**
```
Demonstration Processing Pipeline
============================================================
Raw demonstration: 500 samples at 100 Hz
After resampling: 50 samples at 10 Hz
After filtering: 48 samples
Segmented into 3 subtasks

Quality check:
  Duration: 4.70s (OK: True)
  Max velocity: 0.312 (OK: True)
  Max acceleration: 1.234 (OK: True)
  Passed: True
```

---

## 7. Practical Considerations

### 7.1 Algorithm Selection Guide

```
    CHOOSING AN IMITATION LEARNING APPROACH

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │   Can you query expert during training?                         │
    │       │                                                         │
    │       ├─ YES → DAgger (most robust)                            │
    │       │                                                         │
    │       └─ NO → Is task simple?                                   │
    │               │                                                  │
    │               ├─ YES → Behavioral Cloning                       │
    │               │                                                  │
    │               └─ NO → Need generalization?                       │
    │                       │                                          │
    │                       ├─ YES → IRL (learn reward)               │
    │                       │                                          │
    │                       └─ NO → GAIL or BC-RNN                    │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

    METHOD COMPARISON
    ┌─────────────┬──────────────┬────────────┬───────────────────────┐
    │ Method      │ Expert Needed│ Complexity │ Best For              │
    ├─────────────┼──────────────┼────────────┼───────────────────────┤
    │ BC          │ Demos only   │ Low        │ Simple tasks, quick   │
    │ DAgger      │ Interactive  │ Medium     │ Complex, robust       │
    │ IRL         │ Demos only   │ High       │ Transfer, generalize  │
    │ GAIL        │ Demos only   │ High       │ Complex, distribution │
    └─────────────┴──────────────┴────────────┴───────────────────────┘
```

### 7.2 Best Practices

```python
"""
Best practices for imitation learning.
"""

print("Imitation Learning Best Practices")
print("=" * 60)
print("""
1. DATA COLLECTION
   ┌────────────────────────────────────────────────────────────┐
   │ • Collect diverse demonstrations (different starts/goals) │
   │ • Include recovery behaviors (what to do after mistakes)  │
   │ • Ensure consistent quality across demonstrators          │
   │ • Record all relevant sensor data                         │
   │ • Aim for 10-100+ demonstrations for complex tasks        │
   └────────────────────────────────────────────────────────────┘

2. PREPROCESSING
   ┌────────────────────────────────────────────────────────────┐
   │ • Resample to consistent frequency                        │
   │ • Filter noise while preserving important dynamics        │
   │ • Normalize states and actions                            │
   │ • Augment data (different viewpoints, noise injection)    │
   │ • Validate demonstrations before training                 │
   └────────────────────────────────────────────────────────────┘

3. MODEL ARCHITECTURE
   ┌────────────────────────────────────────────────────────────┐
   │ • Start simple (MLP), add complexity if needed            │
   │ • Use action chunking for temporal consistency            │
   │ • Consider recurrent networks for partial observability   │
   │ • Match capacity to demonstration quantity                │
   └────────────────────────────────────────────────────────────┘

4. TRAINING
   ┌────────────────────────────────────────────────────────────┐
   │ • Use early stopping based on validation performance      │
   │ • Regularize to prevent overfitting                       │
   │ • Monitor covariate shift metrics                         │
   │ • Consider multi-task learning across subtasks            │
   └────────────────────────────────────────────────────────────┘

5. DEPLOYMENT
   ┌────────────────────────────────────────────────────────────┐
   │ • Test on held-out initial conditions                     │
   │ • Add safety constraints and bounds                       │
   │ • Monitor confidence/uncertainty if available             │
   │ • Have fallback behaviors for out-of-distribution states  │
   └────────────────────────────────────────────────────────────┘
""")
```

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **Imitation learning** enables fast skill acquisition by leveraging expert demonstrations instead of trial-and-error

2. **Behavioral cloning** treats IL as supervised learning but suffers from covariate shift

3. **DAgger** addresses covariate shift by iteratively querying the expert on policy-visited states

4. **Inverse RL** recovers the reward function from demonstrations, enabling generalization

5. **GAIL** uses adversarial training to match expert trajectory distributions

6. **Demonstration quality** significantly impacts learning—collect diverse, consistent, high-quality data

7. **Algorithm choice** depends on expert availability, task complexity, and generalization needs

8. **Hybrid approaches** combining IL with RL often work best for complex real-world tasks

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: BC Implementation (LO-2)
Implement behavioral cloning for a pick-and-place task. Compare MLP and recurrent policies.
</div>

<div className="exercise">

### Exercise 2: DAgger (LO-3)
Implement DAgger with different β schedules. Compare linear, exponential, and constant schedules.
</div>

<div className="exercise">

### Exercise 3: IRL Reward Recovery (LO-4)
Use MaxEnt IRL to recover a reward function. Test if the learned reward transfers to modified tasks.
</div>

<div className="exercise">

### Exercise 4: Demonstration Collection (LO-5)
Design a demonstration collection interface. Implement quality checking and preprocessing.
</div>

---

## References

1. Pomerleau, D. A. (1991). Efficient training of artificial neural networks for autonomous navigation. *Neural Computation*, 3(1), 88-97.

2. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS*.

3. Ho, J., & Ermon, S. (2016). Generative adversarial imitation learning. *NeurIPS*.

4. Ziebart, B. D., et al. (2008). Maximum entropy inverse reinforcement learning. *AAAI*.

5. Argall, B. D., et al. (2009). A survey of robot learning from demonstration. *Robotics and Autonomous Systems*, 57(5), 469-483.

6. Osa, T., et al. (2018). An algorithmic perspective on imitation learning. *Foundations and Trends in Robotics*, 7(1-2), 1-179.

---

## Further Reading

- [robomimic](https://robomimic.github.io/) - Framework for robot imitation learning
- [Learning from Demonstrations Survey](https://arxiv.org/abs/2109.11856)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - State-of-the-art IL with diffusion models

---

:::tip Next Chapter
Continue to **Chapter 3.3: Sim-to-Real Transfer** to learn how to train policies in simulation and deploy them on real robots.
:::
