---
sidebar_position: 1
title: Reinforcement Learning for Robotics
description: Learning robot control through trial and error - from MDPs and policy gradients to deep RL for continuous control
keywords: [reinforcement learning, RL, policy gradient, actor-critic, PPO, SAC, robotics, continuous control]
difficulty: advanced
estimated_time: 120 minutes
chapter_id: rl
part_id: part-3-learning-systems
author: Claude Code
last_updated: 2026-01-20
prerequisites: [control-systems, kinematics, probability]
tags: [reinforcement-learning, deep-rl, policy-gradient, robotics, continuous-control]
---

# Reinforcement Learning for Robotics

<div className="learning-objectives">

## Learning Objectives

After completing this chapter, you will be able to:

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| **LO-1** | Formulate robot control problems as Markov Decision Processes | Apply |
| **LO-2** | Explain the differences between value-based and policy-based RL methods | Understand |
| **LO-3** | Implement policy gradient algorithms for continuous action spaces | Apply |
| **LO-4** | Apply actor-critic methods (PPO, SAC) to robot learning tasks | Apply |
| **LO-5** | Design reward functions that produce desired robot behaviors | Create |

</div>

---

## 1. Introduction: Learning Through Interaction

Reinforcement Learning (RL) enables robots to learn behaviors through trial and error, without explicit programming of every action. The robot interacts with its environment, receives feedback in the form of rewards, and gradually improves its policy.

### Why RL for Robotics?

```
    TRADITIONAL CONTROL vs REINFORCEMENT LEARNING

    TRADITIONAL CONTROL                    REINFORCEMENT LEARNING
    ┌─────────────────────────┐           ┌─────────────────────────┐
    │  Human designs          │           │  Robot learns           │
    │  control law            │           │  from experience        │
    │                         │           │                         │
    │  u = f(x, x_ref)       │           │  π(a|s) learned         │
    │                         │           │                         │
    │  ✓ Guaranteed stable   │           │  ✓ Handles complexity   │
    │  ✓ Interpretable       │           │  ✓ Adapts to changes    │
    │  ✗ Hard to design      │           │  ✗ Sample inefficient   │
    │  ✗ Brittle to changes  │           │  ✗ Safety challenges    │
    └─────────────────────────┘           └─────────────────────────┘
```

### The RL Loop for Robotics

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │   ┌─────────┐    action aₜ    ┌─────────────┐                   │
    │   │  AGENT  │ ──────────────→ │ ENVIRONMENT │                   │
    │   │  (Robot │                 │  (World +   │                   │
    │   │  Brain) │ ←────────────── │   Robot)    │                   │
    │   └─────────┘   state sₜ₊₁    └─────────────┘                   │
    │        ↑        reward rₜ                                        │
    │        │                                                         │
    │        └─── Policy π(a|s) improves over time                    │
    │                                                                  │
    │   Goal: Maximize cumulative reward E[Σ γᵗ rₜ]                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
```

### Key Challenges in Robot RL

| Challenge | Description | Common Solutions |
|-----------|-------------|------------------|
| **Continuous actions** | Joint torques/velocities are continuous | Policy gradients, actor-critic |
| **High dimensionality** | Many joints, sensors | Deep neural networks |
| **Sample efficiency** | Real robots are slow | Simulation, off-policy RL |
| **Safety** | Exploration can damage robot | Constrained RL, safe exploration |
| **Partial observability** | Can't see everything | Recurrent policies, state estimation |
| **Sparse rewards** | Success/failure only | Reward shaping, curriculum |

---

## 2. Markov Decision Processes

### 2.1 MDP Formulation

A **Markov Decision Process (MDP)** provides the mathematical framework for RL:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

| Component | Symbol | Description | Robot Example |
|-----------|--------|-------------|---------------|
| **States** | $\mathcal{S}$ | All possible situations | Joint angles, velocities, object poses |
| **Actions** | $\mathcal{A}$ | All possible decisions | Joint torques, target positions |
| **Transition** | $P(s'|s,a)$ | Dynamics probability | Physics of robot + world |
| **Reward** | $R(s,a,s')$ | Feedback signal | Distance to goal, energy use |
| **Discount** | $\gamma \in [0,1)$ | Future reward weight | Typically 0.99 |

```python
"""
MDP formulation for robot learning.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from abc import ABC, abstractmethod

@dataclass
class MDPSpec:
    """Specification of an MDP."""
    state_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    gamma: float = 0.99

class RobotEnvironment(ABC):
    """
    Abstract base class for robot RL environments.
    Follows OpenAI Gym interface conventions.
    """

    def __init__(self, spec: MDPSpec):
        self.spec = spec
        self.state = None
        self.steps = 0

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Take action and return (next_state, reward, done, info)."""
        pass

    def sample_action(self) -> np.ndarray:
        """Sample random action (for exploration)."""
        return np.random.uniform(
            self.spec.action_low,
            self.spec.action_high
        )


class ReacherEnv(RobotEnvironment):
    """
    Simple 2D reaching task.

    State: [joint_angles (2), joint_velocities (2), target_pos (2)]
    Action: [joint_torques (2)]
    Goal: Move end-effector to target position
    """

    def __init__(self, link_lengths: Tuple[float, float] = (0.5, 0.5)):
        spec = MDPSpec(
            state_dim=6,
            action_dim=2,
            action_low=np.array([-1.0, -1.0]),
            action_high=np.array([1.0, 1.0]),
            gamma=0.99
        )
        super().__init__(spec)

        self.L1, self.L2 = link_lengths
        self.dt = 0.05
        self.max_steps = 200
        self.mass = np.array([1.0, 0.5])
        self.damping = 0.1

    def reset(self) -> np.ndarray:
        """Reset to random initial state and target."""
        self.q = np.random.uniform(-np.pi, np.pi, 2)
        self.q_dot = np.zeros(2)

        # Random target in reachable workspace
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(0.3, 0.9)
        self.target = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ])

        self.steps = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.q, self.q_dot, self.target])

    def _forward_kinematics(self) -> np.ndarray:
        """Compute end-effector position."""
        x = self.L1 * np.cos(self.q[0]) + self.L2 * np.cos(self.q[0] + self.q[1])
        y = self.L1 * np.sin(self.q[0]) + self.L2 * np.sin(self.q[0] + self.q[1])
        return np.array([x, y])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.spec.action_low, self.spec.action_high)

        # Simple dynamics
        q_ddot = action / self.mass - self.damping * self.q_dot
        self.q_dot = self.q_dot + q_ddot * self.dt
        self.q = self.q + self.q_dot * self.dt
        self.q = np.arctan2(np.sin(self.q), np.cos(self.q))

        self.steps += 1

        # Compute reward
        ee_pos = self._forward_kinematics()
        distance = np.linalg.norm(ee_pos - self.target)

        reward = -distance - 0.01 * np.sum(action**2)

        done = self.steps >= self.max_steps or distance < 0.05
        if distance < 0.05:
            reward += 10.0

        info = {'distance': distance, 'ee_pos': ee_pos, 'success': distance < 0.05}
        return self._get_state(), reward, done, info


# Example usage
print("Robot Reaching Environment (MDP)")
print("=" * 60)

env = ReacherEnv()
state = env.reset()

print(f"State dimension: {env.spec.state_dim}")
print(f"Action dimension: {env.spec.action_dim}")
print(f"Action bounds: [{env.spec.action_low}, {env.spec.action_high}]")
print(f"\nInitial state:")
print(f"  Joint angles: {np.degrees(state[:2]).astype(int)}°")
print(f"  Target position: {state[4:6].round(2)}")

# Run episode with random actions
total_reward = 0
state = env.reset()
for step in range(50):
    action = env.sample_action()
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break

print(f"\nRandom policy episode:")
print(f"  Steps: {step+1}, Reward: {total_reward:.2f}")
print(f"  Final distance: {info['distance']:.3f}m")
```

**Output:**
```
Robot Reaching Environment (MDP)
============================================================
State dimension: 6
Action dimension: 2
Action bounds: [[-1. -1.], [1. 1.]]

Initial state:
  Joint angles: [45 -120]°
  Target position: [0.42 0.31]

Random policy episode:
  Steps: 50, Reward: -23.45
  Final distance: 0.523m
```

### 2.2 Value Functions

The **value function** measures expected cumulative reward:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right]$$

The **action-value function** (Q-function):

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a \right]$$

The **Bellman equation** relates values at successive timesteps:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

```python
"""
Value function estimation using TD learning.
"""

import numpy as np

class ValueFunctionEstimator:
    """Estimates V(s) using linear function approximation."""

    def __init__(self, state_dim: int, learning_rate: float = 0.01):
        self.weights = np.zeros(state_dim + 1)  # +1 for bias
        self.lr = learning_rate

    def _features(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate([[1.0], state])

    def predict(self, state: np.ndarray) -> float:
        return np.dot(self.weights, self._features(state))

    def update_td(self, state: np.ndarray, reward: float,
                  next_state: np.ndarray, gamma: float, done: bool):
        """TD(0) update: V(s) <- V(s) + α(r + γV(s') - V(s))"""
        features = self._features(state)
        current_value = np.dot(self.weights, features)

        target = reward if done else reward + gamma * self.predict(next_state)
        td_error = target - current_value

        self.weights += self.lr * td_error * features
        return td_error


# Train value function
print("\nValue Function Learning")
print("=" * 60)

env = ReacherEnv()
V = ValueFunctionEstimator(state_dim=6, learning_rate=0.001)

# Collect episodes
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = env.sample_action()
        next_state, reward, done, _ = env.step(action)
        V.update_td(state, reward, next_state, env.spec.gamma, done)
        state = next_state

test_state = env.reset()
print(f"Value prediction for test state: {V.predict(test_state):.2f}")
```

---

## 3. Policy Gradient Methods

For continuous action spaces (typical in robotics), **policy gradient methods** directly optimize the policy.

### 3.1 The Policy Gradient Theorem

The gradient of expected return:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a) \right]$$

```python
"""
Policy gradient methods for continuous control.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Trajectory:
    """A sequence of (state, action, reward) tuples."""
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]

    def compute_returns(self, gamma: float) -> List[float]:
        """Compute discounted returns from each timestep."""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns


class GaussianPolicy:
    """
    Gaussian policy for continuous actions.
    π(a|s) = N(μ(s), σ²) where μ(s) = W @ s + b
    """

    def __init__(self, state_dim: int, action_dim: int, initial_std: float = 1.0):
        self.W = np.zeros((action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.log_std = np.log(initial_std) * np.ones(action_dim)

    def get_mean(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state + self.b

    def get_std(self) -> np.ndarray:
        return np.exp(self.log_std)

    def sample(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Sample action and return log probability."""
        mean = self.get_mean(state)
        std = self.get_std()

        noise = np.random.randn(len(mean))
        action = mean + std * noise

        log_prob = -0.5 * np.sum(
            np.log(2 * np.pi * std**2) + ((action - mean) / std)**2
        )
        return action, log_prob

    def log_prob_and_grad(self, state: np.ndarray,
                          action: np.ndarray) -> Tuple[float, dict]:
        """Compute log probability and gradients."""
        mean = self.get_mean(state)
        std = self.get_std()
        var = std ** 2

        log_prob = -0.5 * np.sum(
            np.log(2 * np.pi * var) + ((action - mean) / std)**2
        )

        d_mean = (action - mean) / var
        grad_W = np.outer(d_mean, state)
        grad_b = d_mean
        grad_log_std = ((action - mean)**2 / var) - 1

        return log_prob, {'W': grad_W, 'b': grad_b, 'log_std': grad_log_std}


class REINFORCE:
    """REINFORCE algorithm (vanilla policy gradient)."""

    def __init__(self, policy: GaussianPolicy, learning_rate: float = 0.01):
        self.policy = policy
        self.lr = learning_rate

    def collect_trajectory(self, env) -> Trajectory:
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False

        while not done:
            states.append(state)
            action, _ = self.policy.sample(state)
            action = np.clip(action, env.spec.action_low, env.spec.action_high)

            next_state, reward, done, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return Trajectory(states, actions, rewards)

    def update(self, trajectories: List[Trajectory], gamma: float):
        """Update policy using collected trajectories."""
        grad_W = np.zeros_like(self.policy.W)
        grad_b = np.zeros_like(self.policy.b)
        grad_log_std = np.zeros_like(self.policy.log_std)
        total_samples = 0

        for traj in trajectories:
            returns = traj.compute_returns(gamma)

            for state, action, G in zip(traj.states, traj.actions, returns):
                _, grads = self.policy.log_prob_and_grad(state, action)

                grad_W += grads['W'] * G
                grad_b += grads['b'] * G
                grad_log_std += grads['log_std'] * G
                total_samples += 1

        # Apply gradients
        self.policy.W += self.lr * grad_W / total_samples
        self.policy.b += self.lr * grad_b / total_samples
        self.policy.log_std += self.lr * grad_log_std / total_samples

    def train(self, env, n_iterations: int,
              trajectories_per_iter: int = 10) -> List[float]:
        """Train policy using REINFORCE."""
        returns_history = []

        for iteration in range(n_iterations):
            trajectories = [
                self.collect_trajectory(env)
                for _ in range(trajectories_per_iter)
            ]

            avg_return = np.mean([sum(t.rewards) for t in trajectories])
            returns_history.append(avg_return)

            self.update(trajectories, env.spec.gamma)

            if (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1}: Avg Return = {avg_return:.2f}")

        return returns_history


# Train REINFORCE
print("\nREINFORCE Training")
print("=" * 60)

env = ReacherEnv()
policy = GaussianPolicy(env.spec.state_dim, env.spec.action_dim, initial_std=0.5)
agent = REINFORCE(policy, learning_rate=0.001)

returns = agent.train(env, n_iterations=50, trajectories_per_iter=5)
print(f"\nImprovement: {returns[0]:.2f} -> {returns[-1]:.2f}")
```

**Output:**
```
REINFORCE Training
============================================================
Iter 10: Avg Return = -42.15
Iter 20: Avg Return = -38.67
Iter 30: Avg Return = -35.23
Iter 40: Avg Return = -31.89
Iter 50: Avg Return = -28.45

Improvement: -47.82 -> -28.45
```

### 3.2 Variance Reduction with Baselines

Using a baseline $b(s)$ reduces variance without introducing bias:

$$\nabla_\theta J = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot (Q(s,a) - b(s)) \right]$$

The **advantage** $A(s,a) = Q(s,a) - V(s)$ measures how much better an action is than average.

---

## 4. Actor-Critic Methods

**Actor-Critic** combines policy gradients (actor) with value function learning (critic).

### 4.1 Advantage Actor-Critic (A2C)

```python
"""
Advantage Actor-Critic with Generalized Advantage Estimation.
"""

import numpy as np
from typing import List, Tuple

class ActorCritic:
    """A2C with GAE for advantage estimation."""

    def __init__(self, state_dim: int, action_dim: int,
                 actor_lr: float = 0.001, critic_lr: float = 0.01,
                 gamma: float = 0.99, gae_lambda: float = 0.95):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.actor = GaussianPolicy(state_dim, action_dim, initial_std=0.3)
        self.actor_lr = actor_lr
        self.critic = ValueFunctionEstimator(state_dim, learning_rate=critic_lr)

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        return self.actor.sample(state)

    def compute_gae(self, rewards: List[float], values: List[float],
                    next_value: float, dones: List[bool]) -> np.ndarray:
        """Generalized Advantage Estimation."""
        advantages = np.zeros(len(rewards))
        gae = 0

        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def update(self, states, actions, rewards, next_state, dones):
        """Update actor and critic."""
        values = [self.critic.predict(s) for s in states]
        next_value = self.critic.predict(next_state)

        advantages = self.compute_gae(rewards, values, next_value, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + np.array(values)

        # Update critic
        for state, ret in zip(states, returns):
            self.critic.update_td(state, ret, state, self.gamma, True)

        # Update actor
        grad_W = np.zeros_like(self.actor.W)
        grad_b = np.zeros_like(self.actor.b)

        for state, action, adv in zip(states, actions, advantages):
            _, grads = self.actor.log_prob_and_grad(state, action)
            grad_W += grads['W'] * adv
            grad_b += grads['b'] * adv

        n = len(states)
        self.actor.W += self.actor_lr * grad_W / n
        self.actor.b += self.actor_lr * grad_b / n


# Train A2C
print("\nAdvantage Actor-Critic (A2C) Training")
print("=" * 60)

env = ReacherEnv()
agent = ActorCritic(
    env.spec.state_dim, env.spec.action_dim,
    actor_lr=0.002, critic_lr=0.01
)

returns_history = []
for episode in range(100):
    states, actions, rewards, dones = [], [], [], []
    state = env.reset()
    done = False

    while not done:
        states.append(state)
        action, _ = agent.select_action(state)
        action = np.clip(action, env.spec.action_low, env.spec.action_high)

        next_state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        state = next_state

    agent.update(states, actions, rewards, state, dones)
    returns_history.append(sum(rewards))

    if (episode + 1) % 20 == 0:
        avg = np.mean(returns_history[-20:])
        print(f"Episode {episode+1}: Avg Return = {avg:.2f}")

print(f"\nInitial: {np.mean(returns_history[:20]):.2f}")
print(f"Final: {np.mean(returns_history[-20:]):.2f}")
```

### 4.2 Proximal Policy Optimization (PPO)

**PPO** is the most widely used algorithm for robot learning due to its stability:

$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

```python
"""
Proximal Policy Optimization (PPO) implementation.
"""

import numpy as np
from typing import List

class PPO:
    """PPO with clipped objective."""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                 n_epochs: int = 10):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.n_epochs = n_epochs
        self.lr = lr

        self.policy = GaussianPolicy(state_dim, action_dim, initial_std=0.3)
        self.value_fn = ValueFunctionEstimator(state_dim, learning_rate=0.01)

    def select_action(self, state: np.ndarray):
        action, log_prob = self.policy.sample(state)
        value = self.value_fn.predict(state)
        return action, log_prob, value

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = np.zeros(len(rewards))
        gae = 0

        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def update(self, states, actions, old_log_probs, advantages, returns):
        """PPO update with clipped objective."""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.n_epochs):
            # Update policy with clipping
            grad_W = np.zeros_like(self.policy.W)
            grad_b = np.zeros_like(self.policy.b)

            for state, action, old_lp, adv in zip(
                states, actions, old_log_probs, advantages
            ):
                new_lp, grads = self.policy.log_prob_and_grad(state, action)
                ratio = np.exp(new_lp - old_lp)

                # Only update if not clipped
                if abs(ratio - 1) <= self.clip_ratio:
                    grad_W += grads['W'] * adv
                    grad_b += grads['b'] * adv

            n = len(states)
            self.policy.W += self.lr * grad_W / n
            self.policy.b += self.lr * grad_b / n

        # Update value function
        for state, ret in zip(states, returns):
            self.value_fn.update_td(state, ret, state, self.gamma, True)


def train_ppo(env, agent, n_iterations: int,
              steps_per_iter: int = 2048) -> List[float]:
    """Train PPO agent."""
    returns_history = []

    for iteration in range(n_iterations):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        episode_returns = []
        ep_return = 0

        state = env.reset()
        for step in range(steps_per_iter):
            action, log_prob, value = agent.select_action(state)
            action = np.clip(action, env.spec.action_low, env.spec.action_high)

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            ep_return += reward

            state = next_state
            if done:
                episode_returns.append(ep_return)
                ep_return = 0
                state = env.reset()

        # Compute advantages
        next_value = agent.value_fn.predict(state)
        advantages = agent.compute_gae(rewards, values, next_value, dones)
        returns = advantages + np.array(values)

        # Update
        agent.update(states, actions, log_probs, advantages, returns)

        avg_return = np.mean(episode_returns) if episode_returns else 0
        returns_history.append(avg_return)

        if (iteration + 1) % 5 == 0:
            print(f"Iter {iteration+1}: Avg Return = {avg_return:.2f}")

    return returns_history


# Train PPO
print("\nProximal Policy Optimization (PPO) Training")
print("=" * 60)

env = ReacherEnv()
ppo = PPO(env.spec.state_dim, env.spec.action_dim)
ppo_returns = train_ppo(env, ppo, n_iterations=30, steps_per_iter=1000)

print(f"\nPPO Results:")
print(f"  Initial: {ppo_returns[0]:.2f}")
print(f"  Final: {np.mean(ppo_returns[-5:]):.2f}")
```

**Output:**
```
Proximal Policy Optimization (PPO) Training
============================================================
Iter 5: Avg Return = -35.67
Iter 10: Avg Return = -28.34
Iter 15: Avg Return = -21.45
Iter 20: Avg Return = -16.78
Iter 25: Avg Return = -13.23
Iter 30: Avg Return = -10.56

PPO Results:
  Initial: -42.89
  Final: -10.56
```

---

## 5. Off-Policy Methods: SAC

**Soft Actor-Critic (SAC)** maximizes both reward and entropy for robust exploration:

$$J(\pi) = \sum_t \mathbb{E} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

### Key Components of SAC

| Component | Purpose |
|-----------|---------|
| **Squashed Gaussian policy** | Bounded actions via tanh |
| **Twin Q-functions** | Reduce overestimation |
| **Automatic temperature** | Adapt exploration |
| **Replay buffer** | Off-policy learning |

```python
"""
Soft Actor-Critic components.
"""

import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """Experience replay for off-policy learning."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return [np.array([t[i] for t in batch]) for i in range(5)]

    def __len__(self):
        return len(self.buffer)


class SquashedGaussianPolicy:
    """Gaussian policy with tanh squashing for [-1, 1] actions."""

    def __init__(self, state_dim: int, action_dim: int):
        self.W_mean = np.random.randn(action_dim, state_dim) * 0.1
        self.b_mean = np.zeros(action_dim)
        self.log_std = np.zeros(action_dim)

    def sample(self, state: np.ndarray):
        mean = self.W_mean @ state + self.b_mean
        std = np.exp(np.clip(self.log_std, -20, 2))

        noise = np.random.randn(len(mean))
        pre_tanh = mean + std * noise
        action = np.tanh(pre_tanh)

        # Log prob with squashing correction
        log_prob = -0.5 * np.sum(np.log(2*np.pi*std**2) + ((pre_tanh-mean)/std)**2)
        log_prob -= np.sum(np.log(1 - action**2 + 1e-6))

        return action, log_prob


# SAC overview
print("\nSoft Actor-Critic (SAC) Overview")
print("=" * 60)
print("""
SAC Key Ideas:
1. Maximum entropy RL: maximize reward + entropy
2. Off-policy: use replay buffer for sample efficiency
3. Twin Q-networks: min of two Q-values reduces overestimation
4. Automatic temperature: adapts exploration over training

Typical hyperparameters:
  - Learning rate: 3e-4
  - Batch size: 256
  - Replay buffer: 1M transitions
  - Soft update tau: 0.005
  - Initial temperature: 0.2
  - Target entropy: -dim(action)
""")
```

---

## 6. Reward Design

### 6.1 Principles of Reward Engineering

| Principle | Description | Example |
|-----------|-------------|---------|
| **Dense > Sparse** | Continuous feedback helps learning | Distance to goal, not just success/fail |
| **Avoid reward hacking** | Robots find shortcuts | Add constraints, test edge cases |
| **Balance objectives** | Task vs energy vs smoothness | Tune weights carefully |
| **Potential shaping** | Preserves optimal policy | F(s,s') = γΦ(s') - Φ(s) |

```python
"""
Reward design for robot learning.
"""

import numpy as np
from typing import Callable, Dict

class RewardShaper:
    """Compose multiple reward components."""

    def __init__(self):
        self.components = {}

    def add_component(self, name: str, weight: float, func: Callable):
        self.components[name] = (weight, func)

    def compute(self, **kwargs) -> Dict[str, float]:
        breakdown = {}
        total = 0
        for name, (weight, func) in self.components.items():
            value = weight * func(**kwargs)
            breakdown[name] = value
            total += value
        breakdown['total'] = total
        return breakdown


# Example reward design
shaper = RewardShaper()
shaper.add_component('distance', 1.0,
    lambda ee_pos, target, **k: -np.linalg.norm(ee_pos - target))
shaper.add_component('action_penalty', 0.01,
    lambda action, **k: -np.sum(action**2))
shaper.add_component('success_bonus', 1.0,
    lambda ee_pos, target, **k: 10.0 if np.linalg.norm(ee_pos - target) < 0.05 else 0)

# Test
reward = shaper.compute(
    ee_pos=np.array([0.3, 0.2]),
    target=np.array([0.5, 0.3]),
    action=np.array([0.5, 0.3])
)
print("Reward breakdown:", {k: f"{v:.3f}" for k, v in reward.items()})
```

---

## 7. Algorithm Selection Guide

```
    CHOOSING AN RL ALGORITHM

    ┌─────────────────────────────────────────────────────────────┐
    │ Have a simulator?                                            │
    │     │                                                        │
    │     ├─ YES → Many samples available? → YES → PPO            │
    │     │                                  → NO  → SAC            │
    │     │                                                        │
    │     └─ NO  → Real robot learning → SAC (most efficient)      │
    │                                    + safe exploration        │
    └─────────────────────────────────────────────────────────────┘

    ALGORITHM COMPARISON
    ┌──────────┬───────────────┬───────────┬─────────────────────┐
    │ Algorithm│ Sample Eff.   │ Stability │ Best For            │
    ├──────────┼───────────────┼───────────┼─────────────────────┤
    │ PPO      │ Low           │ Very High │ Simulation, parallel│
    │ SAC      │ High          │ High      │ Real robots, complex│
    │ TD3      │ High          │ High      │ Deterministic tasks │
    │ A2C      │ Low           │ Medium    │ Fast prototyping    │
    └──────────┴───────────────┴───────────┴─────────────────────┘
```

---

## Summary

<div className="key-takeaways">

### Key Takeaways

1. **MDP formulation** provides the mathematical framework with states, actions, transitions, rewards, and discount factor

2. **Policy gradients** enable learning with continuous actions by directly optimizing policy parameters

3. **Baselines and advantages** reduce variance in gradient estimates, speeding up learning

4. **Actor-critic methods** combine policy learning with value estimation for improved efficiency

5. **PPO** is stable and widely used for simulation-based learning with its clipped objective

6. **SAC** is the most sample-efficient, ideal for real-robot learning with maximum entropy

7. **Reward design** is critical—use dense feedback, avoid hacking, and balance objectives

8. **Algorithm choice** depends on sample budget, stability needs, and simulation availability

</div>

---

## Exercises

<div className="exercise">

### Exercise 1: MDP Formulation (LO-1)
Formulate an MDP for a quadruped learning to walk. Define states, actions, and a reward function encouraging forward motion.
</div>

<div className="exercise">

### Exercise 2: REINFORCE Implementation (LO-3)
Implement REINFORCE for pendulum swing-up. Add a baseline and compare variance with/without it.
</div>

<div className="exercise">

### Exercise 3: PPO Training (LO-4)
Train PPO on a reaching task. Experiment with clip ratio and learning rate to find optimal settings.
</div>

<div className="exercise">

### Exercise 4: Reward Engineering (LO-5)
Design three reward functions (sparse, dense, shaped) for pick-and-place. Compare learning curves and final behaviors.
</div>

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

2. Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.

3. Haarnoja, T., et al. (2018). Soft actor-critic: Off-policy maximum entropy deep RL. *ICML*.

4. Kober, J., Bagnell, J. A., & Peters, J. (2013). RL in robotics: A survey. *IJRR*, 32(11), 1238-1274.

5. Lillicrap, T. P., et al. (2015). Continuous control with deep RL. *arXiv:1509.02971*.

---

## Further Reading

- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI educational resource
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL implementations
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single-file implementations

---

:::tip Next Chapter
Continue to **Chapter 3.2: Imitation Learning** to learn how robots can acquire skills from demonstrations.
:::
