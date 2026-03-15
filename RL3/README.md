# Assignment 3 — Deep Reinforcement Learning for CarRacing

Implementation of a **Deep Reinforcement Learning agent** developed for the *Reinforcement Learning course (A.Y. 2025/2026)* of the MSc in Artificial Intelligence and Robotics.

The goal of this assignment is to train an autonomous agent capable of driving in the **CarRacing-v2 environment** using a policy gradient method implemented with **PyTorch**.

---

# Algorithm Implemented

## Proximal Policy Optimization (PPO)

This project implements **Proximal Policy Optimization (PPO)**, a widely used policy-gradient algorithm for deep reinforcement learning.

PPO improves training stability by limiting how much the policy can change at each update step using a **clipped surrogate objective**.

The objective function is defined as:

\[
L^{CLIP}(\theta) = \mathbb{E} \left[
\min(r_t(\theta)A_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t)
\right]
\]

where:

- \( r_t(\theta) \) is the probability ratio between new and old policies
- \( A_t \) is the advantage estimate
- \( \epsilon \) is the clipping parameter

---

# Neural Network Architecture

The policy uses a **Convolutional Neural Network (CNN)** to process visual observations from the environment.

### Feature Extractor

The network processes RGB frames using convolutional layers:

- Conv2D (32 filters, kernel 8×8, stride 4)
- Conv2D (64 filters, kernel 4×4, stride 2)
- Conv2D (64 filters, kernel 3×3, stride 1)

Followed by:

- Flatten layer
- Fully connected layer (256 units)

### Actor-Critic Structure

The network uses a shared feature extractor followed by two heads:

**Actor**
- Outputs action probabilities (discrete control)

**Critic**
- Estimates the value function \( V(s) \)

This architecture allows the agent to simultaneously learn:

- the **policy**
- the **value function**

---

# Advantage Estimation

Advantages are computed using **Generalized Advantage Estimation (GAE)**:

\[
A_t = \delta_t + (\gamma \lambda)\delta_{t+1} + ...
\]

where:

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

GAE reduces variance while maintaining low bias, improving training stability.

---

# Environment

## CarRacing-v2

The **CarRacing-v2** environment from Gymnasium is a continuous control task where the agent must drive a car around a randomly generated track.

Key characteristics:

- high-dimensional visual observations (RGB images)
- sparse and delayed rewards
- continuous driving dynamics

The objective is to learn a policy that maximizes the cumulative reward by completing the track efficiently.

---

# Training Strategy

The training loop includes:

- **frame skipping** to reduce computational cost
- **reward clipping** to stabilize learning
- **mini-batch PPO updates**
- **gradient clipping** to prevent exploding gradients

Training is performed for up to **7500 episodes**.

Key hyperparameters include:

- learning rate: `1e-4`
- discount factor: `γ = 0.99`
- GAE parameter: `λ = 0.95`
- PPO clipping parameter: `ε = 0.2`

---


## Main Components

- **`Policy`** — Actor-Critic neural network implementing the PPO policy and value function.
- **`forward()`** — Defines the neural network forward pass used to compute action probabilities and state value estimates.
- **`act()`** — Samples an action from the policy distribution given the current state.
- **`compute_advantages()`** — Computes advantages using **Generalized Advantage Estimation (GAE)**.
- **`train()`** — Implements the full PPO training loop including trajectory collection and policy updates.
- **`save()` / `load()`** — Functions used to store and load trained model weights.

---

## Technologies Used

- **Python**
- **PyTorch**
- **NumPy**
- **Gymnasium**
- **SciPy**
