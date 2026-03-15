# Assignment 2 — SARSA-λ and Linear Q Approximation

Implementation of **temporal-difference reinforcement learning algorithms** developed for the *Reinforcement Learning course (A.Y. 2025/2026)* of the MSc in Artificial Intelligence and Robotics.

This assignment focuses on value-based reinforcement learning in both **tabular** and **function approximation** settings.

---

# Algorithms Implemented

## SARSA-λ

SARSA-λ is an **on-policy temporal-difference learning algorithm** that extends SARSA through the use of **eligibility traces**.

The algorithm updates the action-value function using the TD error:

\[
\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
\]

Eligibility traces are used to propagate the TD error backward to previously visited state-action pairs:

\[
E(s,a) \leftarrow \gamma \lambda E(s,a)
\]

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E(s,a)
\]

An **epsilon-greedy exploration strategy** is used during training.

The implementation is tested on the **Taxi-v3** environment.

---

## TD(λ) / Q-Learning with Linear Function Approximation

The second part of the assignment addresses reinforcement learning with **linear value function approximation**.

Since the MountainCar environment has a continuous state space, the action-value function is approximated using:

- **Linear function approximation**
- **RBF (Radial Basis Function) feature encoding**

The feature encoder transforms the raw continuous state into a higher-dimensional representation using multiple RBF samplers with different kernel widths.

The approximated action-value function is represented as:

\[
Q(s,a) = w_a^\top \phi(s)
\]

where:

- \( \phi(s) \) is the RBF feature representation of the state
- \( w_a \) are the weights associated with action \(a\)

Eligibility traces are also used in this setting to improve credit assignment across time steps.

---

# Environments

## Taxi-v3

The **Taxi-v3** environment is a discrete reinforcement learning task where the agent must:

- pick up a passenger
- navigate a grid world
- drop the passenger off at the correct destination

This environment is used to test the **tabular SARSA-λ** implementation.

---

## MountainCar-v0

The **MountainCar-v0** environment is a classical control problem with a **continuous state space**.

The car does not initially have enough power to climb the hill directly, so it must learn to build momentum by moving back and forth.

This environment is used to test **linear value function approximation with RBF features**.

---

# Feature Encoding

## RBF Feature Encoder

To handle continuous states in MountainCar, an **RBF-based feature encoder** is used.

The encoder performs:

1. **State normalization** using `StandardScaler`
2. **Feature transformation** using multiple `RBFSampler` blocks
3. Concatenation of the generated features into a single representation

The implementation uses multiple RBF samplers with different gamma values to capture state information at multiple resolutions.

This allows the agent to generalize across nearby states instead of learning a purely tabular representation.

---

## Main Components

- **`sarsa_lambda()`** — tabular SARSA-λ implementation for the Taxi environment  
- **`epsilon_greedy_action()`** — epsilon-greedy exploration policy used during training  
- **`RBFFeatureEncoder`** — feature extraction for continuous states using RBF kernels  
- **`TDLambda_LVFA`** — TD(λ) agent with Linear Value Function Approximation  
- **`update_transition()`** — weight update using TD error and eligibility traces  
- **`save()` / `load()`** — utilities to save and load trained models  

---

## Technologies Used

- Python  
- NumPy  
- Gymnasium  
- Scikit-learn  
- Pickle  
- tqdm  

---

## Learning Objectives

This assignment focuses on:

- implementing **on-policy temporal-difference control**
- understanding and using **eligibility traces**
- applying **epsilon-greedy exploration strategies**
- using **linear function approximation in reinforcement learning**
- handling **continuous state spaces**
- designing **RBF-based feature representations**
