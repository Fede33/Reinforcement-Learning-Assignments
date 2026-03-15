# Assignment 1 — Policy Iteration and iLQR

Implementation of **Dynamic Programming** and **Optimal Control** algorithms developed for the *Reinforcement Learning course (A.Y. 2025/2026)* of the MSc in Artificial Intelligence and Robotics.

The assignment focuses on implementing reinforcement learning and control algorithms **from scratch**, applying them to classical environments.

---

# Algorithms Implemented

## Policy Iteration
Policy Iteration is a **dynamic programming algorithm** used to compute the optimal policy of a Markov Decision Process (MDP).

The algorithm alternates between two phases:

1. **Policy Evaluation**  
   Compute the value function for the current policy.

2. **Policy Improvement**  
   Update the policy by choosing the action that maximizes the expected value.

The process is repeated until the policy becomes stable.

The implementation is tested on a **custom FrozenLake environment** with stochastic transitions.

---

## Value Iteration
Value Iteration is also implemented for comparison with Policy Iteration.

Instead of evaluating a fixed policy, the algorithm directly updates the value function using the **Bellman optimality equation**:

\[
V(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
\]

The optimal policy is extracted from the resulting value function.

---

## iLQR (Iterative Linear Quadratic Regulator)

The **Iterative Linear Quadratic Regulator (iLQR)** is an optimal control algorithm used to solve nonlinear control problems.

It works by iteratively:

1. Linearizing the system dynamics
2. Approximating the cost function using a quadratic expansion
3. Solving a sequence of **time-varying LQR problems**

The algorithm consists of two main phases:

### Backward Pass
Computes feedback and feedforward gains:

- \(K_t\) : feedback gain
- \(k_t\) : feedforward term

These are obtained by solving the Riccati-like equations.

### Forward Pass
Updates the control trajectory using

\[
u_t = u_t + k_t + K_t (x_t - \hat{x}_t)
\]

This step generates a new trajectory for the next iteration.

---

# Environments

## FrozenLake (Custom Environment)

A modified version of the FrozenLake environment was implemented.

Characteristics:

- Grid-based environment
- Stochastic transitions (slippery surface)
- Goal state with reward
- Holes representing terminal states

The agent must learn the optimal policy to reach the goal while avoiding holes.

---

## Pendulum

The **Pendulum-v1** environment from Gymnasium is used for the iLQR controller.

The system dynamics are defined as:

\[
\dot{\theta}_{t+1} = \dot{\theta}_t +
\left(\frac{3g}{2l}\sin\theta + \frac{3}{ml^2}u\right) dt
\]

\[
\theta_{t+1} = \theta + \dot{\theta}_{t+1} dt
\]

The objective is to **stabilize the pendulum in the upright position** while minimizing control effort.

The cost function used is:

\[
c(x,u) = \theta^2 + 0.1\dot{\theta}^2 + 0.001u^2
\]

---


Main components:

- **policy_iteration()** – implementation of the Policy Iteration algorithm
- **value_iteration()** – implementation of Value Iteration
- **ILqr class** – implementation of the iLQR controller
- **pendulum_dyn()** – nonlinear pendulum dynamics
- **cost()** – cost function for control optimization

---

# Technologies Used

- Python
- NumPy
- Autograd
- Gymnasium
- Matplotlib

---

# Learning Objectives

This assignment focuses on:

- Understanding **Markov Decision Processes**
- Implementing **dynamic programming algorithms**
- Applying **optimal control methods**
- Solving **nonlinear control problems**
- Training agents in simulated environments
