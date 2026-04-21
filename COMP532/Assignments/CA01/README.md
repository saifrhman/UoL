# Multi-Armed Bandits: Exploration, Exploitation, and Action-Value Methods

This repository contains an experimental study of **multi-armed bandit algorithms** based on the framework presented in:

Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction (2nd Edition)*.

The project investigates how different exploration strategies and action-value estimation methods influence learning performance in both stationary and nonstationary bandit environments.

The experiments were implemented in Python and evaluated through learning curves, parameter studies, and summary tables.

---

# Project Overview

The multi-armed bandit problem is a classical reinforcement learning setting in which an agent repeatedly selects from a set of actions with unknown reward distributions.

The agent must balance two competing objectives:

- **Exploration** — trying different actions to gather information  
- **Exploitation** — selecting the action that currently appears to produce the highest reward  

This repository studies this trade-off through three experimental problems.

| Problem | Focus |
|------|------|
| **Problem 1** | Implementation of a multi-armed bandit algorithm |
| **Problem 2** | Exploration strategies |
| **Problem 3** | Action-value estimation methods |

All experiments follow the standard Gaussian bandit testbed and results are averaged across many independent runs.

---

# Repository Structure

```mermaid
flowchart TD

A[Repository Root]

A --> B[notebook.ipynb]
A --> C[report.pdf]
A --> D[outputs]

D --> E[Problem 1]
E --> E1[fig1_q1_eps_greedy_avg_reward.png]
E --> E2[fig1_q1_eps_greedy_pct_optimal.png]
E --> E3[table_problem1_n_sensitivity.csv]

D --> F[Problem 2]
F --> F1[fig2_q2_algorithm_comparison_avg_reward.png]
F --> F2[fig2_q2_algorithm_comparison_pct_optimal.png]
F --> F3[fig3_q2_parameter_sensitivity.png]
F --> F4[table2a_best_parameter.csv]
F --> F5[table2b_scaling_summary.csv]

D --> G[Problem 3]
G --> G1[fig4_q3_stationary_avg_reward.png]
G --> G2[fig4_q3_stationary_pct_optimal.png]
G --> G3[fig5_q3_nonstationary_avg_reward.png]
G --> G4[fig5_q3_nonstationary_pct_optimal.png]
G --> G5[table3_stationary_summary.csv]
G --> G6[table3_nonstationary_summary.csv]
