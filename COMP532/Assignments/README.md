# Bandit Framework Taxonomy

This project implements and analyses a wide range of multi-armed bandit algorithms, following the framework described in
Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto, with additional extensions commonly used in bandit research.

The bandit problem can be understood as a combination of four main components:

Environment – how rewards are generated

Estimator – how action values are estimated

Policy – how actions are selected

Metric – how performance is evaluated


Multi-Armed Bandit Problem
│
├── Environment
│   ├── Stationary bandit
│   │      True action values q*(a) remain constant
│   │
│   ├── Nonstationary bandit
│   │      q*(a) changes over time (random walk)
│   │
│   └── Contextual bandit
│          Reward depends on state s
│          Action values Q(s,a)
│
├── Action-Value Estimation
│   (How the agent estimates Q values)
│
│   ├── Sample-Average Estimator
│   │      Q ← Q + (1/N)(R − Q)
│   │
│   ├── Constant Step-Size Estimator
│   │      Q ← Q + α(R − Q)
│   │
│   ├── Unbiased Constant Step-Size Trick
│   │      Bias-corrected constant α update
│   │
│   └── Sliding Window Estimator
│          Uses most recent W rewards
│
├── Policies / Action Selection
│   (How the next action is chosen)
│
│   ├── Greedy
│   ├── ε-Greedy
│   ├── Optimistic Initial Values
│   ├── Upper Confidence Bound (UCB)
│   ├── Softmax / Boltzmann Policy
│   ├── Gradient Bandit
│   └── Thompson Sampling
│
├── Experiments
│
│   ├── ε-Greedy exploration comparison
│   ├── Algorithm comparison
│   ├── Parameter sensitivity study
│   │      ε, c, α, Q₀
│   ├── Nonstationary tracking experiment
│   └── Contextual bandit experiment
│
└── Evaluation Metrics
    ├── Average reward
    ├── Percentage of optimal action
    └── Cumulative regret
How the Components Interact

Each experiment combines the four core components:
Environment
      +
Estimator
      +
Policy
      +
Metric


Example Configuration
Environment: Stationary bandit
Estimator: Sample-average
Policy: ε-greedy
Metric: Average reward
