# Invariance-aware diverse reward ensemble

This repository is for the RIO project supervised/mentored by Matthew Farrugia-Roberts, for studying reward multiplicity and reward canonicalisation in a deterministic Gridworld setting using STARC.

Overview
- Define a deterministic Gridworld environment
- Generate trajectories under a random policy
- Train an ensemble of reward networks
- Penalise non-diversity between rewards using a canonicalised distance STARC
- Run independent experiments
## Project Structure

```text
.
├── main.py        # Running experiments
├── gridworld.py   # Deterministic Gridworld environment, policies, rewards, trajectories
├── train.py       # Dataset construction, reward networks, training loops
├── starc.py       # STARc canonicalisation and ensemble loss
├── render.py      # Gridworld and reward visualisations
├── README.md
├── LICENSE
└── playground/    # Experiments
    ├── cath/
    ├── lexi/
    └── miguel/
```

## Environment
DeterministicGridWorld

2D grid (default: 5×5)

State = (agent_position, star_position)

4 deterministic actions: up, down, left, right

Star (goal) location can vary across trajectories

## Rewards

Implemented reward functions include:

Star reward: reward = 1 when agent reaches the star

Corner reward: reward = 1 in the bottom-right corner

## Learning Setup

### Dataset

Trajectories are sampled using a fixed policy (e.g. uniform random)

Each transition is converted into a vector:

[pos_x, pos_y, star_x, star_y, action, next_pos_x, next_pos_y, next_star_x, next_star_y]

### Reward Model

SmallRewardNet: a simple MLP predicting scalar rewards

Trained either individually or as an ensemble

### Ensemble Training & STARC

The main experiment trains an ensemble of reward networks with a combined loss:

MSE loss against the ground-truth reward

STARC loss to encourage diversity between canonicalised rewards

STARC works by:

Computing the successor representation under a policy

Canonicalising rewards to remove shaping terms

Penalising similarity between canonical rewards across the ensemble

This helps expose reward multiplicity while keeping behaviour consistent.

Running the Code

## Requirements

Python 3.9+

NumPy

PyTorch

Matplotlib


## Gridworld renderer shows:

⬤ agent start position

★ star (goal) position

Quadrant heatmaps can visualise per-action reward values over the grid

These are useful for inspecting learned reward structure and symmetries.

Notes & Caveats

This is a research prototype, not an optimised RL implementation

Policies are fixed (no planning or control learning)

STARC assumes full knowledge of the environment dynamics

Some files currently assume small state spaces (matrix inversions)
