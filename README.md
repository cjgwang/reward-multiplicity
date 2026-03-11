# Reward Multiplicity via Invariance-Aware Diverse Reward Ensembles

A research project studying **reward multiplicity** and **reward canonicalisation** in a deterministic Gridworld using STARC.

The core question: given a fixed policy and environment, many different reward functions are consistent with observed behaviour. This project trains an ensemble of reward networks with a diversity-promoting loss (STARC) to expose and measure this multiplicity.

## Overview

1. Define a deterministic Gridworld environment
2. Generate trajectories under a fixed policy
3. Train an ensemble of reward networks jointly
4. Penalise similarity between rewards using STARC canonicalised distance
5. Visualise learned reward structure and diversity

## Project Structure

```text
.
├── new/                    # Modular JAX implementation (main codebase)
│   ├── main.py             # Entry point: runs ensemble training experiment
│   ├── gridworld.py        # Environment, policies, reward functions, trajectories
│   ├── train.py            # Dataset construction, SmallRewardNet, training loops
│   ├── starc.py            # STARC canonicalisation and ensemble diversity loss
│   └── render.py           # Gridworld visualisations
│
├── cnn/                    # CNN/PyTorch implementation with experiment scripts
│   ├── gridworld/          # env.py, policies.py, rewards.py, renderer.py
│   ├── starc/              # numpy_ops.py, torch_ops.py
│   ├── training/           # Training loop (loops.py)
│   └── scripts/            # Experiment scripts (ensemble, frozen, bump, etc.)
│
├── jax/                    # Original single-file JAX prototype
│   ├── main.py
│   ├── gridworld.py
│   ├── train.py
│   ├── starc.py
│   └── render.py
│
├── pyproject.toml
├── README.md
└── LICENSE
```

## Environment

**`DeterministicGridWorld`** — a 2D grid (default 5×5).

- **State**: `(agent_position, star_position)` — both as `[x, y]` coordinates
- **State space**: `(rows × cols)²` = 625 states for the default grid
- **Actions**: 4 deterministic actions — up, down, left, right (wall-clipped)
- **Star (goal)**: fixed per trajectory; can vary across trajectories

## Reward Functions

| Name | Description |
|------|-------------|
| `star_reward` | `+1` when agent reaches the star position |
| `corner_reward` | `+1` when agent reaches the bottom-right corner |
| `inverse_reward` | Negation of any reward function |

## Learning Setup

### Dataset

Trajectories are sampled under a policy (e.g. uniform random). Each transition `(s, a, s')` is encoded as a 9-dimensional vector:

```
[pos_x, pos_y, star_x, star_y, action, next_pos_x, next_pos_y, next_star_x, next_star_y]
```

### Reward Model

**`SmallRewardNet`** — a small MLP (pure JAX, no Flax) predicting a scalar reward from transition vectors.

- Default architecture: `9 → 32 → 1` with ReLU activation
- Glorot weight initialisation
- Trained with Adam (optax) + L2 regularisation

### Ensemble Training & STARC Loss

The ensemble is trained jointly with a combined loss:

```
L = MSE(predictions, ground_truth) + λ · STARC_diversity_loss
```

**STARC** works by:
1. Computing the **successor representation** `F = (I − γ P_π)⁻¹` from environment dynamics and policy
2. **Canonicalising** each reward to remove potential-shaping terms: `C = R − V + γ P V`
3. **L2-normalising** the canonical reward (`s-norm`)
4. Penalising **cosine similarity** (small distance) between canonicalised rewards across ensemble members

This encourages the ensemble to find diverse reward functions that are all consistent with the training signal, directly probing reward multiplicity.

The `cnn/` implementation extends this with support for **frozen networks** (previously trained models included in the diversity penalty without being updated).

## Setup

```bash
# Install dependencies (using uv)
uv sync

# Or with pip
pip install numpy torch matplotlib
# For new/ and jax/ modules:
pip install jax jaxlib optax
```

## Running

```bash
# Run the JAX ensemble experiment (new/ module)
python -m new.main
```

For CNN experiments, see the scripts in `cnn/scripts/`:

```bash
python -m cnn.scripts.train_ensemble
python -m cnn.scripts.run_starc_ensemble
python -m cnn.scripts.train_frozen
```

## Visualisation

The renderer (`render.py`) provides:

- Grid visualisation with agent start `⬤` and star goal `★`
- Quadrant heatmaps showing per-action reward values across grid positions — useful for inspecting learned reward structure and symmetries

## Notes

- This is a research prototype, not an optimised RL implementation
- Policies are fixed (no planning or control learning)
- STARC requires full knowledge of environment dynamics (transition matrix + policy)
- Matrix inversion in the successor representation assumes small state spaces
- Authors: MFR, CW, MD, LS
