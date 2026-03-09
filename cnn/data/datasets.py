import random
import numpy as np
from typing import Tuple, Optional
from tqdm import trange

from gridworld.env import DeterministicGridWorld
from gridworld.renderer import render_state_as_image, make_grid_image


def generate_balanced_dataset(star_pos: Tuple[int,int] = (5, 5),
                               n_pos: int = 2000,
                               n_neg: int = 2000,
                               H: int = 7, W: int = 7,
                               agent_allowed: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed-star dataset: agent at star_pos -> reward 1.0, otherwise 0.0.
    agent_allowed: list of (row, col) positions the agent can occupy for negatives.
                   Defaults to all positions except star_pos in rows 1..H-1, cols 1..W-1.
    """
    if agent_allowed is None:
        agent_allowed = [(r, c) for r in range(1, H) for c in range(1, W)]
    neg_positions = [p for p in agent_allowed if p != star_pos]

    X, y = [], []
    for _ in range(n_pos):
        img, r = make_grid_image(agent_pos=star_pos, star_pos=star_pos, H=H, W=W)
        X.append(img); y.append(r)
    for _ in range(n_neg):
        pos = random.choice(neg_positions)
        img, r = make_grid_image(agent_pos=pos, star_pos=star_pos, H=H, W=W)
        X.append(img); y.append(r)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def generate_corner_dataset_aligned(star_pos: Tuple[int,int] = (5, 5),
                                     n_pos: int = 2000,
                                     n_neg: int = 2000,
                                     H: int = 7, W: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligned corner dataset: agent at star_pos (regardless of where the star is) -> 1.0.
    Both positives and negatives use a random star position each sample.
    """
    grid_coords = [(r, c) for r in range(H) for c in range(W)]
    neg_agent_positions = [p for p in grid_coords if p != star_pos]

    X, y = [], []
    for _ in range(n_pos):
        random_star = random.choice(grid_coords)
        img, _ = make_grid_image(agent_pos=star_pos, star_pos=random_star, H=H, W=W)
        X.append(img); y.append(1.0)
    for _ in range(n_neg):
        random_agent = random.choice(neg_agent_positions)
        random_star = random.choice(grid_coords)
        img, _ = make_grid_image(agent_pos=random_agent, star_pos=random_star, H=H, W=W)
        X.append(img); y.append(0.0)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def generate_full_enumeration_dataset(env: DeterministicGridWorld,
                                       channels: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enumerate all (agent_pos, star_pos) pairs using the canonical env.idx_to_state ordering.

    Returns:
        images_full:      (S, C, H, W) float32
        true_rewards_full: (S,) float32 — 1.0 when agent == star, else 0.0
        state_index_map:  (S, 2, 2) int32 — [idx, 0]=agent[x,y], [idx, 1]=star[x,y]
    """
    S = env.num_states
    H, W = env.rows, env.cols
    images_full = np.zeros((S, channels, H, W), dtype=np.float32)
    true_rewards_full = np.zeros(S, dtype=np.float32)
    state_index_map = np.zeros((S, 2, 2), dtype=np.int32)

    for idx in trange(S, desc="Rendering full domain"):
        agent_pos, star_pos = env.idx_to_state(idx)
        img = render_state_as_image(env, (agent_pos, star_pos), channels=channels)
        images_full[idx] = img
        true_rewards_full[idx] = 1.0 if (agent_pos == star_pos).all() else 0.0
        state_index_map[idx, 0] = [int(agent_pos[0]), int(agent_pos[1])]
        state_index_map[idx, 1] = [int(star_pos[0]), int(star_pos[1])]

    return images_full, true_rewards_full, state_index_map
