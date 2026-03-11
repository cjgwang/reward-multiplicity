import random
import numpy as np
from typing import Tuple, Optional
from tqdm import trange

from gridworld.env import DeterministicGridWorld
from gridworld.renderer import render_state_as_image, make_grid_image


def generate_balanced_dataset(star_pos=(5, 5),
                               n_pos: int = 2000,
                               n_neg: int = 2000,
                               H: int = 7, W: int = 7,
                               agent_allowed: Optional[list] = None,
                               env=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supports:
        star_pos = (r,c)
        OR
        star_pos = [(r1,c1), (r2,c2), ...]
    """

    # -------------------------
    # Normalize star_pos input
    # -------------------------
    if isinstance(star_pos, list):
        star_positions = star_pos
    else:
        star_positions = [star_pos]

    all_positions = [(r, c) for r in range(H) for c in range(W)]
    if agent_allowed is None:
        agent_allowed = all_positions

    X, y = [], []

    for sp in star_positions:
        neg_positions = [p for p in agent_allowed if p != sp]

        def _img(agent_pos):
            if env is not None:
                return render_state_as_image(env, (np.array(agent_pos), np.array(sp)))
            return make_grid_image(agent_pos=agent_pos, star_pos=sp, H=H, W=W)[0]

        # positives for this star
        for _ in range(n_pos):
            X.append(_img(sp))
            y.append(1.0)

        # negatives for this star
        for _ in range(n_neg):
            pos = random.choice(neg_positions)
            X.append(_img(pos))
            y.append(0.0)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def generate_corner_dataset_aligned(star_pos: Tuple[int,int] = (5, 5),
                                     n_pos: int = 2000,
                                     n_neg: int = 2000,
                                     H: int = 7, W: int = 7,
                                     env=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligned corner dataset: agent at star_pos (regardless of where the star is) -> 1.0.
    Both positives and negatives use a random star position each sample.
    If env is provided, uses render_state_as_image (consistent with full-domain evaluation).
    """
    grid_coords = [(r, c) for r in range(H) for c in range(W)]
    neg_agent_positions = [p for p in grid_coords if p != star_pos]

    def _img(agent_pos, s_pos):
        if env is not None:
            return render_state_as_image(env, (np.array(agent_pos), np.array(s_pos)))
        return make_grid_image(agent_pos=agent_pos, star_pos=s_pos, H=H, W=W)[0]

    X, y = [], []
    for _ in range(n_pos):
        random_star = random.choice(grid_coords)
        X.append(_img(star_pos, random_star)); y.append(1.0)
    for _ in range(n_neg):
        random_agent = random.choice(neg_agent_positions)
        random_star = random.choice(grid_coords)
        X.append(_img(random_agent, random_star)); y.append(0.0)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def generate_follow_star_dataset(env,
                                  n_pos: int = 2000,
                                  n_neg: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    'Follow the star' dataset: agent at star position → reward=1, else → reward=0.
    Star position is random each sample (covering all positions), so the model
    must learn to compare agent and star channels rather than memorising a fixed
    position. Uses render_state_as_image so images match the full-domain evaluation.
    """
    num_positions = env.num_positions
    X, y = [], []

    for _ in range(n_pos):
        star_idx = np.random.randint(0, num_positions)
        star_pos = env.idx_to_coord(star_idx)
        img = render_state_as_image(env, (star_pos, star_pos))
        X.append(img); y.append(1.0)

    for _ in range(n_neg):
        star_idx = np.random.randint(0, num_positions)
        star_pos = env.idx_to_coord(star_idx)
        # pick a different agent position
        agent_idx = np.random.randint(0, num_positions - 1)
        if agent_idx >= star_idx:
            agent_idx += 1
        agent_pos = env.idx_to_coord(agent_idx)
        img = render_state_as_image(env, (agent_pos, star_pos))
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
