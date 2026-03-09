import numpy as np
from typing import Tuple
from .env import DeterministicGridWorld, State


def render_state_as_image(env: DeterministicGridWorld, state: State,
                          channels: int = 3, dtype=np.float32) -> np.ndarray:
    """
    Renders a (agent_pos, star_pos) state as a (C, H, W) image.
    Channel 0: star one-hot, Channel 1: agent one-hot, Channel 2: empty/reserved.
    """
    agent_pos, star_pos = state
    H, W = env.rows, env.cols

    ch_star  = np.zeros((H, W), dtype=dtype)
    ch_agent = np.zeros((H, W), dtype=dtype)
    ch_misc  = np.zeros((H, W), dtype=dtype)

    ax, ay = int(agent_pos[0]), int(agent_pos[1])
    gx, gy = int(star_pos[0]), int(star_pos[1])
    ch_star[gy, gx] = 1.0
    ch_agent[ay, ax] = 1.0

    img = np.stack([ch_star, ch_agent, ch_misc], axis=0).astype(dtype)

    if channels != 3:
        if channels < 3:
            img = img[:channels]
        else:
            pad = np.zeros((channels - 3, H, W), dtype=dtype)
            img = np.concatenate([img, pad], axis=0)
    return img


def make_grid_image(agent_pos: Tuple[int,int], star_pos: Tuple[int,int],
                    H: int = 7, W: int = 7, channels: int = 3) -> Tuple[np.ndarray, float]:
    """
    Channel order: (R, G, B)
      R (ch 0): star
      G (ch 1): agent
      B (ch 2): border
    Returns (img: np.float32 (channels, H, W), reward: float 0.0/1.0)
    """
    r_agent, c_agent = agent_pos
    img = np.zeros((channels, H, W), dtype=np.float32)

    # Border -> BLUE channel (index 2)
    if channels > 2:
        img[2, 0, :] = 1.0
        img[2, -1, :] = 1.0
        img[2, :, 0] = 1.0
        img[2, :, -1] = 1.0

    # Star -> RED channel (index 0)
    r_star, c_star = star_pos
    img[0, r_star, c_star] = 1.0

    # Agent -> GREEN channel (index 1)
    if channels > 1:
        img[1, r_agent, c_agent] = 1.0

    reward = 1.0 if (r_agent, c_agent) == (r_star, c_star) else 0.0
    return img, reward
