import numpy as np
from .env import DeterministicGridWorld, Reward, Transition, Position


def star_reward(env: DeterministicGridWorld) -> Reward:
    """Reward 1.0 when agent position equals star (goal)."""
    def r(transition: Transition) -> float:
        _, _, s2 = transition
        agent_pos, star_pos = s2
        return float((agent_pos == star_pos).all())
    return r


def corner_reward(env: DeterministicGridWorld) -> Reward:
    """Reward 1.0 when agent reaches bottom-right corner."""
    corner = np.array([env.cols - 1, env.rows - 1], dtype=int)
    def r(transition: Transition) -> float:
        _, _, s2 = transition
        agent_pos, _ = s2
        return float((agent_pos == corner).all())
    return r


def inverse_reward(reward: Reward) -> Reward:
    return lambda t: -reward(t)
