import numpy as np
from .env import DeterministicGridWorld, Policy, Trajectory, State, Action


def uniform_policy(env: DeterministicGridWorld) -> Policy:
    return np.ones((env.num_states, env.num_actions), dtype=float) / float(env.num_actions)


def make_random_policy(env: DeterministicGridWorld, seed: int | None = None) -> Policy:
    rng = np.random.default_rng(seed)
    policy = np.zeros((env.num_states, env.num_actions), dtype=float)
    for s_idx in range(env.num_states):
        a = rng.integers(0, env.num_actions)
        policy[s_idx, a] = 1.0
    return policy


def sample_trajectory(env: DeterministicGridWorld, length: int, policy: Policy,
                      seed: int | None = None,
                      randomize_start: bool = False,
                      randomize_goal: bool = False) -> Trajectory:
    rng = np.random.default_rng(seed)
    start = env.start_coord.copy()
    goal = env.goal_coord.copy()
    if randomize_start:
        start = env.idx_to_coord(rng.integers(0, env.num_positions))
    if randomize_goal:
        goal = env.idx_to_coord(rng.integers(0, env.num_positions))
    state = (start, goal)
    traj: Trajectory = []
    for _ in range(length):
        s_idx = env.state_to_idx(state)
        a = int(rng.choice(env.num_actions, p=policy[s_idx]))
        traj.append((state, a))
        state = env.next_state(state, a)
    return traj
