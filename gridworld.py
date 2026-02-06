# @title Gridworld Environment (setup) {"display-mode":"form"}
import numpy as np
from typing import Tuple, Callable
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ----------------------------------------------------------------------------------------------------------
# Type definitions

Position = np.ndarray
State = Tuple[Position, Position]
Action = int
Transition = Tuple[State, Action, State]
ValueArray = np.ndarray
Reward = Callable[[Transition], float]
Policy = np.ndarray
Trajectory = list[(State, Action)]

# ----------------------------------------------------------------------------------------------------------
# Gridworld class

class DeterministicGridWorld: #deterministic env
    def __init__(self, rows=5, cols=5, start=(0,0), goal=(4,4)):
        self.rows = int(rows)
        self.cols = int(cols)
        self.num_positions = self.rows * self.cols
        self.num_states = self.num_positions ** 2
        self.num_actions = 4
        self._actions = [np.array(x) for x in [(0, 1), (0, -1), (-1, 0), (1, 0)]]

        self.start_coord = np.array(start)
        self.goal_coord = np.array(goal)

    #helpers to transition between coordinates and index
    def coord_to_idx(self, coord: Position) -> int:
        c, r = coord[0], coord[1]
        return int(c + r * self.cols)

    def idx_to_coord(self, idx: int) -> Position:
        return np.array((idx % self.cols, idx // self.cols))

    # Helpers to transition between state and index
    def state_to_idx(self, state: State) -> int:
        pos, star = state
        pos_idx = self.coord_to_idx(pos)
        star_idx = self.coord_to_idx(star)
        return int(pos_idx + star_idx * self.num_positions)

    def idx_to_state(self, idx: int) -> State:
        pos = self.idx_to_coord(idx % self.num_positions)
        star = self.idx_to_coord(idx // self.num_positions)
        return pos, star

    # next coordinate following action a from position pos
    def next_coord(self, pos: Position, a: Action) -> Position:
        if(a < 0 or a >= self.num_actions):
            raise ValueError("Invalid action")
        return np.clip(pos + self._actions[a], 0, [self.cols - 1, self.rows - 1])

    # Expand next_coord to states by not mobing the star
    def next_state(self, s1: State, a: Action) -> State:
        pos, star = s1
        new_pos = self.next_coord(pos, a)
        return new_pos, star

    # Helper function to find next state directly through indexes
    def next_state_from_idx(self, s1: int, a: Action) -> int:
        state = self.idx_to_state(s1)
        new_state = self.next_state(state, a)
        return self.state_to_idx(new_state)

# ----------------------------------------------------------------------------------------------------------
# Standard policies and rewards

# Uniform random policy
def uniform_policy(env) -> Policy:
   return np.ones((env.num_states, env.num_actions)) / env.num_actions

# Random deterministic policy
def make_random_policy(env, seed=None) -> Policy:
    rng = np.random.RandomState(seed)
    policy = np.zeros((env.num_states, env.num_actions))
    for state in range(env.num_states):
        action = rng.randint(0, env.num_actions)
        policy[state][action] = 1
    return policy

# The goal reward
def star_reward(env) -> Reward:
    def r(transition: Transition) -> float:
        _, _, s2 = transition
        pos, star = s2
        return float((pos == star).all())
    return r

# The corner reward
def corner_reward(env) -> Reward:
    corner = [env.cols - 1, env.rows - 1]
    def r(transition: Transition) -> float:
        _, _, s2 = transition
        pos, _ = s2
        return float((pos == corner).all())
    return r

# Returns -reward(transition)
def inverse_reward(reward: Reward) -> Reward:
    return lambda t: -reward(t)

# ----------------------------------------------------------------------------------------------------------
# Trajectories

#used ot build dataset
# Returns a trajectory of the specified length in env following policy
def sample_trajectory(env, length: int, policy: Policy, seed=None) -> Trajectory:
    rng = np.random.RandomState(seed)
    traj = []
    state = (env.start_coord, env.goal_coord)

    for i in range(length):
        state_idx = env.state_to_idx(state)
        a = rng.choice(env.num_actions, p = policy[state_idx])
        traj.append((state, a))
        state = env.next_state(state, a)
    return traj

