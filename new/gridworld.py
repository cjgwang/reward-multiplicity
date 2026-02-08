import jax
import jax.numpy as jnp
from typing import Tuple, Callable, List, Optional, Any

# ----------------------------------------------------------------------------------------------------------
# Type definitions

Position = jnp.ndarray
State = Tuple[Position, Position]
Action = int
Transition = Tuple[State, Action, State]
ValueArray = jnp.ndarray
Reward = Callable[[Transition], float]
Policy = jnp.ndarray
Trajectory = List[Tuple[State, Action]]

# ----------------------------------------------------------------------------------------------------------
# Gridworld class

class DeterministicGridWorld:  
    def __init__(self, rows=5, cols=5, start=(0, 0), goal=(4, 4)):
        self.rows = int(rows)
        self.cols = int(cols)
        self.num_positions = self.rows * self.cols
        self.num_states = self.num_positions ** 2
        self.num_actions = 4
        self._actions = [jnp.array(x) for x in [(0, 1), (0, -1), (-1, 0), (1, 0)]]

        self.start_coord = jnp.array(start)
        self.goal_coord = jnp.array(goal)

    # helpers to transition between coordinates and index
    def coord_to_idx(self, coord: Position) -> int:
        # coord may be jnp.ndarray; use int() to return Python int
        c = int(coord[0])
        r = int(coord[1])
        return int(c + r * self.cols)

    def idx_to_coord(self, idx: int) -> Position:
        # return jnp array of ints
        return jnp.array((int(idx % self.cols), int(idx // self.cols)))

    # Helpers to transition between state and index
    def state_to_idx(self, state: State) -> int:
        pos, star = state
        pos_idx = self.coord_to_idx(pos)
        star_idx = self.coord_to_idx(star)
        return int(pos_idx + star_idx * self.num_positions)

    def idx_to_state(self, idx: int) -> State:
        pos = self.idx_to_coord(int(idx % self.num_positions))
        star = self.idx_to_coord(int(idx // self.num_positions))
        return pos, star

    # next coordinate following action a from position pos
    def next_coord(self, pos: Position, a: Action) -> Position:
        if (a < 0 or a >= self.num_actions):
            raise ValueError("Invalid action")
        # _actions[a] is jnp array; pos may be jnp array
        new = pos + self._actions[a]
        # clip needs array-like bounds; convert to jnp array
        upper = jnp.array([self.cols - 1, self.rows - 1])
        return jnp.clip(new, 0, upper)

    # Expand next_coord to states by not moving the star
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
def uniform_policy(env: DeterministicGridWorld) -> Policy:
    # shape (num_states, num_actions)
    return jnp.ones((env.num_states, env.num_actions), dtype=jnp.float32) / float(env.num_actions)


# Random deterministic policy
def make_random_policy(env: DeterministicGridWorld, seed: Optional[int] = None) -> Policy:
    """
    Returns a deterministic (one-hot) random policy.
    seed: optional integer. If None, uses PRNGKey(0).
    """
    if seed is None:
        key = jax.random.PRNGKey(0)
    else:
        key = jax.random.PRNGKey(int(seed))

    policy = jnp.zeros((env.num_states, env.num_actions), dtype=jnp.float32)
    # iterate states and sample one action per state using split keys
    for state in range(env.num_states):
        key, sub = jax.random.split(key)
        a = int(jax.random.randint(sub, (), 0, env.num_actions))
        policy = policy.at[state, a].set(1.0)
    return policy


# The goal reward
def star_reward(env: DeterministicGridWorld) -> Reward:
    def r(transition: Transition) -> float:
        _, _, s2 = transition
        pos, star = s2
        # (pos == star).all() returns a jnp array, convert to float
        return float(jnp.all(pos == star))
    return r


# The corner reward
def corner_reward(env: DeterministicGridWorld) -> Reward:
    corner = jnp.array([env.cols - 1, env.rows - 1])
    def r(transition: Transition) -> float:
        _, _, s2 = transition
        pos, _ = s2
        return float(jnp.all(pos == corner))
    return r


# Returns inverse reward transition
def inverse_reward(reward: Reward) -> Reward:
    return lambda t: -reward(t)


# ----------------------------------------------------------------------------------------------------------
# Trajectories

# Returns a trajectory of the specified length in env following policy
def sample_trajectory(env, length: int, policy: Policy, key: Optional[Any] = None) -> Trajectory:
    if key is None:
        key = jax.random.PRNGKey(0)
    traj = []
    state = (env.start_coord, env.goal_coord)
    for i in range(length):
        state_idx = env.state_to_idx(state)
        p = policy[state_idx]
        key, sub = jax.random.split(key)
        a = int(jax.random.choice(sub, a=env.num_actions, p=p))
        traj.append((state, a))
        state = env.next_state(state, a)
    return traj
