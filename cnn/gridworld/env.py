import numpy as np
from typing import Tuple, Callable, List

# ---------------------------------------------------------------------
# Conventions (single source of truth)
# Position: numpy array [x, y] where x is column (0..cols-1), y is row (0..rows-1)
# Image array convention for ML: PyTorch-style (C, H, W) with dtype float32.
# Action indexing: 0=up (y-1), 1=down (y+1), 2=left (x-1), 3=right (x+1).
# preds_actions tensor shape (rows, cols, actions) == (H, W, A).
# ---------------------------------------------------------------------

Position = np.ndarray               # [x, y] as integers
State = Tuple[Position, Position]   # (agent_pos, goal_pos)
Action = int
Transition = Tuple[State, Action, State]
ValueArray = np.ndarray
Reward = Callable[[Transition], float]
Policy = np.ndarray
Trajectory = List[Tuple[State, Action]]


class DeterministicGridWorld:
    def __init__(self, rows: int = 5, cols: int = 5, start: Tuple[int,int] = (0,0), goal: Tuple[int,int] = (4,4)):
        self.rows = int(rows)
        self.cols = int(cols)
        self.num_positions = self.rows * self.cols
        self.num_states = self.num_positions ** 2
        self.num_actions = 4
        # action vectors in [dx, dy] with the canonical ordering documented above
        self._actions = [np.array([0, -1], dtype=int),  # up
                         np.array([0,  1], dtype=int),  # down
                         np.array([-1, 0], dtype=int),  # left
                         np.array([1,  0], dtype=int)]  # right

        self.start_coord = np.array(start, dtype=int)
        self.goal_coord = np.array(goal, dtype=int)

    def coord_to_idx(self, coord: Position) -> int:
        x, y = int(coord[0]), int(coord[1])
        return int(x + y * self.cols)

    def idx_to_coord(self, idx: int) -> Position:
        x = int(idx % self.cols)
        y = int(idx // self.cols)
        return np.array([x, y], dtype=int)

    def state_to_idx(self, state: State) -> int:
        agent_pos, star_pos = state
        agent_idx = self.coord_to_idx(agent_pos)
        star_idx  = self.coord_to_idx(star_pos)
        return int(agent_idx + star_idx * self.num_positions)

    def idx_to_state(self, idx: int) -> State:
        agent_idx = int(idx % self.num_positions)
        star_idx  = int(idx // self.num_positions)
        return self.idx_to_coord(agent_idx), self.idx_to_coord(star_idx)

    def next_coord(self, pos: Position, a: Action) -> Position:
        if a < 0 or a >= self.num_actions:
            raise ValueError("Invalid action")
        new = pos + self._actions[a]
        new_clipped = np.clip(new, [0, 0], [self.cols - 1, self.rows - 1])
        return new_clipped.astype(int)

    def next_state(self, s1: State, a: Action) -> State:
        pos, star = s1
        new_pos = self.next_coord(pos, a)
        return (new_pos, star)

    def next_state_from_idx(self, s1_idx: int, a: Action) -> int:
        s = self.idx_to_state(s1_idx)
        ns = self.next_state(s, a)
        return self.state_to_idx(ns)

    def enumerate_states(self):
        for star_idx in range(self.num_positions):
            star = self.idx_to_coord(star_idx)
            for agent_idx in range(self.num_positions):
                agent = self.idx_to_coord(agent_idx)
                yield (agent, star)
