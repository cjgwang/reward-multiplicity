import numpy as np


def build_transition_matrix_np(env) -> np.ndarray:
    """P: shape (S, A, S) with P[s,a,s'] = 1 for deterministic transitions."""
    S, A = env.num_states, env.num_actions
    P = np.zeros((S, A, S), dtype=np.float32)
    for s in range(S):
        for a in range(A):
            s_next = env.next_state_from_idx(s, a)
            P[s, a, s_next] = 1.0
    return P


def build_successor_representation_np(P: np.ndarray, policy: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    P_policy = np.einsum("saj,sa->sj", P, policy)  # (S,S)
    F = np.linalg.inv(np.eye(P_policy.shape[0], dtype=np.float32) - gamma * P_policy + 1e-8 * np.eye(P_policy.shape[0]))
    return F


def canonicalise_reward_np(F: np.ndarray, P: np.ndarray, R: np.ndarray, policy: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    S, A = R.shape
    expected_r = (R * policy).sum(axis=1)   # (S,)
    V = F @ expected_r                       # (S,)
    PV = np.einsum("saj,j->sa", P, V)        # (S,A)
    C = R - V.reshape(S, 1) + gamma * PV
    return C


def s_norm_np(Rc: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(Rc)
    if norm == 0:
        return Rc
    return Rc / norm


def starc_distance_np(env, policy: np.ndarray, r1_state: np.ndarray, r2_state: np.ndarray, gamma: float = 0.9) -> float:
    """
    r1_state, r2_state: arrays of shape (S,) giving r(s) for each canonical state.
    Computes STARC distance (L2 on canonicalised & normalized R matrices).
    """
    r1_state = r1_state.reshape(-1)
    r2_state = r2_state.reshape(-1)
    P = build_transition_matrix_np(env)
    F = build_successor_representation_np(P, policy, gamma)
    S, A, _ = P.shape

    R1 = np.zeros((S, A), dtype=np.float32)
    R2 = np.zeros((S, A), dtype=np.float32)
    for s in range(S):
        for a in range(A):
            ns = env.next_state_from_idx(s, a)
            R1[s, a] = float(r1_state[ns])
            R2[s, a] = float(r2_state[ns])

    C1 = canonicalise_reward_np(F, P, R1, policy, gamma)
    C2 = canonicalise_reward_np(F, P, R2, policy, gamma)
    return float(np.linalg.norm(s_norm_np(C1) - s_norm_np(C2)))
