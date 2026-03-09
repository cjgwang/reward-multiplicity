import numpy as np
import torch
from typing import Optional, List


def build_transition_matrix_torch(env, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns P: (S, A, S) float tensor."""
    device = torch.device("cpu") if device is None else device
    S, A = env.num_states, env.num_actions
    P = torch.zeros((S, A, S), dtype=torch.float32, device=device)
    for s in range(S):
        for a in range(A):
            ns = env.next_state_from_idx(s, a)
            P[s, a, ns] = 1.0
    return P


def build_next_state_idx(env, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns next_state_idx: (S, A) long tensor where entry [s,a] = idx of next state."""
    device = torch.device("cpu") if device is None else device
    S, A = env.num_states, env.num_actions
    idx = np.zeros((S, A), dtype=np.int64)
    for s in range(S):
        for a in range(A):
            idx[s, a] = env.next_state_from_idx(s, a)
    return torch.from_numpy(idx).long().to(device)


def build_successor_representation_torch(P_t: torch.Tensor, policy_t: torch.Tensor, gamma: float = 0.9) -> torch.Tensor:
    """P_t: (S,A,S), policy_t: (S,A) -> F: (S,S)"""
    P_policy = torch.einsum("saj,sa->sj", P_t, policy_t)  # (S,S)
    I = torch.eye(P_policy.size(0), dtype=P_t.dtype, device=P_t.device)
    F = torch.inverse(I - gamma * P_policy)
    return F


def canonicalise_reward_torch(F_t: torch.Tensor, P_t: torch.Tensor,
                               R_t: torch.Tensor, policy_t: torch.Tensor,
                               gamma: float = 0.9) -> torch.Tensor:
    S, A = R_t.shape
    expected_r = (R_t * policy_t).sum(dim=1)   # (S,)
    V = F_t.matmul(expected_r)                  # (S,)
    PV = torch.einsum("saj,j->sa", P_t, V)      # (S,A)
    C_t = R_t - V.view(S, 1) + gamma * PV
    return C_t


def s_norm_torch(Rc_t: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(Rc_t)
    if norm == 0:
        return Rc_t
    return Rc_t / norm


def state_values_to_state_action_rewards(r_state_t: torch.Tensor,
                                          next_state_idx: torch.Tensor) -> torch.Tensor:
    """
    r_state_t: (S,) or (S,1) — predicted reward for each state r(s).
    next_state_idx: (S,A) long tensor where next_state_idx[s,a] = idx of s'.
    Returns R_t: (S,A) where R_t[s,a] = r_state_t[next_state(s,a)].
    """
    r_flat = r_state_t.reshape(-1)
    return r_flat[next_state_idx]


def build_starc_precomputed(env, policy_np: np.ndarray, gamma: float = 0.9,
                             device: Optional[torch.device] = None):
    """
    Precompute all tensors needed for differentiable STARC distance computation.
    Returns a dict with P_t, F_t, policy_t, next_state_idx_t.
    """
    device = torch.device("cpu") if device is None else device
    P_t = build_transition_matrix_torch(env, device=device)
    policy_t = torch.from_numpy(policy_np).float().to(device)
    F_t = build_successor_representation_torch(P_t, policy_t, gamma)
    next_state_idx_t = build_next_state_idx(env, device=device)
    return {"P_t": P_t, "F_t": F_t, "policy_t": policy_t, "next_state_idx_t": next_state_idx_t}


def r_state_to_cnorm_flat(r_state_t: torch.Tensor, precomputed: dict, gamma: float = 0.9) -> torch.Tensor:
    """
    Convert a state-value vector (S,) to a canonicalised, L2-normalised, flattened vector (S*A,).
    Differentiable w.r.t. r_state_t.
    """
    r_flat = r_state_t.reshape(-1)
    R_t = state_values_to_state_action_rewards(r_flat, precomputed["next_state_idx_t"])
    C_t = canonicalise_reward_torch(precomputed["F_t"], precomputed["P_t"],
                                     R_t, precomputed["policy_t"], gamma)
    Cn_t = s_norm_torch(C_t)
    return Cn_t.reshape(-1)


def ensemble_starc_loss_builder(env, policy_np: np.ndarray, gamma: float = 0.9,
                                 device: Optional[torch.device] = None):
    """
    Returns a loss function that computes average pairwise STARC distance
    between a list of model predictions (each shape (S,)).

    Usage:
        loss_fn = ensemble_starc_loss_builder(env, policy_np, gamma, device)
        loss_val, trackers = loss_fn(r_preds)   # r_preds: list of (S,) tensors
    """
    device = torch.device("cpu") if device is None else device
    precomputed = build_starc_precomputed(env, policy_np, gamma, device)

    def loss_fn(r_state_preds: List[torch.Tensor]):
        n = len(r_state_preds)
        C_list = [r_state_to_cnorm_flat(r, precomputed, gamma) for r in r_state_preds]

        total_dist = torch.tensor(0.0, device=device)
        max_dist = torch.tensor(0.0, device=device)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(C_list[i] - C_list[j])
                total_dist += dist
                max_dist = torch.max(max_dist, dist)
                count += 1

        avg_dist = total_dist / float(count) if count > 0 else torch.tensor(0.0, device=device)
        trackers = {
            "avg_dist": float(avg_dist.detach().cpu().item()),
            "max_dist": float(max_dist.detach().cpu().item()),
        }
        return avg_dist, trackers

    loss_fn._precomputed = precomputed
    return loss_fn
