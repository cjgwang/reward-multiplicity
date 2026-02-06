import numpy as np
import torch
from typing import Tuple, Callable, List
from gridword import DeterministicGridWorld, Policy, Reward
from train import SmallRewardNet, transition_to_vector, default_device

Tensor = torch.Tensor
Array = np.ndarray
Output = np.ndarray

def build_transition_matrix(env: DeterministicGridWorld):
    N, A = env.num_states, env.num_actions
    P = np.zeros((N, A, N))
    for state_idx in range(N):
        for a in range(A):
            next_state = env.next_state_from_idx(state_idx, a)
            P[state_idx, a, next_state] = 1
    return P

def build_successor_representation(P, policy: Policy, gamma: float = 0.9):
    N, _ = policy.shape
    P_pol = np.einsum("sax,sa -> sx", P, policy)
    F = np.linalg.inv(np.eye(N) - gamma * P_pol)
    return F

def compute_reward_matrix(env: DeterministicGridWorld, reward: Reward):
    N, A = env.num_states, env.num_actions
    R = np.zeros((N, A))
    for state_idx in range(N):
        state = env.idx_to_state(state_idx)
        for a in range(A):
            next_state = env.next_state(state, a)
            R[state_idx, a] = reward((state, a, next_state))
    return R

def canonicalise_reward(F, P, R, policy, gamma: float = 0.9):
    if isinstance(R, torch.Tensor):
        N, A = R.shape
        V = F @ (R * policy).sum(dim=1)
        C = R - V.unsqueeze(1).expand(N, A) + gamma * P @ V
        return C
    else:
        N, A = R.shape
        V = F @ (R * policy).sum(axis=1)
        C = R - V.reshape(N, 1) + gamma * P @ V
        return C

def s_norm(Rc: Array):
    norm = np.linalg.norm(Rc)
    return Rc if norm == 0 else Rc / norm

def net_s_norm(Rc: Tensor):
    norm = torch.norm(Rc)
    return Rc if norm == 0 else Rc / norm

def ensemble_STARc_loss(env: DeterministicGridWorld, policy: Policy, gamma: float = 0.9, frozen=[], device=default_device):
    N, A = env.num_states, env.num_actions
    P = build_transition_matrix(env)
    F = build_successor_representation(P, policy, gamma)
    X = []
    for s in range(N):
        for a in range(A):
            ns = env.next_state_from_idx(s, a)
            t = (env.idx_to_state(s), a, env.idx_to_state(ns))
            X.append(transition_to_vector(t))

    policy_t = torch.from_numpy(policy).float().to(device)
    P_t = torch.from_numpy(P).float().to(device)
    F_t = torch.from_numpy(F).float().to(device)
    X_t = torch.tensor(np.array(X)).float().to(device)

    frozen_rewards = [net(X_t).detach().reshape(N, A) for net in frozen]
    frozen_canonicalised = [net_s_norm(canonicalise_reward(F_t, P_t, R, policy_t, gamma)) for R in frozen_rewards]
    m = len(frozen_rewards)

    def loss(y: Output, outputs: List[Tuple[SmallRewardNet, Output]]) -> Tuple[Tensor, dict]:
        total_dist = torch.tensor(0.0, device=device)
        max_dist = torch.tensor(0.0, device=device)
        rewards = [net(X_t).reshape(N, A) for net, _ in outputs]
        canonicalised = [net_s_norm(canonicalise_reward(F_t, P_t, R, policy_t, gamma)) for R in rewards]
        n = len(rewards)
        for i in range(n):
            for j in range(i+1, n + m):
                if(j < n):
                    dist = torch.norm(canonicalised[i] - canonicalised[j])
                else:
                    dist = torch.norm(canonicalised[i] - frozen_canonicalised[j - n])
                total_dist += dist
                max_dist = torch.max(dist, max_dist)

        avg_dist = total_dist / (n*m + n*(n-1)/2 + 1e-8)
        trackers = {"avg_dist": avg_dist.detach().item(), "max_dist": max_dist.detach().item()}
        return avg_dist, trackers
    return loss